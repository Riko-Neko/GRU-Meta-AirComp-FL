from collections import deque

import numpy as np
import torch

from aircomp_opt.OTA_sim import AirCompSimulator
from aircomp_opt.f_theta_optim import optimize_beam_ris
from data import deepmimo, pilot_gen
from data.channel import build_ru_channel_evolver_from_config
from fl_core.agg import MetaUpdater
from fl_core.reptile_agg import ReptileAggregator
from fl_core.trainer import GRUTrainer
from model.csi_cnn_gru import CSICNNGRU
from utils.config import Config
from utils.logger import Logger


def main():
    config = Config
    logger = Logger(config=config) if config.log_to_file else Logger()
    # logger = Logger(config.log_file_path) if config.log_to_file else Logger()
    logger.info("Initializing simulation...")
    np.random.seed(0)
    torch.manual_seed(0)

    # Per-user pilot observation noise (SNR heterogeneity)
    if getattr(config, "use_user_pilot_snr_hetero", False):
        snr_pilot_db = np.random.uniform(config.pilot_snr_dB_min, config.pilot_snr_dB_max,
                                         size=(config.num_users,)).astype(float)
    else:
        snr_pilot_db = np.full((config.num_users,), float(getattr(config, "pilot_SNR_dB", 20.0)), dtype=float)

    pilot_noise_std_k = np.power(10.0, -snr_pilot_db / 20.0)  # amplitude std
    logger.info(
        f"Pilot SNR_dB per user: min={snr_pilot_db.min():.2f}, mean={snr_pilot_db.mean():.2f}, max={snr_pilot_db.max():.2f}")

    if config.use_synthetic_data:
        H_BR = (np.random.randn(config.num_ris_elements, config.num_bs_antennas) +
                1j * np.random.randn(config.num_ris_elements, config.num_bs_antennas)) / np.sqrt(2)

        theta_pattern = pilot_gen.generate_pilot_pattern(config.num_pilots, config.num_ris_elements)

        # Initialize user channels (at time 0)
        h_RUs = np.zeros((config.num_users, config.num_ris_elements), dtype=np.complex64)
        for k in range(config.num_users):
            h_RUs[k] = (np.random.randn(config.num_ris_elements) + 1j * np.random.randn(
                config.num_ris_elements)) / np.sqrt(2)

    else:
        H_BR, h_RUs_static = deepmimo.load_data(config.deepmimo_path, num_users=config.num_users)

        # Use initial loaded channels and simulate variation via AR(1)
        H_BR = H_BR.astype(np.complex64)
        if h_RUs_static.ndim == 2:
            h_RUs = h_RUs_static.astype(np.complex64)  # (K, N)
        else:
            # If dataset provided multiple time snapshots, take first for initial
            h_RUs = h_RUs_static[:, 0, :].astype(np.complex64)

        theta_pattern = pilot_gen.generate_pilot_pattern(config.num_pilots, config.num_ris_elements)

    # Set initial BS beamforming vector f (e.g., all ones)
    f_beam = np.ones(config.num_bs_antennas, dtype=np.complex64)

    # Initialize global model
    observation_dim = config.num_pilots  # each pilot yields one observation value (if scalar) or we consider multi-dim
    # Actually, each pilot observation is complex, we consider 2 channels (real & imag)
    obs_dim = config.num_pilots
    output_dim = 2 * config.num_ris_elements

    # Per-user sliding window buffer for GRU
    W = int(getattr(config, "window_length", 1))
    use_time_window = bool(getattr(config, "use_time_window", False))
    pad_val = float(getattr(config, "window_pad_value", 0.0))

    obs_buffers = [deque(maxlen=W) for _ in range(config.num_users)]
    logger.info(f"GRU time-window enabled={use_time_window}, W={W}")
    # Per-user local sample cache: store last S window-samples (X_seq, y)
    S = int(getattr(config, "local_cache_size", 1))
    use_local_cache = bool(getattr(config, "use_local_sample_cache", False)) and (S > 1)
    sample_buffers = [deque(maxlen=S) for _ in range(config.num_users)]
    logger.info(f"Local sample cache enabled={use_local_cache}, S={S}")
    global_model = CSICNNGRU(observation_dim=obs_dim, output_dim=output_dim)

    # Create aggregator (Meta-FL aggregator)
    aircomp_sim = None
    if config.use_aircomp:
        aircomp_sim = AirCompSimulator(noise_std=config.noise_std)
    if config.meta_algorithm.lower() == "reptile":
        aggregator = ReptileAggregator(step_size=config.reptile_step_size, use_aircomp=config.use_aircomp,
                                       aircomp_simulator=aircomp_sim)
    else:
        aggregator = MetaUpdater(meta_algorithm="FedAvg", step_size=1.0, use_aircomp=config.use_aircomp,
                                 aircomp_simulator=aircomp_sim)
    # Trainer for local updates
    trainer = GRUTrainer(learning_rate=config.local_lr, epochs=config.local_epochs, batch_size=config.batch_size)
    ru_evolver = build_ru_channel_evolver_from_config(config)
    # Simulation rounds
    logger.info(f"Starting training for {config.num_rounds} rounds...")

    for round_idx in range(1, config.num_rounds + 1):
        logger.info(f"\033[32mRound {round_idx}\033[0m - Generating pilot observations.")
        # Generate pilot observation and ground truth channel for each user at this round
        local_data = []
        for k in range(config.num_users):
            # Pilot signals for user k
            Y_pilot, cascaded = pilot_gen.simulate_pilot_observation(H_BR, h_RUs[k], f_beam, theta_pattern,
                                                                     noise_std=float(pilot_noise_std_k[k]))

            # Build GRU sequence input: X_seq shape (W, 2, P) --> (seq_len, 2, obs_dim)

            obs_real = np.real(Y_pilot)  # Separate real and imag channels
            obs_imag = np.imag(Y_pilot)
            obs_step = np.stack([obs_real, obs_imag], axis=0).astype(np.float32)  # (2, P)

            if use_time_window and W > 1:
                obs_buffers[k].append(obs_step)  # append current step
                seq = list(obs_buffers[k])  # list of (2,P), length <= W
                X_seq = np.stack(seq, axis=0)  # (len, 2, P)

                # Pad to fixed W (left-padding)
                if X_seq.shape[0] < W:
                    pad_len = W - X_seq.shape[0]
                    pad = np.full((pad_len, 2, obs_dim), pad_val, dtype=np.float32)
                    X_seq = np.concatenate([pad, X_seq], axis=0)  # (W, 2, P)
            else:
                # Fallback: seq_len = 1
                X_seq = obs_step[None, :, :]  # (1, 2, P)

            y = np.concatenate([np.real(cascaded), np.imag(cascaded)], axis=0).astype(np.float32)
            sample = (X_seq.astype(np.float32, copy=False), y.astype(np.float32, copy=False))
            local_data.append(sample)

            if use_time_window and W > 1 and round_idx <= 3 and k == 0:
                logger.info(f"Example X_seq shape for user1: {sample[0].shape}")  # (W,2,P)

            # push into local cache
            if use_local_cache:
                sample_buffers[k].append((sample[0].copy(), sample[1].copy()))  # copy for safety

        # Local training on each user's data
        local_models = []
        losses = []
        for k in range(config.num_users):

            # Model for user k from global weights
            local_model = CSICNNGRU(observation_dim=obs_dim, output_dim=output_dim)
            local_model.load_state_dict(global_model.state_dict())

            # Train on user k's data
            if use_local_cache:
                data_k = list(sample_buffers[k])  # latest S window samples, OK if length < S
            else:
                data_k = [local_data[k]]  # fallback: 1 sample per epoch

            local_model, loss = trainer.train(local_model, data_k)
            losses.append(loss if loss is not None else 0.0)
            local_models.append(local_model)
            logger.info(
                f"User {k + 1} local training loss: \033[34m{loss:.4f}\033[0m" if loss is not None else f"User {k + 1} local training done.")

        # Aggregate updates at server
        logger.info("Aggregating updates at server.")
        old_global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        global_model = aggregator.aggregate(global_model, local_models)
        # If AirComp simulated, we might estimate aggregation error (if aggregator has aircomp)

        if config.use_aircomp:
            # Compute ideal aggregated update Δw (no noise)
            ideal_update = {}
            for key, val in old_global_state.items():
                ideal_update[key] = torch.zeros_like(val)

            for lm in local_models:
                for key, val in lm.state_dict().items():
                    ideal_update[key] += (val.cpu() - old_global_state[key].cpu())

            for key in ideal_update:
                ideal_update[key] /= len(local_models)

            # Actual aggregated update applied to global_model
            actual_update = {}
            for key in ideal_update:
                actual_update[key] = (global_model.state_dict()[key].cpu() - old_global_state[key].cpu())

            err_sum = 0.0
            ref_sum = 0.0
            for key in ideal_update:
                diff = actual_update[key].numpy() - ideal_update[key].numpy()
                err_sum += np.vdot(diff, diff).real
                ref_sum += np.vdot(ideal_update[key].numpy(),
                                   ideal_update[key].numpy()).real

            nmse = err_sum / (ref_sum + 1e-12)
            logger.info(f"AirComp aggregation NMSE (Δw): \033[36m{nmse:.6f}\033[0m")

        # Optimize beamforming vector f and RIS phases theta for next round
        logger.info("Optimizing beamforming and RIS configuration.")
        # Use current true channels for optimization (in real scenario, would use estimated channels)
        f_beam, theta_opt = optimize_beam_ris(H_BR, h_RUs)
        # Update RIS pattern for pilot to new theta (assuming we apply new theta for next round pilot)
        theta_pattern = np.tile(theta_opt, (config.num_pilots, 1))
        logger.info(f"Optimized f: {np.round(f_beam, 4)}")
        logger.info(f"Optimized theta: {np.round(theta_opt, 4)}")

        # Evolve channels for next round (AR(1) with optional dynamic alpha(t))
        h_RUs, alpha_used = ru_evolver.step(h_RUs, round_idx)  # alpha_used: (K,)
        if getattr(config, "use_dynamic_alpha", False):
            logger.info(f"RU channel evolved with dynamic alpha(t) (mode={config.dynamic_alpha_mode}): "
                        f"min={alpha_used.min():.4f}, mean={alpha_used.mean():.4f}, max={alpha_used.max():.4f}"
                        )
        else:
            logger.info(f"RU channel evolved with per-user alpha_k: "
                        f"min={alpha_used.min():.4f}, mean={alpha_used.mean():.4f}, max={alpha_used.max():.4f}"
                        )

    logger.info("Training process completed.")
    # Clean up logger
    logger.close()


if __name__ == "__main__":
    main()
