from collections import deque
import copy

import numpy as np
import torch

from aircomp_opt.OTA_sim import AirCompSimulator
from aircomp_opt.f_theta_optim import optimize_beam_ris
from data import deepmimo, pilot_gen
from data.channel import build_ru_channel_evolver_from_config
from fl_core.agg import MetaUpdater
from fl_core.reptile_agg import ReptileAggregator
from fl_core.model_vector import (
    state_dict_to_vector_backbone,
    model_delta_to_vector_backbone,
)
from fl_core.trainer import GRUTrainer
from model.csi_cnn_gru import CSICNNGRU
from model.csi_cnn_ablation import CSICNNAblation
from utils.config import Config
from utils.logger import Logger


def main():
    config = Config
    logger = Logger(config=config) if config.log_to_file else Logger()
    # logger = Logger(config.log_file_path) if config.log_to_file else Logger()
    logger.info("Initializing simulation...")
    np.random.seed(0)
    torch.manual_seed(0)

    link_switch = getattr(config, "link_switch", [1, 0])
    if link_switch is None or len(link_switch) != 2:
        raise ValueError("Config.link_switch must be length-2: [reflect, direct]")
    reflect_on, direct_on = int(link_switch[0]), int(link_switch[1])
    if (reflect_on not in (0, 1)) or (direct_on not in (0, 1)):
        raise ValueError("Config.link_switch elements must be 0 or 1")
    if reflect_on == 0 and direct_on == 0:
        raise ValueError("Config.link_switch [0,0] is invalid")
    mode_desc = "reflection only" if (reflect_on == 1 and direct_on == 0) else \
        "direct only (no RIS)" if (reflect_on == 0 and direct_on == 1) else "reflection + direct"
    logger.info(f"Link switch [reflect,direct]={list(link_switch)} -> \033[33m{mode_desc}\033[0m")
    if reflect_on == 0:
        logger.info("Reflection link disabled: RIS contribution set to 0.")
    if direct_on == 0:
        logger.info("Direct link disabled.")

    # Per-user pilot observation noise (SNR heterogeneity)
    if getattr(config, "use_user_pilot_snr_hetero", False):
        snr_pilot_db = np.random.uniform(config.pilot_snr_dB_min, config.pilot_snr_dB_max,
                                         size=(config.num_users,)).astype(float)
    else:
        snr_pilot_db = np.full((config.num_users,), float(getattr(config, "pilot_SNR_dB", 20.0)), dtype=float)

    pilot_noise_std_k = np.power(10.0, -snr_pilot_db / 20.0)  # amplitude std
    logger.info(
        f"Pilot SNR_dB per user: min={snr_pilot_db.min():.2f}, mean={snr_pilot_db.mean():.2f}, max={snr_pilot_db.max():.2f}")

    h_BUs = None
    if config.use_synthetic_data:
        # Strictly align baseline main.py: Rayleigh H_BR and channel scaling by ref
        ref = float(getattr(config, "channel_ref_scale", np.sqrt(1e-10)))
        H_BR = ((np.random.randn(config.num_ris_elements, config.num_bs_antennas) +
                 1j * np.random.randn(config.num_ris_elements, config.num_bs_antennas)) / np.sqrt(2)).astype(
            np.complex64)

        theta_pilot = pilot_gen.generate_pilot_pattern(config.num_pilots, config.num_ris_elements)
        theta_ota = np.ones(config.num_ris_elements, dtype=np.complex64)

        # Path-loss samples (no explicit geometry; distances drawn from ranges)
        d_dir = np.random.uniform(getattr(config, "d_direct_min", 50.0),
                                  getattr(config, "d_direct_max", 150.0),
                                  size=(config.num_users,))
        d_ris = np.random.uniform(getattr(config, "d_ris_min", 50.0),
                                  getattr(config, "d_ris_max", 150.0),
                                  size=(config.num_users,))
        pl_direct = np.power(d_dir, -float(getattr(config, "alpha_direct", 3.0)))  # per-user
        pl_ris = np.power(d_ris, -float(getattr(config, "alpha_ris", 2.2)))        # per-user

        # Initialize user channels (at time 0)
        h_RUs = np.zeros((config.num_users, config.num_ris_elements), dtype=np.complex64)
        for k in range(config.num_users):
            h_RUs[k] = ((np.random.randn(config.num_ris_elements) + 1j * np.random.randn(
                config.num_ris_elements)) / np.sqrt(2) * np.sqrt(pl_ris[k]) / ref).astype(np.complex64)

        if direct_on == 1:
            h_BUs = ((np.random.randn(config.num_users, config.num_bs_antennas) +
                      1j * np.random.randn(config.num_users, config.num_bs_antennas)) / np.sqrt(2)).astype(
                np.complex64)
            # Apply per-user path-loss and baseline ref normalization to direct link
            h_BUs = (h_BUs.T * (np.sqrt(pl_direct) / ref)).T.astype(np.complex64)

    else:
        H_BR, h_RUs_static, h_BUs_static = deepmimo.load_data(config.deepmimo_path, num_users=config.num_users)

        # Use initial loaded channels and simulate variation via AR(1)
        H_BR = H_BR.astype(np.complex64)
        if h_RUs_static.ndim == 2:
            h_RUs = h_RUs_static.astype(np.complex64)  # (K, N)
        else:
            # If dataset provided multiple time snapshots, take first for initial
            h_RUs = h_RUs_static[:, 0, :].astype(np.complex64)

        theta_pilot = pilot_gen.generate_pilot_pattern(config.num_pilots, config.num_ris_elements)
        theta_ota = np.ones(config.num_ris_elements, dtype=np.complex64)
        if direct_on == 1:
            if h_BUs_static is None:
                logger.info("Direct link enabled but h_BU not found in dataset; using synthetic BS-UE channels.")
                h_BUs = (np.random.randn(config.num_users, config.num_bs_antennas) +
                         1j * np.random.randn(config.num_users, config.num_bs_antennas)) / np.sqrt(2)
            else:
                if h_BUs_static.ndim == 2:
                    h_BUs = h_BUs_static.astype(np.complex64)
                else:
                    h_BUs = h_BUs_static[:, 0, :].astype(np.complex64)

    # Set initial BS beamforming vector f (e.g., all ones)
    f_beam = np.ones(config.num_bs_antennas, dtype=np.complex64)

    # Initialize global model
    observation_dim = config.num_pilots  # each pilot yields one observation value (if scalar) or we consider multi-dim
    # Actually, each pilot observation is complex, we consider 2 channels (real & imag)
    obs_dim = config.num_pilots
    # Supervision target aligns with received model: [cascaded_vector, direct_scalar], both complex.
    output_dim = 2 * (config.num_ris_elements + 1)

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
    # For warm-start heads: keep a global head template to clone for users
    global_head_state = {k: v.clone() for k, v in global_model.state_dict().items() if k.startswith("head")}
    user_head_states = [copy.deepcopy(global_head_state) for _ in range(config.num_users)]

    enable_cnn_ablation = bool(getattr(config, "enable_cnn_ablation", False))
    ablation_pool_mode = str(getattr(config, "cnn_ablation_pool_mode", "mean")).lower()
    global_model_ablation = None
    user_head_states_ablation = None
    aggregator_ablation = None
    if enable_cnn_ablation:
        global_model_ablation = CSICNNAblation(
            observation_dim=obs_dim,
            output_dim=output_dim,
            pool_mode=ablation_pool_mode,
        )
        global_head_state_ablation = {
            k: v.clone() for k, v in global_model_ablation.state_dict().items() if k.startswith("head")
        }
        user_head_states_ablation = [copy.deepcopy(global_head_state_ablation) for _ in range(config.num_users)]
        if config.meta_algorithm.lower() == "reptile":
            aggregator_ablation = ReptileAggregator(
                step_size=config.reptile_step_size,
                use_aircomp=False,
                aircomp_simulator=None,
            )
        else:
            aggregator_ablation = MetaUpdater(
                meta_algorithm="FedAvg",
                step_size=1.0,
                use_aircomp=False,
                aircomp_simulator=None,
            )
        logger.info(
            f"CNN ablation enabled=True, pool_mode={ablation_pool_mode}. "
            f"Training shares the same per-round samples as CNN+GRU."
        )

    # OTA simulator (Phase1 physical aggregation)
    aircomp_sim = None
    if config.use_aircomp:
        aircomp_sim = AirCompSimulator(
            noise_std=getattr(config, "ota_noise_std", config.noise_std),
            tx_power=getattr(config, "ota_tx_power", 0.1),
            var_floor=getattr(config, "ota_var_floor", 1e-3),
            eps=getattr(config, "ota_eps", 1e-8),
        )
    # Fallback aggregator for non-OTA path
    if config.meta_algorithm.lower() == "reptile":
        aggregator = ReptileAggregator(step_size=config.reptile_step_size, use_aircomp=False, aircomp_simulator=None)
    else:
        aggregator = MetaUpdater(meta_algorithm="FedAvg", step_size=1.0, use_aircomp=False, aircomp_simulator=None)
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
            h_BU_k = h_BUs[k] if h_BUs is not None else None
            Y_pilot, cascaded, direct_eff = pilot_gen.simulate_pilot_observation(
                H_BR, h_RUs[k], f_beam, theta_pilot,
                noise_std=float(pilot_noise_std_k[k]),
                h_BU=h_BU_k,
                link_switch=link_switch,
            )

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

            total_effective = np.concatenate([cascaded, np.asarray([direct_eff], dtype=cascaded.dtype)], axis=0)
            y = np.concatenate([np.real(total_effective), np.imag(total_effective)], axis=0).astype(np.float32)
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
        local_models_ablation = [] if enable_cnn_ablation else None
        losses_ablation = [] if enable_cnn_ablation else None
        for k in range(config.num_users):

            # Model for user k from global weights
            local_model = CSICNNGRU(observation_dim=obs_dim, output_dim=output_dim)
            # load global backbone + user head (warm-start head copied from global template on round1)
            state = global_model.state_dict()
            for hk, hv in user_head_states[k].items():
                state[hk] = hv.clone()
            local_model.load_state_dict(state)

            # Train on user k's data
            if use_local_cache:
                data_k = list(sample_buffers[k])  # latest S window samples, OK if length < S
            else:
                data_k = [local_data[k]]  # fallback: 1 sample per epoch

            local_model, loss = trainer.train(local_model, data_k)
            losses.append(loss if loss is not None else 0.0)
            local_models.append(local_model)
            # cache back the personalized head for user k
            user_head_states[k] = {name: param.detach().clone() for name, param in local_model.state_dict().items() if name.startswith("head")}
            if enable_cnn_ablation:
                local_model_ablation = CSICNNAblation(
                    observation_dim=obs_dim,
                    output_dim=output_dim,
                    pool_mode=ablation_pool_mode,
                )
                state_ablation = global_model_ablation.state_dict()
                for hk, hv in user_head_states_ablation[k].items():
                    state_ablation[hk] = hv.clone()
                local_model_ablation.load_state_dict(state_ablation)

                local_model_ablation, loss_ablation = trainer.train(local_model_ablation, data_k)
                losses_ablation.append(loss_ablation if loss_ablation is not None else 0.0)
                local_models_ablation.append(local_model_ablation)
                user_head_states_ablation[k] = {
                    name: param.detach().clone()
                    for name, param in local_model_ablation.state_dict().items()
                    if name.startswith("head")
                }
                if (loss is not None) and (loss_ablation is not None):
                    logger.info(
                        f"User {k + 1} local loss -> GRU: \033[34m{loss:.4f}\033[0m, "
                        f"CNN-abl: \033[36m{loss_ablation:.4f}\033[0m"
                    )
                else:
                    logger.info(f"User {k + 1} local training done.")
            else:
                logger.info(
                    f"User {k + 1} local training loss: \033[34m{loss:.4f}\033[0m"
                    if loss is not None else f"User {k + 1} local training done."
                )

        if enable_cnn_ablation and losses and losses_ablation:
            logger.info(
                f"Round {round_idx} mean local loss -> GRU: {np.mean(losses):.4f}, "
                f"CNN-abl: {np.mean(losses_ablation):.4f}"
            )

        # Aggregate updates at server
        logger.info("Aggregating GRU updates at server.")
        old_global_vec = state_dict_to_vector_backbone(global_model).detach().cpu()
        old_global_vec_ablation = None
        if enable_cnn_ablation:
            old_global_vec_ablation = state_dict_to_vector_backbone(global_model_ablation).detach().cpu()

        # User weights K_k for both OTA aggregation and the OTA-aware optimizer.
        if getattr(config, "ota_use_weighted_users", True):
            if getattr(config, "user_weight_mode", "uniform") == "random":
                K_vals = np.random.uniform(getattr(config, "user_data_size_min", 5000),
                                           getattr(config, "user_data_size_max", 20000),
                                           size=(config.num_users,))
            else:
                K_vals = np.full((config.num_users,), float(getattr(config, "ota_user_weight", 1.0)))
        else:
            K_vals = np.ones((config.num_users,), dtype=float)
        K_vec = torch.from_numpy(K_vals.astype(np.float32))
        K_norm = K_vec / torch.mean(K_vec)

        # Prepare local update vectors once so the OTA path and the OTA-aware optimizer
        # use the same current-round update statistics.
        delta_list = []
        for lm in local_models:
            delta_list.append(model_delta_to_vector_backbone(lm, global_model).detach().cpu())
        delta_mat = torch.stack(delta_list, dim=0)  # [K, d]
        delta_var = delta_mat.float().var(dim=1, unbiased=False)
        delta_var = torch.clamp(delta_var, min=float(getattr(config, "ota_var_floor", 1e-3)))

        h_eff = None
        if config.use_aircomp and aircomp_sim is not None:
            # Effective channels per user (complex) using theta_ota; reused by ablation branch.
            casc_pref = f_beam.conj() @ H_BR.T
            h_eff_list = []
            for k in range(config.num_users):
                direct = f_beam.conj().dot(h_BUs[k]) if (direct_on == 1 and h_BUs is not None) else 0.0
                reflect = 0.0
                if reflect_on == 1:
                    reflect = np.dot(theta_ota, casc_pref * h_RUs[k])
                h_eff_list.append(direct + reflect)
            h_eff = torch.from_numpy(np.asarray(h_eff_list, dtype=np.complex64))

        if config.use_aircomp and aircomp_sim is not None:
            agg_update, diag = aircomp_sim.aggregate_updates(
                updates=delta_mat.float(),
                h_eff=h_eff,
                user_weights=K_norm,
            )
            # Ideal weighted average (for NMSE logging)
            ideal_update = (delta_mat * K_norm.view(-1, 1)).sum(dim=0) / (K_norm.sum() + 1e-12)
            agg_error_power = torch.norm(agg_update - ideal_update) ** 2
            ideal_power = torch.norm(ideal_update) ** 2
            nmse = agg_error_power / (ideal_power + 1e-12)

            # Apply aggregated delta via FedAvg/Reptile semantics
            aggregator.apply_aggregated_delta(global_model, agg_update, backbone_only=True, prefix="backbone")
            logger.info(
                f"AirComp eta={diag['eta']:.4e}, min|u|^2={diag['min_inner2']:.4e}, "
                f"agg_NMSE={nmse.item():.4e}, agg_err={agg_error_power.item():.4e}, "
                f"ideal_power={ideal_power.item():.4e}"
            )

        else:
            # fallback to FedAvg / Reptile on parameters (backbone only)
            global_model = aggregator.aggregate(global_model, local_models, backbone_only=True, prefix="backbone")

        new_global_vec = state_dict_to_vector_backbone(global_model).detach().cpu()
        logger.info(f"GRU global backbone update norm: {torch.norm(new_global_vec - old_global_vec).item():.4e}")

        if enable_cnn_ablation:
            logger.info("Aggregating CNN-ablation updates at server.")
            delta_list_ablation = []
            for lm in local_models_ablation:
                delta_list_ablation.append(
                    model_delta_to_vector_backbone(lm, global_model_ablation).detach().cpu()
                )
            delta_mat_ablation = torch.stack(delta_list_ablation, dim=0)  # [K, d]

            if config.use_aircomp and aircomp_sim is not None:
                agg_update_ablation, diag_ablation = aircomp_sim.aggregate_updates(
                    updates=delta_mat_ablation.float(),
                    h_eff=h_eff,
                    user_weights=K_norm,
                )
                ideal_update_ablation = (
                    (delta_mat_ablation * K_norm.view(-1, 1)).sum(dim=0) / (K_norm.sum() + 1e-12)
                )
                agg_error_power_ablation = torch.norm(agg_update_ablation - ideal_update_ablation) ** 2
                ideal_power_ablation = torch.norm(ideal_update_ablation) ** 2
                nmse_ablation = agg_error_power_ablation / (ideal_power_ablation + 1e-12)

                aggregator_ablation.apply_aggregated_delta(
                    global_model_ablation,
                    agg_update_ablation,
                    backbone_only=True,
                    prefix="backbone",
                )
                logger.info(
                    f"CNN-abl AirComp eta={diag_ablation['eta']:.4e}, "
                    f"min|u|^2={diag_ablation['min_inner2']:.4e}, "
                    f"agg_NMSE={nmse_ablation.item():.4e}, "
                    f"agg_err={agg_error_power_ablation.item():.4e}, "
                    f"ideal_power={ideal_power_ablation.item():.4e}"
                )
            else:
                global_model_ablation = aggregator_ablation.aggregate(
                    global_model_ablation,
                    local_models_ablation,
                    backbone_only=True,
                    prefix="backbone",
                )

            new_global_vec_ablation = state_dict_to_vector_backbone(global_model_ablation).detach().cpu()
            logger.info(
                "CNN-abl global backbone update norm: "
                f"{torch.norm(new_global_vec_ablation - old_global_vec_ablation).item():.4e}"
            )

        # Optimize beamforming vector f and RIS phases theta for next round
        logger.info("Optimizing beamforming and RIS configuration.")
        # Use current true channels for optimization (in real scenario, would use estimated channels)
        f_beam, theta_ota, nmse_proxy = optimize_beam_ris(
            H_BR, h_RUs, h_BUs=h_BUs, theta_init=theta_ota, f_init=f_beam,
            link_switch=link_switch, user_weights=K_norm.numpy(),
            update_vars=delta_var.numpy(),
            tx_power=getattr(config, "ota_tx_power", 0.1),
            noise_std=getattr(config, "ota_noise_std", config.noise_std),
            var_floor=getattr(config, "ota_var_floor", 1e-3),
            eps=getattr(config, "ota_eps", 1e-8),
        )
        # New pilot pattern independent from OTA theta
        theta_pilot = pilot_gen.generate_pilot_pattern(config.num_pilots, config.num_ris_elements)
        logger.info(f"Optimized f: {np.round(f_beam, 4)}")
        logger.info(f"Optimized theta_ota: {np.round(theta_ota, 4)}, proxy_NMSE={nmse_proxy:.4e}")

        # Evolve channels for next round (AR(1) with optional dynamic alpha(t))
        h_RUs, alpha_used = ru_evolver.step(h_RUs, round_idx)  # alpha_used: (K,)
        if getattr(config, "use_dynamic_alpha", False):
            logger.info(f"RU dynamic alpha(t) (mode={config.dynamic_alpha_mode}): {alpha_used}, "
                        f"min={alpha_used.min():.4f}, max={alpha_used.max():.4f}"
                        )
        else:
            logger.info(f"RU per-user alpha_k: {alpha_used}, "
                        f"min={alpha_used.min():.4f}, max={alpha_used.max():.4f}"
                        )

    logger.info("Training process completed.")
    # Clean up logger
    logger.close()


if __name__ == "__main__":
    main()
