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

    link_switch = config.link_switch
    reflect_on, direct_on = int(link_switch[0]), int(link_switch[1])
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
    if config.use_user_pilot_snr_hetero:
        snr_pilot_db = np.random.uniform(config.pilot_snr_dB_min, config.pilot_snr_dB_max,
                                         size=(config.num_users,)).astype(float)
    else:
        snr_pilot_db = np.full((config.num_users,), float(config.pilot_SNR_dB), dtype=float)

    pilot_noise_std_k = np.power(10.0, -snr_pilot_db / 20.0)  # amplitude std
    logger.info(
        f"Pilot SNR_dB per user: min={snr_pilot_db.min():.2f}, mean={snr_pilot_db.mean():.2f}, max={snr_pilot_db.max():.2f}")

    h_BUs = None
    if config.use_synthetic_data:
        # Strictly align baseline main.py: Rayleigh H_BR and channel scaling by ref
        ref = float(config.channel_ref_scale)
        H_BR = ((np.random.randn(config.num_ris_elements, config.num_bs_antennas) +
                 1j * np.random.randn(config.num_ris_elements, config.num_bs_antennas)) / np.sqrt(2)).astype(
            np.complex64)

        theta_pilot = pilot_gen.generate_pilot_pattern(config.num_pilots, config.num_ris_elements)
        theta_ota = np.ones(config.num_ris_elements, dtype=np.complex64)

        # Path-loss samples (no explicit geometry; distances drawn from ranges)
        d_dir = np.random.uniform(config.d_direct_min,
                                  config.d_direct_max,
                                  size=(config.num_users,))
        d_ris = np.random.uniform(config.d_ris_min,
                                  config.d_ris_max,
                                  size=(config.num_users,))
        pl_direct = np.power(d_dir, -float(config.alpha_direct))  # per-user
        pl_ris = np.power(d_ris, -float(config.alpha_ris))        # per-user

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
    W = int(config.window_length)
    use_time_window = bool(config.use_time_window)
    pad_val = float(config.window_pad_value)
    use_persistent_hidden_state = bool(config.use_persistent_hidden_state)
    stateful_single_step_input = bool(config.stateful_single_step_input)
    reset_hidden_on_round1 = bool(config.reset_hidden_on_round1)
    reset_hidden_on_large_backbone_update = bool(config.reset_hidden_on_large_backbone_update)
    hidden_reset_update_norm_threshold = float(config.hidden_reset_update_norm_threshold)
    if use_persistent_hidden_state and not stateful_single_step_input:
        logger.info("stateful_single_step_input=False is not supported; fallback to single-step mode.")
        stateful_single_step_input = True

    obs_buffers = [deque(maxlen=W) for _ in range(config.num_users)]
    logger.info(
        f"GRU mode: persistent_hidden={use_persistent_hidden_state}, "
        f"time_window={use_time_window}, W={W}"
    )
    # Per-user local sample cache: store last S window-samples (X_seq, y)
    S = int(config.local_cache_size)
    use_local_cache = bool(config.use_local_sample_cache) and (S > 1)
    sample_buffers = [deque(maxlen=S) for _ in range(config.num_users)]
    logger.info(f"Local sample cache enabled={use_local_cache}, S={S}")
    if use_persistent_hidden_state and use_local_cache:
        logger.info("Persistent-hidden GRU path bypasses local sample cache (single-step online update).")
    global_model = CSICNNGRU(observation_dim=obs_dim, output_dim=output_dim)
    user_hidden_states = [None for _ in range(config.num_users)]
    reset_hidden_next_round = False
    # For warm-start heads: keep a global head template to clone for users
    global_head_state = {k: v.clone() for k, v in global_model.state_dict().items() if k.startswith("head")}
    user_head_states = [copy.deepcopy(global_head_state) for _ in range(config.num_users)]

    enable_cnn_ablation = bool(config.enable_cnn_ablation)
    ablation_pool_mode = str(config.cnn_ablation_pool_mode).lower()
    global_model_ablation = None
    user_head_states_ablation = None
    aggregator_ablation = None
    f_beam_ablation = None
    theta_ota_ablation = None
    obs_buffers_ablation = None
    sample_buffers_ablation = None
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
        f_beam_ablation = f_beam.copy()
        theta_ota_ablation = theta_ota.copy()
        obs_buffers_ablation = [deque(maxlen=W) for _ in range(config.num_users)]
        sample_buffers_ablation = [deque(maxlen=S) for _ in range(config.num_users)]
        logger.info(
            f"CNN ablation enabled=True, pool_mode={ablation_pool_mode}. "
            "Training shares the same channel/user realization, but keeps an independent "
            "physical-layer optimization state."
        )

    # OTA simulator (Phase1 physical aggregation)
    aircomp_sim = None
    if config.use_aircomp:
        aircomp_sim = AirCompSimulator(
            noise_std=config.ota_noise_std,
            tx_power=config.ota_tx_power,
            var_floor=config.ota_var_floor,
            eps=config.ota_eps,
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
        if use_persistent_hidden_state:
            if (round_idx == 1 and reset_hidden_on_round1) or reset_hidden_next_round:
                user_hidden_states = [None for _ in range(config.num_users)]
                reason = "round1" if (round_idx == 1 and reset_hidden_on_round1) else "large_backbone_update"
                logger.info(f"Reset persistent hidden states at round {round_idx} (reason={reason}).")
                reset_hidden_next_round = False
        # Generate pilot observation and ground truth channel for each user at this round
        local_data = []
        local_data_ablation = [] if enable_cnn_ablation else None
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

            if use_persistent_hidden_state and stateful_single_step_input:
                X_seq = obs_step[None, :, :]  # (1, 2, P), one newly received step only
            else:
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

            if (not use_persistent_hidden_state) and use_time_window and W > 1 and round_idx <= 3 and k == 0:
                logger.info(f"Example X_seq shape for user1: {sample[0].shape}")  # (W,2,P)

            # push into local cache
            if use_local_cache and (not use_persistent_hidden_state):
                sample_buffers[k].append((sample[0].copy(), sample[1].copy()))  # copy for safety

            if enable_cnn_ablation:
                Y_pilot_ablation, cascaded_ablation, direct_eff_ablation = pilot_gen.simulate_pilot_observation(
                    H_BR, h_RUs[k], f_beam_ablation, theta_pilot,
                    noise_std=float(pilot_noise_std_k[k]),
                    h_BU=h_BU_k,
                    link_switch=link_switch,
                )

                obs_real_ablation = np.real(Y_pilot_ablation)
                obs_imag_ablation = np.imag(Y_pilot_ablation)
                obs_step_ablation = np.stack([obs_real_ablation, obs_imag_ablation], axis=0).astype(np.float32)

                if use_time_window and W > 1:
                    obs_buffers_ablation[k].append(obs_step_ablation)
                    seq_ablation = list(obs_buffers_ablation[k])
                    X_seq_ablation = np.stack(seq_ablation, axis=0)

                    if X_seq_ablation.shape[0] < W:
                        pad_len_ablation = W - X_seq_ablation.shape[0]
                        pad_ablation = np.full((pad_len_ablation, 2, obs_dim), pad_val, dtype=np.float32)
                        X_seq_ablation = np.concatenate([pad_ablation, X_seq_ablation], axis=0)
                else:
                    X_seq_ablation = obs_step_ablation[None, :, :]

                total_effective_ablation = np.concatenate(
                    [cascaded_ablation, np.asarray([direct_eff_ablation], dtype=cascaded_ablation.dtype)],
                    axis=0,
                )
                y_ablation = np.concatenate(
                    [np.real(total_effective_ablation), np.imag(total_effective_ablation)],
                    axis=0,
                ).astype(np.float32)
                sample_ablation = (
                    X_seq_ablation.astype(np.float32, copy=False),
                    y_ablation.astype(np.float32, copy=False),
                )
                local_data_ablation.append(sample_ablation)

                if use_local_cache:
                    sample_buffers_ablation[k].append((sample_ablation[0].copy(), sample_ablation[1].copy()))

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
            if use_persistent_hidden_state:
                sample_k = local_data[k]
                local_model, loss, hidden_next = trainer.train_stateful_step(
                    local_model,
                    sample_k,
                    hidden_state=user_hidden_states[k],
                )
                user_hidden_states[k] = hidden_next
                if (round_idx <= 3) and (k == 0) and (hidden_next is not None):
                    logger.info(f"User1 persistent hidden norm: {torch.norm(hidden_next).item():.4e}")
            else:
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

                if use_local_cache:
                    data_k_ablation = list(sample_buffers_ablation[k])
                else:
                    data_k_ablation = [local_data_ablation[k]]

                local_model_ablation, loss_ablation = trainer.train(local_model_ablation, data_k_ablation)
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
        if config.ota_use_weighted_users:
            if config.user_weight_mode == "random":
                K_vals = np.random.uniform(config.user_data_size_min,
                                           config.user_data_size_max,
                                           size=(config.num_users,))
            else:
                K_vals = np.full((config.num_users,), float(config.ota_user_weight))
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
        delta_var = torch.clamp(delta_var, min=float(config.ota_var_floor))

        delta_mat_ablation = None
        delta_var_ablation = None
        if enable_cnn_ablation:
            delta_list_ablation = []
            for lm in local_models_ablation:
                delta_list_ablation.append(
                    model_delta_to_vector_backbone(lm, global_model_ablation).detach().cpu()
                )
            delta_mat_ablation = torch.stack(delta_list_ablation, dim=0)  # [K, d]
            delta_var_ablation = delta_mat_ablation.float().var(dim=1, unbiased=False)
            delta_var_ablation = torch.clamp(delta_var_ablation, min=float(config.ota_var_floor))

        h_eff = None
        h_eff_ablation = None
        if config.use_aircomp and aircomp_sim is not None:
            # Effective channels per user (complex) using each branch's own OTA variables.
            casc_pref = f_beam.conj() @ H_BR.T
            h_eff_list = []
            for k in range(config.num_users):
                direct = f_beam.conj().dot(h_BUs[k]) if (direct_on == 1 and h_BUs is not None) else 0.0
                reflect = 0.0
                if reflect_on == 1:
                    reflect = np.dot(theta_ota, casc_pref * h_RUs[k])
                h_eff_list.append(direct + reflect)
            h_eff = torch.from_numpy(np.asarray(h_eff_list, dtype=np.complex64))

            if enable_cnn_ablation:
                casc_pref_ablation = f_beam_ablation.conj() @ H_BR.T
                h_eff_list_ablation = []
                for k in range(config.num_users):
                    direct_ablation = (
                        f_beam_ablation.conj().dot(h_BUs[k]) if (direct_on == 1 and h_BUs is not None) else 0.0
                    )
                    reflect_ablation = 0.0
                    if reflect_on == 1:
                        reflect_ablation = np.dot(theta_ota_ablation, casc_pref_ablation * h_RUs[k])
                    h_eff_list_ablation.append(direct_ablation + reflect_ablation)
                h_eff_ablation = torch.from_numpy(np.asarray(h_eff_list_ablation, dtype=np.complex64))

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
        backbone_update_norm = torch.norm(new_global_vec - old_global_vec).item()
        logger.info(f"GRU global backbone update norm: {backbone_update_norm:.4e}")
        if use_persistent_hidden_state and reset_hidden_on_large_backbone_update:
            if backbone_update_norm > hidden_reset_update_norm_threshold:
                reset_hidden_next_round = True
                logger.info(
                    "Persistent hidden states will be reset next round: "
                    f"update_norm={backbone_update_norm:.4e} > threshold={hidden_reset_update_norm_threshold:.4e}"
                )

        if enable_cnn_ablation:
            logger.info("Aggregating CNN-ablation updates at server.")
            if config.use_aircomp and aircomp_sim is not None:
                agg_update_ablation, diag_ablation = aircomp_sim.aggregate_updates(
                    updates=delta_mat_ablation.float(),
                    h_eff=h_eff_ablation,
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
            tx_power=config.ota_tx_power,
            noise_std=config.ota_noise_std,
            var_floor=config.ota_var_floor,
            eps=config.ota_eps,
        )
        # New pilot pattern independent from OTA theta
        theta_pilot = pilot_gen.generate_pilot_pattern(config.num_pilots, config.num_ris_elements)
        logger.info(f"Optimized f: {np.round(f_beam, 4)}")
        logger.info(f"Optimized theta_ota: {np.round(theta_ota, 4)}, proxy_NMSE={nmse_proxy:.4e}")

        if enable_cnn_ablation:
            f_beam_ablation, theta_ota_ablation, nmse_proxy_ablation = optimize_beam_ris(
                H_BR, h_RUs, h_BUs=h_BUs, theta_init=theta_ota_ablation, f_init=f_beam_ablation,
                link_switch=link_switch, user_weights=K_norm.numpy(),
                update_vars=delta_var_ablation.numpy(),
                tx_power=config.ota_tx_power,
                noise_std=config.ota_noise_std,
                var_floor=config.ota_var_floor,
                eps=config.ota_eps,
            )
            logger.info(f"Optimized f (CNN-abl): {np.round(f_beam_ablation, 4)}")
            logger.info(
                f"Optimized theta_ota (CNN-abl): {np.round(theta_ota_ablation, 4)}, "
                f"proxy_NMSE={nmse_proxy_ablation:.4e}"
            )

        # Evolve channels for next round (AR(1) with optional dynamic alpha(t))
        h_RUs, alpha_used = ru_evolver.step(h_RUs, round_idx)  # alpha_used: (K,)
        if config.use_dynamic_alpha:
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
