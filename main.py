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
    state_dict_to_vector,
    state_dict_to_vector_backbone,
    model_delta_to_vector,
    model_delta_to_vector_backbone,
)
from fl_core.trainer import GRUTrainer
from model.csi_cnn_gru import CSICNNGRU
from model.csi_cnn_arch import CSICNNArch
from model.csi_cnn_baseline import CSICNNBaseline
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
    pad_val = float(config.window_pad_value)
    gru_context_mode = str(config.gru_context_mode).lower()
    if gru_context_mode not in {"persistent_hidden", "time_window"}:
        raise ValueError("Config.gru_context_mode must be 'persistent_hidden' or 'time_window'")
    use_persistent_hidden_state = (gru_context_mode == "persistent_hidden")
    use_time_window = (gru_context_mode == "time_window")
    reset_hidden_on_round1 = bool(config.reset_hidden_on_round1)
    reset_hidden_on_large_backbone_update = bool(config.reset_hidden_on_large_backbone_update)
    hidden_reset_update_norm_threshold = float(config.hidden_reset_update_norm_threshold)

    obs_buffers = [deque(maxlen=W) for _ in range(config.num_users)]
    logger.info(
        f"GRU context mode={gru_context_mode}, W={W}"
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

    enable_cnn_arch_ablation = bool(config.enable_cnn_arch_ablation)
    global_model_arch = None
    user_head_states_arch = None
    aggregator_arch = None
    f_beam_arch = None
    theta_ota_arch = None
    obs_buffers_arch = None
    sample_buffers_arch = None
    if enable_cnn_arch_ablation:
        global_model_arch = CSICNNArch(
            observation_dim=obs_dim,
            output_dim=output_dim,
            conv_filters=int(config.cnn_arch_conv_filters),
            conv_kernel=int(config.cnn_arch_conv_kernel),
            hidden_size=int(config.cnn_arch_hidden_size),
            pool_mode=str(config.cnn_arch_pool_mode),
        )
        global_head_state_arch = {
            k: v.clone() for k, v in global_model_arch.state_dict().items() if k.startswith("head")
        }
        user_head_states_arch = [copy.deepcopy(global_head_state_arch) for _ in range(config.num_users)]
        obs_buffers_arch = [deque(maxlen=W) for _ in range(config.num_users)]
        sample_buffers_arch = [deque(maxlen=S) for _ in range(config.num_users)]
        if config.meta_algorithm.lower() == "reptile":
            aggregator_arch = ReptileAggregator(
                step_size=config.reptile_step_size,
                use_aircomp=False,
                aircomp_simulator=None,
            )
        else:
            aggregator_arch = MetaUpdater(
                meta_algorithm="FedAvg",
                step_size=1.0,
                use_aircomp=False,
                aircomp_simulator=None,
            )
        f_beam_arch = f_beam.copy()
        theta_ota_arch = theta_ota.copy()
        logger.info(
            "CNN architecture ablation enabled=True "
            "(replace GRU with non-stateful CNN, keep FL/OTA/physics mechanism unchanged)."
        )

    enable_cnn_baseline = bool(config.enable_cnn_baseline)
    global_model_baseline = None
    aggregator_baseline = None
    f_beam_baseline = None
    theta_ota_baseline = None
    if enable_cnn_baseline:
        global_model_baseline = CSICNNBaseline(
            observation_dim=obs_dim,
            output_dim=output_dim,
            conv_filters=int(config.cnn_baseline_conv_filters),
            conv_kernel=int(config.cnn_baseline_conv_kernel),
            hidden_size=int(config.cnn_baseline_hidden_size),
        )
        aggregator_baseline = MetaUpdater(
            meta_algorithm="FedAvg",
            step_size=1.0,
            use_aircomp=False,
            aircomp_simulator=None,
        )
        f_beam_baseline = f_beam.copy()
        theta_ota_baseline = theta_ota.copy()
        logger.info(
            "Literature CNN baseline enabled=True "
            "(single-step input, full-model FedAvg, non-stateful)."
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
        local_data_arch = [] if enable_cnn_arch_ablation else None
        local_data_baseline = [] if enable_cnn_baseline else None
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

            if use_persistent_hidden_state:
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

            if enable_cnn_arch_ablation:
                Y_pilot_arch, cascaded_arch, direct_eff_arch = pilot_gen.simulate_pilot_observation(
                    H_BR, h_RUs[k], f_beam_arch, theta_pilot,
                    noise_std=float(pilot_noise_std_k[k]),
                    h_BU=h_BU_k,
                    link_switch=link_switch,
                )

                obs_real_arch = np.real(Y_pilot_arch)
                obs_imag_arch = np.imag(Y_pilot_arch)
                obs_step_arch = np.stack([obs_real_arch, obs_imag_arch], axis=0).astype(np.float32)

                # No state is used in architecture ablation.
                # Input mode follows GRU context setting: single-step or window.
                if use_time_window and W > 1:
                    obs_buffers_arch[k].append(obs_step_arch)
                    seq_arch = list(obs_buffers_arch[k])
                    X_seq_arch = np.stack(seq_arch, axis=0)
                    if X_seq_arch.shape[0] < W:
                        pad_len_arch = W - X_seq_arch.shape[0]
                        pad_arch = np.full((pad_len_arch, 2, obs_dim), pad_val, dtype=np.float32)
                        X_seq_arch = np.concatenate([pad_arch, X_seq_arch], axis=0)
                else:
                    X_seq_arch = obs_step_arch[None, :, :]  # (1, 2, P)

                total_effective_arch = np.concatenate(
                    [cascaded_arch, np.asarray([direct_eff_arch], dtype=cascaded_arch.dtype)],
                    axis=0,
                )
                y_arch = np.concatenate(
                    [np.real(total_effective_arch), np.imag(total_effective_arch)],
                    axis=0,
                ).astype(np.float32)
                sample_arch = (
                    X_seq_arch.astype(np.float32, copy=False),
                    y_arch.astype(np.float32, copy=False),
                )
                local_data_arch.append(sample_arch)
                if use_local_cache and use_time_window and W > 1:
                    sample_buffers_arch[k].append((sample_arch[0].copy(), sample_arch[1].copy()))

            if enable_cnn_baseline:
                Y_pilot_baseline, cascaded_baseline, direct_eff_baseline = pilot_gen.simulate_pilot_observation(
                    H_BR, h_RUs[k], f_beam_baseline, theta_pilot,
                    noise_std=float(pilot_noise_std_k[k]),
                    h_BU=h_BU_k,
                    link_switch=link_switch,
                )

                # Literature baseline uses memoryless single-step pilot input.
                obs_real_baseline = np.real(Y_pilot_baseline)
                obs_imag_baseline = np.imag(Y_pilot_baseline)
                obs_step_baseline = np.stack([obs_real_baseline, obs_imag_baseline], axis=0).astype(np.float32)
                X_seq_baseline = obs_step_baseline[None, :, :]  # (1, 2, P)

                total_effective_baseline = np.concatenate(
                    [cascaded_baseline, np.asarray([direct_eff_baseline], dtype=cascaded_baseline.dtype)],
                    axis=0,
                )
                y_baseline = np.concatenate(
                    [np.real(total_effective_baseline), np.imag(total_effective_baseline)],
                    axis=0,
                ).astype(np.float32)
                sample_baseline = (
                    X_seq_baseline.astype(np.float32, copy=False),
                    y_baseline.astype(np.float32, copy=False),
                )
                local_data_baseline.append(sample_baseline)

        # Local training on each user's data
        local_models = []
        losses = []
        local_models_arch = [] if enable_cnn_arch_ablation else None
        losses_arch = [] if enable_cnn_arch_ablation else None
        local_models_baseline = [] if enable_cnn_baseline else None
        losses_baseline = [] if enable_cnn_baseline else None
        for k in range(config.num_users):
            loss_arch = None
            loss_baseline = None

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

            if enable_cnn_arch_ablation:
                local_model_arch = CSICNNArch(
                    observation_dim=obs_dim,
                    output_dim=output_dim,
                    conv_filters=int(config.cnn_arch_conv_filters),
                    conv_kernel=int(config.cnn_arch_conv_kernel),
                    hidden_size=int(config.cnn_arch_hidden_size),
                    pool_mode=str(config.cnn_arch_pool_mode),
                )
                state_arch = global_model_arch.state_dict()
                for hk, hv in user_head_states_arch[k].items():
                    state_arch[hk] = hv.clone()
                local_model_arch.load_state_dict(state_arch)
                if use_local_cache and use_time_window and W > 1:
                    data_k_arch = list(sample_buffers_arch[k])
                else:
                    data_k_arch = [local_data_arch[k]]
                local_model_arch, loss_arch = trainer.train(local_model_arch, data_k_arch)
                losses_arch.append(loss_arch if loss_arch is not None else 0.0)
                local_models_arch.append(local_model_arch)
                user_head_states_arch[k] = {
                    name: param.detach().clone()
                    for name, param in local_model_arch.state_dict().items()
                    if name.startswith("head")
                }

            if enable_cnn_baseline:
                local_model_baseline = CSICNNBaseline(
                    observation_dim=obs_dim,
                    output_dim=output_dim,
                    conv_filters=int(config.cnn_baseline_conv_filters),
                    conv_kernel=int(config.cnn_baseline_conv_kernel),
                    hidden_size=int(config.cnn_baseline_hidden_size),
                )
                local_model_baseline.load_state_dict(global_model_baseline.state_dict())
                data_k_baseline = [local_data_baseline[k]]
                local_model_baseline, loss_baseline = trainer.train(local_model_baseline, data_k_baseline)
                losses_baseline.append(loss_baseline if loss_baseline is not None else 0.0)
                local_models_baseline.append(local_model_baseline)

            loss_parts = []
            if loss is not None:
                loss_parts.append(f"GRU: \033[34m{loss:.4f}\033[0m")
            if enable_cnn_arch_ablation and (loss_arch is not None):
                loss_parts.append(f"CNN-arch: \033[35m{loss_arch:.4f}\033[0m")
            if enable_cnn_baseline and (loss_baseline is not None):
                loss_parts.append(f"CNN-base: \033[36m{loss_baseline:.4f}\033[0m")
            if loss_parts:
                logger.info(f"User {k + 1} local loss -> " + ", ".join(loss_parts))
            else:
                logger.info(f"User {k + 1} local training done.")

        if losses:
            round_loss_parts = [f"Round {round_idx} mean local loss -> GRU: {np.mean(losses):.4f}"]
            if enable_cnn_arch_ablation and losses_arch:
                round_loss_parts.append(f"CNN-arch: {np.mean(losses_arch):.4f}")
            if enable_cnn_baseline and losses_baseline:
                round_loss_parts.append(f"CNN-base: {np.mean(losses_baseline):.4f}")
            logger.info(", ".join(round_loss_parts))

        # Aggregate updates at server
        logger.info("Aggregating GRU updates at server.")
        old_global_vec = state_dict_to_vector_backbone(global_model).detach().cpu()
        old_global_vec_arch = None
        if enable_cnn_arch_ablation:
            old_global_vec_arch = state_dict_to_vector_backbone(global_model_arch).detach().cpu()
        old_global_vec_baseline = None
        if enable_cnn_baseline:
            old_global_vec_baseline = state_dict_to_vector(global_model_baseline).detach().cpu()

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

        delta_mat_arch = None
        delta_var_arch = None
        if enable_cnn_arch_ablation:
            delta_list_arch = []
            for lm in local_models_arch:
                delta_list_arch.append(
                    model_delta_to_vector_backbone(lm, global_model_arch).detach().cpu()
                )
            delta_mat_arch = torch.stack(delta_list_arch, dim=0)  # [K, d]
            delta_var_arch = delta_mat_arch.float().var(dim=1, unbiased=False)
            delta_var_arch = torch.clamp(delta_var_arch, min=float(config.ota_var_floor))

        delta_mat_baseline = None
        delta_var_baseline = None
        if enable_cnn_baseline:
            delta_list_baseline = []
            for lm in local_models_baseline:
                delta_list_baseline.append(
                    model_delta_to_vector(lm, global_model_baseline).detach().cpu()
                )
            delta_mat_baseline = torch.stack(delta_list_baseline, dim=0)  # [K, d]
            delta_var_baseline = delta_mat_baseline.float().var(dim=1, unbiased=False)
            delta_var_baseline = torch.clamp(delta_var_baseline, min=float(config.ota_var_floor))

        h_eff = None
        h_eff_arch = None
        h_eff_baseline = None
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

            if enable_cnn_arch_ablation:
                casc_pref_arch = f_beam_arch.conj() @ H_BR.T
                h_eff_list_arch = []
                for k in range(config.num_users):
                    direct_arch = (
                        f_beam_arch.conj().dot(h_BUs[k]) if (direct_on == 1 and h_BUs is not None) else 0.0
                    )
                    reflect_arch = 0.0
                    if reflect_on == 1:
                        reflect_arch = np.dot(theta_ota_arch, casc_pref_arch * h_RUs[k])
                    h_eff_list_arch.append(direct_arch + reflect_arch)
                h_eff_arch = torch.from_numpy(np.asarray(h_eff_list_arch, dtype=np.complex64))

            if enable_cnn_baseline:
                casc_pref_baseline = f_beam_baseline.conj() @ H_BR.T
                h_eff_list_baseline = []
                for k in range(config.num_users):
                    direct_baseline = (
                        f_beam_baseline.conj().dot(h_BUs[k]) if (direct_on == 1 and h_BUs is not None) else 0.0
                    )
                    reflect_baseline = 0.0
                    if reflect_on == 1:
                        reflect_baseline = np.dot(theta_ota_baseline, casc_pref_baseline * h_RUs[k])
                    h_eff_list_baseline.append(direct_baseline + reflect_baseline)
                h_eff_baseline = torch.from_numpy(np.asarray(h_eff_list_baseline, dtype=np.complex64))

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

        if enable_cnn_arch_ablation:
            logger.info("Aggregating CNN architecture ablation updates at server.")
            if config.use_aircomp and aircomp_sim is not None:
                agg_update_arch, diag_arch = aircomp_sim.aggregate_updates(
                    updates=delta_mat_arch.float(),
                    h_eff=h_eff_arch,
                    user_weights=K_norm,
                )
                ideal_update_arch = (
                    (delta_mat_arch * K_norm.view(-1, 1)).sum(dim=0) / (K_norm.sum() + 1e-12)
                )
                agg_error_power_arch = torch.norm(agg_update_arch - ideal_update_arch) ** 2
                ideal_power_arch = torch.norm(ideal_update_arch) ** 2
                nmse_arch = agg_error_power_arch / (ideal_power_arch + 1e-12)

                aggregator_arch.apply_aggregated_delta(
                    global_model_arch,
                    agg_update_arch,
                    backbone_only=True,
                    prefix="backbone",
                )
                logger.info(
                    f"CNN-arch AirComp eta={diag_arch['eta']:.4e}, "
                    f"min|u|^2={diag_arch['min_inner2']:.4e}, "
                    f"agg_NMSE={nmse_arch.item():.4e}, "
                    f"agg_err={agg_error_power_arch.item():.4e}, "
                    f"ideal_power={ideal_power_arch.item():.4e}"
                )
            else:
                global_model_arch = aggregator_arch.aggregate(
                    global_model_arch,
                    local_models_arch,
                    backbone_only=True,
                    prefix="backbone",
                )

            new_global_vec_arch = state_dict_to_vector_backbone(global_model_arch).detach().cpu()
            logger.info(
                "CNN-arch global backbone update norm: "
                f"{torch.norm(new_global_vec_arch - old_global_vec_arch).item():.4e}"
            )

        if enable_cnn_baseline:
            logger.info("Aggregating literature CNN baseline updates at server.")
            if config.use_aircomp and aircomp_sim is not None:
                agg_update_baseline, diag_baseline = aircomp_sim.aggregate_updates(
                    updates=delta_mat_baseline.float(),
                    h_eff=h_eff_baseline,
                    user_weights=K_norm,
                )
                ideal_update_baseline = (
                    (delta_mat_baseline * K_norm.view(-1, 1)).sum(dim=0) / (K_norm.sum() + 1e-12)
                )
                agg_error_power_baseline = torch.norm(agg_update_baseline - ideal_update_baseline) ** 2
                ideal_power_baseline = torch.norm(ideal_update_baseline) ** 2
                nmse_baseline = agg_error_power_baseline / (ideal_power_baseline + 1e-12)

                aggregator_baseline.apply_aggregated_delta(global_model_baseline, agg_update_baseline, backbone_only=False)
                logger.info(
                    f"CNN-base AirComp eta={diag_baseline['eta']:.4e}, "
                    f"min|u|^2={diag_baseline['min_inner2']:.4e}, "
                    f"agg_NMSE={nmse_baseline.item():.4e}, "
                    f"agg_err={agg_error_power_baseline.item():.4e}, "
                    f"ideal_power={ideal_power_baseline.item():.4e}"
                )
            else:
                global_model_baseline = aggregator_baseline.aggregate(
                    global_model_baseline,
                    local_models_baseline,
                    backbone_only=False,
                )

            new_global_vec_baseline = state_dict_to_vector(global_model_baseline).detach().cpu()
            logger.info(
                "CNN-base global update norm: "
                f"{torch.norm(new_global_vec_baseline - old_global_vec_baseline).item():.4e}"
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

        if enable_cnn_arch_ablation:
            f_beam_arch, theta_ota_arch, nmse_proxy_arch = optimize_beam_ris(
                H_BR, h_RUs, h_BUs=h_BUs, theta_init=theta_ota_arch, f_init=f_beam_arch,
                link_switch=link_switch, user_weights=K_norm.numpy(),
                update_vars=delta_var_arch.numpy(),
                tx_power=config.ota_tx_power,
                noise_std=config.ota_noise_std,
                var_floor=config.ota_var_floor,
                eps=config.ota_eps,
            )
            logger.info(f"Optimized f (CNN-arch): {np.round(f_beam_arch, 4)}")
            logger.info(
                f"Optimized theta_ota (CNN-arch): {np.round(theta_ota_arch, 4)}, "
                f"proxy_NMSE={nmse_proxy_arch:.4e}"
            )

        if enable_cnn_baseline:
            f_beam_baseline, theta_ota_baseline, nmse_proxy_baseline = optimize_beam_ris(
                H_BR, h_RUs, h_BUs=h_BUs, theta_init=theta_ota_baseline, f_init=f_beam_baseline,
                link_switch=link_switch, user_weights=K_norm.numpy(),
                update_vars=delta_var_baseline.numpy(),
                tx_power=config.ota_tx_power,
                noise_std=config.ota_noise_std,
                var_floor=config.ota_var_floor,
                eps=config.ota_eps,
            )
            logger.info(f"Optimized f (CNN-base): {np.round(f_beam_baseline, 4)}")
            logger.info(
                f"Optimized theta_ota (CNN-base): {np.round(theta_ota_baseline, 4)}, "
                f"proxy_NMSE={nmse_proxy_baseline:.4e}"
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
