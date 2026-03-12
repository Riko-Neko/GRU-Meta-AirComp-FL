import hashlib
import json
import math


# Configuration parameters for the simulation
class Config:
    # Data and environment settings
    use_synthetic_data = True  # whether to generate synthetic data (if False, use DeepMIMO data)
    deepmimo_path = "DeepMIMO_O1.npy"  # path to DeepMIMO dataset file (if use_synthetic_data is False)
    num_users = 10  # number of users (K)
    num_bs_antennas = 32  # number of BS antennas (M)
    num_ris_elements = 64  # number of RIS elements (N)
    num_pilots = 8  # number of pilot transmissions per time slot (P)

    # Path-loss settings (synthetic mode)
    alpha_direct = 3.0  # path-loss exponent for BS-UE
    alpha_ris = 2.2  # path-loss exponent for RIS-UE (effective)
    d_direct_min = 100.0
    d_direct_max = 150.0
    d_ris_min = 50.0
    d_ris_max = 100.0
    channel_ref_scale = math.sqrt(1e-10)  # match baseline normalization ref=(1e-10)^0.5

    # Link switch: [reflection, direct]
    # [1,0] reflection only (default), [0,1] direct only (no RIS), [1,1] both, [0,0] invalid
    link_switch = [1, 0]

    # channel settings (RIS->UE link, RU)
    channel_alpha = 0.75  # AR(1)
    use_dynamic_alpha = False
    dynamic_alpha_mode = "sinusoid"  # "sinusoid" | "piecewise"
    alpha_min = 0.50
    alpha_max = 0.98
    alpha_period_rounds = 5
    alpha_piecewise = [(5, 0.98), (10, 0.85), (15, 0.60), (20, 0.90), (50, 0.50), (100, 0.60), (150, 0.70), (200, 0.80),
                       (250, 0.90)]

    # per-user heterogeneity
    # channel dynamic (alpha_k heterogeneity)
    use_user_alpha_hetero = True
    alpha_user_min = 0.60
    alpha_user_max = 0.98
    # Pilot observation noise
    use_user_pilot_snr_hetero = False
    pilot_SNR_dB = 20
    pilot_snr_dB_min = 10
    pilot_snr_dB_max = 30

    # GRU context settings
    gru_context_mode = "persistent_hidden"  # "persistent_hidden" | "time_window"
    gru_csi_target_mode = "t+1"  # GRU output selection: "t" | "t+1" | "uplink_linear"
    uplink_tau_ratio = 0.5  # tau / delta_t for uplink instant, used only when gru_csi_target_mode="uplink_linear"
    window_length = 8  # time window length（5~20）
    window_pad_value = 0.0  # padding value in initial stage
    reset_hidden_on_round1 = True
    reset_hidden_on_large_backbone_update = False
    hidden_reset_update_norm_threshold = 0.5

    # Literature-style CNN baseline settings
    enable_cnn_baseline = True
    cnn_baseline_conv_filters = 8
    cnn_baseline_conv_kernel = 3
    cnn_baseline_hidden_size = 32

    # Pure architecture ablation (replace GRU backbone with non-stateful CNN)
    enable_cnn_arch_ablation = False
    cnn_arch_conv_filters = 8
    cnn_arch_conv_kernel = 3
    cnn_arch_hidden_size = 32
    cnn_arch_pool_mode = "last"  # "last" or "mean"

    # Federated learning settings
    num_rounds = 50  # number of communication rounds
    local_epochs = 3  # local update epochs per round (per user)
    local_lr = 1e-3  # learning rate for local training
    batch_size = 8  # batch size for local training (None means use all data per epoch)
    meta_algorithm = "Reptile"  # Meta-FL algorithm ("Reptile" or "FedAvg")
    reptile_step_size = 0.2  # step size (beta) for Reptile algorithm (if applicable)

    # AirComp and communication settings
    use_aircomp = True  # whether to simulate AirComp aggregation with noise
    SNR_dB = 0  # SNR in dB for AirComp (if use_aircomp is True)
    noise_std = math.pow(10, - (SNR_dB / 20.0))  # noise standard deviation from SNR_dB (20 dB -> ~0.1)
    # OTA aggregation settings (Phase1)
    ota_tx_power = 0.1
    ota_var_floor = 1e-3
    ota_eps = 1e-8
    ota_noise_std = noise_std
    ota_use_weighted_users = True  # if False, always use equal weights (all ones)
    # Optional user-data partition profile (equal/grouped) for experiment bookkeeping.
    # Runtime n_k used by OTA aggregation is computed from actual per-user local sample counts.
    user_data_partition_mode = "grouped"  # "equal" | "grouped"
    user_data_size_equal = 10
    user_group_ratios = [0.3, 0.4, 0.3]  # small/medium/large user fractions, auto-normalized
    user_group_data_sizes = [5, 10, 20]  # small/medium/large n_k for persistent segments

    # OA optimizer mode
    oa_optimizer_mode = "legacy"  # "legacy" | "state_aware"
    oa_ao_iters = 2
    oa_theta_lr = 0.05
    oa_theta_grad_steps = 1
    oa_normalize_f = True

    # State-aware OA settings
    state_beta_z = 0.1
    state_alpha_z = 0.5
    state_alpha_x = 0.5
    state_mu = 0.8
    state_weight_min = 0.7
    state_weight_max = 2.0
    state_fast_clip = 2.0
    state_eps = 1e-8
    state_strategy = "stability"  # "protect" | "stability"

    # Logging settings
    log_to_file = True  # whether to save logs to a file
    log_file_path = "demo.log"  # log file path if enabled

    # Debug visualization: per-user Reptile head parameter projection
    enable_reptile_head_debug_plot = False
    reptile_head_debug_every = 10  # save one projection figure every N rounds
    reptile_head_debug_root = "debug"
    enable_gru_state_diff_debug_plot = False
    gru_state_diff_debug_every = 10  # save one GRU state-diff figure every N rounds

    @classmethod
    def as_dict(cls):
        d = {}
        for k, v in vars(cls).items():
            if k.startswith("_"):
                continue
            if k in {
                "enable_reptile_head_debug_plot",
                "reptile_head_debug_every",
                "reptile_head_debug_root",
                "enable_gru_state_diff_debug_plot",
                "gru_state_diff_debug_every",
            }:
                continue
            if callable(v):
                continue
            if isinstance(v, tuple):
                v = list(v)
            elif isinstance(v, set):
                v = sorted(list(v))
            d[k] = v
        return d

    @classmethod
    def fingerprint(cls, *, length: int = 12) -> str:
        payload = json.dumps(cls.as_dict(), sort_keys=True, default=str, ensure_ascii=True)
        h = hashlib.sha1(payload.encode("utf-8")).hexdigest()
        return h[: max(6, int(length))]

    @classmethod
    def log_prefix(cls) -> str:
        return (f"K{cls.num_users}_M{cls.num_bs_antennas}_N{cls.num_ris_elements}_"
                f"P{cls.num_pilots}_W{cls.window_length}_"
                f"{cls.meta_algorithm}_"
                f"CTX{cls.gru_context_mode}"
                )
