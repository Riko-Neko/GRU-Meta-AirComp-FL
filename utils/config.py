import hashlib
import json
import math


# Configuration parameters for the simulation
class Config:
    # Data and environment settings
    use_synthetic_data = True  # whether to generate synthetic data (if False, use RIS-S21 measurement data)
    risdata_root = "dataset"  # root dir of RIS-S21 dataset when use_synthetic_data is False
    risdata_subset = "specular"
    risdata_result_key = "rand"
    risdata_reference_key = "RISallOff"
    risdata_freq_hz = 5.375e9
    risdata_freq_tol_hz = 30e6
    risdata_max_pattern_samples = None
    risdata_min_snr_db = None
    num_users = 10  # number of users (K)
    num_bs_antennas = 32  # number of BS antennas (M)
    num_ris_elements = 64  # number of RIS elements (N)
    num_pilots = 16  # number of pilot transmissions per time slot (P)

    # Mobility-driven synthetic geometry
    bs_position_xy = [0.0, 0.0]
    ris_position_xy = [30.0, 0.0]
    user_cluster_ratios = [0.5, 0.5]  # near / far user clusters
    # user_cluster_ratios = [1]  # near / far user clusters
    user_cluster_centers_xy = [[40.0, 8.0], [80.0, -12.0]]
    user_cluster_position_jitter_xy = [[10.0, 10.0], [10.0, 10.0]]
    # user_cluster_centers_xy = [[50.0, 0.0]]
    # user_cluster_position_jitter_xy = [[1.0, 1.0]]
    user_speed_range = [10, 20]  # m/s; sampled per-user, then combined with a random direction
    user_motion_direction_deg = 1.5  # None=random direction per-user; float=fixed direction for all users
    # user_speed_user_mask = 1  # 1 means all users move; list of 1-based user ids moves only those users
    user_speed_user_mask = [9, 10]
    channel_time_step = 1e-3  # seconds between consecutive channel samples
    channel_carrier_frequency_hz = 3.5e9
    channel_min_distance = 1.0
    alpha_direct = 3.0  # path-loss exponent for BS-UE direct link
    channel_ref_scale = math.sqrt(1e-10)  # match baseline normalization ref=(1e-10)^0.5

    # Link switch: [reflection, direct]
    # [1,0] reflection only (default), [0,1] direct only (no RIS), [1,1] both, [0,0] invalid
    link_switch = [1, 0]

    # Pilot observation noise
    pilot_SNR_dB = 20

    # GRU context settings
    gru_context_mode = "persistent_hidden"  # "persistent_hidden" | "time_window"
    gru_csi_target_mode = "uplink_linear"  # GRU output selection: "t" | "t+1" | "uplink_linear"
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
    enable_cnn_arch_ablation = True
    cnn_arch_conv_filters = 8
    cnn_arch_conv_kernel = 3
    cnn_arch_hidden_size = 32
    cnn_arch_pool_mode = "last"  # "last" or "mean"

    # Federated learning settings
    num_rounds = 100  # number of communication rounds
    local_epochs = 3  # local update epochs per round (per user)
    local_lr = 1e-3  # learning rate for local training
    batch_size = 8  # batch size for local training (None means use all data per epoch)
    meta_algorithm = "Reptile"  # Meta-FL algorithm ("Reptile" or "FedAvg")
    reptile_step_size = 0.2  # step size (beta) for Reptile algorithm (if applicable)

    # AirComp and communication settings
    use_aircomp = True  # whether to simulate AirComp aggregation with noise
    SNR_dB = 20  # SNR in dB for AirComp (if use_aircomp is True)
    noise_std = math.pow(10, - (SNR_dB / 20.0))  # noise standard deviation from SNR_dB (20 dB -> ~0.1)
    # OTA aggregation settings (Phase1)
    ota_tx_power = 0.1
    ota_var_floor = 1e-3
    ota_eps = 1e-8
    ota_noise_std = noise_std
    ota_use_weighted_users = True  # if False, always use equal weights (all ones)
    # Optional user-data partition profile (equal/grouped) for experiment bookkeeping.
    # Runtime n_k used by OTA aggregation is computed from actual per-user local sample counts.
    user_data_partition_mode = "equal"  # "equal" | "grouped"
    user_data_size_equal = 10
    user_group_ratios = [0.3, 0.4, 0.3]  # small/medium/large user fractions, auto-normalized
    user_group_data_sizes = [1, 10, 50]  # small/medium/large n_k for persistent segments

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
    log_oa_vectors = False  # whether to print optimized beamformer/RIS vectors

    # Debug visualization: per-user Reptile head parameter projection
    enable_reptile_head_debug_plot = False
    reptile_head_debug_every = 10  # save one projection figure every N rounds
    reptile_head_debug_root = "debug"
    enable_gru_dual_target_debug_log = False
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
                "enable_gru_dual_target_debug_log",
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
