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
    num_pilots = 4  # number of pilot transmissions per time slot (P)

    # Mobility-driven synthetic geometry
    bs_position_xy = [0.0, 0.0]
    ris_position_xy = [30.0, 0.0]
    user_cluster_ratios = [0.5, 0.5]  # near / far user clusters
    # user_cluster_ratios = [1]  # near / far user clusters
    user_cluster_centers_xy = [[40.0, 8.0], [80.0, -12.0]]
    user_cluster_position_jitter_xy = [[10.0, 10.0], [10.0, 10.0]]
    # user_cluster_centers_xy = [[50.0, 0.0]]
    # user_cluster_position_jitter_xy = [[10.0, 10.0]]
    user_speed_range = [1, 5]  # m/s; sampled per-user, then combined with a random direction
    user_motion_direction_deg = None  # None=random direction per-user; float=fixed direction for all users
    # user_speed_user_mask = []  # 1 means all users move; list of 1-based user ids moves only those users
    user_speed_user_mask = [7, 8, 9, 10]
    channel_time_step = 1e-3  # seconds between consecutive channel samples
    channel_carrier_frequency_hz = 3.5e9
    channel_min_distance = 1.0
    alpha_direct = 3.0  # path-loss exponent for BS-UE direct link
    channel_ref_scale = math.sqrt(1e-10)  # match baseline normalization ref=(1e-10)^0.5

    # Link switch: [reflection, direct]
    # [1,0] reflection only (default), [0,1] direct only (no RIS), [1,1] both, [0,0] invalid
    link_switch = [1, 1]

    # Pilot observation noise
    pilot_SNR_dB = 20

    # GRU context settings
    gru_context_mode = "persistent_hidden"  # "persistent_hidden" | "time_window"
    gru_csi_target_mode = "uplink_direct"  # GRU output selection: "t" | "uplink_linear" | "uplink_direct"
    uplink_tau_ratio = 0.5  # tau / delta_t for uplink instant; used to generate h_tau and the uplink_linear reference
    enable_mobility_aware_loss_weighting = True
    mobility_tau_loss_weight_gain = 4.0
    mobility_tau_loss_weight_min = 1.0
    mobility_tau_loss_weight_max = 4.0
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

    # Formula-based LMMSE baseline on current pilots, evaluated with hold-last uplink CSI.
    enable_lmmse_baseline = True
    lmmse_prior_var = 1.0

    # Pure architecture ablation (replace GRU backbone with non-stateful CNN)
    enable_cnn_arch_ablation = True
    cnn_arch_conv_filters = 8
    cnn_arch_conv_kernel = 3
    cnn_arch_hidden_size = 32
    cnn_arch_pool_mode = "last"  # "last" or "mean"

    # Federated learning settings
    num_rounds = 100  # number of communication rounds
    local_epochs = 3  # local update epochs per round (per user)
    local_lr = 1e-3
    local_lr_gru = local_lr
    local_lr_arch = local_lr
    local_lr_base = local_lr
    local_optimizer_gru = "adam"  # "adam" | "sgd"
    local_optimizer_arch = "adam"  # "adam" | "sgd"
    local_optimizer_base = "adam"  # "adam" | "sgd"
    local_momentum_gru = 0.0  # used only when optimizer is SGD
    local_momentum_arch = 0.0  # used only when optimizer is SGD
    local_momentum_base = 0.0  # used only when optimizer is SGD
    batch_size = 8  # batch size for local training (None means use all data per epoch)
    meta_algorithm = "Reptile"  # Meta-FL algorithm ("Reptile" or "FedAvg")
    reptile_step_size = 0.2  # step size (beta) for Reptile algorithm (if applicable)
    enable_gru_pl_factorization = True
    gru_pl_loss_weight = 0.05
    gru_pl_eps = 1e-12
    enable_gru_semantic_grouping = True
    gru_group_switch_min_round = 10
    gru_group_switch_patience = 3
    gru_group_switch_ema_lambda = 0.8
    gru_group_switch_tau_b = 0.08
    gru_group_switch_tau_d = 0.08
    gru_group_eps = 1e-8
    gru_group_freeze_after_switch = True
    gru_groupwise_standardization = True
    gru_restart_training_after_switch = False
    gru_head_reset_to_group_mean_on_switch = False
    gru_head_randomize_on_switch = False
    gru_head_disable_persistence_after_switch = False
    gru_log_group_head_dispersion = False
    gru_reset_hidden_on_group_switch = False
    gru_reset_hidden_on_group_change_each_round = False
    gru_disable_persistent_hidden_after_switch = False
    gru_group_lambda_d = 8.0
    gru_group_c_beta = 0.5
    gru_group_c_d = 0.5
    gru_group_lambda_s = 0.0
    gru_group_gamma = 0.0
    gru_group_lambda_h = 0.0
    gru_group_tau = 0.0
    gru_group_k_min = 2
    gru_group_sca_max_iters = 8
    gru_group_sca_tol = 1e-4
    gru_group_relaxation = 0.7
    gru_group_lp_method = "highs"

    # AirComp and communication settings
    use_aircomp = True  # whether to simulate AirComp aggregation with noise
    SNR_dB = 0  # SNR in dB for AirComp (if use_aircomp is True)
    noise_std = math.pow(10, - (SNR_dB / 20.0))  # noise standard deviation from SNR_dB (20 dB -> ~0.1)
    # OTA aggregation settings (Phase1)
    ota_tx_power = 0.1
    ota_var_floor = 1e-3
    ota_eps = 1e-8
    ota_noise_std = noise_std
    ota_use_estimated_h_ru_for_aggregation = True  # if True, OTA h_eff uses branch-specific estimated h_RU; direct h_BU remains true
    ota_use_weighted_users = True  # if False, always use equal weights (all ones)
    oa_iters = 20
    beam_ris_optimizer = "sca"  # "oa" | "sca"
    sca_iters = 20
    sca_threshold = 1e-2
    sca_tau = 1.0
    # Optional user-data partition profile (equal/grouped) for experiment bookkeeping.
    # Runtime n_k used by OTA aggregation is computed from actual per-user local sample counts.
    user_data_partition_mode = "equal"  # "equal" | "grouped"
    user_data_size_equal = 10
    user_group_ratios = [0.3, 0.4, 0.3]  # small/medium/large user fractions, auto-normalized
    user_group_data_sizes = [1, 10, 50]  # small/medium/large n_k for persistent segments

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
    enable_eta_response_snapshot = False
    eta_response_snapshot_root = "debug/eta_response_snapshots"
    eta_response_snapshot_every = 1
    enable_mobility_debug_snapshot = False
    mobility_debug_snapshot_root = "debug/mobility_snapshots"
    mobility_debug_snapshot_every = 1
    enable_delta_motion_debug_snapshot = False
    delta_motion_debug_snapshot_root = "debug/delta_motion_snapshots"
    delta_motion_debug_snapshot_every = 1
    enable_gru_pl_debug_snapshot = False
    gru_pl_debug_snapshot_root = "debug/gru_pl_snapshots"
    gru_pl_debug_snapshot_every = 1

    @staticmethod
    def _slug_value(value) -> str:
        if value is None:
            return "none"
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, float):
            text = f"{value:.6g}"
        elif isinstance(value, (list, tuple)):
            text = "-".join(Config._slug_value(v) for v in value)
        else:
            text = str(value)
        return text.replace(".", "p").replace("-", "m").replace(" ", "")

    @classmethod
    def experiment_exclude_keys(cls):
        return {
            "beam_ris_optimizer",
            "oa_iters",
            "sca_iters",
            "sca_threshold",
            "sca_tau",
            "enable_cnn_baseline",
            "cnn_baseline_conv_filters",
            "cnn_baseline_conv_kernel",
            "cnn_baseline_hidden_size",
            "enable_lmmse_baseline",
            "lmmse_prior_var",
            "enable_cnn_arch_ablation",
            "cnn_arch_conv_filters",
            "cnn_arch_conv_kernel",
            "cnn_arch_hidden_size",
            "cnn_arch_pool_mode",
            "log_to_file",
            "log_file_path",
            "log_oa_vectors",
            "enable_reptile_head_debug_plot",
            "reptile_head_debug_every",
            "reptile_head_debug_root",
            "enable_gru_dual_target_debug_log",
            "enable_gru_state_diff_debug_plot",
            "gru_state_diff_debug_every",
            "enable_eta_response_snapshot",
            "eta_response_snapshot_root",
            "eta_response_snapshot_every",
            "enable_mobility_debug_snapshot",
            "mobility_debug_snapshot_root",
            "mobility_debug_snapshot_every",
            "enable_delta_motion_debug_snapshot",
            "delta_motion_debug_snapshot_root",
            "delta_motion_debug_snapshot_every",
            "enable_gru_pl_debug_snapshot",
            "gru_pl_debug_snapshot_root",
            "gru_pl_debug_snapshot_every",
        }

    @classmethod
    def as_dict(cls, *, exclude_keys=None):
        exclude = set(exclude_keys or [])
        d = {}
        for k, v in vars(cls).items():
            if k.startswith("_"):
                continue
            if k in exclude:
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
    def fingerprint(cls, *, length: int = 12, exclude_keys=None) -> str:
        payload = json.dumps(cls.as_dict(exclude_keys=exclude_keys), sort_keys=True, default=str, ensure_ascii=True)
        h = hashlib.sha1(payload.encode("utf-8")).hexdigest()
        return h[: max(6, int(length))]

    @classmethod
    def experiment_fingerprint(cls, *, length: int = 10) -> str:
        return cls.fingerprint(length=length, exclude_keys=cls.experiment_exclude_keys())

    @classmethod
    def _meta_abbrev(cls) -> str:
        token = str(cls.meta_algorithm).strip().lower().replace("-", "_").replace(" ", "")
        if token in {"fedavg", "fed_avg"}:
            return "FA"
        if token == "reptile":
            return "RP"
        return cls._slug_value(cls.meta_algorithm)

    @classmethod
    def optimizer_tag(cls) -> str:
        mode = str(cls.beam_ris_optimizer).lower()
        if mode == "sca":
            return f"S{int(cls.sca_iters)}T{cls._slug_value(cls.sca_threshold)}U{cls._slug_value(cls.sca_tau)}"
        return f"O{int(cls.oa_iters)}"

    @staticmethod
    def _optimizer_abbrev(name: str) -> str:
        name = str(name).strip().lower()
        if name == "adam":
            return "A"
        if name == "sgd":
            return "S"
        return "X"

    @classmethod
    def log_prefix(cls) -> str:
        data_tag = "SYN" if bool(cls.use_synthetic_data) else f"RIS-{cls._slug_value(cls.risdata_subset)}"
        batch_tag = "ALL" if cls.batch_size is None else cls._slug_value(cls.batch_size)
        if str(cls.user_data_partition_mode).lower() == "equal":
            nk_tag = f"NK{cls._slug_value(cls.user_data_size_equal)}"
        else:
            nk_tag = (
                f"NKR{cls._slug_value(cls.user_group_ratios)}_"
                f"NKS{cls._slug_value(cls.user_group_data_sizes)}"
            )
        opt_tag = (
            f"OP{cls._optimizer_abbrev(cls.local_optimizer_gru)}"
            f"{cls._optimizer_abbrev(cls.local_optimizer_arch)}"
            f"{cls._optimizer_abbrev(cls.local_optimizer_base)}"
        )
        mom_tag = (
            f"MM{cls._slug_value(cls.local_momentum_gru)}-"
            f"{cls._slug_value(cls.local_momentum_arch)}-"
            f"{cls._slug_value(cls.local_momentum_base)}"
        )
        debug_head_tag = ""
        if (
                bool(getattr(cls, "gru_restart_training_after_switch", False))
                or bool(getattr(cls, "gru_head_reset_to_group_mean_on_switch", False))
                or bool(getattr(cls, "gru_head_randomize_on_switch", False))
                or bool(getattr(cls, "gru_head_disable_persistence_after_switch", False))
        ):
            debug_head_tag = (
                f"_GR{int(bool(cls.gru_restart_training_after_switch))}"
                f"H{int(bool(cls.gru_head_reset_to_group_mean_on_switch))}"
                f"{int(bool(cls.gru_head_randomize_on_switch))}"
                f"{int(bool(cls.gru_head_disable_persistence_after_switch))}"
            )
        debug_hidden_tag = ""
        if (
                bool(getattr(cls, "gru_reset_hidden_on_group_switch", False))
                or bool(getattr(cls, "gru_reset_hidden_on_group_change_each_round", False))
                or bool(getattr(cls, "gru_disable_persistent_hidden_after_switch", False))
        ):
            debug_hidden_tag = (
                f"_GX{int(bool(cls.gru_reset_hidden_on_group_switch))}"
                f"{int(bool(cls.gru_reset_hidden_on_group_change_each_round))}"
                f"{int(bool(cls.gru_disable_persistent_hidden_after_switch))}"
            )
        debug_stats_tag = ""
        if bool(getattr(cls, "gru_groupwise_standardization", False)):
            debug_stats_tag = "_GS1"
        return (
            f"{data_tag}_K{int(cls.num_users)}_M{int(cls.num_bs_antennas)}_N{int(cls.num_ris_elements)}_"
            f"P{int(cls.num_pilots)}_L{cls._slug_value(cls.link_switch)}_PSNR{cls._slug_value(cls.pilot_SNR_dB)}_"
            f"CTX{cls._slug_value(cls.gru_context_mode)}_GT{cls._slug_value(cls.gru_csi_target_mode)}_"
            f"RHO{cls._slug_value(cls.uplink_tau_ratio)}_W{int(cls.window_length)}_"
            f"FL{cls._meta_abbrev()}_R{int(cls.num_rounds)}_E{int(cls.local_epochs)}_"
            f"B{batch_tag}_AIR{cls._slug_value(cls.use_aircomp)}_"
            f"SNR{cls._slug_value(cls.SNR_dB)}_TX{cls._slug_value(cls.ota_tx_power)}_"
            f"OC{cls._slug_value(cls.ota_use_estimated_h_ru_for_aggregation)}_"
            f"VF{cls._slug_value(cls.ota_var_floor)}_EPS{cls._slug_value(cls.ota_eps)}_"
            f"PF{cls._slug_value(cls.enable_gru_pl_factorization)}W{cls._slug_value(cls.gru_pl_loss_weight)}_"
            f"GG{cls._slug_value(cls.enable_gru_semantic_grouping)}F{cls._slug_value(cls.gru_group_freeze_after_switch)}_"
            f"WU{cls._slug_value(cls.ota_use_weighted_users)}_DT{cls._slug_value(cls.channel_time_step)}_"
            f"FC{cls._slug_value(cls.channel_carrier_frequency_hz)}_AD{cls._slug_value(cls.alpha_direct)}_"
            f"SPD{cls._slug_value(cls.user_speed_range)}_DIR{cls._slug_value(cls.user_motion_direction_deg)}_"
            f"{opt_tag}_{mom_tag}_"
            f"PART{cls._slug_value(cls.user_data_partition_mode)}_{nk_tag}{debug_head_tag}{debug_hidden_tag}{debug_stats_tag}"
        )
