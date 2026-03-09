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
    num_pilots = 16  # number of pilot transmissions per time slot (P)

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
    channel_alpha = 0.9  # AR(1)
    use_dynamic_alpha = True
    dynamic_alpha_mode = "sinusoid"  # "sinusoid" | "piecewise"
    alpha_min = 0.50
    alpha_max = 0.98
    alpha_period_rounds = 20
    alpha_piecewise = [(5, 0.98), (10, 0.85), (15, 0.60), (20, 0.90)]

    # per-user heterogeneity
    # channel dynamic (alpha_k heterogeneity)
    use_user_alpha_hetero = True
    alpha_user_min = 0.60
    alpha_user_max = 0.98
    # Pilot observation noise
    use_user_pilot_snr_hetero = True
    pilot_SNR_dB = 20
    pilot_snr_dB_min = 10
    pilot_snr_dB_max = 30

    # GRU context settings
    gru_context_mode = "persistent_hidden" # "persistent_hidden" | "time_window"
    window_length = 8  # time window length（5~20）
    window_pad_value = 0.0  # padding value in initial stage
    reset_hidden_on_round1 = True
    reset_hidden_on_large_backbone_update = False
    hidden_reset_update_norm_threshold = 0.5

    # Literature-style CNN baseline settings
    enable_cnn_baseline = True
    cnn_baseline_conv_filters = 16
    cnn_baseline_conv_kernel = 3
    cnn_baseline_hidden_size = 64

    # Pure architecture ablation (replace GRU backbone with non-stateful CNN)
    enable_cnn_arch_ablation = True
    cnn_arch_conv_filters = 8
    cnn_arch_conv_kernel = 3
    cnn_arch_hidden_size = 32
    cnn_arch_pool_mode = "last"  # "last" or "mean"

    # Local sample cache (per-user) for stable local training
    use_local_sample_cache = True
    local_cache_size = 8  # S sample cached per-user

    # Federated learning settings
    num_rounds = 100  # number of communication rounds
    local_epochs = 3  # local update epochs per round (per user)
    local_lr = 1e-3  # learning rate for local training
    batch_size = 32  # batch size for local training (None means use all data per epoch)
    meta_algorithm = "Reptile"  # Meta-FL algorithm ("Reptile" or "FedAvg")
    reptile_step_size = 0.1  # step size (beta) for Reptile algorithm (if applicable)

    # AirComp and communication settings
    use_aircomp = True  # whether to simulate AirComp aggregation with noise
    SNR_dB = 20  # SNR in dB for AirComp (if use_aircomp is True)
    noise_std = math.pow(10, - (SNR_dB / 20.0))  # noise standard deviation from SNR_dB (20 dB -> ~0.1)
    # OTA aggregation settings (Phase1)
    ota_tx_power = 0.1
    ota_var_floor = 1e-3
    ota_eps = 1e-8
    ota_user_weight = 1.0  # scalar base weight; Phase1 uses uniform by default
    ota_noise_std = noise_std
    ota_use_weighted_users = True
    user_weight_mode = "uniform"  # "uniform" or "random"
    user_data_size_min = 5000
    user_data_size_max = 20000

    # Logging settings
    log_to_file = True  # whether to save logs to a file
    log_file_path = "demo.log"  # log file path if enabled

    @classmethod
    def as_dict(cls):
        d = {}
        for k, v in vars(cls).items():
            if k.startswith("_"):
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
                f"S{cls.local_cache_size}_{cls.meta_algorithm}_"
                f"CTX{cls.gru_context_mode}"
                )
