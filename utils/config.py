import hashlib
import json
import math


# Configuration parameters for the simulation
class Config:
    # Data and environment settings
    use_synthetic_data = True  # whether to generate synthetic data (if False, use DeepMIMO data)
    deepmimo_path = "DeepMIMO_O1.npy"  # path to DeepMIMO dataset file (if use_synthetic_data is False)
    num_users = 10  # number of users (K)
    # num_times = 16                     # number of time steps (W) for synthetic data or sequence length
    num_bs_antennas = 32  # number of BS antennas (M)
    num_ris_elements = 64  # number of RIS elements (N)
    num_pilots = 16  # number of pilot transmissions per time slot (P)

    # channel settings (RIS->UE link, RU)
    channel_alpha = 0.9  # AR(1)
    use_user_alpha_hetero = True  # User-level channel dynamic difficulty (alpha_k heterogeneity)
    alpha_user_min = 0.60
    alpha_user_max = 0.98
    use_dynamic_alpha = False
    dynamic_alpha_mode = "sinusoid"  # "sinusoid" | "piecewise"
    alpha_min = 0.50
    alpha_max = 0.98
    alpha_period_rounds = 20
    alpha_piecewise = [(10, 0.98), (25, 0.85), (40, 0.60), (50, 0.90)]

    # Pilot observation noise (per-user heterogeneity)
    use_user_pilot_snr_hetero = True
    pilot_SNR_dB = 20
    pilot_snr_dB_min = 10
    pilot_snr_dB_max = 30

    # GRU time-window settings
    use_time_window = True
    window_length = 8  # time window length（5~20）
    window_pad_value = 0.0  # padding value in initial stage

    # Local sample cache (per-user) for stable local training
    use_local_sample_cache = True
    local_cache_size = 8  # S sample cached per-user

    # Federated learning settings
    num_rounds = 50  # number of communication rounds
    local_epochs = 5  # local update epochs per round (per user)
    local_lr = 1e-4  # learning rate for local training
    batch_size = 32  # batch size for local training (None means use all data per epoch)
    meta_algorithm = "Reptile"  # Meta-FL algorithm ("Reptile" or "FedAvg")
    reptile_step_size = 0.1  # step size (beta) for Reptile algorithm (if applicable)

    # AirComp and communication settings
    use_aircomp = True  # whether to simulate AirComp aggregation with noise
    SNR_dB = 20  # SNR in dB for AirComp (if use_aircomp is True)
    noise_std = math.pow(10,
                         - (SNR_dB / 20.0))  # noise standard deviation (amplitude) computed from SNR_dB (20 dB -> ~0.1)

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
                f"P{cls.num_pilots}_W{getattr(cls, 'window_length', 'NA')}_"
                f"S{getattr(cls, 'local_cache_size', 'NA')}_{cls.meta_algorithm}"
                )
