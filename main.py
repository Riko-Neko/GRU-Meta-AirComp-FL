from collections import deque
import copy
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from aircomp_opt.OTA_sim import AirCompSimulator
from aircomp_opt.f_theta_optim import optimize_beam_ris, optimize_beam_ris_state_aware
from data import deepmimo, pilot_gen
from data.channel import build_ru_channel_evolver_from_config
from fl_core.agg import MetaUpdater
from fl_core.reptile_agg import ReptileAggregator
from fl_core.state_metrics import (
    build_state_weights,
    combine_state_scores,
    compute_head_distance,
    compute_hidden_drift_ratio,
    update_ewma_state,
)
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


def _parse_gru_target_mode(mode):
    token = str(mode).strip().lower().replace(" ", "")
    if token in {"t", "current", "now"}:
        return "t"
    if token in {"t+1", "t_plus_1", "next", "t1"}:
        return "t+1"
    raise ValueError("Config.gru_csi_target_mode must be 't' or 't+1'")


def _allocate_counts_by_ratio(total_count, ratios):
    """Allocate integer counts by ratio with largest-remainder rounding."""
    total_count = int(total_count)
    if total_count <= 0:
        raise ValueError(f"total_count must be positive, got {total_count}")
    ratios = np.asarray(ratios, dtype=np.float64).reshape(-1)
    if ratios.size == 0:
        raise ValueError("ratios must be non-empty")
    if np.any(ratios < 0):
        raise ValueError("ratios must be non-negative")
    ratio_sum = float(np.sum(ratios))
    if ratio_sum <= 0.0:
        raise ValueError("sum(ratios) must be positive")
    ratios = ratios / ratio_sum

    raw = ratios * float(total_count)
    counts = np.floor(raw).astype(np.int64)
    remainder = total_count - int(np.sum(counts))
    if remainder > 0:
        frac = raw - counts.astype(np.float64)
        order = np.argsort(-frac)
        for idx in range(remainder):
            counts[order[idx % order.size]] += 1
    return counts


def _build_user_data_partitions(config):
    """
    Build a per-user virtual data partition and derive n_k from partition lengths.
    Returns:
      n_k: np.ndarray [K], float32
      mode: "equal" or "grouped"
      group_counts: np.ndarray [3], int64, counts for small/medium/large groups
    """
    k_users = int(config.num_users)
    if k_users <= 0:
        raise ValueError(f"Config.num_users must be positive, got {k_users}")

    mode = str(getattr(config, "user_data_partition_mode", "equal")).strip().lower()
    if mode == "equal":
        n_equal = int(getattr(config, "user_data_size_equal", 1))
        if n_equal <= 0:
            raise ValueError(f"Config.user_data_size_equal must be positive, got {n_equal}")
        user_sizes = np.full((k_users,), n_equal, dtype=np.int64)
        group_counts = np.array([k_users, 0, 0], dtype=np.int64)
    elif mode == "grouped":
        group_sizes = np.asarray(getattr(config, "user_group_data_sizes", [1, 1, 1]), dtype=np.int64).reshape(-1)
        group_ratios = np.asarray(getattr(config, "user_group_ratios", [1.0, 1.0, 1.0]), dtype=np.float64).reshape(-1)
        if group_sizes.size != 3:
            raise ValueError(
                f"Config.user_group_data_sizes must contain 3 values [small, medium, large], got {group_sizes.size}"
            )
        if group_ratios.size != 3:
            raise ValueError(
                f"Config.user_group_ratios must contain 3 values [small, medium, large], got {group_ratios.size}"
            )
        if np.any(group_sizes <= 0):
            raise ValueError(f"Config.user_group_data_sizes must be positive, got {group_sizes.tolist()}")
        group_counts = _allocate_counts_by_ratio(k_users, group_ratios)

        user_sizes = np.empty((k_users,), dtype=np.int64)
        start = 0
        for group_idx in range(3):
            end = start + int(group_counts[group_idx])
            user_sizes[start:end] = int(group_sizes[group_idx])
            start = end
    else:
        raise ValueError("Config.user_data_partition_mode must be 'equal' or 'grouped'")

    # Build virtual partition ranges and derive n_k from exact partition lengths.
    partition_ranges = []
    cursor = 0
    for user_idx in range(k_users):
        n_user = int(user_sizes[user_idx])
        start = cursor
        end = start + n_user
        partition_ranges.append((start, end))
        cursor = end
    n_k = np.asarray([float(end - start) for (start, end) in partition_ranges], dtype=np.float32)
    return n_k, mode, group_counts


def _evolve_ru_single(ru_evolver, h_ru_vec, user_idx, time_idx):
    """
    Evolve one user's RU channel by one AR(1) step.
    """
    alpha_global = ru_evolver.traj.get_alpha(time_idx)
    if ru_evolver.alpha_bases is None:
        alpha = float(alpha_global)
    else:
        delta = float(alpha_global - ru_evolver.base_static_alpha)
        alpha = float(ru_evolver.traj._clip_alpha(ru_evolver.alpha_bases[user_idx] + delta))
    beta = math.sqrt(max(0.0, 1.0 - alpha * alpha))
    noise = (np.random.randn(*h_ru_vec.shape) + 1j * np.random.randn(*h_ru_vec.shape)) / math.sqrt(2.0)
    h_next = alpha * h_ru_vec + beta * noise.astype(h_ru_vec.dtype, copy=False)
    return h_next.astype(h_ru_vec.dtype, copy=False), alpha


def _complex_to_ri(vec_complex):
    return np.concatenate([np.real(vec_complex), np.imag(vec_complex)], axis=0).astype(np.float32)


def _ri_to_complex(vec_ri, n):
    vec = np.asarray(vec_ri, dtype=np.float32).reshape(-1)
    if vec.size != 2 * n:
        raise ValueError(f"Expected RI vector size {2 * n}, got {vec.size}")
    return (vec[:n] + 1j * vec[n:]).astype(np.complex64)


def _predict_h_ru_gru(model, x_seq, n_ris, hidden_state=None):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        x_tensor = torch.tensor(x_seq, dtype=torch.float32, device=device).unsqueeze(0)
        if hidden_state is not None:
            pred = model(x_tensor, h0=hidden_state.detach().to(device), return_hidden=False)
        else:
            pred = model(x_tensor, return_hidden=False)
    return _ri_to_complex(pred.squeeze(0).cpu().numpy(), n_ris)


def _predict_h_ru_plain(model, x_seq, n_ris):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        x_tensor = torch.tensor(x_seq, dtype=torch.float32, device=device).unsqueeze(0)
        pred = model(x_tensor)
    return _ri_to_complex(pred.squeeze(0).cpu().numpy(), n_ris)


def _flatten_head_state(head_state):
    """Flatten one user's head state dict to a 1D numpy vector."""
    parts = []
    for key in sorted(head_state.keys()):
        parts.append(head_state[key].detach().cpu().reshape(-1).float())
    if not parts:
        return np.zeros((0,), dtype=np.float32)
    return torch.cat(parts, dim=0).numpy()


def _project_rows_to_2d(x):
    """Project row vectors to 2D using PCA (SVD-based)."""
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={x.shape}")
    n_samples = x.shape[0]
    if n_samples == 0:
        return np.zeros((0, 2), dtype=np.float64), 0.0, 0.0

    centered = x - np.mean(x, axis=0, keepdims=True)
    if centered.shape[1] == 0:
        return np.zeros((n_samples, 2), dtype=np.float64), 0.0, 0.0

    _, s, vh = np.linalg.svd(centered, full_matrices=False)
    n_comp = min(2, vh.shape[0])
    proj = centered @ vh[:n_comp, :].T
    if n_comp < 2:
        proj = np.pad(proj, ((0, 0), (0, 2 - n_comp)), mode="constant")

    ratio1, ratio2 = 0.0, 0.0
    if s.size > 0:
        denom = float(max(1, centered.shape[0] - 1))
        eigvals = (s ** 2) / denom
        total = float(np.sum(eigvals))
        if total > 0.0:
            ratio1 = float(eigvals[0] / total)
            if eigvals.size > 1:
                ratio2 = float(eigvals[1] / total)
    return proj, ratio1, ratio2


def _save_head_projection_plot(user_head_states, round_idx, out_dir, tag):
    """Save 2D projection plot for all users' head parameter states."""
    if not user_head_states:
        return None

    vectors = [_flatten_head_state(hs) for hs in user_head_states]
    mat = np.stack(vectors, axis=0).astype(np.float64, copy=False)
    coords, ratio1, ratio2 = _project_rows_to_2d(mat)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{tag}_head_proj_round_{round_idx:04d}.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    user_ids = np.arange(1, coords.shape[0] + 1)
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=user_ids,
        cmap="tab20",
        s=70,
        edgecolors="black",
        linewidths=0.4,
    )
    for i, (xv, yv) in enumerate(coords, start=1):
        ax.text(float(xv), float(yv), f"U{i}", fontsize=8, ha="left", va="bottom")

    ax.set_title(f"{tag} head projection @ round {round_idx}")
    ax.set_xlabel(f"PC1 ({ratio1 * 100.0:.1f}% var)")
    ax.set_ylabel(f"PC2 ({ratio2 * 100.0:.1f}% var)")
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="User index")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _save_gru_state_delta_plot(delta_vectors, round_idx, out_dir, tag="gru_state_delta"):
    """
    Save per-user GRU hidden-state delta debug figure.
    Delta is expected as (current_hidden - previous_hidden), one vector per user.
    """
    if not delta_vectors:
        return None

    valid = [v for v in delta_vectors if v is not None]
    if not valid:
        return None

    dim = int(valid[0].size)
    num_users = len(delta_vectors)
    mat = np.zeros((num_users, dim), dtype=np.float64)
    has_prev = np.zeros((num_users,), dtype=np.bool_)
    for i, vec in enumerate(delta_vectors):
        if vec is None:
            continue
        arr = np.asarray(vec, dtype=np.float64).reshape(-1)
        if arr.size != dim:
            # Keep debug robust even if shape changes unexpectedly.
            dmin = min(dim, arr.size)
            mat[i, :dmin] = arr[:dmin]
        else:
            mat[i, :] = arr
        has_prev[i] = True

    norms = np.linalg.norm(mat, axis=1)
    coords, ratio1, ratio2 = _project_rows_to_2d(mat)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{tag}_round_{round_idx:04d}.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_bar, ax_scatter = axes
    user_ids = np.arange(1, num_users + 1)

    bar_colors = ["tab:blue" if has_prev[i - 1] else "tab:gray" for i in user_ids]
    ax_bar.bar(user_ids, norms, color=bar_colors, edgecolor="black", linewidth=0.4)
    ax_bar.set_title(f"{tag} norm @ round {round_idx}")
    ax_bar.set_xlabel("User")
    ax_bar.set_ylabel("||delta_h||2")
    ax_bar.grid(True, alpha=0.25, axis="y")

    scatter = ax_scatter.scatter(
        coords[:, 0],
        coords[:, 1],
        c=user_ids,
        cmap="tab20",
        s=70,
        edgecolors="black",
        linewidths=0.4,
    )
    for i, (xv, yv) in enumerate(coords, start=1):
        suffix = "" if has_prev[i - 1] else "*"
        ax_scatter.text(float(xv), float(yv), f"U{i}{suffix}", fontsize=8, ha="left", va="bottom")

    ax_scatter.set_title(f"{tag} PCA @ round {round_idx}")
    ax_scatter.set_xlabel(f"PC1 ({ratio1 * 100.0:.1f}% var)")
    ax_scatter.set_ylabel(f"PC2 ({ratio2 * 100.0:.1f}% var)")
    ax_scatter.set_xlim(-5.0, 5.0)
    ax_scatter.set_ylim(-5.0, 5.0)
    ax_scatter.set_aspect("equal", adjustable="box")
    ax_scatter.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=ax_scatter, label="User index")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main():
    config = Config
    logger = Logger(config=config) if config.log_to_file else Logger()
    # logger = Logger(config.log_file_path) if config.log_to_file else Logger()
    logger.info("Initializing simulation...")
    np.random.seed(0)
    torch.manual_seed(0)

    debug_head_plot_enabled = bool(getattr(config, "enable_reptile_head_debug_plot", False))
    debug_head_plot_every = int(getattr(config, "reptile_head_debug_every", 10))
    if debug_head_plot_every <= 0:
        debug_head_plot_every = 10
    debug_head_plot_root = str(getattr(config, "reptile_head_debug_root", "debug"))
    debug_gru_state_plot_enabled = bool(getattr(config, "enable_gru_state_diff_debug_plot", False))
    debug_gru_state_plot_every = int(getattr(config, "gru_state_diff_debug_every", 10))
    if debug_gru_state_plot_every <= 0:
        debug_gru_state_plot_every = 10
    debug_head_plot_run_dir = None
    debug_head_plot_gru_dir = None
    debug_head_plot_arch_dir = None
    debug_gru_state_plot_dir = None
    if (debug_head_plot_enabled or debug_gru_state_plot_enabled) and str(config.meta_algorithm).lower() == "reptile":
        debug_head_plot_run_dir = os.path.join(debug_head_plot_root, config.fingerprint())
        debug_head_plot_gru_dir = os.path.join(debug_head_plot_run_dir, "head_projection_gru")
        debug_head_plot_arch_dir = os.path.join(debug_head_plot_run_dir, "head_projection_cnn_arch")
        debug_gru_state_plot_dir = os.path.join(debug_head_plot_run_dir, "gru_state_delta")
        if debug_head_plot_enabled:
            os.makedirs(debug_head_plot_gru_dir, exist_ok=True)
            os.makedirs(debug_head_plot_arch_dir, exist_ok=True)
            logger.info(
                f"Reptile head debug projection enabled: every {debug_head_plot_every} rounds, "
                f"out={debug_head_plot_run_dir}"
            )
        if debug_gru_state_plot_enabled:
            os.makedirs(debug_gru_state_plot_dir, exist_ok=True)
            logger.info(
                f"GRU state-diff debug enabled: every {debug_gru_state_plot_every} rounds, "
                f"out={debug_gru_state_plot_dir}"
            )
    elif debug_head_plot_enabled or debug_gru_state_plot_enabled:
        logger.info("Reptile debug plotting skipped: meta_algorithm is not Reptile.")

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
    # Supervision target is RIS-user CSI h_RU (real-imag stacked).
    output_dim = 2 * config.num_ris_elements
    gru_csi_target_mode = _parse_gru_target_mode(config.gru_csi_target_mode)

    # Per-user sliding window buffer for GRU
    W = int(config.window_length)
    pad_val = float(config.window_pad_value)
    gru_context_mode = str(config.gru_context_mode).lower()
    if gru_context_mode not in {"persistent_hidden", "time_window"}:
        raise ValueError("Config.gru_context_mode must be 'persistent_hidden' or 'time_window'")
    use_persistent_hidden_state = (gru_context_mode == "persistent_hidden")
    use_time_window = (gru_context_mode == "time_window")
    oa_optimizer_mode = str(getattr(config, "oa_optimizer_mode", "legacy")).strip().lower()
    if oa_optimizer_mode not in {"legacy", "state_aware"}:
        raise ValueError("Config.oa_optimizer_mode must be 'legacy' or 'state_aware'")
    if oa_optimizer_mode == "state_aware" and not use_persistent_hidden_state:
        raise ValueError("State-aware OA currently supports only gru_context_mode='persistent_hidden'")

    state_beta_z = float(getattr(config, "state_beta_z", 0.1))
    state_alpha_z = float(getattr(config, "state_alpha_z", 0.5))
    state_alpha_x = float(getattr(config, "state_alpha_x", 0.5))
    state_mu = float(getattr(config, "state_mu", 0.5))
    state_weight_min = float(getattr(config, "state_weight_min", 0.5))
    state_weight_max = float(getattr(config, "state_weight_max", 2.0))
    state_fast_clip = float(getattr(config, "state_fast_clip", 2.0))
    state_eps = float(getattr(config, "state_eps", 1e-8))
    state_strategy = str(getattr(config, "state_strategy", "protect")).strip().lower()
    if state_strategy not in {"protect", "stability"}:
        raise ValueError("Config.state_strategy must be 'protect' or 'stability'")
    alpha_sum = state_alpha_z + state_alpha_x
    if alpha_sum <= 0:
        raise ValueError("state_alpha_z + state_alpha_x must be positive")
    state_alpha_z = state_alpha_z / alpha_sum
    state_alpha_x = state_alpha_x / alpha_sum
    oa_ao_iters = int(getattr(config, "oa_ao_iters", 2))
    oa_theta_lr = float(getattr(config, "oa_theta_lr", 0.05))
    oa_theta_grad_steps = int(getattr(config, "oa_theta_grad_steps", 1))
    oa_normalize_f = bool(getattr(config, "oa_normalize_f", True))

    reset_hidden_on_round1 = bool(config.reset_hidden_on_round1)
    reset_hidden_on_large_backbone_update = bool(config.reset_hidden_on_large_backbone_update)
    hidden_reset_update_norm_threshold = float(config.hidden_reset_update_norm_threshold)

    obs_buffers = [deque(maxlen=W) for _ in range(config.num_users)]
    logger.info(
        f"GRU context mode={gru_context_mode}, W={W}"
    )
    logger.info(f"OA optimizer mode={oa_optimizer_mode}")
    logger.info(f"GRU CSI target mode={gru_csi_target_mode}")
    logger.info("CNN-arch/CNN-base CSI target mode=t")
    # Per-user local sample cache for GRU/CNN-arch only.
    # Literature CNN baseline stays memoryless and never uses this cache.
    S = int(config.local_cache_size)
    use_local_cache_seq_models = bool(config.use_local_sample_cache) and (S > 1)
    sample_buffers = [deque(maxlen=S) for _ in range(config.num_users)]
    logger.info(f"Local sample cache (GRU/CNN-arch) enabled={use_local_cache_seq_models}, S={S}")
    if use_persistent_hidden_state and use_local_cache_seq_models:
        logger.info("Persistent-hidden GRU path bypasses local sample cache (continuous-segment update).")
    global_model = CSICNNGRU(observation_dim=obs_dim, output_dim=output_dim)
    user_hidden_states = [None for _ in range(config.num_users)]
    user_segment_time = np.zeros((config.num_users,), dtype=np.int64)
    reset_hidden_next_round = False
    # For warm-start heads: keep a global head template to clone for users
    global_head_state = {k: v.clone() for k, v in global_model.state_dict().items() if k.startswith("head")}
    user_head_states = [copy.deepcopy(global_head_state) for _ in range(config.num_users)]
    state_slow_z = np.zeros((config.num_users,), dtype=np.float32)
    state_fast_x = np.zeros((config.num_users,), dtype=np.float32)
    state_eta = np.ones((config.num_users,), dtype=np.float32)
    state_omega = np.ones((config.num_users,), dtype=np.float32)
    user_n_k_target, user_partition_mode, user_group_counts = _build_user_data_partitions(config)
    logger.info(
        f"User data partition mode={user_partition_mode}, "
        f"target_n_k(min/mean/max)=("
        f"{float(user_n_k_target.min()):.1f}/{float(user_n_k_target.mean()):.1f}/{float(user_n_k_target.max()):.1f})"
    )
    if user_partition_mode == "grouped":
        logger.info(
            "Grouped user partition "
            f"(small/medium/large): counts={user_group_counts.tolist()}, "
            f"n_k={list(np.asarray(config.user_group_data_sizes, dtype=int).tolist())}"
        )
    if not bool(config.ota_use_weighted_users):
        logger.info("OTA user weighting disabled: using equal weights (all ones).")
    if oa_optimizer_mode == "state_aware":
        logger.info(
            "State-aware OA params: "
            f"beta_z={state_beta_z:.3f}, alpha_z={state_alpha_z:.3f}, alpha_x={state_alpha_x:.3f}, "
            f"mu={state_mu:.3f}, clip=[{state_weight_min:.3f},{state_weight_max:.3f}], "
            f"x_clip={state_fast_clip:.3f}, strategy={state_strategy}"
        )
        logger.info("State-aware slow state uses fixed global head template distance.")

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
        logger.info("Literature CNN baseline local cache is disabled by design.")

    # Oracle upper-bound reference branch (true CSI driven AO only, no model training).
    f_beam_oracle = f_beam.copy()
    theta_ota_oracle = theta_ota.copy()
    logger.info("Oracle-true AO reference enabled=True (upper bound, true CSI driven).")

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
        if use_persistent_hidden_state:
            # Persistent mode uses variable-length continuous segments per user.
            h_RUs_next = np.zeros_like(h_RUs)
            alpha_used_next = np.zeros((config.num_users,), dtype=np.float32)
            h_RUs_target_gru = None
        else:
            # One-step channel evolution for non-persistent mode.
            h_RUs_next, alpha_used_next = ru_evolver.step(h_RUs, round_idx)
            h_RUs_target_gru = h_RUs_next if gru_csi_target_mode == "t+1" else h_RUs

        # Generate pilot observation and ground truth channel for each user at this round
        local_data = []
        local_data_stateful = [[] for _ in range(config.num_users)] if use_persistent_hidden_state else None
        local_data_arch = [] if enable_cnn_arch_ablation else None
        local_data_baseline = [None for _ in range(config.num_users)] if enable_cnn_baseline else None
        for k in range(config.num_users):
            # Pilot signals for user k
            h_BU_k = h_BUs[k] if h_BUs is not None else None
            if use_persistent_hidden_state:
                seg_count = max(1, int(round(float(user_n_k_target[k]))))
                h_ru_seg = h_RUs[k].copy()
                seq_samples = []
                baseline_seq_samples = [] if enable_cnn_baseline else None
                alpha_last = 0.0
                for seg_idx in range(seg_count):
                    time_idx = int(user_segment_time[k] + 1)
                    Y_pilot, _, _ = pilot_gen.simulate_pilot_observation(
                        H_BR,
                        h_ru_seg,
                        f_beam,
                        theta_pilot,
                        noise_std=float(pilot_noise_std_k[k]),
                        h_BU=h_BU_k,
                        link_switch=link_switch,
                    )
                    obs_real = np.real(Y_pilot)
                    obs_imag = np.imag(Y_pilot)
                    obs_step = np.stack([obs_real, obs_imag], axis=0).astype(np.float32)  # (2, P)
                    X_seq = obs_step[None, :, :]  # (1, 2, P), one continuous segment

                    h_next_seg, alpha_seg = _evolve_ru_single(
                        ru_evolver=ru_evolver,
                        h_ru_vec=h_ru_seg,
                        user_idx=k,
                        time_idx=time_idx,
                    )
                    y_target = h_next_seg if gru_csi_target_mode == "t+1" else h_ru_seg
                    y = _complex_to_ri(y_target)
                    seq_samples.append((X_seq.astype(np.float32, copy=False), y.astype(np.float32, copy=False)))

                    if enable_cnn_baseline:
                        Y_pilot_baseline, _, _ = pilot_gen.simulate_pilot_observation(
                            H_BR,
                            h_ru_seg,
                            f_beam_baseline,
                            theta_pilot,
                            noise_std=float(pilot_noise_std_k[k]),
                            h_BU=h_BU_k,
                            link_switch=link_switch,
                        )
                        obs_real_baseline = np.real(Y_pilot_baseline)
                        obs_imag_baseline = np.imag(Y_pilot_baseline)
                        obs_step_baseline = np.stack([obs_real_baseline, obs_imag_baseline], axis=0).astype(np.float32)
                        X_seq_baseline = obs_step_baseline[None, :, :]
                        y_baseline = _complex_to_ri(h_ru_seg)
                        baseline_seq_samples.append(
                            (
                                X_seq_baseline.astype(np.float32, copy=False),
                                y_baseline.astype(np.float32, copy=False),
                            )
                        )

                    h_ru_seg = h_next_seg
                    alpha_last = alpha_seg
                    user_segment_time[k] = time_idx

                local_data_stateful[k] = seq_samples
                local_data.append(seq_samples[-1])
                if enable_cnn_baseline:
                    local_data_baseline[k] = baseline_seq_samples
                h_RUs_next[k] = h_ru_seg
                alpha_used_next[k] = float(alpha_last)
            else:
                Y_pilot, _, _ = pilot_gen.simulate_pilot_observation(
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

                y = _complex_to_ri(h_RUs_target_gru[k])
                sample = (X_seq.astype(np.float32, copy=False), y.astype(np.float32, copy=False))
                local_data.append(sample)

            if (not use_persistent_hidden_state) and use_time_window and W > 1 and round_idx <= 3 and k == 0:
                logger.info(f"Example X_seq shape for user1: {sample[0].shape}")  # (W,2,P)

            # push into local cache
            if use_local_cache_seq_models and (not use_persistent_hidden_state):
                sample_buffers[k].append((sample[0].copy(), sample[1].copy()))  # copy for safety

            if enable_cnn_arch_ablation:
                Y_pilot_arch, _, _ = pilot_gen.simulate_pilot_observation(
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

                y_arch = _complex_to_ri(h_RUs[k])
                sample_arch = (
                    X_seq_arch.astype(np.float32, copy=False),
                    y_arch.astype(np.float32, copy=False),
                )
                local_data_arch.append(sample_arch)
                if use_local_cache_seq_models and use_time_window and W > 1:
                    sample_buffers_arch[k].append((sample_arch[0].copy(), sample_arch[1].copy()))

            if enable_cnn_baseline and (not use_persistent_hidden_state):
                Y_pilot_baseline, _, _ = pilot_gen.simulate_pilot_observation(
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

                y_baseline = _complex_to_ri(h_RUs[k])
                sample_baseline = (
                    X_seq_baseline.astype(np.float32, copy=False),
                    y_baseline.astype(np.float32, copy=False),
                )
                local_data_baseline[k] = sample_baseline

        # Local training on each user's data
        local_models = []
        losses = []
        local_models_arch = [] if enable_cnn_arch_ablation else None
        losses_arch = [] if enable_cnn_arch_ablation else None
        local_models_baseline = [] if enable_cnn_baseline else None
        losses_baseline = [] if enable_cnn_baseline else None
        h_RUs_est = np.zeros((config.num_users, config.num_ris_elements), dtype=np.complex64)
        h_RUs_est_arch = np.zeros((config.num_users, config.num_ris_elements), dtype=np.complex64) \
            if enable_cnn_arch_ablation else None
        h_RUs_est_baseline = np.zeros((config.num_users, config.num_ris_elements), dtype=np.complex64) \
            if enable_cnn_baseline else None
        state_fast_x_round = np.zeros((config.num_users,), dtype=np.float32)
        # n_k for this round: number of samples actually used by each user in local training.
        user_sample_count_round = np.zeros((config.num_users,), dtype=np.float32)
        # Debug-only cache: per-user hidden-state delta (current - history), no training-path impact.
        gru_state_delta_vectors = [None for _ in range(config.num_users)] \
            if (debug_gru_state_plot_dir is not None and use_persistent_hidden_state) else None
        for k in range(config.num_users):
            loss_arch = None
            loss_baseline = None
            samples_used_k = 1

            # Model for user k from global weights
            local_model = CSICNNGRU(observation_dim=obs_dim, output_dim=output_dim)
            # load global backbone + user head (warm-start head copied from global template on round1)
            state = global_model.state_dict()
            for hk, hv in user_head_states[k].items():
                state[hk] = hv.clone()
            local_model.load_state_dict(state)

            # Train on user k's data
            if use_persistent_hidden_state:
                seq_k = local_data_stateful[k]
                sample_k = seq_k[-1]
                hidden_prev = user_hidden_states[k]
                local_model, loss, hidden_next, pred_last = trainer.train_stateful_sequence(
                    local_model,
                    seq_k,
                    hidden_state=hidden_prev,
                )
                samples_used_k = max(1, len(seq_k))
                user_hidden_states[k] = hidden_next
                state_fast_x_round[k] = compute_hidden_drift_ratio(
                    hidden_prev=hidden_prev,
                    hidden_next=hidden_next,
                    eps=state_eps,
                    x_max=state_fast_clip,
                )
                if gru_state_delta_vectors is not None and hidden_next is not None:
                    vec_next = hidden_next.detach().cpu().reshape(-1).float()
                    if hidden_prev is None:
                        vec_prev = torch.zeros_like(vec_next)
                    else:
                        vec_prev = hidden_prev.detach().cpu().reshape(-1).float()
                    gru_state_delta_vectors[k] = (vec_next - vec_prev).numpy()
                if (round_idx <= 3) and (k == 0) and (hidden_next is not None):
                    logger.info(f"User1 persistent hidden norm: {torch.norm(hidden_next).item():.4e}")
                if pred_last is not None:
                    h_RUs_est[k] = _ri_to_complex(pred_last.numpy(), config.num_ris_elements)
                else:
                    h_RUs_est[k] = _predict_h_ru_gru(
                        local_model,
                        sample_k[0],
                        config.num_ris_elements,
                        hidden_state=hidden_prev,
                    )
            else:
                if use_local_cache_seq_models:
                    data_k = list(sample_buffers[k])  # latest S window samples, OK if length < S
                else:
                    data_k = [local_data[k]]  # fallback: 1 sample per epoch
                samples_used_k = max(1, len(data_k))
                local_model, loss = trainer.train(local_model, data_k)
                h_RUs_est[k] = _predict_h_ru_gru(
                    local_model,
                    local_data[k][0],
                    config.num_ris_elements,
                    hidden_state=None,
                )
            user_sample_count_round[k] = float(samples_used_k)
            losses.append(loss if loss is not None else 0.0)
            local_models.append(local_model)
            # cache back the personalized head for user k
            user_head_states[k] = {
                name: param.detach().clone()
                for name, param in local_model.state_dict().items()
                if name.startswith("head")
            }
            head_dist_k = compute_head_distance(
                head_state_user=user_head_states[k],
                head_state_global=global_head_state,
                eps=state_eps,
            )
            if round_idx == 1:
                state_slow_z[k] = head_dist_k
            else:
                state_slow_z[k] = update_ewma_state(
                    prev_value=state_slow_z[k],
                    current_value=head_dist_k,
                    beta=state_beta_z,
                )

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
                if use_local_cache_seq_models and use_time_window and W > 1:
                    data_k_arch = list(sample_buffers_arch[k])
                else:
                    data_k_arch = [local_data_arch[k]]
                local_model_arch, loss_arch = trainer.train(local_model_arch, data_k_arch)
                losses_arch.append(loss_arch if loss_arch is not None else 0.0)
                local_models_arch.append(local_model_arch)
                h_RUs_est_arch[k] = _predict_h_ru_plain(
                    local_model_arch,
                    local_data_arch[k][0],
                    config.num_ris_elements,
                )
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
                if use_persistent_hidden_state:
                    data_k_baseline = local_data_baseline[k]
                else:
                    data_k_baseline = [local_data_baseline[k]]
                local_model_baseline, loss_baseline = trainer.train(local_model_baseline, data_k_baseline)
                losses_baseline.append(loss_baseline if loss_baseline is not None else 0.0)
                local_models_baseline.append(local_model_baseline)
                h_RUs_est_baseline[k] = _predict_h_ru_plain(
                    local_model_baseline,
                    data_k_baseline[-1][0],
                    config.num_ris_elements,
                )

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

        state_fast_x = state_fast_x_round.astype(np.float32, copy=False)
        if oa_optimizer_mode == "state_aware":
            state_eta, z_bar, x_bar = combine_state_scores(
                z_state=state_slow_z,
                x_state=state_fast_x,
                alpha_z=state_alpha_z,
                alpha_x=state_alpha_x,
                eps=state_eps,
            )
            state_omega, _ = build_state_weights(
                eta_state=state_eta,
                mu=state_mu,
                weight_min=state_weight_min,
                weight_max=state_weight_max,
                strategy=state_strategy,
            )
            logger.info(
                "State metrics: "
                f"z_bar={z_bar:.4e}, x_bar={x_bar:.4e}, "
                f"eta[min/mean/max]={np.min(state_eta):.3f}/{np.mean(state_eta):.3f}/{np.max(state_eta):.3f}, "
                f"omega[min/mean/max]={np.min(state_omega):.3f}/{np.mean(state_omega):.3f}/{np.max(state_omega):.3f}"
            )
        else:
            state_eta = np.ones((config.num_users,), dtype=np.float32)
            state_omega = np.ones((config.num_users,), dtype=np.float32)

        if debug_head_plot_gru_dir is not None and (round_idx % debug_head_plot_every == 0):
            proj_path_gru = _save_head_projection_plot(
                user_head_states=user_head_states,
                round_idx=round_idx,
                out_dir=debug_head_plot_gru_dir,
                tag="gru",
            )
            if proj_path_gru is not None:
                logger.info(f"[DEBUG] Saved GRU head projection: {proj_path_gru}")
            if enable_cnn_arch_ablation and user_head_states_arch is not None:
                proj_path_arch = _save_head_projection_plot(
                    user_head_states=user_head_states_arch,
                    round_idx=round_idx,
                    out_dir=debug_head_plot_arch_dir,
                    tag="cnn_arch",
                )
                if proj_path_arch is not None:
                    logger.info(f"[DEBUG] Saved CNN-arch head projection: {proj_path_arch}")
        if (debug_gru_state_plot_dir is not None
                and gru_state_delta_vectors is not None
                and (round_idx % debug_gru_state_plot_every == 0)):
            state_diff_path = _save_gru_state_delta_plot(
                delta_vectors=gru_state_delta_vectors,
                round_idx=round_idx,
                out_dir=debug_gru_state_plot_dir,
                tag="gru_state_delta",
            )
            if state_diff_path is not None:
                logger.info(f"[DEBUG] Saved GRU state-diff plot: {state_diff_path}")

        # Aggregate updates at server
        logger.info("Aggregating GRU updates at server.")
        old_global_vec = state_dict_to_vector_backbone(global_model).detach().cpu()
        old_global_vec_arch = None
        if enable_cnn_arch_ablation:
            old_global_vec_arch = state_dict_to_vector_backbone(global_model_arch).detach().cpu()
        old_global_vec_baseline = None
        if enable_cnn_baseline:
            old_global_vec_baseline = state_dict_to_vector(global_model_baseline).detach().cpu()

        # Derive user weights K_k from actual per-user local sample counts n_k in this round.
        if bool(config.ota_use_weighted_users):
            K_vals = np.maximum(user_sample_count_round.astype(np.float32, copy=False), 1.0)
        else:
            K_vals = np.ones((config.num_users,), dtype=np.float32)
        K_vec = torch.from_numpy(K_vals)
        K_norm = K_vec / torch.mean(K_vec)
        logger.info(
            f"Round {round_idx} n_k(min/mean/max)=("
            f"{float(K_vals.min()):.1f}/{float(K_vals.mean()):.1f}/{float(K_vals.max()):.1f})"
        )

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
            # Literature baseline uses full-model weighted FedAvg based on per-user data amount.
            global_model_baseline = aggregator_baseline.aggregate(
                global_model_baseline,
                local_models_baseline,
                backbone_only=False,
                client_weights=user_sample_count_round,
            )

            new_global_vec_baseline = state_dict_to_vector(global_model_baseline).detach().cpu()
            logger.info(
                "CNN-base global update norm: "
                f"{torch.norm(new_global_vec_baseline - old_global_vec_baseline).item():.4e}"
            )

        # Optimize beamforming vector f and RIS phases theta for next round
        logger.info("Optimizing beamforming and RIS configuration.")
        # Use model-estimated h_RU for OTA beam/RIS optimization.
        if oa_optimizer_mode == "state_aware":
            f_beam, theta_ota, nmse_proxy = optimize_beam_ris_state_aware(
                H_BR,
                h_RUs_est,
                h_BUs=h_BUs,
                theta_init=theta_ota,
                f_init=f_beam,
                link_switch=link_switch,
                user_weights=K_norm.numpy(),
                state_weights=state_omega,
                update_vars=delta_var.numpy(),
                tx_power=config.ota_tx_power,
                noise_std=config.ota_noise_std,
                var_floor=config.ota_var_floor,
                ao_iters=oa_ao_iters,
                theta_lr=oa_theta_lr,
                theta_grad_steps=oa_theta_grad_steps,
                normalize_f=oa_normalize_f,
                eps=config.ota_eps,
            )
        else:
            f_beam, theta_ota, nmse_proxy = optimize_beam_ris(
                H_BR,
                h_RUs_est,
                h_BUs=h_BUs,
                theta_init=theta_ota,
                f_init=f_beam,
                link_switch=link_switch,
                user_weights=K_norm.numpy(),
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
                H_BR, h_RUs_est_arch, h_BUs=h_BUs, theta_init=theta_ota_arch, f_init=f_beam_arch,
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
            baseline_user_weights = K_norm.numpy()
            f_beam_baseline, theta_ota_baseline, nmse_proxy_baseline = optimize_beam_ris(
                H_BR, h_RUs_est_baseline, h_BUs=h_BUs, theta_init=theta_ota_baseline, f_init=f_beam_baseline,
                link_switch=link_switch, user_weights=baseline_user_weights,
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

        # Oracle upper-bound AO reference with true channels.
        if use_persistent_hidden_state:
            h_RUs_true_for_oracle = h_RUs_next if gru_csi_target_mode == "t+1" else h_RUs
        else:
            h_RUs_true_for_oracle = h_RUs_target_gru
        f_beam_oracle, theta_ota_oracle, nmse_proxy_oracle = optimize_beam_ris(
            H_BR, h_RUs_true_for_oracle, h_BUs=h_BUs, theta_init=theta_ota_oracle, f_init=f_beam_oracle,
            link_switch=link_switch, user_weights=K_norm.numpy(),
            update_vars=delta_var.numpy(),
            tx_power=config.ota_tx_power,
            noise_std=config.ota_noise_std,
            var_floor=config.ota_var_floor,
            eps=config.ota_eps,
        )
        logger.info(f"Optimized f (Oracle-true): {np.round(f_beam_oracle, 4)}")
        logger.info(
            f"Optimized theta_ota (Oracle-true): {np.round(theta_ota_oracle, 4)}, "
            f"proxy_NMSE={nmse_proxy_oracle:.4e}"
        )

        # Advance channels to the precomputed next-round state.
        h_RUs = h_RUs_next
        alpha_used = alpha_used_next
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
