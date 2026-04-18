from collections import deque
import copy
import math
import os
import time

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MultipleLocator
import numpy as np
import torch

from aircomp_opt.OTA_sim import AirCompSimulator
from aircomp_opt.f_theta_optim import optimize_beam_ris_by_mode
from aircomp_opt.grouping_optim import GroupingSCAConfig, GroupingWarmStart, optimize_risk_grouping_sca
from data import RISdata, pilot_gen
from data.channel import build_ru_channel_evolver_from_config
from fl_core.agg import MetaUpdater
from fl_core.lmmse import estimate_h_ru_lmmse
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
from utils.eta_response_snapshot import build_eta_components, complex_nmse_per_user, save_snapshot_npz
from utils.logger import Logger


PROXY_PLOT_MODEL_ORDER = ("GRU", "CNN-arch", "CNN-base", "LMMSE", "Oracle-true")
PROXY_COMMON_UPDATE_CSV = "05_proxy_nmse_after_optimization_common_update_vars.csv"


def _parse_gru_target_mode(mode):
    token = str(mode).strip().lower().replace(" ", "")
    if token in {"t", "current", "now"}:
        return "t"
    if token in {"uplink_linear", "uplink", "t+tau", "t_plus_tau"}:
        return "uplink_linear"
    if token in {"uplink_direct", "tau", "tau_direct", "direct_tau", "uplink_tau"}:
        return "uplink_direct"
    raise ValueError("Config.gru_csi_target_mode must be 't', 'uplink_linear', or 'uplink_direct'")


def _parse_meta_algorithm(mode):
    token = str(mode).strip().lower().replace("-", "_").replace(" ", "")
    if token in {"fedavg", "fed_avg"}:
        return "fedavg"
    if token == "reptile":
        return "reptile"
    raise ValueError("Config.meta_algorithm must be 'FedAvg' or 'Reptile'")


def _parse_nonnegative_float_grid(values, name):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"Config.{name} must contain at least one value")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Config.{name} must contain only finite values")
    if np.any(arr < 0.0):
        raise ValueError(f"Config.{name} must contain only nonnegative values")
    return np.unique(arr)


BRANCH_ANSI = {
    "GRU": "34",
    "CNN-arch": "35",
    "CNN-base": "36",
    "LMMSE": "31",
    "Oracle-true": "33",
}

WEIGHT_GROUP_ANSI = {
    "equal": "32",
    "small": "33",
    "medium": "36",
    "large": "35",
}


def _ansi(text, code):
    return f"\033[{code}m{text}\033[0m"


def _highlight_metric_value(value, branch_name):
    color = BRANCH_ANSI.get(branch_name, "37")
    return _ansi(f"{float(value):.4e}", f"1;{color}")


def _colorize_branch_line(branch_name, text):
    color = BRANCH_ANSI.get(branch_name)
    if color is None:
        return text
    return _ansi(text, color)


def _build_weight_category_map(raw_weights):
    unique_vals = sorted({int(round(float(v))) for v in np.asarray(raw_weights).reshape(-1).tolist()})
    if len(unique_vals) == 1:
        return {unique_vals[0]: ("E", "equal")}
    if len(unique_vals) == 2:
        return {
            unique_vals[0]: ("S", "small"),
            unique_vals[1]: ("L", "large"),
        }
    category_tokens = [("S", "small"), ("M", "medium"), ("L", "large")]
    mapping = {}
    for idx, raw in enumerate(unique_vals):
        token = category_tokens[idx] if idx < len(category_tokens) else (f"C{idx + 1}", "equal")
        mapping[raw] = token
    return mapping


def _format_ota_weight_logs(raw_weights, norm_weights, *, chunk_size=5):
    raw_arr = np.asarray(raw_weights, dtype=np.float32).reshape(-1)
    norm_arr = np.asarray(norm_weights, dtype=np.float32).reshape(-1)
    category_map = _build_weight_category_map(raw_arr)

    legend_parts = []
    for raw in sorted(category_map.keys()):
        token, label = category_map[raw]
        color = WEIGHT_GROUP_ANSI.get(label, "37")
        legend_parts.append(_ansi(f"{token}:n={raw}", color))
    lines = []
    if legend_parts:
        lines.append("OTA weight groups: " + " | ".join(legend_parts))

    entries = []
    for user_idx, (raw, norm) in enumerate(zip(raw_arr, norm_arr), start=1):
        token, label = category_map[int(round(float(raw)))]
        color = WEIGHT_GROUP_ANSI.get(label, "37")
        entry = f"u{user_idx:02d}[{token}] n={float(raw):.1f}, w={float(norm):.2f}"
        entries.append(_ansi(entry, color))

    for start in range(0, len(entries), chunk_size):
        end = min(start + chunk_size, len(entries))
        lines.append(f"OTA per-user weights [{start + 1:02d}-{end:02d}]: " + " | ".join(entries[start:end]))
    return lines


def _format_speed_doppler_alpha_logs(speed_vals, doppler_vals, alpha_vals, *, chunk_size=5, label="RU mobility"):
    speed_arr = np.asarray(speed_vals, dtype=np.float32).reshape(-1)
    doppler_arr = np.asarray(doppler_vals, dtype=np.float32).reshape(-1)
    alpha_arr = np.asarray(alpha_vals, dtype=np.float32).reshape(-1)
    if not (len(speed_arr) == len(doppler_arr) == len(alpha_arr)):
        raise ValueError("speed/doppler/alpha arrays must have the same length")

    lines = []
    entries = []
    for user_idx, (speed_k, doppler_k, alpha_k) in enumerate(zip(speed_arr, doppler_arr, alpha_arr), start=1):
        entries.append(
            f"u{user_idx:02d}: v={float(speed_k):.3f}m/s, "
            f"fD={float(doppler_k):.3f}Hz, a={float(alpha_k):.6f}"
        )
    for start in range(0, len(entries), chunk_size):
        end = min(start + chunk_size, len(entries))
        lines.append(f"{label} [{start + 1:02d}-{end:02d}]: " + " | ".join(entries[start:end]))
    return lines


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


def _estimate_ris_pathloss_log_bounds(ru_evolver, num_rounds, uplink_tau_ratio, eps=1e-12, margin_ratio=0.05):
    step_values = [0.0]
    rho = float(uplink_tau_ratio)
    for round_idx in range(1, int(num_rounds) + 1):
        step_values.append(float(round_idx - 1))
        step_values.append(float(round_idx - 1) + rho)
    log_pl_values = []
    for step in sorted(set(step_values)):
        pl = np.asarray(ru_evolver.ris_pathloss(ru_evolver.positions_at(step)), dtype=np.float64).reshape(-1)
        log_pl_values.append(np.log(np.maximum(pl, float(eps))))
    stacked = np.concatenate(log_pl_values, axis=0)
    log_pl_min = float(np.min(stacked))
    log_pl_max = float(np.max(stacked))
    if log_pl_max <= log_pl_min:
        log_pl_max = log_pl_min + 1e-6
    span = log_pl_max - log_pl_min
    margin = float(margin_ratio) * span
    return log_pl_min - margin, log_pl_max + margin


def _complex_to_ri(vec_complex):
    return np.concatenate([np.real(vec_complex), np.imag(vec_complex)], axis=0).astype(np.float32)


def _ri_to_complex(vec_ri, n):
    vec = np.asarray(vec_ri, dtype=np.float32).reshape(-1)
    if vec.size != 2 * n:
        raise ValueError(f"Expected RI vector size {2 * n}, got {vec.size}")
    return (vec[:n] + 1j * vec[n:]).astype(np.complex64)


def _split_gru_dual_ri(vec_ri, n_ris):
    vec = np.asarray(vec_ri, dtype=np.float32).reshape(-1)
    csi_dim = 2 * int(n_ris)
    if vec.size != 2 * csi_dim:
        raise ValueError(f"Expected dual-output RI vector size {2 * csi_dim}, got {vec.size}")
    return vec[:csi_dim].astype(np.float32, copy=False), vec[csi_dim:].astype(np.float32, copy=False)


def _build_gru_dual_target(h_t, h_tau):
    h_t_arr = np.asarray(h_t, dtype=np.complex64)
    h_tau_arr = np.asarray(h_tau, dtype=np.complex64)
    return np.concatenate([_complex_to_ri(h_t_arr), _complex_to_ri(h_tau_arr)], axis=0).astype(np.float32)


def _normalize_complex_by_pl(h_vec, pl_value, eps):
    scale = math.sqrt(max(float(pl_value), float(eps)))
    return (np.asarray(h_vec, dtype=np.complex64) / scale).astype(np.complex64, copy=False)


def _apply_pl_to_complex(h_vec_norm, pl_value, eps):
    scale = math.sqrt(max(float(pl_value), float(eps)))
    return (np.asarray(h_vec_norm, dtype=np.complex64) * scale).astype(np.complex64, copy=False)


def _build_gru_dual_target_pl_factorized(h_t, h_tau, pl_value, eps):
    return _build_gru_dual_target(
        _normalize_complex_by_pl(h_t, pl_value, eps),
        _normalize_complex_by_pl(h_tau, pl_value, eps),
    )


def _select_gru_pl_value(pl_t, pl_tau, mode):
    if mode == "t":
        return float(pl_t)
    return float(pl_tau)


def _reconstruct_gru_dual_ri_from_pl(vec_ri_norm, n_ris, pl_value, eps):
    pred_t_ri, pred_tau_ri = _split_gru_dual_ri(vec_ri_norm, n_ris)
    pred_t_c = _apply_pl_to_complex(_ri_to_complex(pred_t_ri, n_ris), pl_value, eps)
    pred_tau_c = _apply_pl_to_complex(_ri_to_complex(pred_tau_ri, n_ris), pl_value, eps)
    return _build_gru_dual_target(pred_t_c, pred_tau_c)


def _select_gru_ri_output(vec_ri, n_ris, mode, uplink_tau_ratio):
    rho = float(uplink_tau_ratio)
    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"uplink_tau_ratio must be in [0, 1], got {uplink_tau_ratio}")
    pred_t, pred_tau = _split_gru_dual_ri(vec_ri, n_ris)
    if mode == "t":
        return pred_t
    if mode == "uplink_linear":
        return (((1.0 - rho) * pred_t) + (rho * pred_tau)).astype(np.float32, copy=False)
    if mode == "uplink_direct":
        return pred_tau.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported GRU output mode: {mode}")


def _predict_gru_dual_ri(model, x_seq, hidden_state=None, return_aux=False):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        x_tensor = torch.tensor(x_seq, dtype=torch.float32, device=device).unsqueeze(0)
        if hidden_state is not None:
            outputs = model(x_tensor, h0=hidden_state.detach().to(device), return_hidden=False)
        else:
            outputs = model(x_tensor, return_hidden=False)
    aux = None
    if isinstance(outputs, tuple) and len(outputs) == 3:
        pred_t, pred_tau, pred_pl = outputs
        aux = {"pl_hat": pred_pl.squeeze(0).cpu().numpy().astype(np.float32, copy=False)}
    else:
        pred_t, pred_tau = outputs
    result = (
        pred_t.squeeze(0).cpu().numpy().astype(np.float32, copy=False),
        pred_tau.squeeze(0).cpu().numpy().astype(np.float32, copy=False),
    )
    if return_aux:
        return result[0], result[1], aux
    return result


def _predict_h_ru_gru(
        model,
        x_seq,
        n_ris,
        mode,
        uplink_tau_ratio,
        hidden_state=None,
        target_stats=None,
        *,
        pl_factorization_enabled=False,
        pl_eps=1e-12,
):
    pred_t, pred_tau, aux = _predict_gru_dual_ri(model, x_seq, hidden_state=hidden_state, return_aux=True)
    pred_dual = np.concatenate([pred_t, pred_tau], axis=0).astype(np.float32, copy=False)
    if target_stats is not None:
        pred_dual = _invert_standardization(pred_dual, target_stats)
    if pl_factorization_enabled:
        if aux is None or "pl_hat" not in aux:
            raise ValueError("PL-factorized GRU prediction requires model to return pl_hat")
        pred_dual = _reconstruct_gru_dual_ri_from_pl(
            pred_dual,
            n_ris,
            float(np.asarray(aux["pl_hat"]).reshape(-1)[0]),
            pl_eps,
        )
    pred = _select_gru_ri_output(pred_dual, n_ris, mode, uplink_tau_ratio)
    return _ri_to_complex(pred, n_ris)


def _predict_h_ru_plain(model, x_seq, n_ris, target_stats=None):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        x_tensor = torch.tensor(x_seq, dtype=torch.float32, device=device).unsqueeze(0)
        pred = model(x_tensor)
    pred_ri = pred.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
    if target_stats is not None:
        pred_ri = _invert_standardization(pred_ri, target_stats)
    return _ri_to_complex(pred_ri, n_ris)


def _complex_nmse(est, target, eps=1e-12):
    est_arr = np.asarray(est, dtype=np.complex64)
    target_arr = np.asarray(target, dtype=np.complex64)
    err_power = float(np.linalg.norm(est_arr - target_arr) ** 2)
    ref_power = float(np.linalg.norm(target_arr) ** 2)
    return err_power / (ref_power + float(eps))


def _proxy_nmse_for_plot_common_update(
    H_BR,
    h_RUs,
    h_BUs,
    f_vec,
    theta_vec,
    link_switch,
    user_weights,
    update_vars,
    tx_power,
    noise_std,
    eps,
):
    h_ru_arr = np.asarray(h_RUs, dtype=np.complex64)
    if h_ru_arr.ndim != 2:
        return None
    H_arr = np.asarray(H_BR, dtype=np.complex64)
    f_arr = np.asarray(f_vec, dtype=np.complex64).reshape(-1)
    theta_arr = np.asarray(theta_vec, dtype=np.complex64).reshape(-1)
    h_bu_arr = None if h_BUs is None else np.asarray(h_BUs, dtype=np.complex64)
    weights = np.asarray(user_weights, dtype=np.float64).reshape(-1)
    common_vars = np.asarray(update_vars, dtype=np.float64).reshape(-1)
    if weights.size != h_ru_arr.shape[0] or common_vars.size != h_ru_arr.shape[0]:
        return None

    reflect_on, direct_on = int(link_switch[0]), int(link_switch[1])
    h_eff = np.zeros((h_ru_arr.shape[0],), dtype=np.complex64)
    for user_idx in range(h_ru_arr.shape[0]):
        h_k = np.zeros((H_arr.shape[1],), dtype=np.complex64)
        if reflect_on:
            h_k = h_k + H_arr.conj().T.dot(theta_arr * h_ru_arr[user_idx])
        if direct_on and h_bu_arr is not None:
            h_k = h_k + h_bu_arr[user_idx]
        h_eff[user_idx] = f_arr.conj().dot(h_k)

    inner2 = np.abs(h_eff).astype(np.float64) ** 2 + float(eps)
    eta_candidates = float(tx_power) * inner2 / (np.square(weights) * common_vars + float(eps))
    eta = float(np.min(eta_candidates).real)
    if not np.isfinite(eta) or eta <= 0.0:
        return None
    noise_power = float((noise_std ** 2) * tx_power)
    return noise_power / (eta * (np.square(weights.sum()) + float(eps)))


def _append_proxy_plot_common_update_csv(log_stem, round_idx, values, figs_root="./figs"):
    if not log_stem or not values:
        return
    out_dir = os.path.join(figs_root, log_stem)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, PROXY_COMMON_UPDATE_CSV)
    write_header = not os.path.exists(out_path)
    with open(out_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write("round," + ",".join(PROXY_PLOT_MODEL_ORDER) + "\n")
        fields = [str(int(round_idx))]
        for model_name in PROXY_PLOT_MODEL_ORDER:
            value = values.get(model_name)
            fields.append("" if value is None or not np.isfinite(value) else f"{float(value):.12e}")
        f.write(",".join(fields) + "\n")


def _compute_standardization_stats(arrays, feature_ndim, eps=1e-6):
    if not arrays:
        raise ValueError("arrays must be non-empty to compute standardization stats")
    packed = []
    feature_shape = None
    for arr in arrays:
        arr_np = np.asarray(arr, dtype=np.float32)
        if arr_np.ndim < feature_ndim:
            raise ValueError(f"Array with shape {arr_np.shape} has fewer than {feature_ndim} feature dims")
        tail_shape = arr_np.shape[-feature_ndim:] if feature_ndim > 0 else ()
        if feature_shape is None:
            feature_shape = tail_shape
        elif tail_shape != feature_shape:
            raise ValueError(f"Inconsistent feature shape: expected {feature_shape}, got {tail_shape}")
        packed.append(arr_np.reshape(-1, *tail_shape))
    stacked = np.concatenate(packed, axis=0)
    mean = stacked.mean(axis=0, dtype=np.float64).astype(np.float32, copy=False)
    std = stacked.std(axis=0, dtype=np.float64).astype(np.float32, copy=False)
    std = np.maximum(std, float(eps)).astype(np.float32, copy=False)
    return {"mean": mean, "std": std}


def _apply_standardization(arr, stats):
    arr_np = np.asarray(arr, dtype=np.float32)
    return ((arr_np - stats["mean"]) / stats["std"]).astype(np.float32, copy=False)


def _invert_standardization(arr, stats):
    arr_np = np.asarray(arr, dtype=np.float32)
    return ((arr_np * stats["std"]) + stats["mean"]).astype(np.float32, copy=False)


def _standardize_sample(sample, input_stats, target_stats):
    if len(sample) == 2:
        x, y = sample
        return (
            _apply_standardization(x, input_stats),
            _apply_standardization(y, target_stats),
        )
    if len(sample) == 3:
        x, y, meta = sample
        return (
            _apply_standardization(x, input_stats),
            _apply_standardization(y, target_stats),
            meta,
        )
    raise ValueError(f"Expected sample as (x, y) or (x, y, meta), got tuple length {len(sample)}")


def _compute_mobility_tau_loss_weight(alpha_tau_k, is_moving, config):
    if not bool(getattr(config, "enable_mobility_aware_loss_weighting", True)):
        return 1.0
    if not bool(is_moving):
        return 1.0
    alpha_abs = abs(float(alpha_tau_k))
    gain = float(getattr(config, "mobility_tau_loss_weight_gain", 4.0))
    min_w = float(getattr(config, "mobility_tau_loss_weight_min", 1.0))
    max_w = float(getattr(config, "mobility_tau_loss_weight_max", 4.0))
    raw = 1.0 + gain * max(0.0, 1.0 - alpha_abs)
    return float(np.clip(raw, min_w, max_w))


def _build_uplink_reference_truth(h_t, h_tau, mode, uplink_tau_ratio):
    h_t_arr = np.asarray(h_t, dtype=np.complex64)
    h_tau_arr = np.asarray(h_tau, dtype=np.complex64)
    if mode == "t":
        return h_t_arr
    if mode == "uplink_direct":
        return h_tau_arr
    if mode == "uplink_linear":
        rho = float(uplink_tau_ratio)
        return (((1.0 - rho) * h_t_arr) + (rho * h_tau_arr)).astype(np.complex64, copy=False)
    raise ValueError(f"Unsupported GRU output mode: {mode}")


def _build_gru_uplink_oracle_prediction(h_t, alpha_tau, pl_t, pl_tau, mode, uplink_tau_ratio, eps=1e-12):
    h_t_arr = np.asarray(h_t, dtype=np.complex64)
    alpha_arr = np.asarray(alpha_tau, dtype=np.float32).reshape(-1, 1)
    pl_t_arr = np.maximum(np.asarray(pl_t, dtype=np.float32).reshape(-1, 1), float(eps))
    pl_tau_arr = np.maximum(np.asarray(pl_tau, dtype=np.float32).reshape(-1, 1), float(eps))
    scale_tau = np.sqrt(pl_tau_arr / pl_t_arr).astype(np.float32, copy=False)
    h_tau_oracle = (alpha_arr * scale_tau * h_t_arr).astype(np.complex64, copy=False)
    return _build_uplink_reference_truth(h_t_arr, h_tau_oracle, mode, uplink_tau_ratio)


def _log_grouped_nmse(logger, branch_name, per_user_nmse, moving_user_mask, metric_label):
    if per_user_nmse is None:
        return
    nmse_arr = np.asarray(per_user_nmse, dtype=np.float64).reshape(-1)
    moving_mask = np.asarray(moving_user_mask, dtype=bool).reshape(-1)
    if nmse_arr.size != moving_mask.size:
        raise ValueError(f"Grouped NMSE mask size mismatch: {nmse_arr.size} vs {moving_mask.size}")
    static_mask = ~moving_mask
    parts = []
    if np.any(static_mask):
        parts.append(
            f"static(n={int(np.sum(static_mask))})={_highlight_metric_value(float(np.mean(nmse_arr[static_mask])), branch_name)}"
        )
    if np.any(moving_mask):
        parts.append(
            f"moving(n={int(np.sum(moving_mask))})={_highlight_metric_value(float(np.mean(nmse_arr[moving_mask])), branch_name)}"
        )
    if parts:
        logger.info(_colorize_branch_line(branch_name, f"{branch_name} {metric_label} groups -> " + ", ".join(parts)))


def _log_partitioned_nmse(logger, branch_name, per_user_nmse, group_assignment, metric_label, labels=("low", "high")):
    if per_user_nmse is None or group_assignment is None:
        return
    nmse_arr = np.asarray(per_user_nmse, dtype=np.float64).reshape(-1)
    group_arr = np.asarray(group_assignment, dtype=np.int64).reshape(-1)
    if nmse_arr.size != group_arr.size:
        raise ValueError(f"Partitioned NMSE group size mismatch: {nmse_arr.size} vs {group_arr.size}")
    parts = []
    for group_value, label in enumerate(labels):
        mask = (group_arr == int(group_value))
        if np.any(mask):
            parts.append(
                f"{label}(n={int(np.sum(mask))})={_highlight_metric_value(float(np.mean(nmse_arr[mask])), branch_name)}"
            )
    if parts:
        logger.info(_colorize_branch_line(branch_name, f"{branch_name} {metric_label} semantic-groups -> " + ", ".join(parts)))


def _build_window_sample(obs_buffer, current_step, W, obs_dim, pad_val):
    seq = list(obs_buffer)
    if seq:
        seq[-1] = current_step
    else:
        seq = [current_step]
    X_seq = np.stack(seq, axis=0)
    if X_seq.shape[0] < W:
        pad_len = W - X_seq.shape[0]
        pad = np.full((pad_len, 2, obs_dim), pad_val, dtype=np.float32)
        X_seq = np.concatenate([pad, X_seq], axis=0)
    return X_seq.astype(np.float32, copy=False)


def _flatten_sample_groups(sample_groups):
    flat = []
    for item in sample_groups:
        if item is None:
            continue
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def _format_min_mean_max(values):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return "min/mean/max=(n/a/n/a/n/a)"
    return (
        f"min/mean/max=({float(arr.min()):.4e}/"
        f"{float(arr.mean()):.4e}/"
        f"{float(arr.max()):.4e})"
    )


def _format_xy(coord):
    xy = np.asarray(coord, dtype=np.float64).reshape(-1)
    if xy.size != 2:
        return "[n/a, n/a]"
    return f"[{float(xy[0]):.3f}, {float(xy[1]):.3f}]"


def _complex_delta_metrics(h_t, h_tau, eps=1e-12):
    h_t_arr = np.asarray(h_t, dtype=np.complex64).reshape(-1)
    h_tau_arr = np.asarray(h_tau, dtype=np.complex64).reshape(-1)
    delta = h_tau_arr - h_t_arr
    delta_norm = float(np.linalg.norm(delta))
    base_norm = float(np.linalg.norm(h_t_arr))
    delta_rel = delta_norm / (base_norm + float(eps))
    return delta_norm, delta_rel


def _build_gru_grouping_config(config):
    return GroupingSCAConfig(
        lambda_d=float(getattr(config, "gru_group_lambda_d", 8.0)),
        c_beta=float(getattr(config, "gru_group_c_beta", 0.5)),
        c_d=float(getattr(config, "gru_group_c_d", 0.5)),
        lambda_s=float(getattr(config, "gru_group_lambda_s", 0.0)),
        gamma=float(getattr(config, "gru_group_gamma", 0.0)),
        lambda_h=float(getattr(config, "gru_group_lambda_h", 0.0)),
        tau=float(getattr(config, "gru_group_tau", 0.0)),
        k_min=int(getattr(config, "gru_group_k_min", 2)),
        sca_max_iters=int(getattr(config, "gru_group_sca_max_iters", 8)),
        sca_tol=float(getattr(config, "gru_group_sca_tol", 1e-4)),
        relaxation=float(getattr(config, "gru_group_relaxation", 0.7)),
        lp_method=str(getattr(config, "gru_group_lp_method", "highs")),
    )


def _relative_scalar_change(curr_value, prev_value, eps):
    return abs(float(curr_value) - float(prev_value)) / (abs(float(prev_value)) + float(eps))


def _build_gru_grouping_proxies(pl_pred_round, delta_norm_round, h_ru_est, eps):
    beta_hat = np.maximum(np.asarray(pl_pred_round, dtype=np.float64).reshape(-1), float(eps))
    delta_norm = np.maximum(np.asarray(delta_norm_round, dtype=np.float64).reshape(-1), 0.0)
    h_norm = np.linalg.norm(np.asarray(h_ru_est, dtype=np.complex128), axis=1)
    d_hat = delta_norm / (h_norm + float(eps))
    B_round = float(np.mean(-np.log(beta_hat)))
    D_round = float(np.mean(np.log1p(d_hat)))
    return beta_hat.astype(np.float64, copy=False), d_hat.astype(np.float64, copy=False), B_round, D_round


def _build_gru_group_beam_matrix(num_users, f_single, group_beams=None, group_assignment=None):
    if group_beams is None or group_assignment is None:
        return np.repeat(np.asarray(f_single, dtype=np.complex64).reshape(1, -1), int(num_users), axis=0)
    assign_arr = np.asarray(group_assignment, dtype=np.int64).reshape(-1)
    if assign_arr.size != int(num_users):
        raise ValueError("group_assignment size mismatch")
    beam_rows = []
    for user_idx in range(int(num_users)):
        group_idx = int(assign_arr[user_idx])
        beam_rows.append(np.asarray(group_beams[group_idx], dtype=np.complex64).reshape(-1))
    return np.stack(beam_rows, axis=0).astype(np.complex64, copy=False)


def _run_gru_grouping_optimizer(
        *,
        beta_hat,
        d_hat,
        beta_ema,
        d_ema,
        H_BR,
        h_ru_est,
        f_single,
        group_beams=None,
        group_assignment=None,
        cfg,
        prev_x_hard=None,
        prev_x_soft=None,
        prev_mu=None,
):
    num_users = int(np.asarray(beta_hat).reshape(-1).size)
    f_prev = _build_gru_group_beam_matrix(
        num_users=num_users,
        f_single=f_single,
        group_beams=group_beams,
        group_assignment=group_assignment,
    )
    warm_start = GroupingWarmStart(
        x_prev=None if prev_x_hard is None else np.asarray(prev_x_hard, dtype=np.float64).reshape(-1),
        beta_ema=np.asarray(beta_ema, dtype=np.float64).reshape(-1),
        d_ema=np.asarray(d_ema, dtype=np.float64).reshape(-1),
        x_init=None if prev_x_soft is None else np.asarray(prev_x_soft, dtype=np.float64).reshape(-1),
        mu_init=None if prev_mu is None else np.asarray(prev_mu, dtype=np.float64).reshape(2),
    )
    return optimize_risk_grouping_sca(
        beta_hat=np.asarray(beta_hat, dtype=np.float64).reshape(-1),
        d_hat=np.asarray(d_hat, dtype=np.float64).reshape(-1),
        H_BR=np.asarray(H_BR, dtype=np.complex64),
        f_prev=f_prev,
        h_ru_est=np.asarray(h_ru_est, dtype=np.complex64),
        cfg=cfg,
        warm_start=warm_start,
    )


def _save_user_location_velocity_plot(
        initial_positions_xy,
        velocity_vectors_xy,
        bs_position_xy,
        ris_position_xy,
        out_path,
):
    """
    Save a 2D map of initial user positions and movement vectors.
    Arrow lengths remain proportional to speed through a single global scale, while the
    coordinate system is rendered as a square equal-aspect geometry map.
    """
    positions = np.asarray(initial_positions_xy, dtype=np.float64).reshape(-1, 2)
    velocities = np.asarray(velocity_vectors_xy, dtype=np.float64).reshape(-1, 2)
    if positions.shape != velocities.shape:
        raise ValueError(
            f"positions and velocities must share shape (K,2), got {positions.shape} vs {velocities.shape}"
        )
    if positions.shape[0] == 0:
        raise ValueError("No user positions to plot")

    bs_xy = np.asarray(bs_position_xy, dtype=np.float64).reshape(2)
    ris_xy = np.asarray(ris_position_xy, dtype=np.float64).reshape(2)

    speed = np.linalg.norm(velocities, axis=1)
    speed_max = float(np.max(speed))

    base_points = np.vstack([positions, bs_xy[None, :], ris_xy[None, :]])
    base_x_min = float(np.min(base_points[:, 0]))
    base_x_max = float(np.max(base_points[:, 0]))
    base_y_min = float(np.min(base_points[:, 1]))
    base_y_max = float(np.max(base_points[:, 1]))
    base_span = max(base_x_max - base_x_min, base_y_max - base_y_min, 100.0)
    vector_plot_scale = 0.18 * base_span / max(speed_max, 1e-12)
    velocities_plot = velocities * vector_plot_scale
    arrow_end = positions + velocities_plot
    all_points = np.vstack([base_points, arrow_end])
    raw_x_min = float(np.min(all_points[:, 0]))
    raw_x_max = float(np.max(all_points[:, 0]))
    raw_y_min = float(np.min(all_points[:, 1]))
    raw_y_max = float(np.max(all_points[:, 1]))

    major_grid_step = 10.0
    minor_grid_step = 5.0
    x_min_aligned = major_grid_step * math.floor(raw_x_min / major_grid_step)
    x_max_aligned = major_grid_step * math.ceil(raw_x_max / major_grid_step)
    y_min_aligned = major_grid_step * math.floor(raw_y_min / major_grid_step)
    y_max_aligned = major_grid_step * math.ceil(raw_y_max / major_grid_step)
    x_span = x_max_aligned - x_min_aligned
    y_span = y_max_aligned - y_min_aligned
    side_span = major_grid_step * math.ceil(max(x_span, y_span, 100.0) / major_grid_step)
    x_pad = 0.5 * (side_span - x_span)
    y_pad = 0.5 * (side_span - y_span)
    x_min = x_min_aligned - x_pad
    x_max = x_max_aligned + x_pad
    y_min = y_min_aligned - y_pad
    y_max = y_max_aligned + y_pad

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        s=16,
        color="#202020",
        edgecolors="black",
        linewidths=0.35,
        zorder=3,
        label="User initial positions",
    )

    nonzero = speed > 1e-12
    quiver = None
    if np.any(nonzero):
        quiver = ax.quiver(
            positions[nonzero, 0],
            positions[nonzero, 1],
            velocities_plot[nonzero, 0],
            velocities_plot[nonzero, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.0032,
            headwidth=4.8,
            headlength=6.0,
            headaxislength=5.2,
            color="#1f5aa6",
            linewidth=0.6,
            zorder=2,
        )

    ax.scatter(
        [bs_xy[0]],
        [bs_xy[1]],
        marker="^",
        s=46,
        facecolors="white",
        edgecolors="#b22222",
        linewidths=1.2,
        label="BS",
        zorder=4,
    )
    ax.scatter(
        [ris_xy[0]],
        [ris_xy[1]],
        marker="D",
        s=34,
        facecolors="white",
        edgecolors="#1f7a3f",
        linewidths=1.2,
        label="RIS",
        zorder=4,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Initial User Geometry and Velocity Vectors")
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MultipleLocator(major_grid_step))
    ax.yaxis.set_major_locator(MultipleLocator(major_grid_step))
    ax.xaxis.set_minor_locator(MultipleLocator(minor_grid_step))
    ax.yaxis.set_minor_locator(MultipleLocator(minor_grid_step))
    ax.grid(which="major", color="#8a8a8a", linestyle="-", linewidth=0.9, alpha=0.9)
    ax.grid(which="minor", color="#c2c2c2", linestyle="--", linewidth=0.55, alpha=0.85)
    ax.tick_params(axis="both", which="major", labelsize=9)
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=8,
        framealpha=0.9,
        handlelength=1.2,
        labelspacing=0.3,
    )
    fig.tight_layout(rect=(0.0, 0.06, 0.84, 1.0))
    if quiver is not None:
        ref_speed = max(5.0, 5.0 * math.ceil(speed_max / 5.0))
        ref_len_axes = (ref_speed * vector_plot_scale) / max(x_max - x_min, 1e-9)
        fig.canvas.draw()
        legend_bbox = legend.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())
        ref_x0 = float(legend_bbox.x0)
        ref_y0 = float(legend_bbox.y0) - 0.08
        ref_arrow = FancyArrowPatch(
            (ref_x0, ref_y0),
            (ref_x0 + ref_len_axes, ref_y0),
            arrowstyle="-|>",
            mutation_scale=12.0,
            linewidth=1.6,
            color="black",
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(ref_arrow)
        ax.text(
            ref_x0,
            ref_y0 - 0.035,
            f"Speed reference: {ref_speed:.1f} m/s",
            fontsize=8,
            color="black",
            ha="left",
            va="top",
            transform=ax.transAxes,
            clip_on=False,
        )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _format_complex_matrix_for_log(vec_complex):
    mat = np.asarray(vec_complex, dtype=np.complex64).reshape(1, -1)
    return np.array2string(
        np.round(mat, 4),
        precision=4,
        separator=", ",
        suppress_small=False,
        threshold=1000000,
        max_line_width=200,
    )


def _log_gru_dual_head_debug(logger, round_idx, user_idx, pred_dual_ri, target_dual_ri, n_ris, loss_t, loss_tau):
    pred_t_ri, pred_tau_ri = _split_gru_dual_ri(pred_dual_ri, n_ris)
    target_t_ri, target_tau_ri = _split_gru_dual_ri(target_dual_ri, n_ris)
    pred_t = _ri_to_complex(pred_t_ri, n_ris)
    pred_tau = _ri_to_complex(pred_tau_ri, n_ris)
    target_t = _ri_to_complex(target_t_ri, n_ris)
    target_tau = _ri_to_complex(target_tau_ri, n_ris)
    total_loss = None
    if (loss_t is not None) and (loss_tau is not None):
        total_loss = 0.5 * (float(loss_t) + float(loss_tau))

    loss_t_text = "n/a" if loss_t is None else f"{float(loss_t):.6f}"
    loss_tau_text = "n/a" if loss_tau is None else f"{float(loss_tau):.6f}"
    total_loss_text = "n/a" if total_loss is None else f"{total_loss:.6f}"
    prefix = f"Round {round_idx} User {user_idx + 1} GRU"
    logger.info(
        f"{prefix} loss detail -> total={total_loss_text}, "
        f"t={loss_t_text}, tau={loss_tau_text}"
    )
    logger.info(f"{prefix} target t matrix:\n{_format_complex_matrix_for_log(target_t)}")
    logger.info(f"{prefix} pred t matrix:\n{_format_complex_matrix_for_log(pred_t)}")
    logger.info(f"{prefix} target tau matrix:\n{_format_complex_matrix_for_log(target_tau)}")
    logger.info(f"{prefix} pred tau matrix:\n{_format_complex_matrix_for_log(pred_tau)}")


def _flatten_head_state(head_state):
    """Flatten one user's head state dict to a 1D numpy vector."""
    parts = []
    for key in sorted(head_state.keys()):
        parts.append(head_state[key].detach().cpu().reshape(-1).float())
    if not parts:
        return np.zeros((0,), dtype=np.float32)
    return torch.cat(parts, dim=0).numpy()


def _mean_head_state(user_head_states, user_ids):
    ids = np.asarray(user_ids, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        raise ValueError("user_ids must be non-empty to compute mean head state")
    ref_state = user_head_states[int(ids[0])]
    mean_state = {}
    for key in sorted(ref_state.keys()):
        stacked = torch.stack(
            [user_head_states[int(user_idx)][key].detach().cpu().float() for user_idx in ids.tolist()],
            dim=0,
        )
        mean_state[key] = stacked.mean(dim=0).to(dtype=ref_state[key].dtype)
    return mean_state


def _replace_heads_with_group_means(user_head_states, group_assignment):
    assign = np.asarray(group_assignment, dtype=np.int64).reshape(-1)
    updated = [copy.deepcopy(head_state) for head_state in user_head_states]
    for group_idx in range(2):
        user_ids = np.flatnonzero(assign == group_idx).astype(np.int64)
        if user_ids.size == 0:
            continue
        mean_state = _mean_head_state(user_head_states, user_ids)
        for user_idx in user_ids.tolist():
            updated[int(user_idx)] = {key: value.detach().clone() for key, value in mean_state.items()}
    return updated


def _randomize_user_heads_from_fresh_model(
        *,
        num_users,
        observation_dim,
        output_dim,
        enable_pl_factorization,
        log_pl_min,
        log_pl_max,
):
    randomized = []
    for _ in range(int(num_users)):
        fresh_model = CSICNNGRU(
            observation_dim=observation_dim,
            output_dim=output_dim,
            enable_pl_factorization=enable_pl_factorization,
            log_pl_min=log_pl_min,
            log_pl_max=log_pl_max,
        )
        randomized.append(
            {
                key: value.detach().clone()
                for key, value in fresh_model.state_dict().items()
                if key.startswith("head")
            }
        )
    return randomized


def _create_fresh_gru_model(*, observation_dim, output_dim, enable_pl_factorization, log_pl_min, log_pl_max):
    return CSICNNGRU(
        observation_dim=observation_dim,
        output_dim=output_dim,
        enable_pl_factorization=enable_pl_factorization,
        log_pl_min=log_pl_min,
        log_pl_max=log_pl_max,
    )


def _group_head_dispersion_stats(user_head_states, group_assignment, eps=1e-12):
    assign = np.asarray(group_assignment, dtype=np.int64).reshape(-1)
    stats = []
    for group_idx in range(2):
        user_ids = np.flatnonzero(assign == group_idx).astype(np.int64)
        if user_ids.size == 0:
            stats.append({"group_idx": int(group_idx), "size": 0, "mean_rel": float("nan"), "max_rel": float("nan")})
            continue
        mean_state = _mean_head_state(user_head_states, user_ids)
        mean_vec = _flatten_head_state(mean_state).astype(np.float64, copy=False)
        mean_norm = float(np.linalg.norm(mean_vec))
        rel_dists = []
        for user_idx in user_ids.tolist():
            user_vec = _flatten_head_state(user_head_states[int(user_idx)]).astype(np.float64, copy=False)
            rel = float(np.linalg.norm(user_vec - mean_vec) / (mean_norm + float(eps)))
            rel_dists.append(rel)
        stats.append(
            {
                "group_idx": int(group_idx),
                "size": int(user_ids.size),
                "mean_rel": float(np.mean(rel_dists)),
                "max_rel": float(np.max(rel_dists)),
            }
        )
    return stats


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
    meta_algorithm = _parse_meta_algorithm(config.meta_algorithm)
    ota_use_estimated_h_ru_for_aggregation = bool(getattr(config, "ota_use_estimated_h_ru_for_aggregation", False))
    logger = Logger(config=config) if config.log_to_file else Logger()
    # logger = Logger(config.log_file_path) if config.log_to_file else Logger()
    logger.info("Initializing simulation...")
    logger.info(f"Config fingerprint(full)={config.fingerprint()}")
    logger.info(f"Experiment prefix={config.log_prefix()}")
    logger.info(f"Optimizer tag={config.optimizer_tag()}")
    logger.info(
        "OTA aggregation h_RU source="
        + ("estimated (branch-specific)" if ota_use_estimated_h_ru_for_aggregation else "true/oracle simulator")
    )
    np.random.seed(0)
    torch.manual_seed(0)

    debug_head_plot_enabled = bool(getattr(config, "enable_reptile_head_debug_plot", False))
    debug_head_plot_every = int(getattr(config, "reptile_head_debug_every", 10))
    if debug_head_plot_every <= 0:
        debug_head_plot_every = 10
    debug_head_plot_root = str(getattr(config, "reptile_head_debug_root", "debug"))
    debug_gru_dual_target_log_enabled = bool(getattr(config, "enable_gru_dual_target_debug_log", False))
    debug_gru_state_plot_enabled = bool(getattr(config, "enable_gru_state_diff_debug_plot", False))
    debug_gru_state_plot_every = int(getattr(config, "gru_state_diff_debug_every", 10))
    if debug_gru_state_plot_every <= 0:
        debug_gru_state_plot_every = 10
    eta_snapshot_enabled = bool(getattr(config, "enable_eta_response_snapshot", False))
    eta_snapshot_root = str(getattr(config, "eta_response_snapshot_root", "debug/eta_response_snapshots"))
    eta_snapshot_every = int(getattr(config, "eta_response_snapshot_every", 1))
    if eta_snapshot_every <= 0:
        raise ValueError("Config.eta_response_snapshot_every must be positive")
    eta_snapshot_run_dir = None
    if eta_snapshot_enabled:
        eta_snapshot_run_dir = os.path.join(eta_snapshot_root, logger.stem or config.log_prefix())
    mobility_snapshot_enabled = bool(getattr(config, "enable_mobility_debug_snapshot", False))
    mobility_snapshot_root = str(getattr(config, "mobility_debug_snapshot_root", "debug/mobility_snapshots"))
    mobility_snapshot_every = int(getattr(config, "mobility_debug_snapshot_every", 1))
    if mobility_snapshot_every <= 0:
        raise ValueError("Config.mobility_debug_snapshot_every must be positive")
    mobility_snapshot_run_dir = None
    if mobility_snapshot_enabled:
        mobility_snapshot_run_dir = os.path.join(mobility_snapshot_root, logger.stem or config.log_prefix())
    delta_motion_snapshot_enabled = bool(getattr(config, "enable_delta_motion_debug_snapshot", False))
    delta_motion_snapshot_root = str(getattr(config, "delta_motion_debug_snapshot_root", "debug/delta_motion_snapshots"))
    delta_motion_snapshot_every = int(getattr(config, "delta_motion_debug_snapshot_every", 1))
    if delta_motion_snapshot_every <= 0:
        raise ValueError("Config.delta_motion_debug_snapshot_every must be positive")
    delta_motion_snapshot_run_dir = None
    if delta_motion_snapshot_enabled:
        delta_motion_snapshot_run_dir = os.path.join(delta_motion_snapshot_root, logger.stem or config.log_prefix())
    gru_pl_snapshot_enabled = bool(getattr(config, "enable_gru_pl_debug_snapshot", False))
    gru_pl_snapshot_root = str(getattr(config, "gru_pl_debug_snapshot_root", "debug/gru_pl_snapshots"))
    gru_pl_snapshot_every = int(getattr(config, "gru_pl_debug_snapshot_every", 1))
    if gru_pl_snapshot_every <= 0:
        raise ValueError("Config.gru_pl_debug_snapshot_every must be positive")
    gru_pl_snapshot_run_dir = None
    if gru_pl_snapshot_enabled:
        gru_pl_snapshot_run_dir = os.path.join(gru_pl_snapshot_root, logger.stem or config.log_prefix())
    gru_group_switch_sensitivity_snapshot_enabled = bool(
        getattr(config, "enable_gru_group_switch_sensitivity_snapshot", False)
    )
    gru_group_switch_sensitivity_snapshot_root = str(
        getattr(config, "gru_group_switch_sensitivity_snapshot_root", "debug/gru_group_switch_sensitivity_snapshots")
    )
    gru_group_switch_sensitivity_snapshot_every = int(
        getattr(config, "gru_group_switch_sensitivity_snapshot_every", 1)
    )
    if gru_group_switch_sensitivity_snapshot_every <= 0:
        raise ValueError("Config.gru_group_switch_sensitivity_snapshot_every must be positive")
    if gru_group_switch_sensitivity_snapshot_enabled:
        gru_group_switch_sensitivity_tau_b_grid = _parse_nonnegative_float_grid(
            getattr(config, "gru_group_switch_sensitivity_tau_b_values", [0.065]),
            "gru_group_switch_sensitivity_tau_b_values",
        )
        gru_group_switch_sensitivity_tau_d_grid = _parse_nonnegative_float_grid(
            getattr(config, "gru_group_switch_sensitivity_tau_d_values", [0.065]),
            "gru_group_switch_sensitivity_tau_d_values",
        )
    else:
        gru_group_switch_sensitivity_tau_b_grid = np.asarray([0.065], dtype=np.float64)
        gru_group_switch_sensitivity_tau_d_grid = np.asarray([0.065], dtype=np.float64)
    gru_group_switch_sensitivity_snapshot_run_dir = None
    if gru_group_switch_sensitivity_snapshot_enabled:
        gru_group_switch_sensitivity_snapshot_run_dir = os.path.join(
            gru_group_switch_sensitivity_snapshot_root,
            logger.stem or config.log_prefix(),
        )
    debug_head_plot_run_dir = None
    debug_head_plot_gru_dir = None
    debug_head_plot_arch_dir = None
    debug_gru_state_plot_dir = None
    if (debug_head_plot_enabled or debug_gru_state_plot_enabled) and meta_algorithm == "reptile":
        debug_run_tag = logger.stem or config.log_prefix()
        debug_head_plot_run_dir = os.path.join(debug_head_plot_root, debug_run_tag)
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
    direct_only_mode = (reflect_on == 0 and direct_on == 1)
    mode_desc = "reflection only" if (reflect_on == 1 and direct_on == 0) else \
        "direct only (no RIS)" if (reflect_on == 0 and direct_on == 1) else "reflection + direct"
    logger.info(f"Link switch [reflect,direct]={list(link_switch)} -> \033[33m{mode_desc}\033[0m")
    if reflect_on == 0:
        logger.info("Reflection link disabled: RIS contribution set to 0.")
    if direct_on == 0:
        logger.info("Direct link disabled.")

    pilot_snr_db = float(config.pilot_SNR_dB)
    pilot_noise_std = float(np.power(10.0, -pilot_snr_db / 20.0))
    logger.info(f"Pilot SNR_dB: {pilot_snr_db:.2f}")
    gru_csi_target_mode = _parse_gru_target_mode(config.gru_csi_target_mode)
    uplink_tau_ratio = float(getattr(config, "uplink_tau_ratio", 0.5))
    if not (0.0 <= uplink_tau_ratio <= 1.0):
        raise ValueError("Config.uplink_tau_ratio must be in [0, 1]")
    use_uplink_linear = (gru_csi_target_mode == "uplink_linear")
    use_uplink_direct = (gru_csi_target_mode == "uplink_direct")
    use_uplink_target = use_uplink_linear or use_uplink_direct
    enable_gru_pl_factorization = bool(getattr(config, "enable_gru_pl_factorization", False))
    gru_pl_loss_weight = float(getattr(config, "gru_pl_loss_weight", 0.05))
    gru_pl_eps = float(getattr(config, "gru_pl_eps", 1e-12))
    enable_gru_semantic_grouping = bool(getattr(config, "enable_gru_semantic_grouping", False))
    gru_group_switch_min_round = int(getattr(config, "gru_group_switch_min_round", 15))
    gru_group_switch_patience = int(getattr(config, "gru_group_switch_patience", 3))
    gru_group_switch_ema_lambda = float(getattr(config, "gru_group_switch_ema_lambda", 0.8))
    gru_group_switch_tau_b = float(getattr(config, "gru_group_switch_tau_b", 0.02))
    gru_group_switch_tau_d = float(getattr(config, "gru_group_switch_tau_d", 0.02))
    gru_group_eps = float(getattr(config, "gru_group_eps", 1e-8))
    gru_group_freeze_after_switch = bool(getattr(config, "gru_group_freeze_after_switch", False))
    gru_groupwise_standardization = bool(getattr(config, "gru_groupwise_standardization", False))
    gru_restart_training_after_switch = bool(getattr(config, "gru_restart_training_after_switch", False))
    gru_head_reset_to_group_mean_on_switch = bool(
        getattr(config, "gru_head_reset_to_group_mean_on_switch", False)
    )
    gru_head_randomize_on_switch = bool(getattr(config, "gru_head_randomize_on_switch", False))
    gru_head_disable_persistence_after_switch = bool(
        getattr(config, "gru_head_disable_persistence_after_switch", False)
    )
    gru_log_group_head_dispersion = bool(getattr(config, "gru_log_group_head_dispersion", False))
    gru_reset_hidden_on_group_switch = bool(getattr(config, "gru_reset_hidden_on_group_switch", False))
    gru_reset_hidden_on_group_change_each_round = bool(
        getattr(config, "gru_reset_hidden_on_group_change_each_round", False)
    )
    gru_disable_persistent_hidden_after_switch = bool(
        getattr(config, "gru_disable_persistent_hidden_after_switch", False)
    )
    if gru_pl_loss_weight < 0.0:
        raise ValueError("Config.gru_pl_loss_weight must be nonnegative")
    if gru_pl_eps <= 0.0:
        raise ValueError("Config.gru_pl_eps must be positive")
    if gru_group_switch_min_round < 1:
        raise ValueError("Config.gru_group_switch_min_round must be at least 1")
    if gru_group_switch_patience < 1:
        raise ValueError("Config.gru_group_switch_patience must be positive")
    if not (0.0 <= gru_group_switch_ema_lambda < 1.0):
        raise ValueError("Config.gru_group_switch_ema_lambda must be in [0, 1)")
    if gru_group_switch_tau_b < 0.0 or gru_group_switch_tau_d < 0.0:
        raise ValueError("Config.gru_group_switch_tau_b/tau_d must be nonnegative")
    if gru_group_eps <= 0.0:
        raise ValueError("Config.gru_group_eps must be positive")
    if gru_group_switch_sensitivity_snapshot_enabled:
        gru_group_switch_sensitivity_tau_b_grid = np.unique(
            np.concatenate([gru_group_switch_sensitivity_tau_b_grid, np.asarray([gru_group_switch_tau_b])])
        )
        gru_group_switch_sensitivity_tau_d_grid = np.unique(
            np.concatenate([gru_group_switch_sensitivity_tau_d_grid, np.asarray([gru_group_switch_tau_d])])
        )
    gru_grouping_cfg = _build_gru_grouping_config(config) if enable_gru_semantic_grouping else None

    ru_evolver = None
    h_BUs = None
    if config.use_synthetic_data:
        ru_evolver = build_ru_channel_evolver_from_config(config)
        H_BR = ru_evolver.initialize_br_channel()
        h_RUs, h_BUs = ru_evolver.initialize_user_channels(include_direct=(direct_on == 1))

        theta_pilot = pilot_gen.generate_pilot_pattern(config.num_pilots, config.num_ris_elements)
        theta_ota = np.ones(config.num_ris_elements, dtype=np.complex64)

    else:
        H_BR, h_RUs_static, h_BUs_static = RISdata.load_data(
            config.risdata_root,
            num_users=config.num_users,
            num_bs_antennas=config.num_bs_antennas,
            subset=str(getattr(config, "risdata_subset", "specular")),
            result_key=str(getattr(config, "risdata_result_key", "rand")),
            reference_key=str(getattr(config, "risdata_reference_key", "RISallOff")),
            freq_hz=float(getattr(config, "risdata_freq_hz", 5.375e9)),
            freq_tol_hz=float(getattr(config, "risdata_freq_tol_hz", 30e6)),
            max_pattern_samples=getattr(config, "risdata_max_pattern_samples", None),
            min_snr_db=getattr(config, "risdata_min_snr_db", None),
        )

        # Use the loaded static measurement-derived channels and simulate variation via AR(1).
        H_BR = H_BR.astype(np.complex64)
        if h_RUs_static.ndim == 2:
            h_RUs = h_RUs_static.astype(np.complex64)  # (K, N)
        else:
            h_RUs = h_RUs_static[:, 0, :].astype(np.complex64)
        if int(config.num_ris_elements) != int(h_RUs.shape[1]):
            logger.info(
                "RISdata adapter overrides num_ris_elements from "
                f"{int(config.num_ris_elements)} to {int(h_RUs.shape[1])} based on measurement patterns."
            )
            config.num_ris_elements = int(h_RUs.shape[1])

        ru_evolver = build_ru_channel_evolver_from_config(config)

        theta_pilot = pilot_gen.generate_pilot_pattern(config.num_pilots, config.num_ris_elements)
        theta_ota = np.ones(config.num_ris_elements, dtype=np.complex64)
        if direct_on == 1:
            if h_BUs_static is None:
                logger.info("Direct link enabled but RISdata adapter has no h_BU; using synthetic BS-UE channels.")
                h_BUs = (np.random.randn(config.num_users, config.num_bs_antennas) +
                         1j * np.random.randn(config.num_users, config.num_bs_antennas)) / np.sqrt(2)
            else:
                if h_BUs_static.ndim == 2:
                    h_BUs = h_BUs_static.astype(np.complex64)
                else:
                    h_BUs = h_BUs_static[:, 0, :].astype(np.complex64)

    cluster_counts = np.bincount(ru_evolver.cluster_ids, minlength=len(config.user_cluster_ratios)).tolist()
    ris_dist_t0 = ru_evolver.ris_distances(ru_evolver.initial_positions)
    speed_mag = ru_evolver.speed_magnitudes
    logger.info(
        "Mobility-driven synthetic channel model: "
        f"BS={np.round(ru_evolver.bs_position, 3).tolist()}, "
        f"RIS={np.round(ru_evolver.ris_position, 3).tolist()}, "
        f"dt={float(config.channel_time_step):.4e}s, "
        f"fc={float(config.channel_carrier_frequency_hz):.4e}Hz"
    )
    logger.info(
        f"User mobility clusters counts={cluster_counts}, "
        f"|v|(min/mean/max)=({float(speed_mag.min()):.3f}/{float(speed_mag.mean()):.3f}/{float(speed_mag.max()):.3f}) m/s"
    )
    direction_mode = (
        "random"
        if ru_evolver.fixed_direction_deg is None
        else f"fixed {float(ru_evolver.fixed_direction_deg):.2f} deg"
    )
    moving_users_text = "all" if len(ru_evolver.moving_user_ids) == int(config.num_users) else str(ru_evolver.moving_user_ids)
    logger.info(f"Mobility direction mode={direction_mode}, moving_users={moving_users_text}")
    for user_idx, velocity_xy in enumerate(ru_evolver.velocity_vectors, start=1):
        logger.info(
            f"User {user_idx} fixed velocity -> "
            f"v=[{float(velocity_xy[0]):.4f}, {float(velocity_xy[1]):.4f}] m/s, "
            f"|v|={float(np.linalg.norm(velocity_xy)):.4f} m/s"
        )
    logger.info(
        f"RIS distance at t0 (min/mean/max)=({float(ris_dist_t0.min()):.3f}/{float(ris_dist_t0.mean()):.3f}/{float(ris_dist_t0.max()):.3f})"
    )
    if direct_on == 1:
        direct_dist_t0 = ru_evolver.direct_distances(ru_evolver.initial_positions)
        logger.info(
            f"BS-user distance at t0 (min/mean/max)=("
            f"{float(direct_dist_t0.min()):.3f}/{float(direct_dist_t0.mean()):.3f}/{float(direct_dist_t0.max()):.3f})"
        )

    gru_log_pl_min = None
    gru_log_pl_max = None
    if enable_gru_pl_factorization:
        gru_log_pl_min, gru_log_pl_max = _estimate_ris_pathloss_log_bounds(
            ru_evolver,
            config.num_rounds,
            uplink_tau_ratio,
            eps=gru_pl_eps,
        )
        logger.info(
            "GRU pathloss factorization enabled: "
            f"log(PL_RIS) bounds=[{gru_log_pl_min:.4f}, {gru_log_pl_max:.4f}], "
            f"pl_loss_weight={gru_pl_loss_weight:.4f}"
        )

    # Plot initial user coordinates and motion vectors (debug/visualization only).
    try:
        location_out_dir = os.path.join("figs", "location")
        run_token = (logger.stem or "run").split("_")[-1]
        location_prefix = f"{config.experiment_fingerprint()}_{config.optimizer_tag()}_{run_token}"
        location_out_path = os.path.join(
            location_out_dir,
            f"{location_prefix}_user_motion.png",
        )
        saved_location_path = _save_user_location_velocity_plot(
            initial_positions_xy=ru_evolver.initial_positions,
            velocity_vectors_xy=ru_evolver.velocity_vectors,
            bs_position_xy=ru_evolver.bs_position,
            ris_position_xy=ru_evolver.ris_position,
            out_path=location_out_path,
        )
        logger.info(f"Saved user location/motion map: {saved_location_path}")
    except Exception as exc:
        logger.info(f"Failed to save user location/motion map: {exc}")

    # Set initial BS beamforming vector f (e.g., all ones)
    f_beam = np.ones(config.num_bs_antennas, dtype=np.complex64)

    # Initialize global model
    observation_dim = config.num_pilots  # each pilot yields one observation value (if scalar) or we consider multi-dim
    # Actually, each pilot observation is complex, we consider 2 channels (real & imag)
    obs_dim = config.num_pilots
    # Single-horizon CSI target is RIS-user CSI h_RU (real-imag stacked).
    # The GRU predicts both t and tau, so its head output dimension is doubled internally.
    output_dim = 2 * config.num_ris_elements
    log_oa_vectors = bool(getattr(config, "log_oa_vectors", False))
    if eta_snapshot_enabled:
        if not bool(config.use_aircomp):
            raise ValueError("Eta-response snapshot requires Config.use_aircomp=True")
        if direct_only_mode:
            raise ValueError("Eta-response snapshot requires a reflection-enabled link (direct-only mode unsupported)")
        if not use_uplink_target:
            raise ValueError("Eta-response snapshot requires gru_csi_target_mode in {'uplink_linear', 'uplink_direct'}")
    if gru_group_switch_sensitivity_snapshot_enabled and not enable_gru_semantic_grouping:
        raise ValueError(
            "GRU group switch sensitivity snapshots require enable_gru_semantic_grouping=True"
        )
    if gru_pl_snapshot_enabled:
        if not enable_gru_pl_factorization:
            raise ValueError("GRU PL debug snapshot requires enable_gru_pl_factorization=True")
        if direct_only_mode:
            raise ValueError("GRU PL debug snapshot requires a reflection-enabled link (direct-only mode unsupported)")

    # Per-user sliding window buffer for GRU
    W = int(config.window_length)
    pad_val = float(config.window_pad_value)
    gru_context_mode = str(config.gru_context_mode).lower()
    if gru_context_mode not in {"persistent_hidden", "time_window"}:
        raise ValueError("Config.gru_context_mode must be 'persistent_hidden' or 'time_window'")
    use_persistent_hidden_state = (gru_context_mode == "persistent_hidden")
    use_time_window = (gru_context_mode == "time_window")
    if enable_gru_semantic_grouping:
        if not enable_gru_pl_factorization:
            raise ValueError("GRU semantic grouping requires enable_gru_pl_factorization=True")
        if not use_persistent_hidden_state:
            raise ValueError("GRU semantic grouping requires gru_context_mode='persistent_hidden'")
        if direct_only_mode:
            raise ValueError("GRU semantic grouping requires a reflection-enabled link (direct-only mode unsupported)")
        if gru_grouping_cfg is None:
            raise RuntimeError("Internal error: missing GRU grouping config")
        if gru_restart_training_after_switch:
            logger.info(
                "GRU grouped-restart debug enabled: "
                "freeze membership after switch and restart GRU model/head/hidden state, "
                "while keeping group physical variables."
            )
        if gru_head_randomize_on_switch and gru_head_reset_to_group_mean_on_switch:
            logger.info(
                "GRU grouped-head diagnostics: both randomize_on_switch and "
                "reset_to_group_mean_on_switch are enabled; randomization will take precedence."
            )
        logger.info(
            "GRU grouped-head diagnostics: "
            f"reset_to_group_mean_on_switch={gru_head_reset_to_group_mean_on_switch}, "
            f"randomize_on_switch={gru_head_randomize_on_switch}, "
            f"disable_head_persistence_after_switch={gru_head_disable_persistence_after_switch}, "
            f"log_group_head_dispersion={gru_log_group_head_dispersion}"
        )
        logger.info(
            "GRU grouped-hidden diagnostics: "
            f"reset_on_group_switch={gru_reset_hidden_on_group_switch}, "
            f"reset_on_group_change_each_round={gru_reset_hidden_on_group_change_each_round}, "
            f"disable_persistent_hidden_after_switch={gru_disable_persistent_hidden_after_switch}"
        )
        logger.info(
            "GRU grouped-standardization diagnostics: "
            f"groupwise_standardization={gru_groupwise_standardization}"
        )
    elif gru_restart_training_after_switch:
        raise ValueError("gru_restart_training_after_switch requires enable_gru_semantic_grouping=True")
        gru_grouping_cfg.validate(int(config.num_users))
        logger.info(
            "GRU semantic grouping enabled: "
            f"switch@r>={gru_group_switch_min_round}, patience={gru_group_switch_patience}, "
            f"ema_lambda={gru_group_switch_ema_lambda:.2f}, "
            f"tau_B={gru_group_switch_tau_b:.4f}, tau_D={gru_group_switch_tau_d:.4f}, "
            f"k_min={gru_grouping_cfg.k_min}, sca_iters={gru_grouping_cfg.sca_max_iters}, "
            f"freeze_after_switch={gru_group_freeze_after_switch}"
        )
    beam_ris_optimizer = str(getattr(config, "beam_ris_optimizer", "oa")).strip().lower()
    if beam_ris_optimizer not in {"oa", "sca", "dc"}:
        raise ValueError("Config.beam_ris_optimizer must be 'oa', 'sca', or 'dc'")
    oa_iters = int(getattr(config, "oa_iters", 5))
    sca_iters = int(getattr(config, "sca_iters", 100))
    sca_threshold = float(getattr(config, "sca_threshold", 1e-2))
    sca_tau = float(getattr(config, "sca_tau", 1.0))
    dc_outer_iters = int(getattr(config, "dc_outer_iters", 20))
    dc_inner_iters = int(getattr(config, "dc_inner_iters", 50))
    dc_tol = float(getattr(config, "dc_tol", 1e-3))
    dc_inner_tol = float(getattr(config, "dc_inner_tol", 1e-8))
    local_lr_gru = float(getattr(config, "local_lr_gru", config.local_lr))
    local_lr_arch = float(getattr(config, "local_lr_arch", config.local_lr))
    local_lr_base = float(getattr(config, "local_lr_base", config.local_lr))
    local_optimizer_gru = str(getattr(config, "local_optimizer_gru", "adam")).strip().lower()
    local_optimizer_arch = str(getattr(config, "local_optimizer_arch", "adam")).strip().lower()
    local_optimizer_base = str(getattr(config, "local_optimizer_base", "adam")).strip().lower()
    local_momentum_gru = float(getattr(config, "local_momentum_gru", 0.0))
    local_momentum_arch = float(getattr(config, "local_momentum_arch", 0.0))
    local_momentum_base = float(getattr(config, "local_momentum_base", 0.0))
    if oa_iters <= 0:
        raise ValueError("Config.oa_iters must be positive")
    if sca_iters <= 0:
        raise ValueError("Config.sca_iters must be positive")
    if sca_threshold < 0:
        raise ValueError("Config.sca_threshold must be nonnegative")
    if sca_tau <= 0:
        raise ValueError("Config.sca_tau must be positive")
    if dc_outer_iters <= 0:
        raise ValueError("Config.dc_outer_iters must be positive")
    if dc_inner_iters <= 0:
        raise ValueError("Config.dc_inner_iters must be positive")
    if dc_tol < 0:
        raise ValueError("Config.dc_tol must be nonnegative")
    if dc_inner_tol <= 0:
        raise ValueError("Config.dc_inner_tol must be positive")
    if local_lr_gru <= 0 or local_lr_arch <= 0 or local_lr_base <= 0:
        raise ValueError("All local learning rates must be positive")
    if local_optimizer_gru not in {"adam", "sgd"} or local_optimizer_arch not in {"adam", "sgd"} or local_optimizer_base not in {"adam", "sgd"}:
        raise ValueError("All local optimizers must be 'adam' or 'sgd'")
    if local_momentum_gru < 0 or local_momentum_arch < 0 or local_momentum_base < 0:
        raise ValueError("All local momentum values must be nonnegative")

    reset_hidden_on_round1 = bool(config.reset_hidden_on_round1)
    reset_hidden_on_large_backbone_update = bool(config.reset_hidden_on_large_backbone_update)
    hidden_reset_update_norm_threshold = float(config.hidden_reset_update_norm_threshold)

    obs_buffers = [deque(maxlen=W) for _ in range(config.num_users)]
    logger.info(
        f"GRU context mode={gru_context_mode}, W={W}"
    )
    logger.info(f"Beam/RIS optimizer={beam_ris_optimizer.upper()}")
    logger.info(f"OA params: iters={oa_iters}")
    if beam_ris_optimizer == "sca":
        logger.info(
            f"SCA params: iters={sca_iters}, threshold={sca_threshold:.2e}, tau={sca_tau:.3f}"
        )
    elif beam_ris_optimizer == "dc":
        logger.info(
            f"DC params: outer_iters={dc_outer_iters}, inner_iters={dc_inner_iters}, "
            f"tol={dc_tol:.2e}, inner_tol={dc_inner_tol:.2e}"
        )
    logger.info(
        f"Local learning rates: GRU={local_lr_gru:.3e}, "
        f"CNN-arch={local_lr_arch:.3e}, CNN-base={local_lr_base:.3e}"
    )
    logger.info(
        "Local optimizers: "
        f"GRU={local_optimizer_gru}(momentum={local_momentum_gru:.3f}), "
        f"CNN-arch={local_optimizer_arch}(momentum={local_momentum_arch:.3f}), "
        f"CNN-base={local_optimizer_base}(momentum={local_momentum_base:.3f})"
    )
    logger.info(f"Meta algorithm={meta_algorithm}")
    logger.info(f"GRU CSI output selection mode={gru_csi_target_mode} (dual-head supervised on t and tau)")
    logger.info(
        f"GRU pathloss factorization={'enabled' if enable_gru_pl_factorization else 'disabled'}"
    )
    if use_uplink_target:
        logger.info(f"GRU uplink instant ratio rho=tau/delta={uplink_tau_ratio:.3f}")
    if bool(getattr(config, "enable_mobility_aware_loss_weighting", True)):
        logger.info(
            "GRU mobility-aware tau-loss weighting enabled: "
            f"gain={float(getattr(config, 'mobility_tau_loss_weight_gain', 4.0)):.3f}, "
            f"clip=[{float(getattr(config, 'mobility_tau_loss_weight_min', 1.0)):.3f}, "
            f"{float(getattr(config, 'mobility_tau_loss_weight_max', 4.0)):.3f}]"
        )
    if mobility_snapshot_enabled:
        logger.info(f"Mobility debug snapshots enabled: {mobility_snapshot_run_dir}")
    if delta_motion_snapshot_enabled:
        logger.info(f"Delta-motion debug snapshots enabled: {delta_motion_snapshot_run_dir}")
    if gru_pl_snapshot_enabled:
        logger.info(f"GRU PL debug snapshots enabled: {gru_pl_snapshot_run_dir}")
    if gru_group_switch_sensitivity_snapshot_enabled:
        logger.info(
            "GRU group switch sensitivity snapshots enabled: "
            f"{gru_group_switch_sensitivity_snapshot_run_dir}, "
            f"tau_B_grid={np.round(gru_group_switch_sensitivity_tau_b_grid, 6).tolist()}, "
            f"tau_D_grid={np.round(gru_group_switch_sensitivity_tau_d_grid, 6).tolist()}"
        )
    logger.info("CNN-arch/CNN-base/LMMSE CSI target mode=t (hold-last for uplink evaluation)")
    if direct_only_mode:
        logger.info(
            "Direct-only mode: bypass h_RU learning for GRU/CNN branches; "
            "retain the direct link as a traditional non-RIS baseline and optimize f only."
        )
    global_model = _create_fresh_gru_model(
        observation_dim=obs_dim,
        output_dim=output_dim,
        enable_pl_factorization=enable_gru_pl_factorization,
        log_pl_min=gru_log_pl_min,
        log_pl_max=gru_log_pl_max,
    )
    user_hidden_states = [None for _ in range(config.num_users)]
    reset_hidden_next_round = False
    global_head_state = {k: v.clone() for k, v in global_model.state_dict().items() if k.startswith("head")}
    user_head_states = [copy.deepcopy(global_head_state) for _ in range(config.num_users)]
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

    enable_cnn_arch_ablation = bool(config.enable_cnn_arch_ablation)
    global_model_arch = None
    user_head_states_arch = None
    aggregator_arch = None
    f_beam_arch = None
    theta_ota_arch = None
    obs_buffers_arch = None
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
        if meta_algorithm == "reptile":
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

    enable_lmmse_baseline = bool(getattr(config, "enable_lmmse_baseline", True))
    lmmse_prior_var = float(getattr(config, "lmmse_prior_var", 1.0))
    f_beam_lmmse = None
    theta_ota_lmmse = None
    if enable_lmmse_baseline:
        if direct_only_mode:
            enable_lmmse_baseline = False
            logger.info("LMMSE baseline bypassed in direct-only mode (no RIS-user channel to estimate).")
        else:
            f_beam_lmmse = f_beam.copy()
            theta_ota_lmmse = theta_ota.copy()
            logger.info(
                "LMMSE baseline enabled=True "
                "(closed-form estimate on current pilots, no training, hold-last uplink evaluation, "
                "equal-variance OTA proxy)."
            )

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
    if meta_algorithm == "reptile":
        aggregator = ReptileAggregator(step_size=config.reptile_step_size, use_aircomp=False, aircomp_simulator=None)
    else:
        aggregator = MetaUpdater(meta_algorithm="FedAvg", step_size=1.0, use_aircomp=False, aircomp_simulator=None)
    # Trainer for local updates
    trainer_gru = GRUTrainer(
        learning_rate=local_lr_gru,
        epochs=config.local_epochs,
        batch_size=config.batch_size,
        optimizer_name=local_optimizer_gru,
        momentum=local_momentum_gru,
    )
    trainer_arch = GRUTrainer(
        learning_rate=local_lr_arch,
        epochs=config.local_epochs,
        batch_size=config.batch_size,
        optimizer_name=local_optimizer_arch,
        momentum=local_momentum_arch,
    )
    trainer_base = GRUTrainer(
        learning_rate=local_lr_base,
        epochs=config.local_epochs,
        batch_size=config.batch_size,
        optimizer_name=local_optimizer_base,
        momentum=local_momentum_base,
    )
    gru_group_mode = "single"
    gru_group_patience_counter = 0
    gru_group_switch_round = None
    gru_group_assignment = np.zeros((config.num_users,), dtype=np.int64)
    gru_group_prev_x_hard = None
    gru_group_prev_x_soft = None
    gru_group_prev_mu = None
    gru_group_beta_ema = None
    gru_group_d_ema = None
    gru_group_scalar_B_ema = None
    gru_group_scalar_D_ema = None
    gru_group_switch_sensitivity_patience_grid = np.zeros(
        (
            int(gru_group_switch_sensitivity_tau_b_grid.size),
            int(gru_group_switch_sensitivity_tau_d_grid.size),
        ),
        dtype=np.int16,
    )
    global_model_gru_groups = None
    f_beam_gru_groups = None
    theta_ota_gru_groups = None
    gru_user_last_group_assignment = np.full((config.num_users,), -1, dtype=np.int64)
    reset_hidden_next_round_reason = None
    # Simulation rounds
    logger.info(f"Starting training for {config.num_rounds} rounds...")

    for round_idx in range(1, config.num_rounds + 1):
        logger.info(f"\033[32mRound {round_idx}\033[0m - Generating pilot observations.")
        gru_group_freeze_effective = gru_group_freeze_after_switch or gru_restart_training_after_switch
        if gru_group_mode == "grouped":
            current_gru_group_assignment = np.asarray(gru_group_assignment, dtype=np.int64).copy()
            current_global_model_gru_groups = global_model_gru_groups
            current_f_beam_gru_groups = f_beam_gru_groups
            current_theta_ota_gru_groups = theta_ota_gru_groups
            group_sizes = [int(np.sum(current_gru_group_assignment == g)) for g in range(2)]
            logger.info(
                f"Round {round_idx} GRU grouping mode=grouped, "
                f"group_sizes=({group_sizes[0]},{group_sizes[1]}), switch_round={gru_group_switch_round}, "
                f"frozen={gru_group_freeze_effective}, "
                f"head_persist={'off' if gru_head_disable_persistence_after_switch else 'on'}, "
                f"hidden_persist={'off' if gru_disable_persistent_hidden_after_switch else 'on'}"
            )
            if gru_log_group_head_dispersion:
                head_dispersion_stats = _group_head_dispersion_stats(
                    user_head_states,
                    current_gru_group_assignment,
                    eps=gru_group_eps,
                )
                logger.info(
                    "GRU grouped-head dispersion -> "
                    + ", ".join(
                        [
                            f"G{stat['group_idx'] + 1}(n={stat['size']},"
                            f"mean_rel={stat['mean_rel']:.4e},max_rel={stat['max_rel']:.4e})"
                            for stat in head_dispersion_stats
                        ]
                    )
                )
        else:
            current_gru_group_assignment = np.zeros((config.num_users,), dtype=np.int64)
            current_global_model_gru_groups = None
            current_f_beam_gru_groups = None
            current_theta_ota_gru_groups = None
            logger.info(f"Round {round_idx} GRU grouping mode=single")
        if use_persistent_hidden_state:
            if (round_idx == 1 and reset_hidden_on_round1) or reset_hidden_next_round:
                user_hidden_states = [None for _ in range(config.num_users)]
                reason = (
                    "round1"
                    if (round_idx == 1 and reset_hidden_on_round1)
                    else (reset_hidden_next_round_reason or "large_backbone_update")
                )
                logger.info(f"Reset persistent hidden states at round {round_idx} (reason={reason}).")
                reset_hidden_next_round = False
                reset_hidden_next_round_reason = None
            if gru_group_mode == "grouped":
                if gru_disable_persistent_hidden_after_switch:
                    user_hidden_states = [None for _ in range(config.num_users)]
                    logger.info("GRU grouped-hidden debug: persistent hidden disabled after switch; using None state for all users.")
                elif gru_reset_hidden_on_group_change_each_round:
                    changed_mask = (
                        (gru_user_last_group_assignment >= 0)
                        & (gru_user_last_group_assignment != current_gru_group_assignment)
                    )
                    changed_users = np.flatnonzero(changed_mask).astype(np.int64)
                    for user_idx in changed_users.tolist():
                        user_hidden_states[int(user_idx)] = None
                    if changed_users.size > 0:
                        logger.info(
                            "GRU grouped-hidden debug: reset hidden for group-changed users -> "
                            f"{(changed_users + 1).tolist()}"
                        )
                gru_user_last_group_assignment = current_gru_group_assignment.copy()
            else:
                gru_user_last_group_assignment.fill(-1)
        h_RUs_uplink = None
        h_BUs_uplink = None
        alpha_used_uplink = None
        step_result = ru_evolver.step_split(
            h_RUs,
            round_idx,
            uplink_tau_ratio,
            h_BUs=h_BUs,
        )
        h_RUs_uplink = step_result.h_ru_tau
        h_RUs_next = step_result.h_ru_next
        h_BUs_uplink = step_result.h_bu_tau
        h_BUs_next = step_result.h_bu_next
        alpha_used_next = step_result.alpha_delta.astype(np.float32, copy=False)
        alpha_used_uplink = step_result.alpha_tau.astype(np.float32, copy=False)

        # Generate pilot observation and ground truth channel for each user at this round.
        # Physical time advances once per round; n_k only controls how many same-round
        # training observations are generated from the current channel block.
        local_data = [None for _ in range(config.num_users)]
        local_data_groups = [[] for _ in range(config.num_users)]
        local_data_arch = [None for _ in range(config.num_users)] if enable_cnn_arch_ablation else None
        local_data_arch_groups = [[] for _ in range(config.num_users)] if enable_cnn_arch_ablation else None
        local_data_baseline = [None for _ in range(config.num_users)] if enable_cnn_baseline else None
        local_data_baseline_groups = [[] for _ in range(config.num_users)] if enable_cnn_baseline else None
        h_RUs_est_lmmse = np.zeros((config.num_users, config.num_ris_elements), dtype=np.complex64) \
            if enable_lmmse_baseline else None
        user_pos_t_for_log = np.zeros((config.num_users, 2), dtype=np.float64)
        user_pos_uplink_for_log = np.zeros((config.num_users, 2), dtype=np.float64)
        h_ru_signal_norms = []
        y_pilot_gru_norms = []
        target_tau_norms = []
        tau_loss_weight_round = np.ones((config.num_users,), dtype=np.float32)
        moving_user_mask = np.asarray(ru_evolver.moving_user_mask, dtype=bool).reshape(-1)
        true_delta_norm_round = np.zeros((config.num_users,), dtype=np.float32)
        true_delta_rel_round = np.zeros((config.num_users,), dtype=np.float32)
        gru_delta_proxy_norm_round = np.full((config.num_users,), np.nan, dtype=np.float32)
        gru_delta_proxy_rel_round = np.full((config.num_users,), np.nan, dtype=np.float32)
        gru_pl_true_round = np.full((config.num_users,), np.nan, dtype=np.float32)
        gru_pl_pred_round = np.full((config.num_users,), np.nan, dtype=np.float32)
        positions_t_round = ru_evolver.positions_at(float(round_idx - 1))
        positions_uplink_round = ru_evolver.positions_at(float(round_idx - 1) + uplink_tau_ratio)
        ris_pl_t_round = ru_evolver.ris_pathloss(positions_t_round).astype(np.float32, copy=False)
        ris_pl_uplink_round = ru_evolver.ris_pathloss(positions_uplink_round).astype(np.float32, copy=False)
        for k in range(config.num_users):
            gru_group_idx_k = int(current_gru_group_assignment[k])
            f_beam_gru_k = (
                np.asarray(current_f_beam_gru_groups[gru_group_idx_k], dtype=np.complex64)
                if current_f_beam_gru_groups is not None
                else f_beam
            )
            sample_count_k = max(1, int(round(float(user_n_k_target[k]))))
            h_BU_k = h_BUs[k] if h_BUs is not None else None
            user_pos_t_for_log[k] = positions_t_round[k]
            user_pos_uplink_for_log[k] = positions_uplink_round[k]
            tau_loss_weight_k = _compute_mobility_tau_loss_weight(alpha_used_uplink[k], moving_user_mask[k], config)
            tau_loss_weight_round[k] = tau_loss_weight_k
            gru_sample_meta = {"tau_loss_weight": tau_loss_weight_k}
            pl_sel_k = _select_gru_pl_value(ris_pl_t_round[k], ris_pl_uplink_round[k], gru_csi_target_mode)
            if enable_gru_pl_factorization:
                gru_pl_true_round[k] = float(pl_sel_k)
            if enable_gru_pl_factorization:
                gru_sample_meta["pl_sel"] = float(pl_sel_k)
                gru_sample_meta["log_pl_sel"] = float(np.log(max(pl_sel_k, gru_pl_eps)))
                gru_sample_meta["pl_loss_scale"] = float(gru_pl_loss_weight)
                y_gru = _build_gru_dual_target_pl_factorized(h_RUs[k], h_RUs_uplink[k], pl_sel_k, gru_pl_eps)
            else:
                y_gru = _build_gru_dual_target(h_RUs[k], h_RUs_uplink[k])
            y_plain = _complex_to_ri(h_RUs[k])
            tau_norm_k = float(np.linalg.norm(h_RUs_uplink[k]))
            true_delta_norm_k, true_delta_rel_k = _complex_delta_metrics(h_RUs[k], h_RUs_uplink[k])
            true_delta_norm_round[k] = true_delta_norm_k
            true_delta_rel_round[k] = true_delta_rel_k

            # Canonical current-round observation used for inference/state update.
            Y_pilot_canon, _, _ = pilot_gen.simulate_pilot_observation(
                H_BR, h_RUs[k], f_beam_gru_k, theta_pilot,
                noise_std=pilot_noise_std,
                h_BU=h_BU_k,
                link_switch=link_switch,
            )
            obs_step_canon = np.stack([np.real(Y_pilot_canon), np.imag(Y_pilot_canon)], axis=0).astype(np.float32)
            if use_time_window and W > 1:
                obs_buffers[k].append(obs_step_canon)
                X_seq_canon = _build_window_sample(obs_buffers[k], obs_step_canon, W, obs_dim, pad_val)
            else:
                X_seq_canon = obs_step_canon[None, :, :]
            local_data[k] = (
                X_seq_canon.astype(np.float32, copy=False),
                y_gru.astype(np.float32, copy=False),
                gru_sample_meta,
            )

            for _ in range(sample_count_k):
                Y_pilot, _, _ = pilot_gen.simulate_pilot_observation(
                    H_BR, h_RUs[k], f_beam_gru_k, theta_pilot,
                    noise_std=pilot_noise_std,
                    h_BU=h_BU_k,
                    link_switch=link_switch,
                )
                obs_step = np.stack([np.real(Y_pilot), np.imag(Y_pilot)], axis=0).astype(np.float32)
                if use_time_window and W > 1:
                    X_seq = _build_window_sample(obs_buffers[k], obs_step, W, obs_dim, pad_val)
                else:
                    X_seq = obs_step[None, :, :]
                local_data_groups[k].append(
                    (
                        X_seq.astype(np.float32, copy=False),
                        y_gru.astype(np.float32, copy=False),
                        gru_sample_meta,
                    )
                )
                h_ru_signal_norms.append(float(np.linalg.norm(h_RUs[k])))
                y_pilot_gru_norms.append(float(np.linalg.norm(Y_pilot)))
                target_tau_norms.append(tau_norm_k)

            if use_time_window and W > 1 and round_idx <= 3 and k == 0:
                logger.info(f"Example X_seq shape for user1: {local_data_groups[k][0][0].shape}")  # (W,2,P)

            if enable_cnn_arch_ablation:
                Y_pilot_arch_canon, _, _ = pilot_gen.simulate_pilot_observation(
                    H_BR, h_RUs[k], f_beam_arch, theta_pilot,
                    noise_std=pilot_noise_std,
                    h_BU=h_BU_k,
                    link_switch=link_switch,
                )
                obs_step_arch_canon = np.stack(
                    [np.real(Y_pilot_arch_canon), np.imag(Y_pilot_arch_canon)],
                    axis=0,
                ).astype(np.float32)
                if use_time_window and W > 1:
                    obs_buffers_arch[k].append(obs_step_arch_canon)
                    X_seq_arch_canon = _build_window_sample(
                        obs_buffers_arch[k], obs_step_arch_canon, W, obs_dim, pad_val
                    )
                else:
                    X_seq_arch_canon = obs_step_arch_canon[None, :, :]
                local_data_arch[k] = (
                    X_seq_arch_canon.astype(np.float32, copy=False),
                    y_plain.astype(np.float32, copy=False),
                )
                for _ in range(sample_count_k):
                    Y_pilot_arch, _, _ = pilot_gen.simulate_pilot_observation(
                        H_BR, h_RUs[k], f_beam_arch, theta_pilot,
                        noise_std=pilot_noise_std,
                        h_BU=h_BU_k,
                        link_switch=link_switch,
                    )
                    obs_step_arch = np.stack([np.real(Y_pilot_arch), np.imag(Y_pilot_arch)], axis=0).astype(np.float32)
                    if use_time_window and W > 1:
                        X_seq_arch = _build_window_sample(obs_buffers_arch[k], obs_step_arch, W, obs_dim, pad_val)
                    else:
                        X_seq_arch = obs_step_arch[None, :, :]
                    local_data_arch_groups[k].append(
                        (X_seq_arch.astype(np.float32, copy=False), y_plain.astype(np.float32, copy=False))
                    )

            if enable_cnn_baseline:
                Y_pilot_baseline_canon, _, _ = pilot_gen.simulate_pilot_observation(
                    H_BR, h_RUs[k], f_beam_baseline, theta_pilot,
                    noise_std=pilot_noise_std,
                    h_BU=h_BU_k,
                    link_switch=link_switch,
                )
                obs_step_baseline_canon = np.stack(
                    [np.real(Y_pilot_baseline_canon), np.imag(Y_pilot_baseline_canon)],
                    axis=0,
                ).astype(np.float32)
                local_data_baseline[k] = (
                    obs_step_baseline_canon[None, :, :].astype(np.float32, copy=False),
                    y_plain.astype(np.float32, copy=False),
                )
                for _ in range(sample_count_k):
                    Y_pilot_baseline, _, _ = pilot_gen.simulate_pilot_observation(
                        H_BR, h_RUs[k], f_beam_baseline, theta_pilot,
                        noise_std=pilot_noise_std,
                        h_BU=h_BU_k,
                        link_switch=link_switch,
                    )
                    obs_step_baseline = np.stack(
                        [np.real(Y_pilot_baseline), np.imag(Y_pilot_baseline)],
                        axis=0,
                    ).astype(np.float32)
                    local_data_baseline_groups[k].append(
                        (obs_step_baseline[None, :, :].astype(np.float32, copy=False), y_plain.astype(np.float32, copy=False))
                    )

            if enable_lmmse_baseline:
                Y_pilot_lmmse_canon, _, _ = pilot_gen.simulate_pilot_observation(
                    H_BR, h_RUs[k], f_beam_lmmse, theta_pilot,
                    noise_std=pilot_noise_std,
                    h_BU=h_BU_k,
                    link_switch=link_switch,
                )
                h_RUs_est_lmmse[k] = estimate_h_ru_lmmse(
                    Y_pilot_lmmse_canon,
                    H_BR,
                    f_beam_lmmse,
                    theta_pilot,
                    pilot_noise_std,
                    h_BU=h_BU_k,
                    link_switch=link_switch,
                    prior_var=lmmse_prior_var,
                )

        gru_input_stats = None
        gru_target_stats = None
        gru_target_stats_by_user = [None for _ in range(config.num_users)]
        if gru_group_mode == "grouped" and gru_groupwise_standardization:
            for group_idx in range(2):
                user_ids = np.flatnonzero(current_gru_group_assignment == group_idx).astype(np.int64)
                if user_ids.size == 0:
                    raise RuntimeError(f"GRU group {group_idx + 1} is empty when computing groupwise standardization")
                group_samples = _flatten_sample_groups([local_data_groups[int(user_idx)] for user_idx in user_ids.tolist()])
                input_stats_g = _compute_standardization_stats([sample[0] for sample in group_samples], feature_ndim=2)
                target_stats_g = _compute_standardization_stats([sample[1] for sample in group_samples], feature_ndim=1)
                for user_idx in user_ids.tolist():
                    local_data_groups[int(user_idx)] = [
                        _standardize_sample(sample, input_stats_g, target_stats_g)
                        for sample in local_data_groups[int(user_idx)]
                    ]
                    local_data[int(user_idx)] = _standardize_sample(
                        local_data[int(user_idx)],
                        input_stats_g,
                        target_stats_g,
                    )
                    gru_target_stats_by_user[int(user_idx)] = target_stats_g
            logger.info("GRU grouped standardization: using per-group input/target stats.")
        else:
            gru_flat_samples = _flatten_sample_groups(local_data_groups)
            gru_input_stats = _compute_standardization_stats([sample[0] for sample in gru_flat_samples], feature_ndim=2)
            gru_target_stats = _compute_standardization_stats([sample[1] for sample in gru_flat_samples], feature_ndim=1)
            local_data_groups = [
                [_standardize_sample(sample, gru_input_stats, gru_target_stats) for sample in seq]
                for seq in local_data_groups
            ]
            local_data = [_standardize_sample(sample, gru_input_stats, gru_target_stats) for sample in local_data]
            gru_target_stats_by_user = [gru_target_stats for _ in range(config.num_users)]

        arch_input_stats = None
        baseline_input_stats = None
        plain_target_stats = None
        plain_target_arrays = []
        if enable_cnn_arch_ablation:
            arch_flat_samples = _flatten_sample_groups(local_data_arch_groups)
            arch_input_stats = _compute_standardization_stats(
                [sample[0] for sample in arch_flat_samples],
                feature_ndim=2,
            )
            plain_target_arrays.extend([sample[1] for sample in arch_flat_samples])
        if enable_cnn_baseline:
            baseline_flat_samples = _flatten_sample_groups(local_data_baseline_groups)
            baseline_input_stats = _compute_standardization_stats(
                [sample[0] for sample in baseline_flat_samples],
                feature_ndim=2,
            )
            plain_target_arrays.extend([sample[1] for sample in baseline_flat_samples])
        if plain_target_arrays:
            plain_target_stats = _compute_standardization_stats(plain_target_arrays, feature_ndim=1)
        if enable_cnn_arch_ablation:
            local_data_arch_groups = [
                [_standardize_sample(sample, arch_input_stats, plain_target_stats) for sample in seq]
                for seq in local_data_arch_groups
            ]
            local_data_arch = [
                _standardize_sample(sample, arch_input_stats, plain_target_stats)
                for sample in local_data_arch
            ]
        if enable_cnn_baseline:
            local_data_baseline_groups = [
                [_standardize_sample(sample, baseline_input_stats, plain_target_stats) for sample in seq]
                for seq in local_data_baseline_groups
            ]
            local_data_baseline = [
                _standardize_sample(sample, baseline_input_stats, plain_target_stats)
                for sample in local_data_baseline
            ]

        # Local training on each user's data
        local_models = []
        losses = []
        gru_losses_t = []
        gru_losses_tau = []
        gru_local_update_norms = []
        local_models_arch = [] if enable_cnn_arch_ablation else None
        losses_arch = [] if enable_cnn_arch_ablation else None
        arch_local_update_norms = [] if enable_cnn_arch_ablation else None
        local_models_baseline = [] if enable_cnn_baseline else None
        losses_baseline = [] if enable_cnn_baseline else None
        baseline_local_update_norms = [] if enable_cnn_baseline else None
        h_RUs_est = np.zeros((config.num_users, config.num_ris_elements), dtype=np.complex64)
        h_RUs_est_arch = np.zeros((config.num_users, config.num_ris_elements), dtype=np.complex64) \
            if enable_cnn_arch_ablation else None
        h_RUs_est_baseline = np.zeros((config.num_users, config.num_ris_elements), dtype=np.complex64) \
            if enable_cnn_baseline else None
        # n_k for this round: number of samples actually used by each user in local training.
        user_sample_count_round = np.zeros((config.num_users,), dtype=np.float32)
        # Debug-only cache: per-user hidden-state delta (current - history), no training-path impact.
        gru_state_delta_vectors = [None for _ in range(config.num_users)] \
            if (debug_gru_state_plot_dir is not None and use_persistent_hidden_state) else None
        for k in range(config.num_users):
            loss_arch = None
            loss_baseline = None
            samples_used_k = 1
            gru_loss_t = None
            gru_loss_tau = None
            pred_dual_for_log = None
            target_dual_for_log = None

            # Model for user k from current global shared weights.
            local_model = CSICNNGRU(
                observation_dim=obs_dim,
                output_dim=output_dim,
                enable_pl_factorization=enable_gru_pl_factorization,
                log_pl_min=gru_log_pl_min,
                log_pl_max=gru_log_pl_max,
            )
            gru_group_idx_k = int(current_gru_group_assignment[k])
            source_global_model = (
                current_global_model_gru_groups[gru_group_idx_k]
                if current_global_model_gru_groups is not None
                else global_model
            )
            state = source_global_model.state_dict()
            for hk, hv in user_head_states[k].items():
                state[hk] = hv.clone()
            local_model.load_state_dict(state)

            # Train on user k's data
            if direct_only_mode:
                loss = 0.0
                samples_used_k = max(1, int(round(float(user_n_k_target[k]))))
            elif use_persistent_hidden_state:
                data_k = local_data_groups[k]
                sample_k = local_data[k]
                gru_target_stats_k = gru_target_stats_by_user[k]
                disable_hidden_persistence_now = (
                    gru_group_mode == "grouped" and gru_disable_persistent_hidden_after_switch
                )
                hidden_prev = None if disable_hidden_persistence_now else user_hidden_states[k]
                local_model, loss = trainer_gru.train_stateful_independent(
                    local_model,
                    data_k,
                    hidden_state=hidden_prev,
                )
                gru_stats = dict(getattr(trainer_gru, "last_loss_stats", {}))
                gru_loss_t = gru_stats.get("loss_t")
                gru_loss_tau = gru_stats.get("loss_tau")
                samples_used_k = max(1, len(data_k))
                pred_aux = None
                if enable_gru_pl_factorization:
                    pred_last, hidden_next, pred_aux = trainer_gru.infer_stateful_sample(
                        local_model,
                        sample_k,
                        hidden_state=hidden_prev,
                        return_aux=True,
                    )
                else:
                    pred_last, hidden_next = trainer_gru.infer_stateful_sample(
                        local_model,
                        sample_k,
                        hidden_state=hidden_prev,
                    )
                if not disable_hidden_persistence_now:
                    user_hidden_states[k] = hidden_next
                if gru_state_delta_vectors is not None and hidden_next is not None:
                    vec_next = hidden_next.detach().cpu().reshape(-1).float()
                    if hidden_prev is None:
                        vec_prev = torch.zeros_like(vec_next)
                    else:
                        vec_prev = hidden_prev.detach().cpu().reshape(-1).float()
                    gru_state_delta_vectors[k] = (vec_next - vec_prev).numpy()
                if (round_idx <= 3) and (k == 0) and (hidden_next is not None):
                    logger.info(f"User1 persistent hidden norm: {torch.norm(hidden_next).item():.4e}")
                target_dual_norm = _invert_standardization(np.asarray(sample_k[1], dtype=np.float32), gru_target_stats_k)
                pred_dual_norm = _invert_standardization(
                    pred_last.numpy().astype(np.float32, copy=False),
                    gru_target_stats_k,
                )
                if enable_gru_pl_factorization:
                    pl_target_sel = float(sample_k[2]["pl_sel"])
                    pl_pred_sel = float(np.asarray(pred_aux["pl_hat"], dtype=np.float32).reshape(-1)[0])
                    gru_pl_pred_round[k] = float(pl_pred_sel)
                    target_dual_for_log = _reconstruct_gru_dual_ri_from_pl(
                        target_dual_norm,
                        config.num_ris_elements,
                        pl_target_sel,
                        gru_pl_eps,
                    )
                    pred_dual_for_log = _reconstruct_gru_dual_ri_from_pl(
                        pred_dual_norm,
                        config.num_ris_elements,
                        pl_pred_sel,
                        gru_pl_eps,
                    )
                else:
                    target_dual_for_log = target_dual_norm
                    pred_dual_for_log = pred_dual_norm
                pred_selected = _select_gru_ri_output(
                    pred_dual_for_log,
                    config.num_ris_elements,
                    gru_csi_target_mode,
                    uplink_tau_ratio,
                )
                pred_t_ri_log, pred_tau_ri_log = _split_gru_dual_ri(pred_dual_for_log, config.num_ris_elements)
                pred_t_c = _ri_to_complex(pred_t_ri_log, config.num_ris_elements)
                pred_tau_c = _ri_to_complex(pred_tau_ri_log, config.num_ris_elements)
                pred_delta_norm_k, pred_delta_rel_k = _complex_delta_metrics(pred_t_c, pred_tau_c)
                gru_delta_proxy_norm_round[k] = pred_delta_norm_k
                gru_delta_proxy_rel_round[k] = pred_delta_rel_k
                h_RUs_est[k] = _ri_to_complex(pred_selected, config.num_ris_elements)
            else:
                data_k = local_data_groups[k]
                samples_used_k = max(1, len(data_k))
                gru_target_stats_k = gru_target_stats_by_user[k]
                local_model, loss = trainer_gru.train(local_model, data_k)
                gru_stats = dict(getattr(trainer_gru, "last_loss_stats", {}))
                gru_loss_t = gru_stats.get("loss_t")
                gru_loss_tau = gru_stats.get("loss_tau")
                target_dual_norm = _invert_standardization(
                    np.asarray(local_data[k][1], dtype=np.float32),
                    gru_target_stats_k,
                )
                pred_aux = None
                if enable_gru_pl_factorization:
                    pred_t_ri, pred_tau_ri, pred_aux = _predict_gru_dual_ri(
                        local_model,
                        local_data[k][0],
                        hidden_state=None,
                        return_aux=True,
                    )
                else:
                    pred_t_ri, pred_tau_ri = _predict_gru_dual_ri(
                        local_model,
                        local_data[k][0],
                        hidden_state=None,
                    )
                pred_dual_norm = _invert_standardization(
                    np.concatenate([pred_t_ri, pred_tau_ri], axis=0).astype(np.float32, copy=False),
                    gru_target_stats_k,
                )
                if enable_gru_pl_factorization:
                    pl_target_sel = float(local_data[k][2]["pl_sel"])
                    pl_pred_sel = float(np.asarray(pred_aux["pl_hat"], dtype=np.float32).reshape(-1)[0])
                    gru_pl_pred_round[k] = float(pl_pred_sel)
                    target_dual_for_log = _reconstruct_gru_dual_ri_from_pl(
                        target_dual_norm,
                        config.num_ris_elements,
                        pl_target_sel,
                        gru_pl_eps,
                    )
                    pred_dual_for_log = _reconstruct_gru_dual_ri_from_pl(
                        pred_dual_norm,
                        config.num_ris_elements,
                        pl_pred_sel,
                        gru_pl_eps,
                    )
                else:
                    target_dual_for_log = target_dual_norm
                    pred_dual_for_log = pred_dual_norm
                pred_selected = _select_gru_ri_output(
                    pred_dual_for_log,
                    config.num_ris_elements,
                    gru_csi_target_mode,
                    uplink_tau_ratio,
                )
                pred_t_ri_log, pred_tau_ri_log = _split_gru_dual_ri(pred_dual_for_log, config.num_ris_elements)
                pred_t_c = _ri_to_complex(pred_t_ri_log, config.num_ris_elements)
                pred_tau_c = _ri_to_complex(pred_tau_ri_log, config.num_ris_elements)
                pred_delta_norm_k, pred_delta_rel_k = _complex_delta_metrics(pred_t_c, pred_tau_c)
                gru_delta_proxy_norm_round[k] = pred_delta_norm_k
                gru_delta_proxy_rel_round[k] = pred_delta_rel_k
                h_RUs_est[k] = _ri_to_complex(pred_selected, config.num_ris_elements)
            user_sample_count_round[k] = float(samples_used_k)
            losses.append(loss if loss is not None else 0.0)
            if gru_loss_t is not None:
                gru_losses_t.append(float(gru_loss_t))
            if gru_loss_tau is not None:
                gru_losses_tau.append(float(gru_loss_tau))
            gru_local_update_norms.append(
                float(torch.norm(model_delta_to_vector_backbone(local_model, source_global_model, prefix="backbone")).item())
            )
            local_models.append(local_model)
            disable_head_persistence_now = (
                gru_group_mode == "grouped" and gru_head_disable_persistence_after_switch
            )
            if not disable_head_persistence_now:
                # cache back the personalized head for user k
                user_head_states[k] = {
                    name: param.detach().clone()
                    for name, param in local_model.state_dict().items()
                    if name.startswith("head")
                }

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
                if direct_only_mode:
                    loss_arch = 0.0
                else:
                    data_k_arch = local_data_arch_groups[k]
                    local_model_arch, loss_arch = trainer_arch.train(local_model_arch, data_k_arch)
                    h_RUs_est_arch[k] = _predict_h_ru_plain(
                        local_model_arch,
                        local_data_arch[k][0],
                        config.num_ris_elements,
                        target_stats=plain_target_stats,
                    )
                losses_arch.append(loss_arch if loss_arch is not None else 0.0)
                arch_local_update_norms.append(
                    float(
                        torch.norm(
                            model_delta_to_vector_backbone(local_model_arch, global_model_arch, prefix="backbone")
                        ).item()
                    )
                )
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
                if direct_only_mode:
                    loss_baseline = 0.0
                else:
                    data_k_baseline = local_data_baseline_groups[k]
                    local_model_baseline, loss_baseline = trainer_base.train(local_model_baseline, data_k_baseline)
                    h_RUs_est_baseline[k] = _predict_h_ru_plain(
                        local_model_baseline,
                        local_data_baseline[k][0],
                        config.num_ris_elements,
                        target_stats=plain_target_stats,
                    )
                losses_baseline.append(loss_baseline if loss_baseline is not None else 0.0)
                baseline_local_update_norms.append(
                    float(torch.norm(model_delta_to_vector(local_model_baseline, global_model_baseline)).item())
                )
                local_models_baseline.append(local_model_baseline)

            loss_parts = []
            if direct_only_mode:
                loss_parts.append("direct-only baseline mode (h_RU training bypassed)")
            elif loss is not None:
                if (gru_loss_t is not None) and (gru_loss_tau is not None):
                    loss_parts.append(
                        "GRU(total/t/tau): "
                        f"\033[34m{loss:.4e}/{gru_loss_t:.4e}/{gru_loss_tau:.4e}\033[0m"
                    )
                else:
                    loss_parts.append(f"GRU: \033[34m{loss:.4e}\033[0m")
            if enable_cnn_arch_ablation and (loss_arch is not None):
                loss_parts.append(f"CNN-arch: \033[35m{loss_arch:.4e}\033[0m")
            if enable_cnn_baseline and (loss_baseline is not None):
                loss_parts.append(f"CNN-base: \033[36m{loss_baseline:.4e}\033[0m")
            pos_text = (
                f"pos_t={_format_xy(user_pos_t_for_log[k])}, "
                f"pos_uplink={_format_xy(user_pos_uplink_for_log[k])}"
            )
            if loss_parts:
                logger.info(f"User {k + 1} local loss -> " + ", ".join(loss_parts) + f", {pos_text}")
            else:
                logger.info(f"User {k + 1} local training done. {pos_text}")
            if (
                debug_gru_dual_target_log_enabled
                and (pred_dual_for_log is not None)
                and (target_dual_for_log is not None)
            ):
                _log_gru_dual_head_debug(
                    logger,
                    round_idx,
                    k,
                    pred_dual_for_log,
                    target_dual_for_log,
                    config.num_ris_elements,
                    gru_loss_t,
                    gru_loss_tau,
                )

        logger.info(
            f"Round {round_idx} GRU signal diagnostics -> "
            f"||h_RU|| {_format_min_mean_max(h_ru_signal_norms)}, "
            f"||Y_pilot|| {_format_min_mean_max(y_pilot_gru_norms)}, "
            f"||target_tau|| {_format_min_mean_max(target_tau_norms)}"
        )
        logger.info(
            f"Round {round_idx} delta-motion diagnostics -> "
            f"true_delta_rel {_format_min_mean_max(true_delta_rel_round)}, "
            f"GRU_delta_proxy_rel {_format_min_mean_max(gru_delta_proxy_rel_round)}"
        )
        logger.info(
            f"Round {round_idx} GRU mobility tau-loss weight -> "
            f"{_format_min_mean_max(tau_loss_weight_round)}"
        )
        update_parts = [f"GRU-backbone {_format_min_mean_max(gru_local_update_norms)}"]
        if enable_cnn_arch_ablation and arch_local_update_norms:
            update_parts.append(f"CNN-arch-backbone {_format_min_mean_max(arch_local_update_norms)}")
        if enable_cnn_baseline and baseline_local_update_norms:
            update_parts.append(f"CNN-base {_format_min_mean_max(baseline_local_update_norms)}")
        logger.info(f"Round {round_idx} local update diagnostics -> " + ", ".join(update_parts))

        if losses:
            gru_round_loss = np.mean(gru_losses_t) if gru_losses_t else np.mean(losses)
            round_loss_parts = [f"Round {round_idx} mean local loss -> GRU: {gru_round_loss:.4f}"]
            if enable_cnn_arch_ablation and losses_arch:
                round_loss_parts.append(f"CNN-arch: {np.mean(losses_arch):.4f}")
            if enable_cnn_baseline and losses_baseline:
                round_loss_parts.append(f"CNN-base: {np.mean(losses_baseline):.4f}")
            logger.info(", ".join(round_loss_parts))
            if gru_losses_t and gru_losses_tau:
                logger.info(
                    f"Round {round_idx} GRU dual-head local loss -> "
                    f"GRU_t: {np.mean(gru_losses_t):.4f}, "
                    f"GRU_tau: {np.mean(gru_losses_tau):.4f}, "
                    f"GRU_total: {np.mean(losses):.4f}"
                )
        uplink_nmse_gru_k = None
        uplink_nmse_gru_oracle_k = None
        uplink_nmse_arch_k = None
        uplink_nmse_baseline_k = None
        uplink_nmse_lmmse_k = None
        uplink_nmse_oracle_k = None
        if (not direct_only_mode) and (h_RUs_uplink is not None):
            uplink_nmse_gru = _complex_nmse(h_RUs_est, h_RUs_uplink)
            uplink_nmse_gru_k = complex_nmse_per_user(h_RUs_est, h_RUs_uplink)
            logger.info(_colorize_branch_line("GRU", f"GRU uplink_true_NMSE={_highlight_metric_value(uplink_nmse_gru, 'GRU')}"))
            _log_grouped_nmse(logger, "GRU", uplink_nmse_gru_k, moving_user_mask, "uplink_true_NMSE")
            if enable_gru_semantic_grouping and gru_group_mode == "grouped":
                _log_partitioned_nmse(
                    logger,
                    "GRU",
                    uplink_nmse_gru_k,
                    current_gru_group_assignment,
                    "uplink_true_NMSE",
                    labels=("low", "high"),
                )
            if alpha_used_uplink is not None:
                h_RUs_oracle_gru = _build_gru_uplink_oracle_prediction(
                    h_t=h_RUs,
                    alpha_tau=alpha_used_uplink,
                    pl_t=ris_pl_t_round,
                    pl_tau=ris_pl_uplink_round,
                    mode=gru_csi_target_mode,
                    uplink_tau_ratio=uplink_tau_ratio,
                    eps=gru_pl_eps,
                )
                uplink_nmse_gru_oracle = _complex_nmse(h_RUs_oracle_gru, h_RUs_uplink)
                uplink_nmse_gru_oracle_k = complex_nmse_per_user(h_RUs_oracle_gru, h_RUs_uplink)
                logger.info(
                    _colorize_branch_line(
                        "GRU",
                        f"GRU oracle_uplink_true_NMSE={_highlight_metric_value(uplink_nmse_gru_oracle, 'GRU')}",
                    )
                )
                if enable_gru_semantic_grouping and gru_group_mode == "grouped":
                    _log_partitioned_nmse(
                        logger,
                        "GRU",
                        uplink_nmse_gru_oracle_k,
                        current_gru_group_assignment,
                        "oracle_uplink_true_NMSE",
                        labels=("low", "high"),
                    )
            if enable_cnn_arch_ablation and (h_RUs_est_arch is not None):
                uplink_nmse_arch = _complex_nmse(h_RUs_est_arch, h_RUs_uplink)
                uplink_nmse_arch_k = complex_nmse_per_user(h_RUs_est_arch, h_RUs_uplink)
                logger.info(
                    _colorize_branch_line(
                        "CNN-arch",
                        f"CNN-arch uplink_true_NMSE={_highlight_metric_value(uplink_nmse_arch, 'CNN-arch')}",
                    )
                )
                _log_grouped_nmse(logger, "CNN-arch", uplink_nmse_arch_k, moving_user_mask, "uplink_true_NMSE")
            if enable_cnn_baseline and (h_RUs_est_baseline is not None):
                uplink_nmse_baseline = _complex_nmse(h_RUs_est_baseline, h_RUs_uplink)
                uplink_nmse_baseline_k = complex_nmse_per_user(h_RUs_est_baseline, h_RUs_uplink)
                logger.info(
                    _colorize_branch_line(
                        "CNN-base",
                        f"CNN-base uplink_true_NMSE={_highlight_metric_value(uplink_nmse_baseline, 'CNN-base')}",
                    )
                )
                _log_grouped_nmse(logger, "CNN-base", uplink_nmse_baseline_k, moving_user_mask, "uplink_true_NMSE")
            if enable_lmmse_baseline and (h_RUs_est_lmmse is not None):
                uplink_nmse_lmmse = _complex_nmse(h_RUs_est_lmmse, h_RUs_uplink)
                uplink_nmse_lmmse_k = complex_nmse_per_user(h_RUs_est_lmmse, h_RUs_uplink)
                logger.info(
                    _colorize_branch_line(
                        "LMMSE",
                        f"LMMSE uplink_true_NMSE={_highlight_metric_value(uplink_nmse_lmmse, 'LMMSE')}",
                    )
                )
                _log_grouped_nmse(logger, "LMMSE", uplink_nmse_lmmse_k, moving_user_mask, "uplink_true_NMSE")
            h_RUs_true_for_oracle_uplink = _build_uplink_reference_truth(
                h_RUs,
                h_RUs_uplink,
                gru_csi_target_mode,
                uplink_tau_ratio,
            )
            uplink_nmse_oracle = _complex_nmse(h_RUs_true_for_oracle_uplink, h_RUs_uplink)
            uplink_nmse_oracle_k = complex_nmse_per_user(h_RUs_true_for_oracle_uplink, h_RUs_uplink)
            logger.info(
                _colorize_branch_line(
                    "Oracle-true",
                    f"Oracle-true uplink_true_NMSE={_highlight_metric_value(uplink_nmse_oracle, 'Oracle-true')}",
                )
            )
            _log_grouped_nmse(logger, "Oracle-true", uplink_nmse_oracle_k, moving_user_mask, "uplink_true_NMSE")

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
        old_global_vec = state_dict_to_vector_backbone(global_model).detach().cpu() if gru_group_mode == "single" else None
        old_global_vec_groups = None
        if gru_group_mode == "grouped":
            old_global_vec_groups = [
                state_dict_to_vector_backbone(global_model_gru_groups[group_idx]).detach().cpu()
                for group_idx in range(2)
            ]
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
        K_norm_gru_effective = K_norm.clone()
        logger.info(
            _ansi(
                f"Round {round_idx} n_k(min/mean/max)=("
                f"{float(K_vals.min()):.1f}/{float(K_vals.mean()):.1f}/{float(K_vals.max()):.1f})",
                "1;32",
            )
        )
        for weight_line in _format_ota_weight_logs(K_vals, K_norm.numpy(), chunk_size=5):
            logger.info(weight_line)

        # Prepare local update vectors once so the OTA path and the OTA-aware optimizer
        # use the same current-round update statistics.
        delta_list = []
        for lm in local_models:
            if gru_group_mode == "single":
                delta_list.append(model_delta_to_vector_backbone(lm, global_model).detach().cpu())
            else:
                group_idx = int(current_gru_group_assignment[len(delta_list)])
                delta_list.append(
                    model_delta_to_vector_backbone(lm, global_model_gru_groups[group_idx]).detach().cpu()
                )
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
        gru_group_user_indices = None
        gru_group_delta_vars = None
        gru_group_user_weights = None
        h_RUs_ota = h_RUs_uplink if use_uplink_target else h_RUs
        h_BUs_ota = h_BUs_uplink if use_uplink_target else h_BUs
        h_RUs_ota_gru = h_RUs_est if ota_use_estimated_h_ru_for_aggregation else h_RUs_ota
        h_RUs_ota_arch = h_RUs_est_arch if ota_use_estimated_h_ru_for_aggregation else h_RUs_ota
        h_RUs_ota_baseline = h_RUs_est_baseline if ota_use_estimated_h_ru_for_aggregation else h_RUs_ota
        if config.use_aircomp and aircomp_sim is not None:
            if gru_group_mode == "single":
                casc_pref = f_beam.conj() @ H_BR.T
                h_eff_list = []
                for k in range(config.num_users):
                    direct = f_beam.conj().dot(h_BUs_ota[k]) if (direct_on == 1 and h_BUs_ota is not None) else 0.0
                    reflect = 0.0
                    if reflect_on == 1:
                        reflect = np.dot(theta_ota, casc_pref * h_RUs_ota_gru[k])
                    h_eff_list.append(direct + reflect)
                h_eff = torch.from_numpy(np.asarray(h_eff_list, dtype=np.complex64))
            else:
                h_eff = torch.zeros((config.num_users,), dtype=torch.complex64)
            if enable_cnn_arch_ablation:
                casc_pref_arch = f_beam_arch.conj() @ H_BR.T
                h_eff_list_arch = []
                for k in range(config.num_users):
                    direct_arch = (
                        f_beam_arch.conj().dot(h_BUs_ota[k]) if (direct_on == 1 and h_BUs_ota is not None) else 0.0
                    )
                    reflect_arch = 0.0
                    if reflect_on == 1:
                        reflect_arch = np.dot(theta_ota_arch, casc_pref_arch * h_RUs_ota_arch[k])
                    h_eff_list_arch.append(direct_arch + reflect_arch)
                h_eff_arch = torch.from_numpy(np.asarray(h_eff_list_arch, dtype=np.complex64))

            if enable_cnn_baseline:
                casc_pref_baseline = f_beam_baseline.conj() @ H_BR.T
                h_eff_list_baseline = []
                for k in range(config.num_users):
                    direct_baseline = (
                        f_beam_baseline.conj().dot(h_BUs_ota[k]) if (direct_on == 1 and h_BUs_ota is not None) else 0.0
                    )
                    reflect_baseline = 0.0
                    if reflect_on == 1:
                        reflect_baseline = np.dot(theta_ota_baseline, casc_pref_baseline * h_RUs_ota_baseline[k])
                    h_eff_list_baseline.append(direct_baseline + reflect_baseline)
                h_eff_baseline = torch.from_numpy(np.asarray(h_eff_list_baseline, dtype=np.complex64))
        if config.use_aircomp and aircomp_sim is not None:
            if gru_group_mode == "single":
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
                aggregator.apply_aggregated_delta(
                    global_model,
                    agg_update,
                    backbone_only=True,
                    prefix="backbone",
                )
                logger.info(
                    _colorize_branch_line(
                        "GRU",
                        f"AirComp eta={diag['eta']:.4e}, min|u|^2={diag['min_inner2']:.4e}, "
                        f"agg_NMSE={_highlight_metric_value(nmse.item(), 'GRU')}, "
                        f"agg_err={agg_error_power.item():.4e}, ideal_power={ideal_power.item():.4e}",
                    )
                )
                new_global_vec = state_dict_to_vector_backbone(global_model).detach().cpu()
                backbone_update_norm = torch.norm(new_global_vec - old_global_vec).item()
                logger.info(f"GRU global backbone update norm: {backbone_update_norm:.4e}")
            else:
                gru_group_user_indices = [
                    np.flatnonzero(current_gru_group_assignment == group_idx).astype(np.int64)
                    for group_idx in range(2)
                ]
                gru_group_delta_vars = [None, None]
                gru_group_user_weights = [None, None]
                group_update_norms = []
                for group_idx in range(2):
                    user_ids = gru_group_user_indices[group_idx]
                    user_ids_t = torch.as_tensor(user_ids, dtype=torch.long)
                    if user_ids.size == 0:
                        raise RuntimeError(f"GRU group {group_idx + 1} is empty in grouped mode")
                    K_vals_g = K_vals[user_ids]
                    K_vec_g = torch.from_numpy(K_vals_g)
                    K_norm_g = K_vec_g / torch.mean(K_vec_g)
                    K_norm_gru_effective[user_ids_t] = K_norm_g
                    gru_group_user_weights[group_idx] = K_norm_g

                    local_models_g = [local_models[int(user_idx)] for user_idx in user_ids.tolist()]
                    delta_list_g = [
                        model_delta_to_vector_backbone(
                            lm,
                            global_model_gru_groups[group_idx],
                            prefix="backbone",
                        ).detach().cpu()
                        for lm in local_models_g
                    ]
                    delta_mat_g = torch.stack(delta_list_g, dim=0)
                    delta_var_g = delta_mat_g.float().var(dim=1, unbiased=False)
                    delta_var_g = torch.clamp(delta_var_g, min=float(config.ota_var_floor))
                    delta_var[user_ids_t] = delta_var_g
                    gru_group_delta_vars[group_idx] = delta_var_g

                    f_group = current_f_beam_gru_groups[group_idx]
                    theta_group = current_theta_ota_gru_groups[group_idx]
                    casc_pref_group = f_group.conj() @ H_BR.T
                    h_eff_list_group = []
                    for user_idx in user_ids.tolist():
                        direct = (
                            f_group.conj().dot(h_BUs_ota[user_idx])
                            if (direct_on == 1 and h_BUs_ota is not None)
                            else 0.0
                        )
                        reflect = 0.0
                        if reflect_on == 1:
                            reflect = np.dot(theta_group, casc_pref_group * h_RUs_ota_gru[user_idx])
                        h_eff_list_group.append(direct + reflect)
                    h_eff_group = torch.from_numpy(np.asarray(h_eff_list_group, dtype=np.complex64))
                    h_eff[user_ids_t] = h_eff_group

                    agg_update_g, diag_g = aircomp_sim.aggregate_updates(
                        updates=delta_mat_g.float(),
                        h_eff=h_eff_group,
                        user_weights=K_norm_g,
                    )
                    ideal_update_g = (
                        (delta_mat_g * K_norm_g.view(-1, 1)).sum(dim=0) / (K_norm_g.sum() + 1e-12)
                    )
                    agg_error_power_g = torch.norm(agg_update_g - ideal_update_g) ** 2
                    ideal_power_g = torch.norm(ideal_update_g) ** 2
                    nmse_g = agg_error_power_g / (ideal_power_g + 1e-12)

                    aggregator.apply_aggregated_delta(
                        global_model_gru_groups[group_idx],
                        agg_update_g,
                        backbone_only=True,
                        prefix="backbone",
                    )
                    logger.info(
                        _colorize_branch_line(
                            "GRU",
                            f"GRU-G{group_idx + 1} AirComp eta={diag_g['eta']:.4e}, "
                            f"min|u|^2={diag_g['min_inner2']:.4e}, "
                            f"agg_NMSE={_highlight_metric_value(nmse_g.item(), 'GRU')}, "
                            f"agg_err={agg_error_power_g.item():.4e}, "
                            f"ideal_power={ideal_power_g.item():.4e}, users={user_ids.tolist()}",
                        )
                    )
                    new_global_vec_group = state_dict_to_vector_backbone(global_model_gru_groups[group_idx]).detach().cpu()
                    update_norm_g = torch.norm(new_global_vec_group - old_global_vec_groups[group_idx]).item()
                    group_update_norms.append(update_norm_g)
                    logger.info(f"GRU-G{group_idx + 1} global backbone update norm: {update_norm_g:.4e}")
                backbone_update_norm = max(group_update_norms) if group_update_norms else 0.0
                logger.info(f"GRU grouped backbone update norm max: {backbone_update_norm:.4e}")
        else:
            # fallback to FedAvg / Reptile on parameters
            if gru_group_mode == "single":
                global_model = aggregator.aggregate(
                    global_model,
                    local_models,
                    backbone_only=True,
                    prefix="backbone",
                )
                new_global_vec = state_dict_to_vector_backbone(global_model).detach().cpu()
                backbone_update_norm = torch.norm(new_global_vec - old_global_vec).item()
                logger.info(f"GRU global backbone update norm: {backbone_update_norm:.4e}")
            else:
                gru_group_user_indices = [
                    np.flatnonzero(current_gru_group_assignment == group_idx).astype(np.int64)
                    for group_idx in range(2)
                ]
                gru_group_delta_vars = [None, None]
                gru_group_user_weights = [None, None]
                group_update_norms = []
                h_eff = torch.zeros((config.num_users,), dtype=torch.complex64)
                for group_idx in range(2):
                    user_ids = gru_group_user_indices[group_idx]
                    user_ids_t = torch.as_tensor(user_ids, dtype=torch.long)
                    if user_ids.size == 0:
                        raise RuntimeError(f"GRU group {group_idx + 1} is empty in grouped mode")
                    local_models_g = [local_models[int(user_idx)] for user_idx in user_ids.tolist()]
                    global_model_gru_groups[group_idx] = aggregator.aggregate(
                        global_model_gru_groups[group_idx],
                        local_models_g,
                        backbone_only=True,
                        prefix="backbone",
                    )
                    delta_var_g = delta_var[user_ids_t]
                    gru_group_delta_vars[group_idx] = delta_var_g
                    K_vals_g = K_vals[user_ids]
                    K_vec_g = torch.from_numpy(K_vals_g)
                    K_norm_g = K_vec_g / torch.mean(K_vec_g)
                    K_norm_gru_effective[user_ids_t] = K_norm_g
                    gru_group_user_weights[group_idx] = K_norm_g
                    new_global_vec_group = state_dict_to_vector_backbone(global_model_gru_groups[group_idx]).detach().cpu()
                    update_norm_g = torch.norm(new_global_vec_group - old_global_vec_groups[group_idx]).item()
                    group_update_norms.append(update_norm_g)
                    logger.info(f"GRU-G{group_idx + 1} global backbone update norm: {update_norm_g:.4e}")
                backbone_update_norm = max(group_update_norms) if group_update_norms else 0.0
                logger.info(f"GRU grouped backbone update norm max: {backbone_update_norm:.4e}")
        if use_persistent_hidden_state and reset_hidden_on_large_backbone_update:
            if backbone_update_norm > hidden_reset_update_norm_threshold:
                reset_hidden_next_round = True
                reset_hidden_next_round_reason = "large_backbone_update"
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
                    _colorize_branch_line(
                        "CNN-arch",
                        f"CNN-arch AirComp eta={diag_arch['eta']:.4e}, "
                        f"min|u|^2={diag_arch['min_inner2']:.4e}, "
                        f"agg_NMSE={_highlight_metric_value(nmse_arch.item(), 'CNN-arch')}, "
                        f"agg_err={agg_error_power_arch.item():.4e}, "
                        f"ideal_power={ideal_power_arch.item():.4e}",
                    )
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
            if (config.use_aircomp and aircomp_sim is not None
                    and h_eff_baseline is not None and delta_mat_baseline is not None):
                # Diagnostic only: CNN-base remains FedAvg below, so this must not alter model updates or RNG state.
                with torch.random.fork_rng(devices=[]):
                    agg_update_baseline_diag, diag_baseline = aircomp_sim.aggregate_updates(
                        updates=delta_mat_baseline.float(),
                        h_eff=h_eff_baseline,
                        user_weights=K_norm,
                    )
                ideal_update_baseline = (
                    (delta_mat_baseline * K_norm.view(-1, 1)).sum(dim=0) / (K_norm.sum() + 1e-12)
                )
                agg_error_power_baseline = torch.norm(agg_update_baseline_diag - ideal_update_baseline) ** 2
                ideal_power_baseline = torch.norm(ideal_update_baseline) ** 2
                nmse_baseline = agg_error_power_baseline / (ideal_power_baseline + 1e-12)
                logger.info(
                    _colorize_branch_line(
                        "CNN-base",
                        f"CNN-base AirComp diagnostic eta={diag_baseline['eta']:.4e}, "
                        f"min|u|^2={diag_baseline['min_inner2']:.4e}, "
                        f"agg_NMSE={_highlight_metric_value(nmse_baseline.item(), 'CNN-base')}, "
                        f"agg_err={agg_error_power_baseline.item():.4e}, "
                        f"ideal_power={ideal_power_baseline.item():.4e}",
                    )
                )
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

        if eta_snapshot_enabled and (round_idx % eta_snapshot_every == 0):
            k_norm_np = K_norm.detach().cpu().numpy().astype(np.float32, copy=False)
            k_norm_gru_np = K_norm_gru_effective.detach().cpu().numpy().astype(np.float32, copy=False)
            snapshot_payload = {
                "round_idx": np.asarray(round_idx, dtype=np.int32),
                "user_id": np.arange(1, config.num_users + 1, dtype=np.int16),
                "K_k": np.asarray(k_norm_np, dtype=np.float32),
                "K_raw": np.asarray(K_vals, dtype=np.float32),
                "gru_group_id": np.asarray(current_gru_group_assignment, dtype=np.int8),
            }
            snapshot_meta = {
                "num_users": np.asarray(config.num_users, dtype=np.int32),
                "beam_ris_optimizer": np.asarray(beam_ris_optimizer),
                "gru_csi_target_mode": np.asarray(gru_csi_target_mode),
                "ota_tx_power": np.asarray(config.ota_tx_power, dtype=np.float32),
                "ota_var_floor": np.asarray(config.ota_var_floor, dtype=np.float32),
                "ota_eps": np.asarray(config.ota_eps, dtype=np.float32),
                "meta_algorithm": np.asarray(str(config.meta_algorithm)),
                "log_stem": np.asarray(logger.stem or config.log_prefix()),
            }

            gru_components = build_eta_components(
                h_eff=h_eff.detach().cpu().numpy(),
                user_weights=k_norm_gru_np,
                update_vars=delta_var.detach().cpu().numpy(),
                tx_power=config.ota_tx_power,
                eps=config.ota_eps,
            )
            for key, value in gru_components.items():
                snapshot_payload[f"gru_{key}"] = np.asarray(value, dtype=np.float32)
            snapshot_payload["gru_uplink_true_nmse_k"] = np.asarray(uplink_nmse_gru_k, dtype=np.float32)

            if enable_cnn_arch_ablation and h_eff_arch is not None and delta_var_arch is not None and uplink_nmse_arch_k is not None:
                arch_components = build_eta_components(
                    h_eff=h_eff_arch.detach().cpu().numpy(),
                    user_weights=k_norm_np,
                    update_vars=delta_var_arch.detach().cpu().numpy(),
                    tx_power=config.ota_tx_power,
                    eps=config.ota_eps,
                )
                for key, value in arch_components.items():
                    snapshot_payload[f"cnn_arch_{key}"] = np.asarray(value, dtype=np.float32)
                snapshot_payload["cnn_arch_uplink_true_nmse_k"] = np.asarray(uplink_nmse_arch_k, dtype=np.float32)

            if enable_cnn_baseline and h_eff_baseline is not None and delta_var_baseline is not None and uplink_nmse_baseline_k is not None:
                baseline_components = build_eta_components(
                    h_eff=h_eff_baseline.detach().cpu().numpy(),
                    user_weights=k_norm_np,
                    update_vars=delta_var_baseline.detach().cpu().numpy(),
                    tx_power=config.ota_tx_power,
                    eps=config.ota_eps,
                )
                for key, value in baseline_components.items():
                    snapshot_payload[f"cnn_base_{key}"] = np.asarray(value, dtype=np.float32)
                snapshot_payload["cnn_base_uplink_true_nmse_k"] = np.asarray(uplink_nmse_baseline_k, dtype=np.float32)

            save_snapshot_npz(
                run_dir=eta_snapshot_run_dir,
                round_idx=round_idx,
                payload=snapshot_payload,
                meta=snapshot_meta,
            )

        if mobility_snapshot_enabled and (round_idx % mobility_snapshot_every == 0):
            user_id = np.arange(1, config.num_users + 1, dtype=np.int16)
            positions_next_round = ru_evolver.positions_at(float(round_idx))
            ris_dist_t = ru_evolver.ris_distances(positions_t_round).astype(np.float32, copy=False)
            ris_dist_tau = ru_evolver.ris_distances(positions_uplink_round).astype(np.float32, copy=False)
            ris_dist_next = ru_evolver.ris_distances(positions_next_round).astype(np.float32, copy=False)
            ris_pl_t = ru_evolver.ris_pathloss(positions_t_round).astype(np.float32, copy=False)
            ris_pl_tau = ru_evolver.ris_pathloss(positions_uplink_round).astype(np.float32, copy=False)
            ris_pl_next = ru_evolver.ris_pathloss(positions_next_round).astype(np.float32, copy=False)

            direct_dist_t = np.full((config.num_users,), np.nan, dtype=np.float32)
            direct_dist_tau = np.full((config.num_users,), np.nan, dtype=np.float32)
            direct_dist_next = np.full((config.num_users,), np.nan, dtype=np.float32)
            direct_pl_t = np.full((config.num_users,), np.nan, dtype=np.float32)
            direct_pl_tau = np.full((config.num_users,), np.nan, dtype=np.float32)
            direct_pl_next = np.full((config.num_users,), np.nan, dtype=np.float32)
            if direct_on == 1:
                direct_dist_t = ru_evolver.direct_distances(positions_t_round).astype(np.float32, copy=False)
                direct_dist_tau = ru_evolver.direct_distances(positions_uplink_round).astype(np.float32, copy=False)
                direct_dist_next = ru_evolver.direct_distances(positions_next_round).astype(np.float32, copy=False)
                direct_pl_t = ru_evolver.direct_pathloss(positions_t_round).astype(np.float32, copy=False)
                direct_pl_tau = ru_evolver.direct_pathloss(positions_uplink_round).astype(np.float32, copy=False)
                direct_pl_next = ru_evolver.direct_pathloss(positions_next_round).astype(np.float32, copy=False)

            mobility_payload = {
                "round_idx": np.asarray(round_idx, dtype=np.int32),
                "user_id": user_id,
                "moving_user_mask": np.asarray(moving_user_mask, dtype=np.int8),
                "speed_mps": np.asarray(ru_evolver.speed_magnitudes, dtype=np.float32),
                "doppler_hz": np.asarray(ru_evolver.current_doppler_vector(), dtype=np.float32),
                "alpha_delta": np.asarray(alpha_used_next, dtype=np.float32),
                "alpha_tau": np.asarray(alpha_used_uplink, dtype=np.float32) if alpha_used_uplink is not None else np.full((config.num_users,), np.nan, dtype=np.float32),
                "tau_loss_weight": np.asarray(tau_loss_weight_round, dtype=np.float32),
                "tau_ratio": np.asarray(uplink_tau_ratio, dtype=np.float32),
                "K_k": np.asarray(K_vals, dtype=np.float32),
                "K_norm": np.asarray(K_norm.detach().cpu().numpy(), dtype=np.float32),
                "pos_t_xy": np.asarray(positions_t_round, dtype=np.float32),
                "pos_tau_xy": np.asarray(positions_uplink_round, dtype=np.float32),
                "pos_next_xy": np.asarray(positions_next_round, dtype=np.float32),
                "ris_distance_t": ris_dist_t,
                "ris_distance_tau": ris_dist_tau,
                "ris_distance_next": ris_dist_next,
                "ris_pathloss_t": ris_pl_t,
                "ris_pathloss_tau": ris_pl_tau,
                "ris_pathloss_next": ris_pl_next,
                "direct_distance_t": direct_dist_t,
                "direct_distance_tau": direct_dist_tau,
                "direct_distance_next": direct_dist_next,
                "direct_pathloss_t": direct_pl_t,
                "direct_pathloss_tau": direct_pl_tau,
                "direct_pathloss_next": direct_pl_next,
            }
            if uplink_nmse_gru_k is not None:
                mobility_payload["gru_uplink_true_nmse_k"] = np.asarray(uplink_nmse_gru_k, dtype=np.float32)
            if uplink_nmse_gru_oracle_k is not None:
                mobility_payload["gru_oracle_uplink_true_nmse_k"] = np.asarray(uplink_nmse_gru_oracle_k, dtype=np.float32)
            if uplink_nmse_arch_k is not None:
                mobility_payload["cnn_arch_uplink_true_nmse_k"] = np.asarray(uplink_nmse_arch_k, dtype=np.float32)
            if uplink_nmse_baseline_k is not None:
                mobility_payload["cnn_base_uplink_true_nmse_k"] = np.asarray(uplink_nmse_baseline_k, dtype=np.float32)
            if uplink_nmse_lmmse_k is not None:
                mobility_payload["lmmse_uplink_true_nmse_k"] = np.asarray(uplink_nmse_lmmse_k, dtype=np.float32)
            if uplink_nmse_oracle_k is not None:
                mobility_payload["oracle_uplink_true_nmse_k"] = np.asarray(uplink_nmse_oracle_k, dtype=np.float32)

            mobility_meta = {
                "num_users": np.asarray(config.num_users, dtype=np.int32),
                "dt_seconds": np.asarray(config.channel_time_step, dtype=np.float32),
                "fc_hz": np.asarray(config.channel_carrier_frequency_hz, dtype=np.float32),
                "tau_ratio": np.asarray(uplink_tau_ratio, dtype=np.float32),
                "gru_csi_target_mode": np.asarray(gru_csi_target_mode),
                "beam_ris_optimizer": np.asarray(beam_ris_optimizer),
                "log_stem": np.asarray(logger.stem or config.log_prefix()),
                "reflect_on": np.asarray(reflect_on, dtype=np.int8),
                "direct_on": np.asarray(direct_on, dtype=np.int8),
            }
            save_snapshot_npz(
                run_dir=mobility_snapshot_run_dir,
                round_idx=round_idx,
                payload=mobility_payload,
                meta=mobility_meta,
            )

        if delta_motion_snapshot_enabled and (round_idx % delta_motion_snapshot_every == 0):
            user_id = np.arange(1, config.num_users + 1, dtype=np.int16)
            positions_next_round = ru_evolver.positions_at(float(round_idx))
            delta_motion_payload = {
                "round_idx": np.asarray(round_idx, dtype=np.int32),
                "user_id": user_id,
                "moving_user_mask": np.asarray(moving_user_mask, dtype=np.int8),
                "speed_mps": np.asarray(ru_evolver.speed_magnitudes, dtype=np.float32),
                "doppler_hz": np.asarray(ru_evolver.current_doppler_vector(), dtype=np.float32),
                "alpha_delta": np.asarray(alpha_used_next, dtype=np.float32),
                "alpha_tau": np.asarray(alpha_used_uplink, dtype=np.float32) if alpha_used_uplink is not None else np.full((config.num_users,), np.nan, dtype=np.float32),
                "tau_ratio": np.asarray(uplink_tau_ratio, dtype=np.float32),
                "tau_loss_weight": np.asarray(tau_loss_weight_round, dtype=np.float32),
                "pos_t_xy": np.asarray(positions_t_round, dtype=np.float32),
                "pos_tau_xy": np.asarray(positions_uplink_round, dtype=np.float32),
                "pos_next_xy": np.asarray(positions_next_round, dtype=np.float32),
                "ris_pathloss_tau": np.asarray(ru_evolver.ris_pathloss(positions_uplink_round), dtype=np.float32),
                "ris_distance_tau": np.asarray(ru_evolver.ris_distances(positions_uplink_round), dtype=np.float32),
                "true_delta_norm_k": np.asarray(true_delta_norm_round, dtype=np.float32),
                "true_delta_rel_norm_k": np.asarray(true_delta_rel_round, dtype=np.float32),
                "gru_delta_proxy_norm_k": np.asarray(gru_delta_proxy_norm_round, dtype=np.float32),
                "gru_delta_proxy_rel_norm_k": np.asarray(gru_delta_proxy_rel_round, dtype=np.float32),
            }
            if direct_on == 1:
                delta_motion_payload["direct_pathloss_tau"] = np.asarray(ru_evolver.direct_pathloss(positions_uplink_round), dtype=np.float32)
                delta_motion_payload["direct_distance_tau"] = np.asarray(ru_evolver.direct_distances(positions_uplink_round), dtype=np.float32)

            delta_motion_meta = {
                "num_users": np.asarray(config.num_users, dtype=np.int32),
                "dt_seconds": np.asarray(config.channel_time_step, dtype=np.float32),
                "fc_hz": np.asarray(config.channel_carrier_frequency_hz, dtype=np.float32),
                "tau_ratio": np.asarray(uplink_tau_ratio, dtype=np.float32),
                "gru_csi_target_mode": np.asarray(gru_csi_target_mode),
                "log_stem": np.asarray(logger.stem or config.log_prefix()),
                "reflect_on": np.asarray(reflect_on, dtype=np.int8),
                "direct_on": np.asarray(direct_on, dtype=np.int8),
            }
            save_snapshot_npz(
                run_dir=delta_motion_snapshot_run_dir,
                round_idx=round_idx,
                payload=delta_motion_payload,
                meta=delta_motion_meta,
            )

        if gru_pl_snapshot_enabled and (round_idx % gru_pl_snapshot_every == 0):
            log_pl_true = np.log(np.maximum(gru_pl_true_round.astype(np.float64), float(gru_pl_eps))).astype(np.float32)
            log_pl_pred = np.log(np.maximum(gru_pl_pred_round.astype(np.float64), float(gru_pl_eps))).astype(np.float32)
            pl_abs_err = np.abs(gru_pl_pred_round - gru_pl_true_round).astype(np.float32)
            pl_rel_err = (pl_abs_err / np.maximum(gru_pl_true_round, float(gru_pl_eps))).astype(np.float32)
            log_pl_signed_err = (log_pl_pred - log_pl_true).astype(np.float32)
            log_pl_abs_err = np.abs(log_pl_signed_err).astype(np.float32)
            snapshot_payload = {
                "round_idx": np.asarray(round_idx, dtype=np.int32),
                "user_id": np.arange(1, config.num_users + 1, dtype=np.int16),
                "gru_group_id": np.asarray(current_gru_group_assignment, dtype=np.int8),
                "moving_user_mask": np.asarray(moving_user_mask, dtype=np.int8),
                "speed_mps": np.asarray(ru_evolver.speed_magnitudes, dtype=np.float32),
                "doppler_hz": np.asarray(ru_evolver.current_doppler_vector(), dtype=np.float32),
                "alpha_tau": np.asarray(alpha_used_uplink, dtype=np.float32) if alpha_used_uplink is not None else np.full((config.num_users,), np.nan, dtype=np.float32),
                "tau_loss_weight": np.asarray(tau_loss_weight_round, dtype=np.float32),
                "ris_pathloss_t": np.asarray(ris_pl_t_round, dtype=np.float32),
                "ris_pathloss_tau": np.asarray(ris_pl_uplink_round, dtype=np.float32),
                "gru_pl_true_sel": np.asarray(gru_pl_true_round, dtype=np.float32),
                "gru_pl_pred_sel": np.asarray(gru_pl_pred_round, dtype=np.float32),
                "gru_log_pl_true_sel": log_pl_true,
                "gru_log_pl_pred_sel": log_pl_pred,
                "gru_pl_abs_err": pl_abs_err,
                "gru_pl_rel_err": pl_rel_err,
                "gru_log_pl_signed_err": log_pl_signed_err,
                "gru_log_pl_abs_err": log_pl_abs_err,
            }
            if uplink_nmse_gru_k is not None:
                snapshot_payload["gru_uplink_true_nmse_k"] = np.asarray(uplink_nmse_gru_k, dtype=np.float32)
            if uplink_nmse_gru_oracle_k is not None:
                snapshot_payload["gru_oracle_uplink_true_nmse_k"] = np.asarray(uplink_nmse_gru_oracle_k, dtype=np.float32)
            snapshot_meta = {
                "num_users": np.asarray(config.num_users, dtype=np.int32),
                "gru_csi_target_mode": np.asarray(gru_csi_target_mode),
                "beam_ris_optimizer": np.asarray(beam_ris_optimizer),
                "meta_algorithm": np.asarray(str(config.meta_algorithm)),
                "log_stem": np.asarray(logger.stem or config.log_prefix()),
                "pl_eps": np.asarray(gru_pl_eps, dtype=np.float32),
                "log_pl_min": np.asarray(gru_log_pl_min, dtype=np.float32),
                "log_pl_max": np.asarray(gru_log_pl_max, dtype=np.float32),
                "reflect_on": np.asarray(reflect_on, dtype=np.int8),
                "direct_on": np.asarray(direct_on, dtype=np.int8),
            }
            save_snapshot_npz(
                run_dir=gru_pl_snapshot_run_dir,
                round_idx=round_idx,
                payload=snapshot_payload,
                meta=snapshot_meta,
            )

        # Optimize beamforming vector f and RIS phases theta for next round
        logger.info("Optimizing beamforming and RIS configuration.")
        # Use model-estimated h_RU for OTA beam/RIS optimization.
        h_BUs_opt = h_BUs_uplink if use_uplink_target else h_BUs
        if gru_group_mode == "single":
            optimize_start = time.perf_counter()
            f_beam, theta_ota, nmse_proxy = optimize_beam_ris_by_mode(
                mode=beam_ris_optimizer,
                H_BR=H_BR,
                h_RUs=h_RUs_est,
                h_BUs=h_BUs_opt,
                theta_init=theta_ota,
                f_init=f_beam,
                link_switch=link_switch,
                user_weights=K_norm.numpy(),
                update_vars=delta_var.numpy(),
                tx_power=config.ota_tx_power,
                noise_std=config.ota_noise_std,
                var_floor=config.ota_var_floor,
                eps=config.ota_eps,
                oa_iters=oa_iters,
                sca_iters=sca_iters,
                sca_threshold=sca_threshold,
                sca_tau=sca_tau,
                dc_outer_iters=dc_outer_iters,
                dc_inner_iters=dc_inner_iters,
                dc_tol=dc_tol,
                dc_inner_tol=dc_inner_tol,
            )
            optimize_elapsed = time.perf_counter() - optimize_start
            logger.info(
                f"Round {round_idx} GRU {beam_ris_optimizer.upper()} solve_time={optimize_elapsed:.4f}s"
            )
        else:
            group_nmse_proxies = []
            for group_idx in range(2):
                user_ids = gru_group_user_indices[group_idx]
                optimize_start_group = time.perf_counter()
                f_beam_gru_groups[group_idx], theta_ota_gru_groups[group_idx], nmse_proxy_group = optimize_beam_ris_by_mode(
                    mode=beam_ris_optimizer,
                    H_BR=H_BR,
                    h_RUs=h_RUs_est[user_ids],
                    h_BUs=None if h_BUs_opt is None else h_BUs_opt[user_ids],
                    theta_init=theta_ota_gru_groups[group_idx],
                    f_init=f_beam_gru_groups[group_idx],
                    link_switch=link_switch,
                    user_weights=gru_group_user_weights[group_idx].numpy(),
                    update_vars=gru_group_delta_vars[group_idx].numpy(),
                    tx_power=config.ota_tx_power,
                    noise_std=config.ota_noise_std,
                    var_floor=config.ota_var_floor,
                    eps=config.ota_eps,
                    oa_iters=oa_iters,
                    sca_iters=sca_iters,
                    sca_threshold=sca_threshold,
                    sca_tau=sca_tau,
                    dc_outer_iters=dc_outer_iters,
                    dc_inner_iters=dc_inner_iters,
                    dc_tol=dc_tol,
                    dc_inner_tol=dc_inner_tol,
                )
                optimize_elapsed_group = time.perf_counter() - optimize_start_group
                group_nmse_proxies.append(float(nmse_proxy_group))
                logger.info(
                    f"Round {round_idx} GRU-G{group_idx + 1} {beam_ris_optimizer.upper()} "
                    f"solve_time={optimize_elapsed_group:.4f}s, users={user_ids.tolist()}"
                )
        # New pilot pattern independent from OTA theta
        theta_pilot = pilot_gen.generate_pilot_pattern(config.num_pilots, config.num_ris_elements)
        if gru_group_mode == "single":
            if log_oa_vectors:
                logger.info(f"Optimized f: {np.round(f_beam, 4)}")
                logger.info(
                    _colorize_branch_line(
                        "GRU",
                        f"Optimized theta_ota: {np.round(theta_ota, 4)}, "
                        f"proxy_NMSE={_highlight_metric_value(nmse_proxy, 'GRU')}",
                    )
                )
            else:
                logger.info(
                    _colorize_branch_line(
                        "GRU",
                        f"GRU proxy_NMSE={_highlight_metric_value(nmse_proxy, 'GRU')}",
                    )
                )
        else:
            for group_idx in range(2):
                group_nmse_proxy = group_nmse_proxies[group_idx]
                if log_oa_vectors:
                    logger.info(f"Optimized f (GRU-G{group_idx + 1}): {np.round(f_beam_gru_groups[group_idx], 4)}")
                    logger.info(
                        _colorize_branch_line(
                            "GRU",
                            f"Optimized theta_ota (GRU-G{group_idx + 1}): {np.round(theta_ota_gru_groups[group_idx], 4)}, "
                            f"proxy_NMSE={_highlight_metric_value(group_nmse_proxy, 'GRU')}",
                        )
                    )
                else:
                    logger.info(
                        _colorize_branch_line(
                            "GRU",
                            f"GRU-G{group_idx + 1} proxy_NMSE={_highlight_metric_value(group_nmse_proxy, 'GRU')}",
                        )
                    )

        if enable_cnn_arch_ablation:
            optimize_start_arch = time.perf_counter()
            f_beam_arch, theta_ota_arch, nmse_proxy_arch = optimize_beam_ris_by_mode(
                mode=beam_ris_optimizer,
                H_BR=H_BR, h_RUs=h_RUs_est_arch, h_BUs=h_BUs_opt, theta_init=theta_ota_arch, f_init=f_beam_arch,
                link_switch=link_switch, user_weights=K_norm.numpy(),
                update_vars=delta_var_arch.numpy(),
                tx_power=config.ota_tx_power,
                noise_std=config.ota_noise_std,
                var_floor=config.ota_var_floor,
                eps=config.ota_eps,
                oa_iters=oa_iters,
                sca_iters=sca_iters,
                sca_threshold=sca_threshold,
                sca_tau=sca_tau,
                dc_outer_iters=dc_outer_iters,
                dc_inner_iters=dc_inner_iters,
                dc_tol=dc_tol,
                dc_inner_tol=dc_inner_tol,
            )
            optimize_elapsed_arch = time.perf_counter() - optimize_start_arch
            logger.info(
                f"Round {round_idx} CNN-arch {beam_ris_optimizer.upper()} solve_time={optimize_elapsed_arch:.4f}s"
            )
            if log_oa_vectors:
                logger.info(f"Optimized f (CNN-arch): {np.round(f_beam_arch, 4)}")
                logger.info(
                    _colorize_branch_line(
                        "CNN-arch",
                        f"Optimized theta_ota (CNN-arch): {np.round(theta_ota_arch, 4)}, "
                        f"proxy_NMSE={_highlight_metric_value(nmse_proxy_arch, 'CNN-arch')}",
                    )
                )
            else:
                logger.info(
                    _colorize_branch_line(
                        "CNN-arch",
                        f"CNN-arch proxy_NMSE={_highlight_metric_value(nmse_proxy_arch, 'CNN-arch')}",
                    )
                )

        if enable_cnn_baseline:
            baseline_user_weights = K_norm.numpy()
            optimize_start_baseline = time.perf_counter()
            f_beam_baseline, theta_ota_baseline, nmse_proxy_baseline = optimize_beam_ris_by_mode(
                mode=beam_ris_optimizer,
                H_BR=H_BR, h_RUs=h_RUs_est_baseline, h_BUs=h_BUs_opt, theta_init=theta_ota_baseline, f_init=f_beam_baseline,
                link_switch=link_switch, user_weights=baseline_user_weights,
                update_vars=delta_var_baseline.numpy(),
                tx_power=config.ota_tx_power,
                noise_std=config.ota_noise_std,
                var_floor=config.ota_var_floor,
                eps=config.ota_eps,
                oa_iters=oa_iters,
                sca_iters=sca_iters,
                sca_threshold=sca_threshold,
                sca_tau=sca_tau,
                dc_outer_iters=dc_outer_iters,
                dc_inner_iters=dc_inner_iters,
                dc_tol=dc_tol,
                dc_inner_tol=dc_inner_tol,
            )
            optimize_elapsed_baseline = time.perf_counter() - optimize_start_baseline
            logger.info(
                f"Round {round_idx} CNN-base {beam_ris_optimizer.upper()} solve_time={optimize_elapsed_baseline:.4f}s"
            )
            if log_oa_vectors:
                logger.info(f"Optimized f (CNN-base): {np.round(f_beam_baseline, 4)}")
                logger.info(
                    _colorize_branch_line(
                        "CNN-base",
                        f"Optimized theta_ota (CNN-base): {np.round(theta_ota_baseline, 4)}, "
                        f"proxy_NMSE={_highlight_metric_value(nmse_proxy_baseline, 'CNN-base')}",
                    )
                )
            else:
                logger.info(
                    _colorize_branch_line(
                        "CNN-base",
                        f"CNN-base proxy_NMSE={_highlight_metric_value(nmse_proxy_baseline, 'CNN-base')}",
                    )
                )

        if enable_lmmse_baseline:
            lmmse_update_vars = np.ones((config.num_users,), dtype=np.float32)
            optimize_start_lmmse = time.perf_counter()
            f_beam_lmmse, theta_ota_lmmse, nmse_proxy_lmmse = optimize_beam_ris_by_mode(
                mode=beam_ris_optimizer,
                H_BR=H_BR,
                h_RUs=h_RUs_est_lmmse,
                h_BUs=h_BUs_opt,
                theta_init=theta_ota_lmmse,
                f_init=f_beam_lmmse,
                link_switch=link_switch,
                user_weights=K_norm.numpy(),
                update_vars=lmmse_update_vars,
                tx_power=config.ota_tx_power,
                noise_std=config.ota_noise_std,
                var_floor=config.ota_var_floor,
                eps=config.ota_eps,
                oa_iters=oa_iters,
                sca_iters=sca_iters,
                sca_threshold=sca_threshold,
                sca_tau=sca_tau,
                dc_outer_iters=dc_outer_iters,
                dc_inner_iters=dc_inner_iters,
                dc_tol=dc_tol,
                dc_inner_tol=dc_inner_tol,
            )
            optimize_elapsed_lmmse = time.perf_counter() - optimize_start_lmmse
            logger.info(
                f"Round {round_idx} LMMSE {beam_ris_optimizer.upper()} solve_time={optimize_elapsed_lmmse:.4f}s"
            )
            if log_oa_vectors:
                logger.info(f"Optimized f (LMMSE): {np.round(f_beam_lmmse, 4)}")
                logger.info(
                    _colorize_branch_line(
                        "LMMSE",
                        f"Optimized theta_ota (LMMSE): {np.round(theta_ota_lmmse, 4)}, "
                        f"proxy_NMSE={_highlight_metric_value(nmse_proxy_lmmse, 'LMMSE')}",
                    )
                )
            else:
                logger.info(
                    _colorize_branch_line(
                        "LMMSE",
                        f"LMMSE proxy_NMSE={_highlight_metric_value(nmse_proxy_lmmse, 'LMMSE')}",
                    )
                )

        # Oracle upper-bound AO reference with true channels.
        h_RUs_true_for_oracle = _build_uplink_reference_truth(
            h_RUs,
            h_RUs_uplink,
            gru_csi_target_mode,
            uplink_tau_ratio,
        )
        optimize_start_oracle = time.perf_counter()
        f_beam_oracle, theta_ota_oracle, nmse_proxy_oracle = optimize_beam_ris_by_mode(
            mode=beam_ris_optimizer,
            H_BR=H_BR, h_RUs=h_RUs_true_for_oracle, h_BUs=h_BUs_opt, theta_init=theta_ota_oracle, f_init=f_beam_oracle,
            link_switch=link_switch, user_weights=K_norm.numpy(),
            update_vars=delta_var.numpy(),
            tx_power=config.ota_tx_power,
            noise_std=config.ota_noise_std,
            var_floor=config.ota_var_floor,
            eps=config.ota_eps,
            oa_iters=oa_iters,
            sca_iters=sca_iters,
            sca_threshold=sca_threshold,
            sca_tau=sca_tau,
            dc_outer_iters=dc_outer_iters,
            dc_inner_iters=dc_inner_iters,
            dc_tol=dc_tol,
            dc_inner_tol=dc_inner_tol,
        )
        optimize_elapsed_oracle = time.perf_counter() - optimize_start_oracle
        logger.info(
            f"Round {round_idx} Oracle-true {beam_ris_optimizer.upper()} solve_time={optimize_elapsed_oracle:.4f}s"
        )
        if log_oa_vectors:
            logger.info(f"Optimized f (Oracle-true): {np.round(f_beam_oracle, 4)}")
            logger.info(
                _colorize_branch_line(
                    "Oracle-true",
                    f"Optimized theta_ota (Oracle-true): {np.round(theta_ota_oracle, 4)}, "
                    f"proxy_NMSE={_highlight_metric_value(nmse_proxy_oracle, 'Oracle-true')}",
                )
            )
        else:
            logger.info(
                _colorize_branch_line(
                    "Oracle-true",
                    f"Oracle-true proxy_NMSE={_highlight_metric_value(nmse_proxy_oracle, 'Oracle-true')}",
                )
            )

        # Plot-only normalization: keep logged proxy values branch-specific, but draw the
        # proxy figure with one common reference update variance for cross-baseline fairness.
        if gru_group_mode == "single":
            common_proxy_update_vars = delta_var.detach().cpu().numpy()
            common_proxy_user_weights = K_norm.detach().cpu().numpy()
            common_proxy_plot_values = {}

            def _record_common_proxy_plot_value(model_name, h_ru_branch, f_branch, theta_branch):
                proxy_value = _proxy_nmse_for_plot_common_update(
                    H_BR=H_BR,
                    h_RUs=h_ru_branch,
                    h_BUs=h_BUs_opt,
                    f_vec=f_branch,
                    theta_vec=theta_branch,
                    link_switch=link_switch,
                    user_weights=common_proxy_user_weights,
                    update_vars=common_proxy_update_vars,
                    tx_power=config.ota_tx_power,
                    noise_std=config.ota_noise_std,
                    eps=config.ota_eps,
                )
                if proxy_value is not None:
                    common_proxy_plot_values[model_name] = proxy_value

            _record_common_proxy_plot_value("GRU", h_RUs_est, f_beam, theta_ota)
            if enable_cnn_arch_ablation and h_RUs_est_arch is not None:
                _record_common_proxy_plot_value("CNN-arch", h_RUs_est_arch, f_beam_arch, theta_ota_arch)
            if enable_cnn_baseline and h_RUs_est_baseline is not None:
                _record_common_proxy_plot_value("CNN-base", h_RUs_est_baseline, f_beam_baseline, theta_ota_baseline)
            if enable_lmmse_baseline and h_RUs_est_lmmse is not None:
                _record_common_proxy_plot_value("LMMSE", h_RUs_est_lmmse, f_beam_lmmse, theta_ota_lmmse)
            _record_common_proxy_plot_value("Oracle-true", h_RUs_true_for_oracle, f_beam_oracle, theta_ota_oracle)
            _append_proxy_plot_common_update_csv(logger.stem, round_idx, common_proxy_plot_values)

        if enable_gru_semantic_grouping:
            beta_hat_round, d_hat_round, B_round, D_round = _build_gru_grouping_proxies(
                pl_pred_round=gru_pl_pred_round,
                delta_norm_round=gru_delta_proxy_norm_round,
                h_ru_est=h_RUs_est,
                eps=gru_group_eps,
            )
            if not np.all(np.isfinite(beta_hat_round)) or not np.all(np.isfinite(d_hat_round)):
                raise RuntimeError("GRU semantic grouping received non-finite proxy values")

            if gru_group_beta_ema is None:
                gru_group_beta_ema = beta_hat_round.copy()
                gru_group_d_ema = d_hat_round.copy()
            else:
                gru_group_beta_ema = (
                    gru_group_switch_ema_lambda * gru_group_beta_ema
                    + (1.0 - gru_group_switch_ema_lambda) * beta_hat_round
                )
                gru_group_d_ema = (
                    gru_group_switch_ema_lambda * gru_group_d_ema
                    + (1.0 - gru_group_switch_ema_lambda) * d_hat_round
                )

            prev_B_ema = gru_group_scalar_B_ema
            prev_D_ema = gru_group_scalar_D_ema
            if prev_B_ema is None or prev_D_ema is None:
                gru_group_scalar_B_ema = float(B_round)
                gru_group_scalar_D_ema = float(D_round)
                delta_B_ema = float("inf")
                delta_D_ema = float("inf")
                plateau_reached = False
                gru_group_patience_counter = 0
            else:
                gru_group_scalar_B_ema = (
                    gru_group_switch_ema_lambda * float(prev_B_ema)
                    + (1.0 - gru_group_switch_ema_lambda) * float(B_round)
                )
                gru_group_scalar_D_ema = (
                    gru_group_switch_ema_lambda * float(prev_D_ema)
                    + (1.0 - gru_group_switch_ema_lambda) * float(D_round)
                )
                delta_B_ema = _relative_scalar_change(gru_group_scalar_B_ema, prev_B_ema, gru_group_eps)
                delta_D_ema = _relative_scalar_change(gru_group_scalar_D_ema, prev_D_ema, gru_group_eps)
                plateau_reached = (
                    delta_B_ema <= gru_group_switch_tau_b and delta_D_ema <= gru_group_switch_tau_d
                )
                if plateau_reached:
                    gru_group_patience_counter += 1
                else:
                    gru_group_patience_counter = 0

            logger.info(
                "GRU semantic proxies -> "
                f"B={B_round:.4e}, D={D_round:.4e}, "
                f"EMA(B,D)=({gru_group_scalar_B_ema:.4e},{gru_group_scalar_D_ema:.4e}), "
                f"delta=({delta_B_ema:.4e},{delta_D_ema:.4e}), "
                f"patience={gru_group_patience_counter}, mode={gru_group_mode}"
            )

            if (
                gru_group_switch_sensitivity_snapshot_enabled
                and gru_group_mode == "single"
                and round_idx < gru_group_switch_min_round
            ):
                plateau_grid = (
                    (delta_B_ema <= gru_group_switch_sensitivity_tau_b_grid[:, None])
                    & (delta_D_ema <= gru_group_switch_sensitivity_tau_d_grid[None, :])
                )
                plateau_grid &= bool(np.isfinite(delta_B_ema) and np.isfinite(delta_D_ema))
                gru_group_switch_sensitivity_patience_grid = np.where(
                    plateau_grid,
                    gru_group_switch_sensitivity_patience_grid + 1,
                    0,
                ).astype(np.int16, copy=False)
                ready_grid = gru_group_switch_sensitivity_patience_grid >= int(gru_group_switch_patience)
                if round_idx % gru_group_switch_sensitivity_snapshot_every == 0:
                    snapshot_payload = {
                        "round_idx": np.asarray(round_idx, dtype=np.int32),
                        "B": np.asarray(float(B_round), dtype=np.float32),
                        "D": np.asarray(float(D_round), dtype=np.float32),
                        "B_ema": np.asarray(float(gru_group_scalar_B_ema), dtype=np.float32),
                        "D_ema": np.asarray(float(gru_group_scalar_D_ema), dtype=np.float32),
                        "prev_B_ema": np.asarray(
                            np.nan if prev_B_ema is None else float(prev_B_ema),
                            dtype=np.float32,
                        ),
                        "prev_D_ema": np.asarray(
                            np.nan if prev_D_ema is None else float(prev_D_ema),
                            dtype=np.float32,
                        ),
                        "delta_B_ema": np.asarray(float(delta_B_ema), dtype=np.float32),
                        "delta_D_ema": np.asarray(float(delta_D_ema), dtype=np.float32),
                        "configured_tau_B": np.asarray(float(gru_group_switch_tau_b), dtype=np.float32),
                        "configured_tau_D": np.asarray(float(gru_group_switch_tau_d), dtype=np.float32),
                        "configured_plateau_reached": np.asarray(bool(plateau_reached), dtype=np.int8),
                        "configured_patience": np.asarray(int(gru_group_patience_counter), dtype=np.int16),
                        "switch_min_round": np.asarray(int(gru_group_switch_min_round), dtype=np.int32),
                        "switch_patience": np.asarray(int(gru_group_switch_patience), dtype=np.int16),
                        "rounds_until_min_round": np.asarray(
                            int(gru_group_switch_min_round - round_idx),
                            dtype=np.int32,
                        ),
                        "tau_B_grid": np.asarray(gru_group_switch_sensitivity_tau_b_grid, dtype=np.float32),
                        "tau_D_grid": np.asarray(gru_group_switch_sensitivity_tau_d_grid, dtype=np.float32),
                        "plateau_grid": np.asarray(plateau_grid, dtype=np.int8),
                        "patience_grid": np.asarray(gru_group_switch_sensitivity_patience_grid, dtype=np.int16),
                        "ready_grid": np.asarray(ready_grid, dtype=np.int8),
                    }
                    snapshot_meta = {
                        "num_users": np.asarray(config.num_users, dtype=np.int32),
                        "log_stem": np.asarray(logger.stem or config.log_prefix()),
                        "switch_min_round": np.asarray(int(gru_group_switch_min_round), dtype=np.int32),
                        "switch_patience": np.asarray(int(gru_group_switch_patience), dtype=np.int16),
                        "ema_lambda": np.asarray(float(gru_group_switch_ema_lambda), dtype=np.float32),
                        "eps": np.asarray(float(gru_group_eps), dtype=np.float32),
                        "configured_tau_B": np.asarray(float(gru_group_switch_tau_b), dtype=np.float32),
                        "configured_tau_D": np.asarray(float(gru_group_switch_tau_d), dtype=np.float32),
                        "tau_B_grid": np.asarray(gru_group_switch_sensitivity_tau_b_grid, dtype=np.float32),
                        "tau_D_grid": np.asarray(gru_group_switch_sensitivity_tau_d_grid, dtype=np.float32),
                    }
                    save_snapshot_npz(
                        run_dir=gru_group_switch_sensitivity_snapshot_run_dir,
                        round_idx=round_idx,
                        payload=snapshot_payload,
                        meta=snapshot_meta,
                    )

            if gru_group_mode == "single":
                if (
                    round_idx >= gru_group_switch_min_round
                    and gru_group_patience_counter >= gru_group_switch_patience
                ):
                    global_model_gru_groups = [copy.deepcopy(global_model), copy.deepcopy(global_model)]
                    f_beam_gru_groups = [f_beam.copy(), f_beam.copy()]
                    theta_ota_gru_groups = [theta_ota.copy(), theta_ota.copy()]
                    grouping_result = _run_gru_grouping_optimizer(
                        beta_hat=beta_hat_round,
                        d_hat=d_hat_round,
                        beta_ema=gru_group_beta_ema,
                        d_ema=gru_group_d_ema,
                        H_BR=H_BR,
                        h_ru_est=h_RUs_est,
                        f_single=f_beam,
                        group_beams=None,
                        group_assignment=None,
                        cfg=gru_grouping_cfg,
                        prev_x_hard=None,
                        prev_x_soft=None,
                        prev_mu=None,
                    )
                    gru_group_assignment = np.asarray(grouping_result.x_hard, dtype=np.int64)
                    gru_group_prev_x_hard = np.asarray(grouping_result.x_hard, dtype=np.float64)
                    gru_group_prev_x_soft = np.asarray(grouping_result.x_soft, dtype=np.float64)
                    gru_group_prev_mu = np.asarray(grouping_result.mu, dtype=np.float64)
                    if gru_restart_training_after_switch:
                        fresh_template = _create_fresh_gru_model(
                            observation_dim=obs_dim,
                            output_dim=output_dim,
                            enable_pl_factorization=enable_gru_pl_factorization,
                            log_pl_min=gru_log_pl_min,
                            log_pl_max=gru_log_pl_max,
                        )
                        global_model_gru_groups = [copy.deepcopy(fresh_template), copy.deepcopy(fresh_template)]
                        fresh_head_state = {
                            key: value.detach().clone()
                            for key, value in fresh_template.state_dict().items()
                            if key.startswith("head")
                        }
                        user_head_states = [copy.deepcopy(fresh_head_state) for _ in range(config.num_users)]
                        user_hidden_states = [None for _ in range(config.num_users)]
                        reset_hidden_next_round = False
                        reset_hidden_next_round_reason = None
                    else:
                        if gru_head_randomize_on_switch:
                            user_head_states = _randomize_user_heads_from_fresh_model(
                                num_users=config.num_users,
                                observation_dim=obs_dim,
                                output_dim=output_dim,
                                enable_pl_factorization=enable_gru_pl_factorization,
                                log_pl_min=gru_log_pl_min,
                                log_pl_max=gru_log_pl_max,
                            )
                        elif gru_head_reset_to_group_mean_on_switch:
                            user_head_states = _replace_heads_with_group_means(
                                user_head_states,
                                gru_group_assignment,
                            )
                    gru_group_mode = "grouped"
                    gru_group_switch_round = int(round_idx)
                    if (not gru_restart_training_after_switch) and gru_reset_hidden_on_group_switch:
                        reset_hidden_next_round = True
                        reset_hidden_next_round_reason = "group_switch"
                    logger.info(
                        "Switching GRU grouping mode -> grouped "
                        f"at round {round_idx}; next-round low={grouping_result.group_low.tolist()}, "
                        f"high={grouping_result.group_high.tolist()}, "
                        f"iters={grouping_result.iterations}, converged={grouping_result.converged}"
                    )
                    if gru_restart_training_after_switch:
                        logger.info(
                            "GRU grouped-restart debug: reinitialized both group GRU models, "
                            "reset all user heads, and cleared all persistent hidden states at switch."
                        )
                    elif gru_head_randomize_on_switch:
                        logger.info("GRU grouped-head debug: randomized all user head states at switch.")
                    elif gru_head_reset_to_group_mean_on_switch:
                        logger.info("GRU grouped-head debug: reset user head states to group means at switch.")
                    if gru_log_group_head_dispersion:
                        head_dispersion_stats = _group_head_dispersion_stats(
                            user_head_states,
                            gru_group_assignment,
                            eps=gru_group_eps,
                        )
                        logger.info(
                            "GRU grouped-head dispersion after switch -> "
                            + ", ".join(
                                [
                                    f"G{stat['group_idx'] + 1}(n={stat['size']},"
                                    f"mean_rel={stat['mean_rel']:.4e},max_rel={stat['max_rel']:.4e})"
                                    for stat in head_dispersion_stats
                                ]
                            )
                        )
            else:
                if gru_group_freeze_effective:
                    logger.info(
                        "GRU grouping frozen after switch; "
                        f"keeping users low={np.flatnonzero(gru_group_assignment == 0).astype(np.int64).tolist()}, "
                        f"high={np.flatnonzero(gru_group_assignment == 1).astype(np.int64).tolist()}"
                    )
                    grouping_result = None
                else:
                    grouping_result = _run_gru_grouping_optimizer(
                        beta_hat=beta_hat_round,
                        d_hat=d_hat_round,
                        beta_ema=gru_group_beta_ema,
                        d_ema=gru_group_d_ema,
                        H_BR=H_BR,
                        h_ru_est=h_RUs_est,
                        f_single=f_beam,
                        group_beams=f_beam_gru_groups,
                        group_assignment=current_gru_group_assignment,
                        cfg=gru_grouping_cfg,
                        prev_x_hard=gru_group_prev_x_hard,
                        prev_x_soft=gru_group_prev_x_soft,
                        prev_mu=gru_group_prev_mu,
                    )
                    gru_group_assignment = np.asarray(grouping_result.x_hard, dtype=np.int64)
                    gru_group_prev_x_hard = np.asarray(grouping_result.x_hard, dtype=np.float64)
                    gru_group_prev_x_soft = np.asarray(grouping_result.x_soft, dtype=np.float64)
                    gru_group_prev_mu = np.asarray(grouping_result.mu, dtype=np.float64)
                    logger.info(
                        "GRU regrouping for next round -> "
                        f"low={grouping_result.group_low.tolist()}, "
                        f"high={grouping_result.group_high.tolist()}, "
                        f"iters={grouping_result.iterations}, converged={grouping_result.converged}"
                    )

        # Advance channels to the precomputed next-round state.
        h_RUs = h_RUs_next
        h_BUs = h_BUs_next
        alpha_used = alpha_used_next
        doppler_used = ru_evolver.current_doppler_vector()
        for alpha_line in _format_speed_doppler_alpha_logs(
                ru_evolver.speed_magnitudes,
                doppler_used,
                alpha_used,
                chunk_size=5,
                label="RU mobility speed/fD/alpha",
        ):
            logger.info(alpha_line)
        logger.info(
            f"RU mobility alpha summary: min={alpha_used.min():.4f}, max={alpha_used.max():.4f}"
        )
        if use_uplink_target and (alpha_used_uplink is not None):
            for alpha_tau_line in _format_speed_doppler_alpha_logs(
                    ru_evolver.speed_magnitudes,
                    doppler_used,
                    alpha_used_uplink,
                    chunk_size=5,
                    label="RU uplink speed/fD/alpha_tau",
            ):
                logger.info(alpha_tau_line)
            logger.info(
                f"RU uplink alpha_tau summary: min={alpha_used_uplink.min():.4f}, max={alpha_used_uplink.max():.4f}"
            )
        user_time_steps = np.full((config.num_users,), float(round_idx), dtype=np.float64)
        current_positions = ru_evolver.positions_at_steps(user_time_steps)
        current_ris_dist = ru_evolver.ris_distances(current_positions)
        logger.info(
            f"RIS distance after round {round_idx} (min/mean/max)=("
            f"{float(current_ris_dist.min()):.3f}/{float(current_ris_dist.mean()):.3f}/{float(current_ris_dist.max()):.3f})"
        )
        if direct_on == 1:
            current_direct_dist = ru_evolver.direct_distances(current_positions)
            logger.info(
                f"BS-user distance after round {round_idx} (min/mean/max)=("
                f"{float(current_direct_dist.min()):.3f}/{float(current_direct_dist.mean()):.3f}/{float(current_direct_dist.max()):.3f})"
            )

    logger.info("Training process completed.")
    # Clean up logger
    logger.close()


if __name__ == "__main__":
    main()
