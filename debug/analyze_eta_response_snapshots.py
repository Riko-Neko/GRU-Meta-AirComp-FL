import argparse
import json
import os
from glob import glob
from typing import Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


BRANCH_LABELS = {
    "gru": "GRU",
    "cnn_arch": "CNN-arch",
    "cnn_base": "CNN-base",
}

BRANCH_REQUIRED_KEYS = (
    "eta_k",
    "h_eff_abs2_k",
    "v_k",
    "uplink_true_nmse_k",
)


def _safe_log10(x, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return np.log10(np.maximum(arr, float(eps)))


def _pearson_corr(x, y) -> float:
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(mask.sum()) < 2:
        return float("nan")
    x_valid = x_arr[mask]
    y_valid = y_arr[mask]
    x_centered = x_valid - x_valid.mean()
    y_centered = y_valid - y_valid.mean()
    denom = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))
    if float(denom) <= 0.0:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denom)


def _standardize_columns(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std <= float(eps), 1.0, std)
    return (x - mean) / std


def _standardized_regression(x: np.ndarray, y: np.ndarray, feature_names) -> dict:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(y_arr)
    for col_idx in range(x_arr.shape[1]):
        valid &= np.isfinite(x_arr[:, col_idx])
    if int(valid.sum()) < (x_arr.shape[1] + 1):
        return {
            "num_samples": int(valid.sum()),
            "intercept": float("nan"),
            "coefficients": {name: float("nan") for name in feature_names},
        }

    x_valid = _standardize_columns(x_arr[valid])
    y_valid = y_arr[valid]
    y_std = float(y_valid.std())
    if y_std <= 1e-12:
        return {
            "num_samples": int(valid.sum()),
            "intercept": 0.0,
            "coefficients": {name: 0.0 for name in feature_names},
        }
    y_valid = (y_valid - y_valid.mean()) / y_std

    design = np.concatenate([np.ones((x_valid.shape[0], 1), dtype=np.float64), x_valid], axis=1)
    beta, _, _, _ = np.linalg.lstsq(design, y_valid, rcond=None)
    return {
        "num_samples": int(valid.sum()),
        "intercept": float(beta[0]),
        "coefficients": {
            name: float(beta[idx + 1]) for idx, name in enumerate(feature_names)
        },
    }


def _quartile_response(eta_k: np.ndarray, nmse_k: np.ndarray) -> list:
    eta_flat = np.asarray(eta_k, dtype=np.float64).reshape(-1)
    nmse_flat = np.asarray(nmse_k, dtype=np.float64).reshape(-1)
    valid = np.isfinite(eta_flat) & np.isfinite(nmse_flat)
    eta_valid = eta_flat[valid]
    nmse_valid = nmse_flat[valid]
    if eta_valid.size == 0:
        return []

    q1, q2, q3 = np.quantile(eta_valid, [0.25, 0.5, 0.75])
    bounds = [(-np.inf, q1), (q1, q2), (q2, q3), (q3, np.inf)]
    result = []
    for idx, (lo, hi) in enumerate(bounds, start=1):
        if idx == 1:
            mask = eta_valid <= hi
        elif idx == 4:
            mask = eta_valid > lo
        else:
            mask = (eta_valid > lo) & (eta_valid <= hi)
        if int(mask.sum()) == 0:
            result.append(
                {
                    "quartile": idx,
                    "count": 0,
                    "eta_mean": float("nan"),
                    "eta_median": float("nan"),
                    "nmse_mean": float("nan"),
                    "nmse_median": float("nan"),
                }
            )
            continue
        result.append(
            {
                "quartile": idx,
                "count": int(mask.sum()),
                "eta_mean": float(np.mean(eta_valid[mask])),
                "eta_median": float(np.median(eta_valid[mask])),
                "nmse_mean": float(np.mean(nmse_valid[mask])),
                "nmse_median": float(np.median(nmse_valid[mask])),
            }
        )
    return result


def _load_npz_scalar(data, key, default=None):
    if key not in data:
        return default
    value = np.asarray(data[key])
    if value.shape == ():
        return value.item()
    if value.size == 1:
        return value.reshape(()).item()
    return value


def _discover_run_dir(run_dir: str, latest: bool = False) -> str:
    abs_path = os.path.abspath(run_dir)
    if glob(os.path.join(abs_path, "round_*.npz")):
        return abs_path
    candidates = [
        path for path in glob(os.path.join(abs_path, "*"))
        if os.path.isdir(path) and glob(os.path.join(path, "round_*.npz"))
    ]
    if not candidates:
        raise FileNotFoundError(f"No snapshot run directory found under: {abs_path}")
    if latest:
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return os.path.abspath(candidates[0])
    raise FileNotFoundError(
        f"Path does not contain snapshots directly: {abs_path}. "
        "Pass a concrete run directory or use --latest."
    )


def _load_branch_series(round_paths, branch_prefix: str):
    required_keys = [f"{branch_prefix}_{name}" for name in BRANCH_REQUIRED_KEYS]
    rounds = []
    user_ids = None
    k_stack = []
    k_raw_stack = []
    eta_stack = []
    abs2_stack = []
    v_stack = []
    nmse_stack = []
    for path in round_paths:
        with np.load(path, allow_pickle=False) as data:
            if not all(key in data for key in required_keys):
                continue
            rounds.append(int(_load_npz_scalar(data, "round_idx", 0)))
            if user_ids is None:
                user_ids = np.asarray(data["user_id"], dtype=np.int32)
            k_stack.append(np.asarray(data["K_k"], dtype=np.float64))
            if "K_raw" in data:
                k_raw_stack.append(np.asarray(data["K_raw"], dtype=np.float64))
            eta_stack.append(np.asarray(data[f"{branch_prefix}_eta_k"], dtype=np.float64))
            abs2_stack.append(np.asarray(data[f"{branch_prefix}_h_eff_abs2_k"], dtype=np.float64))
            v_stack.append(np.asarray(data[f"{branch_prefix}_v_k"], dtype=np.float64))
            nmse_stack.append(np.asarray(data[f"{branch_prefix}_uplink_true_nmse_k"], dtype=np.float64))
    if not rounds:
        return None
    return {
        "round_idx": np.asarray(rounds, dtype=np.int32),
        "user_id": user_ids,
        "K_k": np.stack(k_stack, axis=0),
        "K_raw": np.stack(k_raw_stack, axis=0) if k_raw_stack else None,
        "eta_k": np.stack(eta_stack, axis=0),
        "h_eff_abs2_k": np.stack(abs2_stack, axis=0),
        "v_k": np.stack(v_stack, axis=0),
        "uplink_true_nmse_k": np.stack(nmse_stack, axis=0),
    }


def _plot_heatmap(matrix: np.ndarray, title: str, out_path: str, *, round_idx, user_id, log10_scale: bool = True):
    if not HAS_MATPLOTLIB:
        return
    mat = _safe_log10(matrix) if log10_scale else np.asarray(matrix, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(mat, aspect="auto", origin="lower", interpolation="nearest", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("User")
    ax.set_ylabel("Round index")
    ax.set_xticks(np.arange(len(user_id)))
    ax.set_xticklabels([str(int(uid)) for uid in user_id], rotation=0)
    if len(round_idx) <= 20:
        ax.set_yticks(np.arange(len(round_idx)))
        ax.set_yticklabels([str(int(r)) for r in round_idx])
    fig.colorbar(im, ax=ax, label="log10(value)" if log10_scale else "value")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_scatter(log_eta: np.ndarray, log_nmse: np.ndarray, user_ids, out_path: str, title: str):
    if not HAS_MATPLOTLIB:
        return
    rounds, users = log_eta.shape
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    colors = np.repeat(np.asarray(user_ids, dtype=np.int32)[None, :], rounds, axis=0).reshape(-1)
    scatter = ax.scatter(
        log_eta.reshape(-1),
        log_nmse.reshape(-1),
        c=colors,
        cmap="tab20",
        s=16,
        alpha=0.75,
        linewidths=0.0,
    )
    ax.set_title(title)
    ax.set_xlabel("log10(eta_k)")
    ax.set_ylabel("log10(uplink_true_NMSE_k)")
    ax.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="User")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_bottleneck_bar(counts: np.ndarray, user_ids, out_path: str, title: str):
    if not HAS_MATPLOTLIB:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(np.arange(len(user_ids)), counts, color="tab:red", edgecolor="black", linewidth=0.4)
    ax.set_title(title)
    ax.set_xlabel("User")
    ax.set_ylabel("Times user is min-eta bottleneck")
    ax.set_xticks(np.arange(len(user_ids)))
    ax.set_xticklabels([str(int(uid)) for uid in user_ids])
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def analyze_branch(series: dict) -> Tuple[dict, dict]:
    eta_k = np.asarray(series["eta_k"], dtype=np.float64)
    nmse_k = np.asarray(series["uplink_true_nmse_k"], dtype=np.float64)
    abs2_k = np.asarray(series["h_eff_abs2_k"], dtype=np.float64)
    k_k = np.asarray(series["K_k"], dtype=np.float64)
    v_k = np.asarray(series["v_k"], dtype=np.float64)
    round_idx = np.asarray(series["round_idx"], dtype=np.int32)
    user_id = np.asarray(series["user_id"], dtype=np.int32)

    log_eta = _safe_log10(eta_k)
    log_nmse = _safe_log10(nmse_k)
    log_abs2 = _safe_log10(abs2_k)
    log_k = _safe_log10(k_k)
    log_v = _safe_log10(v_k)

    per_user_corr = np.asarray(
        [_pearson_corr(log_eta[:, idx], log_nmse[:, idx]) for idx in range(log_eta.shape[1])],
        dtype=np.float64,
    )
    per_round_corr = np.asarray(
        [_pearson_corr(log_eta[idx, :], log_nmse[idx, :]) for idx in range(log_eta.shape[0])],
        dtype=np.float64,
    )

    bottleneck_user = np.argmin(eta_k, axis=1)
    worst_nmse_user = np.argmax(nmse_k, axis=1)
    bottleneck_counts = np.bincount(bottleneck_user, minlength=eta_k.shape[1]).astype(np.int32)
    worst_nmse_counts = np.bincount(worst_nmse_user, minlength=eta_k.shape[1]).astype(np.int32)
    best_eta_per_user = np.maximum(eta_k.max(axis=0), 1e-12)
    eta_to_user_best = eta_k / best_eta_per_user.reshape(1, -1)

    component_x = np.stack(
        [
            log_abs2.reshape(-1),
            (-2.0 * log_k).reshape(-1),
            (-log_v).reshape(-1),
        ],
        axis=1,
    )
    component_regression = _standardized_regression(
        component_x,
        log_nmse.reshape(-1),
        feature_names=("log_h_eff_abs2", "minus_2log_K_k", "minus_log_v_k"),
    )

    summary = {
        "num_rounds": int(eta_k.shape[0]),
        "num_users": int(eta_k.shape[1]),
        "overall_corr_log10_eta_vs_log10_nmse": _pearson_corr(log_eta, log_nmse),
        "overall_corr_log10_h_eff_abs2_vs_log10_nmse": _pearson_corr(log_abs2, log_nmse),
        "overall_corr_log10_K_k_vs_log10_nmse": _pearson_corr(log_k, log_nmse),
        "overall_corr_log10_v_k_vs_log10_nmse": _pearson_corr(log_v, log_nmse),
        "per_user_corr_log10_eta_vs_log10_nmse": per_user_corr.tolist(),
        "per_round_corr_log10_eta_vs_log10_nmse": per_round_corr.tolist(),
        "bottleneck_counts_by_user": bottleneck_counts.tolist(),
        "worst_nmse_counts_by_user": worst_nmse_counts.tolist(),
        "bottleneck_matches_worst_nmse_ratio": float(np.mean(bottleneck_user == worst_nmse_user)),
        "mean_eta_to_user_best_ratio": float(np.mean(eta_to_user_best)),
        "mean_min_eta_to_best_round_min_eta_ratio": float(
            np.mean(np.min(eta_k, axis=1) / np.maximum(np.max(np.min(eta_k, axis=1)), 1e-12))
        ),
        "eta_quartile_response": _quartile_response(eta_k, nmse_k),
        "standardized_regression_log10_nmse_on_eta_components": component_regression,
    }

    arrays = {
        "round_idx": round_idx,
        "user_id": user_id,
        "eta_k": eta_k,
        "h_eff_abs2_k": abs2_k,
        "K_k": k_k,
        "v_k": v_k,
        "uplink_true_nmse_k": nmse_k,
        "log10_eta_k": log_eta,
        "log10_uplink_true_nmse_k": log_nmse,
        "per_user_corr_log10_eta_vs_log10_nmse": per_user_corr,
        "per_round_corr_log10_eta_vs_log10_nmse": per_round_corr,
        "bottleneck_counts_by_user": bottleneck_counts,
        "worst_nmse_counts_by_user": worst_nmse_counts,
        "eta_to_user_best_ratio": eta_to_user_best,
    }
    return summary, arrays


def _save_branch_outputs(run_analysis_dir: str, branch_prefix: str, branch_summary: dict, branch_arrays: dict):
    branch_dir = os.path.join(run_analysis_dir, branch_prefix)
    os.makedirs(branch_dir, exist_ok=True)
    with open(os.path.join(branch_dir, "summary.json"), "w", encoding="utf-8") as fp:
        json.dump(branch_summary, fp, indent=2, ensure_ascii=False)
    np.savez_compressed(os.path.join(branch_dir, "analysis_arrays.npz"), **branch_arrays)

    _plot_heatmap(
        branch_arrays["eta_k"],
        title=f"{BRANCH_LABELS[branch_prefix]} eta_k",
        out_path=os.path.join(branch_dir, "eta_heatmap.png"),
        round_idx=branch_arrays["round_idx"],
        user_id=branch_arrays["user_id"],
        log10_scale=True,
    )
    _plot_heatmap(
        branch_arrays["uplink_true_nmse_k"],
        title=f"{BRANCH_LABELS[branch_prefix]} uplink_true_NMSE_k",
        out_path=os.path.join(branch_dir, "uplink_true_nmse_heatmap.png"),
        round_idx=branch_arrays["round_idx"],
        user_id=branch_arrays["user_id"],
        log10_scale=True,
    )
    _plot_scatter(
        branch_arrays["log10_eta_k"],
        branch_arrays["log10_uplink_true_nmse_k"],
        branch_arrays["user_id"],
        out_path=os.path.join(branch_dir, "eta_vs_uplink_true_nmse.png"),
        title=f"{BRANCH_LABELS[branch_prefix]} log10(eta_k) vs log10(uplink_true_NMSE_k)",
    )
    _plot_bottleneck_bar(
        branch_arrays["bottleneck_counts_by_user"],
        branch_arrays["user_id"],
        out_path=os.path.join(branch_dir, "bottleneck_counts.png"),
        title=f"{BRANCH_LABELS[branch_prefix]} min-eta bottleneck frequency",
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze per-user eta response snapshots.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="debug/eta_response_snapshots",
        help="Snapshot run directory, or a root containing multiple run directories.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="If --run-dir points to a root directory, analyze the latest snapshot run.",
    )
    args = parser.parse_args()

    run_dir = _discover_run_dir(args.run_dir, latest=bool(args.latest))
    round_paths = sorted(glob(os.path.join(run_dir, "round_*.npz")))
    if not round_paths:
        raise FileNotFoundError(f"No round snapshots found in: {run_dir}")

    run_meta = {}
    meta_path = os.path.join(run_dir, "run_meta.npz")
    if os.path.exists(meta_path):
        with np.load(meta_path, allow_pickle=False) as data:
            for key in data.files:
                run_meta[key] = _load_npz_scalar(data, key)

    analysis_dir = os.path.join(run_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    top_summary = {
        "run_dir": run_dir,
        "num_snapshot_rounds": len(round_paths),
        "meta": run_meta,
        "available_branches": [],
        "branches": {},
    }

    for branch_prefix in BRANCH_LABELS:
        series = _load_branch_series(round_paths, branch_prefix)
        if series is None:
            continue
        branch_summary, branch_arrays = analyze_branch(series)
        top_summary["available_branches"].append(branch_prefix)
        top_summary["branches"][branch_prefix] = branch_summary
        _save_branch_outputs(analysis_dir, branch_prefix, branch_summary, branch_arrays)

    with open(os.path.join(analysis_dir, "summary.json"), "w", encoding="utf-8") as fp:
        json.dump(top_summary, fp, indent=2, ensure_ascii=False)

    print(f"Snapshot analysis saved to: {analysis_dir}")
    for branch_prefix in top_summary["available_branches"]:
        corr = top_summary["branches"][branch_prefix]["overall_corr_log10_eta_vs_log10_nmse"]
        overlap = top_summary["branches"][branch_prefix]["bottleneck_matches_worst_nmse_ratio"]
        print(
            f"{BRANCH_LABELS[branch_prefix]}: "
            f"corr(log10 eta, log10 uplink_true_NMSE)={corr:.4f}, "
            f"bottleneck==worst_nmse ratio={overlap:.4f}"
        )


if __name__ == "__main__":
    main()
