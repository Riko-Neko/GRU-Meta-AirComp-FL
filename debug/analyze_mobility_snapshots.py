import argparse
import json
import os
from glob import glob

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
    "lmmse": "LMMSE",
    "oracle": "Oracle-true",
}


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
        "coefficients": {name: float(beta[idx + 1]) for idx, name in enumerate(feature_names)},
    }


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
        raise FileNotFoundError(f"No mobility snapshot run directory found under: {abs_path}")
    if latest:
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return os.path.abspath(candidates[0])
    raise FileNotFoundError(
        f"Path does not contain snapshots directly: {abs_path}. "
        "Pass a concrete run directory or use --latest."
    )


def _stack_key(round_paths, key, dtype=np.float64):
    values = []
    for path in round_paths:
        with np.load(path, allow_pickle=False) as data:
            if key not in data:
                raise KeyError(f"Missing required key '{key}' in {path}")
            values.append(np.asarray(data[key], dtype=dtype))
    return np.stack(values, axis=0)


def _load_branch_nmse(round_paths, key):
    values = []
    rounds = []
    for path in round_paths:
        with np.load(path, allow_pickle=False) as data:
            if key not in data:
                return None
            values.append(np.asarray(data[key], dtype=np.float64))
            rounds.append(int(_load_npz_scalar(data, "round_idx", 0)))
    return {"round_idx": np.asarray(rounds, dtype=np.int32), "nmse_k": np.stack(values, axis=0)}


def _plot_heatmap(matrix: np.ndarray, title: str, out_path: str, *, round_idx, user_id, log10_scale: bool = False):
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


def _plot_scatter(x, y, moving_mask, out_path: str, title: str, xlabel: str, ylabel: str):
    if not HAS_MATPLOTLIB:
        return
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    moving = np.asarray(moving_mask, dtype=bool).reshape(-1)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    moving = moving[mask]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(x_arr[~moving], y_arr[~moving], s=18, alpha=0.7, label="static", c="#1f77b4", linewidths=0.0)
    if np.any(moving):
        ax.scatter(x_arr[moving], y_arr[moving], s=22, alpha=0.8, label="moving", c="#d62728", linewidths=0.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def analyze_branch(common: dict, branch_nmse: dict) -> tuple[dict, dict]:
    nmse_k = np.asarray(branch_nmse["nmse_k"], dtype=np.float64)
    speed_k = np.asarray(common["speed_mps"], dtype=np.float64)
    doppler_k = np.asarray(common["doppler_hz"], dtype=np.float64)
    alpha_delta = np.asarray(common["alpha_delta"], dtype=np.float64)
    alpha_tau = np.asarray(common["alpha_tau"], dtype=np.float64)
    ris_pl_tau = np.asarray(common["ris_pathloss_tau"], dtype=np.float64)
    direct_pl_tau = np.asarray(common["direct_pathloss_tau"], dtype=np.float64)
    tau_weight = np.asarray(common["tau_loss_weight"], dtype=np.float64)
    moving_mask = np.asarray(common["moving_user_mask"], dtype=bool)

    log_nmse = _safe_log10(nmse_k)
    log_speed = _safe_log10(speed_k + 1e-6)
    log_doppler = _safe_log10(doppler_k + 1e-6)
    log_ris_pl_tau = _safe_log10(ris_pl_tau)
    log_direct_pl_tau = _safe_log10(np.maximum(np.where(np.isfinite(direct_pl_tau), direct_pl_tau, np.nan), 1e-12))

    static_mask = ~moving_mask
    moving_nmse_mean = float(np.mean(nmse_k[:, moving_mask])) if np.any(moving_mask) else float("nan")
    static_nmse_mean = float(np.mean(nmse_k[:, static_mask])) if np.any(static_mask) else float("nan")
    moving_nmse_median = float(np.median(nmse_k[:, moving_mask])) if np.any(moving_mask) else float("nan")
    static_nmse_median = float(np.median(nmse_k[:, static_mask])) if np.any(static_mask) else float("nan")

    feature_x = np.stack(
        [
            log_speed.reshape(-1),
            log_doppler.reshape(-1),
            alpha_tau.reshape(-1),
            log_ris_pl_tau.reshape(-1),
            tau_weight.reshape(-1),
        ],
        axis=1,
    )
    regression = _standardized_regression(
        feature_x,
        log_nmse.reshape(-1),
        feature_names=("log_speed", "log_doppler", "alpha_tau", "log_ris_pathloss_tau", "tau_loss_weight"),
    )

    summary = {
        "num_rounds": int(nmse_k.shape[0]),
        "num_users": int(nmse_k.shape[1]),
        "overall_corr_alpha_tau_vs_log10_nmse": _pearson_corr(alpha_tau, log_nmse),
        "overall_corr_log10_speed_vs_log10_nmse": _pearson_corr(log_speed, log_nmse),
        "overall_corr_log10_doppler_vs_log10_nmse": _pearson_corr(log_doppler, log_nmse),
        "overall_corr_log10_ris_pathloss_tau_vs_log10_nmse": _pearson_corr(log_ris_pl_tau, log_nmse),
        "overall_corr_tau_loss_weight_vs_log10_nmse": _pearson_corr(tau_weight, log_nmse),
        "moving_nmse_mean": moving_nmse_mean,
        "static_nmse_mean": static_nmse_mean,
        "moving_nmse_median": moving_nmse_median,
        "static_nmse_median": static_nmse_median,
        "moving_over_static_mean_ratio": float(moving_nmse_mean / static_nmse_mean) if np.isfinite(moving_nmse_mean) and np.isfinite(static_nmse_mean) and static_nmse_mean > 0 else float("nan"),
        "moving_over_static_median_ratio": float(moving_nmse_median / static_nmse_median) if np.isfinite(moving_nmse_median) and np.isfinite(static_nmse_median) and static_nmse_median > 0 else float("nan"),
        "standardized_regression_log10_nmse_on_mobility_features": regression,
    }

    arrays = {
        "round_idx": common["round_idx"],
        "user_id": common["user_id"],
        "moving_user_mask": moving_mask,
        "speed_mps": speed_k,
        "doppler_hz": doppler_k,
        "alpha_delta": alpha_delta,
        "alpha_tau": alpha_tau,
        "tau_loss_weight": tau_weight,
        "ris_pathloss_tau": ris_pl_tau,
        "direct_pathloss_tau": direct_pl_tau,
        "uplink_true_nmse_k": nmse_k,
        "log10_uplink_true_nmse_k": log_nmse,
        "log10_speed_mps": log_speed,
        "log10_doppler_hz": log_doppler,
        "log10_ris_pathloss_tau": log_ris_pl_tau,
        "log10_direct_pathloss_tau": log_direct_pl_tau,
    }
    return summary, arrays


def _save_branch_outputs(run_analysis_dir: str, branch_prefix: str, branch_summary: dict, branch_arrays: dict):
    branch_dir = os.path.join(run_analysis_dir, branch_prefix)
    os.makedirs(branch_dir, exist_ok=True)
    with open(os.path.join(branch_dir, "summary.json"), "w", encoding="utf-8") as fp:
        json.dump(branch_summary, fp, indent=2, ensure_ascii=False)
    np.savez_compressed(os.path.join(branch_dir, "analysis_arrays.npz"), **branch_arrays)

    _plot_heatmap(
        branch_arrays["uplink_true_nmse_k"],
        title=f"{BRANCH_LABELS[branch_prefix]} uplink_true_NMSE_k",
        out_path=os.path.join(branch_dir, "uplink_true_nmse_heatmap.png"),
        round_idx=branch_arrays["round_idx"],
        user_id=branch_arrays["user_id"],
        log10_scale=True,
    )
    _plot_heatmap(
        branch_arrays["alpha_tau"],
        title="alpha_tau",
        out_path=os.path.join(branch_dir, "alpha_tau_heatmap.png"),
        round_idx=branch_arrays["round_idx"],
        user_id=branch_arrays["user_id"],
        log10_scale=False,
    )
    _plot_heatmap(
        branch_arrays["ris_pathloss_tau"],
        title="RIS pathloss at tau",
        out_path=os.path.join(branch_dir, "ris_pathloss_tau_heatmap.png"),
        round_idx=branch_arrays["round_idx"],
        user_id=branch_arrays["user_id"],
        log10_scale=True,
    )
    moving_mask_2d = np.repeat(branch_arrays["moving_user_mask"][None, :], branch_arrays["uplink_true_nmse_k"].shape[0], axis=0)
    _plot_scatter(
        branch_arrays["alpha_tau"],
        branch_arrays["log10_uplink_true_nmse_k"],
        moving_mask_2d,
        out_path=os.path.join(branch_dir, "alpha_tau_vs_uplink_true_nmse.png"),
        title=f"{BRANCH_LABELS[branch_prefix]} alpha_tau vs log10 uplink_true_NMSE",
        xlabel="alpha_tau",
        ylabel="log10(uplink_true_NMSE)",
    )
    _plot_scatter(
        branch_arrays["log10_speed_mps"],
        branch_arrays["log10_uplink_true_nmse_k"],
        moving_mask_2d,
        out_path=os.path.join(branch_dir, "speed_vs_uplink_true_nmse.png"),
        title=f"{BRANCH_LABELS[branch_prefix]} log10 speed vs log10 uplink_true_NMSE",
        xlabel="log10(speed m/s)",
        ylabel="log10(uplink_true_NMSE)",
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze per-user mobility/pathloss debug snapshots.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="debug/mobility_snapshots",
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

    common = {
        "round_idx": _stack_key(round_paths, "round_idx", dtype=np.int32).reshape(-1),
        "user_id": _stack_key(round_paths, "user_id", dtype=np.int32)[0],
        "moving_user_mask": _stack_key(round_paths, "moving_user_mask", dtype=np.int8)[0].astype(bool),
        "speed_mps": _stack_key(round_paths, "speed_mps"),
        "doppler_hz": _stack_key(round_paths, "doppler_hz"),
        "alpha_delta": _stack_key(round_paths, "alpha_delta"),
        "alpha_tau": _stack_key(round_paths, "alpha_tau"),
        "tau_loss_weight": _stack_key(round_paths, "tau_loss_weight"),
        "ris_pathloss_tau": _stack_key(round_paths, "ris_pathloss_tau"),
        "direct_pathloss_tau": _stack_key(round_paths, "direct_pathloss_tau"),
    }

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
        branch_series = _load_branch_nmse(round_paths, f"{branch_prefix}_uplink_true_nmse_k")
        if branch_series is None:
            continue
        branch_summary, branch_arrays = analyze_branch(common, branch_series)
        top_summary["available_branches"].append(branch_prefix)
        top_summary["branches"][branch_prefix] = branch_summary
        _save_branch_outputs(analysis_dir, branch_prefix, branch_summary, branch_arrays)

    with open(os.path.join(analysis_dir, "summary.json"), "w", encoding="utf-8") as fp:
        json.dump(top_summary, fp, indent=2, ensure_ascii=False)

    print(f"Mobility snapshot analysis saved to: {analysis_dir}")
    for branch_prefix in top_summary["available_branches"]:
        branch_summary = top_summary["branches"][branch_prefix]
        print(
            f"{BRANCH_LABELS[branch_prefix]}: "
            f"corr(alpha_tau, log10 NMSE)={branch_summary['overall_corr_alpha_tau_vs_log10_nmse']:.4f}, "
            f"moving/static mean ratio={branch_summary['moving_over_static_mean_ratio']:.4f}"
        )


if __name__ == "__main__":
    main()
