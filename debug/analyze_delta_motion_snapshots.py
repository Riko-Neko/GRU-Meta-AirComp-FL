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
        raise FileNotFoundError(f"No delta-motion snapshot run directory found under: {abs_path}")
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


def analyze(common: dict) -> tuple[dict, dict]:
    moving_mask = np.asarray(common["moving_user_mask"], dtype=bool)
    true_delta = np.asarray(common["true_delta_rel_norm_k"], dtype=np.float64)
    gru_delta = np.asarray(common["gru_delta_proxy_rel_norm_k"], dtype=np.float64)
    speed = np.asarray(common["speed_mps"], dtype=np.float64)
    doppler = np.asarray(common["doppler_hz"], dtype=np.float64)
    alpha_tau = np.asarray(common["alpha_tau"], dtype=np.float64)
    one_minus_alpha_tau = 1.0 - np.abs(alpha_tau)
    tau_weight = np.asarray(common["tau_loss_weight"], dtype=np.float64)
    ris_pl_tau = np.asarray(common["ris_pathloss_tau"], dtype=np.float64)

    log_true_delta = _safe_log10(true_delta)
    log_gru_delta = _safe_log10(gru_delta)
    log_speed = _safe_log10(speed + 1e-6)
    log_doppler = _safe_log10(doppler + 1e-6)
    log_ris_pl_tau = _safe_log10(ris_pl_tau)

    static_mask = ~moving_mask
    summary = {
        "num_rounds": int(true_delta.shape[0]),
        "num_users": int(true_delta.shape[1]),
        "true_delta_rel_corr_log10_speed": _pearson_corr(log_true_delta, log_speed),
        "true_delta_rel_corr_log10_doppler": _pearson_corr(log_true_delta, log_doppler),
        "true_delta_rel_corr_alpha_tau": _pearson_corr(true_delta, alpha_tau),
        "true_delta_rel_corr_one_minus_abs_alpha_tau": _pearson_corr(true_delta, one_minus_alpha_tau),
        "true_delta_rel_corr_tau_loss_weight": _pearson_corr(true_delta, tau_weight),
        "true_delta_rel_corr_log10_ris_pathloss_tau": _pearson_corr(log_true_delta, log_ris_pl_tau),
        "gru_delta_proxy_rel_corr_true_delta_rel": _pearson_corr(log_gru_delta, log_true_delta),
        "gru_delta_proxy_rel_corr_log10_speed": _pearson_corr(log_gru_delta, log_speed),
        "gru_delta_proxy_rel_corr_log10_doppler": _pearson_corr(log_gru_delta, log_doppler),
        "gru_delta_proxy_rel_corr_alpha_tau": _pearson_corr(gru_delta, alpha_tau),
        "gru_delta_proxy_rel_corr_one_minus_abs_alpha_tau": _pearson_corr(gru_delta, one_minus_alpha_tau),
        "true_delta_rel_static_mean": float(np.mean(true_delta[:, static_mask])) if np.any(static_mask) else float("nan"),
        "true_delta_rel_moving_mean": float(np.mean(true_delta[:, moving_mask])) if np.any(moving_mask) else float("nan"),
        "gru_delta_proxy_rel_static_mean": float(np.nanmean(gru_delta[:, static_mask])) if np.any(static_mask) else float("nan"),
        "gru_delta_proxy_rel_moving_mean": float(np.nanmean(gru_delta[:, moving_mask])) if np.any(moving_mask) else float("nan"),
        "true_delta_regression": _standardized_regression(
            np.stack(
                [
                    log_speed.reshape(-1),
                    log_doppler.reshape(-1),
                    alpha_tau.reshape(-1),
                    one_minus_alpha_tau.reshape(-1),
                    log_ris_pl_tau.reshape(-1),
                    tau_weight.reshape(-1),
                ],
                axis=1,
            ),
            log_true_delta.reshape(-1),
            feature_names=(
                "log_speed",
                "log_doppler",
                "alpha_tau",
                "one_minus_abs_alpha_tau",
                "log_ris_pathloss_tau",
                "tau_loss_weight",
            ),
        ),
        "gru_delta_proxy_regression": _standardized_regression(
            np.stack(
                [
                    log_speed.reshape(-1),
                    log_doppler.reshape(-1),
                    alpha_tau.reshape(-1),
                    one_minus_alpha_tau.reshape(-1),
                    log_ris_pl_tau.reshape(-1),
                    tau_weight.reshape(-1),
                ],
                axis=1,
            ),
            log_gru_delta.reshape(-1),
            feature_names=(
                "log_speed",
                "log_doppler",
                "alpha_tau",
                "one_minus_abs_alpha_tau",
                "log_ris_pathloss_tau",
                "tau_loss_weight",
            ),
        ),
    }

    arrays = {
        "round_idx": common["round_idx"],
        "user_id": common["user_id"],
        "moving_user_mask": moving_mask,
        "speed_mps": speed,
        "doppler_hz": doppler,
        "alpha_tau": alpha_tau,
        "one_minus_abs_alpha_tau": one_minus_alpha_tau,
        "tau_loss_weight": tau_weight,
        "ris_pathloss_tau": ris_pl_tau,
        "true_delta_norm_k": np.asarray(common["true_delta_norm_k"], dtype=np.float64),
        "true_delta_rel_norm_k": true_delta,
        "gru_delta_proxy_norm_k": np.asarray(common["gru_delta_proxy_norm_k"], dtype=np.float64),
        "gru_delta_proxy_rel_norm_k": gru_delta,
        "log10_true_delta_rel_norm_k": log_true_delta,
        "log10_gru_delta_proxy_rel_norm_k": log_gru_delta,
        "log10_speed_mps": log_speed,
        "log10_doppler_hz": log_doppler,
        "log10_ris_pathloss_tau": log_ris_pl_tau,
    }
    return summary, arrays


def main():
    parser = argparse.ArgumentParser(description="Analyze delta-motion proxy debug snapshots.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="debug/delta_motion_snapshots",
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
        "alpha_tau": _stack_key(round_paths, "alpha_tau"),
        "tau_loss_weight": _stack_key(round_paths, "tau_loss_weight"),
        "ris_pathloss_tau": _stack_key(round_paths, "ris_pathloss_tau"),
        "true_delta_norm_k": _stack_key(round_paths, "true_delta_norm_k"),
        "true_delta_rel_norm_k": _stack_key(round_paths, "true_delta_rel_norm_k"),
        "gru_delta_proxy_norm_k": _stack_key(round_paths, "gru_delta_proxy_norm_k"),
        "gru_delta_proxy_rel_norm_k": _stack_key(round_paths, "gru_delta_proxy_rel_norm_k"),
    }

    summary, arrays = analyze(common)
    analysis_dir = os.path.join(run_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    with open(os.path.join(analysis_dir, "summary.json"), "w", encoding="utf-8") as fp:
        json.dump({"run_dir": run_dir, "meta": run_meta, "summary": summary}, fp, indent=2, ensure_ascii=False)
    np.savez_compressed(os.path.join(analysis_dir, "analysis_arrays.npz"), **arrays)

    moving_mask_2d = np.repeat(arrays["moving_user_mask"][None, :], arrays["true_delta_rel_norm_k"].shape[0], axis=0)
    _plot_heatmap(
        arrays["true_delta_rel_norm_k"],
        title="true delta relative norm",
        out_path=os.path.join(analysis_dir, "true_delta_rel_norm_heatmap.png"),
        round_idx=arrays["round_idx"],
        user_id=arrays["user_id"],
        log10_scale=True,
    )
    _plot_heatmap(
        arrays["gru_delta_proxy_rel_norm_k"],
        title="GRU delta-proxy relative norm",
        out_path=os.path.join(analysis_dir, "gru_delta_proxy_rel_norm_heatmap.png"),
        round_idx=arrays["round_idx"],
        user_id=arrays["user_id"],
        log10_scale=True,
    )
    _plot_scatter(
        arrays["log10_speed_mps"],
        arrays["log10_true_delta_rel_norm_k"],
        moving_mask_2d,
        out_path=os.path.join(analysis_dir, "speed_vs_true_delta_rel.png"),
        title="log10 speed vs log10 true delta relative norm",
        xlabel="log10(speed m/s)",
        ylabel="log10(true delta relative norm)",
    )
    _plot_scatter(
        arrays["alpha_tau"],
        arrays["true_delta_rel_norm_k"],
        moving_mask_2d,
        out_path=os.path.join(analysis_dir, "alpha_tau_vs_true_delta_rel.png"),
        title="alpha_tau vs true delta relative norm",
        xlabel="alpha_tau",
        ylabel="true delta relative norm",
    )
    _plot_scatter(
        arrays["log10_true_delta_rel_norm_k"],
        arrays["log10_gru_delta_proxy_rel_norm_k"],
        moving_mask_2d,
        out_path=os.path.join(analysis_dir, "true_delta_vs_gru_delta_proxy.png"),
        title="log10 true delta relative norm vs log10 GRU delta-proxy",
        xlabel="log10(true delta relative norm)",
        ylabel="log10(GRU delta-proxy relative norm)",
    )

    print(f"Delta-motion snapshot analysis saved to: {analysis_dir}")
    print(
        "corr(true_delta_rel, one_minus_abs_alpha_tau)="
        f"{summary['true_delta_rel_corr_one_minus_abs_alpha_tau']:.4f}, "
        "corr(gru_delta_proxy_rel, true_delta_rel)="
        f"{summary['gru_delta_proxy_rel_corr_true_delta_rel']:.4f}"
    )


if __name__ == "__main__":
    main()
