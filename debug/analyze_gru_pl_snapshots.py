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
        raise FileNotFoundError(f"No GRU PL snapshot run directory found under: {abs_path}")
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


def _plot_scatter(true_log_pl, pred_log_pl, moving_mask, out_path: str):
    if not HAS_MATPLOTLIB:
        return
    x = np.asarray(true_log_pl, dtype=np.float64).reshape(-1)
    y = np.asarray(pred_log_pl, dtype=np.float64).reshape(-1)
    moving = np.asarray(moving_mask, dtype=bool).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    moving = moving[valid]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(x[~moving], y[~moving], s=18, alpha=0.7, label="static", c="#1f77b4", linewidths=0.0)
    if np.any(moving):
        ax.scatter(x[moving], y[moving], s=22, alpha=0.8, label="moving", c="#d62728", linewidths=0.0)
    xy_min = float(min(np.min(x), np.min(y)))
    xy_max = float(max(np.max(x), np.max(y)))
    ax.plot([xy_min, xy_max], [xy_min, xy_max], linestyle="--", color="black", linewidth=1.0)
    ax.set_title("GRU PL: pred vs true (log10 domain)")
    ax.set_xlabel("log10(true PL_sel)")
    ax.set_ylabel("log10(pred PL_sel)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_error_vs_nmse(log_pl_abs_err, uplink_nmse, moving_mask, out_path: str):
    if not HAS_MATPLOTLIB:
        return
    x = np.asarray(log_pl_abs_err, dtype=np.float64).reshape(-1)
    y = _safe_log10(uplink_nmse).reshape(-1)
    moving = np.asarray(moving_mask, dtype=bool).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    moving = moving[valid]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(x[~moving], y[~moving], s=18, alpha=0.7, label="static", c="#1f77b4", linewidths=0.0)
    if np.any(moving):
        ax.scatter(x[moving], y[moving], s=22, alpha=0.8, label="moving", c="#d62728", linewidths=0.0)
    ax.set_title("GRU PL error vs uplink true NMSE")
    ax.set_xlabel("|log(pred PL_sel) - log(true PL_sel)|")
    ax.set_ylabel("log10(uplink_true_NMSE_k)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_user_mean(metric, user_id, out_path: str, title: str, ylabel: str):
    if not HAS_MATPLOTLIB:
        return
    values = np.asarray(metric, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(np.arange(len(user_id)), values, color="#4c78a8", edgecolor="black", linewidth=0.4)
    ax.set_title(title)
    ax.set_xlabel("User")
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(user_id)))
    ax.set_xticklabels([str(int(uid)) for uid in user_id], rotation=0)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze GRU PL debug snapshots.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="debug/gru_pl_snapshots",
        help="Concrete snapshot run directory, or a parent directory containing run subdirectories.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="If --run-dir points to a parent directory, analyze the latest run.",
    )
    args = parser.parse_args()

    run_dir = _discover_run_dir(args.run_dir, latest=args.latest)
    round_paths = sorted(glob(os.path.join(run_dir, "round_*.npz")))
    if not round_paths:
        raise FileNotFoundError(f"No round_*.npz files found under: {run_dir}")

    with np.load(round_paths[0], allow_pickle=False) as first:
        user_id = np.asarray(first["user_id"], dtype=np.int32)
    round_idx = []
    for path in round_paths:
        with np.load(path, allow_pickle=False) as data:
            round_idx.append(int(_load_npz_scalar(data, "round_idx", 0)))
    round_idx = np.asarray(round_idx, dtype=np.int32)

    moving_mask = _stack_key(round_paths, "moving_user_mask", dtype=np.int8).astype(bool)
    speed_mps = _stack_key(round_paths, "speed_mps", dtype=np.float64)
    alpha_tau = _stack_key(round_paths, "alpha_tau", dtype=np.float64)
    pl_true = _stack_key(round_paths, "gru_pl_true_sel", dtype=np.float64)
    pl_pred = _stack_key(round_paths, "gru_pl_pred_sel", dtype=np.float64)
    log_pl_true = _stack_key(round_paths, "gru_log_pl_true_sel", dtype=np.float64)
    log_pl_pred = _stack_key(round_paths, "gru_log_pl_pred_sel", dtype=np.float64)
    pl_abs_err = _stack_key(round_paths, "gru_pl_abs_err", dtype=np.float64)
    pl_rel_err = _stack_key(round_paths, "gru_pl_rel_err", dtype=np.float64)
    log_pl_abs_err = _stack_key(round_paths, "gru_log_pl_abs_err", dtype=np.float64)
    uplink_nmse = None
    try:
        uplink_nmse = _stack_key(round_paths, "gru_uplink_true_nmse_k", dtype=np.float64)
    except KeyError:
        uplink_nmse = None

    static_mask = ~moving_mask
    summary = {
        "run_dir": os.path.abspath(run_dir),
        "num_rounds": int(len(round_idx)),
        "num_users": int(len(user_id)),
        "overall_corr_log10_pred_vs_true": _pearson_corr(_safe_log10(pl_true), _safe_log10(pl_pred)),
        "overall_corr_log_pred_vs_true": _pearson_corr(log_pl_true, log_pl_pred),
        "overall_pl_mae": float(np.mean(pl_abs_err)),
        "overall_pl_mape": float(np.mean(pl_rel_err)),
        "overall_log_pl_mae": float(np.mean(log_pl_abs_err)),
        "moving_log_pl_mae": float(np.mean(log_pl_abs_err[moving_mask])) if np.any(moving_mask) else float("nan"),
        "static_log_pl_mae": float(np.mean(log_pl_abs_err[static_mask])) if np.any(static_mask) else float("nan"),
        "overall_corr_log_pl_abs_err_vs_speed": _pearson_corr(log_pl_abs_err, speed_mps),
        "overall_corr_log_pl_abs_err_vs_alpha_tau": _pearson_corr(log_pl_abs_err, alpha_tau),
    }
    if uplink_nmse is not None:
        summary["overall_corr_log_pl_abs_err_vs_log10_uplink_nmse"] = _pearson_corr(log_pl_abs_err, _safe_log10(uplink_nmse))

    summary["per_user"] = []
    for idx, uid in enumerate(user_id):
        user_summary = {
            "user_id": int(uid),
            "log_pl_mae": float(np.mean(log_pl_abs_err[:, idx])),
            "pl_mape": float(np.mean(pl_rel_err[:, idx])),
            "corr_log_pred_vs_true": _pearson_corr(log_pl_true[:, idx], log_pl_pred[:, idx]),
            "is_moving": bool(np.any(moving_mask[:, idx])),
        }
        if uplink_nmse is not None:
            user_summary["corr_log_pl_abs_err_vs_log10_uplink_nmse"] = _pearson_corr(
                log_pl_abs_err[:, idx],
                _safe_log10(uplink_nmse[:, idx]),
            )
        summary["per_user"].append(user_summary)

    analysis_dir = os.path.join(run_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    with open(os.path.join(analysis_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    np.savez_compressed(
        os.path.join(analysis_dir, "analysis_arrays.npz"),
        round_idx=round_idx,
        user_id=user_id,
        moving_user_mask=moving_mask.astype(np.int8),
        speed_mps=speed_mps,
        alpha_tau=alpha_tau,
        gru_pl_true_sel=pl_true,
        gru_pl_pred_sel=pl_pred,
        gru_log_pl_true_sel=log_pl_true,
        gru_log_pl_pred_sel=log_pl_pred,
        gru_pl_abs_err=pl_abs_err,
        gru_pl_rel_err=pl_rel_err,
        gru_log_pl_abs_err=log_pl_abs_err,
        gru_uplink_true_nmse_k=uplink_nmse if uplink_nmse is not None else np.full_like(pl_true, np.nan),
    )

    _plot_heatmap(pl_true, "GRU PL true", os.path.join(analysis_dir, "pl_true_heatmap.png"), round_idx=round_idx, user_id=user_id, log10_scale=True)
    _plot_heatmap(pl_pred, "GRU PL pred", os.path.join(analysis_dir, "pl_pred_heatmap.png"), round_idx=round_idx, user_id=user_id, log10_scale=True)
    _plot_heatmap(log_pl_abs_err, "GRU |log PL error|", os.path.join(analysis_dir, "log_pl_abs_err_heatmap.png"), round_idx=round_idx, user_id=user_id, log10_scale=False)
    _plot_scatter(_safe_log10(pl_true), _safe_log10(pl_pred), moving_mask, os.path.join(analysis_dir, "pred_vs_true_log10_scatter.png"))
    _plot_user_mean(np.mean(log_pl_abs_err, axis=0), user_id, os.path.join(analysis_dir, "per_user_log_pl_mae.png"), "Per-user mean |log PL error|", "mean abs error")
    _plot_user_mean(np.mean(pl_rel_err, axis=0), user_id, os.path.join(analysis_dir, "per_user_pl_mape.png"), "Per-user mean relative PL error", "mean relative error")
    if uplink_nmse is not None:
        _plot_error_vs_nmse(log_pl_abs_err, uplink_nmse, moving_mask, os.path.join(analysis_dir, "log_pl_err_vs_uplink_nmse.png"))

    print(f"GRU PL snapshot analysis saved to: {analysis_dir}")


if __name__ == "__main__":
    main()
