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
        raise FileNotFoundError(f"No GRU group switch sensitivity snapshots found under: {abs_path}")
    if latest:
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return os.path.abspath(candidates[0])
    raise FileNotFoundError(
        f"Path does not contain snapshots directly: {abs_path}. "
        "Pass a concrete run directory or use --latest."
    )


def _load_series(round_paths):
    rows = []
    tau_b_grid = None
    tau_d_grid = None
    plateau_stack = []
    patience_stack = []
    ready_stack = []

    for path in round_paths:
        with np.load(path, allow_pickle=False) as data:
            if tau_b_grid is None:
                tau_b_grid = np.asarray(data["tau_B_grid"], dtype=np.float64)
                tau_d_grid = np.asarray(data["tau_D_grid"], dtype=np.float64)
            rows.append(
                {
                    "round_idx": int(_load_npz_scalar(data, "round_idx")),
                    "B": float(_load_npz_scalar(data, "B")),
                    "D": float(_load_npz_scalar(data, "D")),
                    "B_ema": float(_load_npz_scalar(data, "B_ema")),
                    "D_ema": float(_load_npz_scalar(data, "D_ema")),
                    "delta_B_ema": float(_load_npz_scalar(data, "delta_B_ema")),
                    "delta_D_ema": float(_load_npz_scalar(data, "delta_D_ema")),
                    "configured_tau_B": float(_load_npz_scalar(data, "configured_tau_B")),
                    "configured_tau_D": float(_load_npz_scalar(data, "configured_tau_D")),
                    "configured_plateau_reached": int(_load_npz_scalar(data, "configured_plateau_reached")),
                    "configured_patience": int(_load_npz_scalar(data, "configured_patience")),
                    "switch_min_round": int(_load_npz_scalar(data, "switch_min_round")),
                    "switch_patience": int(_load_npz_scalar(data, "switch_patience")),
                    "rounds_until_min_round": int(_load_npz_scalar(data, "rounds_until_min_round")),
                }
            )
            plateau_stack.append(np.asarray(data["plateau_grid"], dtype=bool))
            patience_stack.append(np.asarray(data["patience_grid"], dtype=np.int16))
            ready_stack.append(np.asarray(data["ready_grid"], dtype=bool))

    rows.sort(key=lambda row: row["round_idx"])
    return rows, tau_b_grid, tau_d_grid, np.stack(plateau_stack), np.stack(patience_stack), np.stack(ready_stack)


def _first_ready_round(rounds: np.ndarray, ready_stack: np.ndarray) -> np.ndarray:
    first = np.full(ready_stack.shape[1:], np.nan, dtype=np.float64)
    for i in range(ready_stack.shape[1]):
        for j in range(ready_stack.shape[2]):
            ready_idx = np.flatnonzero(ready_stack[:, i, j])
            if ready_idx.size:
                first[i, j] = float(rounds[int(ready_idx[0])])
    return first


def _plot_delta_trace(arrays: dict, out_path: str):
    if not HAS_MATPLOTLIB:
        return
    rounds = arrays["round_idx"]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    eps = 1e-12
    ax.semilogy(rounds, np.maximum(arrays["delta_B_ema"], eps), marker="o", linewidth=1.5, label="delta_B_ema")
    ax.semilogy(rounds, np.maximum(arrays["delta_D_ema"], eps), marker="s", linewidth=1.5, label="delta_D_ema")
    ax.axhline(float(arrays["configured_tau_B"][0]), color="#1f77b4", linestyle="--", linewidth=1.2, label="tau_B configured")
    ax.axhline(float(arrays["configured_tau_D"][0]), color="#d62728", linestyle="--", linewidth=1.2, label="tau_D configured")
    ax.axvline(float(arrays["switch_min_round"][0]), color="#333333", linestyle=":", linewidth=1.1, label="min round")
    ax.set_title("Pre-switch EMA relative changes")
    ax.set_xlabel("Round")
    ax.set_ylabel("Relative change")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_configured_patience(arrays: dict, out_path: str):
    if not HAS_MATPLOTLIB:
        return
    rounds = arrays["round_idx"]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(rounds, arrays["configured_patience"], marker="o", linewidth=1.6, color="#2ca02c", label="configured patience")
    ax.axhline(float(arrays["switch_patience"][0]), color="#d62728", linestyle="--", linewidth=1.2, label="required patience")
    ax.axvline(float(arrays["switch_min_round"][0]), color="#333333", linestyle=":", linewidth=1.1, label="min round")
    plateau_rounds = rounds[arrays["configured_plateau_reached"].astype(bool)]
    if plateau_rounds.size:
        ax.scatter(
            plateau_rounds,
            arrays["configured_patience"][arrays["configured_plateau_reached"].astype(bool)],
            s=42,
            facecolors="none",
            edgecolors="#ff7f0e",
            linewidths=1.2,
            label="plateau true",
        )
    ax.set_title("Configured tau_B/tau_D patience before min round")
    ax.set_xlabel("Round")
    ax.set_ylabel("Patience counter")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_ready_fraction(arrays: dict, out_path: str):
    if not HAS_MATPLOTLIB:
        return
    rounds = arrays["round_idx"]
    ready_stack = arrays["ready_grid"].astype(bool)
    fraction = ready_stack.reshape(ready_stack.shape[0], -1).mean(axis=1)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(rounds, fraction, marker="o", linewidth=1.6, color="#9467bd")
    ax.axvline(float(arrays["switch_min_round"][0]), color="#333333", linestyle=":", linewidth=1.1, label="min round")
    ax.set_title("Fraction of tau grid ready before min round")
    ax.set_xlabel("Round")
    ax.set_ylabel("Ready fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_first_ready_heatmap(arrays: dict, out_path: str):
    if not HAS_MATPLOTLIB:
        return
    first_ready = np.ma.masked_invalid(arrays["first_ready_round"])
    tau_b = arrays["tau_B_grid"]
    tau_d = arrays["tau_D_grid"]
    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(first_ready, origin="lower", aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_title("First pre-min round reaching required patience")
    ax.set_xlabel("tau_D")
    ax.set_ylabel("tau_B")
    ax.set_xticks(np.arange(len(tau_d)))
    ax.set_xticklabels([f"{v:.3g}" for v in tau_d], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(tau_b)))
    ax.set_yticklabels([f"{v:.3g}" for v in tau_b])
    fig.colorbar(im, ax=ax, label="First ready round")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def analyze(run_dir: str, output_dir: str = None) -> str:
    run_dir = _discover_run_dir(run_dir)
    round_paths = sorted(glob(os.path.join(run_dir, "round_*.npz")))
    if not round_paths:
        raise FileNotFoundError(f"No round_*.npz files found in {run_dir}")

    rows, tau_b_grid, tau_d_grid, plateau_stack, patience_stack, ready_stack = _load_series(round_paths)
    rounds = np.asarray([row["round_idx"] for row in rows], dtype=np.int32)
    arrays = {
        "round_idx": rounds,
        "B": np.asarray([row["B"] for row in rows], dtype=np.float64),
        "D": np.asarray([row["D"] for row in rows], dtype=np.float64),
        "B_ema": np.asarray([row["B_ema"] for row in rows], dtype=np.float64),
        "D_ema": np.asarray([row["D_ema"] for row in rows], dtype=np.float64),
        "delta_B_ema": np.asarray([row["delta_B_ema"] for row in rows], dtype=np.float64),
        "delta_D_ema": np.asarray([row["delta_D_ema"] for row in rows], dtype=np.float64),
        "configured_tau_B": np.asarray([row["configured_tau_B"] for row in rows], dtype=np.float64),
        "configured_tau_D": np.asarray([row["configured_tau_D"] for row in rows], dtype=np.float64),
        "configured_plateau_reached": np.asarray([row["configured_plateau_reached"] for row in rows], dtype=np.int8),
        "configured_patience": np.asarray([row["configured_patience"] for row in rows], dtype=np.int16),
        "switch_min_round": np.asarray([row["switch_min_round"] for row in rows], dtype=np.int32),
        "switch_patience": np.asarray([row["switch_patience"] for row in rows], dtype=np.int16),
        "rounds_until_min_round": np.asarray([row["rounds_until_min_round"] for row in rows], dtype=np.int32),
        "tau_B_grid": tau_b_grid,
        "tau_D_grid": tau_d_grid,
        "plateau_grid": plateau_stack.astype(np.int8),
        "patience_grid": patience_stack,
        "ready_grid": ready_stack.astype(np.int8),
    }
    arrays["first_ready_round"] = _first_ready_round(rounds, ready_stack)

    configured_ready_idx = np.flatnonzero(
        arrays["configured_patience"] >= int(arrays["switch_patience"][0])
    )
    finite_first = arrays["first_ready_round"][np.isfinite(arrays["first_ready_round"])]
    summary = {
        "run_dir": os.path.abspath(run_dir),
        "num_round_snapshots": int(rounds.size),
        "round_min": int(rounds.min()),
        "round_max": int(rounds.max()),
        "switch_min_round": int(arrays["switch_min_round"][0]),
        "switch_patience": int(arrays["switch_patience"][0]),
        "configured_tau_B": float(arrays["configured_tau_B"][0]),
        "configured_tau_D": float(arrays["configured_tau_D"][0]),
        "configured_first_ready_round": (
            int(rounds[int(configured_ready_idx[0])]) if configured_ready_idx.size else None
        ),
        "tau_B_grid_size": int(tau_b_grid.size),
        "tau_D_grid_size": int(tau_d_grid.size),
        "ready_pairs_final": int(ready_stack[-1].sum()),
        "ready_pairs_total": int(ready_stack[-1].size),
        "ready_fraction_final": float(ready_stack[-1].mean()),
        "first_ready_round_min": float(np.min(finite_first)) if finite_first.size else None,
        "first_ready_round_max": float(np.max(finite_first)) if finite_first.size else None,
    }

    analysis_dir = os.path.abspath(output_dir) if output_dir else os.path.join(run_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    np.savez_compressed(os.path.join(analysis_dir, "analysis_arrays.npz"), **arrays)
    with open(os.path.join(analysis_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    _plot_delta_trace(arrays, os.path.join(analysis_dir, "delta_ema_trace.png"))
    _plot_configured_patience(arrays, os.path.join(analysis_dir, "configured_patience_trace.png"))
    _plot_ready_fraction(arrays, os.path.join(analysis_dir, "ready_fraction_by_round.png"))
    _plot_first_ready_heatmap(arrays, os.path.join(analysis_dir, "first_ready_round_heatmap.png"))
    return analysis_dir


def main():
    parser = argparse.ArgumentParser(description="Analyze GRU group switch tau sensitivity snapshots.")
    parser.add_argument(
        "run_dir",
        nargs="?",
        default="debug/gru_group_switch_sensitivity_snapshots",
        help="Snapshot run directory, or the root containing run subdirectories.",
    )
    parser.add_argument("--latest", action="store_true", help="Use the newest run under run_dir.")
    parser.add_argument("--output", default=None, help="Optional output analysis directory.")
    args = parser.parse_args()

    run_dir = _discover_run_dir(args.run_dir, latest=args.latest)
    out_dir = analyze(run_dir, output_dir=args.output)
    print(f"Analysis saved to: {out_dir}")
    if not HAS_MATPLOTLIB:
        print("matplotlib is unavailable; only summary.json and analysis_arrays.npz were written.")


if __name__ == "__main__":
    main()
