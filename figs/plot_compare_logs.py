#!/usr/bin/env python3
"""
Overlay round-metric figures from multiple training logs.

Default behavior:
  - read the two newest logs under ../log
  - save the same per-metric figures as utils.log_plotter into ../figs/compare_latest2

Examples:
  python figs/plot_compare_logs.py
  python figs/plot_compare_logs.py --logs log/a.log log/b.log --labels "tau=0.065" "tau=0.08"
  python figs/plot_compare_logs.py --latest-count 4 --labels A B C D --output-dir figs/compare_abcd
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import re
import sys
from typing import Any, List, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.log_plotter import (  # noqa: E402
    LINEAR_Y_TITLES,
    MODEL_COLORS,
    PLOT_SPECS,
    PROXY_ORACLE_GAP_TITLE,
    _build_optimizer_compare_metric,
    _build_proxy_oracle_gap_metric,
    _display_title,
    _find_optimizer_pair_log,
    _parse_log_metrics,
)


MAX_LOGS = 4
LINESTYLES = (
    "-",
    "--",
    (0, (8, 5, 2, 5)),
    (0, (2, 5)),
)
GRU_GROUP_UPLINK_TITLE = "GRU Group Uplink True NMSE"
GRU_GROUP_UPLINK_STEM = "07_gru_group_uplink_true_nmse"
# Low/high keep the original green/orange semantics. The first log uses the
# previous baseline colors; later logs stay in-family with larger contrast.
GRU_GROUP_COMPARE_LOW_COLORS = ("#1b7837", "#7fbf7b", "#005a32", "#4daf4a")
GRU_GROUP_COMPARE_HIGH_COLORS = ("#d95f02", "#fdb863", "#a6611a", "#f7811d")


@dataclass(frozen=True)
class RunMetrics:
    path: Path
    label: str
    metrics: Mapping[str, Mapping[str, Mapping[int, float]]]
    linestyle: Any


def _resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _latest_logs(log_dir: Path, count: int) -> List[Path]:
    log_dir = _resolve_path(log_dir)
    if not log_dir.is_dir():
        raise SystemExit(f"Log directory does not exist: {log_dir}")
    logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        raise SystemExit(f"No .log files found under: {log_dir}")
    return logs[:count]


def _parse_labels(raw_labels: Optional[Sequence[str]], count: int, paths: Sequence[Path]) -> List[str]:
    if not raw_labels:
        return [f"log{i + 1}" for i in range(count)]
    labels = list(raw_labels)
    if len(labels) == 1 and "," in labels[0]:
        labels = [part.strip() for part in labels[0].split(",") if part.strip()]
    if len(labels) != count:
        raise SystemExit(f"--labels count ({len(labels)}) must match log count ({count}).")
    return labels


def _safe_slug(text: str, max_len: int = 36) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    slug = re.sub(r"_+", "_", slug).strip("._-")
    if not slug:
        slug = "run"
    if len(slug) <= max_len:
        return slug
    return slug[:max_len].rstrip("._-")


def _default_output_dir(labels: Sequence[str], explicit_logs: bool) -> Path:
    if not explicit_logs and len(labels) == 2 and labels == ["log1", "log2"]:
        return PROJECT_ROOT / "figs" / "compare_latest2"
    name = "compare_" + "_vs_".join(_safe_slug(label, max_len=18) for label in labels)
    return PROJECT_ROOT / "figs" / name


def _load_run_metrics(path: Path, label: str, linestyle: Any) -> RunMetrics:
    metrics = _parse_log_metrics(str(path))
    paired_log_path = _find_optimizer_pair_log(str(path), metrics.get("_meta", {}))
    paired_metrics = _parse_log_metrics(paired_log_path) if paired_log_path is not None else None
    metrics["DC vs SCA Uplink True NMSE"] = _build_optimizer_compare_metric(metrics, paired_metrics)
    metrics[PROXY_ORACLE_GAP_TITLE] = _build_proxy_oracle_gap_metric(
        metrics.get("Proxy NMSE After Optimization", {})
    )
    return RunMetrics(path=path, label=label, metrics=metrics, linestyle=linestyle)


def _load_runs(log_paths: Sequence[Path], labels: Sequence[str]) -> List[RunMetrics]:
    runs = []
    for idx, (path, label) in enumerate(zip(log_paths, labels)):
        resolved = _resolve_path(path)
        if not resolved.is_file():
            raise SystemExit(f"Log file does not exist: {resolved}")
        runs.append(_load_run_metrics(resolved, label, LINESTYLES[idx]))
    return runs


def _is_gru_group_uplink(title: str, file_stem: str) -> bool:
    return title == GRU_GROUP_UPLINK_TITLE or file_stem == GRU_GROUP_UPLINK_STEM


def _gru_group_compare_color(model: str, run_idx: int, fallback: str) -> str:
    if model.startswith("GRU-low"):
        return GRU_GROUP_COMPARE_LOW_COLORS[run_idx % len(GRU_GROUP_COMPARE_LOW_COLORS)]
    if model.startswith("GRU-high"):
        return GRU_GROUP_COMPARE_HIGH_COLORS[run_idx % len(GRU_GROUP_COMPARE_HIGH_COLORS)]
    return fallback


def _gru_group_compare_label(run_label: str, group_name: str) -> str:
    return f"{run_label} / {group_name}"


def _render_matplotlib(
    runs: Sequence[RunMetrics],
    out_dir: Path,
    specs,
    dpi: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:
        raise SystemExit("matplotlib is required to draw comparison figures.") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    for title, file_stem, model_order, model_colors, model_styles in specs:
        ylog = title not in LINEAR_Y_TITLES
        is_gru_group_uplink = _is_gru_group_uplink(title, file_stem)
        fig, ax = plt.subplots(figsize=(7.2, 6.4))
        fig.subplots_adjust(left=0.09, right=0.98, top=0.91, bottom=0.19)
        any_curve = False
        used_models = set()
        for run_idx, run in enumerate(runs):
            metric = run.metrics.get(title, {})
            for model in model_order:
                points = metric.get(model, {})
                if not points:
                    continue
                rounds = []
                values = []
                for round_idx in sorted(points):
                    value = float(points[round_idx])
                    if not math.isfinite(value):
                        continue
                    if ylog and value <= 0.0:
                        continue
                    rounds.append(int(round_idx))
                    values.append(value)
                if not rounds:
                    continue
                style = (model_styles or {}).get(model, {})
                color = model_colors.get(model, MODEL_COLORS.get(model, "#333333"))
                linestyle = run.linestyle
                if is_gru_group_uplink:
                    color = _gru_group_compare_color(model, run_idx, color)
                    linestyle = style.get("linestyle", "-")
                ax.plot(
                    rounds,
                    values,
                    color=color,
                    linestyle=linestyle,
                    linewidth=style.get("linewidth", 1.8),
                    alpha=0.95,
                )
                used_models.add(model)
                any_curve = True

        display_title = _display_title({}, title)
        ax.set_title(display_title)
        ax.set_xlabel("Round")
        ax.set_ylabel("Oracle gap (dB)" if title == PROXY_ORACLE_GAP_TITLE else "Value")
        if ylog:
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.32)

        if any_curve:
            if is_gru_group_uplink:
                color_handles = []
                for run_idx, run in enumerate(runs):
                    color_handles.extend(
                        [
                            Line2D(
                                [0],
                                [0],
                                color=GRU_GROUP_COMPARE_LOW_COLORS[run_idx],
                                linewidth=2.0,
                                linestyle="-",
                                label=_gru_group_compare_label(run.label, "low"),
                            ),
                            Line2D(
                                [0],
                                [0],
                                color=GRU_GROUP_COMPARE_HIGH_COLORS[run_idx],
                                linewidth=2.0,
                                linestyle="-",
                                label=_gru_group_compare_label(run.label, "high"),
                            ),
                        ]
                    )
                style_handles = [
                    Line2D([0], [0], color="#222222", linewidth=1.8, linestyle="--", label="oracle"),
                    Line2D([0], [0], color="#222222", linewidth=1.8, linestyle="-", label="actual"),
                ]
                color_legend = ax.legend(
                    handles=color_handles,
                    loc="center right",
                    fontsize=8,
                    ncol=2,
                    framealpha=0.92,
                )
                ax.add_artist(color_legend)
                ax.legend(handles=style_handles, loc="lower left", fontsize=8, framealpha=0.92)
            else:
                model_handles = [
                    Line2D(
                        [0],
                        [0],
                        color=model_colors.get(model, MODEL_COLORS.get(model, "#333333")),
                        linewidth=2.0,
                        linestyle="-",
                        label=model,
                    )
                    for model in model_order
                    if model in used_models
                ]
                log_handles = [
                    Line2D([0], [0], color="#222222", linewidth=1.8, linestyle=run.linestyle, label=run.label)
                    for run in runs
                ]
                model_legend = ax.legend(
                    handles=model_handles,
                    fontsize=8,
                    loc="upper right",
                    framealpha=0.92,
                )
                ax.add_artist(model_legend)
                ax.legend(handles=log_handles, fontsize=8, loc="lower left", framealpha=0.92)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)

        fig.text(
            0.01,
            0.015,
            f"logs: {' vs '.join(run.label for run in runs)}",
            fontsize=7,
            color="#555555",
            ha="left",
            va="bottom",
        )
        fig.savefig(out_dir / f"{file_stem}.png", dpi=dpi)
        plt.close(fig)


def _parse_metric_filter(text: Optional[str]):
    if not text:
        return PLOT_SPECS
    wanted = {part.strip() for part in text.split(",") if part.strip()}
    if not wanted:
        return PLOT_SPECS
    selected = []
    for spec in PLOT_SPECS:
        title, file_stem, *_ = spec
        if title in wanted or file_stem in wanted:
            selected.append(spec)
    missing = wanted.difference({spec[0] for spec in selected}).difference({spec[1] for spec in selected})
    if missing:
        raise SystemExit(f"Unknown metric spec(s): {', '.join(sorted(missing))}")
    return tuple(selected)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Overlay the same round-metric figures from 1-4 logs.")
    parser.add_argument("--logs", nargs="+", default=None, help="Log files to compare. Default: latest two logs.")
    parser.add_argument("--log-dir", default="log", help="Directory used when --logs is omitted. Default: log")
    parser.add_argument("--latest-count", type=int, default=2, help="Number of latest logs to use when --logs is omitted.")
    parser.add_argument("--labels", nargs="+", default=None, help="Labels for logs. Use quotes for labels with spaces.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Default: figs/compare_latest2 or labels-based name.")
    parser.add_argument("--dpi", type=int, default=170, help="Output PNG DPI.")
    parser.add_argument(
        "--metrics",
        default=None,
        help="Optional comma-separated metric titles or file stems to draw. Default: all PLOT_SPECS.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.latest_count <= 0:
        raise SystemExit("--latest-count must be positive.")

    explicit_logs = bool(args.logs)
    log_paths = [Path(p) for p in args.logs] if explicit_logs else _latest_logs(Path(args.log_dir), args.latest_count)
    if len(log_paths) > MAX_LOGS:
        raise SystemExit(f"At most {MAX_LOGS} logs are supported because only {MAX_LOGS} distinct linestyles are defined.")
    if not log_paths:
        raise SystemExit("No logs selected.")

    labels = _parse_labels(args.labels, len(log_paths), log_paths)
    out_dir = _resolve_path(Path(args.output_dir)) if args.output_dir else _default_output_dir(labels, explicit_logs)
    specs = _parse_metric_filter(args.metrics)
    runs = _load_runs(log_paths, labels)

    print("Selected logs:")
    for run in runs:
        print(f"  {run.label}: {run.path}")

    _render_matplotlib(runs, out_dir, specs, dpi=args.dpi)
    print(f"Comparison figures saved to: {out_dir} (png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
