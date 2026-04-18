#!/usr/bin/env python3
"""
Plot Uplink True NMSE versus communication rounds for multiple pilot lengths.

Default behavior:
  - scan ../log/*.log
  - group logs by filename after removing pilot length, timestamp, and hash tags
  - choose the latest complete group containing P={4,8,16,32}
  - save ../figs/uplink_true_nmse_vs_rounds_by_pilot.png

For strict paper figures, pass explicit logs:
  python figs/plot_uplink_true_nmse_vs_rounds_by_pilot.py \
    --logs log/P4.log log/P8.log log/P16.log log/P32.log
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.log_plotter import _parse_log_metrics  # noqa: E402


DEFAULT_PILOTS = (4, 8, 16, 32)
DEFAULT_MODELS = ("GRU", "CNN-arch", "CNN-base", "LMMSE")
MODEL_COLORS = {
    "GRU": "#1f77b4",
    "CNN-arch": "#d627aa",
    "CNN-base": "#17becf",
    "LMMSE": "#d62728",
}
PILOT_LINESTYLES = {
    4: "-",
    8: "--",
    16: "-.",
    32: ":",
}

PILOT_RE = re.compile(r"(?:^|_)P(?P<pilot>\d+)(?=_)")
RUN_STAMP_RE = re.compile(r"_\d{8}-\d{6}-\d{6}$")
HASH_SEGMENT_RE = re.compile(r"_H[0-9a-fA-F]{8,16}(?=_)")
EXP_HASH_RE = re.compile(r"_EXP[0-9a-fA-F]+(?=_)")
LEGACY_HASH_SUFFIX_RE = re.compile(r"_[0-9a-fA-F]{12}$")
STABLE_CONFIG_TOKEN_RE = re.compile(
    r"^(SYN|K\d+|M\d+|N\d+|L.+|PSNR.+|CTX.+|GT.+|RHO.+|W\d+|FL.+|R\d+|E\d+|B.+|"
    r"AIR.+|SNR.+|TX.+|OC.+|VF.+|EPS.+|PF.+|GG.+|WU.+|DT.+|FC.+|AD.+|SPD.+|DIR.+|"
    r"PART.+|NK.+)$"
)
OPTIMIZER_RUN_TOKEN_RE = re.compile(r"^[OSD]\d.*$")


@dataclass(frozen=True)
class LogRecord:
    path: Path
    pilot: int
    group_key: str
    uplink_true_nmse: Mapping[str, Mapping[int, float]]
    mtime: float


def _parse_csv_ints(text: str) -> Tuple[int, ...]:
    values = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return tuple(values)


def _parse_csv_strings(text: str) -> Tuple[str, ...]:
    values = tuple(part.strip() for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one model name")
    return values


def _extract_pilot_from_name(path: Path) -> Optional[int]:
    match = PILOT_RE.search(path.stem)
    if not match:
        return None
    return int(match.group("pilot"))


def _canonical_group_token(token: str) -> str:
    # Long log names can truncate DIRnone to DIRnon. Treat them as the same run config.
    if token == "DIRnon":
        return "DIRnone"
    return token


def _normalized_group_key(path: Path) -> str:
    stem = path.stem
    stem = RUN_STAMP_RE.sub("", stem)
    stem = LEGACY_HASH_SUFFIX_RE.sub("", stem)
    stem = HASH_SEGMENT_RE.sub("_H*", stem)
    stem = EXP_HASH_RE.sub("_EXP*", stem)
    tokens = []
    for token in stem.split("_"):
        if not token:
            continue
        if re.fullmatch(r"P\d+", token):
            continue
        if re.fullmatch(r"H[0-9a-fA-F*]+", token):
            continue
        if STABLE_CONFIG_TOKEN_RE.match(token) or OPTIMIZER_RUN_TOKEN_RE.match(token):
            tokens.append(_canonical_group_token(token))
    if len(tokens) >= 10:
        return "_".join(tokens)
    stem = re.sub(r"(?<=_)P\d+(?=_)", "P*", stem)
    stem = re.sub(r"__+", "_", stem)
    return stem.strip("_")


def _uplink_true_nmse_curves(
    log_path: Path,
    models: Sequence[str],
) -> Dict[str, Dict[int, float]]:
    metrics = _parse_log_metrics(str(log_path))
    uplink = metrics.get("Uplink True NMSE", {})
    curves: Dict[str, Dict[int, float]] = {}
    for model in models:
        points = uplink.get(model, {})
        clean_points = {
            int(round_idx): float(value)
            for round_idx, value in points.items()
            if value is not None and math.isfinite(float(value)) and float(value) > 0
        }
        if clean_points:
            curves[model] = dict(sorted(clean_points.items()))
    return curves


def _iter_log_paths(log_dir: Path) -> Iterable[Path]:
    if not log_dir.is_dir():
        return []
    return sorted(log_dir.glob("*.log"))


def _load_records(
    paths: Iterable[Path],
    pilots: Sequence[int],
    models: Sequence[str],
    require_filters: Sequence[str],
) -> List[LogRecord]:
    pilot_set = set(pilots)
    records: List[LogRecord] = []
    for raw_path in paths:
        path = raw_path.expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path = path.resolve()
        if not path.is_file():
            print(f"[skip] missing log: {path}", file=sys.stderr)
            continue
        stem = path.stem
        if any(token not in stem for token in require_filters):
            continue
        pilot = _extract_pilot_from_name(path)
        if pilot not in pilot_set:
            continue
        uplink_true_nmse = _uplink_true_nmse_curves(path, models)
        if not uplink_true_nmse:
            print(f"[skip] no positive uplink_true_NMSE curve found: {path.name}", file=sys.stderr)
            continue
        records.append(
            LogRecord(
                path=path,
                pilot=int(pilot),
                group_key=_normalized_group_key(path),
                uplink_true_nmse=uplink_true_nmse,
                mtime=path.stat().st_mtime,
            )
        )
    return records


def _latest_by_pilot(records: Iterable[LogRecord]) -> Dict[int, LogRecord]:
    selected: Dict[int, LogRecord] = {}
    for record in records:
        prev = selected.get(record.pilot)
        if prev is None or record.mtime > prev.mtime:
            selected[record.pilot] = record
    return selected


def _select_records(
    records: Sequence[LogRecord],
    pilots: Sequence[int],
    strict_group: bool,
) -> Tuple[Dict[int, LogRecord], str]:
    if not records:
        raise SystemExit("No usable logs found.")

    pilot_set = set(pilots)
    groups: Dict[str, List[LogRecord]] = {}
    for record in records:
        groups.setdefault(record.group_key, []).append(record)

    ranked = []
    for key, group_records in groups.items():
        by_pilot = _latest_by_pilot(group_records)
        present = pilot_set.intersection(by_pilot)
        newest = max(record.mtime for record in group_records)
        ranked.append((len(present), newest, key, by_pilot))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)

    for count, _newest, key, by_pilot in ranked:
        if count == len(pilots):
            return {pilot: by_pilot[pilot] for pilot in pilots}, key

    if strict_group:
        best_count, _newest, best_key, _by_pilot = ranked[0]
        raise SystemExit(
            f"No complete same-config group for pilots {list(pilots)}. "
            f"Best group has {best_count}/{len(pilots)}: {best_key}"
        )

    print(
        "[warn] no complete same-config group found; using latest available log per pilot.",
        file=sys.stderr,
    )
    mixed = _latest_by_pilot(records)
    missing = [pilot for pilot in pilots if pilot not in mixed]
    if missing:
        raise SystemExit(f"Missing pilot lengths: {missing}")
    return {pilot: mixed[pilot] for pilot in pilots}, "<mixed-latest-per-pilot>"


def _final_value(curve: Mapping[int, float]) -> Optional[float]:
    if not curve:
        return None
    return float(curve[max(curve)])


def _print_selection(selected: Mapping[int, LogRecord], pilots: Sequence[int], models: Sequence[str]) -> None:
    print("Selected logs:")
    for pilot in pilots:
        record = selected[pilot]
        print(f"  P={pilot}: {record.path.name}")
    print("")

    header = ["P", *models]
    rows = []
    for pilot in pilots:
        record = selected[pilot]
        row = [str(pilot)]
        for model in models:
            curve = record.uplink_true_nmse.get(model, {})
            value = _final_value(curve)
            if value is None:
                row.append("NA")
            else:
                row.append(f"{value:.6e} @R{max(curve)}")
        rows.append(row)
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    print("Final available Uplink True NMSE:")
    print("  ".join(cell.rjust(widths[i]) for i, cell in enumerate(header)))
    for row in rows:
        print("  ".join(cell.rjust(widths[i]) for i, cell in enumerate(row)))


def _plot(
    selected: Mapping[int, LogRecord],
    pilots: Sequence[int],
    models: Sequence[str],
    output_path: Path,
    group_key: str,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(
            "matplotlib is required to generate PNG output. "
            "Install project requirements first, e.g. `pip install -r requirements.txt`."
        ) from exc
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.91, bottom=0.19)
    any_curve = False

    for model in models:
        for pilot in pilots:
            curve = selected[pilot].uplink_true_nmse.get(model, {})
            if not curve:
                continue
            xs = sorted(curve)
            ys = [curve[x] for x in xs]
            ax.plot(
                xs,
                ys,
                linestyle=PILOT_LINESTYLES.get(pilot, "-"),
                linewidth=1.8,
                color=MODEL_COLORS.get(model),
            )
            any_curve = True

    if not any_curve:
        raise SystemExit("No positive Uplink True NMSE curves found for requested models and pilots.")

    ax.set_title("Uplink True NMSE vs. Rounds by Pilot Length")
    ax.set_xlabel("Round")
    ax.set_ylabel("Uplink True NMSE")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.32)

    model_handles = [
        Line2D([0], [0], color=MODEL_COLORS.get(model), linewidth=2.0, label=model)
        for model in models
    ]
    pilot_handles = [
        Line2D([0], [0], color="#333333", linestyle=PILOT_LINESTYLES.get(pilot, "-"), linewidth=2.0, label=f"P={pilot}")
        for pilot in pilots
    ]
    model_legend = ax.legend(handles=model_handles, title="Method", loc="upper right", fontsize=8, title_fontsize=8)
    ax.add_artist(model_legend)
    ax.legend(handles=pilot_handles, title="Pilot length", loc="lower left", fontsize=8, title_fontsize=8)

    fig.text(
        0.01,
        0.015,
        f"group: {group_key}",
        fontsize=7,
        color="#555555",
        ha="left",
        va="bottom",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot Uplink True NMSE against rounds for P={4,8,16,32} from training logs."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=PROJECT_ROOT / "log",
        help="Directory containing training .log files. Ignored when --logs is provided.",
    )
    parser.add_argument(
        "--logs",
        nargs="*",
        type=Path,
        default=None,
        help="Explicit log files to use. This is the safest mode for paper figures.",
    )
    parser.add_argument(
        "--pilots",
        type=_parse_csv_ints,
        default=DEFAULT_PILOTS,
        help="Comma-separated pilot lengths, default: 4,8,16,32.",
    )
    parser.add_argument(
        "--models",
        type=_parse_csv_strings,
        default=DEFAULT_MODELS,
        help="Comma-separated models, default: GRU,CNN-arch,CNN-base,LMMSE.",
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Require this substring in selected log filenames. Can be repeated.",
    )
    parser.add_argument(
        "--strict-group",
        action="store_true",
        help="Fail if no complete same-config group is found.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected logs and final available values without plotting.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "figs" / "uplink_true_nmse_vs_rounds_by_pilot.png",
        help="Output PNG path, default: figs/uplink_true_nmse_vs_rounds_by_pilot.png.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    pilots = tuple(args.pilots)
    models = tuple(args.models)

    if args.logs:
        paths = args.logs
    else:
        paths = list(_iter_log_paths(args.log_dir.expanduser().resolve()))

    records = _load_records(paths, pilots, models, tuple(args.filter))
    selected, group_key = _select_records(records, pilots, strict_group=bool(args.strict_group))
    _print_selection(selected, pilots, models)

    if args.dry_run:
        return 0

    out_path = args.output.expanduser()
    if not out_path.is_absolute():
        out_path = (PROJECT_ROOT / out_path).resolve()
    saved = _plot(selected, pilots, models, out_path, group_key)
    print(f"Saved: {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
