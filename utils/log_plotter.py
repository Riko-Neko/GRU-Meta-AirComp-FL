import math
import os
import re
from typing import Dict, Optional, Tuple


ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
ROUND_RE = re.compile(r"Round\s+(\d+)")
MEAN_LOCAL_LOSS_RE = re.compile(r"Round\s+(\d+)\s+mean local loss\s*->\s*(.+)$")
GRU_DUAL_HEAD_LOSS_RE = re.compile(r"Round\s+(\d+)\s+GRU dual-head local loss\s*->\s*(.+)$")
GLOBAL_UPDATE_RE = re.compile(
    r"(GRU|CNN-arch|CNN-base)\s+global(?:\s+backbone)?\s+update norm:\s*([0-9eE+.\-]+)"
)
AGG_NMSE_RE = re.compile(r"agg_NMSE=([0-9eE+.\-]+)")
PROXY_NMSE_RE = re.compile(r"proxy_NMSE=([0-9eE+.\-]+)")
UPLINK_TRUE_NMSE_RE = re.compile(r"uplink_true_NMSE=([0-9eE+.\-]+)")
MODEL_ORDER = ("GRU", "CNN-arch", "CNN-base", "Oracle-true")
MODEL_COLORS = {"GRU": "#1f77b4", "CNN-arch": "#d627aa", "CNN-base": "#17becf", "Oracle-true": "#ff7f0e"}


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def _escape_xml(text: str) -> str:
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&apos;"))


def _model_from_aircomp_line(msg: str) -> Optional[str]:
    if msg.startswith("CNN-arch AirComp"):
        return "CNN-arch"
    if msg.startswith("CNN-base AirComp"):
        return "CNN-base"
    if msg.startswith("AirComp eta="):
        return "GRU"
    return None


def _model_from_proxy_line(msg: str) -> Optional[str]:
    if "Optimized theta_ota (Oracle-true)" in msg:
        return "Oracle-true"
    if "Optimized theta_ota (CNN-arch)" in msg:
        return "CNN-arch"
    if "Optimized theta_ota (CNN-base)" in msg:
        return "CNN-base"
    if "Optimized theta_ota:" in msg:
        return "GRU"
    if msg.startswith("Oracle-true proxy_NMSE="):
        return "Oracle-true"
    if msg.startswith("CNN-arch proxy_NMSE="):
        return "CNN-arch"
    if msg.startswith("CNN-base proxy_NMSE="):
        return "CNN-base"
    if msg.startswith("GRU proxy_NMSE="):
        return "GRU"
    return None


def _model_from_uplink_line(msg: str) -> Optional[str]:
    if msg.startswith("GRU uplink_true_NMSE="):
        return "GRU"
    if msg.startswith("CNN-arch uplink_true_NMSE="):
        return "CNN-arch"
    if msg.startswith("CNN-base uplink_true_NMSE="):
        return "CNN-base"
    return None


def _append_metric(
    store: Dict[str, Dict[int, float]],
    model: str,
    round_idx: Optional[int],
    value: float,
) -> None:
    if round_idx is None:
        return
    store.setdefault(model, {})[round_idx] = value


def _proxy_model_order(enable_cnn_arch: bool, enable_cnn_baseline: bool):
    order = ["GRU"]
    if enable_cnn_arch:
        order.append("CNN-arch")
    if enable_cnn_baseline:
        order.append("CNN-base")
    order.append("Oracle-true")
    return order


def _parse_log_metrics(log_path: str) -> Dict[str, Dict[str, Dict[int, float]]]:
    round_local_loss: Dict[str, Dict[int, float]] = {}
    round_update_norm: Dict[str, Dict[int, float]] = {}
    round_agg_nmse: Dict[str, Dict[int, float]] = {}
    round_proxy_nmse: Dict[str, Dict[int, float]] = {}
    round_uplink_true_nmse: Dict[str, Dict[int, float]] = {}
    current_round: Optional[int] = None
    pending_proxy_model: Optional[str] = None
    enable_cnn_arch = False
    enable_cnn_baseline = False
    proxy_sequence_idx = 0

    with open(log_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = _strip_ansi(raw.rstrip("\n"))
            msg = line.split("] ", 1)[1] if "] " in line else line

            if "CNN architecture ablation enabled=True" in msg:
                enable_cnn_arch = True
            if "Literature CNN baseline enabled=True" in msg:
                enable_cnn_baseline = True

            m_round = ROUND_RE.search(msg)
            if m_round and "Generating pilot observations" in msg:
                current_round = int(m_round.group(1))

            if "Optimizing beamforming and RIS configuration." in msg:
                proxy_sequence_idx = 0

            m_mean = MEAN_LOCAL_LOSS_RE.search(msg)
            if m_mean:
                r = int(m_mean.group(1))
                parts = [p.strip() for p in m_mean.group(2).split(",")]
                for part in parts:
                    if ":" not in part:
                        continue
                    model, value = part.split(":", 1)
                    model = model.strip()
                    try:
                        v = float(value.strip())
                    except ValueError:
                        continue
                    _append_metric(round_local_loss, model, r, v)
                continue

            m_gru_dual = GRU_DUAL_HEAD_LOSS_RE.search(msg)
            if m_gru_dual:
                r = int(m_gru_dual.group(1))
                parts = [p.strip() for p in m_gru_dual.group(2).split(",")]
                for part in parts:
                    if ":" not in part:
                        continue
                    key, value = part.split(":", 1)
                    if key.strip() != "GRU_t":
                        continue
                    try:
                        v_t = float(value.strip())
                    except ValueError:
                        continue
                    # For plotting GRU loss, prefer t-head loss over total dual-head loss.
                    _append_metric(round_local_loss, "GRU", r, v_t)
                    break
                continue

            m_update = GLOBAL_UPDATE_RE.search(msg)
            if m_update:
                _append_metric(round_update_norm, m_update.group(1), current_round, float(m_update.group(2)))
                continue

            model_aircomp = _model_from_aircomp_line(msg)
            if model_aircomp is not None:
                m_nmse = AGG_NMSE_RE.search(msg)
                if m_nmse:
                    _append_metric(round_agg_nmse, model_aircomp, current_round, float(m_nmse.group(1)))
                continue

            model_proxy = _model_from_proxy_line(msg)
            if model_proxy is not None:
                pending_proxy_model = model_proxy

            m_proxy = PROXY_NMSE_RE.search(msg)
            if m_proxy:
                target_model = model_proxy if model_proxy is not None else pending_proxy_model
                if target_model is None and msg.startswith("proxy_NMSE="):
                    order = _proxy_model_order(enable_cnn_arch, enable_cnn_baseline)
                    if proxy_sequence_idx < len(order):
                        target_model = order[proxy_sequence_idx]
                        proxy_sequence_idx += 1
                if target_model is not None:
                    _append_metric(round_proxy_nmse, target_model, current_round, float(m_proxy.group(1)))
                    pending_proxy_model = None
                continue

            model_uplink = _model_from_uplink_line(msg)
            if model_uplink is not None:
                m_uplink = UPLINK_TRUE_NMSE_RE.search(msg)
                if m_uplink:
                    _append_metric(round_uplink_true_nmse, model_uplink, current_round, float(m_uplink.group(1)))
                continue

    return {
        "Round Mean Local Loss": round_local_loss,
        "Global Update Norm": round_update_norm,
        "AirComp Aggregation NMSE": round_agg_nmse,
        "Proxy NMSE After Optimization": round_proxy_nmse,
        "Uplink True NMSE": round_uplink_true_nmse,
    }


def _plot_metric_matplotlib(ax, metric_dict: Dict[str, Dict[int, float]], title: str, ylog: bool = True):
    any_curve = False
    for model in MODEL_ORDER:
        points = metric_dict.get(model, {})
        if not points:
            continue
        rounds = sorted(points.keys())
        values = [points[r] for r in rounds]
        ax.plot(rounds, values, marker="o", markersize=2, linewidth=1.4, label=model, color=MODEL_COLORS[model])
        any_curve = True

    ax.set_title(title)
    ax.set_xlabel("Round")
    if ylog:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    if any_curve:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)


def _format_tick(v: float) -> str:
    if v == 0:
        return "0"
    if abs(v) >= 1e4 or abs(v) <= 1e-3:
        return f"{v:.1e}"
    return f"{v:.4g}"


def _svg_panel(
    chunks: list,
    metric_dict: Dict[str, Dict[int, float]],
    title: str,
    x: int,
    y: int,
    w: int,
    h: int,
    ylog: bool = True,
) -> None:
    chunks.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#ffffff" stroke="#dddddd"/>')
    chunks.append(f'<text x="{x + 10}" y="{y + 22}" font-size="14" fill="#222">{_escape_xml(title)}</text>')

    pad_l, pad_r, pad_t, pad_b = 55, 14, 34, 40
    px0, py0 = x + pad_l, y + pad_t
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    chunks.append(f'<rect x="{px0}" y="{py0}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#cccccc"/>')

    all_rounds = []
    all_values = []
    for model in MODEL_ORDER:
        points = metric_dict.get(model, {})
        for r, v in points.items():
            if ylog and v <= 0:
                continue
            all_rounds.append(r)
            all_values.append(v)

    if not all_rounds or not all_values:
        chunks.append(
            f'<text x="{x + w // 2}" y="{y + h // 2}" text-anchor="middle" font-size="16" fill="#666">N/A</text>'
        )
        return

    x_min, x_max = min(all_rounds), max(all_rounds)
    if x_min == x_max:
        x_min -= 1
        x_max += 1

    if ylog:
        y_min = min(all_values)
        y_max = max(all_values)
        y_t_min = math.log10(y_min)
        y_t_max = math.log10(y_max)
    else:
        y_min, y_max = min(all_values), max(all_values)
        y_t_min, y_t_max = y_min, y_max

    if y_t_min == y_t_max:
        y_t_min -= 0.5
        y_t_max += 0.5

    def sx(rv: float) -> float:
        return px0 + (rv - x_min) * pw / (x_max - x_min)

    def sy(vv: float) -> float:
        t = math.log10(vv) if ylog else vv
        return py0 + (y_t_max - t) * ph / (y_t_max - y_t_min)

    # Axis ticks
    xticks = sorted(set(int(round(x_min + i * (x_max - x_min) / 4.0)) for i in range(5)))
    for rv in xticks:
        xx = sx(rv)
        chunks.append(f'<line x1="{xx:.2f}" y1="{py0 + ph}" x2="{xx:.2f}" y2="{py0 + ph + 4}" stroke="#666"/>')
        chunks.append(
            f'<text x="{xx:.2f}" y="{py0 + ph + 18}" text-anchor="middle" font-size="10" fill="#444">{rv}</text>'
        )

    for i in range(5):
        t = y_t_min + i * (y_t_max - y_t_min) / 4.0
        val = (10 ** t) if ylog else t
        yy = py0 + (y_t_max - t) * ph / (y_t_max - y_t_min)
        chunks.append(f'<line x1="{px0 - 4}" y1="{yy:.2f}" x2="{px0}" y2="{yy:.2f}" stroke="#666"/>')
        chunks.append(f'<line x1="{px0}" y1="{yy:.2f}" x2="{px0 + pw}" y2="{yy:.2f}" stroke="#eeeeee"/>')
        chunks.append(
            f'<text x="{px0 - 8}" y="{yy + 3:.2f}" text-anchor="end" font-size="10" fill="#444">{_format_tick(val)}</text>'
        )

    # Curves + legend
    legend_y = y + 18
    legend_x = x + w - 190
    for idx, model in enumerate(MODEL_ORDER):
        points = metric_dict.get(model, {})
        if not points:
            continue
        r_sorted = sorted(points.keys())
        vals = [points[r] for r in r_sorted if (points[r] > 0 or not ylog)]
        rounds = [r for r in r_sorted if (points[r] > 0 or not ylog)]
        if not rounds:
            continue
        d = " ".join(
            ("M" if i == 0 else "L") + f"{sx(rounds[i]):.2f},{sy(vals[i]):.2f}"
            for i in range(len(rounds))
        )
        color = MODEL_COLORS[model]
        chunks.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.8"/>')

        ly = legend_y + idx * 14
        chunks.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 16}" y2="{ly}" stroke="{color}" stroke-width="2"/>')
        chunks.append(
            f'<text x="{legend_x + 20}" y="{ly + 3}" font-size="11" fill="#333">{_escape_xml(model)}</text>'
        )


def _render_svg(metrics: Dict[str, Dict[str, Dict[int, float]]], log_stem: str, out_path: str) -> str:
    width, height = 1420, 1400
    panels: Tuple[Tuple[int, int, int, int], ...] = (
        (20, 42, 680, 420),
        (720, 42, 680, 420),
        (20, 500, 680, 420),
        (720, 500, 680, 420),
        (20, 958, 680, 420),
    )
    titles = (
        "Round Mean Local Loss",
        "Global Update Norm",
        "AirComp Aggregation NMSE",
        "Proxy NMSE After Optimization",
        "Uplink True NMSE",
    )

    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#f6f8fb"/>',
        f'<text x="{width / 2}" y="26" text-anchor="middle" font-size="16" fill="#222">{_escape_xml(log_stem)}</text>',
    ]

    for i, title in enumerate(titles):
        x, y, w, h = panels[i]
        _svg_panel(chunks, metrics.get(title, {}), title, x, y, w, h, ylog=True)

    chunks.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(chunks))
    return out_path


def _render_matplotlib(metrics: Dict[str, Dict[str, Dict[int, float]]], log_stem: str, out_path: str) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)
    titles = (
        "Round Mean Local Loss",
        "Global Update Norm",
        "AirComp Aggregation NMSE",
        "Proxy NMSE After Optimization",
        "Uplink True NMSE",
    )
    for ax, title in zip(axes.flat, titles):
        _plot_metric_matplotlib(ax, metrics.get(title, {}), title, ylog=True)
    for ax in axes.flat[len(titles):]:
        ax.axis("off")
    fig.suptitle(log_stem, fontsize=11)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_round_metrics_from_log(log_path: str, figs_root: str = "./figs") -> Optional[str]:
    """
    Parse round-level metrics from a training log and save one comparison figure.
    Output path format:
      ./figs/<log_filename_without_ext>.(png|svg)
    """
    if not os.path.isfile(log_path):
        return None

    metrics = _parse_log_metrics(log_path)
    log_stem = os.path.splitext(os.path.basename(log_path))[0]
    os.makedirs(figs_root, exist_ok=True)

    out_png = os.path.join(figs_root, f"{log_stem}.png")
    png_path = _render_matplotlib(metrics, log_stem, out_png)
    if png_path is not None:
        return png_path

    out_svg = os.path.join(figs_root, f"{log_stem}.svg")
    return _render_svg(metrics, log_stem, out_svg)
