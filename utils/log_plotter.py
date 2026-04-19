import math
import os
import re
from typing import Dict, Optional


ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
ROUND_RE = re.compile(r"Round\s+(\d+)")
MEAN_LOCAL_LOSS_RE = re.compile(r"Round\s+(\d+)\s+mean local loss\s*->\s*(.+)$")
GRU_DUAL_HEAD_LOSS_RE = re.compile(r"Round\s+(\d+)\s+GRU dual-head local loss\s*->\s*(.+)$")
ETA_RE = re.compile(r"eta=([0-9eE+.\-]+)")
AGG_NMSE_RE = re.compile(r"agg_NMSE=([0-9eE+.\-]+)")
AGG_ERR_RE = re.compile(r"agg_err=([0-9eE+.\-]+)")
PROXY_NMSE_RE = re.compile(r"proxy_NMSE=([0-9eE+.\-]+)")
UPLINK_TRUE_NMSE_RE = re.compile(r"uplink_true_NMSE=([0-9eE+.\-]+)")
LOW_GROUP_NMSE_RE = re.compile(r"low\(n=\d+\)=([0-9eE+.\-]+)")
HIGH_GROUP_NMSE_RE = re.compile(r"high\(n=\d+\)=([0-9eE+.\-]+)")
BEAM_RIS_OPT_RE = re.compile(r"Beam/RIS optimizer=(OA|SCA|DC)", re.IGNORECASE)
RUN_STAMP_RE = re.compile(r"^\d{8}-\d{6}-\d{6}$")
LEGACY_LOG_SUFFIX_RE = re.compile(r"^(.*)_([0-9a-f]{12})$")
EXP_HASH_SUFFIX_RE = re.compile(r"^(.*)_EXP[0-9a-f]+$", re.IGNORECASE)
MODEL_ORDER = ("GRU", "CNN-arch", "CNN-base", "LMMSE", "Oracle-true")
UPLINK_MODEL_ORDER = ("GRU", "CNN-arch", "CNN-base", "LMMSE")
MODEL_COLORS = {
    "GRU": "#1f77b4",
    "CNN-arch": "#d627aa",
    "CNN-base": "#17becf",
    "LMMSE": "#d62728",
    "Oracle-true": "#ff7f0e",
}
OPT_COMPARE_ORDER = ("DC-GRU", "SCA-GRU")
OPT_COMPARE_COLORS = {
    "DC-GRU": "#1f77b4",
    "SCA-GRU": "#d62728",
}
GRU_GROUP_ORDER = ("GRU-low", "GRU-high")
GRU_GROUP_COLORS = {
    "GRU-low": "#2ca02c",
    "GRU-high": "#d62728",
}
GRU_GROUP_UPLINK_ORDER = ("GRU-low", "GRU-low-oracle", "GRU-high", "GRU-high-oracle")
GRU_GROUP_UPLINK_COLORS = {
    "GRU-low": "#2ca02c",
    "GRU-low-oracle": "#2ca02c",
    "GRU-high": "#d62728",
    "GRU-high-oracle": "#d62728",
}
GRU_GROUP_UPLINK_STYLES = {
    "GRU-low": {"linestyle": "-", "stroke_dasharray": None},
    "GRU-low-oracle": {"linestyle": "--", "stroke_dasharray": "6,4"},
    "GRU-high": {"linestyle": "-", "stroke_dasharray": None},
    "GRU-high-oracle": {"linestyle": "--", "stroke_dasharray": "6,4"},
}
PROXY_ORACLE_GAP_TITLE = "Proxy Oracle Gap"
LINEAR_Y_TITLES = {PROXY_ORACLE_GAP_TITLE}
PLOT_SPECS = (
    ("Round Mean Local Loss", "01_round_mean_local_loss", MODEL_ORDER, MODEL_COLORS, None),
    ("AirComp Eta", "02_aircomp_eta", MODEL_ORDER, MODEL_COLORS, None),
    ("AirComp Aggregation NMSE", "03_aircomp_aggregation_nmse", MODEL_ORDER, MODEL_COLORS, None),
    ("AirComp Aggregation Error", "04_aircomp_aggregation_error", MODEL_ORDER, MODEL_COLORS, None),
    ("Proxy NMSE After Optimization", "05_proxy_nmse_after_optimization", MODEL_ORDER, MODEL_COLORS, None),
    ("Uplink True NMSE", "06_uplink_true_nmse", UPLINK_MODEL_ORDER, MODEL_COLORS, None),
    (
        "GRU Group Uplink True NMSE",
        "07_gru_group_uplink_true_nmse",
        GRU_GROUP_UPLINK_ORDER,
        GRU_GROUP_UPLINK_COLORS,
        GRU_GROUP_UPLINK_STYLES,
    ),
    ("DC vs SCA Uplink True NMSE", "08_dc_vs_sca_uplink_true_nmse", OPT_COMPARE_ORDER, OPT_COMPARE_COLORS, None),
    ("GRU Group Proxy NMSE", "09_gru_group_proxy_nmse", GRU_GROUP_ORDER, GRU_GROUP_COLORS, None),
    (PROXY_ORACLE_GAP_TITLE, "10_proxy_oracle_gap", UPLINK_MODEL_ORDER, MODEL_COLORS, None),
)
COMMON_PROXY_UPDATE_CSV = "05_proxy_nmse_after_optimization_common_update_vars.csv"
COMMON_PROXY_DISPLAY_TITLE = "Proxy NMSE After Optimization (common GRU update variance)"
PROXY_ORACLE_GAP_DISPLAY_TITLE = "Proxy Oracle Gap to Oracle (dB)"


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
    if msg.startswith("LMMSE AirComp"):
        return "LMMSE"
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
    if "Optimized theta_ota (LMMSE)" in msg:
        return "LMMSE"
    if "Optimized theta_ota:" in msg:
        return "GRU"
    if msg.startswith("Oracle-true proxy_NMSE="):
        return "Oracle-true"
    if msg.startswith("CNN-arch proxy_NMSE="):
        return "CNN-arch"
    if msg.startswith("CNN-base proxy_NMSE="):
        return "CNN-base"
    if msg.startswith("LMMSE proxy_NMSE="):
        return "LMMSE"
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
    if msg.startswith("LMMSE uplink_true_NMSE="):
        return "LMMSE"
    if msg.startswith("Oracle-true uplink_true_NMSE="):
        return "Oracle-true"
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


def _log_prefix_stem(log_path: str) -> str:
    stem = os.path.splitext(os.path.basename(log_path))[0]
    m = LEGACY_LOG_SUFFIX_RE.match(stem)
    stem = m.group(1) if m else stem
    m_exp = EXP_HASH_SUFFIX_RE.match(stem)
    return m_exp.group(1) if m_exp else stem


def _pair_prefix_and_optimizer_from_path(log_path: str, default_optimizer: Optional[str] = None):
    stem = os.path.splitext(os.path.basename(log_path))[0]
    parts = stem.rsplit("_", 2)
    if len(parts) == 3 and RUN_STAMP_RE.match(parts[2]):
        optimizer_tag = parts[1].upper()
        prefix = parts[0]
        m_exp = EXP_HASH_SUFFIX_RE.match(prefix)
        if m_exp:
            prefix = m_exp.group(1)
        if optimizer_tag.startswith("OA-"):
            return prefix, "oa"
        if optimizer_tag.startswith("SCA-"):
            return prefix, "sca"
        if optimizer_tag.startswith("DC-"):
            return prefix, "dc"
        if optimizer_tag.startswith("O"):
            return prefix, "oa"
        if optimizer_tag.startswith("S"):
            return prefix, "sca"
        if optimizer_tag.startswith("D"):
            return prefix, "dc"
        if default_optimizer in {"oa", "sca", "dc"}:
            return prefix, default_optimizer
    legacy_prefix = _log_prefix_stem(log_path)
    return legacy_prefix, default_optimizer


def _read_experiment_prefix_from_log(log_path: str) -> Optional[str]:
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for raw in f:
                msg = _strip_ansi(raw.rstrip("\n"))
                msg = msg.split("] ", 1)[1] if "] " in msg else msg
                if msg.startswith("Experiment prefix="):
                    return msg.split("=", 1)[1].strip()
    except OSError:
        return None
    return None


def _proxy_model_order(enable_cnn_arch: bool, enable_cnn_baseline: bool, enable_lmmse: bool):
    order = ["GRU"]
    if enable_cnn_arch:
        order.append("CNN-arch")
    if enable_cnn_baseline:
        order.append("CNN-base")
    if enable_lmmse:
        order.append("LMMSE")
    order.append("Oracle-true")
    return order


def _parse_log_metrics(log_path: str) -> Dict[str, Dict[str, Dict[int, float]]]:
    round_local_loss: Dict[str, Dict[int, float]] = {}
    round_eta: Dict[str, Dict[int, float]] = {}
    round_agg_nmse: Dict[str, Dict[int, float]] = {}
    round_agg_err: Dict[str, Dict[int, float]] = {}
    round_proxy_nmse: Dict[str, Dict[int, float]] = {}
    round_uplink_true_nmse: Dict[str, Dict[int, float]] = {}
    round_gru_group_uplink_true_nmse: Dict[str, Dict[int, float]] = {}
    round_gru_group_proxy_nmse: Dict[str, Dict[int, float]] = {}
    current_round: Optional[int] = None
    pending_proxy_model: Optional[str] = None
    experiment_prefix: Optional[str] = None
    enable_cnn_arch = False
    enable_cnn_baseline = False
    enable_lmmse = False
    proxy_sequence_idx = 0
    beam_ris_optimizer: Optional[str] = None

    with open(log_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = _strip_ansi(raw.rstrip("\n"))
            msg = line.split("] ", 1)[1] if "] " in line else line

            if "CNN architecture ablation enabled=True" in msg:
                enable_cnn_arch = True
            if "Literature CNN baseline enabled=True" in msg:
                enable_cnn_baseline = True
            if "LMMSE baseline enabled=True" in msg:
                enable_lmmse = True
            if msg.startswith("Experiment prefix="):
                experiment_prefix = msg.split("=", 1)[1].strip()
            m_opt = BEAM_RIS_OPT_RE.search(msg)
            if m_opt:
                beam_ris_optimizer = m_opt.group(1).strip().lower()

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

            model_aircomp = _model_from_aircomp_line(msg)
            if model_aircomp is not None:
                m_eta = ETA_RE.search(msg)
                if m_eta:
                    _append_metric(round_eta, model_aircomp, current_round, float(m_eta.group(1)))
                m_agg_nmse = AGG_NMSE_RE.search(msg)
                if m_agg_nmse:
                    _append_metric(round_agg_nmse, model_aircomp, current_round, float(m_agg_nmse.group(1)))
                m_agg_err = AGG_ERR_RE.search(msg)
                if m_agg_err:
                    _append_metric(round_agg_err, model_aircomp, current_round, float(m_agg_err.group(1)))
                continue

            model_proxy = _model_from_proxy_line(msg)
            if model_proxy is not None:
                pending_proxy_model = model_proxy

            m_proxy = PROXY_NMSE_RE.search(msg)
            if m_proxy:
                if ("GRU-G1" in msg) and ("proxy_NMSE=" in msg):
                    _append_metric(round_gru_group_proxy_nmse, "GRU-low", current_round, float(m_proxy.group(1)))
                    continue
                if ("GRU-G2" in msg) and ("proxy_NMSE=" in msg):
                    _append_metric(round_gru_group_proxy_nmse, "GRU-high", current_round, float(m_proxy.group(1)))
                    continue
                target_model = model_proxy if model_proxy is not None else pending_proxy_model
                if target_model is None and msg.startswith("proxy_NMSE="):
                    order = _proxy_model_order(enable_cnn_arch, enable_cnn_baseline, enable_lmmse)
                    if proxy_sequence_idx < len(order):
                        target_model = order[proxy_sequence_idx]
                        proxy_sequence_idx += 1
                if target_model is not None:
                    _append_metric(round_proxy_nmse, target_model, current_round, float(m_proxy.group(1)))
                    pending_proxy_model = None
                continue

            model_uplink = _model_from_uplink_line(msg)
            if msg.startswith("GRU uplink_true_NMSE semantic-groups ->"):
                m_low = LOW_GROUP_NMSE_RE.search(msg)
                m_high = HIGH_GROUP_NMSE_RE.search(msg)
                if m_low:
                    _append_metric(round_gru_group_uplink_true_nmse, "GRU-low", current_round, float(m_low.group(1)))
                if m_high:
                    _append_metric(round_gru_group_uplink_true_nmse, "GRU-high", current_round, float(m_high.group(1)))
                continue
            if msg.startswith("GRU oracle_uplink_true_NMSE semantic-groups ->"):
                m_low = LOW_GROUP_NMSE_RE.search(msg)
                m_high = HIGH_GROUP_NMSE_RE.search(msg)
                if m_low:
                    _append_metric(round_gru_group_uplink_true_nmse, "GRU-low-oracle", current_round, float(m_low.group(1)))
                if m_high:
                    _append_metric(round_gru_group_uplink_true_nmse, "GRU-high-oracle", current_round, float(m_high.group(1)))
                continue
            if model_uplink is not None:
                m_uplink = UPLINK_TRUE_NMSE_RE.search(msg)
                if m_uplink:
                    _append_metric(round_uplink_true_nmse, model_uplink, current_round, float(m_uplink.group(1)))
                continue

    path_pair_prefix, inferred_optimizer = _pair_prefix_and_optimizer_from_path(log_path, beam_ris_optimizer)
    log_pair_prefix = experiment_prefix or path_pair_prefix
    if beam_ris_optimizer is None:
        beam_ris_optimizer = inferred_optimizer
    return {
        "Round Mean Local Loss": round_local_loss,
        "AirComp Eta": round_eta,
        "AirComp Aggregation NMSE": round_agg_nmse,
        "AirComp Aggregation Error": round_agg_err,
        "Proxy NMSE After Optimization": round_proxy_nmse,
        "Uplink True NMSE": round_uplink_true_nmse,
        "GRU Group Uplink True NMSE": round_gru_group_uplink_true_nmse,
        "GRU Group Proxy NMSE": round_gru_group_proxy_nmse,
        "_meta": {
            "beam_ris_optimizer": beam_ris_optimizer,
            "log_pair_prefix": log_pair_prefix,
        },
    }


def _plot_metric_matplotlib(
    ax,
    metric_dict: Dict[str, Dict[int, float]],
    title: str,
    ylog: bool = True,
    model_order=MODEL_ORDER,
    model_colors=MODEL_COLORS,
    model_styles=None,
):
    any_curve = False
    for model in model_order:
        points = metric_dict.get(model, {})
        if not points:
            continue
        rounds = sorted(points.keys())
        values = [points[r] for r in rounds]
        style = (model_styles or {}).get(model, {})
        ax.plot(
            rounds,
            values,
            marker="o",
            markersize=2,
            linewidth=style.get("linewidth", 1.4),
            linestyle=style.get("linestyle", "-"),
            label=model,
            color=model_colors[model],
        )
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


def _lift_nonpositive_for_log(points: Dict[int, float], floor: float = 1e-15) -> Dict[int, float]:
    lifted = {}
    for round_idx, value in points.items():
        lifted[round_idx] = value if value > 0 else floor
    return lifted


def _display_title(metrics: Dict[str, object], title: str) -> str:
    titles = metrics.get("_plot_titles", {})
    if isinstance(titles, dict):
        display = str(titles.get(title, title))
        if display != title:
            return display
    if title == PROXY_ORACLE_GAP_TITLE:
        return PROXY_ORACLE_GAP_DISPLAY_TITLE
    return title


def _svg_panel(
    chunks: list,
    metric_dict: Dict[str, Dict[int, float]],
    title: str,
    x: int,
    y: int,
    w: int,
    h: int,
    ylog: bool = True,
    model_order=MODEL_ORDER,
    model_colors=MODEL_COLORS,
    model_styles=None,
) -> None:
    chunks.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#ffffff" stroke="#dddddd"/>')
    chunks.append(f'<text x="{x + 10}" y="{y + 22}" font-size="14" fill="#222">{_escape_xml(title)}</text>')

    pad_l, pad_r, pad_t, pad_b = 55, 14, 34, 40
    px0, py0 = x + pad_l, y + pad_t
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    chunks.append(f'<rect x="{px0}" y="{py0}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#cccccc"/>')

    all_rounds = []
    all_values = []
    for model in model_order:
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
    for idx, model in enumerate(model_order):
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
        color = model_colors[model]
        style = (model_styles or {}).get(model, {})
        stroke_dasharray = style.get("stroke_dasharray")
        dash_attr = f' stroke-dasharray="{stroke_dasharray}"' if stroke_dasharray else ""
        chunks.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{style.get("linewidth", 1.8)}"{dash_attr}/>'
        )

        ly = legend_y + idx * 14
        dash_line_attr = f' stroke-dasharray="{stroke_dasharray}"' if stroke_dasharray else ""
        chunks.append(
            f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 16}" y2="{ly}" stroke="{color}" stroke-width="2"{dash_line_attr}/>'
        )
        chunks.append(
            f'<text x="{legend_x + 20}" y="{ly + 3}" font-size="11" fill="#333">{_escape_xml(model)}</text>'
        )


def _read_optimizer_mode_from_log(log_path: str) -> Optional[str]:
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for raw in f:
                msg = _strip_ansi(raw.rstrip("\n"))
                msg = msg.split("] ", 1)[1] if "] " in msg else msg
                m_opt = BEAM_RIS_OPT_RE.search(msg)
                if m_opt:
                    return m_opt.group(1).strip().lower()
    except OSError:
        return None
    _, inferred_optimizer = _pair_prefix_and_optimizer_from_path(log_path, None)
    return inferred_optimizer


def _find_optimizer_pair_log(log_path: str, current_meta: Dict[str, Optional[str]]) -> Optional[str]:
    current_optimizer = current_meta.get("beam_ris_optimizer")
    pair_prefix = current_meta.get("log_pair_prefix")
    if current_optimizer not in {"dc", "sca"} or not pair_prefix:
        return None
    target_optimizer = "sca" if current_optimizer == "dc" else "dc"
    log_dir = os.path.dirname(log_path)
    candidates = []
    try:
        for filename in os.listdir(log_dir):
            if not filename.endswith(".log"):
                continue
            candidate_path = os.path.join(log_dir, filename)
            if os.path.abspath(candidate_path) == os.path.abspath(log_path):
                continue
            candidate_optimizer = _read_optimizer_mode_from_log(candidate_path)
            candidate_prefix = _read_experiment_prefix_from_log(candidate_path)
            if candidate_prefix is None:
                candidate_prefix, _ = _pair_prefix_and_optimizer_from_path(candidate_path, candidate_optimizer)
            if candidate_optimizer != target_optimizer:
                continue
            if candidate_prefix != pair_prefix:
                continue
            candidates.append(candidate_path)
    except OSError:
        return None
    if not candidates:
        return None
    candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return candidates[0]


def _build_optimizer_compare_metric(
    current_metrics: Dict[str, Dict[str, Dict[int, float]]],
    paired_metrics: Optional[Dict[str, Dict[str, Dict[int, float]]]],
) -> Dict[str, Dict[int, float]]:
    if paired_metrics is None:
        return {}

    by_optimizer = {}
    for metrics in (current_metrics, paired_metrics):
        optimizer = metrics.get("_meta", {}).get("beam_ris_optimizer")
        if optimizer in {"dc", "sca"}:
            by_optimizer[optimizer] = metrics
    if "dc" not in by_optimizer or "sca" not in by_optimizer:
        return {}

    compare_metric: Dict[str, Dict[int, float]] = {}
    for optimizer, prefix in (("dc", "DC"), ("sca", "SCA")):
        uplink_metric = by_optimizer[optimizer].get("Uplink True NMSE", {})
        gru_points = uplink_metric.get("GRU", {})
        if gru_points:
            compare_metric[f"{prefix}-GRU"] = dict(gru_points)
    return compare_metric


def _build_proxy_oracle_gap_metric(
    proxy_metric: Dict[str, Dict[int, float]],
) -> Dict[str, Dict[int, float]]:
    oracle_points = proxy_metric.get("Oracle-true", {})
    if not oracle_points:
        return {}

    gap_metric: Dict[str, Dict[int, float]] = {}
    for model in UPLINK_MODEL_ORDER:
        model_points = proxy_metric.get(model, {})
        if not model_points:
            continue
        model_gap = {}
        for round_idx, value in model_points.items():
            oracle_value = oracle_points.get(round_idx)
            if value is None or oracle_value is None or value <= 0.0 or oracle_value <= 0.0:
                continue
            model_gap[round_idx] = abs(10.0 * math.log10(value) - 10.0 * math.log10(oracle_value))
        if model_gap:
            gap_metric[model] = model_gap
    return gap_metric


def _load_common_update_proxy_metric(out_dir: str) -> Optional[Dict[str, Dict[int, float]]]:
    csv_path = os.path.join(out_dir, COMMON_PROXY_UPDATE_CSV)
    if not os.path.isfile(csv_path):
        return None

    metric: Dict[str, Dict[int, float]] = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            if len(header) < 2 or header[0] != "round":
                return None
            models = header[1:]
            for raw in f:
                parts = raw.strip().split(",")
                if len(parts) < 2 or not parts[0]:
                    continue
                try:
                    round_idx = int(parts[0])
                except ValueError:
                    continue
                for model, value_text in zip(models, parts[1:]):
                    if model not in MODEL_ORDER or not value_text:
                        continue
                    try:
                        value = float(value_text)
                    except ValueError:
                        continue
                    if math.isfinite(value) and value > 0:
                        metric.setdefault(model, {})[round_idx] = value
    except OSError:
        return None

    return metric if any(metric.values()) else None


def _remove_common_update_proxy_metric_csv(out_dir: str) -> None:
    csv_path = os.path.join(out_dir, COMMON_PROXY_UPDATE_CSV)
    try:
        os.remove(csv_path)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _render_svg(metrics: Dict[str, Dict[str, Dict[int, float]]], log_stem: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    width, height = 740, 520
    for title, file_stem, model_order, model_colors, model_styles in PLOT_SPECS:
        out_path = os.path.join(out_dir, f"{file_stem}.svg")
        display_title = _display_title(metrics, title)
        chunks = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect x="0" y="0" width="100%" height="100%" fill="#f6f8fb"/>',
            f'<text x="{width / 2}" y="26" text-anchor="middle" font-size="13" fill="#222">{_escape_xml(log_stem)}</text>',
        ]
        _svg_panel(
            chunks,
            metrics.get(title, {}),
            display_title,
            20,
            54,
            680,
            420,
            ylog=title not in LINEAR_Y_TITLES,
            model_order=model_order,
            model_colors=model_colors,
            model_styles=model_styles,
        )
        chunks.append("</svg>")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(chunks))
    return out_dir


def _render_matplotlib(metrics: Dict[str, Dict[str, Dict[int, float]]], log_stem: str, out_dir: str) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    os.makedirs(out_dir, exist_ok=True)
    for title, file_stem, model_order, model_colors, model_styles in PLOT_SPECS:
        out_path = os.path.join(out_dir, f"{file_stem}.png")
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        display_title = _display_title(metrics, title)
        _plot_metric_matplotlib(
            ax,
            metrics.get(title, {}),
            display_title,
            ylog=title not in LINEAR_Y_TITLES,
            model_order=model_order,
            model_colors=model_colors,
            model_styles=model_styles,
        )
        fig.suptitle(log_stem, fontsize=9)
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
    return out_dir


def _plot_round_metrics_from_log_impl(log_path: str, figs_root: str, *, refresh_pair: bool) -> Optional[str]:
    """
    Parse round-level metrics from a training log and save one directory of metric figures.
    Output path format:
      ./figs/<log_filename_without_ext>/*.(png|svg)
    """
    if not os.path.isfile(log_path):
        return None

    metrics = _parse_log_metrics(log_path)
    paired_log_path = _find_optimizer_pair_log(log_path, metrics.get("_meta", {}))
    paired_metrics = _parse_log_metrics(paired_log_path) if paired_log_path is not None else None
    metrics["DC vs SCA Uplink True NMSE"] = _build_optimizer_compare_metric(metrics, paired_metrics)
    log_stem = os.path.splitext(os.path.basename(log_path))[0]
    os.makedirs(figs_root, exist_ok=True)

    out_dir = os.path.join(figs_root, log_stem)
    common_proxy_metric = _load_common_update_proxy_metric(out_dir)
    if common_proxy_metric is not None:
        metrics["Proxy NMSE After Optimization"] = common_proxy_metric
        metrics["_plot_titles"] = {
            "Proxy NMSE After Optimization": COMMON_PROXY_DISPLAY_TITLE,
        }
    metrics[PROXY_ORACLE_GAP_TITLE] = _build_proxy_oracle_gap_metric(
        metrics.get("Proxy NMSE After Optimization", {})
    )

    png_path = _render_matplotlib(metrics, log_stem, out_dir)
    if png_path is not None:
        if common_proxy_metric is not None:
            _remove_common_update_proxy_metric_csv(out_dir)
        if refresh_pair and paired_log_path is not None:
            _plot_round_metrics_from_log_impl(paired_log_path, figs_root, refresh_pair=False)
        return png_path

    svg_path = _render_svg(metrics, log_stem, out_dir)
    if common_proxy_metric is not None:
        _remove_common_update_proxy_metric_csv(out_dir)
    if refresh_pair and paired_log_path is not None:
        _plot_round_metrics_from_log_impl(paired_log_path, figs_root, refresh_pair=False)
    return svg_path


def plot_round_metrics_from_log(log_path: str, figs_root: str = "./figs") -> Optional[str]:
    return _plot_round_metrics_from_log_impl(log_path, figs_root, refresh_pair=True)
