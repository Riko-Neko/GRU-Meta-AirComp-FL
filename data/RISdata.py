from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.io import loadmat


def _pattern_to_row(ris_pattern: Any) -> Optional[np.ndarray]:
    if ris_pattern is None:
        return None
    arr = np.asarray(ris_pattern)
    if arr.size == 0:
        return None
    return arr.reshape(-1).astype(np.float32, copy=False)


def _selected_trace_value(sample: Dict[str, Any]) -> Optional[np.complex64]:
    trace = sample.get("s21_trace")
    freq_idx = sample.get("freq_idx")
    if trace is None or freq_idx is None:
        return None

    trace_arr = np.asarray(trace).reshape(-1)
    idx = int(freq_idx)
    if idx < 0 or idx >= trace_arr.size:
        return None

    value = trace_arr[idx]
    reference = sample.get("reference_trace")
    if reference is not None:
        ref_arr = np.asarray(reference).reshape(-1)
        if 0 <= idx < ref_arr.size:
            value = value - ref_arr[idx]
    return np.complex64(value)


def _estimate_effective_channel(samples: Sequence[Dict[str, Any]]) -> Optional[np.ndarray]:
    rows: List[np.ndarray] = []
    values: List[np.complex64] = []

    for sample in samples:
        row = _pattern_to_row(sample.get("ris_pattern"))
        value = _selected_trace_value(sample)
        if row is None or value is None:
            continue
        rows.append(row)
        values.append(value)

    if not rows:
        return None

    n_elem = rows[0].size
    if any(row.size != n_elem for row in rows):
        return None

    design = np.stack(rows, axis=0).astype(np.float32, copy=False)
    response = np.asarray(values, dtype=np.complex64).reshape(-1)
    channel, _, _, _ = np.linalg.lstsq(design, response, rcond=1e-6)
    return channel.astype(np.complex64, copy=False)


def load_data(
        file_path,
        num_users=None,
        *,
        num_bs_antennas=32,
        subset="specular",
        result_key="rand",
        reference_key="RISallOff",
        freq_hz=5.375e9,
        freq_tol_hz=30e6,
        max_pattern_samples=None,
        min_snr_db=None,
):
    """
    Adapter used by `main.py`.

    The RIS-S21 dataset does not provide a decomposed `(H_BR, h_RU, h_BU)` tuple directly.
    This wrapper reconstructs one effective reflection-channel vector per geometry by solving
    a least-squares inverse problem from `(RIS pattern, measured S21)` pairs at the chosen
    frequency bin, and then packs the result into the existing simulation interface:

    - `H_BR`: deterministic placeholder with unit gain on the first BS antenna
    - `h_RUs_static`: per-user effective reflection channels recovered from measurements
    - `h_BUs_static`: `None` (direct path is not available from this dataset wrapper)
    """
    dataset = load_ris_s21_dataset(
        root_dir=file_path,
        subset=subset,
        result_key=result_key,
        reference_key=reference_key,
        freq_hz=freq_hz,
        freq_tol_hz=freq_tol_hz,
        max_pattern_samples=max_pattern_samples,
        min_snr_db=min_snr_db,
    )

    grouped: Dict[Any, List[Dict[str, Any]]] = {}
    for sample in dataset["samples"]:
        grouped.setdefault(sample["geometry_id"], []).append(sample)

    geometry_ids = sorted(grouped.keys(), key=lambda x: str(x))
    effective_channels: List[np.ndarray] = []
    kept_geometry_ids: List[Any] = []
    for geometry_id in geometry_ids:
        channel = _estimate_effective_channel(grouped[geometry_id])
        if channel is None:
            continue
        effective_channels.append(channel)
        kept_geometry_ids.append(geometry_id)

    if not effective_channels:
        raise ValueError("RISdata.load_data could not reconstruct any effective reflection channels.")

    if num_users is not None:
        k = int(num_users)
        if k <= 0:
            raise ValueError(f"num_users must be positive, got {num_users}")
        if len(effective_channels) < k:
            raise ValueError(
                f"RISdata provides only {len(effective_channels)} valid geometries, "
                f"which is fewer than requested num_users={k}."
            )
        effective_channels = effective_channels[:k]
        kept_geometry_ids = kept_geometry_ids[:k]

    h_RUs_static = np.stack(effective_channels, axis=0).astype(np.complex64, copy=False)
    num_ris_elements = int(h_RUs_static.shape[1])
    num_bs_antennas = int(num_bs_antennas)
    if num_bs_antennas <= 0:
        raise ValueError(f"num_bs_antennas must be positive, got {num_bs_antennas}")

    # Use a deterministic placeholder BS-RIS matrix so the existing simulator can reuse
    # the measured effective reflection channel as `h_RU`.
    H_BR = np.zeros((num_ris_elements, num_bs_antennas), dtype=np.complex64)
    H_BR[:, 0] = 1.0 + 0.0j

    _ = kept_geometry_ids  # retained for debugging convenience when stepping through the loader
    return H_BR, h_RUs_static, None


@dataclass
class RISS21LoadConfig:
    """
    Loader config for:
    'A comprehensive dataset of RIS-based channel measurements in the 5GHz band' (2023)

    Defaults are chosen to be permissive:
    - load from 'specular'
    - load all geometries
    - use random patterns
    - no geometric filtering
    - no SNR filtering unless user provides an estimator
    """
    root_dir: Union[str, Path] = "dataset"
    subset: str = "specular"  # {"specular", "nonSpecular", "rotatingStage"}
    geometry_ids: Optional[Sequence[int]] = None

    # Which measurement family to load from results/patterns
    result_key: str = "rand"  # {"rand", "algoSEmax", "algoSEmin", "algoGreedyMax", ...}
    reference_key: str = "RISallOff"  # {"noPlate", "Plate", "RISallOff", "RISallOn"}

    # Frequency selection
    freq_hz: float = 5.375e9  # default center freq used frequently in the repo docs
    freq_tol_hz: float = 30e6  # loose by default, user can tighten later

    # Geometry filters (None = no filtering)
    dist_ant1_range: Optional[Tuple[float, float]] = None
    dist_ant2_range: Optional[Tuple[float, float]] = None
    az_ant1_range_deg: Optional[Tuple[float, float]] = None
    az_ant2_range_deg: Optional[Tuple[float, float]] = None
    el_ant1_range_deg: Optional[Tuple[float, float]] = None
    el_ant2_range_deg: Optional[Tuple[float, float]] = None
    height_ant1_range: Optional[Tuple[float, float]] = None
    height_ant2_range: Optional[Tuple[float, float]] = None

    # Optional sample cap
    max_pattern_samples: Optional[int] = None

    # SNR interface:
    # The official dataset docs do NOT define a canonical SNR field.
    # So we expose an estimator hook and optional threshold.
    snr_estimator: Optional[Callable[[Dict[str, Any]], Optional[float]]] = None
    min_snr_db: Optional[float] = None


# =========================
# Helpers
# =========================

def _squeeze(x: Any) -> Any:
    """Recursively squeeze singleton dimensions where useful."""
    if isinstance(x, np.ndarray):
        return np.squeeze(x)
    return x


def _matobj_to_dict(obj: Any) -> Any:
    """
    Convert scipy.io.loadmat MATLAB structs into nested Python dicts/lists/arrays.
    Handles:
      - mat_struct-like objects with _fieldnames
      - object arrays
      - numeric arrays
    """
    # MATLAB struct object
    if hasattr(obj, "_fieldnames"):
        out = {}
        for name in obj._fieldnames:
            out[name] = _matobj_to_dict(getattr(obj, name))
        return out

    # numpy array
    if isinstance(obj, np.ndarray):
        obj = np.squeeze(obj)

        # Object array: decode element-wise
        if obj.dtype == np.object_:
            if obj.ndim == 0:
                return _matobj_to_dict(obj.item())
            return [_matobj_to_dict(v) for v in obj.flat]

        # Numeric / logical / char arrays
        return obj

    return obj


def _load_mat_file(mat_path: Union[str, Path]) -> Dict[str, Any]:
    raw = loadmat(
        mat_path,
        struct_as_record=False,
        squeeze_me=True
    )
    clean = {}
    for k, v in raw.items():
        if k.startswith("__"):
            continue
        clean[k] = _matobj_to_dict(v)
    return clean


def _to_scalar(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    arr = np.asarray(x)
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    return None


def _in_range(val: Optional[float], rng: Optional[Tuple[float, float]]) -> bool:
    if rng is None:
        return True
    if val is None:
        return False
    lo, hi = rng
    return lo <= val <= hi


def _nearest_freq_index(freq_points_hz: np.ndarray, target_hz: float) -> int:
    freq_points_hz = np.asarray(freq_points_hz, dtype=float).reshape(-1)
    return int(np.argmin(np.abs(freq_points_hz - target_hz)))


def _safe_get(d: Dict[str, Any], key: str, default=None):
    return d[key] if isinstance(d, dict) and key in d else default


def _extract_geometry_meta(geometry: Dict[str, Any]) -> Dict[str, Optional[float]]:
    return {
        "dist_ant1": _to_scalar(_safe_get(geometry, "distAnt1")),
        "dist_ant2": _to_scalar(_safe_get(geometry, "distAnt2")),
        "az_ant1_deg": _to_scalar(_safe_get(geometry, "AzimuthAngAnt1")),
        "az_ant2_deg": _to_scalar(_safe_get(geometry, "AzimuthAngAnt2")),
        "el_ant1_deg": _to_scalar(_safe_get(geometry, "ElevationAngAnt1")),
        "el_ant2_deg": _to_scalar(_safe_get(geometry, "ElevationAngAnt2")),
        "height_ant1": _to_scalar(_safe_get(geometry, "heightAnt1")),
        "height_ant2": _to_scalar(_safe_get(geometry, "heightAnt2")),
    }


def _passes_geometry_filter(meta: Dict[str, Optional[float]], cfg: RISS21LoadConfig) -> bool:
    return (
            _in_range(meta["dist_ant1"], cfg.dist_ant1_range) and
            _in_range(meta["dist_ant2"], cfg.dist_ant2_range) and
            _in_range(meta["az_ant1_deg"], cfg.az_ant1_range_deg) and
            _in_range(meta["az_ant2_deg"], cfg.az_ant2_range_deg) and
            _in_range(meta["el_ant1_deg"], cfg.el_ant1_range_deg) and
            _in_range(meta["el_ant2_deg"], cfg.el_ant2_range_deg) and
            _in_range(meta["height_ant1"], cfg.height_ant1_range) and
            _in_range(meta["height_ant2"], cfg.height_ant2_range)
    )


def _normalize_patterns(patterns_obj: Any) -> List[np.ndarray]:
    """
    Each pattern is typically a 16x16 binary RIS state.
    Returns a list of 2D arrays.
    """
    if patterns_obj is None:
        return []

    if isinstance(patterns_obj, list):
        out = []
        for p in patterns_obj:
            arr = np.asarray(p)
            if arr.size > 0:
                out.append(arr.astype(np.int8))
        return out

    arr = np.asarray(patterns_obj)
    if arr.ndim == 2:
        return [arr.astype(np.int8)]
    if arr.ndim == 3:
        return [arr[i].astype(np.int8) for i in range(arr.shape[0])]
    return [arr.astype(np.int8)]


def _extract_complex_trace(result_entry: Any) -> Optional[np.ndarray]:
    """
    Best-effort extraction of a complex S21 trace.

    Since the public format doc tells us that results.* is a struct but does not fully specify
    the inner field names for every result variant, this function tries several common shapes.

    Supported cases:
    - direct complex ndarray
    - dict containing keys like 'S21', 's21', 'trace', 'data', 'values'
    """
    if result_entry is None:
        return None

    if isinstance(result_entry, np.ndarray):
        if np.iscomplexobj(result_entry) or np.issubdtype(result_entry.dtype, np.number):
            return np.asarray(result_entry)

    if isinstance(result_entry, dict):
        for key in ("S21", "s21", "trace", "data", "values"):
            if key in result_entry:
                arr = np.asarray(result_entry[key])
                return arr

    return None


def _extract_result_list(results_block: Any) -> List[Any]:
    """
    Turn one result family into a Python list of entries.
    """
    if results_block is None:
        return []

    if isinstance(results_block, list):
        return results_block

    if isinstance(results_block, np.ndarray) and results_block.dtype == np.object_:
        return [x for x in results_block.flat]

    return [results_block]


def default_snr_proxy(sample: Dict[str, Any]) -> Optional[float]:
    """
    A pragmatic SNR-like proxy, not an official dataset field.

    It compares the selected sample's magnitude at the chosen frequency bin
    against the chosen reference trace at the same bin.

    Returns:
        20*log10(|sample| / max(|reference|, eps))
    """
    trace = sample.get("s21_trace")
    ref_trace = sample.get("reference_trace")
    freq_idx = sample.get("freq_idx")

    if trace is None or ref_trace is None or freq_idx is None:
        return None

    trace = np.asarray(trace).reshape(-1)
    ref_trace = np.asarray(ref_trace).reshape(-1)

    if freq_idx >= len(trace) or freq_idx >= len(ref_trace):
        return None

    eps = 1e-12
    num = np.abs(trace[freq_idx])
    den = max(np.abs(ref_trace[freq_idx]), eps)
    return 20.0 * np.log10(max(num, eps) / den)


def load_ris_s21_dataset(
        root_dir: Union[str, Path] = "dataset",
        subset: str = "specular",
        geometry_ids: Optional[Sequence[int]] = None,
        result_key: str = "rand",
        reference_key: str = "RISallOff",
        freq_hz: float = 5.375e9,
        freq_tol_hz: float = 30e6,
        dist_ant1_range: Optional[Tuple[float, float]] = None,
        dist_ant2_range: Optional[Tuple[float, float]] = None,
        az_ant1_range_deg: Optional[Tuple[float, float]] = None,
        az_ant2_range_deg: Optional[Tuple[float, float]] = None,
        el_ant1_range_deg: Optional[Tuple[float, float]] = None,
        el_ant2_range_deg: Optional[Tuple[float, float]] = None,
        height_ant1_range: Optional[Tuple[float, float]] = None,
        height_ant2_range: Optional[Tuple[float, float]] = None,
        max_pattern_samples: Optional[int] = None,
        snr_estimator: Optional[Callable[[Dict[str, Any]], Optional[float]]] = None,
        min_snr_db: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Load the 2023 RIS-S21 measurement dataset.

    Returns a dict with:
      - config
      - files
      - samples
      - summary

    Each sample exposes:
      - geometry_id
      - subset
      - geometry_meta (distance / angle / height)
      - freq_points_hz
      - freq_idx
      - freq_selected_hz
      - pattern_index
      - ris_pattern
      - s21_trace
      - reference_trace
      - snr_db (optional, derived via estimator)
    """
    cfg = RISS21LoadConfig(
        root_dir=root_dir,
        subset=subset,
        geometry_ids=geometry_ids,
        result_key=result_key,
        reference_key=reference_key,
        freq_hz=freq_hz,
        freq_tol_hz=freq_tol_hz,
        dist_ant1_range=dist_ant1_range,
        dist_ant2_range=dist_ant2_range,
        az_ant1_range_deg=az_ant1_range_deg,
        az_ant2_range_deg=az_ant2_range_deg,
        el_ant1_range_deg=el_ant1_range_deg,
        el_ant2_range_deg=el_ant2_range_deg,
        height_ant1_range=height_ant1_range,
        height_ant2_range=height_ant2_range,
        max_pattern_samples=max_pattern_samples,
        snr_estimator=snr_estimator,
        min_snr_db=min_snr_db,
    )

    root = Path(cfg.root_dir)
    subset_dir = root / cfg.subset
    if not subset_dir.exists():
        raise FileNotFoundError(
            f"Subset directory not found: {subset_dir}. "
            f"Expected structure like dataset/specular or dataset/nonSpecular."
        )

    mat_files = sorted(subset_dir.glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found under {subset_dir}")

    if cfg.geometry_ids is not None:
        allowed = {int(x) for x in cfg.geometry_ids}
        mat_files = [p for p in mat_files if p.stem.isdigit() and int(p.stem) in allowed]

    if not mat_files:
        raise ValueError("No matching geometry files after geometry_ids filtering.")

    if cfg.snr_estimator is None and cfg.min_snr_db is not None:
        # If user wants SNR filtering but did not supply an estimator,
        # use a simple magnitude-ratio proxy against the chosen reference.
        cfg.snr_estimator = default_snr_proxy

    samples: List[Dict[str, Any]] = []
    kept_files: List[str] = []

    for mat_path in mat_files:
        d = _load_mat_file(mat_path)

        freq_points_hz = np.asarray(d.get("freqPoints"), dtype=float).reshape(-1)
        if freq_points_hz.size == 0:
            continue

        freq_idx = _nearest_freq_index(freq_points_hz, cfg.freq_hz)
        freq_selected_hz = float(freq_points_hz[freq_idx])

        if abs(freq_selected_hz - cfg.freq_hz) > cfg.freq_tol_hz:
            continue

        geometry = d.get("geometry", {})
        geometry_meta = _extract_geometry_meta(geometry)
        if not _passes_geometry_filter(geometry_meta, cfg):
            continue

        patterns_struct = d.get("patterns", {})
        results_struct = d.get("results", {})
        reference_struct = d.get("reference", {})

        ris_patterns = _normalize_patterns(_safe_get(patterns_struct, cfg.result_key))
        result_entries = _extract_result_list(_safe_get(results_struct, cfg.result_key))

        # Reference trace: may be a nested struct or direct array
        ref_entry = _safe_get(reference_struct, cfg.reference_key)
        reference_trace = _extract_complex_trace(ref_entry)

        # Align lengths conservatively
        n = min(len(ris_patterns), len(result_entries)) if ris_patterns else len(result_entries)
        if n == 0:
            continue

        if not ris_patterns:
            # Some result families may not expose per-pattern arrays in an obvious form.
            # Keep placeholder None for pattern if unavailable.
            ris_patterns = [None] * n

        if cfg.max_pattern_samples is not None:
            n = min(n, int(cfg.max_pattern_samples))

        geometry_id = int(mat_path.stem) if mat_path.stem.isdigit() else mat_path.stem

        for idx in range(n):
            result_entry = result_entries[idx]
            s21_trace = _extract_complex_trace(result_entry)

            sample = {
                "geometry_id": geometry_id,
                "subset": cfg.subset,
                "file_path": str(mat_path),
                "geometry_meta": geometry_meta,
                "freq_points_hz": freq_points_hz,
                "freq_idx": freq_idx,
                "freq_selected_hz": freq_selected_hz,
                "pattern_index": idx,
                "ris_pattern": ris_patterns[idx] if idx < len(ris_patterns) else None,
                "s21_trace": s21_trace,
                "reference_trace": reference_trace,
                "snr_db": None,
            }

            if cfg.snr_estimator is not None:
                try:
                    sample["snr_db"] = cfg.snr_estimator(sample)
                except Exception:
                    sample["snr_db"] = None

            if cfg.min_snr_db is not None:
                snr_db = sample["snr_db"]
                if snr_db is None or snr_db < cfg.min_snr_db:
                    continue

            samples.append(sample)

        kept_files.append(str(mat_path))

    if not samples:
        raise ValueError(
            "No samples matched the requested configuration. "
            "Relax geometry/frequency/SNR filters and try again."
        )

    summary = {
        "num_files": len(kept_files),
        "num_samples": len(samples),
        "subset": cfg.subset,
        "result_key": cfg.result_key,
        "reference_key": cfg.reference_key,
        "freq_hz_target": cfg.freq_hz,
        "freq_tol_hz": cfg.freq_tol_hz,
        "snr_filter_enabled": cfg.min_snr_db is not None,
    }

    return {
        "config": cfg,
        "files": kept_files,
        "samples": samples,
        "summary": summary,
    }
