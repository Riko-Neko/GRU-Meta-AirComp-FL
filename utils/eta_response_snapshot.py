import os
from typing import Dict, Optional

import numpy as np


def complex_nmse_per_user(est, target, eps: float = 1e-12) -> np.ndarray:
    est_arr = np.asarray(est, dtype=np.complex64)
    target_arr = np.asarray(target, dtype=np.complex64)
    if est_arr.ndim == 1:
        est_arr = est_arr.reshape(1, -1)
    if target_arr.ndim == 1:
        target_arr = target_arr.reshape(1, -1)
    err_power = np.sum(np.abs(est_arr - target_arr) ** 2, axis=1, dtype=np.float64)
    ref_power = np.sum(np.abs(target_arr) ** 2, axis=1, dtype=np.float64)
    return err_power / (ref_power + float(eps))


def build_eta_components(h_eff, user_weights, update_vars, tx_power: float, eps: float = 1e-8) -> Dict[str, np.ndarray]:
    h_eff_arr = np.asarray(h_eff, dtype=np.complex64).reshape(-1)
    weight_arr = np.asarray(user_weights, dtype=np.float64).reshape(-1)
    var_arr = np.asarray(update_vars, dtype=np.float64).reshape(-1)
    abs2 = np.abs(h_eff_arr) ** 2
    eta_k = float(tx_power) * abs2 / (np.square(weight_arr) * var_arr + float(eps))
    return {
        "h_eff_abs2_k": abs2.astype(np.float64),
        "eta_k": eta_k.astype(np.float64),
        "v_k": var_arr.astype(np.float64),
    }


def save_snapshot_npz(run_dir: str, round_idx: int, payload: Dict[str, object], meta: Optional[Dict[str, object]] = None) -> str:
    os.makedirs(run_dir, exist_ok=True)
    if meta is not None:
        meta_path = os.path.join(run_dir, "run_meta.npz")
        if not os.path.exists(meta_path):
            np.savez_compressed(meta_path, **meta)

    out_path = os.path.join(run_dir, f"round_{int(round_idx):04d}.npz")
    np.savez_compressed(out_path, **payload)
    return out_path
