from __future__ import annotations

import csv
import itertools
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class GroupingSCAConfig:
    """
    Hyper-parameters for the two-group robust risk grouping problem.

    The implementation follows the user's proposed formulation:

        min_x,mu  sum_k (1-x_k)(|r_k-mu_1|+eps_k)^2
                + sum_k x_k(|r_k-mu_2|+eps_k)^2
                - lambda_s (mu_2 - mu_1)
                + gamma sum_{i<j} (1-|x_i-x_j|)(1-rho_ij)
                + lambda_h sum_k |x_k-x_k_prev|
                + tau sum_k x_k (1-x_k)

    solved via SCA with a linear-program subproblem per outer iteration.
    """

    lambda_d: float = 1.0
    c_beta: float = 1.0
    c_d: float = 1.0
    lambda_s: float = 0.25
    gamma: float = 0.10
    lambda_h: float = 0.10
    tau: float = 0.20
    eps_beta: float = 1e-12
    eps_d: float = 1e-12
    eps_rho: float = 1e-8
    k_min: int = 1
    sca_max_iters: int = 30
    sca_tol: float = 1e-4
    relaxation: float = 1.0
    lp_method: str = "highs"

    def validate(self, num_users: int) -> None:
        if int(num_users) <= 1:
            raise ValueError("num_users must be greater than 1")
        if int(self.k_min) <= 0:
            raise ValueError("k_min must be positive")
        if 2 * int(self.k_min) > int(num_users):
            raise ValueError(f"k_min={self.k_min} is infeasible for num_users={num_users}")
        if float(self.eps_beta) <= 0.0:
            raise ValueError("eps_beta must be positive")
        if float(self.eps_d) <= 0.0:
            raise ValueError("eps_d must be positive")
        if float(self.eps_rho) <= 0.0:
            raise ValueError("eps_rho must be positive")
        if int(self.sca_max_iters) <= 0:
            raise ValueError("sca_max_iters must be positive")
        if float(self.sca_tol) <= 0.0:
            raise ValueError("sca_tol must be positive")
        if not (0.0 < float(self.relaxation) <= 1.0):
            raise ValueError("relaxation must lie in (0, 1]")


@dataclass
class GroupingWarmStart:
    x_prev: Optional[np.ndarray] = None
    beta_ema: Optional[np.ndarray] = None
    d_ema: Optional[np.ndarray] = None
    x_init: Optional[np.ndarray] = None
    mu_init: Optional[Sequence[float]] = None


@dataclass
class GroupingResult:
    x_soft: np.ndarray
    x_hard: np.ndarray
    mu: np.ndarray
    nominal_risk: np.ndarray
    risk_radius: np.ndarray
    proxy_features: np.ndarray
    compatibility: np.ndarray
    beta_ema: np.ndarray
    d_ema: np.ndarray
    objective_history: np.ndarray
    converged: bool
    iterations: int
    lp_status: str
    lp_success: bool
    group_low: np.ndarray
    group_high: np.ndarray


@dataclass
class SweepScoreWeights:
    risk_gap: float = 1.0
    top_risk_precision: float = 0.75
    compatibility: float = 0.25
    migration: float = 0.25
    balance: float = 0.10


@dataclass
class _LPVarIndex:
    x: slice
    mu1: int
    mu2: int
    q1: slice
    q2: slice
    u: slice
    s: slice
    num_vars: int


def _normalize_complex(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.complex128).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= float(eps):
        return np.ones_like(arr, dtype=np.complex128) / np.sqrt(max(1, arr.size))
    return arr / norm


def _pair_indices(num_users: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(num_users) for j in range(i + 1, num_users)]


def _build_lp_index(num_users: int, pairs: Sequence[Tuple[int, int]]) -> _LPVarIndex:
    start = 0
    x = slice(start, start + num_users)
    start = x.stop
    mu1 = start
    mu2 = start + 1
    start = mu2 + 1
    q1 = slice(start, start + num_users)
    start = q1.stop
    q2 = slice(start, start + num_users)
    start = q2.stop
    u = slice(start, start + len(pairs))
    start = u.stop
    s = slice(start, start + num_users)
    start = s.stop
    return _LPVarIndex(x=x, mu1=mu1, mu2=mu2, q1=q1, q2=q2, u=u, s=s, num_vars=start)


def build_proxy_features(
        beta_hat: np.ndarray,
        d_hat: np.ndarray,
        *,
        eps_beta: float = 1e-12,
        eps_d: float = 1e-12,
) -> np.ndarray:
    beta_arr = np.maximum(np.asarray(beta_hat, dtype=np.float64).reshape(-1), float(eps_beta))
    d_arr = np.maximum(np.asarray(d_hat, dtype=np.float64).reshape(-1), 0.0)
    z_beta = -np.log(beta_arr)
    z_d = np.log1p(d_arr)
    return np.stack([z_beta, z_d], axis=1)


def build_nominal_risk(
        beta_hat: np.ndarray,
        d_hat: np.ndarray,
        *,
        lambda_d: float = 1.0,
        eps_beta: float = 1e-12,
        eps_d: float = 1e-12,
) -> np.ndarray:
    z = build_proxy_features(beta_hat, d_hat, eps_beta=eps_beta, eps_d=eps_d)
    return z[:, 0] + float(lambda_d) * z[:, 1]


def build_risk_radius(
        beta_hat: np.ndarray,
        d_hat: np.ndarray,
        beta_ema: np.ndarray,
        d_ema: np.ndarray,
        *,
        c_beta: float = 1.0,
        c_d: float = 1.0,
        eps_beta: float = 1e-12,
        eps_d: float = 1e-12,
) -> np.ndarray:
    beta_now = np.maximum(np.asarray(beta_hat, dtype=np.float64).reshape(-1), float(eps_beta))
    d_now = np.maximum(np.asarray(d_hat, dtype=np.float64).reshape(-1), 0.0)
    beta_avg = np.maximum(np.asarray(beta_ema, dtype=np.float64).reshape(-1), float(eps_beta))
    d_avg = np.maximum(np.asarray(d_ema, dtype=np.float64).reshape(-1), 0.0)
    if beta_now.shape != beta_avg.shape or d_now.shape != d_avg.shape or beta_now.shape != d_now.shape:
        raise ValueError("beta_hat/d_hat and EMA arrays must share the same shape")
    beta_term = np.abs((-np.log(beta_now)) - (-np.log(beta_avg)))
    d_term = np.abs(np.log1p(d_now) - np.log1p(d_avg))
    return float(c_beta) * beta_term + float(c_d) * d_term


def build_compatibility_matrix(
        H_BR: np.ndarray,
        f_prev: np.ndarray,
        h_ru_est: np.ndarray,
        *,
        eps_rho: float = 1e-8,
) -> np.ndarray:
    h_br = np.asarray(H_BR, dtype=np.complex128)
    h_ru = np.asarray(h_ru_est, dtype=np.complex128)
    if h_br.ndim != 2:
        raise ValueError("H_BR must be a 2-D matrix")
    if h_ru.ndim != 2:
        raise ValueError("h_ru_est must be a 2-D array with shape [K, N]")
    if h_br.shape[0] != h_ru.shape[1]:
        raise ValueError(f"H_BR shape {h_br.shape} is incompatible with h_ru_est shape {h_ru.shape}")
    f_arr = np.asarray(f_prev, dtype=np.complex128)
    if f_arr.ndim == 1:
        f_vec = _normalize_complex(f_arr, eps=eps_rho)
        if h_br.shape[1] != f_vec.size:
            raise ValueError(f"H_BR shape {h_br.shape} is incompatible with f_prev size {f_vec.size}")
        cascaded_prefix = np.conj(h_br @ f_vec).reshape(1, -1)
        signatures = cascaded_prefix * h_ru
    elif f_arr.ndim == 2:
        if f_arr.shape[0] != h_ru.shape[0] or f_arr.shape[1] != h_br.shape[1]:
            raise ValueError(
                f"H_BR shape {h_br.shape}, f_prev shape {f_arr.shape}, "
                f"and h_ru_est shape {h_ru.shape} are incompatible"
            )
        f_norm = np.stack([_normalize_complex(row, eps=eps_rho) for row in f_arr], axis=0)
        cascaded_prefix = np.conj(f_norm @ h_br.T)
        signatures = cascaded_prefix * h_ru
    else:
        raise ValueError("f_prev must be a beamforming vector [M] or per-user beamformer matrix [K, M]")
    norms = np.linalg.norm(signatures, axis=1)
    gram = signatures @ signatures.conj().T
    denom = norms[:, None] * norms[None, :] + float(eps_rho)
    rho = np.abs(gram) / denom
    rho = np.clip(rho.real, 0.0, 1.0)
    np.fill_diagonal(rho, 1.0)
    return rho


def initialize_grouping_from_nominal_risk(risk: np.ndarray, k_min: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constrained 1-D two-means initialization.

    Since the nominal risk is scalar, the optimal split for 2-means with a size
    constraint is contiguous after sorting.
    """

    risk_arr = np.asarray(risk, dtype=np.float64).reshape(-1)
    num_users = risk_arr.size
    if 2 * int(k_min) > num_users:
        raise ValueError(f"k_min={k_min} is infeasible for num_users={num_users}")

    order = np.argsort(risk_arr, kind="mergesort")
    sorted_risk = risk_arr[order]
    prefix = np.concatenate([[0.0], np.cumsum(sorted_risk)])
    prefix2 = np.concatenate([[0.0], np.cumsum(np.square(sorted_risk))])

    best_cut = None
    best_sse = None
    for cut in range(int(k_min), num_users - int(k_min) + 1):
        count_low = cut
        count_high = num_users - cut

        sum_low = prefix[cut] - prefix[0]
        sum_high = prefix[num_users] - prefix[cut]
        sum2_low = prefix2[cut] - prefix2[0]
        sum2_high = prefix2[num_users] - prefix2[cut]

        sse_low = sum2_low - (sum_low * sum_low) / max(1, count_low)
        sse_high = sum2_high - (sum_high * sum_high) / max(1, count_high)
        total_sse = float(sse_low + sse_high)
        if best_sse is None or total_sse < best_sse:
            best_sse = total_sse
            best_cut = cut

    if best_cut is None:
        raise RuntimeError("failed to initialize the 2-group split")

    low_ids = order[:best_cut]
    high_ids = order[best_cut:]
    x_init = np.zeros((num_users,), dtype=np.float64)
    x_init[high_ids] = 1.0

    mu1 = float(np.mean(risk_arr[low_ids]))
    mu2 = float(np.mean(risk_arr[high_ids]))
    return x_init, np.asarray([mu1, max(mu1, mu2)], dtype=np.float64)


def evaluate_grouping_objective(
        x_soft: np.ndarray,
        mu: np.ndarray,
        nominal_risk: np.ndarray,
        risk_radius: np.ndarray,
        compatibility: np.ndarray,
        x_prev: np.ndarray,
        cfg: GroupingSCAConfig,
) -> float:
    x_arr = np.clip(np.asarray(x_soft, dtype=np.float64).reshape(-1), 0.0, 1.0)
    mu_arr = np.asarray(mu, dtype=np.float64).reshape(2)
    risk_arr = np.asarray(nominal_risk, dtype=np.float64).reshape(-1)
    radius_arr = np.asarray(risk_radius, dtype=np.float64).reshape(-1)
    prev_arr = np.clip(np.asarray(x_prev, dtype=np.float64).reshape(-1), 0.0, 1.0)
    rho = np.asarray(compatibility, dtype=np.float64)

    term_low = np.sum((1.0 - x_arr) * np.square(np.abs(risk_arr - mu_arr[0]) + radius_arr))
    term_high = np.sum(x_arr * np.square(np.abs(risk_arr - mu_arr[1]) + radius_arr))

    compat_term = 0.0
    for i, j in _pair_indices(x_arr.size):
        compat_term += (1.0 - abs(x_arr[i] - x_arr[j])) * (1.0 - float(rho[i, j]))

    migrate_term = np.sum(np.abs(x_arr - prev_arr))
    shrink_term = np.sum(x_arr * (1.0 - x_arr))

    return float(
        term_low
        + term_high
        - float(cfg.lambda_s) * (mu_arr[1] - mu_arr[0])
        + float(cfg.gamma) * compat_term
        + float(cfg.lambda_h) * migrate_term
        + float(cfg.tau) * shrink_term
    )


def _project_previous_grouping(x_prev: Optional[np.ndarray], num_users: int, k_min: int) -> np.ndarray:
    if x_prev is None:
        raise ValueError("x_prev cannot be None here")
    x_arr = np.clip(np.asarray(x_prev, dtype=np.float64).reshape(-1), 0.0, 1.0)
    if x_arr.size != num_users:
        raise ValueError(f"x_prev has size {x_arr.size}, expected {num_users}")
    total = float(np.sum(x_arr))
    if total < float(k_min):
        order = np.argsort(-x_arr)
        x_arr[:] = 0.0
        x_arr[order[:k_min]] = 1.0
    elif total > float(num_users - k_min):
        order = np.argsort(x_arr)
        x_arr[:] = 1.0
        x_arr[order[:k_min]] = 0.0
    return x_arr


def _repair_hard_grouping(x_soft: np.ndarray, k_min: int) -> np.ndarray:
    x_arr = np.clip(np.asarray(x_soft, dtype=np.float64).reshape(-1), 0.0, 1.0)
    num_users = x_arr.size
    hard = (x_arr >= 0.5).astype(np.int64)
    num_high = int(np.sum(hard))

    if num_high < int(k_min):
        promote_order = np.argsort(-x_arr)
        for idx in promote_order:
            if hard[idx] == 0:
                hard[idx] = 1
                num_high += 1
                if num_high >= int(k_min):
                    break
    elif num_high > int(num_users - k_min):
        demote_order = np.argsort(x_arr)
        for idx in demote_order:
            if hard[idx] == 1:
                hard[idx] = 0
                num_high -= 1
                if num_high <= int(num_users - k_min):
                    break
    return hard.astype(np.int64)


def _solve_linearized_subproblem(
        x_curr: np.ndarray,
        q1_curr: np.ndarray,
        q2_curr: np.ndarray,
        nominal_risk: np.ndarray,
        risk_radius: np.ndarray,
        compatibility: np.ndarray,
        x_prev: np.ndarray,
        cfg: GroupingSCAConfig,
) -> Tuple[np.ndarray, np.ndarray, str, bool]:
    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        raise ImportError(
            "SciPy is required for the SCA grouping optimizer. Install scipy in the project environment."
        ) from exc

    num_users = int(np.asarray(nominal_risk).size)
    pairs = _pair_indices(num_users)
    index = _build_lp_index(num_users, pairs)

    x_curr = np.asarray(x_curr, dtype=np.float64).reshape(-1)
    q1_curr = np.asarray(q1_curr, dtype=np.float64).reshape(-1)
    q2_curr = np.asarray(q2_curr, dtype=np.float64).reshape(-1)
    risk_arr = np.asarray(nominal_risk, dtype=np.float64).reshape(-1)
    radius_arr = np.asarray(risk_radius, dtype=np.float64).reshape(-1)
    prev_arr = np.asarray(x_prev, dtype=np.float64).reshape(-1)

    c = np.zeros((index.num_vars,), dtype=np.float64)

    c[index.mu1] = float(cfg.lambda_s)
    c[index.mu2] = -float(cfg.lambda_s)

    for k in range(num_users):
        q1_shift = q1_curr[k] + radius_arr[k]
        q2_shift = q2_curr[k] + radius_arr[k]
        c[index.x.start + k] = (
                -(q1_shift ** 2)
                + (q2_shift ** 2)
                + float(cfg.tau) * (1.0 - 2.0 * x_curr[k])
        )
        c[index.q1.start + k] = 2.0 * (1.0 - x_curr[k]) * q1_shift
        c[index.q2.start + k] = 2.0 * x_curr[k] * q2_shift
        c[index.s.start + k] = float(cfg.lambda_h)

    for pair_offset, (i, j) in enumerate(pairs):
        c[index.u.start + pair_offset] = -float(cfg.gamma) * (1.0 - float(compatibility[i, j]))

    A_ub: List[np.ndarray] = []
    b_ub: List[float] = []

    # Group-size constraints.
    row = np.zeros((index.num_vars,), dtype=np.float64)
    row[index.x] = 1.0
    A_ub.append(row)
    b_ub.append(float(num_users - cfg.k_min))

    row = np.zeros((index.num_vars,), dtype=np.float64)
    row[index.x] = -1.0
    A_ub.append(row)
    b_ub.append(float(-cfg.k_min))

    # mu_2 >= mu_1.
    row = np.zeros((index.num_vars,), dtype=np.float64)
    row[index.mu1] = 1.0
    row[index.mu2] = -1.0
    A_ub.append(row)
    b_ub.append(0.0)

    # q_{k,g} >= +/- (r_k - mu_g)
    for k in range(num_users):
        # q1_k >= r_k - mu1  => -q1_k - mu1 <= -r_k
        row = np.zeros((index.num_vars,), dtype=np.float64)
        row[index.q1.start + k] = -1.0
        row[index.mu1] = -1.0
        A_ub.append(row)
        b_ub.append(-float(risk_arr[k]))

        # q1_k >= mu1 - r_k => -q1_k + mu1 <= r_k
        row = np.zeros((index.num_vars,), dtype=np.float64)
        row[index.q1.start + k] = -1.0
        row[index.mu1] = 1.0
        A_ub.append(row)
        b_ub.append(float(risk_arr[k]))

        # q2_k >= r_k - mu2  => -q2_k - mu2 <= -r_k
        row = np.zeros((index.num_vars,), dtype=np.float64)
        row[index.q2.start + k] = -1.0
        row[index.mu2] = -1.0
        A_ub.append(row)
        b_ub.append(-float(risk_arr[k]))

        # q2_k >= mu2 - r_k => -q2_k + mu2 <= r_k
        row = np.zeros((index.num_vars,), dtype=np.float64)
        row[index.q2.start + k] = -1.0
        row[index.mu2] = 1.0
        A_ub.append(row)
        b_ub.append(float(risk_arr[k]))

    # u_ij >= |x_i - x_j|
    for pair_offset, (i, j) in enumerate(pairs):
        row = np.zeros((index.num_vars,), dtype=np.float64)
        row[index.x.start + i] = 1.0
        row[index.x.start + j] = -1.0
        row[index.u.start + pair_offset] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

        row = np.zeros((index.num_vars,), dtype=np.float64)
        row[index.x.start + i] = -1.0
        row[index.x.start + j] = 1.0
        row[index.u.start + pair_offset] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    # s_k >= |x_k - x_prev_k|
    for k in range(num_users):
        row = np.zeros((index.num_vars,), dtype=np.float64)
        row[index.x.start + k] = 1.0
        row[index.s.start + k] = -1.0
        A_ub.append(row)
        b_ub.append(float(prev_arr[k]))

        row = np.zeros((index.num_vars,), dtype=np.float64)
        row[index.x.start + k] = -1.0
        row[index.s.start + k] = -1.0
        A_ub.append(row)
        b_ub.append(float(-prev_arr[k]))

    bounds: List[Tuple[Optional[float], Optional[float]]] = []
    bounds.extend([(0.0, 1.0) for _ in range(num_users)])
    bounds.append((None, None))  # mu1
    bounds.append((None, None))  # mu2
    bounds.extend([(0.0, None) for _ in range(num_users)])  # q1
    bounds.extend([(0.0, None) for _ in range(num_users)])  # q2
    bounds.extend([(0.0, 1.0) for _ in pairs])  # u
    bounds.extend([(0.0, 1.0) for _ in range(num_users)])  # s

    result = linprog(
        c=c,
        A_ub=np.asarray(A_ub, dtype=np.float64),
        b_ub=np.asarray(b_ub, dtype=np.float64),
        bounds=bounds,
        method=str(cfg.lp_method),
    )
    if result.x is None:
        raise RuntimeError(f"linprog failed without a primal point: {result.message}")
    return np.asarray(result.x, dtype=np.float64), c, str(result.message), bool(result.success)


def optimize_risk_grouping_sca(
        beta_hat: np.ndarray,
        d_hat: np.ndarray,
        H_BR: np.ndarray,
        f_prev: np.ndarray,
        h_ru_est: np.ndarray,
        cfg: GroupingSCAConfig,
        *,
        warm_start: Optional[GroupingWarmStart] = None,
) -> GroupingResult:
    beta_arr = np.maximum(np.asarray(beta_hat, dtype=np.float64).reshape(-1), float(cfg.eps_beta))
    d_arr = np.maximum(np.asarray(d_hat, dtype=np.float64).reshape(-1), 0.0)
    num_users = int(beta_arr.size)
    cfg.validate(num_users)

    if d_arr.size != num_users:
        raise ValueError("beta_hat and d_hat must share the same number of users")

    if warm_start is None:
        warm_start = GroupingWarmStart()

    beta_ema = beta_arr.copy() if warm_start.beta_ema is None else np.maximum(
        np.asarray(warm_start.beta_ema, dtype=np.float64).reshape(-1),
        float(cfg.eps_beta),
    )
    d_ema = d_arr.copy() if warm_start.d_ema is None else np.maximum(
        np.asarray(warm_start.d_ema, dtype=np.float64).reshape(-1),
        0.0,
    )
    if beta_ema.size != num_users or d_ema.size != num_users:
        raise ValueError("EMA arrays must have the same length as beta_hat/d_hat")

    proxy_features = build_proxy_features(beta_arr, d_arr, eps_beta=cfg.eps_beta, eps_d=cfg.eps_d)
    nominal_risk = build_nominal_risk(
        beta_arr,
        d_arr,
        lambda_d=cfg.lambda_d,
        eps_beta=cfg.eps_beta,
        eps_d=cfg.eps_d,
    )
    risk_radius = build_risk_radius(
        beta_arr,
        d_arr,
        beta_ema,
        d_ema,
        c_beta=cfg.c_beta,
        c_d=cfg.c_d,
        eps_beta=cfg.eps_beta,
        eps_d=cfg.eps_d,
    )
    compatibility = build_compatibility_matrix(H_BR, f_prev, h_ru_est, eps_rho=cfg.eps_rho)

    if warm_start.x_prev is None:
        x_prev, _ = initialize_grouping_from_nominal_risk(nominal_risk, cfg.k_min)
    else:
        x_prev = _project_previous_grouping(warm_start.x_prev, num_users, cfg.k_min)

    if warm_start.x_init is None:
        x_curr = x_prev.copy()
    else:
        x_curr = _project_previous_grouping(warm_start.x_init, num_users, cfg.k_min)

    if warm_start.mu_init is None:
        _, mu_seed = initialize_grouping_from_nominal_risk(nominal_risk, cfg.k_min)
        mu_curr = mu_seed.copy()
    else:
        mu_curr = np.asarray(warm_start.mu_init, dtype=np.float64).reshape(2)
        mu_curr[1] = max(mu_curr[1], mu_curr[0])

    q1_curr = np.abs(nominal_risk - mu_curr[0])
    q2_curr = np.abs(nominal_risk - mu_curr[1])
    objective_history: List[float] = [
        evaluate_grouping_objective(x_curr, mu_curr, nominal_risk, risk_radius, compatibility, x_prev, cfg)
    ]

    converged = False
    lp_status = "not_started"
    lp_success = False

    for _iter in range(int(cfg.sca_max_iters)):
        solution, _, lp_status, lp_success = _solve_linearized_subproblem(
            x_curr=x_curr,
            q1_curr=q1_curr,
            q2_curr=q2_curr,
            nominal_risk=nominal_risk,
            risk_radius=risk_radius,
            compatibility=compatibility,
            x_prev=x_prev,
            cfg=cfg,
        )
        if not lp_success:
            raise RuntimeError(f"SCA LP subproblem failed: {lp_status}")

        pairs = _pair_indices(num_users)
        index = _build_lp_index(num_users, pairs)
        x_lp = np.clip(solution[index.x], 0.0, 1.0)
        mu_lp = np.asarray([solution[index.mu1], solution[index.mu2]], dtype=np.float64)
        mu_lp[1] = max(mu_lp[1], mu_lp[0])

        relax = float(cfg.relaxation)
        x_next = relax * x_lp + (1.0 - relax) * x_curr
        mu_next = relax * mu_lp + (1.0 - relax) * mu_curr
        mu_next[1] = max(mu_next[1], mu_next[0])

        q1_next = np.abs(nominal_risk - mu_next[0])
        q2_next = np.abs(nominal_risk - mu_next[1])
        objective_history.append(
            evaluate_grouping_objective(x_next, mu_next, nominal_risk, risk_radius, compatibility, x_prev, cfg)
        )

        if float(np.linalg.norm(x_next - x_curr)) <= float(cfg.sca_tol):
            x_curr = x_next
            mu_curr = mu_next
            q1_curr = q1_next
            q2_curr = q2_next
            converged = True
            break

        x_curr = x_next
        mu_curr = mu_next
        q1_curr = q1_next
        q2_curr = q2_next

    x_hard = _repair_hard_grouping(x_curr, cfg.k_min)
    group_low = np.flatnonzero(x_hard == 0).astype(np.int64) + 1
    group_high = np.flatnonzero(x_hard == 1).astype(np.int64) + 1

    return GroupingResult(
        x_soft=x_curr.astype(np.float64, copy=False),
        x_hard=x_hard.astype(np.int64, copy=False),
        mu=mu_curr.astype(np.float64, copy=False),
        nominal_risk=nominal_risk.astype(np.float64, copy=False),
        risk_radius=risk_radius.astype(np.float64, copy=False),
        proxy_features=proxy_features.astype(np.float64, copy=False),
        compatibility=compatibility.astype(np.float64, copy=False),
        beta_ema=beta_ema.astype(np.float64, copy=False),
        d_ema=d_ema.astype(np.float64, copy=False),
        objective_history=np.asarray(objective_history, dtype=np.float64),
        converged=bool(converged),
        iterations=len(objective_history) - 1,
        lp_status=lp_status,
        lp_success=bool(lp_success),
        group_low=group_low,
        group_high=group_high,
    )


def _relative_channel_delta_norm(
        h_ref: np.ndarray,
        h_next: np.ndarray,
        *,
        eps_d: float = 1e-12,
) -> np.ndarray:
    ref = np.asarray(h_ref, dtype=np.complex128)
    nxt = np.asarray(h_next, dtype=np.complex128)
    if ref.shape != nxt.shape:
        raise ValueError("h_ref and h_next must share the same shape")
    delta = np.linalg.norm(nxt - ref, axis=1)
    denom = np.linalg.norm(nxt, axis=1) + float(eps_d)
    return (delta / denom).astype(np.float64, copy=False)


def _build_smoke_test_case(
        smoke_cfg,
        *,
        seed: int = 7,
        round_idx: int = 3,
        ema_alpha: float = 0.8,
        estimate_noise_scale: float = 0.05,
) -> Dict[str, np.ndarray]:
    from data.channel import build_ru_channel_evolver_from_config

    if int(round_idx) < 2:
        raise ValueError("round_idx must be at least 2 so that EMA and previous grouping are defined")

    rng = np.random.default_rng(int(seed))
    np.random.seed(int(seed))

    evolver = build_ru_channel_evolver_from_config(smoke_cfg)
    H_BR = evolver.initialize_br_channel()
    h_ru_state, _ = evolver.initialize_user_channels(include_direct=False)

    f_prev = _normalize_complex(np.ones((int(smoke_cfg.num_bs_antennas),), dtype=np.complex128)).astype(np.complex64)
    rho = float(smoke_cfg.uplink_tau_ratio)

    beta_prev = None
    d_prev = None
    h_ru_tau_curr = None
    beta_curr = None
    d_curr = None

    for step_round in range(1, int(round_idx) + 1):
        h_ru_t = h_ru_state.copy()
        step = evolver.step_split(h_ru_state, step_round, rho, h_BUs=None)
        beta_round = np.asarray(
            evolver.ris_pathloss(evolver.positions_at(float(step_round - 1) + rho)),
            dtype=np.float64,
        )
        d_round = _relative_channel_delta_norm(h_ru_t, step.h_ru_tau, eps_d=1e-12)

        if step_round == int(round_idx) - 1:
            beta_prev = beta_round.copy()
            d_prev = d_round.copy()
        if step_round == int(round_idx):
            beta_curr = beta_round.copy()
            d_curr = d_round.copy()
            h_ru_tau_curr = step.h_ru_tau.copy()

        h_ru_state = step.h_ru_next

    if beta_prev is None or d_prev is None or beta_curr is None or d_curr is None or h_ru_tau_curr is None:
        raise RuntimeError("failed to construct smoke-test proxies")

    beta_hat = np.maximum(beta_curr * np.exp(0.03 * rng.standard_normal(beta_curr.shape)), 1e-12)
    d_hat = np.maximum(d_curr * np.exp(0.05 * rng.standard_normal(d_curr.shape)), 0.0)
    beta_ema = float(ema_alpha) * beta_prev + (1.0 - float(ema_alpha)) * beta_hat
    d_ema = float(ema_alpha) * d_prev + (1.0 - float(ema_alpha)) * d_hat

    signal_scale = np.linalg.norm(h_ru_tau_curr, axis=1, keepdims=True) / np.sqrt(max(1, h_ru_tau_curr.shape[1]))
    signal_scale = np.maximum(signal_scale, 1e-8)
    noise = (
                    rng.standard_normal(h_ru_tau_curr.shape) + 1j * rng.standard_normal(h_ru_tau_curr.shape)
            ) / np.sqrt(2.0)
    h_ru_est = h_ru_tau_curr + float(estimate_noise_scale) * signal_scale * noise

    return {
        "H_BR": np.asarray(H_BR, dtype=np.complex64),
        "f_prev": np.asarray(f_prev, dtype=np.complex64),
        "h_ru_est": np.asarray(h_ru_est, dtype=np.complex64),
        "beta_prev": np.asarray(beta_prev, dtype=np.float64),
        "d_prev": np.asarray(d_prev, dtype=np.float64),
        "beta_hat": np.asarray(beta_hat, dtype=np.float64),
        "d_hat": np.asarray(d_hat, dtype=np.float64),
        "beta_ema": np.asarray(beta_ema, dtype=np.float64),
        "d_ema": np.asarray(d_ema, dtype=np.float64),
        "cluster_ids": np.asarray(evolver.cluster_ids, dtype=np.int64),
        "positions_tau": np.asarray(evolver.positions_at(float(round_idx - 1) + rho), dtype=np.float64),
        "speed_mps": np.asarray(evolver.speed_magnitudes, dtype=np.float64),
        "round_idx": np.asarray(int(round_idx), dtype=np.int64),
    }


def _format_vector(arr: np.ndarray, precision: int = 4) -> str:
    return np.array2string(np.asarray(arr), precision=precision, separator=", ", suppress_small=False)


def _mean_intragroup_compatibility(compatibility: np.ndarray, hard_assign: np.ndarray, group_value: int) -> float:
    group_ids = np.flatnonzero(np.asarray(hard_assign, dtype=np.int64) == int(group_value))
    if group_ids.size <= 1:
        return 1.0
    vals: List[float] = []
    for local_i in range(group_ids.size):
        for local_j in range(local_i + 1, group_ids.size):
            vals.append(float(compatibility[group_ids[local_i], group_ids[local_j]]))
    if not vals:
        return 1.0
    return float(np.mean(vals))


def _build_trial_warm_start(
        smoke_case: Dict[str, np.ndarray],
        grouping_cfg: GroupingSCAConfig,
) -> GroupingWarmStart:
    x_prev, mu_prev = initialize_grouping_from_nominal_risk(
        build_nominal_risk(
            smoke_case["beta_prev"],
            smoke_case["d_prev"],
            lambda_d=grouping_cfg.lambda_d,
            eps_beta=grouping_cfg.eps_beta,
            eps_d=grouping_cfg.eps_d,
        ),
        k_min=grouping_cfg.k_min,
    )
    return GroupingWarmStart(
        x_prev=x_prev,
        beta_ema=smoke_case["beta_ema"],
        d_ema=smoke_case["d_ema"],
        mu_init=mu_prev,
    )


def _run_single_trial(
        smoke_case: Dict[str, np.ndarray],
        grouping_cfg: GroupingSCAConfig,
) -> Tuple[GroupingResult, np.ndarray]:
    warm_start = _build_trial_warm_start(smoke_case, grouping_cfg)
    result = optimize_risk_grouping_sca(
        beta_hat=smoke_case["beta_hat"],
        d_hat=smoke_case["d_hat"],
        H_BR=smoke_case["H_BR"],
        f_prev=smoke_case["f_prev"],
        h_ru_est=smoke_case["h_ru_est"],
        cfg=grouping_cfg,
        warm_start=warm_start,
    )
    if warm_start.x_prev is None:
        raise RuntimeError("warm_start.x_prev should not be None")
    return result, np.asarray(warm_start.x_prev, dtype=np.float64)


def _summarize_trial(
        result: GroupingResult,
        x_prev: np.ndarray,
        score_weights: SweepScoreWeights,
) -> Dict[str, float]:
    hard = np.asarray(result.x_hard, dtype=np.int64)
    risk = np.asarray(result.nominal_risk, dtype=np.float64)
    num_users = hard.size
    num_high = int(np.sum(hard))
    num_low = int(num_users - num_high)

    high_ids = np.flatnonzero(hard == 1)
    low_ids = np.flatnonzero(hard == 0)
    if high_ids.size == 0 or low_ids.size == 0:
        raise RuntimeError("invalid hard partition: one group is empty")

    mean_risk_high = float(np.mean(risk[high_ids]))
    mean_risk_low = float(np.mean(risk[low_ids]))
    risk_gap = mean_risk_high - mean_risk_low

    top_ids = np.argsort(-risk, kind="mergesort")[:num_high]
    top_risk_precision = float(np.mean(hard[top_ids]))

    compat_high = _mean_intragroup_compatibility(result.compatibility, hard, 1)
    compat_low = _mean_intragroup_compatibility(result.compatibility, hard, 0)
    compat_mean = 0.5 * (compat_high + compat_low)

    migration_rate = float(np.mean(np.abs(hard.astype(np.float64) - np.asarray(x_prev, dtype=np.float64))))
    balance_score = 1.0 - abs(float(num_high) - 0.5 * float(num_users)) / max(1.0, 0.5 * float(num_users))

    objective_init = float(result.objective_history[0])
    objective_final = float(result.objective_history[-1])
    objective_delta = objective_init - objective_final

    sweep_score = (
            float(score_weights.risk_gap) * risk_gap
            + float(score_weights.top_risk_precision) * top_risk_precision
            + float(score_weights.compatibility) * compat_mean
            - float(score_weights.migration) * migration_rate
            + float(score_weights.balance) * balance_score
    )
    return {
        "num_high": float(num_high),
        "num_low": float(num_low),
        "risk_gap": float(risk_gap),
        "mean_risk_high": mean_risk_high,
        "mean_risk_low": mean_risk_low,
        "top_risk_precision": top_risk_precision,
        "compat_high": float(compat_high),
        "compat_low": float(compat_low),
        "compat_mean": float(compat_mean),
        "migration_rate": migration_rate,
        "balance_score": float(balance_score),
        "objective_init": objective_init,
        "objective_final": objective_final,
        "objective_delta": objective_delta,
        "sweep_score": float(sweep_score),
    }


def _print_single_trial(
        seed: int,
        smoke_cfg,
        grouping_cfg: GroupingSCAConfig,
        smoke_case: Dict[str, np.ndarray],
        result: GroupingResult,
        x_prev: np.ndarray,
) -> None:
    cluster_ids = np.asarray(smoke_case["cluster_ids"], dtype=np.int64)
    unique_clusters = np.unique(cluster_ids)

    print("=== Grouping Optimizer Smoke Test ===")
    print(
        f"seed={int(seed)}, round_idx={int(smoke_case['round_idx'])}, "
        f"num_users={int(smoke_cfg.num_users)}, k_min={int(grouping_cfg.k_min)}"
    )
    print(
        "smoke_cfg="
        f"{{M={int(smoke_cfg.num_bs_antennas)}, N={int(smoke_cfg.num_ris_elements)}, "
        f"clusters={smoke_cfg.user_cluster_ratios}, speed={smoke_cfg.user_speed_range}, "
        f"dir={smoke_cfg.user_motion_direction_deg}, move_mask={smoke_cfg.user_speed_user_mask}}}"
    )
    print(
        "grouping_cfg="
        f"{{lambda_d={grouping_cfg.lambda_d}, lambda_s={grouping_cfg.lambda_s}, gamma={grouping_cfg.gamma}, "
        f"lambda_h={grouping_cfg.lambda_h}, tau={grouping_cfg.tau}, "
        f"sca_max_iters={grouping_cfg.sca_max_iters}, sca_tol={grouping_cfg.sca_tol}}}"
    )
    print(f"cluster_ids={cluster_ids.tolist()}")
    print(f"speed_mps={_format_vector(smoke_case['speed_mps'], precision=3)}")
    print(f"beta_hat={_format_vector(smoke_case['beta_hat'], precision=6)}")
    print(f"d_hat={_format_vector(smoke_case['d_hat'], precision=6)}")
    print(f"nominal_risk={_format_vector(result.nominal_risk, precision=6)}")
    print(f"risk_radius={_format_vector(result.risk_radius, precision=6)}")
    print(f"x_prev={_format_vector(x_prev, precision=3)}")
    print(f"x_soft={_format_vector(result.x_soft, precision=6)}")
    print(f"x_hard={result.x_hard.tolist()}")
    print(f"group_low(user ids)={result.group_low.tolist()}")
    print(f"group_high(user ids)={result.group_high.tolist()}")
    print(f"mu={_format_vector(result.mu, precision=6)}")
    print(f"objective_history={_format_vector(result.objective_history, precision=6)}")
    print(f"converged={result.converged}, iterations={result.iterations}, lp_status={result.lp_status}")

    if unique_clusters.size == 2:
        cluster_risk_means = {}
        for cluster_id in unique_clusters.tolist():
            mask = cluster_ids == int(cluster_id)
            cluster_risk_means[int(cluster_id)] = float(np.mean(result.nominal_risk[mask]))
        high_cluster = max(cluster_risk_means.items(), key=lambda item: item[1])[0]
        true_high = (cluster_ids == int(high_cluster)).astype(np.int64)
        agreement = float(np.mean(true_high == result.x_hard))
        print(f"cluster_mean_risk={cluster_risk_means}")
        print(f"true_high_cluster={int(high_cluster)}")
        print(f"true_high_mask={true_high.tolist()}")
        print(f"hard_group_agreement_vs_cluster={agreement:.4f}")
    else:
        print("cluster_mean_risk=skipped (smoke-test comparison expects exactly 2 true clusters)")


def _print_sweep_results(
        ranked_records: List[Dict[str, object]],
        *,
        top_k: int,
) -> None:
    print("=== Grouping Parameter Sweep ===")
    print(
        "rank | score | lambda_d | gamma | lambda_h | tau | k_min | num_high | "
        "risk_gap | top_prec | compat | migration | obj_delta | group_high"
    )
    for rank, record in enumerate(ranked_records[: int(top_k)], start=1):
        cfg = record["grouping_cfg"]
        metrics = record["metrics"]
        result = record["result"]
        print(
            f"{rank:>4d} | "
            f"{metrics['sweep_score']:.4f} | "
            f"{cfg.lambda_d:>8.3f} | "
            f"{cfg.gamma:>5.3f} | "
            f"{cfg.lambda_h:>8.3f} | "
            f"{cfg.tau:>4.3f} | "
            f"{int(cfg.k_min):>5d} | "
            f"{int(metrics['num_high']):>8d} | "
            f"{metrics['risk_gap']:.4f} | "
            f"{metrics['top_risk_precision']:.4f} | "
            f"{metrics['compat_mean']:.4f} | "
            f"{metrics['migration_rate']:.4f} | "
            f"{metrics['objective_delta']:.4f} | "
            f"{result.group_high.tolist()}"
        )


def _to_builtin(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(val) for val in value]
    return value


def _make_grouping_debug_dir(seed: int, round_idx: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = Path(__file__).resolve().parents[1] / "debug" / "grouping_optim" / (
        f"seed{int(seed)}_round{int(round_idx)}_{stamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_sweep_csv(
        csv_path: Path,
        ranked_records: List[Dict[str, object]],
) -> None:
    fieldnames = [
        "rank",
        "lambda_d",
        "lambda_s",
        "gamma",
        "lambda_h",
        "tau",
        "c_beta",
        "c_d",
        "k_min",
        "relaxation",
        "sca_max_iters",
        "sca_tol",
        "num_high",
        "num_low",
        "risk_gap",
        "mean_risk_high",
        "mean_risk_low",
        "top_risk_precision",
        "compat_high",
        "compat_low",
        "compat_mean",
        "migration_rate",
        "balance_score",
        "objective_init",
        "objective_final",
        "objective_delta",
        "sweep_score",
        "converged",
        "iterations",
        "lp_success",
        "lp_status",
        "group_low",
        "group_high",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for rank, record in enumerate(ranked_records, start=1):
            cfg = record["grouping_cfg"]
            result = record["result"]
            metrics = record["metrics"]
            writer.writerow(
                {
                    "rank": int(rank),
                    "lambda_d": float(cfg.lambda_d),
                    "lambda_s": float(cfg.lambda_s),
                    "gamma": float(cfg.gamma),
                    "lambda_h": float(cfg.lambda_h),
                    "tau": float(cfg.tau),
                    "c_beta": float(cfg.c_beta),
                    "c_d": float(cfg.c_d),
                    "k_min": int(cfg.k_min),
                    "relaxation": float(cfg.relaxation),
                    "sca_max_iters": int(cfg.sca_max_iters),
                    "sca_tol": float(cfg.sca_tol),
                    "num_high": int(metrics["num_high"]),
                    "num_low": int(metrics["num_low"]),
                    "risk_gap": float(metrics["risk_gap"]),
                    "mean_risk_high": float(metrics["mean_risk_high"]),
                    "mean_risk_low": float(metrics["mean_risk_low"]),
                    "top_risk_precision": float(metrics["top_risk_precision"]),
                    "compat_high": float(metrics["compat_high"]),
                    "compat_low": float(metrics["compat_low"]),
                    "compat_mean": float(metrics["compat_mean"]),
                    "migration_rate": float(metrics["migration_rate"]),
                    "balance_score": float(metrics["balance_score"]),
                    "objective_init": float(metrics["objective_init"]),
                    "objective_final": float(metrics["objective_final"]),
                    "objective_delta": float(metrics["objective_delta"]),
                    "sweep_score": float(metrics["sweep_score"]),
                    "converged": bool(result.converged),
                    "iterations": int(result.iterations),
                    "lp_success": bool(result.lp_success),
                    "lp_status": str(result.lp_status),
                    "group_low": json.dumps(result.group_low.tolist(), ensure_ascii=True),
                    "group_high": json.dumps(result.group_high.tolist(), ensure_ascii=True),
                }
            )


def _write_sweep_summary(
        summary_path: Path,
        best_path: Path,
        *,
        seed: int,
        round_idx: int,
        run_dir: Path,
        smoke_cfg,
        sweep_grid: Dict[str, List[object]],
        sweep_score_weights: SweepScoreWeights,
        total_sweep_trials: int,
        ranked_records: List[Dict[str, object]],
) -> None:
    best_record = ranked_records[0]
    best_cfg = best_record["grouping_cfg"]
    best_result = best_record["result"]
    best_metrics = best_record["metrics"]

    summary = {
        "seed": int(seed),
        "round_idx": int(round_idx),
        "run_dir": str(run_dir),
        "total_sweep_trials": int(total_sweep_trials),
        "smoke_cfg": _to_builtin(vars(smoke_cfg)),
        "sweep_grid": _to_builtin(sweep_grid),
        "sweep_score_weights": _to_builtin(sweep_score_weights.__dict__),
        "best_grouping_cfg": _to_builtin(best_cfg.__dict__),
        "best_metrics": _to_builtin(best_metrics),
        "best_group_low": _to_builtin(best_result.group_low),
        "best_group_high": _to_builtin(best_result.group_high),
        "best_lp_status": str(best_result.lp_status),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    best_trial = {
        "grouping_cfg": _to_builtin(best_cfg.__dict__),
        "metrics": _to_builtin(best_metrics),
        "x_prev": _to_builtin(best_record["x_prev"]),
        "x_soft": _to_builtin(best_result.x_soft),
        "x_hard": _to_builtin(best_result.x_hard),
        "mu": _to_builtin(best_result.mu),
        "group_low": _to_builtin(best_result.group_low),
        "group_high": _to_builtin(best_result.group_high),
        "nominal_risk": _to_builtin(best_result.nominal_risk),
        "risk_radius": _to_builtin(best_result.risk_radius),
        "objective_history": _to_builtin(best_result.objective_history),
        "converged": bool(best_result.converged),
        "iterations": int(best_result.iterations),
        "lp_status": str(best_result.lp_status),
        "lp_success": bool(best_result.lp_success),
    }
    best_path.write_text(json.dumps(best_trial, indent=2), encoding="utf-8")


def _save_sweep_plots(
        run_dir: Path,
        ranked_records: List[Dict[str, object]],
) -> None:
    try:
        mpl_cache_dir = Path(__file__).resolve().parents[1] / "debug" / "grouping_optim" / ".mplconfig"
        mpl_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required to render sweep visualizations for grouping_optim."
        ) from exc

    sweep_scores = np.asarray([record["metrics"]["sweep_score"] for record in ranked_records], dtype=np.float64)
    risk_gaps = np.asarray([record["metrics"]["risk_gap"] for record in ranked_records], dtype=np.float64)
    top_precisions = np.asarray(
        [record["metrics"]["top_risk_precision"] for record in ranked_records],
        dtype=np.float64,
    )
    compat_means = np.asarray([record["metrics"]["compat_mean"] for record in ranked_records], dtype=np.float64)
    migration_rates = np.asarray(
        [record["metrics"]["migration_rate"] for record in ranked_records],
        dtype=np.float64,
    )
    objective_deltas = np.asarray(
        [record["metrics"]["objective_delta"] for record in ranked_records],
        dtype=np.float64,
    )
    lambda_ds = np.asarray([record["grouping_cfg"].lambda_d for record in ranked_records], dtype=np.float64)
    k_mins = np.asarray([record["grouping_cfg"].k_min for record in ranked_records], dtype=np.float64)

    num_bins = int(np.clip(np.sqrt(max(1, sweep_scores.size)), 10, 80))

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.hist(sweep_scores, bins=num_bins, color="#1f77b4", alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_title("Grouping Sweep Score Distribution")
    ax.set_xlabel("Sweep Score")
    ax.set_ylabel("Trial Count")
    ax.grid(alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(run_dir / "sweep_score_hist.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    scatter = ax.scatter(
        risk_gaps,
        top_precisions,
        c=sweep_scores,
        s=20,
        alpha=0.55,
        cmap="viridis",
        edgecolors="none",
    )
    ax.set_title("Risk Gap vs Top-Risk Precision")
    ax.set_xlabel("Risk Gap")
    ax.set_ylabel("Top-Risk Precision")
    ax.grid(alpha=0.25, linestyle=":")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Sweep Score")
    fig.tight_layout()
    fig.savefig(run_dir / "risk_gap_vs_top_precision.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    scatter = ax.scatter(
        lambda_ds,
        sweep_scores,
        c=k_mins,
        s=22,
        alpha=0.55,
        cmap="plasma",
        edgecolors="none",
    )
    ax.set_title("lambda_d vs Sweep Score")
    ax.set_xlabel("lambda_d")
    ax.set_ylabel("Sweep Score")
    ax.grid(alpha=0.25, linestyle=":")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("k_min")
    fig.tight_layout()
    fig.savefig(run_dir / "lambda_d_vs_sweep_score.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    scatter = ax.scatter(
        compat_means,
        migration_rates,
        c=objective_deltas,
        s=22,
        alpha=0.55,
        cmap="coolwarm",
        edgecolors="none",
    )
    ax.set_title("Compatibility vs Migration")
    ax.set_xlabel("Mean Intra-Group Compatibility")
    ax.set_ylabel("Migration Rate")
    ax.grid(alpha=0.25, linestyle=":")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Objective Delta (init - final)")
    fig.tight_layout()
    fig.savefig(run_dir / "compatibility_vs_migration.png", dpi=180)
    plt.close(fig)


def _save_sweep_artifacts(
        *,
        seed: int,
        round_idx: int,
        smoke_cfg,
        sweep_grid: Dict[str, List[object]],
        sweep_score_weights: SweepScoreWeights,
        total_sweep_trials: int,
        ranked_records: List[Dict[str, object]],
        run_dir: Optional[Path] = None,
) -> Path:
    if run_dir is None:
        run_dir = _make_grouping_debug_dir(seed=seed, round_idx=round_idx)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_sweep_csv(run_dir / "sweep_results.csv", ranked_records)
    _write_sweep_summary(
        run_dir / "summary.json",
        run_dir / "best_trial.json",
        seed=seed,
        round_idx=round_idx,
        run_dir=run_dir,
        smoke_cfg=smoke_cfg,
        sweep_grid=sweep_grid,
        sweep_score_weights=sweep_score_weights,
        total_sweep_trials=total_sweep_trials,
        ranked_records=ranked_records,
    )
    _save_sweep_plots(run_dir, ranked_records)
    return run_dir


def _value_signature(value) -> object:
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 12)
    return value


def _build_refined_axis_values(base_values: Sequence[object], center_value: object) -> List[object]:
    values = list(base_values)
    if not values:
        raise ValueError("base_values must be non-empty")

    int_like = all(isinstance(v, (int, np.integer)) for v in values)
    numeric_values = sorted({_value_signature(v) for v in values}, key=float)
    center = _value_signature(center_value)
    if center not in numeric_values:
        numeric_values.append(center)
        numeric_values = sorted(numeric_values, key=float)
    center_idx = numeric_values.index(center)

    refined = {center}
    if center_idx > 0:
        left = numeric_values[center_idx - 1]
        refined.add(left)
        if int_like:
            for val in range(int(min(left, center)) + 1, int(max(left, center))):
                refined.add(int(val))
        else:
            refined.add(0.5 * (float(left) + float(center)))
    if center_idx + 1 < len(numeric_values):
        right = numeric_values[center_idx + 1]
        refined.add(right)
        if int_like:
            for val in range(int(min(center, right)) + 1, int(max(center, right))):
                refined.add(int(val))
        else:
            refined.add(0.5 * (float(center) + float(right)))

    if int_like:
        return sorted({int(v) for v in refined})
    return sorted({float(v) for v in refined})


def _build_refined_trial_configs(
        grouping_cfg_base: GroupingSCAConfig,
        sweep_grid: Dict[str, List[object]],
        ranked_records: List[Dict[str, object]],
        *,
        refine_top_k: int,
        refine_max_order: int = 2,
        include_center: bool = True,
) -> List[GroupingSCAConfig]:
    sweep_keys = list(sweep_grid.keys())
    max_order = max(1, int(refine_max_order))
    seen = set()
    refined_cfgs: List[GroupingSCAConfig] = []

    for record in ranked_records[: int(refine_top_k)]:
        center_cfg = record["grouping_cfg"]
        center_values = {key: getattr(center_cfg, key) for key in sweep_keys}
        axis_candidates = {
            key: _build_refined_axis_values(sweep_grid[key], center_values[key])
            for key in sweep_keys
        }

        def _append_cfg(values_dict: Dict[str, object]) -> None:
            cfg_kwargs = dict(grouping_cfg_base.__dict__)
            cfg_kwargs.update(values_dict)
            signature = tuple(_value_signature(cfg_kwargs[key]) for key in sweep_keys)
            if signature in seen:
                return
            seen.add(signature)
            refined_cfgs.append(GroupingSCAConfig(**cfg_kwargs))

        if include_center:
            _append_cfg(center_values)

        varying_keys = [key for key in sweep_keys if len(axis_candidates[key]) > 1]
        for order in range(1, min(max_order, len(varying_keys)) + 1):
            for key_subset in itertools.combinations(varying_keys, order):
                subset_values: List[List[object]] = []
                valid_subset = True
                for key in key_subset:
                    non_center_values = [
                        value
                        for value in axis_candidates[key]
                        if _value_signature(value) != _value_signature(center_values[key])
                    ]
                    if not non_center_values:
                        valid_subset = False
                        break
                    subset_values.append(non_center_values)
                if not valid_subset:
                    continue

                for combo in itertools.product(*subset_values):
                    trial_values = dict(center_values)
                    for key, value in zip(key_subset, combo):
                        trial_values[key] = value
                    _append_cfg(trial_values)

    return refined_cfgs


def _sort_sweep_records(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        records,
        key=lambda item: (
            -float(item["metrics"]["sweep_score"]),
            -float(item["metrics"]["risk_gap"]),
            -float(item["metrics"]["top_risk_precision"]),
        ),
    )


def _evaluate_sweep_configs(
        smoke_case: Dict[str, np.ndarray],
        trial_cfgs: Sequence[GroupingSCAConfig],
        sweep_score_weights: SweepScoreWeights,
        *,
        progress_desc: str,
) -> List[Dict[str, object]]:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    cfg_list = list(trial_cfgs)
    cfg_iter = cfg_list
    if tqdm is not None:
        cfg_iter = tqdm(cfg_list, total=len(cfg_list), desc=progress_desc, unit="trial")

    records: List[Dict[str, object]] = []
    for trial_cfg in cfg_iter:
        result, x_prev = _run_single_trial(smoke_case, trial_cfg)
        metrics = _summarize_trial(result, x_prev, sweep_score_weights)
        records.append(
            {
                "grouping_cfg": trial_cfg,
                "result": result,
                "x_prev": x_prev,
                "metrics": metrics,
            }
        )
    return records


def _smoke_test_main() -> None:
    # Editable smoke-test geometry and mobility block.
    smoke_cfg = SimpleNamespace(
        num_users=10,
        num_bs_antennas=32,
        num_ris_elements=64,
        bs_position_xy=[0.0, 0.0],
        ris_position_xy=[30.0, 0.0],
        user_cluster_ratios=[0.5, 0.5],
        user_cluster_centers_xy=[[50.0, 8.0], [90.0, -12]],
        user_cluster_position_jitter_xy=[[10.0, 10.0], [10.0, 10.0]],
        user_speed_range=[10.0, 20.0],
        user_motion_direction_deg=None,
        user_speed_user_mask=[9, 10],
        alpha_direct=2.0,
        channel_ref_scale=float(np.sqrt(1e-10)),
        channel_time_step=1e-3,
        channel_carrier_frequency_hz=3.5e9,
        channel_min_distance=1.0,
        uplink_tau_ratio=0.5,
    )

    # Editable smoke-test synthesis block.
    seed = 7
    round_idx = 5
    ema_alpha = 0.8
    estimate_noise_scale = 0.05

    run_mode = "sweep"  # "single" | "sweep"

    # Editable grouping hyper-parameter block.
    grouping_cfg_base = GroupingSCAConfig(
        lambda_d=7.0,
        c_beta=1.0,
        c_d=1.0,
        lambda_s=0.25,
        gamma=0.10,
        lambda_h=0.10,
        tau=0.20,
        k_min=max(1, int(smoke_cfg.num_users) // 4),
        sca_max_iters=30,
        sca_tol=1e-4,
        relaxation=1.0,
        lp_method="highs",
    )

    # Editable sweep block.
    sweep_score_weights = SweepScoreWeights(
        risk_gap=1.0,
        top_risk_precision=0.75,
        compatibility=0.25,
        migration=0.25,
        balance=0.10,
    )
    sweep_grid = {
        "lambda_d": [0.5, 1.0, 2.0, 4.0, 6.0, 8.0],
        "lambda_s": [0.0, 0.1, 0.25, 0.5],
        "gamma": [0.0, 0.05, 0.1, 0.2],
        "lambda_h": [0.0, 0.05, 0.1, 0.2],
        "tau": [0.0, 0.1, 0.2, 0.4],
        "c_beta": [0.5, 1.0],
        "c_d": [0.5, 1.0],
        "k_min": [2, 3, 4, 5],
        "relaxation": [0.7, 1.0],
    }
    sweep_top_k = 20
    refine_enabled = True
    refine_top_k = 8
    refine_max_order = 2

    smoke_case = _build_smoke_test_case(
        smoke_cfg,
        seed=int(seed),
        round_idx=int(round_idx),
        ema_alpha=float(ema_alpha),
        estimate_noise_scale=float(estimate_noise_scale),
    )
    if str(run_mode).strip().lower() == "single":
        result, x_prev = _run_single_trial(smoke_case, grouping_cfg_base)
        _print_single_trial(int(seed), smoke_cfg, grouping_cfg_base, smoke_case, result, x_prev)
        return

    if str(run_mode).strip().lower() != "sweep":
        raise ValueError("run_mode must be 'single' or 'sweep'")

    sweep_keys = list(sweep_grid.keys())
    sweep_values = [list(sweep_grid[key]) for key in sweep_keys]
    total_sweep_trials = int(np.prod([len(v) for v in sweep_values], dtype=np.int64))

    coarse_trial_cfgs: List[GroupingSCAConfig] = []
    for combo in itertools.product(*sweep_values):
        trial_kwargs = dict(grouping_cfg_base.__dict__)
        for key, value in zip(sweep_keys, combo):
            trial_kwargs[key] = value
        coarse_trial_cfgs.append(GroupingSCAConfig(**trial_kwargs))

    coarse_records = _evaluate_sweep_configs(
        smoke_case,
        coarse_trial_cfgs,
        sweep_score_weights,
        progress_desc="Grouping coarse sweep",
    )
    coarse_ranked_records = _sort_sweep_records(coarse_records)
    debug_root_dir = _make_grouping_debug_dir(seed=int(seed), round_idx=int(round_idx))

    print(f"seed={int(seed)}")
    print(
        "smoke_cfg="
        f"{{M={int(smoke_cfg.num_bs_antennas)}, N={int(smoke_cfg.num_ris_elements)}, "
        f"clusters={smoke_cfg.user_cluster_ratios}, speed={smoke_cfg.user_speed_range}, "
        f"dir={smoke_cfg.user_motion_direction_deg}, move_mask={smoke_cfg.user_speed_user_mask}}}"
    )
    print(
        "sweep_score_weights="
        f"{{risk_gap={sweep_score_weights.risk_gap}, top_risk_precision={sweep_score_weights.top_risk_precision}, "
        f"compatibility={sweep_score_weights.compatibility}, migration={sweep_score_weights.migration}, "
        f"balance={sweep_score_weights.balance}}}"
    )
    print(f"sweep_grid={sweep_grid}")
    print(f"total_sweep_trials={total_sweep_trials}")
    print(
        f"refine_enabled={bool(refine_enabled)}, refine_top_k={int(refine_top_k)}, "
        f"refine_max_order={int(refine_max_order)}"
    )
    print("")
    print("=== Coarse Sweep ===")
    _print_sweep_results(coarse_ranked_records, top_k=sweep_top_k)

    coarse_debug_dir = _save_sweep_artifacts(
        seed=int(seed),
        round_idx=int(round_idx),
        smoke_cfg=smoke_cfg,
        sweep_grid=sweep_grid,
        sweep_score_weights=sweep_score_weights,
        total_sweep_trials=total_sweep_trials,
        ranked_records=coarse_ranked_records,
        run_dir=debug_root_dir / "coarse",
    )
    print(f"coarse_debug_dir={coarse_debug_dir}")

    best_record = coarse_ranked_records[0]
    if bool(refine_enabled):
        refined_trial_cfgs = _build_refined_trial_configs(
            grouping_cfg_base,
            sweep_grid,
            coarse_ranked_records,
            refine_top_k=int(refine_top_k),
            refine_max_order=int(refine_max_order),
            include_center=True,
        )
        refined_total_trials = len(refined_trial_cfgs)
        refined_records = _evaluate_sweep_configs(
            smoke_case,
            refined_trial_cfgs,
            sweep_score_weights,
            progress_desc="Grouping refine sweep",
        )
        refined_ranked_records = _sort_sweep_records(refined_records)

        print("")
        print("=== Refine Sweep ===")
        print(f"refined_total_trials={int(refined_total_trials)}")
        _print_sweep_results(refined_ranked_records, top_k=sweep_top_k)

        refine_debug_dir = _save_sweep_artifacts(
            seed=int(seed),
            round_idx=int(round_idx),
            smoke_cfg=smoke_cfg,
            sweep_grid={"refined_from_top_k": [int(refine_top_k)], **sweep_grid},
            sweep_score_weights=sweep_score_weights,
            total_sweep_trials=refined_total_trials,
            ranked_records=refined_ranked_records,
            run_dir=debug_root_dir / "refine",
        )
        print(f"refine_debug_dir={refine_debug_dir}")
        best_record = refined_ranked_records[0]

    print("")
    print("=== Best Sweep Trial Detail ===")
    _print_single_trial(
        int(seed),
        smoke_cfg,
        best_record["grouping_cfg"],
        smoke_case,
        best_record["result"],
        best_record["x_prev"],
    )


if __name__ == "__main__":
    _smoke_test_main()
