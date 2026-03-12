"""
Time-varying channel evolution utilities.

Extracts the per-round AR(1) update for the RIS->UE link h_RU,
and supports an optional dynamic alpha(t) trajectory controlled by Config.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np


@dataclass
class AlphaTrajectoryConfig:
    """Configuration for alpha(t) used in AR(1) channel evolution."""
    use_dynamic_alpha: bool = False
    static_alpha: float = 0.8

    # Dynamic alpha trajectory
    dynamic_alpha_mode: str = "sinusoid"  # "sinusoid" | "piecewise"

    # Sinusoidal alpha(t) parameters
    alpha_min: float = 0.50
    alpha_max: float = 0.98
    alpha_period_rounds: int = 20

    # Piecewise alpha(t) parameters: list of (end_round_inclusive, alpha_value)
    alpha_piecewise: Optional[List[Tuple[int, float]]] = None


class AlphaTrajectory:
    """Deterministic alpha(t) generator (shared across users by default)."""

    def __init__(self, cfg: AlphaTrajectoryConfig):
        self.cfg = cfg
        self._validate()

    def _validate(self) -> None:
        if not (0.0 <= self.cfg.static_alpha < 1.0):
            raise ValueError(f"static_alpha must be in [0,1), got {self.cfg.static_alpha}")

        if self.cfg.use_dynamic_alpha:
            mode = self.cfg.dynamic_alpha_mode.lower()
            if mode not in {"sinusoid", "piecewise"}:
                raise ValueError(f"Unsupported dynamic_alpha_mode: {self.cfg.dynamic_alpha_mode}")

            if mode == "sinusoid":
                if self.cfg.alpha_period_rounds <= 0:
                    raise ValueError("alpha_period_rounds must be positive")
                if not (0.0 <= self.cfg.alpha_min <= self.cfg.alpha_max < 1.0):
                    raise ValueError("Require 0 <= alpha_min <= alpha_max < 1")

            if mode == "piecewise":
                if not self.cfg.alpha_piecewise:
                    raise ValueError("alpha_piecewise must be provided for piecewise mode")
                ends = [e for e, _ in self.cfg.alpha_piecewise]
                if any(ends[i] <= ends[i - 1] for i in range(1, len(ends))):
                    raise ValueError("alpha_piecewise end_rounds must be strictly increasing")
                for _, a in self.cfg.alpha_piecewise:
                    if not (0.0 <= a < 1.0):
                        raise ValueError("alpha_piecewise values must be in [0,1)")

    @staticmethod
    def _clip_alpha(alpha: float) -> float:
        # Avoid sqrt(1-alpha^2) going negative due to numerical issues near 1
        return float(min(max(alpha, 0.0), 0.999999))

    def get_alpha(self, round_idx: int) -> float:
        """Return alpha(t) for a given round index (1-based)."""
        if not self.cfg.use_dynamic_alpha:
            return self._clip_alpha(self.cfg.static_alpha)

        t = max(int(round_idx), 1)
        mode = self.cfg.dynamic_alpha_mode.lower()

        if mode == "sinusoid":
            # Smooth periodic alpha(t): high alpha => slow fading; low alpha => fast fading
            a_min, a_max = self.cfg.alpha_min, self.cfg.alpha_max
            a_mean = 0.5 * (a_min + a_max)
            a_amp = 0.5 * (a_max - a_min)
            phase = 2.0 * math.pi * (t - 1) / float(self.cfg.alpha_period_rounds)
            alpha = a_mean + a_amp * math.sin(phase)
            return self._clip_alpha(alpha)

        # piecewise
        for end_round, alpha in self.cfg.alpha_piecewise or []:
            if t <= end_round:
                return self._clip_alpha(alpha)
        return self._clip_alpha((self.cfg.alpha_piecewise or [(-1, self.cfg.static_alpha)])[-1][1])


class RUChannelEvolverAR1:
    """
    AR(1) evolver for RIS->UE channels h_RU.
    h_RU[k] <- alpha(t)*h_RU[k] + sqrt(1-alpha(t)^2) * w,   w ~ CN(0, I)
    """

    def __init__(self, num_ris_elements, traj, *, alpha_bases=None, base_static_alpha=0.8):
        self.num_ris_elements = int(num_ris_elements)
        self.traj = traj
        self.alpha_bases = None if alpha_bases is None else np.asarray(alpha_bases, dtype=float)
        self.base_static_alpha = float(base_static_alpha)

    @staticmethod
    def _validate_step_ratio(step_ratio: float) -> float:
        ratio = float(step_ratio)
        if not (0.0 <= ratio <= 1.0):
            raise ValueError(f"step_ratio must be in [0, 1], got {step_ratio}")
        return ratio

    def _alpha_vec_for_round(self, round_idx: int, num_users: int) -> np.ndarray:
        alpha_global = self.traj.get_alpha(round_idx)

        # alpha_k(t) = clip(alpha_k_base + (alpha_global(t) - base_static_alpha))
        if self.alpha_bases is None:
            return np.full((num_users,), alpha_global, dtype=float)

        delta = float(alpha_global - self.base_static_alpha)
        return np.array([self.traj._clip_alpha(a + delta) for a in self.alpha_bases[:num_users]], dtype=float)

    def alpha_vec_at(self, round_idx: int, num_users: int) -> np.ndarray:
        return self._alpha_vec_for_round(round_idx, num_users)

    def alpha_at_user(self, round_idx: int, user_idx: int) -> float:
        return float(self._alpha_vec_for_round(round_idx, user_idx + 1)[user_idx])

    @staticmethod
    def _fractional_alpha(alpha_delta, step_ratio: float):
        ratio = float(step_ratio)
        if ratio <= 0.0:
            return np.ones_like(alpha_delta, dtype=float)
        if ratio >= 1.0:
            return np.asarray(alpha_delta, dtype=float).copy()
        return np.power(np.asarray(alpha_delta, dtype=float), ratio)

    def step(self, h_RUs: np.ndarray, round_idx: int):
        """
        Evolve h_RUs one step.
        Args:
            h_RUs: shape (K, N), complex
            round_idx: 1-based communication round
        Returns:
            (h_RUs_new, alpha_used)
        """
        alpha_vec = self._alpha_vec_for_round(round_idx, h_RUs.shape[0])

        beta_vec = np.sqrt(np.maximum(0.0, 1.0 - alpha_vec * alpha_vec)).astype(float)
        noise = (np.random.randn(*h_RUs.shape) + 1j * np.random.randn(*h_RUs.shape)) / np.sqrt(2.0)

        h_new = (alpha_vec[:, None] * h_RUs) + (beta_vec[:, None] * noise.astype(h_RUs.dtype, copy=False))
        return h_new.astype(h_RUs.dtype, copy=False), alpha_vec

    def step_split(self, h_RUs: np.ndarray, round_idx: int, tau_ratio: float):
        """
        Split one full downlink interval into t+tau and t+1 on a consistent path.
        Args:
            h_RUs: shape (K, N), complex
            round_idx: 1-based communication round
            tau_ratio: tau / delta_t in [0, 1]
        Returns:
            (h_tau, h_next, alpha_delta, alpha_tau)
        """
        rho = self._validate_step_ratio(tau_ratio)
        alpha_delta = self._alpha_vec_for_round(round_idx, h_RUs.shape[0])
        alpha_tau = self._fractional_alpha(alpha_delta, rho)
        alpha_rem = self._fractional_alpha(alpha_delta, 1.0 - rho)

        beta_tau = np.sqrt(np.maximum(0.0, 1.0 - alpha_tau * alpha_tau)).astype(float)
        noise_tau = (np.random.randn(*h_RUs.shape) + 1j * np.random.randn(*h_RUs.shape)) / np.sqrt(2.0)
        h_tau = (alpha_tau[:, None] * h_RUs) + (beta_tau[:, None] * noise_tau.astype(h_RUs.dtype, copy=False))

        beta_rem = np.sqrt(np.maximum(0.0, 1.0 - alpha_rem * alpha_rem)).astype(float)
        noise_rem = (np.random.randn(*h_RUs.shape) + 1j * np.random.randn(*h_RUs.shape)) / np.sqrt(2.0)
        h_next = (alpha_rem[:, None] * h_tau) + (beta_rem[:, None] * noise_rem.astype(h_RUs.dtype, copy=False))

        return (
            h_tau.astype(h_RUs.dtype, copy=False),
            h_next.astype(h_RUs.dtype, copy=False),
            alpha_delta.astype(float, copy=False),
            np.asarray(alpha_tau, dtype=float),
        )

    def step_single_split(self, h_ru: np.ndarray, user_idx: int, round_idx: int, tau_ratio: float):
        """
        Single-user split-step evolution matching step_split semantics.
        Returns:
            (h_tau, h_next, alpha_delta, alpha_tau)
        """
        rho = self._validate_step_ratio(tau_ratio)
        alpha_delta = self.alpha_at_user(round_idx, user_idx)
        alpha_tau = float(self._fractional_alpha(np.asarray([alpha_delta], dtype=float), rho)[0])
        alpha_rem = float(self._fractional_alpha(np.asarray([alpha_delta], dtype=float), 1.0 - rho)[0])

        beta_tau = math.sqrt(max(0.0, 1.0 - alpha_tau * alpha_tau))
        noise_tau = (np.random.randn(*h_ru.shape) + 1j * np.random.randn(*h_ru.shape)) / math.sqrt(2.0)
        h_tau = alpha_tau * h_ru + beta_tau * noise_tau.astype(h_ru.dtype, copy=False)

        beta_rem = math.sqrt(max(0.0, 1.0 - alpha_rem * alpha_rem))
        noise_rem = (np.random.randn(*h_ru.shape) + 1j * np.random.randn(*h_ru.shape)) / math.sqrt(2.0)
        h_next = alpha_rem * h_tau + beta_rem * noise_rem.astype(h_ru.dtype, copy=False)

        return (
            h_tau.astype(h_ru.dtype, copy=False),
            h_next.astype(h_ru.dtype, copy=False),
            float(alpha_delta),
            float(alpha_tau),
        )


def build_ru_channel_evolver_from_config(config):
    traj_cfg = AlphaTrajectoryConfig(
        use_dynamic_alpha=config.use_dynamic_alpha,
        static_alpha=config.channel_alpha,
        dynamic_alpha_mode=config.dynamic_alpha_mode,
        alpha_min=config.alpha_min,
        alpha_max=config.alpha_max,
        alpha_period_rounds=config.alpha_period_rounds,
        alpha_piecewise=config.alpha_piecewise,
    )
    traj = AlphaTrajectory(traj_cfg)

    alpha_bases = None
    if config.use_user_alpha_hetero:
        a_min = float(config.alpha_user_min)
        a_max = float(config.alpha_user_max)
        K = int(config.num_users)
        alpha_bases = np.random.uniform(a_min, a_max, size=(K,)).astype(float)

    return RUChannelEvolverAR1(
        num_ris_elements=config.num_ris_elements,
        traj=traj,
        alpha_bases=alpha_bases,
        base_static_alpha=config.channel_alpha,
    )
