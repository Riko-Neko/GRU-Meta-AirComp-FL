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

    def step(self, h_RUs: np.ndarray, round_idx: int):
        """
        Evolve h_RUs one step.
        Args:
            h_RUs: shape (K, N), complex
            round_idx: 1-based communication round
        Returns:
            (h_RUs_new, alpha_used)
        """
        alpha_global = self.traj.get_alpha(round_idx)

        # alpha_k(t) = clip(alpha_k_base + (alpha_global(t) - base_static_alpha))
        if self.alpha_bases is None:
            alpha_vec = np.full((h_RUs.shape[0],), alpha_global, dtype=float)
        else:
            delta = float(alpha_global - self.base_static_alpha)
            alpha_vec = np.array([self.traj._clip_alpha(a + delta) for a in self.alpha_bases], dtype=float)

        beta_vec = np.sqrt(np.maximum(0.0, 1.0 - alpha_vec * alpha_vec)).astype(float)
        noise = (np.random.randn(*h_RUs.shape) + 1j * np.random.randn(*h_RUs.shape)) / np.sqrt(2.0)

        h_new = (alpha_vec[:, None] * h_RUs) + (beta_vec[:, None] * noise.astype(h_RUs.dtype, copy=False))
        return h_new.astype(h_RUs.dtype, copy=False), alpha_vec


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
