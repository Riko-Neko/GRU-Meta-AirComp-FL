"""
Mobility-driven time-varying channel evolution utilities.

This module replaces the old hand-crafted alpha(t) scheduler with an explicit
user-mobility model:

- user positions evolve from initial coordinates and velocity vectors;
- path-loss is recomputed from the updated positions every step;
- small-scale fading follows a mobility-driven, first-order-correlation-matched
  complex AR(1) approximation, with Jakes/J0 correlation used to map user
  speed into per-step correlation coefficients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np


_C_LIGHT = 299_792_458.0
_RIS_PATHLOSS_POWER = 2.0


def _bessel_j0(x) -> np.ndarray:
    """
    Numerical J0 approximation via the integral representation:
        J0(x) = (1 / pi) * integral_0^pi cos(x sin(theta)) dtheta
    This avoids adding a scipy dependency while remaining accurate enough for
    the mobility ranges used in this project.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    flat = np.abs(x_arr.reshape(-1))
    if flat.size == 0:
        return np.asarray(x_arr, dtype=np.float64)
    theta = np.linspace(0.0, math.pi, num=512, dtype=np.float64)
    integrand = np.cos(flat[:, None] * np.sin(theta)[None, :])
    vals = np.trapz(integrand, theta, axis=1) / math.pi
    return vals.reshape(x_arr.shape)


@dataclass
class MobilityDrivenChannelConfig:
    num_users: int
    num_ris_elements: int
    num_bs_antennas: int
    bs_position_xy: Tuple[float, float]
    ris_position_xy: Tuple[float, float]
    user_cluster_ratios: Sequence[float]
    user_cluster_centers_xy: Sequence[Sequence[float]]
    user_cluster_position_jitter_xy: Sequence[Sequence[float]]
    user_speed_range: Sequence[float]
    user_motion_direction_deg: Optional[float]
    user_speed_user_mask: Optional[Union[int, Sequence[int]]]
    direct_pathloss_exponent: float
    channel_ref_scale: float
    channel_time_step: float
    carrier_frequency_hz: float
    min_distance: float = 1.0


@dataclass
class UserChannelStepResult:
    h_ru_next: np.ndarray
    alpha_delta: np.ndarray
    h_bu_next: Optional[np.ndarray] = None
    h_ru_tau: Optional[np.ndarray] = None
    alpha_tau: Optional[np.ndarray] = None
    h_bu_tau: Optional[np.ndarray] = None


@dataclass
class SingleUserChannelStepResult:
    h_ru_tau: np.ndarray
    h_ru_next: np.ndarray
    alpha_delta: float
    alpha_tau: float
    h_bu_tau: Optional[np.ndarray] = None
    h_bu_next: Optional[np.ndarray] = None


def _allocate_counts_by_ratio(total_count: int, ratios: Sequence[float]) -> np.ndarray:
    ratio_arr = np.asarray(ratios, dtype=np.float64).reshape(-1)
    if ratio_arr.size == 0:
        raise ValueError("Cluster ratios must be non-empty.")
    if np.any(ratio_arr < 0):
        raise ValueError("Cluster ratios must be non-negative.")
    if float(np.sum(ratio_arr)) <= 0.0:
        raise ValueError("Cluster ratios must sum to a positive value.")

    ratio_arr = ratio_arr / np.sum(ratio_arr)
    raw = ratio_arr * int(total_count)
    base = np.floor(raw).astype(np.int64)
    remainder = int(total_count) - int(np.sum(base))
    if remainder > 0:
        frac = raw - base
        order = np.argsort(-frac)
        for idx in order[:remainder]:
            base[idx] += 1
    return base


class MobilityDrivenChannelEvolver:
    """
    Explicit user-mobility channel model.

    Large-scale fading is recomputed from positions every step, while
    small-scale fading uses a user-level first-order-correlation-matched
    complex AR(1) evolution with Jakes/J0 correlation mapped from user speed.
    """

    def __init__(self, cfg: MobilityDrivenChannelConfig):
        self.cfg = cfg
        self._validate()
        self.bs_position = np.asarray(cfg.bs_position_xy, dtype=np.float64).reshape(2)
        self.ris_position = np.asarray(cfg.ris_position_xy, dtype=np.float64).reshape(2)
        self.ref_scale = float(cfg.channel_ref_scale)
        self.time_step = float(cfg.channel_time_step)
        self.carrier_frequency_hz = float(cfg.carrier_frequency_hz)
        self.min_distance = float(cfg.min_distance)
        self.direct_pathloss_exponent = float(cfg.direct_pathloss_exponent)

        self.cluster_ids = self._sample_cluster_ids()
        self.initial_positions = self._sample_initial_positions()
        self.fixed_direction_deg = None if cfg.user_motion_direction_deg is None else float(cfg.user_motion_direction_deg)
        self.moving_user_mask = self._resolve_motion_user_mask()
        self.moving_user_ids = (np.flatnonzero(self.moving_user_mask).astype(np.int64) + 1).tolist()
        self.velocity_vectors = self._sample_velocity_vectors()
        self.speed_magnitudes = np.linalg.norm(self.velocity_vectors, axis=1)
        self.doppler_hz = self._speed_to_doppler(self.speed_magnitudes)
        self.alpha_delta = self._speed_to_alpha(self.speed_magnitudes)

    def _validate(self) -> None:
        if int(self.cfg.num_users) <= 0:
            raise ValueError("num_users must be positive")
        if int(self.cfg.num_ris_elements) <= 0:
            raise ValueError("num_ris_elements must be positive")
        if int(self.cfg.num_bs_antennas) <= 0:
            raise ValueError("num_bs_antennas must be positive")
        if float(self.cfg.channel_ref_scale) <= 0.0:
            raise ValueError("channel_ref_scale must be positive")
        if float(self.cfg.channel_time_step) <= 0.0:
            raise ValueError("channel_time_step must be positive")
        if float(self.cfg.carrier_frequency_hz) <= 0.0:
            raise ValueError("carrier_frequency_hz must be positive")
        if float(self.cfg.direct_pathloss_exponent) <= 0.0:
            raise ValueError("direct_pathloss_exponent must be positive")
        if float(self.cfg.min_distance) <= 0.0:
            raise ValueError("min_distance must be positive")

        cluster_count = len(self.cfg.user_cluster_ratios)
        if cluster_count <= 0:
            raise ValueError("At least one user cluster must be configured.")

        expected_lengths = {
            "user_cluster_centers_xy": len(self.cfg.user_cluster_centers_xy),
            "user_cluster_position_jitter_xy": len(self.cfg.user_cluster_position_jitter_xy),
        }
        for name, length in expected_lengths.items():
            if length != cluster_count:
                raise ValueError(f"{name} length must match user_cluster_ratios ({cluster_count}), got {length}")

        for seq_name in ("user_cluster_centers_xy", "user_cluster_position_jitter_xy"):
            seq = getattr(self.cfg, seq_name)
            for idx, item in enumerate(seq):
                arr = np.asarray(item, dtype=np.float64).reshape(-1)
                if arr.size != 2:
                    raise ValueError(f"{seq_name}[{idx}] must contain exactly 2 values, got {arr.size}")

        speed_range = np.asarray(self.cfg.user_speed_range, dtype=np.float64).reshape(-1)
        if speed_range.size != 2:
            raise ValueError(f"user_speed_range must contain exactly 2 values, got {speed_range.size}")
        if not np.all(np.isfinite(speed_range)):
            raise ValueError("user_speed_range must contain finite values")
        if float(speed_range[0]) > float(speed_range[1]):
            raise ValueError("user_speed_range must be ordered as [min_speed, max_speed]")

        if self.cfg.user_motion_direction_deg is not None:
            direction_deg = float(self.cfg.user_motion_direction_deg)
            if not np.isfinite(direction_deg):
                raise ValueError("user_motion_direction_deg must be finite when provided")

    def _resolve_motion_user_mask(self) -> np.ndarray:
        num_users = int(self.cfg.num_users)
        raw_mask = self.cfg.user_speed_user_mask
        if raw_mask is None:
            return np.ones((num_users,), dtype=bool)

        if np.isscalar(raw_mask):
            user_idx = int(raw_mask)
            if user_idx == 1:
                return np.ones((num_users,), dtype=bool)
            if not (1 <= user_idx <= num_users):
                raise ValueError(f"user_speed_user_mask scalar must be 1 or a 1-based user id in [1,{num_users}]")
            mask = np.zeros((num_users,), dtype=bool)
            mask[user_idx - 1] = True
            return mask

        raw_arr = np.asarray(list(raw_mask), dtype=np.int64).reshape(-1)
        if raw_arr.size == 0:
            return np.zeros((num_users,), dtype=bool)
        mask = np.zeros((num_users,), dtype=bool)
        for user_idx in raw_arr.tolist():
            if not (1 <= int(user_idx) <= num_users):
                raise ValueError(f"user_speed_user_mask ids must be 1-based and <= {num_users}, got {user_idx}")
            mask[int(user_idx) - 1] = True
        return mask

    @staticmethod
    def _validate_step_ratio(step_ratio: float) -> float:
        ratio = float(step_ratio)
        if not (0.0 <= ratio <= 1.0):
            raise ValueError(f"step_ratio must be in [0, 1], got {step_ratio}")
        return ratio

    @staticmethod
    def _clip_alpha(alpha: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(alpha, dtype=np.float64), -0.999999, 0.999999)

    def _sample_cluster_ids(self) -> np.ndarray:
        counts = _allocate_counts_by_ratio(int(self.cfg.num_users), self.cfg.user_cluster_ratios)
        cluster_ids = np.empty((int(self.cfg.num_users),), dtype=np.int64)
        start = 0
        for cluster_idx, count in enumerate(counts.tolist()):
            end = start + int(count)
            cluster_ids[start:end] = int(cluster_idx)
            start = end
        return cluster_ids

    def _sample_initial_positions(self) -> np.ndarray:
        positions = np.zeros((int(self.cfg.num_users), 2), dtype=np.float64)
        for user_idx, cluster_idx in enumerate(self.cluster_ids.tolist()):
            center = np.asarray(self.cfg.user_cluster_centers_xy[cluster_idx], dtype=np.float64).reshape(2)
            jitter = np.asarray(self.cfg.user_cluster_position_jitter_xy[cluster_idx], dtype=np.float64).reshape(2)
            offset = np.random.uniform(-jitter, jitter, size=(2,))
            positions[user_idx] = center + offset
        return positions

    def _sample_velocity_vectors(self) -> np.ndarray:
        velocities = np.zeros((int(self.cfg.num_users), 2), dtype=np.float64)
        speed_min, speed_max = np.asarray(self.cfg.user_speed_range, dtype=np.float64).reshape(2)
        for user_idx, _cluster_idx in enumerate(self.cluster_ids.tolist()):
            if not bool(self.moving_user_mask[user_idx]):
                continue
            signed_speed = np.random.uniform(speed_min, speed_max)
            if self.fixed_direction_deg is None:
                direction = np.random.uniform(0.0, 2.0 * math.pi)
            else:
                direction = math.radians(self.fixed_direction_deg)
            velocities[user_idx] = signed_speed * np.array([math.cos(direction), math.sin(direction)], dtype=np.float64)
        return velocities

    def _speed_to_doppler(self, speed_magnitudes: np.ndarray) -> np.ndarray:
        return np.asarray(speed_magnitudes, dtype=np.float64) * self.carrier_frequency_hz / _C_LIGHT

    def _speed_to_alpha_for_interval(self, speed_magnitudes: np.ndarray, interval_seconds: float) -> np.ndarray:
        doppler_hz = self._speed_to_doppler(speed_magnitudes)
        arg = 2.0 * math.pi * doppler_hz * float(interval_seconds)
        alpha = _bessel_j0(arg)
        return self._clip_alpha(alpha)

    def _speed_to_alpha(self, speed_magnitudes: np.ndarray) -> np.ndarray:
        return self._speed_to_alpha_for_interval(speed_magnitudes, self.time_step)

    def positions_at(self, step_value: float) -> np.ndarray:
        return self.initial_positions + (self.velocity_vectors * (self.time_step * float(step_value)))

    def positions_at_steps(self, step_values: Sequence[float]) -> np.ndarray:
        step_arr = np.asarray(step_values, dtype=np.float64).reshape(-1, 1)
        if step_arr.shape[0] != int(self.cfg.num_users):
            raise ValueError(
                f"Expected one step value per user ({int(self.cfg.num_users)}), got {step_arr.shape[0]}"
            )
        return self.initial_positions + (self.velocity_vectors * (self.time_step * step_arr))

    def ris_distances(self, positions_xy: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions_xy, dtype=np.float64)
        return np.linalg.norm(positions - self.ris_position[None, :], axis=1)

    def direct_distances(self, positions_xy: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions_xy, dtype=np.float64)
        return np.linalg.norm(positions - self.bs_position[None, :], axis=1)

    def ris_pathloss(self, positions_xy: np.ndarray) -> np.ndarray:
        distances = np.maximum(self.ris_distances(positions_xy), self.min_distance)
        return np.power(distances, -_RIS_PATHLOSS_POWER).astype(np.float64, copy=False)

    def direct_pathloss(self, positions_xy: np.ndarray) -> np.ndarray:
        distances = np.maximum(self.direct_distances(positions_xy), self.min_distance)
        return np.power(distances, -self.direct_pathloss_exponent).astype(np.float64, copy=False)

    def initialize_br_channel(self) -> np.ndarray:
        return (
            (np.random.randn(int(self.cfg.num_ris_elements), int(self.cfg.num_bs_antennas)) +
             1j * np.random.randn(int(self.cfg.num_ris_elements), int(self.cfg.num_bs_antennas)))
            / math.sqrt(2.0)
        ).astype(np.complex64)

    def initialize_user_channels(self, *, include_direct: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        pl_ris = self.ris_pathloss(self.initial_positions)
        g_ru0 = self._complex_noise((int(self.cfg.num_users), int(self.cfg.num_ris_elements)))
        h_ru0 = self._apply_pathloss(g_ru0, pl_ris)

        h_bu0 = None
        if include_direct:
            pl_direct = self.direct_pathloss(self.initial_positions)
            g_bu0 = self._complex_noise((int(self.cfg.num_users), int(self.cfg.num_bs_antennas)))
            h_bu0 = self._apply_pathloss(g_bu0, pl_direct)
        return h_ru0, h_bu0

    def current_alpha_vector(self) -> np.ndarray:
        return self.alpha_delta.astype(np.float32, copy=False)

    def current_doppler_vector(self) -> np.ndarray:
        return self.doppler_hz.astype(np.float32, copy=False)

    def _complex_noise(self, shape: Tuple[int, ...]) -> np.ndarray:
        return (
            (np.random.randn(*shape) + 1j * np.random.randn(*shape))
            / math.sqrt(2.0)
        ).astype(np.complex64)

    def _apply_pathloss(self, g_small: np.ndarray, pathloss: np.ndarray) -> np.ndarray:
        scale = np.sqrt(np.asarray(pathloss, dtype=np.float64)).astype(np.float32, copy=False)
        return (g_small * scale[:, None] / self.ref_scale).astype(np.complex64, copy=False)

    def _remove_pathloss(self, h_full: np.ndarray, pathloss: np.ndarray) -> np.ndarray:
        scale = np.sqrt(np.maximum(np.asarray(pathloss, dtype=np.float64), 1e-20)).astype(np.float32, copy=False)
        return (h_full * self.ref_scale / scale[:, None]).astype(np.complex64, copy=False)

    def _evolve_batch(
            self,
            current_h: np.ndarray,
            pathloss_current: np.ndarray,
            pathloss_target: np.ndarray,
            alpha_delta: np.ndarray,
    ) -> np.ndarray:
        g_current = self._remove_pathloss(current_h, pathloss_current)
        beta = np.sqrt(np.maximum(0.0, 1.0 - np.square(np.abs(alpha_delta)))).astype(np.float64)
        noise = self._complex_noise(current_h.shape)
        g_next = (alpha_delta[:, None] * g_current) + (beta[:, None] * noise)
        return self._apply_pathloss(g_next, pathloss_target)

    def _evolve_batch_split(
            self,
            current_h: np.ndarray,
            pathloss_current: np.ndarray,
            pathloss_tau: np.ndarray,
            pathloss_next: np.ndarray,
            alpha_tau: np.ndarray,
            alpha_delta: np.ndarray,
            alpha_rem: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        g_current = self._remove_pathloss(current_h, pathloss_current)
        noise_tau = self._complex_noise(current_h.shape)
        noise_next = self._complex_noise(current_h.shape)

        beta_tau = np.sqrt(np.maximum(0.0, 1.0 - np.square(np.abs(alpha_tau)))).astype(np.float64)
        g_tau = (alpha_tau[:, None] * g_current) + (beta_tau[:, None] * noise_tau)

        g_next = np.empty_like(g_current)
        near_static_tau = beta_tau < 1e-8
        if np.any(near_static_tau):
            beta_delta = np.sqrt(np.maximum(0.0, 1.0 - np.square(np.abs(alpha_delta[near_static_tau])))).astype(np.float64)
            g_next[near_static_tau] = (
                alpha_delta[near_static_tau, None] * g_current[near_static_tau]
                + beta_delta[:, None] * noise_next[near_static_tau]
            )
        if np.any(~near_static_tau):
            beta_tau_safe = beta_tau[~near_static_tau]
            coupling = (
                alpha_rem[~near_static_tau] - alpha_tau[~near_static_tau] * alpha_delta[~near_static_tau]
            ) / beta_tau_safe
            residual_var = np.maximum(
                0.0,
                1.0 - np.square(np.abs(alpha_delta[~near_static_tau])) - np.square(np.abs(coupling)),
            )
            g_next[~near_static_tau] = (
                alpha_delta[~near_static_tau, None] * g_current[~near_static_tau]
                + coupling[:, None] * noise_tau[~near_static_tau]
                + np.sqrt(residual_var)[:, None] * noise_next[~near_static_tau]
            )

        h_tau = self._apply_pathloss(g_tau, pathloss_tau)
        h_next = self._apply_pathloss(g_next, pathloss_next)
        return h_tau, h_next, alpha_tau

    def step(self, h_RUs: np.ndarray, round_idx: int, h_BUs: Optional[np.ndarray] = None) -> UserChannelStepResult:
        positions_current = self.positions_at(float(round_idx - 1))
        positions_next = self.positions_at(float(round_idx))
        alpha_delta = self.alpha_delta.copy()

        pl_ris_current = self.ris_pathloss(positions_current)
        pl_ris_next = self.ris_pathloss(positions_next)
        h_ru_next = self._evolve_batch(h_RUs, pl_ris_current, pl_ris_next, alpha_delta)

        h_bu_next = None
        if h_BUs is not None:
            pl_direct_current = self.direct_pathloss(positions_current)
            pl_direct_next = self.direct_pathloss(positions_next)
            h_bu_next = self._evolve_batch(h_BUs, pl_direct_current, pl_direct_next, alpha_delta)

        return UserChannelStepResult(
            h_ru_next=h_ru_next,
            alpha_delta=alpha_delta.astype(np.float32, copy=False),
            h_bu_next=h_bu_next,
        )

    def step_split(
            self,
            h_RUs: np.ndarray,
            round_idx: int,
            tau_ratio: float,
            h_BUs: Optional[np.ndarray] = None,
    ) -> UserChannelStepResult:
        rho = self._validate_step_ratio(tau_ratio)
        positions_current = self.positions_at(float(round_idx - 1))
        positions_tau = self.positions_at(float(round_idx - 1) + rho)
        positions_next = self.positions_at(float(round_idx))
        alpha_delta = self.alpha_delta.copy()
        alpha_tau = self._speed_to_alpha_for_interval(self.speed_magnitudes, rho * self.time_step)
        alpha_rem = self._speed_to_alpha_for_interval(self.speed_magnitudes, (1.0 - rho) * self.time_step)

        pl_ris_current = self.ris_pathloss(positions_current)
        pl_ris_tau = self.ris_pathloss(positions_tau)
        pl_ris_next = self.ris_pathloss(positions_next)
        h_ru_tau, h_ru_next, alpha_tau = self._evolve_batch_split(
            h_RUs,
            pl_ris_current,
            pl_ris_tau,
            pl_ris_next,
            alpha_tau,
            alpha_delta,
            alpha_rem,
        )

        h_bu_tau = None
        h_bu_next = None
        if h_BUs is not None:
            pl_direct_current = self.direct_pathloss(positions_current)
            pl_direct_tau = self.direct_pathloss(positions_tau)
            pl_direct_next = self.direct_pathloss(positions_next)
            h_bu_tau, h_bu_next, _ = self._evolve_batch_split(
                h_BUs,
                pl_direct_current,
                pl_direct_tau,
                pl_direct_next,
                alpha_tau,
                alpha_delta,
                alpha_rem,
            )

        return UserChannelStepResult(
            h_ru_next=h_ru_next,
            alpha_delta=alpha_delta.astype(np.float32, copy=False),
            h_bu_next=h_bu_next,
            h_ru_tau=h_ru_tau,
            alpha_tau=np.asarray(alpha_tau, dtype=np.float32),
            h_bu_tau=h_bu_tau,
        )

    def step_single_split(
            self,
            h_ru: np.ndarray,
            user_idx: int,
            round_idx: int,
            tau_ratio: float,
            h_bu: Optional[np.ndarray] = None,
    ) -> SingleUserChannelStepResult:
        rho = self._validate_step_ratio(tau_ratio)
        user = int(user_idx)
        alpha_delta = float(self.alpha_delta[user])
        alpha_tau = float(self._speed_to_alpha_for_interval(np.asarray([self.speed_magnitudes[user]], dtype=np.float64), rho * self.time_step)[0])
        alpha_rem = float(
            self._speed_to_alpha_for_interval(
                np.asarray([self.speed_magnitudes[user]], dtype=np.float64),
                (1.0 - rho) * self.time_step,
            )[0]
        )

        positions_current = self.positions_at(float(round_idx - 1))[user:user + 1]
        positions_tau = self.positions_at(float(round_idx - 1) + rho)[user:user + 1]
        positions_next = self.positions_at(float(round_idx))[user:user + 1]

        h_ru_tau, h_ru_next, _ = self._evolve_batch_split(
            h_ru.reshape(1, -1),
            self.ris_pathloss(positions_current),
            self.ris_pathloss(positions_tau),
            self.ris_pathloss(positions_next),
            np.asarray([alpha_tau], dtype=np.float64),
            np.asarray([alpha_delta], dtype=np.float64),
            np.asarray([alpha_rem], dtype=np.float64),
        )

        h_bu_tau = None
        h_bu_next = None
        if h_bu is not None:
            h_bu_tau, h_bu_next, _ = self._evolve_batch_split(
                h_bu.reshape(1, -1),
                self.direct_pathloss(positions_current),
                self.direct_pathloss(positions_tau),
                self.direct_pathloss(positions_next),
                np.asarray([alpha_tau], dtype=np.float64),
                np.asarray([alpha_delta], dtype=np.float64),
                np.asarray([alpha_rem], dtype=np.float64),
            )
            h_bu_tau = h_bu_tau[0]
            h_bu_next = h_bu_next[0]

        return SingleUserChannelStepResult(
            h_ru_tau=h_ru_tau[0],
            h_ru_next=h_ru_next[0],
            alpha_delta=alpha_delta,
            alpha_tau=alpha_tau,
            h_bu_tau=h_bu_tau,
            h_bu_next=h_bu_next,
        )


def build_ru_channel_evolver_from_config(config) -> MobilityDrivenChannelEvolver:
    cfg = MobilityDrivenChannelConfig(
        num_users=int(config.num_users),
        num_ris_elements=int(config.num_ris_elements),
        num_bs_antennas=int(config.num_bs_antennas),
        bs_position_xy=tuple(getattr(config, "bs_position_xy", (0.0, 0.0))),
        ris_position_xy=tuple(getattr(config, "ris_position_xy", (50.0, 0.0))),
        user_cluster_ratios=list(getattr(config, "user_cluster_ratios", [0.5, 0.5])),
        user_cluster_centers_xy=list(getattr(config, "user_cluster_centers_xy", [(40.0, 10.0), (90.0, -10.0)])),
        user_cluster_position_jitter_xy=list(
            getattr(config, "user_cluster_position_jitter_xy", [(5.0, 5.0), (10.0, 10.0)])
        ),
        user_speed_range=list(getattr(config, "user_speed_range", [0.5, 8.0])),
        user_motion_direction_deg=getattr(config, "user_motion_direction_deg", None),
        user_speed_user_mask=getattr(config, "user_speed_user_mask", 1),
        direct_pathloss_exponent=float(getattr(config, "alpha_direct", 3.0)),
        channel_ref_scale=float(getattr(config, "channel_ref_scale", math.sqrt(1e-10))),
        channel_time_step=float(getattr(config, "channel_time_step", 1e-3)),
        carrier_frequency_hz=float(getattr(config, "channel_carrier_frequency_hz", 3.5e9)),
        min_distance=float(getattr(config, "channel_min_distance", 1.0)),
    )
    return MobilityDrivenChannelEvolver(cfg)
