"""CMTR-v1: causal modal Track/Reacquire estimator.

Development-only implementation for manual concept validation.

TRACK:
    past accepted modal trajectory + current SSP
    -> causal prediction
    -> SSP-aware local search window
    -> present-frame projection-residual refinement

REACQUIRE:
    if TRACK violates data-only residual / innovation / ordering checks,
    call the existing full instantaneous estimator through a callback.

No true k, KRAKEN roots, seabed parameters, or future samples are used.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import time
from typing import Callable

import numpy as np

from mode_fri.mode_solver import solve_mode


@dataclass(frozen=True)
class CMTRConfig:
    beta_pred: float = 0.50
    delta0_rad_m: float = 1.20e-3
    delta_min_rad_m: float = 8.00e-4
    delta_max_rad_m: float = 4.00e-3
    lambda_k: float = 1.50
    lambda_ssp: float = 0.15
    minimum_mode_separation_rad_m: float = 3.50e-3

    local_grid_points: int = 9
    local_coordinate_sweeps: int = 2
    enable_quadratic_subgrid: bool = True

    residual_ema_alpha: float = 0.25
    residual_mean_multiplier: float = 1.60
    residual_worst_multiplier: float = 1.80
    residual_mean_margin: float = 0.01
    residual_worst_margin: float = 0.02
    innovation_limit_rad_m: float = 3.00e-3

    search_lower_fraction: float = 0.95
    search_upper_fraction: float = 0.999999


@dataclass
class CMTRState:
    previous_k: np.ndarray | None = None
    previous_previous_k: np.ndarray | None = None
    previous_ssp: np.ndarray | None = None
    residual_mean_ema: float | None = None
    residual_worst_ema: float | None = None
    last_state: str = "UNINITIALIZED"
    frame_index: int = -1


@dataclass(frozen=True)
class CMTRStepResult:
    k_hat: np.ndarray
    state: str
    predicted_k: np.ndarray
    search_half_widths: np.ndarray
    ssp_change: float
    innovation_rms_rad_m: float
    residual_mean: float
    residual_worst: float
    residual_mean_threshold: float
    residual_worst_threshold: float
    local_runtime_seconds: float
    reacquire_runtime_seconds: float
    total_runtime_seconds: float
    local_candidate: np.ndarray
    local_accepted: bool


def _canonicalize_column(column: np.ndarray) -> np.ndarray:
    x = np.asarray(column, dtype=np.complex128).reshape(-1)
    nrm = float(np.linalg.norm(x))
    if not np.isfinite(nrm) or nrm <= 1e-14:
        raise FloatingPointError("Invalid mode atom norm.")
    x = x / nrm
    pidx = int(np.argmax(np.abs(x)))
    pivot = x[pidx]
    if abs(pivot) > 0:
        x = x * np.exp(-1j * np.angle(pivot))
    if np.real(x[pidx]) < 0:
        x = -x
    return x


def _receiver_indices(depths: np.ndarray, dz: float, water_depth: float) -> np.ndarray:
    idx = np.rint(np.asarray(depths, dtype=float) / float(dz)).astype(int)
    max_idx = int(round(float(water_depth) / float(dz)))
    if np.any(idx < 0) or np.any(idx > max_idx):
        raise ValueError("Receiver depth outside water column.")
    return idx


def water_column_k_interval(
    frequency_hz: float, ssp: np.ndarray, config: CMTRConfig
) -> tuple[float, float]:
    c = np.asarray(ssp, dtype=float).reshape(-1)
    omega = 2.0 * math.pi * float(frequency_hz)
    k_min = config.search_lower_fraction * omega / float(np.max(c))
    k_max = config.search_upper_fraction * omega / float(np.min(c))
    if not (0 < k_min < k_max):
        raise ValueError("Invalid k interval.")
    return float(k_min), float(k_max)


def project_descending(
    values: np.ndarray,
    *,
    k_min: float,
    k_max: float,
    min_sep: float,
) -> np.ndarray:
    x = np.sort(np.asarray(values, dtype=float).reshape(-1))[::-1].copy()
    x = np.clip(x, k_min, k_max)
    for _ in range(max(1, 8 * x.size)):
        changed = False
        for i in range(x.size - 1):
            if x[i] - x[i + 1] < min_sep:
                center = 0.5 * (x[i] + x[i + 1])
                x[i] = center + 0.5 * min_sep
                x[i + 1] = center - 0.5 * min_sep
                changed = True
        x = np.clip(x, k_min, k_max)
        if not changed:
            break
    return np.sort(x)[::-1].copy()


def normalized_projection_scores(Y: np.ndarray, basis: np.ndarray) -> tuple[float, float]:
    y = np.asarray(Y, dtype=np.complex128)
    phi = np.asarray(basis, dtype=np.complex128)
    coeff, _, _, _ = np.linalg.lstsq(phi, y, rcond=None)
    residual = y - phi @ coeff
    sig = np.sum(np.abs(y) ** 2, axis=0)
    err = np.sum(np.abs(residual) ** 2, axis=0)
    q = err / np.maximum(sig, 1e-30)
    return float(np.mean(q)), float(np.max(q))


class LocalModeAtomFactory:
    def __init__(
        self,
        *,
        frequency_hz: float,
        ssp: np.ndarray,
        dz: float,
        receiver_depths: np.ndarray,
        water_depth: float,
    ):
        self.omega = 2.0 * math.pi * float(frequency_hz)
        self.ssp = np.asarray(ssp, dtype=float).copy()
        self.dz = float(dz)
        self.idx = _receiver_indices(receiver_depths, dz, water_depth)
        self.cache: dict[float, np.ndarray] = {}

    def atom(self, k: float) -> np.ndarray:
        key = round(float(k), 11)
        if key not in self.cache:
            full = solve_mode(
                horizontal_wavenumber=float(k),
                omega=self.omega,
                sound_speed_profile=self.ssp,
                dz=self.dz,
            )
            self.cache[key] = _canonicalize_column(
                np.asarray(full[self.idx], dtype=np.complex128)
            )
        return self.cache[key]

    def basis(self, k_values: np.ndarray) -> np.ndarray:
        return np.column_stack([self.atom(k) for k in k_values])


def causal_prediction(
    previous_k: np.ndarray,
    previous_previous_k: np.ndarray | None,
    *,
    k_min: float,
    k_max: float,
    config: CMTRConfig,
) -> np.ndarray:
    prev = np.asarray(previous_k, dtype=float).reshape(-1)
    if previous_previous_k is None:
        pred = prev.copy()
    else:
        older = np.asarray(previous_previous_k, dtype=float).reshape(-1)
        pred = prev + config.beta_pred * (prev - older)
    return project_descending(
        pred,
        k_min=k_min,
        k_max=k_max,
        min_sep=config.minimum_mode_separation_rad_m,
    )


def ssp_relative_change(current_ssp: np.ndarray, previous_ssp: np.ndarray | None) -> float:
    if previous_ssp is None:
        return 0.0
    cur = np.asarray(current_ssp, dtype=float)
    prv = np.asarray(previous_ssp, dtype=float)
    return float(np.linalg.norm(cur - prv) / max(np.linalg.norm(prv), 1e-15))


def adaptive_search_half_widths(
    previous_k: np.ndarray,
    previous_previous_k: np.ndarray | None,
    ssp_change: float,
    config: CMTRConfig,
) -> np.ndarray:
    prev = np.asarray(previous_k, dtype=float).reshape(-1)
    if previous_previous_k is None:
        dk = np.zeros_like(prev)
    else:
        dk = np.abs(prev - np.asarray(previous_previous_k, dtype=float).reshape(-1))
    widths = (
        config.delta0_rad_m
        + config.lambda_k * dk
        + config.lambda_ssp * float(ssp_change)
    )
    return np.clip(widths, config.delta_min_rad_m, config.delta_max_rad_m)


def _mode_bounds(
    current_k: np.ndarray,
    predicted_k: np.ndarray,
    widths: np.ndarray,
    mode_index: int,
    *,
    k_min: float,
    k_max: float,
    min_sep: float,
) -> tuple[float, float]:
    m = int(mode_index)
    lo = max(k_min, float(predicted_k[m] - widths[m]))
    hi = min(k_max, float(predicted_k[m] + widths[m]))
    if m > 0:
        hi = min(hi, float(current_k[m - 1] - min_sep))
    if m < current_k.size - 1:
        lo = max(lo, float(current_k[m + 1] + min_sep))
    if lo > hi:
        center = float(np.clip(predicted_k[m], k_min, k_max))
        return center, center
    return float(lo), float(hi)


def _quadratic_vertex(xs: np.ndarray, ys: np.ndarray) -> float | None:
    try:
        a, b, _ = np.polyfit(xs, ys, 2)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(a) or abs(a) < 1e-18:
        return None
    value = -b / (2.0 * a)
    return float(value) if np.isfinite(value) else None


def local_coordinate_refinement(
    Y: np.ndarray,
    predicted_k: np.ndarray,
    half_widths: np.ndarray,
    *,
    frequency_hz: float,
    ssp: np.ndarray,
    dz: float,
    receiver_depths: np.ndarray,
    water_depth: float,
    config: CMTRConfig,
) -> tuple[np.ndarray, tuple[float, float]]:
    k_min, k_max = water_column_k_interval(frequency_hz, ssp, config)
    current = project_descending(
        predicted_k,
        k_min=k_min,
        k_max=k_max,
        min_sep=config.minimum_mode_separation_rad_m,
    )
    factory = LocalModeAtomFactory(
        frequency_hz=frequency_hz,
        ssp=ssp,
        dz=dz,
        receiver_depths=receiver_depths,
        water_depth=water_depth,
    )

    ngrid = max(3, int(config.local_grid_points))
    if ngrid % 2 == 0:
        ngrid += 1

    def score(k_values: np.ndarray) -> tuple[float, float]:
        return normalized_projection_scores(Y, factory.basis(k_values))

    for _ in range(max(1, int(config.local_coordinate_sweeps))):
        for m in range(current.size):
            lo, hi = _mode_bounds(
                current,
                predicted_k,
                half_widths,
                m,
                k_min=k_min,
                k_max=k_max,
                min_sep=config.minimum_mode_separation_rad_m,
            )
            if math.isclose(lo, hi, abs_tol=1e-14):
                current[m] = lo
                continue

            grid = np.linspace(lo, hi, ngrid)
            means = []
            for value in grid:
                proposal = current.copy()
                proposal[m] = value
                means.append(score(proposal)[0])

            best = int(np.argmin(means))
            best_value = float(grid[best])
            best_mean = float(means[best])

            if config.enable_quadratic_subgrid and 0 < best < len(grid) - 1:
                vertex = _quadratic_vertex(
                    grid[best - 1:best + 2],
                    np.asarray(means[best - 1:best + 2]),
                )
                if vertex is not None and lo <= vertex <= hi:
                    proposal = current.copy()
                    proposal[m] = vertex
                    vmean, _ = score(proposal)
                    if vmean < best_mean:
                        best_value = float(vertex)

            current[m] = best_value

        current = project_descending(
            current,
            k_min=k_min,
            k_max=k_max,
            min_sep=config.minimum_mode_separation_rad_m,
        )

    return current.copy(), score(current)


class CausalModalTrackReacquire:
    def __init__(self, config: CMTRConfig | None = None):
        self.config = config or CMTRConfig()
        self.state = CMTRState()

    def initialize_from_full_estimate(
        self,
        *,
        k_hat: np.ndarray,
        SSP_t: np.ndarray,
        Y_t: np.ndarray,
        frequency_hz: float,
        dz: float,
        receiver_depths: np.ndarray,
        water_depth: float,
        full_runtime_seconds: float,
    ) -> CMTRStepResult:
        cfg = self.config
        k_min, k_max = water_column_k_interval(frequency_hz, SSP_t, cfg)
        k0 = project_descending(
            k_hat, k_min=k_min, k_max=k_max,
            min_sep=cfg.minimum_mode_separation_rad_m,
        )
        factory = LocalModeAtomFactory(
            frequency_hz=frequency_hz, ssp=SSP_t, dz=dz,
            receiver_depths=receiver_depths, water_depth=water_depth,
        )
        mean_res, worst_res = normalized_projection_scores(Y_t, factory.basis(k0))

        self.state.previous_k = k0.copy()
        self.state.previous_ssp = np.asarray(SSP_t, dtype=float).copy()
        self.state.residual_mean_ema = mean_res
        self.state.residual_worst_ema = worst_res
        self.state.last_state = "INIT_REACQUIRE"
        self.state.frame_index = 0

        return CMTRStepResult(
            k_hat=k0.copy(), state="INIT_REACQUIRE",
            predicted_k=k0.copy(), search_half_widths=np.zeros_like(k0),
            ssp_change=0.0, innovation_rms_rad_m=0.0,
            residual_mean=mean_res, residual_worst=worst_res,
            residual_mean_threshold=mean_res, residual_worst_threshold=worst_res,
            local_runtime_seconds=0.0,
            reacquire_runtime_seconds=float(full_runtime_seconds),
            total_runtime_seconds=float(full_runtime_seconds),
            local_candidate=k0.copy(), local_accepted=True,
        )

    def step(
        self,
        *,
        Y_t: np.ndarray,
        SSP_t: np.ndarray,
        frequency_hz: float,
        dz: float,
        receiver_depths: np.ndarray,
        water_depth: float,
        reacquire_fn: Callable[[], tuple[np.ndarray, float]],
    ) -> CMTRStepResult:
        if self.state.previous_k is None:
            raise RuntimeError("Initialize CMTR first.")

        cfg = self.config
        current_ssp = np.asarray(SSP_t, dtype=float).copy()
        k_min, k_max = water_column_k_interval(frequency_hz, current_ssp, cfg)
        ssp_change = ssp_relative_change(current_ssp, self.state.previous_ssp)

        pred = causal_prediction(
            self.state.previous_k,
            self.state.previous_previous_k,
            k_min=k_min, k_max=k_max, config=cfg,
        )
        widths = adaptive_search_half_widths(
            self.state.previous_k,
            self.state.previous_previous_k,
            ssp_change,
            cfg,
        )

        t0 = time.perf_counter()
        local_candidate, (local_mean, local_worst) = local_coordinate_refinement(
            Y_t, pred, widths,
            frequency_hz=frequency_hz,
            ssp=current_ssp,
            dz=dz,
            receiver_depths=receiver_depths,
            water_depth=water_depth,
            config=cfg,
        )
        local_runtime = time.perf_counter() - t0

        innovation = float(np.sqrt(np.mean((local_candidate - pred) ** 2)))
        mean_ema = float(self.state.residual_mean_ema)
        worst_ema = float(self.state.residual_worst_ema)
        mean_threshold = mean_ema * cfg.residual_mean_multiplier + cfg.residual_mean_margin
        worst_threshold = worst_ema * cfg.residual_worst_multiplier + cfg.residual_worst_margin

        separation_ok = bool(
            np.all(
                local_candidate[:-1] - local_candidate[1:]
                >= cfg.minimum_mode_separation_rad_m - 1e-12
            )
        )

        accept_track = (
            local_mean <= mean_threshold
            and local_worst <= worst_threshold
            and innovation <= cfg.innovation_limit_rad_m
            and separation_ok
        )

        reacquire_runtime = 0.0
        if accept_track:
            final_k = local_candidate.copy()
            final_mean, final_worst = local_mean, local_worst
            state_name = "TRACK"
        else:
            reacquired_k, reacquire_runtime = reacquire_fn()
            final_k = project_descending(
                reacquired_k, k_min=k_min, k_max=k_max,
                min_sep=cfg.minimum_mode_separation_rad_m,
            )
            factory = LocalModeAtomFactory(
                frequency_hz=frequency_hz, ssp=current_ssp, dz=dz,
                receiver_depths=receiver_depths, water_depth=water_depth,
            )
            final_mean, final_worst = normalized_projection_scores(
                Y_t, factory.basis(final_k)
            )
            state_name = "REACQUIRE"

        a = cfg.residual_ema_alpha
        self.state.residual_mean_ema = (1 - a) * mean_ema + a * final_mean
        self.state.residual_worst_ema = (1 - a) * worst_ema + a * final_worst
        self.state.previous_previous_k = self.state.previous_k.copy()
        self.state.previous_k = final_k.copy()
        self.state.previous_ssp = current_ssp.copy()
        self.state.last_state = state_name
        self.state.frame_index += 1

        return CMTRStepResult(
            k_hat=final_k.copy(),
            state=state_name,
            predicted_k=pred.copy(),
            search_half_widths=widths.copy(),
            ssp_change=ssp_change,
            innovation_rms_rad_m=innovation,
            residual_mean=float(final_mean),
            residual_worst=float(final_worst),
            residual_mean_threshold=float(mean_threshold),
            residual_worst_threshold=float(worst_threshold),
            local_runtime_seconds=float(local_runtime),
            reacquire_runtime_seconds=float(reacquire_runtime),
            total_runtime_seconds=float(local_runtime + reacquire_runtime),
            local_candidate=local_candidate.copy(),
            local_accepted=bool(accept_track),
        )
