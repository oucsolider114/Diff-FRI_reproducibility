# -*- coding: utf-8 -*-
"""
CSMT-v2: Causal SSP-aware Modal Tracking.

Final-method candidate derived after CMTR-v1 ablation.

Initialization
--------------
Frame 0 uses one Full Proposed estimate supplied by the caller.

Online update for t >= 1
------------------------
past accepted modal trajectory + current SSP
    -> causal trajectory prediction
    -> SSP-aware adaptive local search window
    -> present-frame continuous local refinement
    -> accept local result directly

There is NO residual-triggered Full-Proposed REACQUIRE path in CSMT-v2.

Scientific restrictions
-----------------------
- causal only: no future Y / SSP / k;
- no true k or KRAKEN roots;
- no bottom sound speed / density / attenuation;
- tracking parameters are inherited unchanged from frozen CMTR-v1:
  beta_pred, delta0/min/max, lambda_k, lambda_ssp, minimum separation,
  local-grid points, coordinate sweeps and quadratic subgrid option.

The removal of REACQUIRE is a structural v2 change motivated by the completed
CMTR-v1 ablation. It is not parameter retuning.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from mode_fri.online_track_reacquire import (
    CMTRConfig,
    LocalModeAtomFactory,
    adaptive_search_half_widths,
    causal_prediction,
    local_coordinate_refinement,
    normalized_projection_scores,
    project_descending,
    ssp_relative_change,
    water_column_k_interval,
)


@dataclass(frozen=True)
class CSMTConfig:
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

    search_lower_fraction: float = 0.95
    search_upper_fraction: float = 0.999999

    def as_cmtr_local_config(self) -> CMTRConfig:
        """
        Reuse the already-tested local-search helper implementation.
        Residual-gate fields are inert because CSMT never calls a gate.
        """
        return CMTRConfig(
            beta_pred=float(self.beta_pred),
            delta0_rad_m=float(self.delta0_rad_m),
            delta_min_rad_m=float(self.delta_min_rad_m),
            delta_max_rad_m=float(self.delta_max_rad_m),
            lambda_k=float(self.lambda_k),
            lambda_ssp=float(self.lambda_ssp),
            minimum_mode_separation_rad_m=float(
                self.minimum_mode_separation_rad_m
            ),
            local_grid_points=int(self.local_grid_points),
            local_coordinate_sweeps=int(self.local_coordinate_sweeps),
            enable_quadratic_subgrid=bool(self.enable_quadratic_subgrid),
            # Inert diagnostic/gate values; CSMT does not gate.
            residual_ema_alpha=0.25,
            residual_mean_multiplier=1.0,
            residual_worst_multiplier=1.0,
            residual_mean_margin=0.0,
            residual_worst_margin=0.0,
            innovation_limit_rad_m=1.0,
            search_lower_fraction=float(self.search_lower_fraction),
            search_upper_fraction=float(self.search_upper_fraction),
        )


@dataclass
class CSMTState:
    previous_k: np.ndarray | None = None
    previous_previous_k: np.ndarray | None = None
    previous_ssp: np.ndarray | None = None
    frame_index: int = -1


@dataclass(frozen=True)
class CSMTStepResult:
    k_hat: np.ndarray
    state: str
    predicted_k: np.ndarray
    search_half_widths: np.ndarray
    ssp_change: float
    innovation_rms_rad_m: float
    residual_mean: float
    residual_worst: float
    local_runtime_seconds: float
    total_runtime_seconds: float


class CausalSSPModalTracking:
    """
    Causal local modal tracker with SSP-aware adaptive search.
    """

    def __init__(self, config: CSMTConfig | None = None):
        self.config = config or CSMTConfig()
        self.state = CSMTState()

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
    ) -> CSMTStepResult:
        cfg = self.config.as_cmtr_local_config()
        k_min, k_max = water_column_k_interval(
            frequency_hz,
            SSP_t,
            cfg,
        )
        k0 = project_descending(
            k_hat,
            k_min=k_min,
            k_max=k_max,
            min_sep=cfg.minimum_mode_separation_rad_m,
        )

        factory = LocalModeAtomFactory(
            frequency_hz=frequency_hz,
            ssp=SSP_t,
            dz=dz,
            receiver_depths=receiver_depths,
            water_depth=water_depth,
        )
        mean_res, worst_res = normalized_projection_scores(
            Y_t,
            factory.basis(k0),
        )

        self.state.previous_k = k0.copy()
        self.state.previous_previous_k = None
        self.state.previous_ssp = np.asarray(
            SSP_t,
            dtype=np.float64,
        ).copy()
        self.state.frame_index = 0

        return CSMTStepResult(
            k_hat=k0.copy(),
            state="INIT_FULL",
            predicted_k=k0.copy(),
            search_half_widths=np.zeros_like(k0),
            ssp_change=0.0,
            innovation_rms_rad_m=0.0,
            residual_mean=float(mean_res),
            residual_worst=float(worst_res),
            local_runtime_seconds=0.0,
            total_runtime_seconds=float(full_runtime_seconds),
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
    ) -> CSMTStepResult:
        if self.state.previous_k is None:
            raise RuntimeError("Initialize CSMT-v2 before step().")

        local_cfg = self.config.as_cmtr_local_config()
        current_ssp = np.asarray(
            SSP_t,
            dtype=np.float64,
        ).copy()

        k_min, k_max = water_column_k_interval(
            frequency_hz,
            current_ssp,
            local_cfg,
        )
        ssp_change = ssp_relative_change(
            current_ssp,
            self.state.previous_ssp,
        )

        predicted = causal_prediction(
            self.state.previous_k,
            self.state.previous_previous_k,
            k_min=k_min,
            k_max=k_max,
            config=local_cfg,
        )
        widths = adaptive_search_half_widths(
            self.state.previous_k,
            self.state.previous_previous_k,
            ssp_change,
            local_cfg,
        )

        t0 = time.perf_counter()
        local_k, (mean_res, worst_res) = local_coordinate_refinement(
            Y_t,
            predicted,
            widths,
            frequency_hz=frequency_hz,
            ssp=current_ssp,
            dz=dz,
            receiver_depths=receiver_depths,
            water_depth=water_depth,
            config=local_cfg,
        )
        local_runtime = float(time.perf_counter() - t0)

        innovation = float(
            np.sqrt(np.mean((local_k - predicted) ** 2))
        )

        self.state.previous_previous_k = (
            self.state.previous_k.copy()
        )
        self.state.previous_k = local_k.copy()
        self.state.previous_ssp = current_ssp.copy()
        self.state.frame_index += 1

        return CSMTStepResult(
            k_hat=local_k.copy(),
            state="TRACK",
            predicted_k=predicted.copy(),
            search_half_widths=widths.copy(),
            ssp_change=float(ssp_change),
            innovation_rms_rad_m=innovation,
            residual_mean=float(mean_res),
            residual_worst=float(worst_res),
            local_runtime_seconds=local_runtime,
            total_runtime_seconds=local_runtime,
        )
