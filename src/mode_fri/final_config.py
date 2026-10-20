"""Single source of truth for the frozen final experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "final_config.json"


@dataclass(frozen=True)
class RunSpec:
    mode: str
    nominal_k: tuple[int, ...]
    nominal_snr_db: tuple[float, ...]
    nominal_trials: int
    baseline_snr_db: tuple[float, ...]
    baseline_trials: int
    ablation_snr_db: tuple[float, ...]
    ablation_trials: int
    environment_snr_db: float
    environment_trials: int
    environment_ssp_offsets_mps: tuple[float, ...]
    environment_bottom_cp_offsets_percent: tuple[float, ...]
    environment_bottom_density_offsets_percent: tuple[float, ...]


@dataclass(frozen=True)
class FinalConfig:
    raw: dict[str, Any]
    source_path: Path

    @property
    def base_seed(self) -> int:
        return int(self.raw["base_seed"])

    @property
    def atoms(self) -> int:
        return int(self.raw["M"])

    @property
    def snapshots(self) -> int:
        return int(self.raw["L"])

    @property
    def dz_m(self) -> float:
        return float(self.raw["dz_m"])

    @property
    def lambda_bottom(self) -> float:
        return float(self.raw["lambda_bottom"])

    @property
    def selector_gamma(self) -> float:
        return float(self.raw["selector_physics_gamma"])

    @property
    def success_tolerance(self) -> float:
        return float(self.raw["success_tolerance"])

    def run_spec(self, mode: str, extended_snr: bool = False) -> RunSpec:
        mode = mode.lower()
        if mode not in {"quick", "full"}:
            raise ValueError("mode must be 'quick' or 'full'")
        block = self.raw[mode]
        snr_values = block["nominal_snr_db"]
        if mode == "full" and extended_snr:
            snr_values = block["extended_nominal_snr_db"]
        environment = self.raw["environment"]
        return RunSpec(
            mode=mode,
            nominal_k=tuple(int(v) for v in block["nominal_K"]),
            nominal_snr_db=tuple(float(v) for v in snr_values),
            nominal_trials=int(block["nominal_trials"]),
            baseline_snr_db=tuple(float(v) for v in block["baseline_snr_db"]),
            baseline_trials=int(block["baseline_trials"]),
            ablation_snr_db=tuple(float(v) for v in block["ablation_snr_db"]),
            ablation_trials=int(block["ablation_trials"]),
            environment_snr_db=float(block["environment_snr_db"]),
            environment_trials=int(block["environment_trials"]),
            environment_ssp_offsets_mps=tuple(
                float(v) for v in block.get(
                    "environment_ssp_offsets_mps", environment["ssp_offsets_mps"]
                )
            ),
            environment_bottom_cp_offsets_percent=tuple(
                float(v) for v in block.get(
                    "environment_bottom_cp_offsets_percent",
                    environment["bottom_cp_offsets_percent"],
                )
            ),
            environment_bottom_density_offsets_percent=tuple(
                float(v) for v in block.get(
                    "environment_bottom_density_offsets_percent",
                    environment["bottom_density_offsets_percent"],
                )
            ),
        )


def load_final_config(path: Path | str = DEFAULT_CONFIG_PATH) -> FinalConfig:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    required = {
        "base_seed", "M", "L", "dz_m", "lambda_bottom",
        "selector_physics_gamma", "safe_root_refinement", "quick", "full",
        "baseline", "ablation", "environment",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise ValueError(f"Missing final configuration keys: {missing}")
    if float(raw["lambda_bottom"]) != 0.1:
        raise ValueError("Final lambda_bottom is frozen at 0.1")
    if float(raw["selector_physics_gamma"]) != 0.0:
        raise ValueError("Final selector physics gamma is frozen at 0")
    if float(raw["dz_m"]) != 0.5:
        raise ValueError("Final Numerov dz is frozen at 0.5 m")
    if bool(raw["safe_root_refinement"]):
        raise ValueError("Safe-root refinement must remain disabled")
    return FinalConfig(raw=raw, source_path=source)
