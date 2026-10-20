"""Validated observation loaders for measured array data.

This module only loads observations and metadata.  It never synthesizes data,
calls KRAKEN, or accepts seabed acoustic parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ExperimentalObservation:
    Y: np.ndarray
    frequency: float
    array_geometry: np.ndarray
    water_depth: float
    SSP: np.ndarray
    ssp_depths: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)
    reference_wavenumbers: np.ndarray | None = None

    def validated(self) -> "ExperimentalObservation":
        y = np.asarray(self.Y, dtype=np.complex128)
        geometry = np.asarray(self.array_geometry, dtype=np.float64).reshape(-1)
        ssp = np.asarray(self.SSP, dtype=np.float64).reshape(-1)
        ssp_depths = np.asarray(self.ssp_depths, dtype=np.float64).reshape(-1)
        if y.ndim != 2 or min(y.shape) < 1:
            raise ValueError("Y must have shape [num_receivers, num_snapshots]")
        if not np.all(np.isfinite(y.real)) or not np.all(np.isfinite(y.imag)):
            raise ValueError("Y contains NaN or infinite values")
        if geometry.size != y.shape[0]:
            raise ValueError("array_geometry length must equal Y receiver dimension")
        if not np.all(np.isfinite(geometry)):
            raise ValueError("array_geometry contains non-finite values")
        if np.any(geometry <= 0) or np.any(geometry >= float(self.water_depth)):
            raise ValueError("receiver depths must lie strictly within the water column")
        if np.any(np.diff(geometry) <= 0):
            raise ValueError("receiver depths must be strictly increasing; channel reordering is forbidden")
        if float(self.frequency) <= 0 or not np.isfinite(self.frequency):
            raise ValueError("frequency must be positive and finite")
        if float(self.water_depth) <= 0 or not np.isfinite(self.water_depth):
            raise ValueError("water_depth must be positive and finite")
        if ssp.size < 2 or ssp.size != ssp_depths.size:
            raise ValueError("SSP values and depths must have equal length >= 2")
        if not np.all(np.isfinite(ssp)) or np.any(ssp <= 0):
            raise ValueError("SSP values must be positive and finite")
        if not np.all(np.isfinite(ssp_depths)) or np.any(np.diff(ssp_depths) <= 0):
            raise ValueError("SSP depths must be finite and strictly increasing")
        if ssp_depths[0] > 1e-9 or ssp_depths[-1] < float(self.water_depth) - 1e-9:
            raise ValueError("SSP must cover the full water column from surface to bottom")
        if ssp_depths[0] < 0 or ssp_depths[-1] > float(self.water_depth) + 1e-9:
            raise ValueError("SSP depths must lie within the water column")
        reference = None
        if self.reference_wavenumbers is not None:
            reference = np.asarray(self.reference_wavenumbers, dtype=np.float64).reshape(-1)
            if reference.size < 1 or not np.all(np.isfinite(reference)):
                raise ValueError("reference_wavenumbers must be finite when supplied")
        return ExperimentalObservation(
            Y=y,
            frequency=float(self.frequency),
            array_geometry=geometry,
            water_depth=float(self.water_depth),
            SSP=ssp,
            ssp_depths=ssp_depths,
            metadata=dict(self.metadata),
            reference_wavenumbers=reference,
        )


def _required(metadata: Mapping[str, Any], key: str) -> Any:
    if key not in metadata or metadata[key] is None:
        raise ValueError(f"missing required metadata: {key}")
    return metadata[key]


def _metadata_from_sidecar(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        raise ValueError(f"missing required metadata sidecar: {sidecar.name}")
    return json.loads(sidecar.read_text(encoding="utf-8"))


def _complex_csv(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError("CSV observation file is empty")
    try:
        return np.asarray([[complex(value.strip().replace("i", "j")) for value in row] for row in rows])
    except ValueError as exc:
        raise ValueError("CSV must contain a rectangular matrix of real or complex values") from exc


def load_experimental_observation(
    path: str | Path,
    metadata_override: Mapping[str, Any] | None = None,
) -> ExperimentalObservation:
    """Load .npz/.npy/.csv/.mat observations with strict metadata validation."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower()
    override = dict(metadata_override or {})
    payload: dict[str, Any] = {}

    if suffix == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            payload = {key: archive[key] for key in archive.files}
    elif suffix == ".npy":
        payload = {"Y": np.load(source, allow_pickle=False), **_metadata_from_sidecar(source)}
    elif suffix == ".csv":
        payload = {"Y": _complex_csv(source), **_metadata_from_sidecar(source)}
    elif suffix == ".mat":
        try:
            from scipy.io import loadmat
        except ImportError as exc:
            raise ImportError("scipy is required to load MATLAB .mat observations") from exc
        payload = {key: value for key, value in loadmat(source).items() if not key.startswith("__")}
    else:
        raise ValueError(f"unsupported observation format: {suffix}")

    payload.update(override)
    y = _required(payload, "Y")
    frequency = payload.get("frequency", payload.get("frequency_hz"))
    geometry = payload.get("array_geometry", payload.get("array_depths_m"))
    water_depth = payload.get("water_depth", payload.get("water_depth_m"))
    ssp = payload.get("SSP", payload.get("ssp_values_m_s"))
    ssp_depths = payload.get("ssp_depths", payload.get("ssp_depths_m"))
    reference = payload.get("reference_wavenumbers")
    observation = ExperimentalObservation(
        Y=y,
        frequency=float(_required({"frequency": frequency}, "frequency")),
        array_geometry=_required({"array_geometry": geometry}, "array_geometry"),
        water_depth=float(_required({"water_depth": water_depth}, "water_depth")),
        SSP=_required({"SSP": ssp}, "SSP"),
        ssp_depths=_required({"ssp_depths": ssp_depths}, "ssp_depths"),
        metadata={"source_path": str(source), **dict(payload.get("metadata", {}))},
        reference_wavenumbers=reference,
    )
    return observation.validated()
