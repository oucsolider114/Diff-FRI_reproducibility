"""Output and metadata helpers for reproducible final experiments."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Iterable, Mapping, Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_OUTPUT = PROJECT_ROOT / "outputs" / "final"
CSV_DIR = FINAL_OUTPUT / "csv"
FIGURE_DIR = FINAL_OUTPUT / "figures"
TABLE_DIR = FINAL_OUTPUT / "tables"
LOG_DIR = FINAL_OUTPUT / "logs"


def ensure_output_tree() -> None:
    for path in (CSV_DIR, FIGURE_DIR, TABLE_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def save_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def array_text(values: Any) -> str:
    array = np.asarray(values).reshape(-1)
    return ";".join(f"{float(value):.12f}" for value in array)


def config_fingerprint(raw: Mapping[str, Any]) -> str:
    # Publication-only derived comparisons do not alter the frozen estimator
    # or primary Monte Carlo scope.  This exclusion is the historical rule
    # used by the existing 910 FULL rows and keeps their identifier stable.
    primary = {key: value for key, value in raw.items() if key != "derived_comparisons"}
    payload = json.dumps(primary, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def write_manifest(config: Mapping[str, Any], mode: str) -> Path:
    ensure_output_tree()
    manifest = {
        "mode": mode,
        "config_sha256_16": config_fingerprint(config),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": "cpu",
        "config": config,
    }
    path = LOG_DIR / f"run_manifest_{mode}.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def common_metadata(config, run_spec, *, seed: int, k: int, snr_db: float, trial: int) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "K": int(k),
        "SNR_db": float(snr_db),
        "trial": int(trial),
        "M": int(config.atoms),
        "L": int(config.snapshots),
        "dz_m": float(config.dz_m),
        "lambda_bottom": float(config.lambda_bottom),
        "selector_physics_gamma": float(config.selector_gamma),
        "safe_root_refinement": 0,
        "run_mode": run_spec.mode,
        "config_id": config_fingerprint(config.raw),
    }


def classify_failure(*, exception: Exception | None, finite: bool, selection_success: bool, max_error: float, tolerance: float) -> str:
    if exception is not None:
        if isinstance(exception, (FloatingPointError, OverflowError)):
            return "NUMERICAL_FAILURE"
        return "CODE_OR_INPUT_ERROR"
    if not finite:
        return "NUMERICAL_FAILURE"
    if not selection_success:
        return "RANDOM_OPTIMIZATION_OR_SELECTOR_FAILURE"
    if max_error > tolerance:
        return "METHOD_LIMITATION_OR_MODEL_MISMATCH"
    return "SUCCESS"
