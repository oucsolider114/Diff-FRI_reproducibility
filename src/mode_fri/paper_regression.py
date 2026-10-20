"""Numerical regression against the previously validated QUICK result set."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .final_io import PROJECT_ROOT, save_csv
from .paper_io import PAPER_CSV_DIR


REFERENCE_CSV_DIR = PROJECT_ROOT / "outputs" / "final" / "csv"
L20_REFERENCE_CSV_DIR = PROJECT_ROOT / "tests" / "reference" / "final_paper_L20"
METHOD_TO_REFERENCE = {
    "Grid-GOMP": "Grid-512",
    "Numerov+stratified+lambda0.1": "Numerov+stratified+moderate physics",
}


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Regression reference/result is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _key(dataset: str, row: dict[str, str]) -> tuple[Any, ...]:
    method = METHOD_TO_REFERENCE.get(row.get("method", ""), row.get("method", ""))
    common = (method, row.get("K", ""), row.get("SNR_db", ""), row.get("trial", ""))
    if dataset == "environment":
        return common + (row.get("environment_kind", ""), row.get("environment_offset", ""))
    return common


def _equivalent(reference: str, candidate: str, tolerance: float) -> tuple[bool, float]:
    try:
        ref_value = float(reference)
        new_value = float(candidate)
    except (TypeError, ValueError):
        return reference == candidate, 0.0 if reference == candidate else float("inf")
    if np.isnan(ref_value) and np.isnan(new_value):
        return True, 0.0
    if np.isinf(ref_value) or np.isinf(new_value):
        return ref_value == new_value, 0.0 if ref_value == new_value else float("inf")
    difference = abs(ref_value - new_value)
    return difference <= tolerance, difference


def verify_quick_regression(
    *,
    paper_csv_dir: Path = PAPER_CSV_DIR,
    reference_csv_dir: Path = REFERENCE_CSV_DIR,
    tolerance: float = 1.0e-12,
) -> list[dict[str, Any]]:
    """Compare QUICK estimators with the appropriate frozen reference set.

    Nominal and environment remain strict pre/post-refactor checks against the
    original reference. Baseline and ablation use a separate corrected-L20
    reference because the legacy moving-source generator had silently supplied
    64 snapshots despite the frozen final configuration L=20.
    """
    datasets = {
        "nominal": ("nominal_trials_quick.csv", ("success", "rmse", "max_error", "estimated_k", "true_k")),
        "baseline": ("baseline_trials_quick.csv", ("success", "rmse", "max_error", "estimated_k", "true_k")),
        "ablation": ("ablation_trials_quick.csv", ("success", "rmse", "max_error", "estimated_k", "true_k")),
        "environment": (
            "environment_trials_quick.csv",
            ("success", "rmse", "max_error", "estimated_k", "true_k", "calibration_accepted", "decision_status"),
        ),
    }
    comparisons: list[dict[str, Any]] = []
    for dataset, (filename, columns) in datasets.items():
        paper_rows = _read(paper_csv_dir / filename)
        dataset_reference_dir = (
            L20_REFERENCE_CSV_DIR
            if dataset in {"baseline", "ablation"}
            else reference_csv_dir
        )
        reference_rows = _read(dataset_reference_dir / filename)
        reference_index = {_key(dataset, row): row for row in reference_rows}
        for paper_row in paper_rows:
            row_key = _key(dataset, paper_row)
            if row_key not in reference_index:
                comparisons.append({
                    "dataset": dataset, "reference_set": dataset_reference_dir.name,
                    "key": repr(row_key), "column": "ROW",
                    "reference": "MISSING", "paper": "PRESENT", "abs_difference": float("inf"), "passed": 0,
                })
                continue
            reference_row = reference_index[row_key]
            for column in columns:
                passed, difference = _equivalent(
                    reference_row.get(column, ""), paper_row.get(column, ""), tolerance
                )
                comparisons.append({
                    "dataset": dataset,
                    "reference_set": dataset_reference_dir.name,
                    "key": repr(row_key),
                    "column": column,
                    "reference": reference_row.get(column, ""),
                    "paper": paper_row.get(column, ""),
                    "abs_difference": difference,
                    "passed": int(passed),
                })
    output_path = paper_csv_dir / "regression_quick.csv"
    save_csv(output_path, comparisons)
    failures = [row for row in comparisons if not row["passed"]]
    if failures:
        raise AssertionError(
            f"QUICK regression failed in {len(failures)} of {len(comparisons)} comparisons; "
            f"see {output_path}"
        )
    return comparisons
