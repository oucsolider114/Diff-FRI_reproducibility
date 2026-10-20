"""Paths and manifests for the non-destructive final-paper result set."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch

from .final_io import PROJECT_ROOT, config_fingerprint


FINAL_PAPER_OUTPUT = PROJECT_ROOT / "outputs" / "final_paper"
PAPER_CSV_DIR = FINAL_PAPER_OUTPUT / "csv"
PAPER_FIGURE_DIR = FINAL_PAPER_OUTPUT / "figures"
PAPER_TABLE_DIR = FINAL_PAPER_OUTPUT / "tables"
PAPER_LOG_DIR = FINAL_PAPER_OUTPUT / "logs"
PAPER_SUMMARY_PATH = FINAL_PAPER_OUTPUT / "final_results_summary.md"


def ensure_paper_output_tree() -> None:
    for path in (
        PAPER_CSV_DIR,
        PAPER_FIGURE_DIR,
        PAPER_TABLE_DIR,
        PAPER_LOG_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def write_paper_manifest(config: dict, mode: str) -> Path:
    ensure_paper_output_tree()
    manifest = {
        "mode": mode,
        "result_set": "final_paper",
        "config_sha256_16": config_fingerprint(config),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": "cpu",
        "config": config,
    }
    path = PAPER_LOG_DIR / f"run_manifest_{mode}.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
