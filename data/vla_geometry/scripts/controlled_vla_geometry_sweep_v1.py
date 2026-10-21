# -*- coding: utf-8 -*-
"""
Controlled VLA aperture / receiver-sampling sweep for NN-FRI + CSMT.

Purpose
-------
Answer one narrow reviewer/advisor question without retuning the method:
how does the frozen NN-FRI -> CSMT workflow behave when only the receiver
mask changes?

Scientific design
-----------------
- Uses the already frozen canonical 40-m / 300-Hz evolving environment.
- Generates ONE full-array noisy observation Y for each trial/frame.
- Every geometry receives a ROW-SUBSET of that exact same Y.
  No signal or noise is regenerated after masking.
- The supplied SSP, oracle/model-reference branches, random coefficients,
  full-array noise realization, estimator seed, K=3, search parameters,
  score definitions, and success threshold are identical across masks.
- Only receiver locations / array aperture / receiver count change.
- NN-FRI Full is run once at frame 0 for every mask.
- CSMT then performs local online updates for frames 1..T-1 with NO
  reacquisition and NO geometry-specific retuning.
- KRAKEN/model-reference roots are used only after estimation for scoring.

This is intentionally a single-factor geometry audit.  If a shortened or
sparse array fails, do NOT retune the local window / beam search and rerun
under the same experiment name.  A changed parameter set is a new protocol.

Expected repository layout
--------------------------
Run this file from the NN_FRI project repository, e.g.

    python experiments/controlled_vla_geometry_sweep_v1.py

Required existing files from the frozen project:
    outputs/canonical_dynamic_benchmark_v1/FROZEN_PROTOCOL.json
    outputs/canonical_dynamic_benchmark_v1/cache/environment.npz
    experiments/dynamic_local_basis_coupling_suite.py
    src/mode_fri/causal_ssp_modal_tracking.py

If the canonical cache is missing, first run the already-frozen canonical
benchmark that created it.  This script never overwrites canonical outputs.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Repository and frozen dependencies
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mode_fri.causal_ssp_modal_tracking import (  # noqa: E402
    CSMTConfig,
    CausalSSPModalTracking,
)

CANONICAL_ROOT = ROOT / "outputs" / "canonical_dynamic_benchmark_v1"
CANONICAL_PROTOCOL = CANONICAL_ROOT / "FROZEN_PROTOCOL.json"
CANONICAL_ENV = CANONICAL_ROOT / "cache" / "environment.npz"
DYN_HELPER_PATH = ROOT / "experiments" / "dynamic_local_basis_coupling_suite.py"

CONFIG_PATH = ROOT / "configs" / "controlled_vla_geometry_sweep_v1.json"

UINT32_SEED_MODULUS = 2 ** 32


def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DYN = load_module("controlled_vla_geometry_dynamic_helpers", DYN_HELPER_PATH)


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(value) -> str:
    blob = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(str(arr.shape).encode("ascii"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def seed32(namespace: str, *parts) -> int:
    text = "|".join([str(namespace), *[str(x) for x in parts]])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def legacy_numpy_seed(seed: int) -> int:
    return int(seed) % UINT32_SEED_MODULUS


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise", restval="")
        writer.writeheader()
        writer.writerows(rows)


def ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "root": root,
        "cache": root / "cache",
        "csv": root / "csv",
        "figures": root / "figures",
        "audit": root / "audit",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    samples: int,
    level: float,
) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(samples), dtype=np.float64)
    for i in range(int(samples)):
        idx = rng.integers(0, x.size, size=x.size)
        boot[i] = float(np.mean(x[idx]))
    alpha = 0.5 * (1.0 - float(level))
    return (
        float(np.quantile(boot, alpha)),
        float(np.quantile(boot, 1.0 - alpha)),
    )


# -----------------------------------------------------------------------------
# Frozen environment and protocol
# -----------------------------------------------------------------------------
def load_canonical_environment() -> tuple[dict, dict, dict, dict]:
    if not CANONICAL_PROTOCOL.exists() or not CANONICAL_ENV.exists():
        raise FileNotFoundError(
            "The frozen canonical environment is missing. Expected:\n"
            f"  {CANONICAL_PROTOCOL}\n"
            f"  {CANONICAL_ENV}\n"
            "Run the already-frozen canonical_dynamic_benchmark_v1 first."
        )

    frozen = load_json(CANONICAL_PROTOCOL)
    canonical_cfg = frozen["canonical_config"]
    dynamic_cfg = frozen["resolved_dynamic_config"]

    data = np.load(CANONICAL_ENV, allow_pickle=False)
    env = {
        "times": np.asarray(data["times_s"], dtype=np.float64),
        "ssp": np.asarray(data["ssp_matrix_m_s"], dtype=np.float64),
        "oracle_k": np.asarray(data["oracle_k"], dtype=np.float64),
        "oracle_basis": np.asarray(data["oracle_basis"], dtype=np.complex128),
        "receiver_depths": np.asarray(data["receiver_depths_m"], dtype=np.float64),
    }

    if env["oracle_k"].shape[1] != 3:
        raise RuntimeError("This frozen geometry audit is preregistered for K=3 only.")
    if env["oracle_basis"].shape[1] != env["receiver_depths"].size:
        raise RuntimeError("Canonical oracle basis / receiver-depth shape mismatch.")

    return frozen, canonical_cfg, dynamic_cfg, env


def validate_masks(cfg: dict, receiver_depths: np.ndarray) -> dict[str, np.ndarray]:
    n = int(receiver_depths.size)
    masks: dict[str, np.ndarray] = {}
    for item in cfg["geometry_variants"]:
        name = str(item["name"])
        idx = np.asarray(item["zero_based_indices"], dtype=np.int64)
        if idx.ndim != 1 or idx.size < 3:
            raise RuntimeError(f"Mask {name}: need at least 3 one-dimensional receiver indices.")
        if np.any(idx < 0) or np.any(idx >= n):
            raise RuntimeError(f"Mask {name}: receiver index outside [0, {n-1}].")
        if np.unique(idx).size != idx.size:
            raise RuntimeError(f"Mask {name}: duplicate receiver index.")
        if not np.all(np.diff(idx) > 0):
            raise RuntimeError(f"Mask {name}: indices must be strictly increasing.")
        masks[name] = idx

    baseline = str(cfg["baseline_variant"])
    if baseline not in masks:
        raise RuntimeError(f"baseline_variant={baseline!r} is not defined.")
    if masks[baseline].size != n or not np.array_equal(masks[baseline], np.arange(n)):
        raise RuntimeError("Baseline geometry must contain every canonical receiver in order.")
    return masks


def csmt_config_from_json(cfg: dict) -> CSMTConfig:
    c = cfg["csmt_frozen_parameters"]
    return CSMTConfig(
        beta_pred=float(c["beta_pred"]),
        delta0_rad_m=float(c["delta0_rad_m"]),
        delta_min_rad_m=float(c["delta_min_rad_m"]),
        delta_max_rad_m=float(c["delta_max_rad_m"]),
        lambda_k=float(c["lambda_k"]),
        lambda_ssp=float(c["lambda_ssp"]),
        minimum_mode_separation_rad_m=float(c["minimum_mode_separation_rad_m"]),
        local_grid_points=int(c["local_grid_points"]),
        local_coordinate_sweeps=int(c["local_coordinate_sweeps"]),
        enable_quadratic_subgrid=bool(c["enable_quadratic_subgrid"]),
    )


def protocol_payload(
    cfg: dict,
    frozen_canonical: dict,
    masks: dict[str, np.ndarray],
    env: dict,
) -> dict:
    geometry = []
    full_depths = env["receiver_depths"]
    full_span = float(full_depths[-1] - full_depths[0])
    for item in cfg["geometry_variants"]:
        name = str(item["name"])
        idx = masks[name]
        depths = full_depths[idx]
        geometry.append(
            {
                "name": name,
                "description": str(item["description"]),
                "zero_based_indices": [int(x) for x in idx],
                "one_based_sensor_numbers": [int(x) + 1 for x in idx],
                "receiver_depths_m": [float(x) for x in depths],
                "receiver_count": int(idx.size),
                "physical_aperture_m": float(depths[-1] - depths[0]),
                "aperture_fraction_of_full": (
                    float((depths[-1] - depths[0]) / full_span) if full_span > 0 else 1.0
                ),
            }
        )

    return {
        "experiment": str(cfg["experiment"]),
        "scientific_question": (
            "Single-factor receiver-mask audit of NN-FRI initialization and complete CSMT sequence."
        ),
        "script_sha256": sha256_file(Path(__file__)),
        "canonical_protocol_hash": str(frozen_canonical["protocol_hash"]),
        "canonical_environment_sha256": sha256_file(CANONICAL_ENV),
        "dynamic_helper_sha256": sha256_file(DYN_HELPER_PATH),
        "trial_count": int(cfg["trial_count"]),
        "seed_namespace": str(cfg["seed_namespace"]),
        "success_threshold_rad_m": float(cfg["success_threshold_rad_m"]),
        "K": 3,
        "same_full_array_Y_then_mask_rows": True,
        "regenerate_noise_after_masking": False,
        "retune_any_algorithm_parameter_by_geometry": False,
        "full_initialization_frames_per_sequence": 1,
        "reacquisition_after_initialization": False,
        "csmt_frozen_parameters": cfg["csmt_frozen_parameters"],
        "geometry_variants": geometry,
        "statistics": cfg["statistics"],
        "notes": cfg.get("notes", []),
    }


def lock_protocol(dirs: dict[str, Path], payload: dict) -> str:
    p_hash = sha256_json(payload)
    path = dirs["root"] / "PROTOCOL_LOCK.json"
    if path.exists():
        old = load_json(path)
        if str(old["protocol_hash"]) != p_hash:
            raise RuntimeError(
                "PROTOCOL LOCK MISMATCH. Do not modify masks, seeds, or parameters after "
                "the geometry sweep has started. Use a new experiment version instead."
            )
    else:
        path.write_text(
            json.dumps(
                {
                    "protocol_hash": p_hash,
                    "locked_payload": payload,
                    "note": "Frozen before the first receiver-mask trial is evaluated.",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (dirs["root"] / "EXPERIMENT_STARTED.marker").write_text(
            f"controlled VLA geometry sweep started\nprotocol_hash={p_hash}\n",
            encoding="utf-8",
        )
    return p_hash


# -----------------------------------------------------------------------------
# Observation and frozen Full estimator
# -----------------------------------------------------------------------------
def make_full_array_observation(
    *,
    oracle_basis_full: np.ndarray,
    snapshots: int,
    snr_db: float,
    coefficient_seed: int,
    noise_seed: int,
) -> np.ndarray:
    # IMPORTANT: noise is added on the full canonical array BEFORE masking.
    C = DYN.random_coefficients(3, int(snapshots), int(coefficient_seed))
    clean = np.asarray(oracle_basis_full, dtype=np.complex128) @ C
    noisy, _ = DYN.add_complex_noise(clean, float(snr_db), int(noise_seed))
    return np.asarray(noisy, dtype=np.complex128)


def run_full_nn_fri(
    *,
    Y: np.ndarray,
    SSP: np.ndarray,
    receiver_depths: np.ndarray,
    dynamic_cfg: dict,
    optimizer_seed: int,
) -> tuple[np.ndarray, float, bool]:
    metadata = DYN.DynamicMetadata(
        K=3,
        frequency=float(dynamic_cfg["frequency_hz"]),
        array_geometry=np.asarray(receiver_depths, dtype=np.float64).copy(),
        water_depth=float(dynamic_cfg["water_depth_m"]),
        SSP=np.asarray(SSP, dtype=np.float64).copy(),
    )

    t0 = time.perf_counter()
    estimate, _coarse, accepted, _coarse_scores, _refined_scores = DYN.dynamic_topn_refined(
        np.asarray(Y, dtype=np.complex128),
        metadata,
        legacy_numpy_seed(int(optimizer_seed)),
        dynamic_cfg["estimation"],
        float(dynamic_cfg["dz_m"]),
    )
    elapsed = float(time.perf_counter() - t0)
    return np.asarray(estimate, dtype=np.float64), elapsed, bool(accepted)


# -----------------------------------------------------------------------------
# Trial execution
# -----------------------------------------------------------------------------
def score_frame(k_hat: np.ndarray, k_ref: np.ndarray, threshold: float) -> dict:
    err = np.asarray(k_hat, dtype=np.float64) - np.asarray(k_ref, dtype=np.float64)
    abs_err = np.abs(err)
    return {
        "rmse_rad_m": float(np.sqrt(np.mean(err ** 2))),
        "max_component_abs_error_rad_m": float(np.max(abs_err)),
        "success": int(float(np.max(abs_err)) < float(threshold)),
    }


def run_geometry_trial(
    *,
    trial: int,
    variant_name: str,
    mask: np.ndarray,
    cfg: dict,
    canonical_cfg: dict,
    dynamic_cfg: dict,
    env: dict,
    csmt_cfg: CSMTConfig,
    protocol_hash: str,
    dirs: dict[str, Path],
) -> tuple[list[dict], dict]:
    cache_path = dirs["cache"] / f"{variant_name}_trial_{trial:03d}.npz"
    threshold = float(cfg["success_threshold_rad_m"])
    receiver_depths = env["receiver_depths"][mask]
    T = int(env["times"].size)

    # Same seeds for all geometries.  Only row masking changes.
    obs_full: list[np.ndarray] = []
    y_full_hashes: list[str] = []
    y_mask_hashes: list[str] = []
    opt_seeds: list[int] = []
    for ti in range(T):
        coeff_seed = seed32(cfg["seed_namespace"], "coeff", trial, ti)
        noise_seed = seed32(cfg["seed_namespace"], "noise", trial, ti)
        opt_seed = seed32(cfg["seed_namespace"], "optimizer", trial, ti)
        Y_full = make_full_array_observation(
            oracle_basis_full=env["oracle_basis"][ti],
            snapshots=int(canonical_cfg["estimation"]["snapshots"]),
            snr_db=float(canonical_cfg["estimation"]["snr_db"]),
            coefficient_seed=coeff_seed,
            noise_seed=noise_seed,
        )
        obs_full.append(Y_full)
        y_full_hashes.append(sha256_array(Y_full))
        y_mask_hashes.append(sha256_array(Y_full[mask, :]))
        opt_seeds.append(opt_seed)

    if cache_path.exists() and bool(cfg.get("resume", True)):
        data = np.load(cache_path, allow_pickle=False)
        if str(data["protocol_hash"].item()) != protocol_hash:
            raise RuntimeError(f"Cache protocol hash mismatch: {cache_path}")
        # Full-Y hashes are geometry invariant and prove paired observations.
        cached_full = [str(x) for x in data["y_full_hashes"]]
        if cached_full != y_full_hashes:
            raise RuntimeError(f"Full observation hash mismatch: {cache_path}")
        k_series = np.asarray(data["k_series"], dtype=np.float64)
        runtime = np.asarray(data["runtime_s"], dtype=np.float64)
        residual_mean = np.asarray(data["residual_mean"], dtype=np.float64)
        residual_worst = np.asarray(data["residual_worst"], dtype=np.float64)
        init_gate_accepted = bool(data["init_gate_accepted"].item())
    else:
        Y0 = obs_full[0][mask, :]
        k0, init_runtime, init_gate_accepted = run_full_nn_fri(
            Y=Y0,
            SSP=env["ssp"][0],
            receiver_depths=receiver_depths,
            dynamic_cfg=dynamic_cfg,
            optimizer_seed=opt_seeds[0],
        )

        processor = CausalSSPModalTracking(csmt_cfg)
        init = processor.initialize_from_full_estimate(
            k_hat=k0,
            SSP_t=env["ssp"][0],
            Y_t=Y0,
            frequency_hz=float(dynamic_cfg["frequency_hz"]),
            dz=float(dynamic_cfg["dz_m"]),
            receiver_depths=receiver_depths,
            water_depth=float(dynamic_cfg["water_depth_m"]),
            full_runtime_seconds=float(init_runtime),
        )

        k_series = np.zeros((T, 3), dtype=np.float64)
        runtime = np.zeros(T, dtype=np.float64)
        residual_mean = np.zeros(T, dtype=np.float64)
        residual_worst = np.zeros(T, dtype=np.float64)

        k_series[0] = init.k_hat
        runtime[0] = init.total_runtime_seconds
        residual_mean[0] = init.residual_mean
        residual_worst[0] = init.residual_worst

        for ti in range(1, T):
            result = processor.step(
                Y_t=obs_full[ti][mask, :],
                SSP_t=env["ssp"][ti],
                frequency_hz=float(dynamic_cfg["frequency_hz"]),
                dz=float(dynamic_cfg["dz_m"]),
                receiver_depths=receiver_depths,
                water_depth=float(dynamic_cfg["water_depth_m"]),
            )
            k_series[ti] = result.k_hat
            runtime[ti] = result.total_runtime_seconds
            residual_mean[ti] = result.residual_mean
            residual_worst[ti] = result.residual_worst

        np.savez_compressed(
            cache_path,
            protocol_hash=np.asarray(protocol_hash),
            variant_name=np.asarray(variant_name),
            mask=np.asarray(mask, dtype=np.int64),
            receiver_depths_m=np.asarray(receiver_depths, dtype=np.float64),
            k_series=k_series,
            runtime_s=runtime,
            residual_mean=residual_mean,
            residual_worst=residual_worst,
            init_gate_accepted=np.asarray(init_gate_accepted),
            y_full_hashes=np.asarray(y_full_hashes),
            y_mask_hashes=np.asarray(y_mask_hashes),
        )

    frame_rows: list[dict] = []
    frame_rmse = []
    frame_success = []
    frame_maxerr = []
    for ti in range(T):
        score = score_frame(k_series[ti], env["oracle_k"][ti], threshold)
        frame_rmse.append(score["rmse_rad_m"])
        frame_success.append(score["success"])
        frame_maxerr.append(score["max_component_abs_error_rad_m"])
        frame_rows.append(
            {
                "trial": int(trial),
                "variant": variant_name,
                "frame": int(ti),
                "time_s": float(env["times"][ti]),
                "receiver_count": int(mask.size),
                "min_receiver_depth_m": float(receiver_depths[0]),
                "max_receiver_depth_m": float(receiver_depths[-1]),
                "aperture_m": float(receiver_depths[-1] - receiver_depths[0]),
                "k1_hat_rad_m": float(k_series[ti, 0]),
                "k2_hat_rad_m": float(k_series[ti, 1]),
                "k3_hat_rad_m": float(k_series[ti, 2]),
                "k1_ref_rad_m": float(env["oracle_k"][ti, 0]),
                "k2_ref_rad_m": float(env["oracle_k"][ti, 1]),
                "k3_ref_rad_m": float(env["oracle_k"][ti, 2]),
                "frame_rmse_rad_m": score["rmse_rad_m"],
                "max_component_abs_error_rad_m": score["max_component_abs_error_rad_m"],
                "frame_success": score["success"],
                "runtime_s": float(runtime[ti]),
                "projection_residual_mean": float(residual_mean[ti]),
                "projection_residual_worst": float(residual_worst[ti]),
                "full_array_Y_sha256": y_full_hashes[ti],
                "masked_Y_sha256": y_mask_hashes[ti],
            }
        )

    frame_rmse = np.asarray(frame_rmse, dtype=np.float64)
    frame_success = np.asarray(frame_success, dtype=np.float64)
    frame_maxerr = np.asarray(frame_maxerr, dtype=np.float64)

    summary = {
        "trial": int(trial),
        "variant": variant_name,
        "receiver_count": int(mask.size),
        "min_receiver_depth_m": float(receiver_depths[0]),
        "max_receiver_depth_m": float(receiver_depths[-1]),
        "aperture_m": float(receiver_depths[-1] - receiver_depths[0]),
        "static_initializer_rmse_rad_m": float(frame_rmse[0]),
        "static_initializer_max_component_error_rad_m": float(frame_maxerr[0]),
        "static_initializer_success": int(frame_success[0]),
        "static_refinement_gate_accepted": int(init_gate_accepted),
        "csmt_all_frame_mean_rmse_rad_m": float(np.mean(frame_rmse)),
        "csmt_tracking_only_mean_rmse_rad_m": float(np.mean(frame_rmse[1:])),
        "csmt_tracking_only_success_proportion": float(np.mean(frame_success[1:])),
        "csmt_tracking_only_max_component_error_rad_m": float(np.max(frame_maxerr[1:])),
        "full_initialization_runtime_s": float(runtime[0]),
        "csmt_update_runtime_mean_s": float(np.mean(runtime[1:])),
        "csmt_update_runtime_max_s": float(np.max(runtime[1:])),
        "mean_projection_residual_updates": float(np.mean(residual_mean[1:])),
    }
    return frame_rows, summary


# -----------------------------------------------------------------------------
# Aggregation and figures
# -----------------------------------------------------------------------------
def aggregate_results(
    trial_rows: list[dict],
    cfg: dict,
    masks: dict[str, np.ndarray],
    receiver_depths: np.ndarray,
) -> tuple[list[dict], list[dict]]:
    variant_order = [str(x["name"]) for x in cfg["geometry_variants"]]
    baseline = str(cfg["baseline_variant"])
    summary_rows: list[dict] = []

    metrics = [
        "static_initializer_rmse_rad_m",
        "static_initializer_success",
        "csmt_all_frame_mean_rmse_rad_m",
        "csmt_tracking_only_mean_rmse_rad_m",
        "csmt_tracking_only_success_proportion",
        "full_initialization_runtime_s",
        "csmt_update_runtime_mean_s",
    ]

    for variant in variant_order:
        rows = [r for r in trial_rows if r["variant"] == variant]
        idx = masks[variant]
        depths = receiver_depths[idx]
        out = {
            "variant": variant,
            "receiver_count": int(idx.size),
            "receiver_depths_m": ";".join(f"{x:g}" for x in depths),
            "aperture_m": float(depths[-1] - depths[0]),
            "aperture_fraction_of_full": float(
                (depths[-1] - depths[0]) / (receiver_depths[-1] - receiver_depths[0])
            ),
            "trials": int(len(rows)),
        }
        for metric in metrics:
            values = np.asarray([float(r[metric]) for r in rows], dtype=np.float64)
            out[f"mean_{metric}"] = float(np.mean(values))
            out[f"std_{metric}"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        summary_rows.append(out)

    # Paired seed-level differences against full array.
    pair_rows: list[dict] = []
    base_by_trial = {
        int(r["trial"]): r for r in trial_rows if r["variant"] == baseline
    }
    stat_cfg = cfg["statistics"]
    for variant in variant_order:
        if variant == baseline:
            continue
        rows = sorted(
            [r for r in trial_rows if r["variant"] == variant],
            key=lambda r: int(r["trial"]),
        )
        for metric in [
            "static_initializer_rmse_rad_m",
            "csmt_tracking_only_mean_rmse_rad_m",
            "csmt_tracking_only_success_proportion",
        ]:
            diffs = np.asarray(
                [
                    float(r[metric]) - float(base_by_trial[int(r["trial"])][metric])
                    for r in rows
                ],
                dtype=np.float64,
            )
            lo, hi = bootstrap_mean_ci(
                diffs,
                seed=seed32(cfg["seed_namespace"], "bootstrap", variant, metric),
                samples=int(stat_cfg["bootstrap_samples"]),
                level=float(stat_cfg["ci_level"]),
            )
            pair_rows.append(
                {
                    "variant_minus_baseline": f"{variant} - {baseline}",
                    "metric": metric,
                    "mean_paired_difference": float(np.mean(diffs)),
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                    "trial_pairs": int(diffs.size),
                    "interpretation_sign": (
                        "positive=worse for RMSE; positive=better for success"
                        if "success" in metric
                        else "positive=worse"
                    ),
                }
            )

    return summary_rows, pair_rows


def plot_geometry_masks(
    path: Path,
    cfg: dict,
    masks: dict[str, np.ndarray],
    receiver_depths: np.ndarray,
) -> None:
    names = [str(x["name"]) for x in cfg["geometry_variants"]]
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    for xi, name in enumerate(names):
        depths = receiver_depths[masks[name]]
        ax.scatter(np.full(depths.shape, xi, dtype=float), depths, s=30)
    ax.set_xticks(range(len(names)), names, rotation=15, ha="right")
    ax.set_ylabel("Receiver depth (m)")
    ax.set_title("Predeclared receiver masks: only array geometry changes")
    ax.invert_yaxis()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_trial_metric(
    path: Path,
    trial_rows: list[dict],
    cfg: dict,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    names = [str(x["name"]) for x in cfg["geometry_variants"]]
    trials = sorted({int(r["trial"]) for r in trial_rows})
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    x = np.arange(len(names), dtype=float)

    # Thin gray paired trajectories show the same stochastic trial across masks.
    for trial in trials:
        vals = []
        for name in names:
            row = next(r for r in trial_rows if int(r["trial"]) == trial and r["variant"] == name)
            vals.append(float(row[metric]))
        ax.plot(x, vals, marker="o", linewidth=0.8, alpha=0.30)

    means = []
    for name in names:
        vals = [float(r[metric]) for r in trial_rows if r["variant"] == name]
        means.append(float(np.mean(vals)))
    ax.plot(x, means, marker="D", linewidth=2.2, label="Across-trial mean")

    ax.set_xticks(x, names, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_success(path: Path, summary_rows: list[dict]) -> None:
    names = [r["variant"] for r in summary_rows]
    static = [float(r["mean_static_initializer_success"]) for r in summary_rows]
    track = [float(r["mean_csmt_tracking_only_success_proportion"]) for r in summary_rows]
    x = np.arange(len(names), dtype=float)
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    ax.bar(x - width / 2, static, width, label="First-frame NN-FRI success")
    ax.bar(x + width / 2, track, width, label="CSMT update-frame success")
    ax.set_xticks(x, names, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Success proportion")
    ax.set_title("Geometry sensitivity under one fixed max-component error criterion")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_markdown_summary(
    path: Path,
    cfg: dict,
    summary_rows: list[dict],
    pair_rows: list[dict],
    protocol_hash: str,
) -> None:
    lines = [
        "# Controlled VLA geometry sweep v1",
        "",
        f"Protocol hash: `{protocol_hash}`",
        "",
        "## Scientific scope",
        "",
        "This is a receiver-mask-only audit. The full-array acoustic observation is generated once per trial/frame and then row-masked. No noise is regenerated after masking and no NN-FRI/CSMT parameter is retuned by geometry.",
        "",
        "KRAKEN branches are model-reference quantities used only for scoring after estimation. They are not field truth.",
        "",
        "## Geometry summary",
        "",
        "| Variant | N | Aperture (m) | Static RMSE | Static success | CSMT tracking RMSE | CSMT tracking success |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(
            "| {variant} | {receiver_count} | {aperture_m:.1f} | {srmse:.6e} | {ssucc:.3f} | {trmse:.6e} | {tsucc:.3f} |".format(
                variant=r["variant"],
                receiver_count=int(r["receiver_count"]),
                aperture_m=float(r["aperture_m"]),
                srmse=float(r["mean_static_initializer_rmse_rad_m"]),
                ssucc=float(r["mean_static_initializer_success"]),
                trmse=float(r["mean_csmt_tracking_only_mean_rmse_rad_m"]),
                tsucc=float(r["mean_csmt_tracking_only_success_proportion"]),
            )
        )

    lines += [
        "",
        "## Paired differences relative to full array",
        "",
        "| Contrast | Metric | Mean difference | 95% bootstrap CI |",
        "|---|---|---:|---:|",
    ]
    for r in pair_rows:
        lines.append(
            f"| {r['variant_minus_baseline']} | {r['metric']} | {float(r['mean_paired_difference']):.6e} | [{float(r['bootstrap_ci_low']):.6e}, {float(r['bootstrap_ci_high']):.6e}] |"
        )

    lines += [
        "",
        "## Figure semantics",
        "",
        "- `01_geometry_masks.png`: each point is one retained receiver at its physical depth; columns are the four preregistered masks.",
        "- `02_static_initializer_rmse.png`: each thin connected trajectory is the same stochastic trial across masks; diamonds are across-trial means.",
        "- `03_csmt_tracking_rmse.png`: same paired-trial convention for update-frame CSMT RMSE; diamonds are across-trial means.",
        "- `04_success_proportions.png`: left bar in each pair is first-frame NN-FRI success; right bar is the mean CSMT update-frame success proportion. Both use the same frozen max-component threshold.",
        "",
        "## Interpretation rule",
        "",
        "Do not tune the algorithm to rescue a weak geometry. A degradation under a shorter contiguous aperture is itself the requested observability/aperture boundary. The sparse-full-span variant helps separate receiver-count loss from physical-aperture loss, but it is still one controlled synthetic environment rather than a universal array-design theorem.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    cfg = load_json(CONFIG_PATH)
    output_root = ROOT / cfg["output_root"]
    dirs = ensure_dirs(output_root)

    complete_marker = dirs["root"] / "COMPLETE.marker"
    if complete_marker.exists() and not bool(cfg.get("force_recompute", False)):
        print("Controlled VLA geometry sweep already complete.")
        print(f"Output: {output_root}")
        return

    frozen_canonical, canonical_cfg, dynamic_cfg, env = load_canonical_environment()
    masks = validate_masks(cfg, env["receiver_depths"])
    csmt_cfg = csmt_config_from_json(cfg)

    payload = protocol_payload(cfg, frozen_canonical, masks, env)
    p_hash = lock_protocol(dirs, payload)

    # Geometry protocol CSV is written before any scientific result table.
    geometry_rows = []
    full_span = float(env["receiver_depths"][-1] - env["receiver_depths"][0])
    descriptions = {str(x["name"]): str(x["description"]) for x in cfg["geometry_variants"]}
    for name in [str(x["name"]) for x in cfg["geometry_variants"]]:
        idx = masks[name]
        depths = env["receiver_depths"][idx]
        geometry_rows.append(
            {
                "variant": name,
                "description": descriptions[name],
                "zero_based_indices": ";".join(str(int(x)) for x in idx),
                "one_based_sensor_numbers": ";".join(str(int(x) + 1) for x in idx),
                "receiver_depths_m": ";".join(f"{x:g}" for x in depths),
                "receiver_count": int(idx.size),
                "aperture_m": float(depths[-1] - depths[0]),
                "aperture_fraction_of_full": float((depths[-1] - depths[0]) / full_span),
            }
        )
    write_csv(dirs["csv"] / "geometry_protocol.csv", geometry_rows)
    plot_geometry_masks(
        dirs["figures"] / "01_geometry_masks.png",
        cfg,
        masks,
        env["receiver_depths"],
    )

    all_frame_rows: list[dict] = []
    all_trial_rows: list[dict] = []
    variants = [str(x["name"]) for x in cfg["geometry_variants"]]

    print("=" * 92)
    print("CONTROLLED VLA GEOMETRY SWEEP v1")
    print("=" * 92)
    print(f"protocol hash : {p_hash}")
    print(f"trials        : {int(cfg['trial_count'])}")
    print(f"variants      : {', '.join(variants)}")
    print("pairing rule  : generate full-array Y once, then mask rows")
    print("retuning      : NO")
    print("reacquisition : NO")

    for trial in range(int(cfg["trial_count"])):
        print(f"\ntrial {trial:02d}")
        for variant in variants:
            frame_rows, trial_summary = run_geometry_trial(
                trial=trial,
                variant_name=variant,
                mask=masks[variant],
                cfg=cfg,
                canonical_cfg=canonical_cfg,
                dynamic_cfg=dynamic_cfg,
                env=env,
                csmt_cfg=csmt_cfg,
                protocol_hash=p_hash,
                dirs=dirs,
            )
            all_frame_rows.extend(frame_rows)
            all_trial_rows.append(trial_summary)
            print(
                f"  {variant:20s} | init={trial_summary['static_initializer_rmse_rad_m']:.3e} "
                f"| track={trial_summary['csmt_tracking_only_mean_rmse_rad_m']:.3e} "
                f"| succ={trial_summary['csmt_tracking_only_success_proportion']:.3f}"
            )

        # Incremental recovery tables in case a later trial is interrupted.
        write_csv(dirs["csv"] / "frame_metrics.partial.csv", all_frame_rows)
        write_csv(dirs["csv"] / "trial_summary.partial.csv", all_trial_rows)

    summary_rows, pair_rows = aggregate_results(
        all_trial_rows,
        cfg,
        masks,
        env["receiver_depths"],
    )

    write_csv(dirs["csv"] / "frame_metrics.csv", all_frame_rows)
    write_csv(dirs["csv"] / "trial_summary.csv", all_trial_rows)
    write_csv(dirs["csv"] / "geometry_summary.csv", summary_rows)
    write_csv(dirs["csv"] / "paired_differences_vs_full.csv", pair_rows)

    plot_trial_metric(
        dirs["figures"] / "02_static_initializer_rmse.png",
        all_trial_rows,
        cfg,
        "static_initializer_rmse_rad_m",
        "First-frame model-reference k-RMSE (rad/m)",
        "NN-FRI first-frame sensitivity to receiver mask",
    )
    plot_trial_metric(
        dirs["figures"] / "03_csmt_tracking_rmse.png",
        all_trial_rows,
        cfg,
        "csmt_tracking_only_mean_rmse_rad_m",
        "Mean update-frame model-reference k-RMSE (rad/m)",
        "CSMT sequence sensitivity to receiver mask",
    )
    plot_success(
        dirs["figures"] / "04_success_proportions.png",
        summary_rows,
    )

    save_markdown_summary(
        dirs["root"] / "RESULT_SUMMARY.md",
        cfg,
        summary_rows,
        pair_rows,
        p_hash,
    )

    result = {
        "experiment": cfg["experiment"],
        "protocol_hash": p_hash,
        "completed": True,
        "trial_count": int(cfg["trial_count"]),
        "geometry_summary": summary_rows,
        "paired_differences_vs_full": pair_rows,
        "claim_boundary": [
            "Single controlled evolving KRAKEN-forward environment only.",
            "Receiver mask is the only changed experimental factor.",
            "KRAKEN branches are model-reference quantities, not field truth.",
            "No geometry-specific retuning was performed.",
            "A failure of a short/sparse mask is retained as an operating-boundary result.",
        ],
    }
    (dirs["root"] / "RESULT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Hash principal outputs for reproducibility.
    principal = [
        dirs["root"] / "PROTOCOL_LOCK.json",
        dirs["root"] / "RESULT.json",
        dirs["root"] / "RESULT_SUMMARY.md",
        dirs["csv"] / "geometry_protocol.csv",
        dirs["csv"] / "frame_metrics.csv",
        dirs["csv"] / "trial_summary.csv",
        dirs["csv"] / "geometry_summary.csv",
        dirs["csv"] / "paired_differences_vs_full.csv",
        dirs["figures"] / "01_geometry_masks.png",
        dirs["figures"] / "02_static_initializer_rmse.png",
        dirs["figures"] / "03_csmt_tracking_rmse.png",
        dirs["figures"] / "04_success_proportions.png",
    ]
    hash_rows = [
        {
            "relative_path": str(p.relative_to(dirs["root"])),
            "sha256": sha256_file(p),
            "bytes": int(p.stat().st_size),
        }
        for p in principal
    ]
    write_csv(dirs["root"] / "RESULT_HASHES.csv", hash_rows)

    (dirs["csv"] / "frame_metrics.partial.csv").unlink(missing_ok=True)
    (dirs["csv"] / "trial_summary.partial.csv").unlink(missing_ok=True)
    complete_marker.write_text(
        f"controlled VLA geometry sweep complete\nprotocol_hash={p_hash}\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 92)
    print("COMPLETE")
    print("=" * 92)
    for r in summary_rows:
        print(
            f"{r['variant']:20s} | N={int(r['receiver_count']):2d} "
            f"| aperture={float(r['aperture_m']):5.1f} m "
            f"| init={float(r['mean_static_initializer_rmse_rad_m']):.3e} "
            f"| track={float(r['mean_csmt_tracking_only_mean_rmse_rad_m']):.3e} "
            f"| track success={float(r['mean_csmt_tracking_only_success_proportion']):.3f}"
        )
    print(f"\nResults: {output_root}")


if __name__ == "__main__":
    main()
