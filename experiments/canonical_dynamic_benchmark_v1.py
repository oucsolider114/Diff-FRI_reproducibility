
"""Canonical Dynamic Benchmark v1.

Purpose
-------
Freeze ONE stochastic protocol before any temporal regularization is introduced
inside the modal-wavenumber estimator.

This benchmark deliberately resolves the inconsistency identified by
`audit_dynamic_baseline_consistency.py`:

- one fixed 21-point time grid;
- one fixed physical environment;
- one deterministic seed schedule;
- the exact same observation Y for every method at a given trial/time;
- the exact same decomposition pressure for every basis at a given
  trial/time/scenario;
- 20 independent paired trials;
- no estimator modification;
- no oracle information in Proposed or Alpha-beta tracking.

Compared bases
--------------
1) fixed
2) instantaneous_dynamic
3) alpha_beta_dynamic
4) oracle_dynamic

Scenarios
---------
1) no_coupling
2) known_coupling (fixed 0.55-rad imposed truth event)

The benchmark is intentionally K=3 only. K=5 is already known to be a boundary
regime and is not used to decide whether temporal estimator development is worth
pursuing.

Outputs
-------
outputs/canonical_dynamic_benchmark_v1/

Once COMPLETE.marker exists, the benchmark refuses to overwrite itself unless
`force_recompute` is explicitly enabled in the config.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CONFIG_PATH = ROOT / "configs" / "canonical_dynamic_benchmark_v1.json"


# -----------------------------------------------------------------------------
# Dynamic imports: reuse the already-tested experiment implementation.
# -----------------------------------------------------------------------------
def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


DYN = load_module(
    "canonical_dynamic_helpers",
    ROOT / "experiments" / "dynamic_local_basis_coupling_suite.py",
)
TRK = load_module(
    "canonical_tracking_helpers",
    ROOT / "experiments" / "dynamic_temporal_tracking_coupling_suite.py",
)


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(str(arr.shape).encode("ascii"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                names.append(key)
                seen.add(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=names,
            extrasaction="raise",
            restval="",
        )
        writer.writeheader()
        writer.writerows(rows)


def ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "root": root,
        "cache": root / "cache",
        "csv": root / "csv",
        "arrays": root / "arrays",
        "figures": root / "figures",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def protocol_hash(canonical_cfg: dict, base_cfg_path: Path) -> str:
    payload = {
        "canonical_config": canonical_cfg,
        "base_dynamic_config_sha256": sha256_file(base_cfg_path),
        "benchmark_script_sha256": sha256_file(Path(__file__)),
        "dynamic_helper_sha256": sha256_file(
            ROOT / "experiments" / "dynamic_local_basis_coupling_suite.py"
        ),
        "tracking_helper_sha256": sha256_file(
            ROOT / "experiments" / "dynamic_temporal_tracking_coupling_suite.py"
        ),
    }
    blob = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(blob)


def bootstrap_mean_ci(
    values: np.ndarray,
    seed: int,
    samples: int = 20000,
    level: float = 0.95,
) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    n = x.size
    boot = np.empty(int(samples), dtype=np.float64)
    for i in range(int(samples)):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(np.mean(x[idx]))
    alpha = 0.5 * (1.0 - float(level))
    return (
        float(np.quantile(boot, alpha)),
        float(np.quantile(boot, 1.0 - alpha)),
    )


def mean_cmr_db(leakage: float) -> float:
    leak = max(float(leakage), 1.0e-30)
    desired = max(1.0 - leak, 1.0e-30)
    return 10.0 * math.log10(desired / leak)


def finite_or_none(x: float):
    return float(x) if np.isfinite(float(x)) else None


# -----------------------------------------------------------------------------
# Protocol resolution
# -----------------------------------------------------------------------------
def resolve_protocol() -> tuple[dict, dict, Path]:
    canonical = load_json(CONFIG_PATH)
    base_path = ROOT / canonical["base_dynamic_config"]
    base = load_json(base_path)

    cfg = deepcopy(base)
    cfg["experiment"] = "canonical_dynamic_benchmark_v1"
    cfg["output_root"] = canonical["output_root"]
    cfg["K_values"] = [3]
    cfg["time_points"] = int(canonical["time_points"])
    cfg["time_span_s"] = float(canonical["time_span_s"])
    cfg["ssp_dynamic"]["amplitude_m"] = float(
        canonical["environment"]["amplitude_m"]
    )
    cfg["estimation"]["snr_db"] = float(canonical["estimation"]["snr_db"])
    cfg["estimation"]["snapshots"] = int(
        canonical["estimation"]["snapshots"]
    )
    cfg["estimation"]["base_seed"] = int(
        canonical["estimation"]["base_seed"]
    )
    cfg["tracking"]["snr_db"] = float(
        canonical["decomposition"]["snr_db"]
    )
    cfg["tracking"]["base_seed"] = int(
        canonical["decomposition"]["base_seed"]
    )
    cfg["coupling"]["max_mixing_angle_rad"] = float(
        canonical["coupling"]["strength_rad"]
    )

    # This benchmark uses one estimator run per trial/time. The trial loop is
    # managed here rather than by the older dynamic-suite repeat field.
    cfg["estimation"]["repeats"] = 1

    return canonical, cfg, base_path


# -----------------------------------------------------------------------------
# Environment generation
# -----------------------------------------------------------------------------
def make_environment(cfg: dict, dirs: dict[str, Path], p_hash: str) -> dict:
    cache_path = dirs["cache"] / "environment.npz"

    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        cached_hash = str(cached["protocol_hash"].item())
        if cached_hash == p_hash:
            return {
                "depths": np.asarray(cached["depths_m"], dtype=np.float64),
                "receiver_depths": np.asarray(
                    cached["receiver_depths_m"], dtype=np.float64
                ),
                "times": np.asarray(cached["times_s"], dtype=np.float64),
                "fractions": np.asarray(
                    cached["fractions"], dtype=np.float64
                ),
                "ssp": np.asarray(
                    cached["ssp_matrix_m_s"], dtype=np.float64
                ),
                "eta": np.asarray(
                    cached["thermocline_displacement_m"],
                    dtype=np.float64,
                ),
                "oracle_k": np.asarray(
                    cached["oracle_k"], dtype=np.float64
                ),
                "oracle_basis": np.asarray(
                    cached["oracle_basis"], dtype=np.complex128
                ),
                "fixed_basis": np.asarray(
                    cached["fixed_basis"], dtype=np.complex128
                ),
            }

    frequency = float(cfg["frequency_hz"])
    omega = 2.0 * math.pi * frequency
    water_depth = float(cfg["water_depth_m"])
    dz = float(cfg["dz_m"])

    depths = np.arange(
        0.0,
        water_depth + 0.5 * dz,
        dz,
        dtype=np.float64,
    )
    receiver_cfg = cfg["receiver_array"]
    receiver_depths = np.arange(
        float(receiver_cfg["start_depth_m"]),
        float(receiver_cfg["end_depth_m"])
        + 0.5 * float(receiver_cfg["spacing_m"]),
        float(receiver_cfg["spacing_m"]),
        dtype=np.float64,
    )

    times = np.linspace(
        0.0,
        float(cfg["time_span_s"]),
        int(cfg["time_points"]),
        dtype=np.float64,
    )
    fractions = np.linspace(
        0.0,
        1.0,
        int(cfg["time_points"]),
        dtype=np.float64,
    )

    ssp = np.zeros((times.size, depths.size), dtype=np.float64)
    eta = np.zeros(times.size, dtype=np.float64)
    oracle_k = np.zeros((times.size, 3), dtype=np.float64)
    oracle_basis = np.zeros(
        (times.size, receiver_depths.size, 3),
        dtype=np.complex128,
    )

    for ti, fraction in enumerate(fractions):
        ssp[ti], eta[ti] = DYN.make_dynamic_ssp(
            depths,
            float(fraction),
            cfg["ssp_dynamic"],
        )
        oracle_k[ti] = DYN.find_oracle_roots(
            K=3,
            omega=omega,
            ssp=ssp[ti],
            dz=dz,
            bottom=cfg["forward_bottom"],
            scan_points=int(cfg["oracle_root_scan_points"]),
        )
        oracle_basis[ti] = DYN.build_water_column_basis(
            oracle_k[ti],
            omega,
            ssp[ti],
            dz,
            receiver_depths,
            water_depth,
        )

    fixed_basis = oracle_basis[0].copy()

    np.savez_compressed(
        cache_path,
        protocol_hash=np.asarray(p_hash),
        depths_m=depths,
        receiver_depths_m=receiver_depths,
        times_s=times,
        fractions=fractions,
        ssp_matrix_m_s=ssp,
        thermocline_displacement_m=eta,
        oracle_k=oracle_k,
        oracle_basis=oracle_basis,
        fixed_basis=fixed_basis,
    )

    return {
        "depths": depths,
        "receiver_depths": receiver_depths,
        "times": times,
        "fractions": fractions,
        "ssp": ssp,
        "eta": eta,
        "oracle_k": oracle_k,
        "oracle_basis": oracle_basis,
        "fixed_basis": fixed_basis,
    }


# -----------------------------------------------------------------------------
# Estimator + tracker for one trial
# -----------------------------------------------------------------------------
def estimator_seeds(base_seed: int, trial: int, time_index: int):
    root = int(base_seed) + 100000 * int(trial) + 10 * int(time_index)
    return root + 1, root + 2, root + 3


def pressure_seed(base_seed: int, trial: int, scenario_index: int, time_index: int):
    return (
        int(base_seed)
        + 1_000_000 * int(trial)
        + 100_000 * int(scenario_index)
        + int(time_index)
    )


def run_or_load_trial(
    trial: int,
    canonical: dict,
    cfg: dict,
    env: dict,
    dirs: dict[str, Path],
    p_hash: str,
) -> dict:
    cache_path = dirs["cache"] / f"trial_{trial:03d}.npz"

    if cache_path.exists() and bool(canonical["resume"]):
        cached = np.load(cache_path, allow_pickle=False)
        if str(cached["protocol_hash"].item()) == p_hash:
            return {
                "raw_k": np.asarray(cached["raw_k"], dtype=np.float64),
                "tracked_k": np.asarray(
                    cached["tracked_k"], dtype=np.float64
                ),
                "instant_basis": np.asarray(
                    cached["instant_basis"], dtype=np.complex128
                ),
                "tracked_basis": np.asarray(
                    cached["tracked_basis"], dtype=np.complex128
                ),
                "gate_accepted": np.asarray(
                    cached["gate_accepted"], dtype=np.int64
                ),
                "runtime_seconds": np.asarray(
                    cached["runtime_seconds"], dtype=np.float64
                ),
            }

    frequency = float(cfg["frequency_hz"])
    omega = 2.0 * math.pi * frequency
    water_depth = float(cfg["water_depth_m"])
    dz = float(cfg["dz_m"])
    est_cfg = cfg["estimation"]

    times = env["times"]
    receiver_depths = env["receiver_depths"]
    oracle_basis = env["oracle_basis"]
    ssp = env["ssp"]

    T = times.size
    raw_k = np.zeros((T, 3), dtype=np.float64)
    instant_basis = np.zeros(
        (T, receiver_depths.size, 3),
        dtype=np.complex128,
    )
    gate = np.zeros(T, dtype=np.int64)
    runtime = np.zeros(T, dtype=np.float64)

    for ti in range(T):
        coeff_seed, noise_seed, opt_seed = estimator_seeds(
            int(canonical["estimation"]["base_seed"]),
            trial,
            ti,
        )

        C = DYN.random_coefficients(
            3,
            int(canonical["estimation"]["snapshots"]),
            coeff_seed,
        )
        clean_Y = oracle_basis[ti] @ C
        noisy_Y, _ = DYN.add_complex_noise(
            clean_Y,
            float(canonical["estimation"]["snr_db"]),
            noise_seed,
        )

        metadata = DYN.DynamicMetadata(
            K=3,
            frequency=frequency,
            array_geometry=receiver_depths.copy(),
            water_depth=water_depth,
            SSP=ssp[ti].copy(),
        )

        t0 = time.perf_counter()
        estimate, _, accepted, _, _ = DYN.dynamic_topn_refined(
            noisy_Y,
            metadata,
            opt_seed,
            est_cfg,
            dz,
        )
        runtime[ti] = time.perf_counter() - t0
        raw_k[ti] = estimate
        gate[ti] = int(accepted)

        instant_basis[ti] = DYN.build_water_column_basis(
            estimate,
            omega,
            ssp[ti],
            dz,
            receiver_depths,
            water_depth,
        )

    tracked_k = TRK.alpha_beta_track(
        raw_k,
        times,
        canonical["alpha_beta_tracker"],
    )

    tracked_basis = np.zeros_like(instant_basis)
    for ti in range(T):
        tracked_basis[ti] = DYN.build_water_column_basis(
            tracked_k[ti],
            omega,
            ssp[ti],
            dz,
            receiver_depths,
            water_depth,
        )

    np.savez_compressed(
        cache_path,
        protocol_hash=np.asarray(p_hash),
        raw_k=raw_k,
        tracked_k=tracked_k,
        instant_basis=instant_basis,
        tracked_basis=tracked_basis,
        gate_accepted=gate,
        runtime_seconds=runtime,
    )

    return {
        "raw_k": raw_k,
        "tracked_k": tracked_k,
        "instant_basis": instant_basis,
        "tracked_basis": tracked_basis,
        "gate_accepted": gate,
        "runtime_seconds": runtime,
    }


# -----------------------------------------------------------------------------
# Metrics for one trial
# -----------------------------------------------------------------------------
def build_true_coefficients(
    env: dict,
    cfg: dict,
    scenario: str,
) -> np.ndarray:
    fractions = env["fractions"]
    base = DYN.initial_tracking_coefficients(3)
    out = np.zeros((fractions.size, 3), dtype=np.complex128)

    for ti, fraction in enumerate(fractions):
        if scenario == "no_coupling":
            out[ti] = base
        else:
            out[ti], _ = DYN.apply_known_coupling(
                base,
                float(fraction),
                cfg["coupling"],
            )
    return out


def trial_metrics(
    trial: int,
    canonical: dict,
    cfg: dict,
    env: dict,
    trial_data: dict,
    manifest_rows: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    times = env["times"]
    oracle_k = env["oracle_k"]
    oracle_basis = env["oracle_basis"]
    fixed_basis = env["fixed_basis"]

    raw_k = trial_data["raw_k"]
    tracked_k = trial_data["tracked_k"]
    instant_basis = trial_data["instant_basis"]
    tracked_basis = trial_data["tracked_basis"]

    # Estimator / basis quality rows.
    time_rows: list[dict] = []
    for ti, time_s in enumerate(times):
        raw_macs = [
            DYN.column_mac(
                oracle_basis[ti, :, m],
                instant_basis[ti, :, m],
            )
            for m in range(3)
        ]
        tracked_macs = [
            DYN.column_mac(
                oracle_basis[ti, :, m],
                tracked_basis[ti, :, m],
            )
            for m in range(3)
        ]
        time_rows.append(
            {
                "trial": trial,
                "time_index": ti,
                "time_s": float(time_s),
                "raw_k_rmse_rad_m": float(
                    np.sqrt(np.mean((raw_k[ti] - oracle_k[ti]) ** 2))
                ),
                "tracked_k_rmse_rad_m": float(
                    np.sqrt(
                        np.mean((tracked_k[ti] - oracle_k[ti]) ** 2)
                    )
                ),
                "instantaneous_mean_basis_mac": float(np.mean(raw_macs)),
                "tracked_mean_basis_mac": float(np.mean(tracked_macs)),
                "gate_accepted": int(trial_data["gate_accepted"][ti]),
                "runtime_seconds": float(
                    trial_data["runtime_seconds"][ti]
                ),
            }
        )

        # Observation hash manifest: regenerate the exact Y deterministically.
        coeff_seed, noise_seed, opt_seed = estimator_seeds(
            int(canonical["estimation"]["base_seed"]),
            trial,
            ti,
        )
        C = DYN.random_coefficients(
            3,
            int(canonical["estimation"]["snapshots"]),
            coeff_seed,
        )
        clean_Y = oracle_basis[ti] @ C
        noisy_Y, _ = DYN.add_complex_noise(
            clean_Y,
            float(canonical["estimation"]["snr_db"]),
            noise_seed,
        )
        manifest_rows.append(
            {
                "trial": trial,
                "time_index": ti,
                "time_s": float(time_s),
                "coefficient_seed": coeff_seed,
                "noise_seed": noise_seed,
                "optimizer_seed": opt_seed,
                "Y_sha256": sha256_array(noisy_Y),
            }
        )

    scenario_rows: list[dict] = []
    summary_rows: list[dict] = []

    for scenario_index, scenario in enumerate(
        ("no_coupling", "known_coupling")
    ):
        true_coeff = build_true_coefficients(env, cfg, scenario)

        estimates = {
            "fixed": np.zeros_like(true_coeff),
            "instantaneous_dynamic": np.zeros_like(true_coeff),
            "alpha_beta_dynamic": np.zeros_like(true_coeff),
            "oracle_dynamic": np.zeros_like(true_coeff),
        }

        leakage_series = {
            key: np.zeros(times.size, dtype=np.float64)
            for key in estimates
        }

        for ti, time_s in enumerate(times):
            p_seed = pressure_seed(
                int(canonical["decomposition"]["base_seed"]),
                trial,
                scenario_index,
                ti,
            )
            clean_pressure = oracle_basis[ti] @ true_coeff[ti]
            pressure, _ = DYN.add_complex_noise(
                clean_pressure,
                float(canonical["decomposition"]["snr_db"]),
                p_seed,
            )

            bases = {
                "fixed": fixed_basis,
                "instantaneous_dynamic": instant_basis[ti],
                "alpha_beta_dynamic": tracked_basis[ti],
                "oracle_dynamic": oracle_basis[ti],
            }

            row = {
                "trial": trial,
                "scenario": scenario,
                "time_index": ti,
                "time_s": float(time_s),
                "pressure_seed": p_seed,
                "pressure_sha256": sha256_array(pressure),
            }

            for basis_name, basis in bases.items():
                estimate = DYN.decompose_coefficients(basis, pressure)
                estimates[basis_name][ti] = estimate

                leakage, cmr = DYN.single_mode_leakage_metrics(
                    basis,
                    oracle_basis[ti],
                )
                leakage_series[basis_name][ti] = leakage

                row[f"{basis_name}_coeff_rel_error"] = (
                    DYN.coefficient_relative_error(
                        true_coeff[ti],
                        estimate,
                    )
                )
                row[f"{basis_name}_leakage"] = leakage
                row[f"{basis_name}_cmr_db"] = cmr

            scenario_rows.append(row)

        coupling_modes = [
            int(v) - 1 for v in cfg["coupling"]["mode_pair_1based"]
        ]

        for basis_name, estimated in estimates.items():
            coeff_errors = np.asarray(
                [
                    DYN.coefficient_relative_error(
                        true_coeff[ti],
                        estimated[ti],
                    )
                    for ti in range(times.size)
                ],
                dtype=np.float64,
            )

            if scenario == "no_coupling":
                mean_vector = np.mean(
                    estimated,
                    axis=0,
                    keepdims=True,
                )
                base = DYN.initial_tracking_coefficients(3)
                spurious = float(
                    np.linalg.norm(estimated - mean_vector)
                    / max(
                        np.linalg.norm(
                            np.tile(base, (times.size, 1))
                        ),
                        1.0e-15,
                    )
                )
                corr = float("nan")
            else:
                spurious = float("nan")
                corr = DYN.energy_correlation(
                    true_coeff,
                    estimated,
                    coupling_modes,
                )

            mean_leak = float(np.mean(leakage_series[basis_name]))
            summary_rows.append(
                {
                    "trial": trial,
                    "scenario": scenario,
                    "basis": basis_name,
                    "mean_coefficient_relative_error": float(
                        np.mean(coeff_errors)
                    ),
                    "max_coefficient_relative_error": float(
                        np.max(coeff_errors)
                    ),
                    "mean_offmode_energy_fraction": mean_leak,
                    "mean_cross_mode_rejection_db": mean_cmr_db(mean_leak),
                    "spurious_temporal_variation_no_coupling": spurious,
                    "coupled_mode_energy_correlation": corr,
                }
            )

    return time_rows, scenario_rows, summary_rows


# -----------------------------------------------------------------------------
# Paired statistical summary / decision
# -----------------------------------------------------------------------------
def row_lookup(rows: list[dict], scenario: str, basis: str) -> dict:
    match = [
        row
        for row in rows
        if row["scenario"] == scenario and row["basis"] == basis
    ]
    if len(match) != 1:
        raise RuntimeError(
            f"Expected one row for {scenario}/{basis}, got {len(match)}"
        )
    return match[0]


def paired_stats(
    trial_summaries: list[dict],
    canonical: dict,
) -> tuple[list[dict], dict]:
    trials = sorted({int(row["trial"]) for row in trial_summaries})
    paired_rows: list[dict] = []

    # Trial-level paired differences.
    for trial in trials:
        rows = [r for r in trial_summaries if int(r["trial"]) == trial]

        for scenario in ("no_coupling", "known_coupling"):
            fixed = row_lookup(rows, scenario, "fixed")
            inst = row_lookup(rows, scenario, "instantaneous_dynamic")
            track = row_lookup(rows, scenario, "alpha_beta_dynamic")
            oracle = row_lookup(rows, scenario, "oracle_dynamic")

            paired_rows.append(
                {
                    "trial": trial,
                    "scenario": scenario,
                    "tracked_minus_fixed_coeff_error": float(
                        track["mean_coefficient_relative_error"]
                    )
                    - float(fixed["mean_coefficient_relative_error"]),
                    "tracked_minus_instantaneous_coeff_error": float(
                        track["mean_coefficient_relative_error"]
                    )
                    - float(inst["mean_coefficient_relative_error"]),
                    "tracked_minus_fixed_leakage": float(
                        track["mean_offmode_energy_fraction"]
                    )
                    - float(fixed["mean_offmode_energy_fraction"]),
                    "tracked_minus_instantaneous_leakage": float(
                        track["mean_offmode_energy_fraction"]
                    )
                    - float(inst["mean_offmode_energy_fraction"]),
                    "tracked_minus_fixed_spurious": (
                        float(
                            track[
                                "spurious_temporal_variation_no_coupling"
                            ]
                        )
                        - float(
                            fixed[
                                "spurious_temporal_variation_no_coupling"
                            ]
                        )
                        if scenario == "no_coupling"
                        else float("nan")
                    ),
                    "tracked_minus_fixed_coupling_corr": (
                        float(track["coupled_mode_energy_correlation"])
                        - float(fixed["coupled_mode_energy_correlation"])
                        if scenario == "known_coupling"
                        else float("nan")
                    ),
                    "fixed_coeff_error": float(
                        fixed["mean_coefficient_relative_error"]
                    ),
                    "instantaneous_coeff_error": float(
                        inst["mean_coefficient_relative_error"]
                    ),
                    "tracked_coeff_error": float(
                        track["mean_coefficient_relative_error"]
                    ),
                    "oracle_coeff_error": float(
                        oracle["mean_coefficient_relative_error"]
                    ),
                }
            )

    # Aggregate.
    stats_rows: list[dict] = []
    for scenario in ("no_coupling", "known_coupling"):
        subset = [r for r in paired_rows if r["scenario"] == scenario]

        for metric in (
            "tracked_minus_fixed_coeff_error",
            "tracked_minus_instantaneous_coeff_error",
            "tracked_minus_fixed_leakage",
            "tracked_minus_instantaneous_leakage",
            "tracked_minus_fixed_spurious",
            "tracked_minus_fixed_coupling_corr",
        ):
            values = np.asarray(
                [
                    float(r[metric])
                    for r in subset
                    if np.isfinite(float(r[metric]))
                ],
                dtype=np.float64,
            )
            if values.size == 0:
                continue

            ci_low, ci_high = bootstrap_mean_ci(
                values,
                seed=int(canonical["statistics"]["bootstrap_seed"])
                + sum(ord(c) for c in (scenario + metric)),
                samples=int(
                    canonical["statistics"]["bootstrap_samples"]
                ),
                level=float(canonical["statistics"]["ci_level"]),
            )

            higher_is_better = metric.endswith("coupling_corr")
            win_rate = float(
                np.mean(values > 0.0)
                if higher_is_better
                else np.mean(values < 0.0)
            )

            stats_rows.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "n_trials": int(values.size),
                    "mean_paired_difference": float(np.mean(values)),
                    "median_paired_difference": float(
                        np.median(values)
                    ),
                    "std_paired_difference": float(
                        np.std(values, ddof=1)
                    )
                    if values.size > 1
                    else 0.0,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "win_rate": win_rate,
                }
            )

    def stat(scenario: str, metric: str) -> dict:
        match = [
            row
            for row in stats_rows
            if row["scenario"] == scenario
            and row["metric"] == metric
        ]
        if len(match) != 1:
            raise RuntimeError(f"Missing stat {scenario}/{metric}")
        return match[0]

    no_coeff = stat(
        "no_coupling", "tracked_minus_fixed_coeff_error"
    )
    no_spurious = stat(
        "no_coupling", "tracked_minus_fixed_spurious"
    )
    no_leak = stat(
        "no_coupling", "tracked_minus_fixed_leakage"
    )

    known_coeff = stat(
        "known_coupling", "tracked_minus_fixed_coeff_error"
    )
    known_leak = stat(
        "known_coupling", "tracked_minus_fixed_leakage"
    )
    known_corr = stat(
        "known_coupling", "tracked_minus_fixed_coupling_corr"
    )
    known_vs_inst = stat(
        "known_coupling",
        "tracked_minus_instantaneous_coeff_error",
    )

    corr_tolerance = float(
        canonical["decision_rule"][
            "coupling_correlation_tolerance"
        ]
    )

    tracked_beats_instantaneous_stably = (
        float(known_vs_inst["bootstrap_ci_high"]) < 0.0
    )

    tracked_beats_fixed_composite = (
        float(known_coeff["mean_paired_difference"]) < 0.0
        and float(known_leak["mean_paired_difference"]) < 0.0
        and float(known_corr["mean_paired_difference"]) >= -corr_tolerance
        and float(no_coeff["mean_paired_difference"]) < 0.0
        and float(no_spurious["mean_paired_difference"]) < 0.0
        and float(no_leak["mean_paired_difference"]) < 0.0
    )

    tracked_beats_fixed_statistically_on_coeff = (
        float(known_coeff["bootstrap_ci_high"]) < 0.0
        and float(no_coeff["bootstrap_ci_high"]) < 0.0
    )

    if (
        tracked_beats_fixed_composite
        and tracked_beats_fixed_statistically_on_coeff
    ):
        classification = "SUPPORTED"
    elif tracked_beats_instantaneous_stably:
        classification = "PARTIAL_SUPPORT"
    else:
        classification = "NOT_SUPPORTED"

    decision = {
        "classification": classification,
        "tracked_beats_instantaneous_stably": bool(
            tracked_beats_instantaneous_stably
        ),
        "tracked_beats_fixed_composite": bool(
            tracked_beats_fixed_composite
        ),
        "tracked_beats_fixed_statistically_on_coefficient_error": bool(
            tracked_beats_fixed_statistically_on_coeff
        ),
        "coupling_correlation_tolerance": corr_tolerance,
        "interpretation": {
            "SUPPORTED": (
                "Causal tracked dynamic basis is reproducibly better than "
                "the fixed basis under the canonical paired protocol."
            ),
            "PARTIAL_SUPPORT": (
                "Temporal tracking reproducibly improves the instantaneous "
                "dynamic basis, but does not yet establish stable superiority "
                "over the fixed basis."
            ),
            "NOT_SUPPORTED": (
                "The post-hoc tracking layer does not provide reproducible "
                "paired improvement even over the instantaneous dynamic basis."
            ),
        }[classification],
    }

    return paired_rows, {"stats": stats_rows, "decision": decision}


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def plot_paired_coefficients(
    trial_summaries: list[dict],
    scenario: str,
    path: Path,
) -> None:
    trials = sorted({int(r["trial"]) for r in trial_summaries})
    bases = [
        "fixed",
        "instantaneous_dynamic",
        "alpha_beta_dynamic",
        "oracle_dynamic",
    ]
    labels = [
        "Fixed",
        "Instantaneous",
        "Alpha-beta",
        "Oracle",
    ]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x = np.asarray(trials, dtype=np.float64)

    for basis, label in zip(bases, labels):
        y = []
        for trial in trials:
            row = [
                r
                for r in trial_summaries
                if int(r["trial"]) == trial
                and r["scenario"] == scenario
                and r["basis"] == basis
            ][0]
            y.append(float(row["mean_coefficient_relative_error"]))
        ax.plot(x, y, marker="o", linewidth=1.5, label=label)

    ax.set_xlabel("Independent trial")
    ax.set_ylabel("Mean coefficient relative error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_paired_difference(
    paired_rows: list[dict],
    scenario: str,
    path: Path,
) -> None:
    subset = [r for r in paired_rows if r["scenario"] == scenario]
    x = np.asarray([int(r["trial"]) for r in subset])
    y = np.asarray(
        [
            float(r["tracked_minus_fixed_coeff_error"])
            for r in subset
        ]
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.axhline(0.0, linewidth=1.2)
    ax.plot(x, y, marker="o", linewidth=1.4)
    ax.set_xlabel("Independent trial")
    ax.set_ylabel("Tracked - fixed coefficient error")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_k_quality(time_rows: list[dict], path: Path) -> None:
    times = sorted({float(r["time_s"]) for r in time_rows})
    raw_rmse = []
    track_rmse = []
    raw_mac = []
    track_mac = []

    for t in times:
        rows = [r for r in time_rows if float(r["time_s"]) == t]
        raw_rmse.append(
            np.mean([float(r["raw_k_rmse_rad_m"]) for r in rows])
        )
        track_rmse.append(
            np.mean(
                [float(r["tracked_k_rmse_rad_m"]) for r in rows]
            )
        )
        raw_mac.append(
            np.mean(
                [
                    float(r["instantaneous_mean_basis_mac"])
                    for r in rows
                ]
            )
        )
        track_mac.append(
            np.mean(
                [float(r["tracked_mean_basis_mac"]) for r in rows]
            )
        )

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(times, raw_rmse, marker="o", label="Instantaneous k RMSE")
    ax.plot(times, track_rmse, marker="s", label="Tracked k RMSE")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean k RMSE (rad/m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)

    mac_path = path.with_name("04_mean_basis_mac_vs_time.png")
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(times, raw_mac, marker="o", label="Instantaneous basis")
    ax.plot(times, track_mac, marker="s", label="Tracked basis")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean basis MAC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(mac_path, dpi=220)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    canonical, cfg, base_cfg_path = resolve_protocol()
    output_root = ROOT / canonical["output_root"]
    dirs = ensure_dirs(output_root)

    complete_marker = dirs["root"] / "COMPLETE.marker"
    if complete_marker.exists() and not bool(
        canonical.get("force_recompute", False)
    ):
        print("=" * 92)
        print("CANONICAL DYNAMIC BENCHMARK v1 IS ALREADY FROZEN")
        print(f"output = {dirs['root']}")
        print(
            "Set force_recompute=true explicitly only if you intend "
            "to invalidate the frozen benchmark."
        )
        print("=" * 92)
        return

    if bool(canonical.get("force_recompute", False)):
        # Cache overwrite is intentional when force_recompute is explicitly set.
        for path in dirs["cache"].glob("trial_*.npz"):
            path.unlink(missing_ok=True)
        (dirs["cache"] / "environment.npz").unlink(missing_ok=True)
        complete_marker.unlink(missing_ok=True)

    p_hash = protocol_hash(canonical, base_cfg_path)

    resolved_protocol = {
        "protocol_name": "Canonical Dynamic Benchmark v1",
        "protocol_hash": p_hash,
        "canonical_config": canonical,
        "resolved_dynamic_config": cfg,
        "base_dynamic_config_path": str(base_cfg_path),
        "base_dynamic_config_sha256": sha256_file(base_cfg_path),
        "benchmark_script_sha256": sha256_file(Path(__file__)),
        "dynamic_helper_sha256": sha256_file(
            ROOT / "experiments" / "dynamic_local_basis_coupling_suite.py"
        ),
        "tracking_helper_sha256": sha256_file(
            ROOT
            / "experiments"
            / "dynamic_temporal_tracking_coupling_suite.py"
        ),
        "data_fairness_rule": (
            "At each trial/time, one estimator observation Y is generated and "
            "used once by Proposed. At each trial/time/scenario, one pressure "
            "realization is generated and shared by Fixed, Instantaneous, "
            "Alpha-beta, and Oracle bases."
        ),
        "oracle_policy": (
            "Bottom parameters and oracle k/bases are used only to construct "
            "controlled truth and evaluation. They are never inputs to Proposed "
            "or Alpha-beta tracking."
        ),
    }
    (dirs["root"] / "FROZEN_PROTOCOL.json").write_text(
        json.dumps(resolved_protocol, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    env = make_environment(cfg, dirs, p_hash)

    trial_count = int(canonical["trial_count"])
    all_time_rows: list[dict] = []
    all_scenario_rows: list[dict] = []
    all_summary_rows: list[dict] = []
    manifest_rows: list[dict] = []

    print("=" * 92)
    print("CANONICAL DYNAMIC BENCHMARK v1")
    print("=" * 92)
    print(f"protocol hash     : {p_hash}")
    print(f"K                 : 3")
    print(f"time grid         : {env['times'].size} points / {cfg['time_span_s']} s")
    print(f"estimation SNR    : {canonical['estimation']['snr_db']:.1f} dB")
    print(f"decomposition SNR : {canonical['decomposition']['snr_db']:.1f} dB")
    print(f"trials            : {trial_count}")
    print("paired data       : YES")
    print("future samples    : NOT used by Alpha-beta")
    print("bottom in Proposed: NO")
    print("-" * 92)

    for trial in range(trial_count):
        t0 = time.perf_counter()
        trial_data = run_or_load_trial(
            trial,
            canonical,
            cfg,
            env,
            dirs,
            p_hash,
        )
        time_rows, scenario_rows, summary_rows = trial_metrics(
            trial,
            canonical,
            cfg,
            env,
            trial_data,
            manifest_rows,
        )
        all_time_rows.extend(time_rows)
        all_scenario_rows.extend(scenario_rows)
        all_summary_rows.extend(summary_rows)

        known_fixed = row_lookup(
            summary_rows, "known_coupling", "fixed"
        )
        known_track = row_lookup(
            summary_rows, "known_coupling", "alpha_beta_dynamic"
        )
        print(
            f"trial {trial + 1:02d}/{trial_count} | "
            f"known coeff fixed={float(known_fixed['mean_coefficient_relative_error']):.4e} "
            f"tracked={float(known_track['mean_coefficient_relative_error']):.4e} | "
            f"elapsed={time.perf_counter() - t0:.1f}s"
        )

        # Persist partial tables every trial for crash safety.
        write_csv(
            dirs["csv"] / "time_metrics.partial.csv",
            all_time_rows,
        )
        write_csv(
            dirs["csv"] / "decomposition_metrics.partial.csv",
            all_scenario_rows,
        )
        write_csv(
            dirs["csv"] / "trial_summary.partial.csv",
            all_summary_rows,
        )
        write_csv(
            dirs["csv"] / "observation_manifest.partial.csv",
            manifest_rows,
        )

    paired_rows, paired_result = paired_stats(
        all_summary_rows,
        canonical,
    )

    # Final canonical tables.
    write_csv(dirs["csv"] / "time_metrics.csv", all_time_rows)
    write_csv(
        dirs["csv"] / "decomposition_metrics.csv",
        all_scenario_rows,
    )
    write_csv(
        dirs["csv"] / "trial_summary.csv",
        all_summary_rows,
    )
    write_csv(
        dirs["csv"] / "observation_manifest.csv",
        manifest_rows,
    )
    write_csv(
        dirs["csv"] / "paired_trial_differences.csv",
        paired_rows,
    )
    write_csv(
        dirs["csv"] / "paired_statistics.csv",
        paired_result["stats"],
    )

    # Remove partial tables after successful completion.
    for path in dirs["csv"].glob("*.partial.csv"):
        path.unlink(missing_ok=True)

    # Plots.
    plot_paired_coefficients(
        all_summary_rows,
        "known_coupling",
        dirs["figures"] / "01_known_coupling_coeff_error_by_trial.png",
    )
    plot_paired_coefficients(
        all_summary_rows,
        "no_coupling",
        dirs["figures"] / "02_no_coupling_coeff_error_by_trial.png",
    )
    plot_paired_difference(
        paired_rows,
        "known_coupling",
        dirs["figures"] / "03_known_coupling_tracked_minus_fixed.png",
    )
    plot_k_quality(
        all_time_rows,
        dirs["figures"] / "04_mean_k_rmse_vs_time.png",
    )

    # Aggregate basis means.
    aggregate_rows = []
    for scenario in ("no_coupling", "known_coupling"):
        for basis in (
            "fixed",
            "instantaneous_dynamic",
            "alpha_beta_dynamic",
            "oracle_dynamic",
        ):
            rows = [
                r
                for r in all_summary_rows
                if r["scenario"] == scenario and r["basis"] == basis
            ]
            aggregate_rows.append(
                {
                    "scenario": scenario,
                    "basis": basis,
                    "n_trials": len(rows),
                    "mean_coefficient_relative_error": float(
                        np.mean(
                            [
                                float(
                                    r[
                                        "mean_coefficient_relative_error"
                                    ]
                                )
                                for r in rows
                            ]
                        )
                    ),
                    "std_coefficient_relative_error": float(
                        np.std(
                            [
                                float(
                                    r[
                                        "mean_coefficient_relative_error"
                                    ]
                                )
                                for r in rows
                            ],
                            ddof=1,
                        )
                    ),
                    "mean_offmode_energy_fraction": float(
                        np.mean(
                            [
                                float(
                                    r[
                                        "mean_offmode_energy_fraction"
                                    ]
                                )
                                for r in rows
                            ]
                        )
                    ),
                    "mean_cross_mode_rejection_db": float(
                        np.mean(
                            [
                                float(
                                    r[
                                        "mean_cross_mode_rejection_db"
                                    ]
                                )
                                for r in rows
                            ]
                        )
                    ),
                    "mean_spurious_temporal_variation": (
                        float(
                            np.mean(
                                [
                                    float(
                                        r[
                                            "spurious_temporal_variation_no_coupling"
                                        ]
                                    )
                                    for r in rows
                                ]
                            )
                        )
                        if scenario == "no_coupling"
                        else float("nan")
                    ),
                    "mean_coupling_energy_correlation": (
                        float(
                            np.mean(
                                [
                                    float(
                                        r[
                                            "coupled_mode_energy_correlation"
                                        ]
                                    )
                                    for r in rows
                                ]
                            )
                        )
                        if scenario == "known_coupling"
                        else float("nan")
                    ),
                }
            )
    write_csv(
        dirs["csv"] / "aggregate_summary.csv",
        aggregate_rows,
    )

    result = {
        "protocol_hash": p_hash,
        "trial_count": trial_count,
        "decision": paired_result["decision"],
        "aggregate_summary": aggregate_rows,
        "paired_statistics": paired_result["stats"],
    }
    (dirs["root"] / "CANONICAL_RESULT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Final file hashes.
    result_files = [
        dirs["root"] / "FROZEN_PROTOCOL.json",
        dirs["root"] / "CANONICAL_RESULT.json",
        dirs["csv"] / "time_metrics.csv",
        dirs["csv"] / "decomposition_metrics.csv",
        dirs["csv"] / "trial_summary.csv",
        dirs["csv"] / "observation_manifest.csv",
        dirs["csv"] / "paired_trial_differences.csv",
        dirs["csv"] / "paired_statistics.csv",
        dirs["csv"] / "aggregate_summary.csv",
    ]
    file_hash_rows = [
        {
            "relative_path": str(path.relative_to(dirs["root"])),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in result_files
    ]
    write_csv(dirs["root"] / "RESULT_HASHES.csv", file_hash_rows)

    decision = paired_result["decision"]
    report = [
        "# Canonical Dynamic Benchmark v1",
        "",
        f"- Protocol hash: `{p_hash}`",
        f"- Trials: {trial_count}",
        "- K: 3",
        f"- Time points: {env['times'].size}",
        f"- Estimation SNR: {canonical['estimation']['snr_db']} dB",
        f"- Decomposition SNR: {canonical['decomposition']['snr_db']} dB",
        "- Shared paired observations across all compared bases: yes",
        "- Alpha-beta uses only current/past instantaneous estimates: yes",
        "- Bottom parameters supplied to Proposed / Alpha-beta: no",
        "",
        "## Decision",
        "",
        f"**{decision['classification']}**",
        "",
        decision["interpretation"],
        "",
        "The canonical benchmark is the only dynamic protocol that should be "
        "used to decide whether a new temporally regularized estimator is better "
        "than the current instantaneous estimator.",
        "",
        "## Next action",
        "",
    ]

    if decision["classification"] in ("SUPPORTED", "PARTIAL_SUPPORT"):
        report.append(
            "Proceed to a temporally regularized / warm-started modal-wavenumber "
            "estimator, but preserve this benchmark unchanged for paired comparison."
        )
    else:
        report.append(
            "Do not proceed to a more complex dynamic estimator until the failure "
            "mode is reviewed; post-hoc temporal tracking is not reproducibly useful "
            "under the frozen canonical protocol."
        )

    (dirs["root"] / "REPORT.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    complete_marker.write_text(
        f"Canonical Dynamic Benchmark v1 complete\nprotocol_hash={p_hash}\n",
        encoding="utf-8",
    )

    print()
    print("=" * 92)
    print("CANONICAL BENCHMARK COMPLETE")
    print("=" * 92)
    print(f"classification : {decision['classification']}")
    print(
        "track > instant : "
        f"{decision['tracked_beats_instantaneous_stably']}"
    )
    print(
        "track > fixed composite : "
        f"{decision['tracked_beats_fixed_composite']}"
    )
    print(
        "track > fixed coeff statistically : "
        f"{decision['tracked_beats_fixed_statistically_on_coefficient_error']}"
    )
    print(f"protocol hash : {p_hash}")
    print(f"outputs       : {dirs['root']}")
    print("=" * 92)


if __name__ == "__main__":
    main()
