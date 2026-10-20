"""Final unified robustness suite for dynamic local-basis modal decomposition.

This experiment consolidates the remaining controlled-simulation questions into
ONE resumable suite.  It does not modify the frozen estimator.  It reuses the
already-created experiment helpers and evaluates the operating envelope of the
causal dynamic-basis idea across:

    - estimator SNR,
    - thermocline displacement amplitude,
    - environment variation rate,
    - imposed modal-coupling strength,
    - K = 3 main regime and K = 5 boundary regime.

Compared bases:
    1) fixed
    2) instantaneous_dynamic
    3) alpha_beta_dynamic   (causal temporal tracking)
    4) oracle_dynamic       (controlled upper reference only)

Important scientific boundary:
    Bottom parameters are used only by the controlled forward/oracle root
    generator.  They are never supplied to the Proposed estimator, the causal
    tracker, or local-mode recovery from estimated k.

The suite is checkpointed condition-by-condition.  If a run is interrupted,
relaunching the same script resumes from completed caches instead of rerunning
finished estimator conditions.
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


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CONFIG_PATH = ROOT / "configs" / "final_dynamic_robustness_suite.json"


def _load_python_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required helper script not found: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DYN = _load_python_module(
    "dynamic_local_basis_coupling_suite_helpers",
    ROOT / "experiments" / "dynamic_local_basis_coupling_suite.py",
)
TRK = _load_python_module(
    "dynamic_temporal_tracking_coupling_suite_helpers",
    ROOT / "experiments" / "dynamic_temporal_tracking_coupling_suite.py",
)


# -----------------------------------------------------------------------------
# Generic I/O
# -----------------------------------------------------------------------------
def load_config(path: Path = CONFIG_PATH) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    required = [
        "output_root",
        "base_dynamic_config",
        "time_points",
        "estimator_repeats",
        "design",
        "alpha_beta_tracker",
        "coupling_strengths_rad",
    ]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise KeyError(f"Missing config keys: {missing}")
    return cfg


def ensure_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "cache_env": root / "cache" / "environment",
        "cache_est": root / "cache" / "estimation",
        "csv": root / "csv",
        "figures": root / "figures",
        "arrays": root / "arrays",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


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
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="raise", restval="")
        writer.writeheader()
        writer.writerows(rows)


def stable_int(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def condition_id(K: int, snr_db: float, amplitude_m: float, rate_cycles: float) -> str:
    def clean(value: float) -> str:
        return str(value).replace("-", "m").replace(".", "p")
    return (
        f"K{int(K)}_SNR{clean(float(snr_db))}_"
        f"A{clean(float(amplitude_m))}_R{clean(float(rate_cycles))}"
    )


def environment_id(K: int, amplitude_m: float, rate_cycles: float) -> str:
    return condition_id(K, 999.0, amplitude_m, rate_cycles).replace("_SNR999p0", "")


# -----------------------------------------------------------------------------
# Experiment design
# -----------------------------------------------------------------------------
def build_conditions(cfg: dict) -> list[dict]:
    design = cfg["design"]
    conditions: dict[tuple[int, float, float, float], dict] = {}

    def add(K: int, snr: float, amp: float, rate: float, sweep: str) -> None:
        key = (int(K), float(snr), float(amp), float(rate))
        row = conditions.setdefault(
            key,
            {
                "K": int(K),
                "snr_db": float(snr),
                "amplitude_m": float(amp),
                "rate_cycles_per_span": float(rate),
                "sweeps": set(),
            },
        )
        row["sweeps"].add(sweep)

    b3 = design["K3_baseline"]
    for snr in design["K3_snr_sweep_db"]:
        add(3, snr, b3["amplitude_m"], b3["rate_cycles_per_span"], "K3_snr")
    for amp in design["K3_amplitude_sweep_m"]:
        add(3, b3["snr_db"], amp, b3["rate_cycles_per_span"], "K3_amplitude")
    for rate in design["K3_rate_sweep_cycles_per_span"]:
        add(3, b3["snr_db"], b3["amplitude_m"], rate, "K3_rate")

    b5 = design["K5_baseline"]
    for snr in design["K5_snr_sweep_db"]:
        add(5, snr, b5["amplitude_m"], b5["rate_cycles_per_span"], "K5_snr")
    for stress in design.get("K5_stress_cases", []):
        add(
            5,
            stress.get("snr_db", b5["snr_db"]),
            stress.get("amplitude_m", b5["amplitude_m"]),
            stress.get("rate_cycles_per_span", b5["rate_cycles_per_span"]),
            "K5_stress",
        )

    result = []
    for row in conditions.values():
        row = dict(row)
        row["sweeps"] = ";".join(sorted(row["sweeps"]))
        row["condition_id"] = condition_id(
            row["K"], row["snr_db"], row["amplitude_m"], row["rate_cycles_per_span"]
        )
        result.append(row)
    return sorted(result, key=lambda r: (r["K"], r["snr_db"], r["amplitude_m"], r["rate_cycles_per_span"]))


# -----------------------------------------------------------------------------
# Dynamic SSP with an explicit rate multiplier
# -----------------------------------------------------------------------------
def make_dynamic_ssp_rate(
    depths: np.ndarray,
    time_fraction: float,
    ssp_cfg: dict,
    amplitude_m: float,
    rate_cycles: float,
) -> tuple[np.ndarray, float]:
    z = np.asarray(depths, dtype=np.float64)
    c0 = DYN.reference_ssp(z, ssp_cfg["reference_profile"])
    harmonic = float(ssp_cfg.get("second_harmonic_fraction", 0.25))
    phase = float(ssp_cfg.get("second_harmonic_phase_rad", 0.35))
    x = float(rate_cycles) * float(time_fraction)
    raw = math.sin(2.0 * math.pi * x) + harmonic * math.sin(4.0 * math.pi * x + phase)
    eta = float(amplitude_m) * raw / (1.0 + abs(harmonic))
    center = float(ssp_cfg["center_depth_m"])
    sigma = float(ssp_cfg["vertical_taper_sigma_m"])
    taper = np.exp(-0.5 * ((z - center) / sigma) ** 2)
    shifted_depth = np.clip(z - eta * taper, z[0], z[-1])
    c_t = np.interp(shifted_depth, z, c0)
    return c_t.astype(np.float64), float(eta)


# -----------------------------------------------------------------------------
# Cached environment and estimator runs
# -----------------------------------------------------------------------------
def generate_or_load_environment(
    dirs: dict[str, Path],
    base_cfg: dict,
    suite_cfg: dict,
    K: int,
    amplitude_m: float,
    rate_cycles: float,
) -> dict[str, np.ndarray]:
    eid = environment_id(K, amplitude_m, rate_cycles)
    cache = dirs["cache_env"] / f"{eid}.npz"
    if cache.exists() and bool(suite_cfg.get("resume", True)) and not bool(suite_cfg.get("force_recompute", False)):
        data = np.load(cache, allow_pickle=False)
        return {key: np.asarray(data[key]) for key in data.files}

    frequency = float(base_cfg["frequency_hz"])
    omega = 2.0 * math.pi * frequency
    water_depth = float(base_cfg["water_depth_m"])
    dz = float(base_cfg["dz_m"])
    depths = np.arange(0.0, water_depth + 0.5 * dz, dz, dtype=np.float64)
    receiver_cfg = base_cfg["receiver_array"]
    receiver_depths = np.arange(
        float(receiver_cfg["start_depth_m"]),
        float(receiver_cfg["end_depth_m"]) + 0.5 * float(receiver_cfg["spacing_m"]),
        float(receiver_cfg["spacing_m"]),
        dtype=np.float64,
    )
    time_points = int(suite_cfg["time_points"])
    time_span_s = float(suite_cfg.get("time_span_s", base_cfg["time_span_s"]))
    times_s = np.linspace(0.0, time_span_s, time_points, dtype=np.float64)
    fractions = np.linspace(0.0, 1.0, time_points, dtype=np.float64)

    ssp_matrix = np.zeros((time_points, depths.size), dtype=np.float64)
    eta = np.zeros(time_points, dtype=np.float64)
    for ti, frac in enumerate(fractions):
        ssp_matrix[ti], eta[ti] = make_dynamic_ssp_rate(
            depths,
            float(frac),
            base_cfg["ssp_dynamic"],
            amplitude_m,
            rate_cycles,
        )

    oracle_k = np.zeros((time_points, int(K)), dtype=np.float64)
    oracle_basis = np.zeros((time_points, receiver_depths.size, int(K)), dtype=np.complex128)
    for ti in range(time_points):
        oracle_k[ti] = DYN.find_oracle_roots(
            K=int(K),
            omega=omega,
            ssp=ssp_matrix[ti],
            dz=dz,
            bottom=base_cfg["forward_bottom"],
            scan_points=int(suite_cfg.get("oracle_root_scan_points", base_cfg["oracle_root_scan_points"])),
        )
        oracle_basis[ti] = DYN.build_water_column_basis(
            oracle_k[ti], omega, ssp_matrix[ti], dz, receiver_depths, water_depth
        )

    payload = {
        "depths_m": depths,
        "receiver_depths_m": receiver_depths,
        "times_s": times_s,
        "fractions": fractions,
        "ssp_matrix_m_s": ssp_matrix,
        "eta_m": eta,
        "oracle_k": oracle_k,
        "oracle_basis": oracle_basis,
        "fixed_basis": oracle_basis[0],
    }
    np.savez_compressed(cache, **payload)
    return payload


def run_or_load_estimation_condition(
    dirs: dict[str, Path],
    base_cfg: dict,
    suite_cfg: dict,
    condition: dict,
    env: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    cid = condition["condition_id"]
    cache = dirs["cache_est"] / f"{cid}.npz"
    if cache.exists() and bool(suite_cfg.get("resume", True)) and not bool(suite_cfg.get("force_recompute", False)):
        data = np.load(cache, allow_pickle=False)
        return {key: np.asarray(data[key]) for key in data.files}

    K = int(condition["K"])
    snr_db = float(condition["snr_db"])
    repeats = int(suite_cfg["estimator_repeats"])
    frequency = float(base_cfg["frequency_hz"])
    water_depth = float(base_cfg["water_depth_m"])
    dz = float(base_cfg["dz_m"])
    omega = 2.0 * math.pi * frequency
    est_cfg = dict(base_cfg["estimation"])
    est_cfg["snr_db"] = snr_db
    est_cfg["repeats"] = repeats

    oracle_k = np.asarray(env["oracle_k"], dtype=np.float64)
    oracle_basis = np.asarray(env["oracle_basis"], dtype=np.complex128)
    ssp_matrix = np.asarray(env["ssp_matrix_m_s"], dtype=np.float64)
    receiver_depths = np.asarray(env["receiver_depths_m"], dtype=np.float64)
    times_s = np.asarray(env["times_s"], dtype=np.float64)
    T = times_s.size

    raw_k = np.zeros((repeats, T, K), dtype=np.float64)
    raw_basis = np.zeros((repeats, T, receiver_depths.size, K), dtype=np.complex128)
    accepted = np.zeros((repeats, T), dtype=np.int8)
    runtimes = np.zeros((repeats, T), dtype=np.float64)

    seed_offset = stable_int(cid) % 500_000_000
    base_seed = int(suite_cfg.get("base_seed", base_cfg["estimation"]["base_seed"])) + seed_offset

    for repeat in range(repeats):
        for ti in range(T):
            coefficient_seed = base_seed + 10000 * repeat + 10 * ti + 1
            noise_seed = base_seed + 10000 * repeat + 10 * ti + 2
            optimization_seed = base_seed + 10000 * repeat + 10 * ti + 3

            C = DYN.random_coefficients(K, int(est_cfg["snapshots"]), coefficient_seed)
            clean_Y = oracle_basis[ti] @ C
            noisy_Y, _ = DYN.add_complex_noise(clean_Y, snr_db, noise_seed)
            metadata = DYN.DynamicMetadata(
                K=K,
                frequency=frequency,
                array_geometry=receiver_depths.copy(),
                water_depth=water_depth,
                SSP=ssp_matrix[ti].copy(),
            )
            start = time.perf_counter()
            estimate, _, gate_accepted, _, _ = DYN.dynamic_topn_refined(
                noisy_Y, metadata, optimization_seed, est_cfg, dz
            )
            runtimes[repeat, ti] = time.perf_counter() - start
            raw_k[repeat, ti] = estimate
            accepted[repeat, ti] = int(gate_accepted)
            raw_basis[repeat, ti] = DYN.build_water_column_basis(
                estimate, omega, ssp_matrix[ti], dz, receiver_depths, water_depth
            )

    tracked_k = np.zeros_like(raw_k)
    tracked_basis = np.zeros_like(raw_basis)
    for repeat in range(repeats):
        tracked_k[repeat] = TRK.alpha_beta_track(
            raw_k[repeat], times_s, suite_cfg["alpha_beta_tracker"]
        )
        for ti in range(T):
            tracked_basis[repeat, ti] = DYN.build_water_column_basis(
                tracked_k[repeat, ti],
                omega,
                ssp_matrix[ti],
                dz,
                receiver_depths,
                water_depth,
            )

    payload = {
        "raw_k": raw_k,
        "raw_basis": raw_basis,
        "tracked_k": tracked_k,
        "tracked_basis": tracked_basis,
        "accepted": accepted,
        "runtime_seconds": runtimes,
    }
    np.savez_compressed(cache, **payload)
    return payload


# -----------------------------------------------------------------------------
# Decomposition and summary metrics
# -----------------------------------------------------------------------------
def build_true_coefficients(
    K: int,
    fractions: np.ndarray,
    coupling_strength: float,
    base_coupling_cfg: dict,
) -> np.ndarray:
    base = DYN.initial_tracking_coefficients(K)
    result = np.zeros((fractions.size, K), dtype=np.complex128)
    cfg = dict(base_coupling_cfg)
    cfg["max_mixing_angle_rad"] = float(coupling_strength)
    for ti, frac in enumerate(fractions):
        if abs(float(coupling_strength)) <= 1.0e-15:
            result[ti] = base
        else:
            result[ti], _ = DYN.apply_known_coupling(base, float(frac), cfg)
    return result


def mean_basis_mac(oracle_series: np.ndarray, candidate_series: np.ndarray) -> float:
    T, _, K = oracle_series.shape
    values = []
    for ti in range(T):
        for m in range(K):
            values.append(DYN.column_mac(oracle_series[ti, :, m], candidate_series[ti, :, m]))
    return float(np.mean(values))


def evaluate_condition(
    base_cfg: dict,
    suite_cfg: dict,
    condition: dict,
    env: dict[str, np.ndarray],
    estimates: dict[str, np.ndarray],
) -> tuple[list[dict], list[dict]]:
    K = int(condition["K"])
    repeats = int(suite_cfg["estimator_repeats"])
    oracle_basis = np.asarray(env["oracle_basis"], dtype=np.complex128)
    fixed_basis = np.asarray(env["fixed_basis"], dtype=np.complex128)
    oracle_k = np.asarray(env["oracle_k"], dtype=np.float64)
    fractions = np.asarray(env["fractions"], dtype=np.float64)
    times_s = np.asarray(env["times_s"], dtype=np.float64)
    raw_k = np.asarray(estimates["raw_k"], dtype=np.float64)
    raw_basis = np.asarray(estimates["raw_basis"], dtype=np.complex128)
    tracked_k = np.asarray(estimates["tracked_k"], dtype=np.float64)
    tracked_basis = np.asarray(estimates["tracked_basis"], dtype=np.complex128)
    T = times_s.size
    tracking_snr = float(suite_cfg.get("decomposition_snr_db", base_cfg["tracking"]["snr_db"]))
    coupling_modes = [int(v) - 1 for v in base_cfg["coupling"]["mode_pair_1based"]]

    time_rows: list[dict] = []
    summary_rows: list[dict] = []

    for coupling_strength in [float(v) for v in suite_cfg["coupling_strengths_rad"]]:
        true_series = build_true_coefficients(K, fractions, coupling_strength, base_cfg["coupling"])
        coupling_label = "no_coupling" if abs(coupling_strength) <= 1e-15 else f"coupling_{coupling_strength:.3f}"

        for repeat in range(repeats):
            estimated_series = {
                "fixed": np.zeros((T, K), dtype=np.complex128),
                "instantaneous_dynamic": np.zeros((T, K), dtype=np.complex128),
                "alpha_beta_dynamic": np.zeros((T, K), dtype=np.complex128),
                "oracle_dynamic": np.zeros((T, K), dtype=np.complex128),
            }
            basis_series = {
                "fixed": np.repeat(fixed_basis[None, :, :], T, axis=0),
                "instantaneous_dynamic": raw_basis[repeat],
                "alpha_beta_dynamic": tracked_basis[repeat],
                "oracle_dynamic": oracle_basis,
            }

            for ti in range(T):
                seed = (
                    int(suite_cfg.get("tracking_base_seed", base_cfg["tracking"]["base_seed"]))
                    + stable_int(condition["condition_id"] + coupling_label) % 100_000_000
                    + 1000 * repeat
                    + ti
                )
                clean_pressure = oracle_basis[ti] @ true_series[ti]
                pressure, _ = DYN.add_complex_noise(clean_pressure, tracking_snr, seed)

                row = {
                    **condition,
                    "repeat": repeat,
                    "coupling_strength_rad": coupling_strength,
                    "coupling_label": coupling_label,
                    "time_index": ti,
                    "time_s": float(times_s[ti]),
                }
                for name, basis in basis_series.items():
                    coeff = DYN.decompose_coefficients(basis[ti], pressure)
                    estimated_series[name][ti] = coeff
                    leak, cmr = DYN.single_mode_leakage_metrics(basis[ti], oracle_basis[ti])
                    row[f"{name}_coeff_rel_error"] = DYN.coefficient_relative_error(true_series[ti], coeff)
                    row[f"{name}_leakage"] = leak
                    row[f"{name}_cmr_db"] = cmr
                time_rows.append(row)

            for name, series in estimated_series.items():
                coeff_errors = [
                    DYN.coefficient_relative_error(true_series[ti], series[ti])
                    for ti in range(T)
                ]
                basis_mac = mean_basis_mac(oracle_basis, basis_series[name])
                leak_values = []
                for ti in range(T):
                    leak, _ = DYN.single_mode_leakage_metrics(basis_series[name][ti], oracle_basis[ti])
                    leak_values.append(leak)
                mean_leak = float(np.mean(leak_values))
                cmr = 10.0 * math.log10(max(1.0 - mean_leak, 1e-30) / max(mean_leak, 1e-30))

                if abs(coupling_strength) <= 1e-15:
                    mean_vector = np.mean(series, axis=0, keepdims=True)
                    spurious = float(
                        np.linalg.norm(series - mean_vector)
                        / max(np.linalg.norm(np.tile(true_series[0], (T, 1))), 1e-15)
                    )
                    energy_corr = float("nan")
                else:
                    spurious = float("nan")
                    energy_corr = DYN.energy_correlation(true_series, series, coupling_modes)

                if name == "fixed" or name == "oracle_dynamic":
                    k_rmse = float("nan")
                elif name == "instantaneous_dynamic":
                    k_rmse = float(np.sqrt(np.mean((raw_k[repeat] - oracle_k) ** 2)))
                else:
                    k_rmse = float(np.sqrt(np.mean((tracked_k[repeat] - oracle_k) ** 2)))

                summary_rows.append(
                    {
                        **condition,
                        "repeat": repeat,
                        "coupling_strength_rad": coupling_strength,
                        "coupling_label": coupling_label,
                        "basis": name,
                        "mean_k_rmse_rad_m": k_rmse,
                        "mean_basis_mac": basis_mac,
                        "mean_coefficient_relative_error": float(np.mean(coeff_errors)),
                        "max_coefficient_relative_error": float(np.max(coeff_errors)),
                        "mean_offmode_energy_fraction": mean_leak,
                        "mean_cross_mode_rejection_db": cmr,
                        "spurious_temporal_variation": spurious,
                        "coupled_mode_energy_correlation": energy_corr,
                        "gate_acceptance_rate": float(np.mean(estimates["accepted"][repeat])),
                        "mean_estimator_runtime_seconds": float(np.mean(estimates["runtime_seconds"][repeat])),
                    }
                )

    return time_rows, summary_rows


def aggregate_repeats(rows: list[dict]) -> list[dict]:
    keys = sorted(
        {
            (
                r["condition_id"],
                float(r["coupling_strength_rad"]),
                r["basis"],
            )
            for r in rows
        }
    )
    output = []
    for cid, coupling, basis in keys:
        group = [
            r for r in rows
            if r["condition_id"] == cid
            and float(r["coupling_strength_rad"]) == coupling
            and r["basis"] == basis
        ]
        first = group[0]

        def avg(field: str) -> float:
            vals = np.asarray([float(r[field]) for r in group], dtype=np.float64)
            return float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else float("nan")

        output.append(
            {
                "condition_id": cid,
                "K": int(first["K"]),
                "snr_db": float(first["snr_db"]),
                "amplitude_m": float(first["amplitude_m"]),
                "rate_cycles_per_span": float(first["rate_cycles_per_span"]),
                "sweeps": first["sweeps"],
                "coupling_strength_rad": coupling,
                "coupling_label": first["coupling_label"],
                "basis": basis,
                "mean_k_rmse_rad_m": avg("mean_k_rmse_rad_m"),
                "mean_basis_mac": avg("mean_basis_mac"),
                "mean_coefficient_relative_error": avg("mean_coefficient_relative_error"),
                "max_coefficient_relative_error": float(np.nanmax([float(r["max_coefficient_relative_error"]) for r in group])),
                "mean_offmode_energy_fraction": avg("mean_offmode_energy_fraction"),
                "mean_cross_mode_rejection_db": avg("mean_cross_mode_rejection_db"),
                "mean_spurious_temporal_variation": avg("spurious_temporal_variation"),
                "mean_coupled_mode_energy_correlation": avg("coupled_mode_energy_correlation"),
                "gate_acceptance_rate": avg("gate_acceptance_rate"),
                "mean_estimator_runtime_seconds": avg("mean_estimator_runtime_seconds"),
            }
        )
    return output


def build_working_envelope(aggregate: list[dict], primary_coupling: float) -> list[dict]:
    rows = []
    condition_ids = sorted({r["condition_id"] for r in aggregate})
    for cid in condition_ids:
        for coupling in (0.0, float(primary_coupling)):
            subset = [
                r for r in aggregate
                if r["condition_id"] == cid
                and math.isclose(float(r["coupling_strength_rad"]), coupling, abs_tol=1e-12)
            ]
            by_basis = {r["basis"]: r for r in subset}
            if not {"fixed", "instantaneous_dynamic", "alpha_beta_dynamic", "oracle_dynamic"}.issubset(by_basis):
                continue
            f = by_basis["fixed"]
            i = by_basis["instantaneous_dynamic"]
            a = by_basis["alpha_beta_dynamic"]
            o = by_basis["oracle_dynamic"]

            coeff_vs_fixed = float(a["mean_coefficient_relative_error"]) / max(float(f["mean_coefficient_relative_error"]), 1e-15)
            coeff_vs_inst = float(a["mean_coefficient_relative_error"]) / max(float(i["mean_coefficient_relative_error"]), 1e-15)
            leak_vs_fixed = float(a["mean_offmode_energy_fraction"]) / max(float(f["mean_offmode_energy_fraction"]), 1e-15)

            if coupling == 0.0:
                spurious_vs_fixed = float(a["mean_spurious_temporal_variation"]) / max(float(f["mean_spurious_temporal_variation"]), 1e-15)
                corr_adv = float("nan")
                overall = coeff_vs_fixed < 1.0 and leak_vs_fixed < 1.0 and spurious_vs_fixed < 1.0
            else:
                spurious_vs_fixed = float("nan")
                corr_adv = float(a["mean_coupled_mode_energy_correlation"]) - float(f["mean_coupled_mode_energy_correlation"])
                overall = coeff_vs_fixed < 1.0 and leak_vs_fixed < 1.0 and corr_adv >= -0.005

            rows.append(
                {
                    "condition_id": cid,
                    "K": int(a["K"]),
                    "snr_db": float(a["snr_db"]),
                    "amplitude_m": float(a["amplitude_m"]),
                    "rate_cycles_per_span": float(a["rate_cycles_per_span"]),
                    "sweeps": a["sweeps"],
                    "coupling_strength_rad": coupling,
                    "tracked_coeff_error_ratio_vs_fixed": coeff_vs_fixed,
                    "tracked_coeff_error_ratio_vs_instantaneous": coeff_vs_inst,
                    "tracked_leakage_ratio_vs_fixed": leak_vs_fixed,
                    "tracked_spurious_ratio_vs_fixed": spurious_vs_fixed,
                    "tracked_coupling_corr_advantage_vs_fixed": corr_adv,
                    "tracked_beats_fixed_overall": int(overall),
                    "oracle_coeff_error": float(o["mean_coefficient_relative_error"]),
                }
            )
    return rows


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
def select_rows(aggregate: list[dict], *, K: int, coupling: float, sweep_tag: str) -> list[dict]:
    return [
        r for r in aggregate
        if int(r["K"]) == int(K)
        and math.isclose(float(r["coupling_strength_rad"]), float(coupling), abs_tol=1e-12)
        and sweep_tag in r["sweeps"]
    ]


def plot_metric_sweep(
    path: Path,
    rows: list[dict],
    x_field: str,
    ylabel: str,
    metric: str,
) -> None:
    if not rows:
        return
    bases = ["fixed", "instantaneous_dynamic", "alpha_beta_dynamic", "oracle_dynamic"]
    labels = {
        "fixed": "Fixed",
        "instantaneous_dynamic": "Instantaneous",
        "alpha_beta_dynamic": "Alpha-beta",
        "oracle_dynamic": "Oracle dynamic",
    }
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    styles = ["-", "--", "-.", ":"]
    for idx, basis in enumerate(bases):
        part = [r for r in rows if r["basis"] == basis]
        part = sorted(part, key=lambda r: float(r[x_field]))
        if not part:
            continue
        ax.plot(
            [float(r[x_field]) for r in part],
            [float(r[metric]) for r in part],
            linestyle=styles[idx],
            marker="o",
            markersize=4,
            label=labels[basis],
        )
    ax.set_xlabel(x_field.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_coupling_correlation(path: Path, aggregate: list[dict], baseline: dict) -> None:
    rows = [
        r for r in aggregate
        if int(r["K"]) == 3
        and math.isclose(float(r["snr_db"]), float(baseline["snr_db"]), abs_tol=1e-12)
        and math.isclose(float(r["amplitude_m"]), float(baseline["amplitude_m"]), abs_tol=1e-12)
        and math.isclose(float(r["rate_cycles_per_span"]), float(baseline["rate_cycles_per_span"]), abs_tol=1e-12)
        and float(r["coupling_strength_rad"]) > 0.0
    ]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for basis, label, style in [
        ("fixed", "Fixed", "-"),
        ("instantaneous_dynamic", "Instantaneous", "--"),
        ("alpha_beta_dynamic", "Alpha-beta", "-."),
        ("oracle_dynamic", "Oracle dynamic", ":"),
    ]:
        part = sorted([r for r in rows if r["basis"] == basis], key=lambda r: float(r["coupling_strength_rad"]))
        ax.plot(
            [float(r["coupling_strength_rad"]) for r in part],
            [float(r["mean_coupled_mode_energy_correlation"]) for r in part],
            linestyle=style,
            marker="o",
            label=label,
        )
    ax.set_xlabel("Maximum coupling angle (rad)")
    ax.set_ylabel("Coupled-mode energy correlation")
    ax.set_ylim(0.0, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Report generation
# -----------------------------------------------------------------------------
def summarize_primary_envelope(envelope: list[dict], cfg: dict) -> list[str]:
    primary = float(cfg["primary_coupling_strength_rad"])
    design = cfg["design"]
    b3 = design["K3_baseline"]

    def matches(r: dict, tag: str, coupling: float = primary) -> bool:
        return int(r["K"]) == 3 and tag in r["sweeps"] and math.isclose(float(r["coupling_strength_rad"]), coupling, abs_tol=1e-12)

    lines = []
    snr_rows = sorted([r for r in envelope if matches(r, "K3_snr")], key=lambda r: float(r["snr_db"]))
    winners = [float(r["snr_db"]) for r in snr_rows if int(r["tracked_beats_fixed_overall"]) == 1]
    if winners:
        lines.append(f"- Lowest tested K=3 SNR where tracked dynamic beats fixed at primary coupling: {min(winners):.1f} dB.")
    else:
        lines.append("- Tracked dynamic did not beat fixed at any tested K=3 SNR under the primary coupling condition.")

    amp_rows = sorted([r for r in envelope if matches(r, "K3_amplitude")], key=lambda r: float(r["amplitude_m"]))
    amp_win = [float(r["amplitude_m"]) for r in amp_rows if int(r["tracked_beats_fixed_overall"]) == 1]
    if amp_win:
        lines.append(f"- Largest tested thermocline displacement amplitude with a tracked-basis win: {max(amp_win):.2f} m.")

    rate_rows = sorted([r for r in envelope if matches(r, "K3_rate")], key=lambda r: float(r["rate_cycles_per_span"]))
    rate_win = [float(r["rate_cycles_per_span"]) for r in rate_rows if int(r["tracked_beats_fixed_overall"]) == 1]
    if rate_win:
        lines.append(f"- Fastest tested environment-rate factor with a tracked-basis win: {max(rate_win):.2f} cycles/span.")

    # Boundary regime K=5 at baseline-like conditions.
    b5 = design["K5_baseline"]
    k5 = [
        r for r in envelope
        if int(r["K"]) == 5
        and math.isclose(float(r["coupling_strength_rad"]), primary, abs_tol=1e-12)
        and math.isclose(float(r["amplitude_m"]), float(b5["amplitude_m"]), abs_tol=1e-12)
        and math.isclose(float(r["rate_cycles_per_span"]), float(b5["rate_cycles_per_span"]), abs_tol=1e-12)
    ]
    k5_wins = [r for r in k5 if int(r["tracked_beats_fixed_overall"]) == 1]
    if k5_wins:
        lines.append("- K=5 achieves a tracked-basis win in at least one tested baseline-environment SNR condition.")
    else:
        lines.append("- K=5 remains a boundary regime: no overall tracked-basis win was found in the tested baseline-environment SNR sweep.")
    return lines


def main() -> None:
    suite_cfg = load_config()
    base_cfg_path = ROOT / suite_cfg["base_dynamic_config"]
    base_cfg = json.loads(base_cfg_path.read_text(encoding="utf-8"))
    dirs = ensure_dirs(ROOT / suite_cfg["output_root"])

    conditions = build_conditions(suite_cfg)
    write_csv(dirs["csv"] / "condition_manifest.csv", conditions)

    print("=" * 104)
    print("FINAL UNIFIED DYNAMIC-BASIS ROBUSTNESS SUITE")
    print("=" * 104)
    print(f"conditions        : {len(conditions)}")
    print(f"time points       : {int(suite_cfg['time_points'])}")
    print(f"estimator repeats : {int(suite_cfg['estimator_repeats'])}")
    print(f"resume enabled    : {bool(suite_cfg.get('resume', True))}")
    print("bottom parameters : controlled forward/oracle generation ONLY")
    print("main practical basis: causal alpha-beta dynamic")
    print()

    all_time_rows: list[dict] = []
    all_repeat_rows: list[dict] = []

    for index, condition in enumerate(conditions, start=1):
        print("-" * 104)
        print(
            f"[{index:02d}/{len(conditions):02d}] {condition['condition_id']} | "
            f"sweeps={condition['sweeps']}"
        )
        env = generate_or_load_environment(
            dirs,
            base_cfg,
            suite_cfg,
            int(condition["K"]),
            float(condition["amplitude_m"]),
            float(condition["rate_cycles_per_span"]),
        )
        estimates = run_or_load_estimation_condition(dirs, base_cfg, suite_cfg, condition, env)
        time_rows, repeat_rows = evaluate_condition(base_cfg, suite_cfg, condition, env, estimates)
        all_time_rows.extend(time_rows)
        all_repeat_rows.extend(repeat_rows)

        raw_rmse = float(np.sqrt(np.mean((estimates["raw_k"] - env["oracle_k"][None, :, :]) ** 2)))
        tracked_rmse = float(np.sqrt(np.mean((estimates["tracked_k"] - env["oracle_k"][None, :, :]) ** 2)))
        print(f"  kRMSE instantaneous={raw_rmse:.4e} | alpha-beta={tracked_rmse:.4e}")

    aggregate = aggregate_repeats(all_repeat_rows)
    envelope = build_working_envelope(aggregate, float(suite_cfg["primary_coupling_strength_rad"]))

    write_csv(dirs["csv"] / "robustness_time_metrics.csv", all_time_rows)
    write_csv(dirs["csv"] / "robustness_summary_by_repeat.csv", all_repeat_rows)
    write_csv(dirs["csv"] / "robustness_summary.csv", aggregate)
    write_csv(dirs["csv"] / "working_envelope.csv", envelope)

    primary = float(suite_cfg["primary_coupling_strength_rad"])
    plot_metric_sweep(
        dirs["figures"] / "01_K3_snr_coefficient_error.png",
        select_rows(aggregate, K=3, coupling=primary, sweep_tag="K3_snr"),
        "snr_db",
        "Mean coefficient relative error",
        "mean_coefficient_relative_error",
    )
    plot_metric_sweep(
        dirs["figures"] / "02_K3_amplitude_coefficient_error.png",
        select_rows(aggregate, K=3, coupling=primary, sweep_tag="K3_amplitude"),
        "amplitude_m",
        "Mean coefficient relative error",
        "mean_coefficient_relative_error",
    )
    plot_metric_sweep(
        dirs["figures"] / "03_K3_rate_coefficient_error.png",
        select_rows(aggregate, K=3, coupling=primary, sweep_tag="K3_rate"),
        "rate_cycles_per_span",
        "Mean coefficient relative error",
        "mean_coefficient_relative_error",
    )
    plot_metric_sweep(
        dirs["figures"] / "04_K3_snr_leakage.png",
        select_rows(aggregate, K=3, coupling=primary, sweep_tag="K3_snr"),
        "snr_db",
        "Mean off-mode energy fraction",
        "mean_offmode_energy_fraction",
    )
    plot_coupling_correlation(
        dirs["figures"] / "05_K3_coupling_strength_energy_correlation.png",
        aggregate,
        suite_cfg["design"]["K3_baseline"],
    )
    plot_metric_sweep(
        dirs["figures"] / "06_K5_snr_coefficient_error.png",
        select_rows(aggregate, K=5, coupling=primary, sweep_tag="K5_snr"),
        "snr_db",
        "Mean coefficient relative error",
        "mean_coefficient_relative_error",
    )

    report_lines = [
        "# Final Dynamic-Basis Robustness Report",
        "",
        "This suite is a controlled synthetic robustness study. It does not contain SWARM-95 measured-data results.",
        "",
        "## Design",
        "",
        f"- Conditions: {len(conditions)}",
        f"- Time points per condition: {int(suite_cfg['time_points'])}",
        f"- Estimator repeats: {int(suite_cfg['estimator_repeats'])}",
        f"- Primary coupling strength: {primary:.3f} rad",
        "- Compared bases: fixed, instantaneous dynamic, causal alpha-beta dynamic, oracle dynamic.",
        "- Decomposition noise is held at high SNR to isolate basis/tracking effects.",
        "- Bottom parameters are used only for controlled oracle/forward truth.",
        "",
        "## Working-envelope summary",
        "",
    ]
    report_lines.extend(summarize_primary_envelope(envelope, suite_cfg))
    report_lines += [
        "",
        "## Interpretation rule",
        "",
        "A practical tracked-dynamic win requires lower coefficient error and lower leakage than the fixed basis; for coupled cases the energy-correlation loss must also be negligible.",
        "",
        "If K=3 wins over a coherent SNR/amplitude/rate region while K=5 remains mixed, the scientifically defensible claim is an operating-envelope result rather than universal superiority.",
        "",
        "SWARM-95 measured-data validation remains pending.",
    ]
    (dirs["root"] / "FINAL_ROBUSTNESS_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

    summary_json = {
        "experiment": suite_cfg.get("experiment", "final_dynamic_robustness_suite"),
        "condition_count": len(conditions),
        "time_points": int(suite_cfg["time_points"]),
        "estimator_repeats": int(suite_cfg["estimator_repeats"]),
        "primary_coupling_strength_rad": primary,
        "working_envelope_summary": summarize_primary_envelope(envelope, suite_cfg),
        "scientific_boundary": {
            "measured_data": False,
            "bottom_parameters_in_estimator": False,
            "future_samples_in_alpha_beta_tracker": False,
        },
    }
    (dirs["root"] / "summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print()
    print("=" * 104)
    print("WORKING-ENVELOPE SUMMARY")
    print("=" * 104)
    for line in summarize_primary_envelope(envelope, suite_cfg):
        print(line)
    print("-" * 104)
    print(f"outputs = {dirs['root']}")
    print("SWARM-95 measured-data validation remains pending.")
    print("=" * 104)


if __name__ == "__main__":
    main()
