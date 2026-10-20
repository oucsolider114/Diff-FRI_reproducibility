# -*- coding: utf-8 -*-
"""
CSMT-v2 + realistic SSP measurement-chain end-to-end validation v1.

Goal
----
Unify the final frozen CSMT-v2 algorithm with the realistic SSP acquisition
chain developed on WOA23 profiles.

Truth world:
    true WOA-based time-varying SSP
        -> oracle modal wavenumbers / oracle local modes
        -> acoustic observation Y(t) and modal pressure p(t)

Algorithm world:
    measurement chain(true SSP)
        -> believed SSP_tilde(t)
    Y(t), SSP_tilde(t)
        -> frozen CSMT-v2
        -> k_hat(t)
    k_hat(t), SSP_tilde(t)
        -> estimated dynamic local-mode basis Phi_hat(t)
    Phi_hat(t)^dagger p(t)
        -> modal coefficients a_hat(t)

Scientific restrictions
-----------------------
- CSMT-v2 parameters are loaded from its completed independent-holdout lock.
- The source holdout must say PROMOTE_CSMT_V2.
- No CSMT parameter retuning is allowed.
- True SSP, oracle k and bottom parameters are used only to create controlled
  truth and evaluation targets.
- Every SSP-chain condition uses the SAME acoustic Y and downstream pressure
  realization within each paired trial.
- This remains a controlled data-informed experiment, NOT measured acoustic
  validation.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from mode_fri.causal_ssp_modal_tracking import (
    CSMTConfig,
    CausalSSPModalTracking,
)

import ssp_environment as ENV
import ssp_measurement as MEAS


CONFIG_PATH = ROOT / "configs" / "csmt_ssp_measurement_chain_v1.json"
CSMT_HOLDOUT_ROOT = ROOT / "outputs" / "csmt_v2_independent_holdout"


# ---------------------------------------------------------------------------
# Dynamic-suite helpers already present in the main NN_FRI project
# ---------------------------------------------------------------------------
def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


DYN = load_module(
    "joint_dynamic_helpers",
    ROOT / "experiments" / "dynamic_local_basis_coupling_suite.py",
)
CSMT_HOLD = load_module(
    "joint_csmt_holdout_helpers",
    ROOT / "experiments" / "csmt_v2_independent_holdout.py",
)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_json(value) -> str:
    blob = json.dumps(
        value, ensure_ascii=False, sort_keys=True,
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


def ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "root": root,
        "cache": root / "cache",
        "csv": root / "csv",
        "figures": root / "figures",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields, extrasaction="raise", restval=""
        )
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    samples: int,
    ci_level: float,
) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(samples), dtype=np.float64)
    for i in range(int(samples)):
        idx = rng.integers(0, x.size, size=x.size)
        boot[i] = float(np.mean(x[idx]))
    alpha = 0.5 * (1.0 - float(ci_level))
    return (
        float(np.quantile(boot, alpha)),
        float(np.quantile(boot, 1.0 - alpha)),
    )


# ---------------------------------------------------------------------------
# Frozen CSMT-v2
# ---------------------------------------------------------------------------
def load_frozen_csmt() -> tuple[dict, dict, CSMTConfig]:
    lock = load_json(CSMT_HOLDOUT_ROOT / "CSMT_V2_PARAMETER_LOCK.json")
    result = load_json(CSMT_HOLDOUT_ROOT / "HOLDOUT_RESULT.json")
    if str(result["decision"]) != "PROMOTE_CSMT_V2":
        raise RuntimeError(
            "Joint validation requires completed PROMOTE_CSMT_V2."
        )
    if str(result["csmt_v2_parameter_hash"]) != str(
        lock["parameter_hash"]
    ):
        raise RuntimeError("CSMT-v2 result/lock hash mismatch.")

    c = lock["locked_payload"]["csmt_v2"]
    cfg = CSMTConfig(
        beta_pred=float(c["beta_pred"]),
        delta0_rad_m=float(c["delta0_rad_m"]),
        delta_min_rad_m=float(c["delta_min_rad_m"]),
        delta_max_rad_m=float(c["delta_max_rad_m"]),
        lambda_k=float(c["lambda_k"]),
        lambda_ssp=float(c["lambda_ssp"]),
        minimum_mode_separation_rad_m=float(
            c["minimum_mode_separation_rad_m"]
        ),
        local_grid_points=int(c["local_grid_points"]),
        local_coordinate_sweeps=int(c["local_coordinate_sweeps"]),
        enable_quadratic_subgrid=bool(c["enable_quadratic_subgrid"]),
    )
    return lock, result, cfg


# ---------------------------------------------------------------------------
# Protocol / measurement-chain conditions
# ---------------------------------------------------------------------------
def chain_depths(entry: dict) -> np.ndarray:
    presets = MEAS.chain_depth_presets()
    preset = str(entry["preset"])
    if preset not in presets:
        raise KeyError(f"Unknown SSP-chain preset: {preset}")
    return np.asarray(presets[preset], dtype=np.float64)


def chain_spec(entry: dict) -> MEAS.ChainSpec:
    merged = dict(entry)
    merged["sensor_depths_m"] = chain_depths(entry)
    return MEAS.chain_from_dict(merged)


def condition_entries(cfg: dict) -> list[dict]:
    out = []
    for entry in cfg["conditions"]:
        e = dict(entry)
        if e["name"] == "exact_ssp":
            e["kind"] = "exact"
        else:
            e["kind"] = "measurement_chain"
            e["_spec"] = chain_spec(e)
        out.append(e)
    return out


def lock_protocol(
    cfg: dict,
    dirs: dict[str, Path],
    csmt_lock: dict,
) -> str:
    serializable_conditions = []
    for e in cfg["conditions"]:
        serializable_conditions.append(dict(e))
    payload = {
        "experiment": cfg["experiment"],
        "frozen_csmt_v2_hash": csmt_lock["parameter_hash"],
        "references": cfg["references"],
        "time_variation": cfg["time_variation"],
        "trials_per_reference": cfg["trials_per_reference"],
        "estimation": cfg["estimation"],
        "decomposition": cfg["decomposition"],
        "conditions": serializable_conditions,
        "decision_rule": cfg["decision_rule"],
        "statistics": cfg["statistics"],
        "scientific_rule": (
            "Frozen CSMT-v2 + WOA23-based truth + realistic SSP acquisition; "
            "only SSP input quality changes across conditions."
        ),
    }
    h = sha256_json(payload)
    path = dirs["root"] / "JOINT_PROTOCOL_LOCK.json"
    if path.exists():
        old = load_json(path)
        if str(old["protocol_hash"]) != h:
            raise RuntimeError(
                "Joint protocol changed after start. Create v2 instead."
            )
    else:
        path.write_text(
            json.dumps(
                {"protocol_hash": h, "locked_payload": payload},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        (dirs["root"] / "JOINT_STARTED.marker").write_text(
            f"protocol_hash={h}\n",
            encoding="utf-8",
        )
    return h


# ---------------------------------------------------------------------------
# Environment truth
# ---------------------------------------------------------------------------
def reference_from_entry(entry: dict) -> ENV.ReferenceProfile:
    return ENV.reference_from_spec(entry)


def build_environment(
    reference: ENV.ReferenceProfile,
    cfg: dict,
    base_cfg: dict,
) -> dict:
    dz = float(base_cfg["dz_m"])
    depth = float(base_cfg["water_depth_m"])
    z = np.arange(0.0, depth + 0.5 * dz, dz)
    r_cfg = base_cfg["receiver_array"]
    receivers = np.arange(
        float(r_cfg["start_depth_m"]),
        float(r_cfg["end_depth_m"]) + 0.5 * float(r_cfg["spacing_m"]),
        float(r_cfg["spacing_m"]),
    )
    times = np.linspace(
        float(cfg["time_variation"]["start_s"]),
        float(cfg["time_variation"]["end_s"]),
        int(cfg["time_variation"]["time_points"]),
    )
    model = ENV.EnvironmentModel(
        reference=reference,
        z=z,
        amplitude_m=float(cfg["time_variation"]["amplitude_m"]),
        waveform=str(cfg["time_variation"]["waveform"]),
        span_s=float(cfg["time_variation"]["span_s"]),
        taper_sigma_m=float(cfg["time_variation"]["taper_sigma_m"]),
        fundamental_phase_rad=float(
            cfg["time_variation"]["fundamental_phase_rad"]
        ),
    )

    K = int(cfg["estimation"]["K"])
    omega = 2.0 * math.pi * float(base_cfg["frequency_hz"])
    ssp = np.zeros((times.size, z.size), dtype=np.float64)
    oracle_k = np.zeros((times.size, K), dtype=np.float64)
    oracle_basis = np.zeros(
        (times.size, receivers.size, K),
        dtype=np.complex128,
    )

    for ti, t in enumerate(times):
        ssp[ti] = MEAS.truth_at(model, float(t))
        oracle_k[ti] = DYN.find_oracle_roots(
            K=K,
            omega=omega,
            ssp=ssp[ti],
            dz=dz,
            bottom=base_cfg["forward_bottom"],
            scan_points=int(cfg["estimation"]["oracle_root_scan_points"]),
        )
        oracle_basis[ti] = DYN.build_water_column_basis(
            oracle_k[ti],
            omega,
            ssp[ti],
            dz,
            receivers,
            depth,
        )

    return {
        "reference": reference,
        "model": model,
        "z": z,
        "receiver_depths": receivers,
        "times": times,
        "fractions": (
            (times - times[0])
            / max(float(times[-1] - times[0]), 1.0)
        ),
        "ssp": ssp,
        "oracle_k": oracle_k,
        "oracle_basis": oracle_basis,
    }


def true_coefficients(
    env: dict,
    base_cfg: dict,
    strength_rad: float,
) -> np.ndarray:
    K = int(base_cfg.get("K_values", [3])[0])
    K = 3  # joint protocol is explicitly K=3
    base = DYN.initial_tracking_coefficients(K)
    coupling_cfg = dict(base_cfg["coupling"])
    coupling_cfg["max_mixing_angle_rad"] = float(strength_rad)
    out = np.zeros((env["times"].size, K), dtype=np.complex128)
    for ti, frac in enumerate(env["fractions"]):
        if abs(float(strength_rad)) <= 1.0e-15:
            out[ti] = base
        else:
            out[ti], _ = DYN.apply_known_coupling(
                base, float(frac), coupling_cfg
            )
    return out


def coupling_fraction(coefficients: np.ndarray) -> np.ndarray:
    c = np.asarray(coefficients, dtype=np.complex128)
    e1 = np.abs(c[:, 0]) ** 2
    e2 = np.abs(c[:, 1]) ** 2
    return e2 / np.maximum(e1 + e2, 1.0e-15)


# ---------------------------------------------------------------------------
# Paired acoustic data
# ---------------------------------------------------------------------------
def make_observations(
    *,
    env: dict,
    cfg: dict,
    reference_name: str,
    trial: int,
) -> tuple[list[np.ndarray], list[str]]:
    observations = []
    hashes = []
    ns = str(cfg["seed_namespace"])
    snapshots = int(cfg["estimation"]["snapshots"])
    snr_db = float(cfg["estimation"]["snr_db"])
    for ti in range(env["times"].size):
        Y = CSMT_HOLD.make_observation(
            oracle_basis=env["oracle_basis"][ti],
            snapshots=snapshots,
            snr_db=snr_db,
            coefficient_seed=seed32(
                ns, reference_name, trial, ti, "Y_coeff"
            ),
            noise_seed=seed32(
                ns, reference_name, trial, ti, "Y_noise"
            ),
        )
        observations.append(Y)
        hashes.append(sha256_array(Y))
    return observations, hashes


def make_pressure_series(
    *,
    env: dict,
    cfg: dict,
    base_cfg: dict,
    reference_name: str,
    trial: int,
    scenario: str,
) -> tuple[np.ndarray, np.ndarray]:
    if scenario == "no_coupling":
        strength = 0.0
    elif scenario == "known_coupling":
        strength = float(cfg["decomposition"]["coupling_strength_rad"])
    else:
        raise ValueError(scenario)

    truth = true_coefficients(env, base_cfg, strength)
    pressure = np.zeros(
        (env["times"].size, env["receiver_depths"].size),
        dtype=np.complex128,
    )
    ns = str(cfg["seed_namespace"])
    for ti in range(env["times"].size):
        clean = env["oracle_basis"][ti] @ truth[ti]
        pressure[ti], _ = DYN.add_complex_noise(
            clean,
            float(cfg["decomposition"]["snr_db"]),
            seed32(
                ns, reference_name, trial, scenario, ti, "pressure"
            ),
        )
    return truth, pressure


# ---------------------------------------------------------------------------
# SSP beliefs
# ---------------------------------------------------------------------------
def believed_ssp_series(
    *,
    env: dict,
    cfg: dict,
    condition: dict,
    reference_name: str,
    trial: int,
) -> tuple[np.ndarray, list[dict]]:
    if condition["kind"] == "exact":
        details = [
            {
                "profile_age_s": 0.0,
                "delta_rms_m_s": 0.0,
                "delta_max_m_s": 0.0,
                "sensor_count": env["z"].size,
                "formula_used": "truth_teos10",
            }
            for _ in env["times"]
        ]
        return env["ssp"].copy(), details

    spec = condition["_spec"]
    out = np.zeros_like(env["ssp"])
    details = []
    ns = str(cfg["seed_namespace"])
    for ti, t in enumerate(env["times"]):
        rng = np.random.default_rng(
            seed32(
                ns,
                reference_name,
                condition["name"],
                trial,
                ti,
                "ssp_measurement",
            )
        )
        out[ti], detail = MEAS.belief_at(
            env["model"],
            spec,
            float(t),
            env["z"],
            rng,
        )
        details.append(detail)
    return out, details


# ---------------------------------------------------------------------------
# Frozen CSMT-v2 execution
# ---------------------------------------------------------------------------
def run_csmt(
    *,
    observations: list[np.ndarray],
    believed_ssp: np.ndarray,
    env: dict,
    base_cfg: dict,
    csmt_cfg: CSMTConfig,
    cfg: dict,
    reference_name: str,
    condition_name: str,
    trial: int,
) -> dict:
    T = env["times"].size
    K = int(cfg["estimation"]["K"])
    k_hat = np.zeros((T, K), dtype=np.float64)
    runtime = np.zeros(T, dtype=np.float64)
    states = []
    basis = np.zeros_like(env["oracle_basis"])

    # One Full Proposed initialization, using only the believed SSP.
    init_seed = seed32(
        cfg["seed_namespace"],
        reference_name,
        condition_name,
        trial,
        "full_init",
    )
    k0, full_runtime = CSMT_HOLD.run_full(
        Y=observations[0],
        SSP=believed_ssp[0],
        receiver_depths=env["receiver_depths"],
        dynamic_cfg=base_cfg,
        optimizer_seed=init_seed,
    )

    tracker = CausalSSPModalTracking(csmt_cfg)
    init = tracker.initialize_from_full_estimate(
        k_hat=k0,
        SSP_t=believed_ssp[0],
        Y_t=observations[0],
        frequency_hz=float(base_cfg["frequency_hz"]),
        dz=float(base_cfg["dz_m"]),
        receiver_depths=env["receiver_depths"],
        water_depth=float(base_cfg["water_depth_m"]),
        full_runtime_seconds=float(full_runtime),
    )
    k_hat[0] = init.k_hat
    runtime[0] = init.total_runtime_seconds
    states.append(init.state)

    omega = 2.0 * math.pi * float(base_cfg["frequency_hz"])
    basis[0] = DYN.build_water_column_basis(
        k_hat[0],
        omega,
        believed_ssp[0],
        float(base_cfg["dz_m"]),
        env["receiver_depths"],
        float(base_cfg["water_depth_m"]),
    )

    for ti in range(1, T):
        result = tracker.step(
            Y_t=observations[ti],
            SSP_t=believed_ssp[ti],
            frequency_hz=float(base_cfg["frequency_hz"]),
            dz=float(base_cfg["dz_m"]),
            receiver_depths=env["receiver_depths"],
            water_depth=float(base_cfg["water_depth_m"]),
        )
        k_hat[ti] = result.k_hat
        runtime[ti] = result.total_runtime_seconds
        states.append(result.state)
        basis[ti] = DYN.build_water_column_basis(
            k_hat[ti],
            omega,
            believed_ssp[ti],
            float(base_cfg["dz_m"]),
            env["receiver_depths"],
            float(base_cfg["water_depth_m"]),
        )

    return {
        "k_hat": k_hat,
        "runtime": runtime,
        "states": np.asarray(states),
        "basis": basis,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def front_end_metrics(
    *,
    result: dict,
    env: dict,
    details: list[dict],
    cfg: dict,
) -> dict:
    frame_rmse = []
    success = []
    frame_mac = []
    threshold = float(cfg["estimation"]["success_threshold_rad_m"])

    for ti in range(env["times"].size):
        err = np.asarray(
            result["k_hat"][ti] - env["oracle_k"][ti],
            dtype=np.float64,
        )
        frame_rmse.append(float(np.sqrt(np.mean(err ** 2))))
        success.append(
            float(np.max(np.abs(err))) <= threshold
        )
        macs = [
            float(
                DYN.column_mac(
                    env["oracle_basis"][ti, :, m],
                    result["basis"][ti, :, m],
                )
            )
            for m in range(result["basis"].shape[2])
        ]
        frame_mac.append(float(np.mean(macs)))

    return {
        "mean_k_rmse": float(np.mean(frame_rmse)),
        "frame_success_rate": float(np.mean(success)),
        "mean_basis_mac": float(np.mean(frame_mac)),
        "mean_runtime_s": float(np.mean(result["runtime"])),
        "mean_profile_delta_rms_m_s": float(
            np.mean([d["delta_rms_m_s"] for d in details])
        ),
        "worst_profile_delta_rms_m_s": float(
            np.max([d["delta_rms_m_s"] for d in details])
        ),
        "mean_profile_age_s": float(
            np.mean([d["profile_age_s"] for d in details])
        ),
        "worst_profile_age_s": float(
            np.max([d["profile_age_s"] for d in details])
        ),
    }


def downstream_metrics(
    *,
    result: dict,
    truth_coefficients: np.ndarray,
    pressure: np.ndarray,
    env: dict,
    base_cfg: dict,
    event_threshold: float,
    scenario: str,
) -> dict:
    estimate = np.zeros_like(truth_coefficients)
    errors = []
    leakages = []

    for ti in range(env["times"].size):
        estimate[ti] = DYN.decompose_coefficients(
            result["basis"][ti],
            pressure[ti],
        )
        errors.append(
            float(
                DYN.coefficient_relative_error(
                    truth_coefficients[ti],
                    estimate[ti],
                )
            )
        )
        leakage, _ = DYN.single_mode_leakage_metrics(
            result["basis"][ti],
            env["oracle_basis"][ti],
        )
        leakages.append(float(leakage))

    truth_fraction = coupling_fraction(truth_coefficients)
    est_fraction = coupling_fraction(estimate)

    if scenario == "no_coupling":
        base_fraction = float(truth_fraction[0])
        deviation = np.abs(est_fraction - base_fraction)
        false_alarm_rate = float(
            np.mean(deviation > float(event_threshold))
        )
        false_coupling_peak = float(np.max(deviation))
        detection_rate = float("nan")
        trajectory_rmse = float(
            np.sqrt(np.mean((est_fraction - truth_fraction) ** 2))
        )
        corr = float("nan")
        mean_vector = np.mean(estimate, axis=0, keepdims=True)
        spurious = float(
            np.linalg.norm(estimate - mean_vector)
            / max(np.linalg.norm(truth_coefficients), 1.0e-15)
        )
    else:
        no_event_baseline = float(coupling_fraction(
            true_coefficients(env, base_cfg, 0.0)
        )[0])
        true_dev = np.abs(truth_fraction - no_event_baseline)
        est_dev = np.abs(est_fraction - no_event_baseline)
        active = true_dev > float(event_threshold)
        if np.any(active):
            detection_rate = float(
                np.mean(est_dev[active] > float(event_threshold))
            )
        else:
            detection_rate = float("nan")
        false_alarm_rate = float("nan")
        false_coupling_peak = float("nan")
        trajectory_rmse = float(
            np.sqrt(np.mean((est_fraction - truth_fraction) ** 2))
        )
        coupling_modes = [
            int(v) - 1
            for v in base_cfg["coupling"]["mode_pair_1based"]
        ]
        corr = float(
            DYN.energy_correlation(
                truth_coefficients,
                estimate,
                coupling_modes,
            )
        )
        spurious = float("nan")

    return {
        "mean_coefficient_relative_error": float(np.mean(errors)),
        "max_coefficient_relative_error": float(np.max(errors)),
        "mean_offmode_energy_fraction": float(np.mean(leakages)),
        "spurious_temporal_variation": spurious,
        "coupling_energy_correlation": corr,
        "coupling_fraction_trajectory_rmse": trajectory_rmse,
        "false_alarm_rate": false_alarm_rate,
        "false_coupling_peak": false_coupling_peak,
        "detection_rate": detection_rate,
    }


# ---------------------------------------------------------------------------
# Trial cache
# ---------------------------------------------------------------------------
def cache_path(
    dirs: dict[str, Path],
    reference: str,
    condition: str,
    trial: int,
) -> Path:
    folder = dirs["cache"] / reference / condition
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"trial_{trial:03d}.npz"


def run_or_load(
    *,
    dirs: dict[str, Path],
    protocol_hash: str,
    reference_name: str,
    condition: dict,
    trial: int,
    observations: list[np.ndarray],
    y_hashes: list[str],
    env: dict,
    base_cfg: dict,
    csmt_cfg: CSMTConfig,
    cfg: dict,
) -> tuple[dict, list[dict]]:
    path = cache_path(
        dirs, reference_name, condition["name"], trial
    )
    if path.exists() and bool(cfg["resume"]):
        d = np.load(path, allow_pickle=False)
        if str(d["protocol_hash"].item()) != protocol_hash:
            raise RuntimeError(f"Cache protocol mismatch: {path}")
        result = {
            "k_hat": np.asarray(d["k_hat"]),
            "runtime": np.asarray(d["runtime"]),
            "states": np.asarray(d["states"]),
            "basis": np.asarray(d["basis"]),
        }
        details = json.loads(str(d["details_json"].item()))
        # Same paired Y must be regenerated identically.
        cached_hashes = [str(v) for v in d["y_hashes"]]
        if cached_hashes != list(y_hashes):
            raise RuntimeError(
                f"Y SHA256 mismatch in resumed cache: {path}"
            )
        return result, details

    believed, details = believed_ssp_series(
        env=env,
        cfg=cfg,
        condition=condition,
        reference_name=reference_name,
        trial=trial,
    )
    result = run_csmt(
        observations=observations,
        believed_ssp=believed,
        env=env,
        base_cfg=base_cfg,
        csmt_cfg=csmt_cfg,
        cfg=cfg,
        reference_name=reference_name,
        condition_name=condition["name"],
        trial=trial,
    )

    np.savez_compressed(
        path,
        protocol_hash=np.asarray(protocol_hash),
        k_hat=result["k_hat"],
        runtime=result["runtime"],
        states=result["states"],
        basis=result["basis"],
        believed_ssp=believed,
        details_json=np.asarray(
            json.dumps(details, ensure_ascii=False)
        ),
        y_hashes=np.asarray(y_hashes),
    )
    return result, details


# ---------------------------------------------------------------------------
# Aggregation / decisions
# ---------------------------------------------------------------------------
def aggregate_rows(
    trial_rows: list[dict],
    downstream_rows: list[dict],
    cfg: dict,
) -> list[dict]:
    out = []
    refs = sorted({r["reference"] for r in trial_rows})
    conds = [c["name"] for c in cfg["conditions"]]

    for reference in refs:
        exact_front = [
            r for r in trial_rows
            if r["reference"] == reference
            and r["condition"] == "exact_ssp"
        ]
        exact_known = [
            r for r in downstream_rows
            if r["reference"] == reference
            and r["condition"] == "exact_ssp"
            and r["scenario"] == "known_coupling"
        ]
        exact_corr = float(np.mean(
            [r["coupling_energy_correlation"] for r in exact_known]
        ))
        exact_k = float(np.mean(
            [r["mean_k_rmse"] for r in exact_front]
        ))

        for condition in conds:
            front = [
                r for r in trial_rows
                if r["reference"] == reference
                and r["condition"] == condition
            ]
            no = [
                r for r in downstream_rows
                if r["reference"] == reference
                and r["condition"] == condition
                and r["scenario"] == "no_coupling"
            ]
            known = [
                r for r in downstream_rows
                if r["reference"] == reference
                and r["condition"] == condition
                and r["scenario"] == "known_coupling"
            ]

            def avg(rows, key):
                values = [
                    float(r[key]) for r in rows
                    if np.isfinite(float(r[key]))
                ]
                return (
                    float(np.mean(values))
                    if values else float("nan")
                )

            row = {
                "reference": reference,
                "condition": condition,
                "n_trials": len(front),
                "mean_k_rmse": avg(front, "mean_k_rmse"),
                "frame_success_rate":
                    avg(front, "frame_success_rate"),
                "mean_basis_mac": avg(front, "mean_basis_mac"),
                "mean_runtime_s": avg(front, "mean_runtime_s"),
                "mean_profile_delta_rms_m_s":
                    avg(front, "mean_profile_delta_rms_m_s"),
                "worst_profile_delta_rms_m_s":
                    max(
                        float(r["worst_profile_delta_rms_m_s"])
                        for r in front
                    ),
                "mean_profile_age_s":
                    avg(front, "mean_profile_age_s"),
                "worst_profile_age_s":
                    max(
                        float(r["worst_profile_age_s"])
                        for r in front
                    ),
                "no_coupling_coeff_error":
                    avg(no, "mean_coefficient_relative_error"),
                "known_coupling_coeff_error":
                    avg(known, "mean_coefficient_relative_error"),
                "known_coupling_correlation":
                    avg(known, "coupling_energy_correlation"),
                "no_coupling_false_alarm_rate":
                    avg(no, "false_alarm_rate"),
                "known_coupling_detection_rate":
                    avg(known, "detection_rate"),
                "known_coupling_trajectory_rmse":
                    avg(known, "coupling_fraction_trajectory_rmse"),
                "k_ratio_vs_exact":
                    avg(front, "mean_k_rmse")
                    / max(exact_k, 1.0e-15),
                "correlation_drop_vs_exact":
                    exact_corr
                    - avg(known, "coupling_energy_correlation"),
            }

            rule = cfg["decision_rule"]
            primary = float(rule["coefficient_error_max"])
            strong = (
                row["no_coupling_coeff_error"] <= primary
                and row["known_coupling_coeff_error"] <= primary
                and row["known_coupling_correlation"]
                    >= exact_corr - float(
                        rule["correlation_drop_max"]
                    )
                and row["no_coupling_false_alarm_rate"]
                    <= float(rule["false_alarm_rate_max"])
                and row["known_coupling_detection_rate"]
                    >= float(rule["detection_rate_min"])
                and row["frame_success_rate"]
                    >= float(rule["frame_success_rate_min"])
            )
            marginal = (
                row["no_coupling_coeff_error"]
                    <= float(rule["marginal_error_ratio"]) * primary
                and row["known_coupling_coeff_error"]
                    <= float(rule["marginal_error_ratio"]) * primary
                and row["known_coupling_correlation"]
                    >= exact_corr - float(
                        rule["marginal_correlation_drop_max"]
                    )
                and row["no_coupling_false_alarm_rate"]
                    <= float(rule["marginal_false_alarm_rate_max"])
                and row["known_coupling_detection_rate"]
                    >= float(rule["marginal_detection_rate_min"])
                and row["frame_success_rate"]
                    >= float(rule["marginal_frame_success_rate_min"])
            )
            row["classification"] = (
                "PASS" if strong
                else ("MARGINAL" if marginal else "FAIL")
            )
            out.append(row)
    return out


def paired_statistics(
    trial_rows: list[dict],
    downstream_rows: list[dict],
    cfg: dict,
) -> list[dict]:
    out = []
    stat = cfg["statistics"]
    refs = sorted({r["reference"] for r in trial_rows})
    conds = [
        c["name"] for c in cfg["conditions"]
        if c["name"] != "exact_ssp"
    ]

    for reference in refs:
        exact_front = sorted(
            [
                r for r in trial_rows
                if r["reference"] == reference
                and r["condition"] == "exact_ssp"
            ],
            key=lambda r: int(r["trial"]),
        )
        exact_known = sorted(
            [
                r for r in downstream_rows
                if r["reference"] == reference
                and r["condition"] == "exact_ssp"
                and r["scenario"] == "known_coupling"
            ],
            key=lambda r: int(r["trial"]),
        )

        for condition in conds:
            test_front = sorted(
                [
                    r for r in trial_rows
                    if r["reference"] == reference
                    and r["condition"] == condition
                ],
                key=lambda r: int(r["trial"]),
            )
            test_known = sorted(
                [
                    r for r in downstream_rows
                    if r["reference"] == reference
                    and r["condition"] == condition
                    and r["scenario"] == "known_coupling"
                ],
                key=lambda r: int(r["trial"]),
            )

            comparisons = [
                (
                    "k_rmse_chain_minus_exact",
                    np.asarray([
                        a["mean_k_rmse"] - b["mean_k_rmse"]
                        for a, b in zip(test_front, exact_front)
                    ]),
                ),
                (
                    "known_coeff_error_chain_minus_exact",
                    np.asarray([
                        a["mean_coefficient_relative_error"]
                        - b["mean_coefficient_relative_error"]
                        for a, b in zip(test_known, exact_known)
                    ]),
                ),
                (
                    "known_correlation_exact_minus_chain",
                    np.asarray([
                        b["coupling_energy_correlation"]
                        - a["coupling_energy_correlation"]
                        for a, b in zip(test_known, exact_known)
                    ]),
                ),
            ]
            for name, values in comparisons:
                low, high = bootstrap_mean_ci(
                    values,
                    seed=seed32(
                        stat["bootstrap_seed_namespace"],
                        reference,
                        condition,
                        name,
                    ),
                    samples=int(stat["bootstrap_samples"]),
                    ci_level=float(stat["ci_level"]),
                )
                out.append({
                    "reference": reference,
                    "condition": condition,
                    "comparison": name,
                    "n_trials": int(values.size),
                    "mean_difference": float(np.mean(values)),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "significant_positive_degradation": bool(low > 0.0),
                })
    return out


# ---------------------------------------------------------------------------
# Figures / report
# ---------------------------------------------------------------------------
def make_figures(
    dirs: dict[str, Path],
    summary: list[dict],
    cfg: dict,
) -> None:
    refs = sorted({r["reference"] for r in summary})
    conds = [c["name"] for c in cfg["conditions"]]
    x = np.arange(len(conds), dtype=np.float64)
    width = 0.8 / max(len(refs), 1)

    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    for ri, ref in enumerate(refs):
        values = [
            next(
                r["known_coupling_coeff_error"]
                for r in summary
                if r["reference"] == ref
                and r["condition"] == c
            )
            for c in conds
        ]
        ax.bar(
            x + (ri - (len(refs)-1)/2) * width,
            values,
            width=width,
            label=ref,
        )
    ax.axhline(
        float(cfg["decision_rule"]["coefficient_error_max"]),
        linestyle="--",
        label="error budget",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=25, ha="right")
    ax.set_ylabel("Known-coupling coefficient error")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        dirs["figures"] / "01_known_coupling_error.png",
        dpi=220,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    for ref in refs:
        rows = [
            r for r in summary if r["reference"] == ref
        ]
        ax.plot(
            [r["mean_profile_age_s"] for r in rows],
            [r["known_coupling_coeff_error"] for r in rows],
            marker="o",
            label=ref,
        )
    ax.axhline(
        float(cfg["decision_rule"]["coefficient_error_max"]),
        linestyle="--",
    )
    ax.set_xlabel("Mean SSP profile age (s)")
    ax.set_ylabel("Known-coupling coefficient error")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        dirs["figures"] / "02_profile_age_vs_error.png",
        dpi=220,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    for ri, ref in enumerate(refs):
        far = [
            next(
                r["no_coupling_false_alarm_rate"]
                for r in summary
                if r["reference"] == ref
                and r["condition"] == c
            )
            for c in conds
        ]
        det = [
            next(
                r["known_coupling_detection_rate"]
                for r in summary
                if r["reference"] == ref
                and r["condition"] == c
            )
            for c in conds
        ]
        ax.plot(x, det, marker="o", label=f"{ref} detection")
        ax.plot(x, far, marker="s", label=f"{ref} false alarm")
    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=25, ha="right")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(
        dirs["figures"] / "03_detection_false_alarm.png",
        dpi=220,
    )
    plt.close(fig)


def write_report(
    *,
    dirs: dict[str, Path],
    cfg: dict,
    protocol_hash: str,
    csmt_hash: str,
    summary: list[dict],
) -> None:
    lines = [
        "# CSMT-v2 + SSP Measurement Chain End-to-End Validation v1",
        "",
        f"- Joint protocol hash: `{protocol_hash}`",
        f"- Frozen CSMT-v2 hash: `{csmt_hash}`",
        f"- Trials per reference: {cfg['trials_per_reference']}",
        "- CSMT-v2 retuned: NO",
        "- Reacquire: NO",
        "- WOA23 T/S-derived time-varying SSP used as controlled environmental truth",
        "- Measured acoustic data used: NO",
        "",
        "## Condition summary",
        "",
        "| reference | condition | SSP age | SSP rms error | kRMSE | basis MAC | "
        "no-cpl err | known-cpl err | corr | FAR | detection | verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in summary:
        lines.append(
            f"| {r['reference']} | {r['condition']} | "
            f"{r['mean_profile_age_s']:.1f} | "
            f"{r['mean_profile_delta_rms_m_s']:.3f} | "
            f"{r['mean_k_rmse']:.3e} | "
            f"{r['mean_basis_mac']:.5f} | "
            f"{r['no_coupling_coeff_error']:.4f} | "
            f"{r['known_coupling_coeff_error']:.4f} | "
            f"{r['known_coupling_correlation']:.4f} | "
            f"{r['no_coupling_false_alarm_rate']:.3f} | "
            f"{r['known_coupling_detection_rate']:.3f} | "
            f"**{r['classification']}** |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "The exact-SSP condition is the frozen CSMT-v2 application control. "
        "All other conditions change only the SSP acquisition chain.",
        "",
        "A PASS therefore means that a realistic SSP acquisition configuration "
        "preserves the complete CSMT -> local-basis -> modal-decomposition "
        "application chain under the pre-registered budget. A FAIL does not "
        "invalidate CSMT-v2 itself; it identifies the SSP acquisition chain as "
        "an application bottleneck.",
        "",
        "## Scientific boundary",
        "",
        "This experiment uses real WOA23 hydrographic profile products to define "
        "the environmental state, but the acoustic observations are still "
        "synthetic. It must be described as controlled/data-informed application "
        "validation, not real-sea acoustic validation.",
    ]
    (dirs["root"] / "REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = load_json(CONFIG_PATH)
    dirs = ensure_dirs(ROOT / cfg["output_root"])

    # dependency / project checks
    try:
        import gsw  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "The joint WOA/TEOS-10 experiment requires Python package 'gsw'. "
            "Install it in the same .venv before running."
        ) from exc

    csmt_lock, csmt_result, csmt_cfg = load_frozen_csmt()
    base_cfg = load_json(
        ROOT / "configs" / "dynamic_local_basis_coupling_suite.json"
    )
    protocol_hash = lock_protocol(cfg, dirs, csmt_lock)

    if (dirs["root"] / "COMPLETE.marker").exists() and not bool(
        cfg["force_recompute"]
    ):
        print("=" * 100)
        print("JOINT CSMT + SSP CHAIN VALIDATION ALREADY COMPLETE")
        print(f"outputs = {dirs['root']}")
        print("=" * 100)
        return

    conditions = condition_entries(cfg)

    print("=" * 108)
    print("CSMT-v2 + REALISTIC SSP MEASUREMENT CHAIN — END-TO-END v1")
    print("=" * 108)
    print(f"source CSMT decision : {csmt_result['decision']}")
    print(f"frozen CSMT hash     : {csmt_lock['parameter_hash']}")
    print(f"joint protocol hash  : {protocol_hash}")
    print(f"references           : {[r['name'] for r in cfg['references']]}")
    print(f"conditions/reference : {len(conditions)}")
    print(f"trials/reference     : {cfg['trials_per_reference']}")
    print(f"frames/trial         : {cfg['time_variation']['time_points']}")
    print("CSMT retuned          : NO")
    print("measured acoustics    : NO")
    print("-" * 108)

    trial_rows = []
    downstream_rows = []

    for ref_entry in cfg["references"]:
        reference = reference_from_entry(ref_entry)
        ref_name = str(ref_entry["name"])
        env = build_environment(reference, cfg, base_cfg)

        no_truth_proto = true_coefficients(env, base_cfg, 0.0)
        known_truth_proto = true_coefficients(
            env,
            base_cfg,
            float(cfg["decomposition"]["coupling_strength_rad"]),
        )
        baseline_fraction = float(
            coupling_fraction(no_truth_proto)[0]
        )
        known_dev = np.abs(
            coupling_fraction(known_truth_proto) - baseline_fraction
        )
        event_threshold = float(
            cfg["decision_rule"]["detection_threshold_fraction_of_true_peak"]
            * np.max(known_dev)
        )

        print()
        print(
            f"REFERENCE {ref_name}: contrast={reference.contrast_m_s:.2f} m/s, "
            f"thermocline={env['model'].taper_centre_m:.1f} m, "
            f"detection threshold={event_threshold:.4f}"
        )

        for trial in range(int(cfg["trials_per_reference"])):
            t0 = time.perf_counter()

            observations, y_hashes = make_observations(
                env=env,
                cfg=cfg,
                reference_name=ref_name,
                trial=trial,
            )
            downstream_truth = {}
            downstream_pressure = {}
            for scenario in ("no_coupling", "known_coupling"):
                truth, pressure = make_pressure_series(
                    env=env,
                    cfg=cfg,
                    base_cfg=base_cfg,
                    reference_name=ref_name,
                    trial=trial,
                    scenario=scenario,
                )
                downstream_truth[scenario] = truth
                downstream_pressure[scenario] = pressure

            trial_snapshot = []
            for condition in conditions:
                result, details = run_or_load(
                    dirs=dirs,
                    protocol_hash=protocol_hash,
                    reference_name=ref_name,
                    condition=condition,
                    trial=trial,
                    observations=observations,
                    y_hashes=y_hashes,
                    env=env,
                    base_cfg=base_cfg,
                    csmt_cfg=csmt_cfg,
                    cfg=cfg,
                )
                front = front_end_metrics(
                    result=result,
                    env=env,
                    details=details,
                    cfg=cfg,
                )
                frow = {
                    "reference": ref_name,
                    "condition": condition["name"],
                    "trial": trial,
                    **front,
                }
                trial_rows.append(frow)
                trial_snapshot.append(frow)

                for scenario in ("no_coupling", "known_coupling"):
                    metrics = downstream_metrics(
                        result=result,
                        truth_coefficients=downstream_truth[scenario],
                        pressure=downstream_pressure[scenario],
                        env=env,
                        base_cfg=base_cfg,
                        event_threshold=event_threshold,
                        scenario=scenario,
                    )
                    downstream_rows.append({
                        "reference": ref_name,
                        "condition": condition["name"],
                        "trial": trial,
                        "scenario": scenario,
                        **metrics,
                    })

            msg = " | ".join(
                f"{r['condition']}={r['mean_k_rmse']:.2e}"
                for r in trial_snapshot
            )
            print(
                f"  trial {trial+1:02d}/{cfg['trials_per_reference']} | "
                f"{msg} | elapsed={time.perf_counter()-t0:.1f}s"
            )
            write_csv(
                dirs["csv"] / "trial_summary.partial.csv",
                trial_rows,
            )
            write_csv(
                dirs["csv"] / "downstream.partial.csv",
                downstream_rows,
            )

    summary = aggregate_rows(
        trial_rows, downstream_rows, cfg
    )
    paired = paired_statistics(
        trial_rows, downstream_rows, cfg
    )

    write_csv(dirs["csv"] / "trial_summary.csv", trial_rows)
    write_csv(
        dirs["csv"] / "downstream_trial_summary.csv",
        downstream_rows,
    )
    write_csv(dirs["csv"] / "condition_summary.csv", summary)
    write_csv(dirs["csv"] / "paired_statistics.csv", paired)
    for p in dirs["csv"].glob("*.partial.csv"):
        p.unlink(missing_ok=True)

    result = {
        "experiment": cfg["experiment"],
        "protocol_hash": protocol_hash,
        "frozen_csmt_v2_hash": csmt_lock["parameter_hash"],
        "source_csmt_decision": csmt_result["decision"],
        "condition_summary": summary,
        "scientific_boundary": {
            "csmt_v2_retuned": False,
            "csmt_v2_has_reacquire": False,
            "real_woa_hydrography_used": True,
            "measured_acoustic_data_used": False,
            "truth_bottom_parameters_enter_algorithm": False,
        },
    }
    (dirs["root"] / "JOINT_RESULT.json").write_text(
        json.dumps(
            result, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    make_figures(dirs, summary, cfg)
    write_report(
        dirs=dirs,
        cfg=cfg,
        protocol_hash=protocol_hash,
        csmt_hash=str(csmt_lock["parameter_hash"]),
        summary=summary,
    )
    (dirs["root"] / "COMPLETE.marker").write_text(
        f"protocol_hash={protocol_hash}\n",
        encoding="utf-8",
    )

    print()
    print("=" * 108)
    print("JOINT VALIDATION COMPLETE")
    print("=" * 108)
    for r in summary:
        print(
            f"{r['reference']:11s} | {r['condition']:22s} | "
            f"{r['classification']:8s} | "
            f"k={r['mean_k_rmse']:.3e} | "
            f"known err={r['known_coupling_coeff_error']:.4f} | "
            f"corr={r['known_coupling_correlation']:.4f} | "
            f"FAR={r['no_coupling_false_alarm_rate']:.3f} | "
            f"DET={r['known_coupling_detection_rate']:.3f}"
        )
    print("-" * 108)
    print("CSMT-v2 retuned : NO")
    print("measured acoustics: NO")
    print(f"outputs         : {dirs['root']}")
    print("=" * 108)


if __name__ == "__main__":
    main()
