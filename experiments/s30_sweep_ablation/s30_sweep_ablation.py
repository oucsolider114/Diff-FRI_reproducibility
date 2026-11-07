"""S30: sensitivity of the local update to the number of modewise sweeps.

Question
--------
The main text uses two modewise sweeps per update. Is one insufficient,
and is a third redundant?

Design
------
Same environment as the canonical dynamic benchmark
(configs/dynamic_local_basis_coupling_suite.json): 40 m, 300 Hz,
40 receivers on 1 m, 21 frames, 20 realizations, tracking SNR 30 dB.
The first-frame state is taken from data/synthetic/frame_metrics.csv
(column csmt_k* at time_index 0) so that all sweep counts start from the
identical state on identical observations.

Two forward conditions are run:
  fine025 : observations synthesised on a 0.25 m depth grid,
            tracked with the 1.0 m water-column basis used everywhere else.
  native  : observations synthesised on the same 1.0 m grid.

The check is self-consistent: observations are synthesised with the same
water-column recurrence that the tracker uses, perturbed only by additive
complex noise.  The 0.25 m condition is a step-size control, not a
forward-model mismatch, because the cosine finite-difference recurrence is
exact for a constant sound-speed interval at any step.  A basis-mismatch
condition cannot be constructed from the available solvers: solve_mode and
solve_mode_impedance return pointwise identical columns at fixed k, since
they differ only in how the root is selected.  The KRAKEN-versus-water-column
mismatch of the full pipeline is not reproduced here.  Only relative
differences between sweep counts are interpreted.

Outputs
-------
results/s30_sweep_summary.csv
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from mode_fri.mode_solver import solve_mode, solve_mode_impedance  # noqa: E402
from mode_fri.causal_ssp_modal_tracking import (  # noqa: E402
    CSMTConfig,
    CausalSSPModalTracking,
)

CONFIG = REPO / "configs" / "dynamic_local_basis_coupling_suite.json"
METRICS = REPO / "data" / "synthetic" / "frame_metrics.csv"
OUT = Path(__file__).resolve().parent / "results"


# --------------------------------------------------------------------------
# Environment (re-implemented here so the script runs without matplotlib/torch)
# --------------------------------------------------------------------------
def reference_ssp(depths, cfg):
    z = np.asarray(depths, float)
    surface_speed = float(cfg["surface_speed_m_s"])
    deep_speed = float(cfg["deep_speed_m_s"])
    mixed_bottom = float(cfg["mixed_layer_bottom_m"])
    thermo_bottom = float(cfg["thermocline_bottom_m"])
    c = np.empty_like(z)
    c[z <= mixed_bottom] = surface_speed
    middle = (z > mixed_bottom) & (z <= thermo_bottom)
    frac = (z[middle] - mixed_bottom) / max(thermo_bottom - mixed_bottom, 1.0e-12)
    c[middle] = surface_speed + frac * (deep_speed - surface_speed)
    c[z > thermo_bottom] = deep_speed
    return c


def ssp_displacement(t, cfg):
    amp = float(cfg["amplitude_m"])
    harmonic = float(cfg.get("second_harmonic_fraction", 0.25))
    phase = float(cfg.get("second_harmonic_phase_rad", 0.35))
    raw = math.sin(2.0 * math.pi * t) + harmonic * math.sin(
        4.0 * math.pi * t + phase
    )
    return amp * raw / (1.0 + abs(harmonic))


def make_dynamic_ssp(depths, t, cfg):
    z = np.asarray(depths, float)
    c0 = reference_ssp(z, cfg["reference_profile"])
    eta = ssp_displacement(t, cfg)
    center = float(cfg["center_depth_m"])
    sigma = float(cfg["vertical_taper_sigma_m"])
    taper = np.exp(-0.5 * ((z - center) / sigma) ** 2)
    shifted = np.clip(z - eta * taper, z[0], z[-1])
    return np.interp(shifted, z, c0).astype(np.float64), float(eta)


def oracle_bottom_residual(k, omega, ssp, dz, bottom):
    kw = dict(
        water_density=float(bottom["water_density_kg_m3"]),
        bottom_sound_speed=float(bottom["sound_speed_m_s"]),
        bottom_density=float(bottom["density_kg_m3"]),
    )
    psi = solve_mode_impedance(
        horizontal_wavenumber=float(k),
        omega=float(omega),
        sound_speed_profile=ssp,
        dz=float(dz),
        **kw,
    )
    return impedance_bottom_residual(
        psi=psi,
        horizontal_wavenumber=float(k),
        omega=float(omega),
        sound_speed_profile=ssp,
        dz=float(dz),
        **kw,
    )


def _bisect(func, lo, hi, iterations=70):
    a, b = float(lo), float(hi)
    fa = float(func(a))
    for _ in range(int(iterations)):
        m = 0.5 * (a + b)
        fm = float(func(m))
        if fa * fm <= 0.0:
            b = m
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def find_oracle_roots(K, omega, ssp, dz, bottom, scan_points):
    bulk = omega / float(bottom["sound_speed_m_s"])
    lo = max(bulk * 1.00001, 0.90 * omega / float(np.max(ssp)))
    hi = 0.999999 * omega / float(np.min(ssp))
    grid = np.linspace(lo, hi, int(scan_points))
    residual = lambda k: oracle_bottom_residual(k, omega, ssp, dz, bottom)  # noqa: E731
    values = np.asarray([residual(float(k)) for k in grid], dtype=np.float64)
    roots = []
    for i in range(grid.size - 1):
        fa, fb = values[i], values[i + 1]
        if not np.isfinite(fa) or not np.isfinite(fb):
            continue
        if fa == 0.0 or fa * fb < 0.0:
            root = _bisect(residual, float(grid[i]), float(grid[i + 1]))
            if not roots or min(abs(root - e) for e in roots) > 1.0e-7:
                roots.append(root)
    roots = sorted(roots, reverse=True)
    if len(roots) < int(K):
        raise RuntimeError("insufficient oracle roots")
    return np.asarray(roots[: int(K)], dtype=np.float64)


def canonicalize_column(column):
    x = np.asarray(column, dtype=np.complex128).reshape(-1)
    x = x / float(np.linalg.norm(x))
    pivot = int(np.argmax(np.abs(x)))
    x = x * np.exp(-1j * np.angle(x[pivot]))
    if np.real(x[pivot]) < 0.0:
        x = -x
    return x


def build_basis(ks, omega, ssp, dz, receiver_depths, water_depth, bottom=None):
    indices = np.rint(np.asarray(receiver_depths, float) / dz).astype(np.int64)
    columns = []
    for k in np.asarray(ks, dtype=np.float64):
        if bottom is None:
            psi = solve_mode(float(k), float(omega), ssp, float(dz))
        else:
            psi = solve_mode_impedance(
                horizontal_wavenumber=float(k),
                omega=float(omega),
                sound_speed_profile=ssp,
                dz=float(dz),
                water_density=float(bottom["water_density_kg_m3"]),
                bottom_sound_speed=float(bottom["sound_speed_m_s"]),
                bottom_density=float(bottom["density_kg_m3"]),
            )
        columns.append(canonicalize_column(psi[indices]))
    return np.column_stack(columns).astype(np.complex128)


def random_coefficients(K, snapshots, seed):
    rng = np.random.default_rng(int(seed))
    row_scale = np.exp(-0.22 * np.arange(int(K), dtype=np.float64))
    coeff = (
        rng.standard_normal((int(K), int(snapshots)))
        + 1j * rng.standard_normal((int(K), int(snapshots)))
    ) / math.sqrt(2.0)
    return (coeff * row_scale[:, None]).astype(np.complex128)


def add_complex_noise(data, snr_db, seed):
    clean = np.asarray(data, dtype=np.complex128)
    rng = np.random.default_rng(int(seed))
    noise = (
        rng.standard_normal(clean.shape) + 1j * rng.standard_normal(clean.shape)
    ) / math.sqrt(2.0)
    signal = float(np.linalg.norm(clean) ** 2)
    noise_power = float(np.linalg.norm(noise) ** 2)
    scale = math.sqrt(signal / max(noise_power * 10.0 ** (snr_db / 10.0), 1.0e-30))
    return (clean + scale * noise).astype(np.complex128)


# --------------------------------------------------------------------------
def load_reference():
    rows = list(
        csv.DictReader(open(METRICS, newline="", encoding="utf-8-sig"))
    )
    by_repeat = {}
    for row in rows:
        by_repeat.setdefault(int(row["repeat"]), {})[int(row["time_index"])] = row
    n_rep = len(by_repeat)
    n_time = len(next(iter(by_repeat.values())))
    kraken = np.zeros((n_rep, n_time, 3))
    k0 = np.zeros((n_rep, 3))
    for r, frames in by_repeat.items():
        for ti in range(n_time):
            row = frames[ti]
            kraken[r, ti] = [
                float(row["kraken_k1"]),
                float(row["kraken_k2"]),
                float(row["kraken_k3"]),
            ]
        k0[r] = [
            float(frames[0]["csmt_k1"]),
            float(frames[0]["csmt_k2"]),
            float(frames[0]["csmt_k3"]),
        ]
    return kraken, k0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text())
    frequency = float(cfg["frequency_hz"])
    water_depth = float(cfg["water_depth_m"])
    dz = float(cfg["dz_m"])
    bottom = cfg["forward_bottom"]
    snr = float(cfg["tracking"]["snr_db"])

    depths = np.arange(0.0, water_depth + 0.5 * dz, dz)
    array = cfg["receiver_array"]
    receivers = np.arange(
        array["start_depth_m"],
        array["end_depth_m"] + 0.5 * array["spacing_m"],
        array["spacing_m"],
    )
    n_time = int(cfg["time_points"])
    fractions = np.linspace(0.0, 1.0, n_time)
    omega = 2.0 * math.pi * frequency

    ssp = np.zeros((n_time, depths.size))
    for ti in range(n_time):
        ssp[ti], _ = make_dynamic_ssp(depths, float(fractions[ti]), cfg["ssp_dynamic"])

    kraken, k0 = load_reference()
    n_rep = kraken.shape[0]

    seed_sets = {
        "A": (7000, 9000),
        "B": (55000, 77000),
    }
    conditions = {"fine025": 0.25, "native": 1.0}
    sweeps_list = (1, 2, 3, 5, 10)

    rows_out = []
    for condition, dz_generator in conditions.items():
        generator_depths = np.arange(0.0, water_depth + 0.5 * dz_generator, dz_generator)
        generator_indices = np.rint(receivers / dz_generator).astype(np.int64)
        generator_basis = []
        for ti in range(n_time):
            ssp_generator, _ = make_dynamic_ssp(
                generator_depths, float(fractions[ti]), cfg["ssp_dynamic"]
            )
            columns = [
                canonicalize_column(
                    solve_mode(float(k), omega, ssp_generator, dz_generator)[
                        generator_indices
                    ]
                )
                for k in kraken[0, ti]
            ]
            generator_basis.append(
                np.column_stack(columns).astype(np.complex128)
            )
        for set_name, (c_off, n_off) in seed_sets.items():
            observations = [
                [
                    add_complex_noise(
                        generator_basis[ti]
                        @ random_coefficients(3, 20, c_off + 100 * r + ti),
                        snr,
                        n_off + 100 * r + ti,
                    )
                    for ti in range(n_time)
                ]
                for r in range(n_rep)
            ]
            previous = None
            for sweeps in sweeps_list:
                started = time.perf_counter()
                per_repeat = []
                per_frame = np.zeros(n_time)
                for r in range(n_rep):
                    tracker = CausalSSPModalTracking(
                        CSMTConfig(local_coordinate_sweeps=int(sweeps))
                    )
                    tracker.initialize_from_full_estimate(
                        k_hat=k0[r],
                        SSP_t=ssp[0],
                        Y_t=observations[r][0],
                        frequency_hz=frequency,
                        dz=dz,
                        receiver_depths=receivers,
                        water_depth=water_depth,
                        full_runtime_seconds=0.0,
                    )
                    history = np.zeros((n_time, 3))
                    history[0] = k0[r]
                    for ti in range(1, n_time):
                        result = tracker.step(
                            Y_t=observations[r][ti],
                            SSP_t=ssp[ti],
                            frequency_hz=frequency,
                            dz=dz,
                            receiver_depths=receivers,
                            water_depth=water_depth,
                        )
                        history[ti] = result.k_hat
                    error = (history - kraken[r]) ** 2
                    per_frame += error.mean(axis=1)
                    per_repeat.append(float(np.sqrt(error[1:].mean())))
                per_frame /= n_rep
                mean_rmse = float(np.mean(per_repeat))
                rows_out.append(
                    {
                        "condition": condition,
                        "seed_set": set_name,
                        "sweeps": int(sweeps),
                        "update_rmse_mean_rad_m": mean_rmse,
                        "update_rmse_median_rad_m": float(np.median(per_repeat)),
                        "all_frame_rmse_rad_m": float(np.sqrt(per_frame.mean())),
                        "delta_vs_one_sweep_pct": 0.0,
                        "delta_vs_previous_pct": 0.0,
                        "runtime_s": round(time.perf_counter() - started, 2),
                    }
                )
                if sweeps == 1:
                    baseline = mean_rmse
                rows_out[-1]["delta_vs_one_sweep_pct"] = 100.0 * (
                    mean_rmse / baseline - 1.0
                )
                if previous is not None:
                    rows_out[-1]["delta_vs_previous_pct"] = 100.0 * (
                        mean_rmse / previous - 1.0
                    )
                previous = mean_rmse
                print(
                    "%-8s %s sweeps=%d  %.4e  (vs1 %+.2f%%)"
                    % (
                        condition,
                        set_name,
                        sweeps,
                        mean_rmse,
                        100.0 * (mean_rmse / baseline - 1.0),
                    )
                )

    out_csv = OUT / "s30_sweep_summary.csv"
    with open(out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print("wrote", out_csv)


if __name__ == "__main__":
    main()
