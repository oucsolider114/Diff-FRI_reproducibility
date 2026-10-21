"""Unified dynamic local-mode-basis and coupling-separation experiment.

Purpose
-------
This experiment extends the frozen seabed-parameter-free modal-wavenumber
estimator without modifying its source code.  It tests the full controlled chain

    time-varying SSP -> oracle k(t) -> oracle local modes -> synthetic VLA data
                       -> Proposed k_hat(t) -> estimated local modes
                       -> modal coefficient decomposition

and compares three modal bases:

    1) Fixed basis:              oracle basis frozen at the reference time;
    2) Oracle dynamic basis:     basis generated from oracle k(t) + current SSP;
    3) Estimated dynamic basis:  basis generated from Proposed k_hat(t) + current SSP.

Two tracking scenarios are evaluated with the *same* estimated dynamic basis:

    A) environment varies but true modal coefficients are constant (no coupling);
    B) environment varies and a known unitary two-mode coupling event is present.

The forward/oracle root solver uses bottom geoacoustic parameters only to create
controlled truth.  Those bottom parameters never enter the Proposed estimator,
the dynamic coarse dictionary, continuous refinement, gate, or local-mode
recovery from k_hat.

The script writes new files only under outputs/dynamic_local_basis_coupling_suite.
It does not modify frozen estimators, configs, or benchmark outputs.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mode_fri import direct_estimator  # noqa: E402
from mode_fri.mode_solver import (  # noqa: E402
    impedance_bottom_residual,
    solve_mode,
    solve_mode_impedance,
)


CONFIG_PATH = ROOT / "configs" / "dynamic_local_basis_coupling_suite.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "dynamic_local_basis_coupling_suite"


@dataclass(frozen=True)
class DynamicMetadata:
    K: int
    frequency: float
    array_geometry: np.ndarray
    water_depth: float
    SSP: np.ndarray


# -----------------------------------------------------------------------------
# Configuration / I/O
# -----------------------------------------------------------------------------
def load_config(path: Path = CONFIG_PATH) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))

    required = [
        "frequency_hz",
        "water_depth_m",
        "dz_m",
        "K_values",
        "time_points",
        "time_span_s",
        "ssp_dynamic",
        "forward_bottom",
        "estimation",
        "tracking",
        "coupling",
    ]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise KeyError(f"Missing config keys: {missing}")

    if int(cfg["time_points"]) < 5:
        raise ValueError("time_points must be >= 5")
    if any(int(k) not in (3, 5) for k in cfg["K_values"]):
        raise ValueError("This controlled dynamic suite currently supports K=3 and/or K=5")
    if float(cfg["dz_m"]) <= 0:
        raise ValueError("dz_m must be positive")

    est = cfg["estimation"]
    if int(est["snapshots"]) < 5:
        raise ValueError("estimation.snapshots must be >= 5")
    if int(est["repeats"]) < 1:
        raise ValueError("estimation.repeats must be >= 1")
    if int(est["top_n_hypotheses"]) != 5:
        raise ValueError("Frozen Top-N setting must remain 5")
    if int(est["beam_branch_factor"]) != 6:
        raise ValueError("Frozen beam branch factor must remain 6")
    if int(est["beam_width"]) != 24:
        raise ValueError("Frozen beam width must remain 24")

    return cfg


def ensure_dirs(output_root: Path) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    figures = output_root / "figures"
    csv_dir = output_root / "csv"
    arrays = output_root / "arrays"
    for path in (figures, csv_dir, arrays):
        path.mkdir(parents=True, exist_ok=True)
    return {"root": output_root, "figures": figures, "csv": csv_dir, "arrays": arrays}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    names = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------------
# Environment model
# -----------------------------------------------------------------------------
def reference_ssp(depths: np.ndarray, cfg: dict) -> np.ndarray:
    """Piecewise reference SSP used only for this controlled dynamic experiment."""
    z = np.asarray(depths, dtype=np.float64)
    surface_speed = float(cfg["surface_speed_m_s"])
    deep_speed = float(cfg["deep_speed_m_s"])
    mixed_bottom = float(cfg["mixed_layer_bottom_m"])
    thermo_bottom = float(cfg["thermocline_bottom_m"])
    if not (z[0] <= mixed_bottom < thermo_bottom <= z[-1]):
        raise ValueError("Invalid reference SSP layer depths")
    c = np.empty_like(z)
    c[z <= mixed_bottom] = surface_speed
    middle = (z > mixed_bottom) & (z <= thermo_bottom)
    fraction = (z[middle] - mixed_bottom) / max(thermo_bottom - mixed_bottom, 1.0e-12)
    c[middle] = surface_speed + fraction * (deep_speed - surface_speed)
    c[z > thermo_bottom] = deep_speed
    return c


def ssp_displacement(time_fraction: float, cfg: dict) -> float:
    """Controlled nonstationary thermocline displacement in metres."""
    amp = float(cfg["amplitude_m"])
    harmonic = float(cfg.get("second_harmonic_fraction", 0.25))
    phase = float(cfg.get("second_harmonic_phase_rad", 0.35))
    raw = (
        math.sin(2.0 * math.pi * time_fraction)
        + harmonic * math.sin(4.0 * math.pi * time_fraction + phase)
    )
    normalization = 1.0 + abs(harmonic)
    return amp * raw / normalization


def make_dynamic_ssp(
    depths: np.ndarray,
    time_fraction: float,
    cfg: dict,
) -> tuple[np.ndarray, float]:
    """Vertically displace the baseline SSP primarily around the thermocline."""
    z = np.asarray(depths, dtype=np.float64)
    c0 = reference_ssp(z, cfg["reference_profile"])
    eta = ssp_displacement(time_fraction, cfg)
    center = float(cfg["center_depth_m"])
    sigma = float(cfg["vertical_taper_sigma_m"])
    taper = np.exp(-0.5 * ((z - center) / sigma) ** 2)
    shifted_depth = np.clip(z - eta * taper, z[0], z[-1])
    c_t = np.interp(shifted_depth, z, c0)
    return c_t.astype(np.float64), float(eta)


# -----------------------------------------------------------------------------
# Controlled forward truth: bottom parameters used ONLY here to choose roots.
# -----------------------------------------------------------------------------
def oracle_bottom_residual(k: float, omega: float, ssp: np.ndarray, dz: float, bottom: dict) -> float:
    psi = solve_mode_impedance(
        horizontal_wavenumber=float(k),
        omega=float(omega),
        sound_speed_profile=ssp,
        dz=float(dz),
        water_density=float(bottom["water_density_kg_m3"]),
        bottom_sound_speed=float(bottom["sound_speed_m_s"]),
        bottom_density=float(bottom["density_kg_m3"]),
    )
    return impedance_bottom_residual(
        psi=psi,
        horizontal_wavenumber=float(k),
        omega=float(omega),
        sound_speed_profile=ssp,
        dz=float(dz),
        water_density=float(bottom["water_density_kg_m3"]),
        bottom_sound_speed=float(bottom["sound_speed_m_s"]),
        bottom_density=float(bottom["density_kg_m3"]),
    )


def _bisect_root(func, left: float, right: float, iterations: int = 70) -> float:
    f_left = float(func(left))
    f_right = float(func(right))
    if not np.isfinite(f_left) or not np.isfinite(f_right):
        raise FloatingPointError("Non-finite root bracket residual")
    if f_left == 0.0:
        return float(left)
    if f_right == 0.0:
        return float(right)
    if f_left * f_right > 0.0:
        raise ValueError("Root bracket does not change sign")
    a, b = float(left), float(right)
    fa = f_left
    for _ in range(int(iterations)):
        m = 0.5 * (a + b)
        fm = float(func(m))
        if fa * fm <= 0.0:
            b = m
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def find_oracle_roots(
    K: int,
    omega: float,
    ssp: np.ndarray,
    dz: float,
    bottom: dict,
    scan_points: int,
) -> np.ndarray:
    """Find the highest K trapped roots using the known-bottom forward model."""
    bottom_bulk = omega / float(bottom["sound_speed_m_s"])
    lower = max(
        bottom_bulk * 1.00001,
        0.90 * omega / float(np.max(ssp)),
    )
    upper = 0.999999 * omega / float(np.min(ssp))
    if lower >= upper:
        raise RuntimeError("Invalid oracle-root search interval")

    grid = np.linspace(lower, upper, int(scan_points), dtype=np.float64)

    def residual(k_value: float) -> float:
        return oracle_bottom_residual(k_value, omega, ssp, dz, bottom)

    values = np.asarray([residual(float(k)) for k in grid], dtype=np.float64)
    roots: list[float] = []
    for index in range(grid.size - 1):
        fa, fb = values[index], values[index + 1]
        if not np.isfinite(fa) or not np.isfinite(fb):
            continue
        if fa == 0.0 or fa * fb < 0.0:
            root = _bisect_root(residual, float(grid[index]), float(grid[index + 1]))
            if not roots or min(abs(root - existing) for existing in roots) > 1.0e-7:
                roots.append(root)

    roots = sorted(roots, reverse=True)
    if len(roots) < int(K):
        raise RuntimeError(f"Only {len(roots)} oracle roots found; K={K}")
    return np.asarray(roots[: int(K)], dtype=np.float64)


# -----------------------------------------------------------------------------
# Water-column modal bases.  No bottom parameter is used here.
# -----------------------------------------------------------------------------
def receiver_indices(receiver_depths: np.ndarray, dz: float, water_depth: float) -> np.ndarray:
    indices = np.rint(np.asarray(receiver_depths, dtype=np.float64) / float(dz)).astype(np.int64)
    max_index = int(round(float(water_depth) / float(dz)))
    if np.any(indices < 0) or np.any(indices > max_index):
        raise ValueError("Receiver depth lies outside water column")
    return indices


def canonicalize_column(column: np.ndarray) -> np.ndarray:
    """Normalize and apply a deterministic phase/sign convention."""
    x = np.asarray(column, dtype=np.complex128).reshape(-1)
    norm = float(np.linalg.norm(x))
    if not np.isfinite(norm) or norm <= 1.0e-14:
        raise FloatingPointError("Mode column has zero/non-finite norm")
    x = x / norm
    pivot_index = int(np.argmax(np.abs(x)))
    pivot = x[pivot_index]
    if abs(pivot) > 0.0:
        x = x * np.exp(-1j * np.angle(pivot))
    if np.real(x[pivot_index]) < 0.0:
        x = -x
    return x


def build_water_column_basis(
    wavenumbers: np.ndarray,
    omega: float,
    ssp: np.ndarray,
    dz: float,
    receiver_depths: np.ndarray,
    water_depth: float,
) -> np.ndarray:
    indices = receiver_indices(receiver_depths, dz, water_depth)
    columns: list[np.ndarray] = []
    for k in np.asarray(wavenumbers, dtype=np.float64).reshape(-1):
        full = solve_mode(float(k), float(omega), ssp, float(dz))
        columns.append(canonicalize_column(full[indices]))
    return np.column_stack(columns).astype(np.complex128)


def column_mac(reference: np.ndarray, candidate: np.ndarray) -> float:
    a = np.asarray(reference, dtype=np.complex128).reshape(-1)
    b = np.asarray(candidate, dtype=np.complex128).reshape(-1)
    denominator = float(np.linalg.norm(a) ** 2 * np.linalg.norm(b) ** 2)
    if denominator <= 1.0e-30:
        return float("nan")
    return float(abs(np.vdot(a, b)) ** 2 / denominator)


# -----------------------------------------------------------------------------
# Dynamic-aware Top-N + Data-Gated Continuous Refinement adapter.
# The algorithm is unchanged; dictionaries/gate are evaluated with current SSP.
# -----------------------------------------------------------------------------
def normalize_snapshot_columns(Y: np.ndarray) -> np.ndarray:
    data = np.asarray(Y, dtype=np.complex128)
    norms = np.maximum(np.linalg.norm(data, axis=0), 1.0e-15)
    return data / norms[None, :]


def dynamic_grid_dictionary(metadata: DynamicMetadata, grid_size: int, dz: float) -> tuple[np.ndarray, np.ndarray]:
    omega = 2.0 * math.pi * float(metadata.frequency)
    grid_min = direct_estimator.SEARCH_LOWER_FRACTION * omega / float(np.max(metadata.SSP))
    grid_max = direct_estimator.SEARCH_UPPER_FRACTION * omega / float(np.min(metadata.SSP))
    grid = np.linspace(grid_max, grid_min, int(grid_size), dtype=np.float64)
    dictionary = build_water_column_basis(
        grid,
        omega,
        metadata.SSP,
        dz,
        metadata.array_geometry,
        metadata.water_depth,
    )
    return grid, dictionary


def projection_scores(Y: np.ndarray, dictionary: np.ndarray) -> tuple[float, float]:
    observations = np.asarray(Y, dtype=np.complex128)
    atoms = np.asarray(dictionary, dtype=np.complex128)
    coefficients, _, _, _ = np.linalg.lstsq(atoms, observations, rcond=None)
    residual = observations - atoms @ coefficients
    numerator = np.sum(np.abs(residual) ** 2, axis=0)
    denominator = np.maximum(np.sum(np.abs(observations) ** 2, axis=0), 1.0e-30)
    ratios = numerator / denominator
    return float(np.mean(ratios)), float(np.max(ratios))


def dynamic_beam_hypotheses(
    Y: np.ndarray,
    metadata: DynamicMetadata,
    grid_size: int,
    top_n: int,
    branch_factor: int,
    beam_width: int,
    minimum_separation: float,
    dz: float,
) -> list[np.ndarray]:
    grid, dictionary = dynamic_grid_dictionary(metadata, grid_size, dz)
    observations = normalize_snapshot_columns(Y)
    reference_energy = max(float(np.linalg.norm(observations) ** 2), 1.0e-30)
    states: list[tuple[tuple[int, ...], np.ndarray, float]] = [(tuple(), observations.copy(), 1.0)]

    for _ in range(int(metadata.K)):
        expanded: dict[tuple[int, ...], tuple[tuple[int, ...], np.ndarray, float]] = {}
        for selected_indices, residual, _ in states:
            correlations = dictionary.conj().T @ residual
            scores = np.linalg.norm(correlations, axis=1)
            if selected_indices:
                selected_k = grid[np.asarray(selected_indices, dtype=np.int64)]
                for grid_index, candidate_k in enumerate(grid):
                    if np.min(np.abs(selected_k - float(candidate_k))) < float(minimum_separation):
                        scores[grid_index] = -np.inf
            finite = np.flatnonzero(np.isfinite(scores))
            if finite.size == 0:
                continue
            order = finite[np.argsort(scores[finite])[::-1]][: min(int(branch_factor), finite.size)]
            for grid_index in order:
                canonical = tuple(sorted((*selected_indices, int(grid_index))))
                selected_dictionary = dictionary[:, list(canonical)]
                coefficients, _, _, _ = np.linalg.lstsq(selected_dictionary, observations, rcond=None)
                new_residual = observations - selected_dictionary @ coefficients
                residual_score = float(np.linalg.norm(new_residual) ** 2 / reference_energy)
                previous = expanded.get(canonical)
                if previous is None or residual_score < previous[2]:
                    expanded[canonical] = (canonical, new_residual, residual_score)
        if not expanded:
            raise RuntimeError("Dynamic beam-GOMP could not generate candidates")
        states = sorted(expanded.values(), key=lambda item: item[2])[: int(beam_width)]

    hypotheses: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for selected_indices, _, _ in states:
        candidate = np.sort(grid[np.asarray(selected_indices, dtype=np.int64)])[::-1]
        key = tuple(float(x) for x in np.round(candidate, 12))
        if key not in seen:
            seen.add(key)
            hypotheses.append(candidate.astype(np.float64, copy=True))
        if len(hypotheses) >= int(top_n):
            break
    if len(hypotheses) < int(top_n):
        raise RuntimeError("Insufficient unique dynamic Top-N hypotheses")
    return hypotheses


def dynamic_topn_coarse(
    Y: np.ndarray,
    metadata: DynamicMetadata,
    grid_size: int,
    top_n: int,
    branch_factor: int,
    beam_width: int,
    minimum_separation: float,
    dz: float,
) -> np.ndarray:
    hypotheses = dynamic_beam_hypotheses(
        Y, metadata, grid_size, top_n, branch_factor, beam_width, minimum_separation, dz
    )
    scored: list[tuple[float, float, np.ndarray]] = []
    omega = 2.0 * math.pi * float(metadata.frequency)
    for candidate in hypotheses:
        dictionary = build_water_column_basis(
            candidate,
            omega,
            metadata.SSP,
            dz,
            metadata.array_geometry,
            metadata.water_depth,
        )
        mean_score, worst_score = projection_scores(Y, dictionary)
        scored.append((worst_score, mean_score, candidate))
    best = min(scored, key=lambda item: (item[0], item[1]))
    return np.sort(np.asarray(best[2], dtype=np.float64))[::-1]


def dynamic_topn_refined(
    Y: np.ndarray,
    metadata: DynamicMetadata,
    seed: int,
    cfg: dict,
    dz: float,
) -> tuple[np.ndarray, np.ndarray, bool, tuple[float, float], tuple[float, float]]:
    coarse = dynamic_topn_coarse(
        Y=Y,
        metadata=metadata,
        grid_size=int(cfg["grid_size"]),
        top_n=int(cfg["top_n_hypotheses"]),
        branch_factor=int(cfg["beam_branch_factor"]),
        beam_width=int(cfg["beam_width"]),
        minimum_separation=float(cfg["grid_min_separation_rad_m"]),
        dz=dz,
    )

    refined = direct_estimator.estimate_from_coarse(
        Y,
        metadata.K,
        metadata.frequency,
        metadata.array_geometry,
        metadata.water_depth,
        metadata.SSP,
        int(seed),
        coarse,
    )

    omega = 2.0 * math.pi * float(metadata.frequency)
    coarse_basis = build_water_column_basis(
        coarse, omega, metadata.SSP, dz, metadata.array_geometry, metadata.water_depth
    )
    refined_basis = build_water_column_basis(
        refined, omega, metadata.SSP, dz, metadata.array_geometry, metadata.water_depth
    )
    coarse_scores = projection_scores(Y, coarse_basis)
    refined_scores = projection_scores(Y, refined_basis)
    tolerance = float(cfg.get("gate_tolerance", 1.0e-12))
    accepted = (
        refined_scores[0] < coarse_scores[0] - tolerance
        and refined_scores[1] <= coarse_scores[1] + tolerance
    )
    final = refined if accepted else coarse
    return (
        np.sort(np.asarray(final, dtype=np.float64))[::-1],
        np.sort(np.asarray(coarse, dtype=np.float64))[::-1],
        bool(accepted),
        coarse_scores,
        refined_scores,
    )


# -----------------------------------------------------------------------------
# Synthetic estimation blocks and tracking signals
# -----------------------------------------------------------------------------
def random_coefficients(K: int, snapshots: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    row_scale = np.exp(-0.22 * np.arange(int(K), dtype=np.float64))
    coefficients = (
        rng.standard_normal((int(K), int(snapshots)))
        + 1j * rng.standard_normal((int(K), int(snapshots)))
    ) / math.sqrt(2.0)
    coefficients *= row_scale[:, None]
    return coefficients.astype(np.complex128)


def add_complex_noise(data: np.ndarray, snr_db: float | None, seed: int) -> tuple[np.ndarray, float]:
    clean = np.asarray(data, dtype=np.complex128)
    if snr_db is None:
        return clean.copy(), float("inf")
    rng = np.random.default_rng(int(seed))
    noise = (
        rng.standard_normal(clean.shape) + 1j * rng.standard_normal(clean.shape)
    ) / math.sqrt(2.0)
    signal_power = float(np.linalg.norm(clean) ** 2)
    noise_power = float(np.linalg.norm(noise) ** 2)
    target_ratio = 10.0 ** (float(snr_db) / 10.0)
    scale = math.sqrt(signal_power / max(noise_power * target_ratio, 1.0e-30))
    noisy = clean + scale * noise
    achieved = 10.0 * math.log10(signal_power / max(float(np.linalg.norm(scale * noise) ** 2), 1.0e-30))
    return noisy.astype(np.complex128), float(achieved)


def initial_tracking_coefficients(K: int) -> np.ndarray:
    amplitudes = np.exp(-0.32 * np.arange(int(K), dtype=np.float64))
    phases = np.asarray([0.0, 0.55, -0.70, 0.95, -1.10], dtype=np.float64)[: int(K)]
    coeffs = amplitudes * np.exp(1j * phases)
    return coeffs.astype(np.complex128)


def coupling_angle(time_fraction: float, cfg: dict) -> float:
    center = float(cfg["event_center_fraction"])
    width = float(cfg["event_width_fraction"])
    max_angle = float(cfg["max_mixing_angle_rad"])
    if width <= 0.0:
        raise ValueError("coupling.event_width_fraction must be positive")
    return max_angle * math.exp(-0.5 * ((time_fraction - center) / width) ** 2)


def apply_known_coupling(base: np.ndarray, time_fraction: float, cfg: dict) -> tuple[np.ndarray, float]:
    coeffs = np.asarray(base, dtype=np.complex128).copy()
    pair = [int(v) - 1 for v in cfg["mode_pair_1based"]]
    if len(pair) != 2 or min(pair) < 0 or max(pair) >= coeffs.size:
        raise ValueError("Invalid coupling mode pair")
    i, j = pair
    theta = coupling_angle(time_fraction, cfg)
    phase = float(cfg["coupling_phase_rad"])
    ci, cj = coeffs[i], coeffs[j]
    coeffs[i] = math.cos(theta) * ci - np.exp(-1j * phase) * math.sin(theta) * cj
    coeffs[j] = np.exp(1j * phase) * math.sin(theta) * ci + math.cos(theta) * cj
    return coeffs, float(theta)


def decompose_coefficients(basis: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    coefficients, _, _, _ = np.linalg.lstsq(
        np.asarray(basis, dtype=np.complex128),
        np.asarray(pressure, dtype=np.complex128),
        rcond=None,
    )
    return np.asarray(coefficients, dtype=np.complex128).reshape(-1)


def coefficient_relative_error(true: np.ndarray, estimate: np.ndarray) -> float:
    numerator = float(np.linalg.norm(np.asarray(estimate) - np.asarray(true)))
    denominator = max(float(np.linalg.norm(true)), 1.0e-15)
    return numerator / denominator


def energy_correlation(true_series: np.ndarray, estimate_series: np.ndarray, mode_indices: list[int]) -> float:
    correlations: list[float] = []
    for index in mode_indices:
        true_energy = np.abs(true_series[:, index]) ** 2
        est_energy = np.abs(estimate_series[:, index]) ** 2
        if np.std(true_energy) <= 1.0e-12 or np.std(est_energy) <= 1.0e-12:
            continue
        correlations.append(float(np.corrcoef(true_energy, est_energy)[0, 1]))
    if not correlations:
        return float("nan")
    return float(np.mean(correlations))


def single_mode_leakage_metrics(basis: np.ndarray, oracle_basis: np.ndarray) -> tuple[float, float]:
    """Average off-mode energy fraction and cross-mode rejection over true single modes."""
    K = oracle_basis.shape[1]
    leakages: list[float] = []
    rejections: list[float] = []
    for mode_index in range(K):
        pressure = oracle_basis[:, mode_index]
        estimate = decompose_coefficients(basis, pressure)
        power = np.abs(estimate) ** 2
        desired = float(power[mode_index])
        off = float(np.sum(power) - desired)
        leakage = off / max(float(np.sum(power)), 1.0e-30)
        rejection = 10.0 * math.log10(max(desired, 1.0e-30) / max(off, 1.0e-30))
        leakages.append(leakage)
        rejections.append(rejection)
    return float(np.mean(leakages)), float(np.mean(rejections))


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def save_ssp_figure(path: Path, depths: np.ndarray, times_s: np.ndarray, ssp_matrix: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    im = ax.imshow(
        ssp_matrix.T,
        aspect="auto",
        origin="upper",
        extent=[float(times_s[0]), float(times_s[-1]), float(depths[-1]), float(depths[0])],
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Depth (m)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Sound speed (m/s)")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_k_tracking_figure(path: Path, rows: list[dict], K: int) -> None:
    subset = [r for r in rows if int(r["K"]) == int(K)]
    if not subset:
        return
    times = sorted({float(r["time_s"]) for r in subset})
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for mode in range(1, K + 1):
        true_values = []
        est_values = []
        for t in times:
            match = [r for r in subset if float(r["time_s"]) == t]
            true_values.append(float(match[0][f"true_k{mode}"]))
            est_values.append(float(np.mean([float(r[f"est_k{mode}"]) for r in match])))
        ax.plot(times, true_values, linestyle="-", label=f"Mode {mode} oracle")
        ax.plot(times, est_values, linestyle="--", marker="o", markersize=3, label=f"Mode {mode} estimated")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Horizontal wavenumber (rad/m)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_basis_mac_figure(path: Path, rows: list[dict], K: int) -> None:
    subset = [r for r in rows if int(r["K"]) == int(K)]
    if not subset:
        return
    times = sorted({float(r["time_s"]) for r in subset})
    fixed = []
    estimated = []
    for t in times:
        match = [r for r in subset if float(r["time_s"]) == t]
        fixed.append(float(match[0]["fixed_basis_mean_mac"]))
        estimated.append(float(np.mean([float(r["estimated_basis_mean_mac"]) for r in match])))
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(times, fixed, marker="s", linestyle="--", label="Fixed basis")
    ax.plot(times, estimated, marker="o", linestyle="-", label="Estimated dynamic basis")
    ax.axhline(1.0, linestyle=":", label="Oracle dynamic basis")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean MAC to oracle basis")
    ax.set_ylim(0.0, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_coefficient_energy_figure(
    path: Path,
    tracking_rows: list[dict],
    K: int,
    scenario: str,
    mode_indices: list[int],
) -> None:
    subset = [r for r in tracking_rows if int(r["K"]) == int(K) and r["scenario"] == scenario]
    if not subset:
        return
    times = sorted({float(r["time_s"]) for r in subset})
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for mode in mode_indices:
        true_energy = []
        fixed_energy = []
        oracle_energy = []
        estimated_energy = []
        for t in times:
            match = [r for r in subset if float(r["time_s"]) == t]
            true_energy.append(float(match[0][f"true_energy_m{mode}"]))
            fixed_energy.append(float(match[0][f"fixed_energy_m{mode}"]))
            oracle_energy.append(float(match[0][f"oracle_energy_m{mode}"]))
            estimated_energy.append(float(np.mean([float(r[f"estimated_energy_m{mode}"]) for r in match])))
        ax.plot(times, true_energy, linestyle="-", linewidth=2.0, label=f"True M{mode}")
        ax.plot(times, fixed_energy, linestyle=":", label=f"Fixed M{mode}")
        ax.plot(times, oracle_energy, linestyle="--", label=f"Oracle dyn M{mode}")
        ax.plot(times, estimated_energy, marker="o", markersize=3, linestyle="-.", label=f"Estimated dyn M{mode}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Modal coefficient energy")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_summary_bar(path: Path, summary_rows: list[dict], K: int) -> None:
    rows = [r for r in summary_rows if int(r["K"]) == int(K)]
    if not rows:
        return
    labels = [f"{r['scenario']}\n{r['basis']}" for r in rows]
    values = [float(r["mean_coefficient_relative_error"]) for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Mean coefficient relative error")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------
def main() -> None:
    cfg = load_config()
    output_root = (ROOT / cfg.get("output_root", str(DEFAULT_OUTPUT.relative_to(ROOT)))).resolve()
    dirs = ensure_dirs(output_root)

    frequency = float(cfg["frequency_hz"])
    omega = 2.0 * math.pi * frequency
    water_depth = float(cfg["water_depth_m"])
    dz = float(cfg["dz_m"])
    depths = np.arange(0.0, water_depth + 0.5 * dz, dz, dtype=np.float64)
    if not math.isclose(float(depths[-1]), water_depth, abs_tol=1.0e-12):
        raise ValueError("water_depth_m must be divisible by dz_m")

    receiver_cfg = cfg.get("receiver_array", {"start_depth_m": 1.0, "end_depth_m": water_depth, "spacing_m": 1.0})
    receiver_depths = np.arange(
        float(receiver_cfg["start_depth_m"]),
        float(receiver_cfg["end_depth_m"]) + 0.5 * float(receiver_cfg["spacing_m"]),
        float(receiver_cfg["spacing_m"]),
        dtype=np.float64,
    )

    time_points = int(cfg["time_points"])
    times_s = np.linspace(0.0, float(cfg["time_span_s"]), time_points, dtype=np.float64)
    fractions = np.linspace(0.0, 1.0, time_points, dtype=np.float64)
    est_cfg = cfg["estimation"]
    coupling_cfg = cfg["coupling"]
    tracking_cfg = cfg["tracking"]
    bottom_cfg = cfg["forward_bottom"]

    print("=" * 96)
    print("UNIFIED DYNAMIC LOCAL-BASIS + COUPLING-SEPARATION SUITE")
    print("=" * 96)
    print(f"frequency        : {frequency:.3f} Hz")
    print(f"water depth      : {water_depth:.3f} m")
    print(f"receivers        : {receiver_depths.size}")
    print(f"K values         : {cfg['K_values']}")
    print(f"time points      : {time_points}")
    print(f"estimation SNR   : {float(est_cfg['snr_db']):.1f} dB")
    print(f"estimator repeats: {int(est_cfg['repeats'])}")
    print("bottom params    : forward/oracle root generation ONLY")
    print("Proposed input   : Y, K, f, array, H, current SSP, seed")

    # Precompute time-varying SSPs.
    ssp_matrix = np.zeros((time_points, depths.size), dtype=np.float64)
    eta_values = np.zeros(time_points, dtype=np.float64)
    for ti, fraction in enumerate(fractions):
        ssp_matrix[ti], eta_values[ti] = make_dynamic_ssp(depths, float(fraction), cfg["ssp_dynamic"])
    save_ssp_figure(dirs["figures"] / "01_ssp_evolution.png", depths, times_s, ssp_matrix)

    root_rows: list[dict] = []
    estimation_rows: list[dict] = []
    tracking_rows: list[dict] = []
    leakage_rows: list[dict] = []
    summary_rows: list[dict] = []

    # Store arrays for reproducibility.
    npz_payload: dict[str, np.ndarray] = {
        "depths_m": depths,
        "receiver_depths_m": receiver_depths,
        "times_s": times_s,
        "time_fraction": fractions,
        "ssp_matrix_m_s": ssp_matrix,
        "thermocline_displacement_m": eta_values,
    }

    for K in [int(value) for value in cfg["K_values"]]:
        print("\n" + "-" * 96)
        print(f"K = {K}")
        print("-" * 96)

        oracle_k = np.zeros((time_points, K), dtype=np.float64)
        oracle_basis = np.zeros((time_points, receiver_depths.size, K), dtype=np.complex128)

        # Controlled truth for every environmental time point.
        for ti in range(time_points):
            oracle_k[ti] = find_oracle_roots(
                K=K,
                omega=omega,
                ssp=ssp_matrix[ti],
                dz=dz,
                bottom=bottom_cfg,
                scan_points=int(cfg["oracle_root_scan_points"]),
            )
            oracle_basis[ti] = build_water_column_basis(
                oracle_k[ti], omega, ssp_matrix[ti], dz, receiver_depths, water_depth
            )
            row = {
                "K": K,
                "time_index": ti,
                "time_s": float(times_s[ti]),
                "time_fraction": float(fractions[ti]),
                "thermocline_displacement_m": float(eta_values[ti]),
            }
            for mode in range(K):
                row[f"true_k{mode + 1}"] = float(oracle_k[ti, mode])
            root_rows.append(row)

        fixed_basis = oracle_basis[0].copy()
        npz_payload[f"oracle_k_K{K}"] = oracle_k
        npz_payload[f"oracle_basis_K{K}"] = oracle_basis
        npz_payload[f"fixed_basis_K{K}"] = fixed_basis

        estimated_basis_all = np.zeros(
            (int(est_cfg["repeats"]), time_points, receiver_depths.size, K), dtype=np.complex128
        )
        estimated_k_all = np.zeros((int(est_cfg["repeats"]), time_points, K), dtype=np.float64)

        # Full Proposed k-estimation chain for each time/repeat.
        for repeat in range(int(est_cfg["repeats"])):
            print(f"  Estimator repeat {repeat + 1}/{int(est_cfg['repeats'])}")
            for ti in range(time_points):
                coefficient_seed = int(est_cfg["base_seed"]) + 100000 * K + 1000 * repeat + 10 * ti + 1
                noise_seed = int(est_cfg["base_seed"]) + 100000 * K + 1000 * repeat + 10 * ti + 2
                optimization_seed = int(est_cfg["base_seed"]) + 100000 * K + 1000 * repeat + 10 * ti + 3

                C = random_coefficients(K, int(est_cfg["snapshots"]), coefficient_seed)
                clean_Y = oracle_basis[ti] @ C
                noisy_Y, achieved_snr = add_complex_noise(clean_Y, float(est_cfg["snr_db"]), noise_seed)
                metadata = DynamicMetadata(
                    K=K,
                    frequency=frequency,
                    array_geometry=receiver_depths.copy(),
                    water_depth=water_depth,
                    SSP=ssp_matrix[ti].copy(),
                )

                start = time.perf_counter()
                estimate, coarse, accepted, coarse_scores, refined_scores = dynamic_topn_refined(
                    noisy_Y, metadata, optimization_seed, est_cfg, dz
                )
                runtime = time.perf_counter() - start
                estimated_k_all[repeat, ti] = estimate
                estimated_basis_all[repeat, ti] = build_water_column_basis(
                    estimate, omega, ssp_matrix[ti], dz, receiver_depths, water_depth
                )

                k_rmse = float(np.sqrt(np.mean((estimate - oracle_k[ti]) ** 2)))
                fixed_macs = [column_mac(oracle_basis[ti, :, m], fixed_basis[:, m]) for m in range(K)]
                estimated_macs = [
                    column_mac(oracle_basis[ti, :, m], estimated_basis_all[repeat, ti, :, m])
                    for m in range(K)
                ]
                row = {
                    "K": K,
                    "repeat": repeat,
                    "time_index": ti,
                    "time_s": float(times_s[ti]),
                    "time_fraction": float(fractions[ti]),
                    "thermocline_displacement_m": float(eta_values[ti]),
                    "achieved_estimation_snr_db": float(achieved_snr),
                    "k_rmse_rad_m": k_rmse,
                    "fixed_basis_mean_mac": float(np.mean(fixed_macs)),
                    "estimated_basis_mean_mac": float(np.mean(estimated_macs)),
                    "estimated_basis_min_mac": float(np.min(estimated_macs)),
                    "refinement_accepted": int(accepted),
                    "coarse_mean_residual": float(coarse_scores[0]),
                    "coarse_worst_residual": float(coarse_scores[1]),
                    "refined_mean_residual": float(refined_scores[0]),
                    "refined_worst_residual": float(refined_scores[1]),
                    "runtime_seconds": float(runtime),
                }
                for mode in range(K):
                    row[f"true_k{mode + 1}"] = float(oracle_k[ti, mode])
                    row[f"coarse_k{mode + 1}"] = float(coarse[mode])
                    row[f"est_k{mode + 1}"] = float(estimate[mode])
                    row[f"basis_mac_m{mode + 1}"] = float(estimated_macs[mode])
                estimation_rows.append(row)

                print(
                    f"    t={times_s[ti]:7.1f}s | eta={eta_values[ti]:+5.2f}m | "
                    f"kRMSE={k_rmse:.3e} | meanMAC={np.mean(estimated_macs):.5f} | "
                    f"gate={'R' if accepted else 'C'} | {runtime:.2f}s"
                )

        npz_payload[f"estimated_k_K{K}"] = estimated_k_all
        npz_payload[f"estimated_basis_K{K}"] = estimated_basis_all

        # Tracking coefficients: same bases are reused for no-coupling and true-coupling scenarios.
        base_coeff = initial_tracking_coefficients(K)
        true_series_by_scenario: dict[str, np.ndarray] = {}
        for scenario in ("no_coupling", "known_coupling"):
            true_series = np.zeros((time_points, K), dtype=np.complex128)
            theta_series = np.zeros(time_points, dtype=np.float64)
            for ti, fraction in enumerate(fractions):
                if scenario == "no_coupling":
                    true_series[ti] = base_coeff
                    theta_series[ti] = 0.0
                else:
                    true_series[ti], theta_series[ti] = apply_known_coupling(
                        base_coeff, float(fraction), coupling_cfg
                    )
            true_series_by_scenario[scenario] = true_series
            npz_payload[f"true_coefficients_{scenario}_K{K}"] = true_series
            npz_payload[f"coupling_angle_{scenario}_K{K}"] = theta_series

            for repeat in range(int(est_cfg["repeats"])):
                fixed_est_series = np.zeros((time_points, K), dtype=np.complex128)
                oracle_est_series = np.zeros((time_points, K), dtype=np.complex128)
                dynamic_est_series = np.zeros((time_points, K), dtype=np.complex128)

                for ti in range(time_points):
                    tracking_seed = (
                        int(tracking_cfg["base_seed"])
                        + 100000 * K
                        + 10000 * (0 if scenario == "no_coupling" else 1)
                        + 1000 * repeat
                        + ti
                    )
                    clean_pressure = oracle_basis[ti] @ true_series[ti]
                    tracking_snr = tracking_cfg.get("snr_db", None)
                    if tracking_snr is not None:
                        tracking_snr = float(tracking_snr)
                    pressure, achieved_tracking_snr = add_complex_noise(
                        clean_pressure, tracking_snr, tracking_seed
                    )

                    fixed_est = decompose_coefficients(fixed_basis, pressure)
                    oracle_est = decompose_coefficients(oracle_basis[ti], pressure)
                    dynamic_est = decompose_coefficients(estimated_basis_all[repeat, ti], pressure)
                    fixed_est_series[ti] = fixed_est
                    oracle_est_series[ti] = oracle_est
                    dynamic_est_series[ti] = dynamic_est

                    fixed_leakage, fixed_cmr = single_mode_leakage_metrics(fixed_basis, oracle_basis[ti])
                    oracle_leakage, oracle_cmr = single_mode_leakage_metrics(oracle_basis[ti], oracle_basis[ti])
                    dynamic_leakage, dynamic_cmr = single_mode_leakage_metrics(
                        estimated_basis_all[repeat, ti], oracle_basis[ti]
                    )

                    row = {
                        "K": K,
                        "repeat": repeat,
                        "scenario": scenario,
                        "time_index": ti,
                        "time_s": float(times_s[ti]),
                        "time_fraction": float(fractions[ti]),
                        "thermocline_displacement_m": float(eta_values[ti]),
                        "coupling_angle_rad": float(theta_series[ti]),
                        "achieved_tracking_snr_db": float(achieved_tracking_snr),
                        "fixed_coeff_rel_error": coefficient_relative_error(true_series[ti], fixed_est),
                        "oracle_coeff_rel_error": coefficient_relative_error(true_series[ti], oracle_est),
                        "estimated_coeff_rel_error": coefficient_relative_error(true_series[ti], dynamic_est),
                    }
                    for mode in range(K):
                        row[f"true_energy_m{mode + 1}"] = float(abs(true_series[ti, mode]) ** 2)
                        row[f"fixed_energy_m{mode + 1}"] = float(abs(fixed_est[mode]) ** 2)
                        row[f"oracle_energy_m{mode + 1}"] = float(abs(oracle_est[mode]) ** 2)
                        row[f"estimated_energy_m{mode + 1}"] = float(abs(dynamic_est[mode]) ** 2)
                    tracking_rows.append(row)

                    leakage_rows.extend(
                        [
                            {
                                "K": K,
                                "repeat": repeat,
                                "scenario": scenario,
                                "time_index": ti,
                                "time_s": float(times_s[ti]),
                                "basis": "fixed",
                                "mean_offmode_energy_fraction": fixed_leakage,
                                "mean_cross_mode_rejection_db": fixed_cmr,
                            },
                            {
                                "K": K,
                                "repeat": repeat,
                                "scenario": scenario,
                                "time_index": ti,
                                "time_s": float(times_s[ti]),
                                "basis": "oracle_dynamic",
                                "mean_offmode_energy_fraction": oracle_leakage,
                                "mean_cross_mode_rejection_db": oracle_cmr,
                            },
                            {
                                "K": K,
                                "repeat": repeat,
                                "scenario": scenario,
                                "time_index": ti,
                                "time_s": float(times_s[ti]),
                                "basis": "estimated_dynamic",
                                "mean_offmode_energy_fraction": dynamic_leakage,
                                "mean_cross_mode_rejection_db": dynamic_cmr,
                            },
                        ]
                    )

                # Summary rows per repeat/scenario/basis.
                bases = {
                    "fixed": fixed_est_series,
                    "oracle_dynamic": oracle_est_series,
                    "estimated_dynamic": dynamic_est_series,
                }
                coupling_modes = [int(v) - 1 for v in coupling_cfg["mode_pair_1based"]]
                for basis_name, estimates in bases.items():
                    rel_errors = [
                        coefficient_relative_error(true_series[ti], estimates[ti])
                        for ti in range(time_points)
                    ]
                    if scenario == "no_coupling":
                        mean_vector = np.mean(estimates, axis=0, keepdims=True)
                        spurious_variation = float(
                            np.linalg.norm(estimates - mean_vector)
                            / max(np.linalg.norm(np.tile(base_coeff, (time_points, 1))), 1.0e-15)
                        )
                    else:
                        spurious_variation = float("nan")
                    energy_corr = energy_correlation(true_series, estimates, coupling_modes)
                    matching_leak = [
                        r for r in leakage_rows
                        if int(r["K"]) == K
                        and int(r["repeat"]) == repeat
                        and r["scenario"] == scenario
                        and r["basis"] == basis_name
                    ]
                    summary_rows.append(
                        {
                            "K": K,
                            "repeat": repeat,
                            "scenario": scenario,
                            "basis": basis_name,
                            "mean_coefficient_relative_error": float(np.mean(rel_errors)),
                            "max_coefficient_relative_error": float(np.max(rel_errors)),
                            "spurious_temporal_variation_no_coupling": spurious_variation,
                            "coupled_mode_energy_correlation": energy_corr,
                            "mean_offmode_energy_fraction": float(
                                np.mean([float(r["mean_offmode_energy_fraction"]) for r in matching_leak])
                            ),
                            "mean_cross_mode_rejection_db": float(
                                np.mean([float(r["mean_cross_mode_rejection_db"]) for r in matching_leak])
                            ),
                        }
                    )

    # Save all raw results.
    write_csv(dirs["csv"] / "dynamic_oracle_roots.csv", root_rows)
    write_csv(dirs["csv"] / "dynamic_estimation_metrics.csv", estimation_rows)
    write_csv(dirs["csv"] / "dynamic_tracking_metrics.csv", tracking_rows)
    write_csv(dirs["csv"] / "dynamic_cross_mode_leakage.csv", leakage_rows)
    write_csv(dirs["csv"] / "dynamic_summary_by_repeat.csv", summary_rows)
    np.savez_compressed(dirs["arrays"] / "dynamic_suite_arrays.npz", **npz_payload)

    # Aggregate summary across repeats.
    aggregate_rows: list[dict] = []
    keys = sorted({(int(r["K"]), r["scenario"], r["basis"]) for r in summary_rows})
    for K, scenario, basis in keys:
        rows = [r for r in summary_rows if int(r["K"]) == K and r["scenario"] == scenario and r["basis"] == basis]
        aggregate_rows.append(
            {
                "K": K,
                "scenario": scenario,
                "basis": basis,
                "mean_coefficient_relative_error": float(np.mean([float(r["mean_coefficient_relative_error"]) for r in rows])),
                "max_coefficient_relative_error": float(np.max([float(r["max_coefficient_relative_error"]) for r in rows])),
                "mean_spurious_temporal_variation_no_coupling": float(
                    np.nanmean([float(r["spurious_temporal_variation_no_coupling"]) for r in rows])
                ) if scenario == "no_coupling" else float("nan"),
                "mean_coupled_mode_energy_correlation": float(
                    np.nanmean([float(r["coupled_mode_energy_correlation"]) for r in rows])
                ) if scenario == "known_coupling" else float("nan"),
                "mean_offmode_energy_fraction": float(np.mean([float(r["mean_offmode_energy_fraction"]) for r in rows])),
                "mean_cross_mode_rejection_db": float(
                    10.0 * math.log10(
                        max(1.0 - float(np.mean([float(r["mean_offmode_energy_fraction"]) for r in rows])), 1.0e-30)
                        / max(float(np.mean([float(r["mean_offmode_energy_fraction"]) for r in rows])), 1.0e-30)
                    )
                ),
            }
        )
    write_csv(dirs["csv"] / "dynamic_summary.csv", aggregate_rows)

    # Figures focus on K=3, with K=5 carried in CSV/summary as robustness.
    if 3 in [int(v) for v in cfg["K_values"]]:
        save_k_tracking_figure(dirs["figures"] / "02_k_tracking_K3.png", estimation_rows, 3)
        save_basis_mac_figure(dirs["figures"] / "03_basis_mac_K3.png", estimation_rows, 3)
        save_coefficient_energy_figure(
            dirs["figures"] / "04_no_coupling_modal_energy_K3.png",
            tracking_rows,
            3,
            "no_coupling",
            [1, 2, 3],
        )
        save_coefficient_energy_figure(
            dirs["figures"] / "05_known_coupling_modal_energy_K3.png",
            tracking_rows,
            3,
            "known_coupling",
            [1, 2],
        )
        save_summary_bar(dirs["figures"] / "06_summary_coefficient_error_K3.png", aggregate_rows, 3)

    # Human-readable summary.
    estimation_summary = {}
    for K in [int(v) for v in cfg["K_values"]]:
        rows = [r for r in estimation_rows if int(r["K"]) == K]
        estimation_summary[f"K{K}"] = {
            "mean_k_rmse_rad_m": float(np.mean([float(r["k_rmse_rad_m"]) for r in rows])),
            "max_k_rmse_rad_m": float(np.max([float(r["k_rmse_rad_m"]) for r in rows])),
            "mean_estimated_basis_mac": float(np.mean([float(r["estimated_basis_mean_mac"]) for r in rows])),
            "min_estimated_basis_mac": float(np.min([float(r["estimated_basis_min_mac"]) for r in rows])),
            "refinement_acceptance_rate": float(np.mean([int(r["refinement_accepted"]) for r in rows])),
            "mean_runtime_seconds": float(np.mean([float(r["runtime_seconds"]) for r in rows])),
        }

    payload = {
        "experiment": cfg.get("experiment", "dynamic_local_basis_coupling_suite"),
        "label": "controlled synthetic dynamic-environment validation; not measured-data validation",
        "frequency_hz": frequency,
        "water_depth_m": water_depth,
        "K_values": [int(v) for v in cfg["K_values"]],
        "time_points": time_points,
        "estimation_snr_db": float(est_cfg["snr_db"]),
        "estimator_repeats": int(est_cfg["repeats"]),
        "forward_bottom_policy": "bottom parameters used only for oracle/forward root generation",
        "estimated_basis_policy": "current SSP + Proposed k_hat; no bottom parameters",
        "estimation_summary": estimation_summary,
        "aggregate_decomposition_summary": aggregate_rows,
    }
    (dirs["root"] / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# Dynamic Local-Mode Basis and Coupling-Separation Suite",
        "",
        "Controlled synthetic proof-of-concept. This is not measured SWARM-95 validation.",
        "",
        "## Estimation / basis summary",
    ]
    for key, value in estimation_summary.items():
        report_lines.append(
            f"- {key}: mean k-RMSE={value['mean_k_rmse_rad_m']:.6e} rad/m; "
            f"mean estimated-basis MAC={value['mean_estimated_basis_mac']:.6f}; "
            f"min MAC={value['min_estimated_basis_mac']:.6f}; "
            f"gate acceptance={value['refinement_acceptance_rate']:.3f}."
        )
    report_lines += ["", "## Decomposition summary", ""]
    for row in aggregate_rows:
        report_lines.append(
            f"- K={row['K']}, {row['scenario']}, {row['basis']}: "
            f"mean coeff rel.err={row['mean_coefficient_relative_error']:.6e}; "
            f"off-mode fraction={row['mean_offmode_energy_fraction']:.6e}; "
            f"CMR={row['mean_cross_mode_rejection_db']:.2f} dB; "
            f"spurious(no-coupling)={row['mean_spurious_temporal_variation_no_coupling']}; "
            f"coupled-energy corr={row['mean_coupled_mode_energy_correlation']}."
        )
    report_lines += [
        "",
        "## Interpretation rule",
        "",
        "A dynamic basis is useful only if it reduces coefficient error / spurious variation relative to the fixed basis while the known-coupling case still tracks the imposed modal-energy exchange. Oracle dynamic basis is an upper reference, not an estimator.",
        "",
        "The forward bottom model is used only to construct controlled oracle roots. It is never supplied to the Proposed estimator or to local-mode recovery from estimated k.",
    ]
    (dirs["root"] / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("\n" + "=" * 96)
    print("COMPLETE")
    print(f"outputs = {dirs['root']}")
    for K, stats in estimation_summary.items():
        print(
            f"{K}: mean k-RMSE={stats['mean_k_rmse_rad_m']:.3e}, "
            f"mean basis MAC={stats['mean_estimated_basis_mac']:.6f}, "
            f"min basis MAC={stats['min_estimated_basis_mac']:.6f}"
        )
    print("\nAggregate decomposition summary:")
    for row in aggregate_rows:
        print(
            f"  K={row['K']} | {row['scenario']:14s} | {row['basis']:17s} | "
            f"coeff.err={row['mean_coefficient_relative_error']:.3e} | "
            f"leak={row['mean_offmode_energy_fraction']:.3e} | "
            f"CMR={row['mean_cross_mode_rejection_db']:.2f} dB"
        )
    print("=" * 96)


if __name__ == "__main__":
    main()
