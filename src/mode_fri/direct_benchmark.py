"""K=3, 20-trial benchmark for low-SNR direct modal-wavenumber estimation.

This stage combines the validated data-only Top-N beam search with continuous
off-grid refinement.  The Top-N selector preserves several plausible coarse mode
hypotheses and chooses one using only multi-snapshot projection residuals.  The
selected coarse wavenumbers are then refined continuously using the existing
seabed-parameter-free ModeFRINET data-fitting path.

No seabed acoustic parameter, physical root, root-derived cell, Robin residual,
or KRAKEN truth is available to any estimator.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from mode_fri import direct_estimator
from mode_fri.backends import baseline_comparison_final as baseline


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "direct_wavenumber_estimation.json"
PROTECTED_OUTPUT = (ROOT / "outputs" / "final_paper").resolve()


@dataclass(frozen=True)
class AllowedMetadata:
    K: int
    frequency: float
    array_geometry: np.ndarray
    water_depth: float
    SSP: np.ndarray


@dataclass(frozen=True)
class MethodResult:
    method: str
    estimated_k: np.ndarray
    runtime_seconds: float
    optimization_seed: int
    input_y_hash: str
    exception: str


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if int(config["K"]) != 3:
        raise ValueError("The accepted first-round scope is K=3 only")
    if tuple(float(value) for value in config["snr_db"]) != (-5.0, 0.0, 5.0, 10.0, 15.0):
        raise ValueError("The accepted K=3 SNR set has changed")
    if int(config["trials_per_snr"]) != 20:
        raise ValueError("The current Monte Carlo scope is 20 trials per SNR")
    if int(config["number_of_atoms"]) != direct_estimator.NUMBER_OF_ATOMS:
        raise ValueError("Estimator and experiment atom counts disagree")
    if int(config.get("top_n_hypotheses", 0)) != 5:
        raise ValueError("The accepted Top-N scope is 5 coarse hypotheses")
    if int(config.get("beam_branch_factor", 0)) != 6:
        raise ValueError("The accepted beam branch factor is 6")
    if int(config.get("beam_width", 0)) != 24:
        raise ValueError("The accepted beam width is 24")
    return config


def _output_root(config: dict) -> Path:
    output = (ROOT / str(config["output_root"])).resolve()
    if output == PROTECTED_OUTPUT or PROTECTED_OUTPUT in output.parents:
        raise RuntimeError("Direct experiment output must not enter outputs/final_paper")
    return output


def hash_Y(Y: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(Y, dtype=np.complex128))
    digest = hashlib.sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(str(canonical.dtype).encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _manifest_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        return digest.hexdigest()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        stat = item.stat()
        digest.update(str(item.relative_to(path)).replace("\\", "/").encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _ssp_for_water_grid(water_depth: float) -> tuple[np.ndarray, np.ndarray]:
    depths = np.arange(0.0, float(water_depth) + 0.5, 1.0, dtype=np.float64)
    if not math.isclose(float(depths[-1]), float(water_depth), abs_tol=1.0e-12):
        raise ValueError("The existing water-column backend requires integer-metre depth")
    return depths, np.asarray(baseline.sound_speed_profile(depths), dtype=np.float64)


def prepare_forward_observation(K: int, snapshots: int, water_depth: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create the KRAKEN-simulated observation template without exporting truth.

    KRAKEN modal quantities are confined to this forward-simulation function.
    Only the clean pressure matrix and allowed receiver/SSP metadata leave it.
    """
    old_forward_modes = int(baseline.NUMBER_OF_FORWARD_MODES)
    old_target_modes = int(baseline.NUMBER_OF_TARGET_MODES)
    try:
        baseline.NUMBER_OF_FORWARD_MODES = int(K)
        baseline.NUMBER_OF_TARGET_MODES = int(K)
        environment = baseline.load_kraken_environment()
        forward = baseline.generate_clean_moving_source_field(
            receiver_modes=environment["receiver_modes"],
            source_modes=environment["source_modes"],
            complex_k=environment["complex_k"],
        )
        clean_Y = np.asarray(forward["clean_receiver"], dtype=np.complex128)[:, : int(snapshots)].copy()
        receiver_depths = np.asarray(environment["receiver_depths"], dtype=np.float64).copy()
    finally:
        baseline.NUMBER_OF_FORWARD_MODES = old_forward_modes
        baseline.NUMBER_OF_TARGET_MODES = old_target_modes
    _, ssp = _ssp_for_water_grid(water_depth)
    if clean_Y.shape != (receiver_depths.size, int(snapshots)):
        raise RuntimeError(f"Unexpected clean observation shape {clean_Y.shape}")
    return clean_Y, receiver_depths, ssp


def add_shared_noise(clean_Y: np.ndarray, snr_db: float, data_seed: int) -> tuple[np.ndarray, float]:
    noisy, achieved = baseline.add_noise(np.asarray(clean_Y, dtype=np.complex128), float(snr_db), int(data_seed))
    return np.asarray(noisy, dtype=np.complex128), float(achieved)


def _fd_svd(Y: np.ndarray, metadata: AllowedMetadata, _: int) -> np.ndarray:
    if not math.isclose(metadata.frequency, float(baseline.FREQUENCY_HZ), abs_tol=1.0e-12):
        raise ValueError("FD-SVD backend frequency differs from the experiment")
    old_target_modes = int(baseline.NUMBER_OF_TARGET_MODES)
    try:
        baseline.NUMBER_OF_TARGET_MODES = int(metadata.K)
        result = baseline.fd_svd_paper(np.asarray(Y, dtype=np.complex128), metadata.array_geometry)
    finally:
        baseline.NUMBER_OF_TARGET_MODES = old_target_modes
    return np.sort(np.asarray(result["estimated_k"], dtype=np.float64))[::-1]


def _grid_gomp(Y: np.ndarray, metadata: AllowedMetadata, _: int, grid_size: int) -> np.ndarray:
    omega = 2.0 * math.pi * float(metadata.frequency)
    grid_min = direct_estimator.SEARCH_LOWER_FRACTION * omega / float(np.max(metadata.SSP))
    grid_max = direct_estimator.SEARCH_UPPER_FRACTION * omega / float(np.min(metadata.SSP))
    old_values = (
        int(baseline.NUMBER_OF_TARGET_MODES),
        int(baseline.GRID_SIZE),
        float(baseline.GRID_MIN),
        float(baseline.GRID_MAX),
    )
    try:
        baseline.NUMBER_OF_TARGET_MODES = int(metadata.K)
        baseline.GRID_SIZE = int(grid_size)
        baseline.GRID_MIN = float(grid_min)
        baseline.GRID_MAX = float(grid_max)
        normalized = baseline.normalize_snapshot_columns(np.asarray(Y, dtype=np.complex128))
        result = baseline.grid_512_gomp(normalized)
    finally:
        (
            baseline.NUMBER_OF_TARGET_MODES,
            baseline.GRID_SIZE,
            baseline.GRID_MIN,
            baseline.GRID_MAX,
        ) = old_values
    return np.sort(np.asarray(result["estimated_k"], dtype=np.float64))[::-1]


def _legacy(Y: np.ndarray, metadata: AllowedMetadata, seed: int) -> np.ndarray:
    return direct_estimator.estimate_legacy(
        Y,
        metadata.K,
        metadata.frequency,
        metadata.array_geometry,
        metadata.water_depth,
        metadata.SSP,
        seed,
    )


def _direct_ms(Y: np.ndarray, metadata: AllowedMetadata, seed: int) -> np.ndarray:
    return direct_estimator.estimate(
        Y,
        metadata.K,
        metadata.frequency,
        metadata.array_geometry,
        metadata.water_depth,
        metadata.SSP,
        seed,
    )


def _grid_refined(
    Y: np.ndarray,
    metadata: AllowedMetadata,
    seed: int,
    grid_size: int,
) -> np.ndarray:
    """Grid-GOMP coarse estimate followed by seabed-free continuous refinement."""
    coarse_k = _grid_gomp(Y, metadata, seed, grid_size)
    return direct_estimator.estimate_from_coarse(
        Y,
        metadata.K,
        metadata.frequency,
        metadata.array_geometry,
        metadata.water_depth,
        metadata.SSP,
        seed,
        coarse_k,
    )


def _grid_dictionary(
    metadata: AllowedMetadata,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the exact water-column grid dictionary used by Grid-GOMP.

    Only frequency, SSP and the water-column inverse model are used.  No
    seabed parameters or modal roots enter this function.
    """
    omega = 2.0 * math.pi * float(metadata.frequency)
    grid_min = direct_estimator.SEARCH_LOWER_FRACTION * omega / float(np.max(metadata.SSP))
    grid_max = direct_estimator.SEARCH_UPPER_FRACTION * omega / float(np.min(metadata.SSP))

    old_values = (
        int(baseline.NUMBER_OF_TARGET_MODES),
        int(baseline.GRID_SIZE),
        float(baseline.GRID_MIN),
        float(baseline.GRID_MAX),
    )
    try:
        baseline.NUMBER_OF_TARGET_MODES = int(metadata.K)
        baseline.GRID_SIZE = int(grid_size)
        baseline.GRID_MIN = float(grid_min)
        baseline.GRID_MAX = float(grid_max)
        grid = np.linspace(grid_max, grid_min, int(grid_size), dtype=np.float64)
        dictionary = np.asarray(
            baseline.EXP26.build_mode_matrix(grid),
            dtype=np.complex128,
        )
    finally:
        (
            baseline.NUMBER_OF_TARGET_MODES,
            baseline.GRID_SIZE,
            baseline.GRID_MIN,
            baseline.GRID_MAX,
        ) = old_values

    atom_norms = np.maximum(np.linalg.norm(dictionary, axis=0), 1.0e-15)
    dictionary = dictionary / atom_norms[None, :]
    return grid, dictionary


def _projection_snapshot_scores(
    Y: np.ndarray,
    dictionary: np.ndarray,
) -> tuple[float, float]:
    """Return mean and worst per-snapshot relative projection residuals."""
    observations = np.asarray(Y, dtype=np.complex128)
    atoms = np.asarray(dictionary, dtype=np.complex128)
    coefficients, _, _, _ = np.linalg.lstsq(atoms, observations, rcond=None)
    residual = observations - atoms @ coefficients
    numerator = np.sum(np.abs(residual) ** 2, axis=0)
    denominator = np.maximum(np.sum(np.abs(observations) ** 2, axis=0), 1.0e-30)
    ratios = numerator / denominator
    return float(np.mean(ratios)), float(np.max(ratios))


def _grid_beam_hypotheses(
    Y: np.ndarray,
    metadata: AllowedMetadata,
    grid_size: int,
    top_n: int,
    branch_factor: int,
    beam_width: int,
) -> list[np.ndarray]:
    """Generate several data-only K-mode hypotheses with beam GOMP.

    At every mode-selection stage each surviving state branches to its strongest
    correlation candidates.  States are ranked only by reconstruction residual.
    This preserves several plausible low-SNR mode combinations instead of
    committing irreversibly to the greedy first choice.
    """
    if top_n <= 0 or branch_factor <= 0 or beam_width <= 0:
        raise ValueError("top_n, branch_factor and beam_width must be positive")

    grid, dictionary = _grid_dictionary(metadata, grid_size)
    observations = baseline.normalize_snapshot_columns(np.asarray(Y, dtype=np.complex128))
    reference_energy = max(float(np.linalg.norm(observations) ** 2), 1.0e-30)
    minimum_separation = float(baseline.GRID_MIN_SEPARATION)

    # state = (canonical selected grid indices, residual, relative residual score)
    states: list[tuple[tuple[int, ...], np.ndarray, float]] = [
        (tuple(), observations.copy(), 1.0)
    ]

    for _ in range(int(metadata.K)):
        expanded: dict[tuple[int, ...], tuple[tuple[int, ...], np.ndarray, float]] = {}
        for selected_indices, residual, _ in states:
            correlations = dictionary.conj().T @ residual
            group_scores = np.linalg.norm(correlations, axis=1)

            if selected_indices:
                selected_k = grid[np.asarray(selected_indices, dtype=np.int64)]
                for grid_index, candidate_k in enumerate(grid):
                    if np.min(np.abs(selected_k - float(candidate_k))) < minimum_separation:
                        group_scores[grid_index] = -np.inf

            finite = np.flatnonzero(np.isfinite(group_scores))
            if finite.size == 0:
                continue
            order = finite[np.argsort(group_scores[finite])[::-1]]
            order = order[: min(int(branch_factor), order.size)]

            for grid_index in order:
                canonical = tuple(sorted((*selected_indices, int(grid_index))))
                selected_dictionary = dictionary[:, list(canonical)]
                coefficients, _, _, _ = np.linalg.lstsq(
                    selected_dictionary,
                    observations,
                    rcond=None,
                )
                new_residual = observations - selected_dictionary @ coefficients
                residual_score = float(
                    np.linalg.norm(new_residual) ** 2 / reference_energy
                )
                previous = expanded.get(canonical)
                if previous is None or residual_score < previous[2]:
                    expanded[canonical] = (canonical, new_residual, residual_score)

        if not expanded:
            raise RuntimeError("Beam-GOMP could not produce enough separated candidates")
        states = sorted(expanded.values(), key=lambda item: item[2])[: int(beam_width)]

    hypotheses: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for selected_indices, _, _ in states:
        candidate = np.sort(grid[np.asarray(selected_indices, dtype=np.int64)])[::-1]
        key = tuple(float(value) for value in np.round(candidate, 12))
        if key in seen:
            continue
        seen.add(key)
        hypotheses.append(candidate.astype(np.float64, copy=True))
        if len(hypotheses) >= int(top_n):
            break

    if len(hypotheses) < int(top_n):
        raise RuntimeError(
            f"Beam-GOMP returned only {len(hypotheses)} unique hypotheses; expected {top_n}"
        )
    return hypotheses


def _topn_robust_grid(
    Y: np.ndarray,
    metadata: AllowedMetadata,
    _: int,
    grid_size: int,
    top_n: int,
    branch_factor: int,
    beam_width: int,
) -> np.ndarray:
    """Select a Top-N grid hypothesis using data-only multi-snapshot robustness.

    Selection criterion:
      1. minimize the worst per-snapshot relative projection residual;
      2. break ties by the mean per-snapshot relative residual.

    The criterion deliberately uses no true wavenumber, seabed parameter or
    physical root.  It targets the failure observed at low SNR, where greedy
    Grid-GOMP can make an early wrong-mode decision.
    """
    hypotheses = _grid_beam_hypotheses(
        Y,
        metadata,
        grid_size,
        top_n,
        branch_factor,
        beam_width,
    )
    _, full_dictionary = _grid_dictionary(metadata, grid_size)
    # Map grid values back to columns by nearest grid point.  This keeps the
    # projection score identical to the dictionary used to generate hypotheses.
    grid, _ = _grid_dictionary(metadata, grid_size)

    scored: list[tuple[float, float, np.ndarray]] = []
    for candidate in hypotheses:
        indices = np.asarray(
            [int(np.argmin(np.abs(grid - float(value)))) for value in candidate],
            dtype=np.int64,
        )
        mean_score, worst_score = _projection_snapshot_scores(
            Y,
            full_dictionary[:, indices],
        )
        scored.append((worst_score, mean_score, candidate))

    best = min(scored, key=lambda item: (item[0], item[1]))
    return np.sort(np.asarray(best[2], dtype=np.float64))[::-1]


def _continuous_candidate_scores(
    Y: np.ndarray,
    candidate_k: np.ndarray,
) -> tuple[float, float]:
    """Score an arbitrary continuous-k candidate with data projection residuals.

    This is deliberately environment-light: the dictionary is generated from
    the water-column propagation model at the supplied candidate wavenumbers.
    No seabed acoustic parameter, modal root, KRAKEN truth or evaluation metric
    enters the score.
    """
    candidate = np.sort(np.asarray(candidate_k, dtype=np.float64).reshape(-1))[::-1]
    dictionary = np.asarray(
        baseline.EXP26.build_mode_matrix(candidate),
        dtype=np.complex128,
    )
    atom_norms = np.maximum(np.linalg.norm(dictionary, axis=0), 1.0e-15)
    dictionary = dictionary / atom_norms[None, :]
    return _projection_snapshot_scores(Y, dictionary)


def _topn_refined(
    Y: np.ndarray,
    metadata: AllowedMetadata,
    seed: int,
    grid_size: int,
    top_n: int,
    branch_factor: int,
    beam_width: int,
) -> np.ndarray:
    """Top-N coarse selection + conservative continuous off-grid refinement.

    The Top-N stage first solves the low-SNR wrong-mode selection problem using
    data-only beam hypotheses.  The selected coarse hypothesis is then refined
    continuously with ``estimate_from_coarse``.

    To prevent continuous optimization from degrading a borderline but correct
    low-SNR coarse solution, the refined result is accepted only when it improves
    BOTH data-only projection criteria:

      * mean per-snapshot relative residual; and
      * worst per-snapshot relative residual.

    Otherwise the robust Top-N coarse result is retained.  This gate never uses
    true wavenumbers, success labels, seabed parameters, physical roots or cells.
    """
    coarse_k = _topn_robust_grid(
        Y,
        metadata,
        seed,
        grid_size,
        top_n,
        branch_factor,
        beam_width,
    )

    refined_k = direct_estimator.estimate_from_coarse(
        Y,
        metadata.K,
        metadata.frequency,
        metadata.array_geometry,
        metadata.water_depth,
        metadata.SSP,
        seed,
        coarse_k,
    )

    coarse_mean, coarse_worst = _continuous_candidate_scores(Y, coarse_k)
    refined_mean, refined_worst = _continuous_candidate_scores(Y, refined_k)

    # Conservative, truth-free acceptance gate.  A refinement is used only if
    # it improves the average fit and does not sacrifice the worst snapshot.
    mean_improved = refined_mean < coarse_mean - 1.0e-12
    worst_not_worse = refined_worst <= coarse_worst + 1.0e-12

    if mean_improved and worst_not_worse:
        return np.sort(np.asarray(refined_k, dtype=np.float64))[::-1]
    return np.sort(np.asarray(coarse_k, dtype=np.float64))[::-1]


def run_estimators_on_shared_Y(
    Y: np.ndarray,
    metadata: AllowedMetadata,
    optimization_seed: int,
    grid_size: int,
    method_overrides: dict[str, Callable[[np.ndarray, AllowedMetadata, int], np.ndarray]] | None = None,
) -> list[MethodResult]:
    """Run every estimator before evaluation truth is made available."""
    config = load_config()
    top_n = int(config["top_n_hypotheses"])
    branch_factor = int(config["beam_branch_factor"])
    beam_width = int(config["beam_width"])
    methods: list[tuple[str, Callable[[np.ndarray, AllowedMetadata, int], np.ndarray]]] = [
        ("FD-SVD", _fd_svd),
        ("Grid-GOMP", lambda data, meta, seed: _grid_gomp(data, meta, seed, grid_size)),
        (
            "Top-N Robust Grid-GOMP",
            lambda data, meta, seed: _topn_robust_grid(
                data, meta, seed, grid_size, top_n, branch_factor, beam_width
            ),
        ),
        (
            "Top-N + Data-Gated Continuous Refinement",
            lambda data, meta, seed: _topn_refined(
                data, meta, seed, grid_size, top_n, branch_factor, beam_width
            ),
        ),
    ]
    if method_overrides:
        methods = [(name, method_overrides.get(name, function)) for name, function in methods]

    expected_hash = hash_Y(Y)
    results: list[MethodResult] = []
    for name, estimator in methods:
        method_input = np.asarray(Y, dtype=np.complex128).copy()
        input_hash = hash_Y(method_input)
        if input_hash != expected_hash:
            raise RuntimeError("Method input differs before estimator execution")
        start = time.perf_counter()
        exception = ""
        try:
            estimated = np.asarray(estimator(method_input, metadata, int(optimization_seed)), dtype=np.float64)
            estimated = np.sort(estimated.reshape(-1))[::-1]
        except Exception as caught:  # negative results are preserved in the CSV
            estimated = np.asarray([], dtype=np.float64)
            exception = repr(caught)
        runtime = time.perf_counter() - start
        if hash_Y(method_input) != expected_hash:
            raise RuntimeError(f"{name} modified its input observation matrix")
        results.append(
            MethodResult(
                method=name,
                estimated_k=estimated,
                runtime_seconds=float(runtime),
                optimization_seed=int(optimization_seed),
                input_y_hash=input_hash,
                exception=exception,
            )
        )
    if len({result.input_y_hash for result in results}) != 1:
        raise RuntimeError("Estimators did not receive identical observation matrices")
    return results


def load_truth_for_evaluation(K: int) -> np.ndarray:
    """Load KRAKEN wavenumbers only after all estimators have returned."""
    environment = baseline.load_kraken_environment()
    truth = np.real(np.asarray(environment["complex_k"], dtype=np.complex128)[: int(K)])
    return np.sort(np.asarray(truth, dtype=np.float64))[::-1]


def _evaluate(estimate: np.ndarray, truth: np.ndarray, threshold: float) -> dict:
    estimate = np.sort(np.asarray(estimate, dtype=np.float64).reshape(-1))[::-1]
    truth = np.sort(np.asarray(truth, dtype=np.float64).reshape(-1))[::-1]
    if estimate.size != truth.size or not np.all(np.isfinite(estimate)):
        return {
            "rmse": float("nan"),
            "max_abs_error": float("nan"),
            "success": False,
            "errors": np.full(truth.size, np.nan, dtype=np.float64),
        }
    errors = np.abs(estimate - truth)
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "max_abs_error": float(np.max(errors)),
        "success": bool(np.max(errors) <= float(threshold)),
        "errors": errors,
    }


TRIAL_FIELDS = (
    "K",
    "SNR_db",
    "trial",
    "method",
    "RMSE_rad_m",
    "max_abs_error_rad_m",
    "success",
    "runtime_seconds",
    "data_seed",
    "optimization_seed",
    "input_Y_hash",
    "achieved_SNR_db",
    "M",
    "L",
    "frequency_hz",
    "water_depth_m",
    "success_threshold_rad_m",
    "estimated_k",
    "true_k",
    "exception",
)

MODE_FIELDS = (
    "K",
    "SNR_db",
    "trial",
    "method",
    "mode",
    "abs_error_rad_m",
    "estimated_k",
    "true_k",
    "success",
    "input_Y_hash",
)


def _array_text(values: np.ndarray) -> str:
    return ";".join(f"{float(value):.15g}" for value in np.asarray(values).reshape(-1))


def _write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _summary_rows(trial_rows: list[dict], methods: tuple[str, ...], snr_values: tuple[float, ...]) -> list[dict]:
    rows: list[dict] = []
    for method in methods:
        for snr_db in snr_values:
            subset = [
                row
                for row in trial_rows
                if row["method"] == method and float(row["SNR_db"]) == float(snr_db)
            ]
            rmse = np.asarray([float(row["RMSE_rad_m"]) for row in subset], dtype=np.float64)
            maximum = np.asarray([float(row["max_abs_error_rad_m"]) for row in subset], dtype=np.float64)
            runtime = np.asarray([float(row["runtime_seconds"]) for row in subset], dtype=np.float64)
            success = np.asarray([str(row["success"]).lower() in {"true", "1"} for row in subset])
            finite_rmse = rmse[np.isfinite(rmse)]
            finite_maximum = maximum[np.isfinite(maximum)]
            rows.append(
                {
                    "method": method,
                    "SNR_db": float(snr_db),
                    "trials": len(subset),
                    "finite_estimates": int(finite_rmse.size),
                    "mean_RMSE_rad_m": float(np.mean(finite_rmse)) if finite_rmse.size else float("nan"),
                    "median_RMSE_rad_m": float(np.median(finite_rmse)) if finite_rmse.size else float("nan"),
                    "mean_max_abs_error_rad_m": float(np.mean(finite_maximum)) if finite_maximum.size else float("nan"),
                    "success_rate": float(np.mean(success)) if success.size else float("nan"),
                    "mean_runtime_seconds": float(np.mean(runtime)) if runtime.size else float("nan"),
                }
            )
    return rows


SUMMARY_FIELDS = (
    "method",
    "SNR_db",
    "trials",
    "finite_estimates",
    "mean_RMSE_rad_m",
    "median_RMSE_rad_m",
    "mean_max_abs_error_rad_m",
    "success_rate",
    "mean_runtime_seconds",
)


def generate_figures_from_csv(trial_csv: Path, mode_csv: Path, figure_dir: Path) -> None:
    """Generate all requested figures from saved CSV files only."""
    trial_rows = _read_csv(trial_csv)
    mode_rows = _read_csv(mode_csv)
    if not trial_rows or not mode_rows:
        raise RuntimeError("CSV results are required before figure generation")
    figure_dir.mkdir(parents=True, exist_ok=True)
    methods = tuple(dict.fromkeys(row["method"] for row in trial_rows))
    snr_values = np.asarray(sorted({float(row["SNR_db"]) for row in trial_rows}), dtype=np.float64)
    colors = plt.get_cmap("tab10").colors

    figure, axis = plt.subplots(figsize=(8.0, 5.2))
    for method_index, method in enumerate(methods):
        means = []
        for snr_db in snr_values:
            values = np.asarray(
                [
                    float(row["RMSE_rad_m"])
                    for row in trial_rows
                    if row["method"] == method and float(row["SNR_db"]) == snr_db
                ],
                dtype=np.float64,
            )
            values = values[np.isfinite(values)]
            means.append(float(np.mean(values)) if values.size else np.nan)
        axis.plot(snr_values, means, marker="o", label=method, color=colors[method_index])
    axis.set_yscale("log")
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("Mean wavenumber RMSE (rad/m)")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figure_dir / "rmse_vs_snr.png", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.0, 5.2))
    for method_index, method in enumerate(methods):
        rates = []
        for snr_db in snr_values:
            values = [
                str(row["success"]).lower() in {"true", "1"}
                for row in trial_rows
                if row["method"] == method and float(row["SNR_db"]) == snr_db
            ]
            rates.append(float(np.mean(values)) if values else np.nan)
        axis.plot(snr_values, rates, marker="o", label=method, color=colors[method_index])
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("Success rate")
    axis.set_ylim(-0.03, 1.03)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figure_dir / "success_rate_vs_snr.png", dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), sharex=True, sharey=True)
    for method_index, (axis, method) in enumerate(zip(axes.flat, methods)):
        for mode in (1, 2, 3):
            means = []
            for snr_db in snr_values:
                values = np.asarray(
                    [
                        float(row["abs_error_rad_m"])
                        for row in mode_rows
                        if row["method"] == method
                        and int(row["mode"]) == mode
                        and float(row["SNR_db"]) == snr_db
                    ],
                    dtype=np.float64,
                )
                values = values[np.isfinite(values)]
                means.append(float(np.mean(values)) if values.size else np.nan)
            axis.plot(snr_values, means, marker="o", label=f"Mode {mode}")
        axis.set_title(method, fontsize=10)
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)
    figure.supxlabel("SNR (dB)")
    figure.supylabel("Mean absolute error (rad/m)")
    figure.tight_layout()
    figure.savefig(figure_dir / "k3_mode_wise_error_vs_snr.png", dpi=220)
    plt.close(figure)

    runtime_means = []
    runtime_stds = []
    for method in methods:
        values = np.asarray(
            [float(row["runtime_seconds"]) for row in trial_rows if row["method"] == method],
            dtype=np.float64,
        )
        runtime_means.append(float(np.mean(values)))
        runtime_stds.append(float(np.std(values)))
    figure, axis = plt.subplots(figsize=(9.0, 5.2))
    positions = np.arange(len(methods))
    axis.bar(positions, runtime_means, yerr=runtime_stds, capsize=4, color=colors[: len(methods)])
    axis.set_xticks(positions, methods, rotation=18, ha="right")
    axis.set_ylabel("Runtime per realization (s)")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(figure_dir / "runtime_comparison.png", dpi=220)
    plt.close(figure)


def _write_markdown_summary(output_root: Path, summary_rows: list[dict], mode_rows: list[dict]) -> None:
    lookup = {(row["method"], float(row["SNR_db"])): row for row in summary_rows}
    low_snr = (-5.0, 0.0, 5.0)
    grid_name = "Grid-GOMP"
    topn_name = "Top-N Robust Grid-GOMP"
    final_name = "Top-N + Data-Gated Continuous Refinement"

    lines = [
        "# Direct Wavenumber Estimation — Top-N + Data-Gated Continuous Refinement",
        "",
        "All quantitative statements below are generated from the saved CSV files.",
        "",
        "## Per-SNR comparison",
        "",
        "| SNR (dB) | FD-SVD RMSE | Grid RMSE | Top-N RMSE | Top-N+Continuous RMSE | Grid success | Top-N success | Top-N+Continuous success |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for snr_db in (-5.0, 0.0, 5.0, 10.0, 15.0):
        fd = lookup[("FD-SVD", snr_db)]
        grid = lookup[(grid_name, snr_db)]
        topn = lookup[(topn_name, snr_db)]
        final = lookup[(final_name, snr_db)]
        lines.append(
            f"| {snr_db:g} | {float(fd['mean_RMSE_rad_m']):.6e} | "
            f"{float(grid['mean_RMSE_rad_m']):.6e} | "
            f"{float(topn['mean_RMSE_rad_m']):.6e} | "
            f"{float(final['mean_RMSE_rad_m']):.6e} | "
            f"{float(grid['success_rate']):.3f} | "
            f"{float(topn['success_rate']):.3f} | "
            f"{float(final['success_rate']):.3f} |"
        )

    lines.extend(["", "## Low-SNR gate (-5/0/5 dB)", ""])
    for snr_db in low_snr:
        grid = lookup[(grid_name, snr_db)]
        topn = lookup[(topn_name, snr_db)]
        final = lookup[(final_name, snr_db)]
        lines.append(
            f"- {snr_db:g} dB: Grid success={float(grid['success_rate']):.3f}, "
            f"Top-N success={float(topn['success_rate']):.3f}, "
            f"Top-N+Continuous success={float(final['success_rate']):.3f}; "
            f"Top-N RMSE={float(topn['mean_RMSE_rad_m']):.6e}, "
            f"Top-N+Continuous RMSE={float(final['mean_RMSE_rad_m']):.6e}."
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- Top-N tests whether multiple data-only coarse hypotheses reduce low-SNR wrong-mode selection.",
            "- The continuous stage tests whether off-grid refinement reduces the remaining grid quantization error after robust coarse selection. A data-only conservative gate retains the coarse Top-N result whenever the refinement does not improve both mean and worst per-snapshot projection residuals.",
            "- The estimator uses frequency, SSP, water depth, array geometry and the observed multi-snapshot pressure matrix only.",
            "- No seabed sound speed, seabed density, attenuation, physical modal root, root-derived cell, Robin residual, root/physics score, truth-guided refinement, or root-guided refinement enters estimation.",
            "- KRAKEN wavenumbers are loaded only after every estimator has returned and are used only for evaluation.",
            "- Failed trials remain in the CSV and are not re-seeded.",
            "",
            "## Scope",
            "",
            "- K=3; SNR=-5/0/5/10/15 dB; 20 fixed-seed trials per SNR.",
            "- Top-N settings: 5 final hypotheses, branch factor 6, beam width 24.",
            "- Continuous refinement is initialized only by the selected Top-N data hypothesis; acceptance is based only on observed-data projection residuals.",
        ]
    )

    (output_root / "quick_results_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_quick(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    config = load_config(config_path)
    output_root = _output_root(config)
    csv_dir = output_root / "csv"
    figure_dir = output_root / "figures"
    log_dir = output_root / "logs"
    csv_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    protected_before = _manifest_fingerprint(PROTECTED_OUTPUT)
    trial_csv = csv_dir / "direct_wavenumber_trials_quick.csv"
    mode_csv = csv_dir / "direct_wavenumber_mode_errors_quick.csv"
    summary_csv = csv_dir / "direct_wavenumber_summary_quick.csv"
    trial_rows: list[dict] = [dict(row) for row in _read_csv(trial_csv)]
    mode_rows: list[dict] = [dict(row) for row in _read_csv(mode_csv)]

    config_snapshot = {
        **config,
        "direct_schedule": direct_estimator.DIRECT_SCHEDULE.__dict__,
        "legacy_steps": direct_estimator.LEGACY_STEPS,
        "search_interval_rule": {
            "lower": "0.95 * omega / max(SSP)",
            "upper": "0.999999 * omega / min(SSP)",
        },
        "selection": "greedy Grid-GOMP; Top-N beam GOMP with worst-snapshot data-residual selection; Top-N-guided continuous off-grid refinement",
        "forbidden_paths": [
            "seabed acoustic parameters",
            "physical modal roots",
            "root-derived cells",
            "root/physics candidate scores",
            "truth-guided or root-guided refinement",
            "KRAKEN truth during estimation",
        ],
    }
    (output_root / "run_config_snapshot.json").write_text(
        json.dumps(config_snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    K = int(config["K"])
    snapshots = int(config["snapshots"])
    water_depth = float(config["water_depth_m"])
    frequency = float(config["frequency_hz"])
    methods = tuple(str(value) for value in config["methods"])
    snr_values = tuple(float(value) for value in config["snr_db"])
    threshold = float(config["success_max_abs_error_rad_m"])
    grid_size = int(config["grid_size"])
    base_seed = int(config["base_seed"])

    clean_Y, receiver_depths, ssp = prepare_forward_observation(K, snapshots, water_depth)
    metadata = AllowedMetadata(K, frequency, receiver_depths, water_depth, ssp)
    total_workpoints = len(snr_values) * int(config["trials_per_snr"])
    workpoint_index = 0

    for snr_index, snr_db in enumerate(snr_values):
        for trial in range(1, int(config["trials_per_snr"]) + 1):
            workpoint_index += 1
            existing_methods = {
                row["method"]
                for row in trial_rows
                if float(row["SNR_db"]) == snr_db and int(float(row["trial"])) == trial
            }
            if existing_methods == set(methods):
                print(f"[SKIP {workpoint_index:02d}/{total_workpoints}] SNR={snr_db:g}, trial={trial}")
                continue

            data_seed = base_seed + trial * 10_000 + int((snr_db + 20.0) * 100.0)
            optimization_seed = base_seed + 500_000 + snr_index * 10_000 + trial * 100
            noisy_Y, achieved_snr = add_shared_noise(clean_Y, snr_db, data_seed)
            print(
                f"[RUN  {workpoint_index:02d}/{total_workpoints}] SNR={snr_db:g}, trial={trial}, "
                f"data_seed={data_seed}, optimization_seed={optimization_seed}",
                flush=True,
            )

            # This call completes all four estimators without a truth argument.
            method_results = run_estimators_on_shared_Y(
                noisy_Y, metadata, optimization_seed, grid_size
            )

            # Evaluation truth becomes available only after every estimator returned.
            truth = load_truth_for_evaluation(K)
            trial_rows = [
                row
                for row in trial_rows
                if not (float(row["SNR_db"]) == snr_db and int(float(row["trial"])) == trial)
            ]
            mode_rows = [
                row
                for row in mode_rows
                if not (float(row["SNR_db"]) == snr_db and int(float(row["trial"])) == trial)
            ]

            for result in method_results:
                metrics = _evaluate(result.estimated_k, truth, threshold)
                trial_rows.append(
                    {
                        "K": K,
                        "SNR_db": snr_db,
                        "trial": trial,
                        "method": result.method,
                        "RMSE_rad_m": metrics["rmse"],
                        "max_abs_error_rad_m": metrics["max_abs_error"],
                        "success": metrics["success"],
                        "runtime_seconds": result.runtime_seconds,
                        "data_seed": data_seed,
                        "optimization_seed": result.optimization_seed,
                        "input_Y_hash": result.input_y_hash,
                        "achieved_SNR_db": achieved_snr,
                        "M": (
                            K
                            if result.method == "Top-N + Data-Gated Continuous Refinement"
                            else 0
                        ),
                        "L": snapshots,
                        "frequency_hz": frequency,
                        "water_depth_m": water_depth,
                        "success_threshold_rad_m": threshold,
                        "estimated_k": _array_text(result.estimated_k),
                        "true_k": _array_text(truth),
                        "exception": result.exception,
                    }
                )
                for mode_index in range(K):
                    estimated_value = (
                        float(result.estimated_k[mode_index])
                        if result.estimated_k.size == K
                        else float("nan")
                    )
                    mode_rows.append(
                        {
                            "K": K,
                            "SNR_db": snr_db,
                            "trial": trial,
                            "method": result.method,
                            "mode": mode_index + 1,
                            "abs_error_rad_m": float(metrics["errors"][mode_index]),
                            "estimated_k": estimated_value,
                            "true_k": float(truth[mode_index]),
                            "success": metrics["success"],
                            "input_Y_hash": result.input_y_hash,
                        }
                    )
                print(
                    f"  {result.method}: rmse={metrics['rmse']:.6e}, "
                    f"max={metrics['max_abs_error']:.6e}, success={metrics['success']}, "
                    f"runtime={result.runtime_seconds:.2f}s",
                    flush=True,
                )

            trial_rows.sort(key=lambda row: (float(row["SNR_db"]), int(float(row["trial"])), methods.index(row["method"])))
            mode_rows.sort(
                key=lambda row: (
                    float(row["SNR_db"]),
                    int(float(row["trial"])),
                    methods.index(row["method"]),
                    int(float(row["mode"])),
                )
            )
            _write_csv(trial_csv, trial_rows, TRIAL_FIELDS)
            _write_csv(mode_csv, mode_rows, MODE_FIELDS)

    expected_trial_rows = total_workpoints * len(methods)
    expected_mode_rows = expected_trial_rows * K
    if len(trial_rows) != expected_trial_rows or len(mode_rows) != expected_mode_rows:
        raise RuntimeError(
            f"Incomplete K=3 Monte Carlo coverage: trials={len(trial_rows)}/{expected_trial_rows}, "
            f"mode_rows={len(mode_rows)}/{expected_mode_rows}"
        )
    for snr_db in snr_values:
        for trial in range(1, int(config["trials_per_snr"]) + 1):
            hashes = {
                row["input_Y_hash"]
                for row in trial_rows
                if float(row["SNR_db"]) == snr_db and int(float(row["trial"])) == trial
            }
            if len(hashes) != 1:
                raise RuntimeError(f"Input Y hash mismatch at SNR={snr_db}, trial={trial}")

    summary_rows = _summary_rows(trial_rows, methods, snr_values)
    _write_csv(summary_csv, summary_rows, SUMMARY_FIELDS)
    generate_figures_from_csv(trial_csv, mode_csv, figure_dir)
    _write_markdown_summary(output_root, summary_rows, mode_rows)

    protected_after = _manifest_fingerprint(PROTECTED_OUTPUT)
    if protected_after != protected_before:
        raise RuntimeError("outputs/final_paper changed during the direct experiment")
    completion = {
        "status": "complete",
        "trial_rows": len(trial_rows),
        "mode_rows": len(mode_rows),
        "final_paper_manifest_before": protected_before,
        "final_paper_manifest_after": protected_after,
        "truth_loaded_only_after_all_estimators_per_workpoint": True,
    }
    (output_root / "completion.json").write_text(
        json.dumps(completion, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(completion, indent=2, ensure_ascii=False), flush=True)
    return completion


def regenerate_figures(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    config = load_config(config_path)
    output_root = _output_root(config)
    generate_figures_from_csv(
        output_root / "csv" / "direct_wavenumber_trials_quick.csv",
        output_root / "csv" / "direct_wavenumber_mode_errors_quick.csv",
        output_root / "figures",
    )
