"""Frozen final experiment orchestration.

This module calls the validated numerical backends while enforcing the final
configuration: Numerov4 dz=0.5 m, lambda_bottom=0.1, selector gamma=0 and no
safe-root refinement.
"""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import torch

from .final_config import FinalConfig, RunSpec
from .final_io import (
    CSV_DIR,
    array_text,
    classify_failure,
    common_metadata,
    config_fingerprint,
    ensure_output_tree,
    save_csv,
)


def _master():
    from .backends import mode_frinet_master_pipeline as module
    return module


def _baseline():
    from .backends import baseline_comparison_final as module
    return module


def _physics_validation():
    from .backends import high_order_bottom_discretization_validation as module
    return module


def _evaluate(estimated: Any, truth: Any, tolerance: float) -> dict[str, Any]:
    estimated = np.sort(np.asarray(estimated, dtype=np.float64).reshape(-1))[::-1]
    truth = np.sort(np.asarray(truth, dtype=np.float64).reshape(-1))[::-1]
    if estimated.size != truth.size or not np.all(np.isfinite(estimated)):
        return {
            "estimated": estimated,
            "rmse": float("inf"),
            "max_error": float("inf"),
            "success": 0,
            "errors": np.full(truth.shape, np.inf),
        }
    errors = np.abs(estimated - truth)
    return {
        "estimated": estimated,
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "max_error": float(np.max(errors)),
        "success": int(float(np.max(errors)) <= tolerance),
        "errors": errors,
    }


def _summarize(rows: Iterable[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for key, subset in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        finite_rmse = np.asarray(
            [float(row["rmse"]) for row in subset if np.isfinite(float(row["rmse"]))],
            dtype=np.float64,
        )
        runtimes = np.asarray([float(row["runtime_seconds"]) for row in subset], dtype=np.float64)
        summary = {name: value for name, value in zip(group_keys, key)}
        summary.update(
            runs=len(subset),
            success_rate=float(np.mean([int(row["success"]) for row in subset])),
            rmse_mean=float(np.mean(finite_rmse)) if finite_rmse.size else float("nan"),
            rmse_median=float(np.median(finite_rmse)) if finite_rmse.size else float("nan"),
            rmse_p90=float(np.quantile(finite_rmse, 0.9)) if finite_rmse.size else float("nan"),
            runtime_seconds_mean=float(np.mean(runtimes)),
        )
        output.append(summary)
    return output


def configure_backends(config: FinalConfig) -> None:
    master = _master()
    base = master.BASE
    base.PHYSICS_DZ = config.dz_m
    base.NUMBER_OF_ATOMS = config.atoms
    base.NUMBER_OF_SNAPSHOTS = config.snapshots
    base.BOTTOM_LAMBDA = config.lambda_bottom
    base.PHYSICS_SCORE_GAMMA = config.selector_gamma
    master.FINAL_BOTTOM_LAMBDA = config.lambda_bottom
    master.FINAL_SELECTOR_GAMMA = config.selector_gamma
    master.BOTTOM_LAMBDA_BY_K = {3: config.lambda_bottom, 5: config.lambda_bottom, 8: config.lambda_bottom}
    master.BOTTOM_LAMBDA_CANDIDATES_BY_K = {
        3: [config.lambda_bottom], 5: [config.lambda_bottom], 8: [config.lambda_bottom]
    }
    master.RESTARTS_BY_K = {3: 1, 5: 1, 8: 1}


def run_nominal(
    config: FinalConfig,
    spec: RunSpec,
    *,
    csv_dir: Path = CSV_DIR,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    master = _master()
    configure_backends(config)
    kraken = master.BASE.load_kraken()
    complex_k = np.asarray(kraken["complex_k"], dtype=np.complex128)
    mode_matrix = np.asarray(kraken["mode_matrix"], dtype=np.complex128)
    rows: list[dict[str, Any]] = []
    mode_rows: list[dict[str, Any]] = []
    for k_count in spec.nominal_k:
        truth = np.real(complex_k[:k_count]).astype(np.float64)
        forward = mode_matrix[:, :k_count]
        for snr_db in spec.nominal_snr_db:
            for trial in range(1, spec.nominal_trials + 1):
                master.restore_nominal_environment()
                roots = master.BASE.calculate_physical_roots()
                noisy, seed = master.create_noisy_dataset(forward, k_count, snr_db, trial)
                start = time.perf_counter()
                exception: Exception | None = None
                try:
                    result = master.run_mode_frinet(
                        noisy_data=noisy,
                        physical_roots=roots,
                        number_of_modes=k_count,
                        random_seed=seed,
                        bottom_lambda=config.lambda_bottom,
                    )
                    estimate = result["k"]
                    selector_fit = float(result["fit"])
                    selection_success = bool(result["selection_success"])
                except Exception as error:  # preserved in output, not hidden
                    exception = error
                    estimate = np.asarray([], dtype=np.float64)
                    selector_fit = float("inf")
                    selection_success = False
                runtime = time.perf_counter() - start
                metrics = _evaluate(estimate, truth, config.success_tolerance)
                metadata = common_metadata(config, spec, seed=seed, k=k_count, snr_db=snr_db, trial=trial)
                row = {
                    **metadata,
                    "experiment": "nominal",
                    "method": "HO-PG Mode-FRINET",
                    "success": metrics["success"],
                    "rmse": metrics["rmse"],
                    "max_error": metrics["max_error"],
                    "selector_fit": selector_fit,
                    "runtime_seconds": runtime,
                    "failure_class": classify_failure(
                        exception=exception,
                        finite=np.isfinite(metrics["rmse"]),
                        selection_success=selection_success,
                        max_error=metrics["max_error"],
                        tolerance=config.success_tolerance,
                    ),
                    "exception": "" if exception is None else repr(exception),
                    "estimated_k": array_text(metrics["estimated"]),
                    "true_k": array_text(truth),
                }
                rows.append(row)
                for index, true_value in enumerate(truth):
                    estimate_value = metrics["estimated"][index] if metrics["estimated"].size == truth.size else float("nan")
                    mode_rows.append(
                        {
                            **metadata,
                            "method": row["method"],
                            "mode": index + 1,
                            "true_k": float(true_value),
                            "estimated_k": float(estimate_value),
                            "abs_error": float(abs(estimate_value - true_value)),
                        }
                    )
                save_csv(csv_dir / f"nominal_trials_{spec.mode}.csv", rows)
                save_csv(csv_dir / f"nominal_modewise_{spec.mode}.csv", mode_rows)
    summary = _summarize(rows, ("method", "K", "SNR_db"))
    save_csv(csv_dir / f"nominal_summary_{spec.mode}.csv", summary)
    return rows, summary


def _baseline_dataset(bl, snapshots: int, snr_db: float, trial: int):
    kraken = bl.load_kraken_environment()
    forward = bl.generate_clean_moving_source_field(
        receiver_modes=kraken["receiver_modes"],
        source_modes=kraken["source_modes"],
        complex_k=kraken["complex_k"],
    )
    clean_receiver = np.asarray(forward["clean_receiver"], dtype=np.complex128)
    if clean_receiver.ndim != 2 or clean_receiver.shape[1] < snapshots:
        raise ValueError(
            f"Baseline dataset has {clean_receiver.shape}; cannot provide L={snapshots} snapshots"
        )
    # The legacy moving-source generator produces 64 range snapshots.  The
    # frozen final configuration is L=20, so every baseline consumes the same
    # deterministic first 20 snapshots before noise is added.
    clean_receiver = clean_receiver[:, :snapshots].copy()
    seed = bl.BASE_RANDOM_SEED + int((snr_db + 20.0) * 1000) + trial * 100
    noisy, achieved = bl.add_noise(clean_receiver, snr_db, seed)
    receiver = bl.normalize_snapshot_columns(noisy)
    inverse = _to_fd2_grid(receiver)
    return kraken, forward, receiver, inverse, seed, achieved


def _configure_baseline_backend(bl, config: FinalConfig) -> None:
    """Synchronize the legacy FD2 backend with the frozen final dimensions."""

    bl.EXP26.NUMBER_OF_ATOMS = config.atoms
    bl.EXP26.NUMBER_OF_SNAPSHOTS = config.snapshots


def _to_fd2_grid(receiver_data: np.ndarray) -> np.ndarray:
    """Validate/copy the fixed 40-receiver FD2 grid without any truth input."""
    receiver_data = np.asarray(receiver_data, dtype=np.complex128)
    if receiver_data.ndim != 2 or receiver_data.shape[0] != 40:
        raise ValueError(f"FD2 expects the fixed 40-receiver grid, got {receiver_data.shape}")
    return receiver_data.copy()


def _to_ho_grid(inverse_data: np.ndarray) -> np.ndarray:
    """Add the exact pressure-release boundary required by the 41-row HO model."""
    inverse_data = np.asarray(inverse_data, dtype=np.complex128)
    if inverse_data.shape[0] == 41:
        return inverse_data.copy()
    if inverse_data.shape[0] == 40:
        return np.vstack(
            [np.zeros((1, inverse_data.shape[1]), dtype=np.complex128), inverse_data]
        )
    raise ValueError(f"Unsupported inverse-data row count: {inverse_data.shape[0]}")


def _baseline_row(config, spec, *, method, seed, snr_db, trial, estimate, truth, runtime, extra=None):
    metrics = _evaluate(estimate, truth, config.success_tolerance)
    metadata = common_metadata(config, spec, seed=seed, k=len(truth), snr_db=snr_db, trial=trial)
    row = {
        **metadata,
        "experiment": "baseline",
        "method": method,
        "success": metrics["success"],
        "rmse": metrics["rmse"],
        "max_error": metrics["max_error"],
        "runtime_seconds": float(runtime),
        "failure_class": classify_failure(
            exception=None,
            finite=np.isfinite(metrics["rmse"]),
            selection_success=metrics["estimated"].size == len(truth),
            max_error=metrics["max_error"],
            tolerance=config.success_tolerance,
        ),
        "estimated_k": array_text(metrics["estimated"]),
        "true_k": array_text(truth),
    }
    if extra:
        row.update(extra)
    return row


def run_baselines(
    config: FinalConfig,
    spec: RunSpec,
    *,
    csv_dir: Path = CSV_DIR,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bl = _baseline()
    master = _master()
    configure_backends(config)
    _configure_baseline_backend(bl, config)
    bl.NUMBER_OF_RESTARTS = 1
    rows: list[dict[str, Any]] = []
    for snr_db in spec.baseline_snr_db:
        for trial in range(1, spec.baseline_trials + 1):
            kraken, forward, receiver, inverse, seed, achieved = _baseline_dataset(
                bl, config.snapshots, snr_db, trial
            )
            truth = np.asarray(forward["true_k"], dtype=np.float64)
            actual_l = int(inverse.shape[1])
            fd = bl.fd_svd_paper(receiver, np.asarray(kraken["receiver_depths"]))
            rows.append(_baseline_row(
                config, spec, method="FD-SVD", seed=seed, snr_db=snr_db, trial=trial,
                estimate=fd["estimated_k"], truth=truth, runtime=fd["runtime"],
                extra={"L": actual_l, "achieved_SNR_db": achieved, "grid_size": 0},
            ))
            for grid_size in config.raw["baseline"]["grid_sizes"]:
                bl.GRID_SIZE = int(grid_size)
                grid = bl.grid_512_gomp(inverse)
                grid_label = config.raw["baseline"].get(
                    "grid_method_label", f"Grid-{grid_size}"
                )
                rows.append(_baseline_row(
                    config, spec, method=grid_label, seed=seed, snr_db=snr_db, trial=trial,
                    estimate=grid["estimated_k"], truth=truth, runtime=grid["runtime"],
                    extra={"L": actual_l, "achieved_SNR_db": achieved, "grid_size": int(grid_size)},
                ))
            fd2_seed = config.base_seed + 500_000 + trial * 10_000 + int((snr_db + 20) * 1000)
            fd2 = bl.mode_frinet_estimate(inverse, fd2_seed, enable_refinement=False)
            rows.append(_baseline_row(
                config, spec, method="FD2 Mode-FRINET", seed=fd2_seed, snr_db=snr_db, trial=trial,
                estimate=fd2["raw_k"], truth=truth, runtime=fd2["runtime"],
                extra={"L": actual_l, "achieved_SNR_db": achieved, "grid_size": 0},
            ))
            master.restore_nominal_environment()
            roots = master.BASE.calculate_physical_roots()
            ho_inverse = _to_ho_grid(inverse)
            ho_seed = config.base_seed + 700_000 + trial * 10_000 + int((snr_db + 20) * 1000)
            start = time.perf_counter()
            ho = master.run_mode_frinet(ho_inverse, roots, len(truth), ho_seed, config.lambda_bottom)
            runtime = time.perf_counter() - start
            rows.append(_baseline_row(
                config, spec, method="HO-PG Mode-FRINET", seed=ho_seed, snr_db=snr_db, trial=trial,
                estimate=ho["k"], truth=truth, runtime=runtime,
                extra={"L": actual_l, "achieved_SNR_db": achieved, "grid_size": 0},
            ))
            save_csv(csv_dir / f"baseline_trials_{spec.mode}.csv", rows)
    summary = _summarize(rows, ("method", "K", "SNR_db"))
    save_csv(csv_dir / f"baseline_summary_{spec.mode}.csv", summary)
    return rows, summary


def run_grid_resolution(
    config: FinalConfig,
    spec: RunSpec,
    *,
    csv_dir: Path = CSV_DIR,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare fixed grid resolutions on the exact baseline realizations.

    This is a lightweight derived comparison.  It does not rerun FD-SVD,
    FD2, or HO-PG and does not use KRAKEN truth until post-estimation
    evaluation in ``_baseline_row``.
    """
    bl = _baseline()
    configure_backends(config)
    sizes = tuple(
        int(value)
        for value in config.raw.get("derived_comparisons", {}).get(
            "grid_resolution_sizes", (128, 256, 512)
        )
    )
    if sizes != (128, 256, 512):
        raise ValueError("Final grid-resolution comparison is frozen at 128/256/512")

    rows: list[dict[str, Any]] = []
    original_grid_size = int(bl.GRID_SIZE)
    try:
        for snr_db in spec.baseline_snr_db:
            for trial in range(1, spec.baseline_trials + 1):
                _, forward, _, inverse, seed, achieved = _baseline_dataset(
                    bl, config.snapshots, snr_db, trial
                )
                truth = np.asarray(forward["true_k"], dtype=np.float64)
                actual_l = int(inverse.shape[1])
                for grid_size in sizes:
                    bl.GRID_SIZE = grid_size
                    result = bl.grid_512_gomp(inverse)
                    row = _baseline_row(
                        config,
                        spec,
                        method=f"Grid-{grid_size}",
                        seed=seed,
                        snr_db=snr_db,
                        trial=trial,
                        estimate=result["estimated_k"],
                        truth=truth,
                        runtime=result["runtime"],
                        extra={
                            "L": actual_l,
                            "achieved_SNR_db": achieved,
                            "grid_size": grid_size,
                        },
                    )
                    row["experiment"] = "grid_resolution"
                    rows.append(row)
                save_csv(csv_dir / f"grid_resolution_trials_{spec.mode}.csv", rows)
    finally:
        bl.GRID_SIZE = original_grid_size

    summary = _summarize(rows, ("method", "grid_size", "K", "SNR_db"))
    save_csv(csv_dir / f"grid_resolution_summary_{spec.mode}.csv", summary)
    return rows, summary


def run_physics_validation(
    config: FinalConfig,
    spec: RunSpec,
    *,
    csv_dir: Path = CSV_DIR,
) -> dict[str, Path]:
    """Run the existing deterministic FD2/Numerov discretization validation.

    The validated backend remains the sole numerical implementation.  This
    adapter only redirects its deterministic CSV and diagnostic figure outputs
    into the final-paper tree; no estimator setting or Monte Carlo result is
    changed.
    """

    del spec  # deterministic and identical in QUICK/FULL
    validation = _physics_validation()
    settings = config.raw.get("derived_comparisons", {}).get("physics_validation", {})
    expected = {
        "baseline_solver": "FD2",
        "baseline_dz_m": 1.0,
        "final_solver": "NUMEROV4",
        "final_dz_m": 0.5,
        "mode_range": [6, 8],
    }
    if settings != expected:
        raise ValueError(f"Unexpected frozen physics-validation settings: {settings}")

    csv_dir.mkdir(parents=True, exist_ok=True)
    raw_figure_dir = csv_dir.parent / "logs" / "physics_validation_raw_figures"
    raw_figure_dir.mkdir(parents=True, exist_ok=True)
    redirected = {
        "OUTPUT_DIR": csv_dir,
        "FIGURE_DIR": raw_figure_dir,
        "MODEWISE_CSV": csv_dir / "high_order_discretization_modewise.csv",
        "SUMMARY_CSV": csv_dir / "high_order_discretization_summary.csv",
        "BOTTOM_CSV": csv_dir / "bottom_derivative_residual_comparison.csv",
        "BEST_CSV": csv_dir / "best_physics_variant_summary.csv",
    }
    original = {name: getattr(validation, name) for name in redirected}
    try:
        for name, value in redirected.items():
            setattr(validation, name, value)
        validation.main()
    finally:
        for name, value in original.items():
            setattr(validation, name, value)

    required = {
        name: path for name, path in redirected.items() if name.endswith("_CSV")
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Physics validation did not generate: {missing}")
    return required


def run_fd_svd_native_reference(
    config: FinalConfig,
    spec: RunSpec,
    *,
    csv_dir: Path = CSV_DIR,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Paper-native NR=64 FD-SVD sanity check, separate from the L=20 baseline."""

    bl = _baseline()
    settings = config.raw.get("derived_comparisons", {}).get(
        "fd_svd_native_reference", {}
    )
    required = {
        "K": 3,
        "NR": 64,
        "R0_m": 4000.0,
        "range_aperture_m": 1698.4,
        "SNR_db": [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 30.0],
        "trials": 10,
    }
    if settings != required:
        raise ValueError(f"Unexpected frozen FD-SVD native-reference settings: {settings}")
    if (
        int(bl.NUMBER_OF_TARGET_MODES) != 3
        or int(bl.NUMBER_OF_FORWARD_MODES) != 3
        or int(bl.NUMBER_OF_RANGE_SAMPLES) != 64
        or not math.isclose(float(bl.INITIAL_RANGE_M), 4000.0)
        or not math.isclose(float(bl.RANGE_APERTURE_M), 1698.4)
    ):
        raise ValueError("The existing FD-SVD backend is not at its paper-native geometry")

    snr_values = tuple(float(value) for value in settings["SNR_db"])
    trials = int(settings["trials"])
    if spec.mode == "quick":
        snr_values = (15.0,)
        trials = 1

    kraken = bl.load_kraken_environment()
    forward = bl.generate_clean_moving_source_field(
        receiver_modes=kraken["receiver_modes"],
        source_modes=kraken["source_modes"],
        complex_k=kraken["complex_k"],
    )
    clean = np.asarray(forward["clean_receiver"], dtype=np.complex128)
    truth = np.asarray(forward["true_k"], dtype=np.float64)
    receiver_depths = np.asarray(kraken["receiver_depths"], dtype=np.float64)
    if clean.shape != (40, 64):
        raise ValueError(f"Expected the paper-native 40x64 field, got {clean.shape}")

    reference_id = config_fingerprint(settings)
    rows: list[dict[str, Any]] = []
    for snr_db in snr_values:
        for trial in range(1, trials + 1):
            seed = bl.BASE_RANDOM_SEED + int((snr_db + 20.0) * 1000) + trial * 100
            noisy, achieved = bl.add_noise(clean, snr_db, seed)
            receiver = bl.normalize_snapshot_columns(noisy)
            result = bl.fd_svd_paper(receiver, receiver_depths)
            metrics = _evaluate(result["estimated_k"], truth, config.success_tolerance)
            rows.append(
                {
                    "seed": seed,
                    "K": 3,
                    "SNR_db": snr_db,
                    "trial": trial,
                    "NR": 64,
                    "R0_m": 4000.0,
                    "range_aperture_m": 1698.4,
                    "range_step_m": float(bl.RANGE_STEP_M),
                    "run_mode": spec.mode,
                    "config_id": config_fingerprint(config.raw),
                    "reference_config_id": reference_id,
                    "experiment": "fd_svd_native_reference",
                    "method": "FD-SVD (paper-native NR=64)",
                    "data_budget_label": "paper-native reference; not equal-budget L=20",
                    "success": metrics["success"],
                    "rmse": metrics["rmse"],
                    "max_error": metrics["max_error"],
                    "runtime_seconds": float(result["runtime"]),
                    "achieved_SNR_db": achieved,
                    "failure_class": classify_failure(
                        exception=None,
                        finite=np.isfinite(metrics["rmse"]),
                        selection_success=metrics["estimated"].size == truth.size,
                        max_error=metrics["max_error"],
                        tolerance=config.success_tolerance,
                    ),
                    "estimated_k": array_text(metrics["estimated"]),
                    "true_k": array_text(truth),
                }
            )
            save_csv(csv_dir / f"fd_svd_native_reference_trials_{spec.mode}.csv", rows)

    summary = _summarize(rows, ("method", "K", "SNR_db"))
    for row in summary:
        row.update(
            NR=64,
            R0_m=4000.0,
            range_aperture_m=1698.4,
            data_budget_label="paper-native reference; not equal-budget L=20",
            reference_config_id=reference_id,
        )
    save_csv(csv_dir / f"fd_svd_native_reference_summary_{spec.mode}.csv", summary)
    return rows, summary


def _select_random_numerov(base, noisy, roots, k_count, seed):
    trained = base.train_one_restart(noisy, k_count, seed)
    selection = base.select_mode_order_beam(
        trained=trained,
        noisy_data=noisy,
        physical_roots=roots,
        number_of_modes=k_count,
    )
    estimate = selection["selected"] if selection["success"] else np.asarray([])
    return estimate, float(trained["runtime_seconds"]), bool(selection["success"])


def run_ablation(
    config: FinalConfig,
    spec: RunSpec,
    *,
    csv_dir: Path = CSV_DIR,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bl = _baseline()
    master = _master()
    configure_backends(config)
    _configure_baseline_backend(bl, config)
    k_count = int(config.raw["ablation"]["K"])
    rows: list[dict[str, Any]] = []
    for snr_db in spec.ablation_snr_db:
        for trial in range(1, spec.ablation_trials + 1):
            _, forward, _, inverse, data_seed, achieved = _baseline_dataset(
                bl, config.snapshots, snr_db, trial
            )
            truth = np.asarray(forward["true_k"][:k_count], dtype=np.float64)
            ho_inverse = _to_ho_grid(inverse)
            roots = master.BASE.calculate_physical_roots()
            seed = config.base_seed + 900_000 + trial * 10_000 + int((snr_db + 20) * 1000)
            methods: list[tuple[str, Any, float, bool, float]] = []

            old_fd2_lambda = float(bl.EXP26.ROBIN_LAMBDA)
            old_fd2_gamma = float(bl.EXP26.PHYSICS_SCORE_GAMMA)
            bl.EXP26.ROBIN_LAMBDA = 0.0
            bl.EXP26.PHYSICS_SCORE_GAMMA = 0.0
            bl.NUMBER_OF_RESTARTS = 1
            fd2 = bl.mode_frinet_estimate(inverse, seed, enable_refinement=False)
            methods.append(("FD2+random", fd2["raw_k"], fd2["runtime"], len(fd2["raw_k"]) == k_count, 0.0))
            bl.EXP26.ROBIN_LAMBDA = old_fd2_lambda
            bl.EXP26.PHYSICS_SCORE_GAMMA = old_fd2_gamma

            old_lambda = float(master.BASE.BOTTOM_LAMBDA)
            master.BASE.BOTTOM_LAMBDA = 0.0
            random_k, random_runtime, random_ok = _select_random_numerov(master.BASE, ho_inverse, roots, k_count, seed + 1)
            methods.append(("Numerov+random", random_k, random_runtime, random_ok, 0.0))
            stratified = master.run_mode_frinet(ho_inverse, roots, k_count, seed + 2, bottom_lambda=0.0)
            methods.append(("Numerov+stratified", stratified["k"], stratified["runtime_seconds"], stratified["selection_success"], 0.0))
            moderate = master.run_mode_frinet(ho_inverse, roots, k_count, seed + 3, bottom_lambda=config.lambda_bottom)
            final_ablation_label = config.raw["ablation"]["methods"][3]
            methods.append((final_ablation_label, moderate["k"], moderate["runtime_seconds"], moderate["selection_success"], config.lambda_bottom))
            master.BASE.BOTTOM_LAMBDA = old_lambda

            for method, estimate, runtime, selection_ok, method_lambda in methods:
                metrics = _evaluate(estimate, truth, config.success_tolerance)
                metadata = common_metadata(config, spec, seed=seed, k=k_count, snr_db=snr_db, trial=trial)
                rows.append({
                    **metadata,
                    "experiment": "ablation",
                    "method": method,
                    "lambda_bottom": method_lambda,
                    "success": metrics["success"],
                    "rmse": metrics["rmse"],
                    "max_error": metrics["max_error"],
                    "runtime_seconds": float(runtime),
                    "achieved_SNR_db": achieved,
                    "failure_class": classify_failure(
                        exception=None, finite=np.isfinite(metrics["rmse"]), selection_success=selection_ok,
                        max_error=metrics["max_error"], tolerance=config.success_tolerance,
                    ),
                    "estimated_k": array_text(metrics["estimated"]),
                    "true_k": array_text(truth),
                    "data_seed": data_seed,
                })
            save_csv(csv_dir / f"ablation_trials_{spec.mode}.csv", rows)
    summary = _summarize(rows, ("method", "K", "SNR_db"))
    save_csv(csv_dir / f"ablation_summary_{spec.mode}.csv", summary)
    return rows, summary


def _set_bottom(base, cp: float, density: float) -> None:
    base.BOTTOM_CP = float(cp)
    base.BOTTOM_DENSITY = float(density)
    base.BOTTOM_BULK_K = float(base.OMEGA / base.BOTTOM_CP)


def run_environment(
    config: FinalConfig,
    spec: RunSpec,
    *,
    csv_dir: Path = CSV_DIR,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    master = _master()
    configure_backends(config)
    base = master.BASE
    original_cp = float(base.BOTTOM_CP)
    original_density = float(base.BOTTOM_DENSITY)
    kraken = base.load_kraken()
    k_count = int(config.raw["environment"]["K"])
    truth = np.real(np.asarray(kraken["complex_k"])[:k_count]).astype(np.float64)
    forward = np.asarray(kraken["mode_matrix"])[:, :k_count]
    scenario_specs = []
    for value in spec.environment_ssp_offsets_mps:
        scenario_specs.append(("SSP", float(value)))
    for value in spec.environment_bottom_cp_offsets_percent:
        if float(value) != 0.0:
            scenario_specs.append(("BOTTOM_CP", float(value)))
    for value in spec.environment_bottom_density_offsets_percent:
        if float(value) != 0.0:
            scenario_specs.append(("BOTTOM_DENSITY", float(value)))
    rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    try:
        for kind, value in scenario_specs:
            for trial in range(1, spec.environment_trials + 1):
                noisy, seed = master.create_noisy_dataset(forward, k_count, spec.environment_snr_db, trial)
                master.restore_nominal_environment()
                _set_bottom(base, original_cp, original_density)
                if kind == "SSP":
                    master.set_inverse_ssp_offset(value)
                elif kind == "BOTTOM_CP":
                    _set_bottom(base, original_cp * (1.0 + value / 100.0), original_density)
                else:
                    _set_bottom(base, original_cp, original_density * (1.0 + value / 100.0))
                start = time.perf_counter()
                decision_status = "NO_SINGLE_FREQUENCY_CALIBRATION_ATTEMPTED"
                calibration_accepted = 0
                relative_depth = float("nan")
                relative_span = float("nan")
                best_delta = float("nan")
                exception: Exception | None = None
                profile_exception: Exception | None = None
                result: dict[str, Any] = {"fit": float("nan"), "selection_success": False}
                try:
                    roots = base.calculate_physical_roots()
                    result = master.run_mode_frinet(noisy, roots, k_count, seed, config.lambda_bottom)
                    metrics = _evaluate(result["k"], truth, config.success_tolerance)
                except Exception as caught:
                    # Environment mismatch can make the requested K physically
                    # infeasible (for example only seven ordered roots for K=8).
                    # Preserve this as a negative result instead of aborting the
                    # remaining scenarios or silently changing K.
                    exception = caught
                    metrics = _evaluate([], truth, config.success_tolerance)
                    if "physical root" in str(caught).lower() or "物理根" in str(caught):
                        decision_status = "PHYSICAL_ROOT_INFEASIBLE_RECORDED"
                    else:
                        decision_status = "ENVIRONMENT_EVALUATION_FAILED_RECORDED"

                if exception is None and kind == "SSP":
                    try:
                        profile = master.scan_environment_profile(
                            noisy_data=noisy,
                            inverse_base_offset=value,
                            scenario_name=f"SSP_{value:+.1f}_pctNA",
                            snr_db=spec.environment_snr_db,
                            trial=trial,
                        )
                        decision = master.decide_environment_calibration(profile)
                        decision_status = decision["status"]
                        calibration_accepted = int(decision["accepted"])
                        relative_depth = float(profile.get("relative_depth", float("nan")))
                        relative_span = float(profile.get("relative_span", float("nan")))
                        best_delta = float(profile.get("best_delta", float("nan")))
                        for profile_row in profile.get("profile_rows", []):
                            profile_payload = dict(profile_row)
                            # Avoid a case-only duplicate with the standard
                            # metadata field SNR_db. Some CSV readers (notably
                            # PowerShell Import-Csv) treat names case-insensitively.
                            if "snr_db" in profile_payload:
                                profile_payload["profile_snr_db"] = profile_payload.pop("snr_db")
                            profile_rows.append({
                                **common_metadata(config, spec, seed=seed, k=k_count, snr_db=spec.environment_snr_db, trial=trial),
                                "environment_kind": kind,
                                "environment_offset": value,
                                **profile_payload,
                            })
                    except Exception as caught:
                        # The estimate remains valid even if the diagnostic
                        # profile itself fails; reject calibration safely and
                        # retain both the estimate and diagnostic exception.
                        profile_exception = caught
                        decision_status = "PROFILE_EVALUATION_FAILED_FALLBACK"
                        calibration_accepted = 0
                runtime = time.perf_counter() - start
                metadata = common_metadata(config, spec, seed=seed, k=k_count, snr_db=spec.environment_snr_db, trial=trial)
                rows.append({
                    **metadata,
                    "experiment": "environment",
                    "method": "HO-PG Mode-FRINET",
                    "environment_kind": kind,
                    "environment_offset": value,
                    "success": metrics["success"],
                    "rmse": metrics["rmse"],
                    "max_error": metrics["max_error"],
                    "runtime_seconds": runtime,
                    "selector_fit": float(result.get("fit", float("nan"))),
                    "calibration_accepted": calibration_accepted,
                    "decision_status": decision_status,
                    "profile_relative_depth": relative_depth,
                    "profile_relative_span": relative_span,
                    "profile_best_delta_mps": best_delta,
                    "failure_class": (
                        "METHOD_LIMITATION_OR_MODEL_MISMATCH"
                        if exception is not None and decision_status == "PHYSICAL_ROOT_INFEASIBLE_RECORDED"
                        else classify_failure(
                            exception=exception, finite=np.isfinite(metrics["rmse"]),
                            selection_success=bool(result.get("selection_success", False)),
                            max_error=metrics["max_error"], tolerance=config.success_tolerance,
                        )
                    ),
                    "exception": "" if exception is None else repr(exception),
                    "profile_exception": "" if profile_exception is None else repr(profile_exception),
                    "estimated_k": array_text(metrics["estimated"]),
                    "true_k": array_text(truth),
                })
                save_csv(csv_dir / f"environment_trials_{spec.mode}.csv", rows)
                save_csv(csv_dir / f"environment_identifiability_profile_{spec.mode}.csv", profile_rows)
    finally:
        master.restore_nominal_environment()
        _set_bottom(base, original_cp, original_density)
    summary = _summarize(rows, ("environment_kind", "environment_offset", "K", "SNR_db"))
    save_csv(csv_dir / f"environment_summary_{spec.mode}.csv", summary)
    return rows, summary


def run_sections(
    config: FinalConfig,
    spec: RunSpec,
    sections: Iterable[str],
    *,
    csv_dir: Path = CSV_DIR,
) -> dict[str, Any]:
    csv_dir.mkdir(parents=True, exist_ok=True)
    configure_backends(config)
    selected = tuple(dict.fromkeys(section.lower() for section in sections))
    output: dict[str, Any] = {}
    if "nominal" in selected:
        output["nominal"] = run_nominal(config, spec, csv_dir=csv_dir)
    if "baseline" in selected:
        output["baseline"] = run_baselines(config, spec, csv_dir=csv_dir)
    if "grid_resolution" in selected:
        output["grid_resolution"] = run_grid_resolution(config, spec, csv_dir=csv_dir)
    if "physics_validation" in selected:
        output["physics_validation"] = run_physics_validation(config, spec, csv_dir=csv_dir)
    if "fd_svd_native_reference" in selected:
        output["fd_svd_native_reference"] = run_fd_svd_native_reference(
            config, spec, csv_dir=csv_dir
        )
    if "ablation" in selected:
        output["ablation"] = run_ablation(config, spec, csv_dir=csv_dir)
    if "environment" in selected:
        output["environment"] = run_environment(config, spec, csv_dir=csv_dir)
    return output
