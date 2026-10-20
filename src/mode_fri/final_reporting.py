"""Generate the final figures and publication-ready Markdown tables."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .final_io import CSV_DIR, FIGURE_DIR, TABLE_DIR, ensure_output_tree


def _read(name: str, csv_dir: Path = CSV_DIR) -> list[dict[str, str]]:
    path = csv_dir / name
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _save(fig, name: str, figure_dir: Path = FIGURE_DIR) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_dir / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _line_by_k(rows, metric, ylabel, title, name, *, log=False, figure_dir: Path = FIGURE_DIR):
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for k in sorted({int(float(row["K"])) for row in rows}):
        subset = sorted((row for row in rows if int(float(row["K"])) == k), key=lambda row: _float(row, "SNR_db"))
        ax.plot([_float(row, "SNR_db") for row in subset], [_float(row, metric) for row in subset], marker="o", label=f"K={k}")
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, name, figure_dir)


def _plot_modewise(mode: str, csv_dir: Path, figure_dir: Path) -> None:
    rows = _read(f"nominal_modewise_{mode}.csv", csv_dir)
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for k in sorted({int(float(row["K"])) for row in rows}):
        grouped: dict[int, list[float]] = {}
        for row in rows:
            if int(float(row["K"])) == k:
                grouped.setdefault(int(row["mode"]), []).append(_float(row, "abs_error"))
        modes = sorted(grouped)
        ax.plot(
            modes,
            [float(np.median(grouped[index])) for index in modes],
            marker="o",
            label=f"K={k}",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Mode order")
    ax.set_ylabel("Median |wavenumber error| (rad/m)")
    ax.set_title("Mode-wise HO-PG Mode-FRINET Error")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    _save(fig, "mode_wise_error.png", figure_dir)


def _plot_grid(mode: str, csv_dir: Path, figure_dir: Path) -> None:
    rows = _read(f"grid_resolution_summary_{mode}.csv", csv_dir)
    if not rows:
        rows = _read(f"baseline_summary_{mode}.csv", csv_dir)
    subset = [
        row for row in rows
        if row.get("method", "").startswith("Grid-")
        and row.get("method", "").split("-")[-1].isdigit()
    ]
    if not subset:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    for snr in sorted({_float(row, "SNR_db") for row in subset}):
        snr_rows = [row for row in subset if _float(row, "SNR_db") == snr]
        snr_rows.sort(key=lambda row: int(row["method"].split("-")[1]))
        sizes = [int(row["method"].split("-")[1]) for row in snr_rows]
        ax.plot(
            sizes,
            [_float(row, "rmse_median") for row in snr_rows],
            marker="o",
            label=f"{snr:g} dB",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Grid size")
    ax.set_ylabel("Median RMSE (rad/m)")
    ax.set_title("Grid-resolution Comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, "grid_resolution_comparison.png", figure_dir)


def _plot_baseline(mode: str, csv_dir: Path, figure_dir: Path) -> None:
    rows = _read(f"baseline_summary_{mode}.csv", csv_dir)
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for method in sorted({row.get("method", "") for row in rows}):
        subset = sorted(
            (row for row in rows if row.get("method") == method),
            key=lambda row: _float(row, "SNR_db"),
        )
        ax.plot(
            [_float(row, "SNR_db") for row in subset],
            [_float(row, "rmse_median") for row in subset],
            marker="o",
            label=method,
        )
    ax.set_yscale("log")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Median RMSE (rad/m)")
    ax.set_title("Fair Baseline Comparison")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, "baseline_comparison.png", figure_dir)


def _plot_fd2_numerov(mode: str, csv_dir: Path, figure_dir: Path) -> None:
    rows = _read(f"ablation_summary_{mode}.csv", csv_dir)
    if not rows:
        return
    order = [
        "FD2+random",
        "Numerov+random",
        "Numerov+stratified",
        "Numerov+stratified+moderate physics",
        "Numerov+stratified+lambda0.1",
    ]
    values = []
    labels = []
    for method in order:
        subset = [_float(row, "rmse_median") for row in rows if row.get("method") == method]
        if subset:
            labels.append(method.replace("+", "+\n"))
            values.append(float(np.median(subset)))
    if not values:
        return
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.bar(np.arange(len(values)), values)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(values)), labels)
    ax.set_ylabel("Median RMSE (rad/m)")
    ax.set_title("FD2 vs Numerov and Atom-formation Ablation")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "fd2_vs_numerov_mismatch.png", figure_dir)


def _plot_high_order_physics(csv_dir: Path, figure_dir: Path) -> None:
    """Plot the frozen FD2 dz=1 vs Numerov4 dz=0.5 mode-shape evidence."""

    rows = _read("high_order_discretization_modewise.csv", csv_dir)
    variants = (
        ("FD2", 1.0, "FD2, dz=1.0 m"),
        ("NUMEROV4", 0.5, "Numerov4, dz=0.5 m"),
    )
    selected = []
    for solver, dz_m, label in variants:
        subset = sorted(
            (
                row
                for row in rows
                if row.get("solver") == solver
                and np.isclose(_float(row, "dz_m"), dz_m, rtol=0.0, atol=1e-12)
            ),
            key=lambda row: int(float(row["mode"])),
        )
        if subset:
            selected.append((label, subset))
    if len(selected) != 2:
        return
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for label, subset in selected:
        ax.plot(
            [int(float(row["mode"])) for row in subset],
            [_float(row, "shape_mismatch") for row in subset],
            marker="o",
            label=label,
        )
    ax.axvspan(5.5, 8.5, color="tab:orange", alpha=0.08, label="high-order modes 6-8")
    ax.set_yscale("log")
    ax.set_xlabel("Mode order")
    ax.set_ylabel("KRAKEN/inverse mode-shape mismatch")
    ax.set_title("High-order Realistic-bottom Modal Discretization Validation")
    ax.set_xticks(range(1, 9))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, "high_order_mode_shape_mismatch.png", figure_dir)


def _plot_environment(mode: str, csv_dir: Path, figure_dir: Path) -> None:
    rows = _read(f"environment_summary_{mode}.csv", csv_dir)
    if not rows:
        return
    finite_rmse = [
        _float(row, "rmse_median")
        for row in rows
        if np.isfinite(_float(row, "rmse_median")) and _float(row, "rmse_median") > 0
    ]
    infeasible_y = min(finite_rmse) / 2.0 if finite_rmse else 1.0e-6
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), sharey=True)
    for ax, kind, xlabel in zip(
        axes,
        ("SSP", "BOTTOM_CP", "BOTTOM_DENSITY"),
        ("SSP offset (m/s)", "Bottom cp offset (%)", "Bottom density offset (%)"),
    ):
        subset = sorted((row for row in rows if row.get("environment_kind") == kind), key=lambda row: _float(row, "environment_offset"))
        if subset:
            offsets = np.asarray([_float(row, "environment_offset") for row in subset], dtype=float)
            values = np.asarray([_float(row, "rmse_median") for row in subset], dtype=float)
            valid = np.isfinite(values) & (values > 0)
            if np.any(valid):
                ax.plot(offsets[valid], values[valid], marker="o")
            invalid = ~valid
            if np.any(invalid):
                ax.scatter(
                    offsets[invalid], np.full(np.count_nonzero(invalid), infeasible_y),
                    color="crimson", marker="x", s=75, linewidths=2.0,
                    label="infeasible / no finite RMSE",
                )
                for offset in offsets[invalid]:
                    ax.annotate(
                        "infeasible", (offset, infeasible_y), xytext=(0, 7),
                        textcoords="offset points", ha="center", color="crimson", fontsize=7,
                    )
                ax.legend(fontsize=7, loc="best")
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
    axes[0].set_ylabel("Median RMSE (rad/m)")
    fig.suptitle("Environment Mismatch (No Claim of Single-frequency Identification)")
    _save(fig, "environment_mismatch.png", figure_dir)


def _plot_identifiability(mode: str, csv_dir: Path, figure_dir: Path) -> None:
    rows = _read(f"environment_identifiability_profile_{mode}.csv", csv_dir)
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    # Plot one representative trial for every SSP scenario. Taking the first
    # eight raw (scenario, trial) keys would over-represent the first scenario
    # in FULL, where each scenario has five trials.
    representative: dict[str, str] = {}
    for row in rows:
        scenario = row.get("scenario", "")
        trial = row.get("trial", "")
        if scenario and scenario not in representative:
            representative[scenario] = trial
    for scenario, trial in representative.items():
        subset = [row for row in rows if row.get("scenario") == scenario and row.get("trial") == trial and row.get("stage") == "COARSE"]
        subset.sort(key=lambda row: _float(row, "delta_c_mps"))
        if subset:
            offset = _float(subset[0], "environment_offset")
            label = f"SSP {offset:+g} m/s, trial {trial}"
            ax.plot([_float(row, "delta_c_mps") for row in subset], [_float(row, "fit") for row in subset], label=label)
    ax.set_xlabel("Candidate SSP correction (m/s)")
    ax.set_ylabel("Root-constrained relative residual")
    ax.set_title("Single-frequency Environment-identifiability Profiles")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    _save(fig, "environment_identifiability_profile.png", figure_dir)


def _markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    if not rows:
        return "No results were generated.\n"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            try:
                number = float(value)
                value = f"{number:.6g}"
            except (TypeError, ValueError):
                pass
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _sort_rows(rows: list[dict[str, str]], columns: tuple[str, ...]) -> list[dict[str, str]]:
    """Sort report rows numerically where possible without changing source CSVs."""

    def value(row: dict[str, str], column: str) -> tuple[int, float | str]:
        raw = row.get(column, "")
        try:
            number = float(raw)
        except (TypeError, ValueError):
            return (1, raw)
        if np.isfinite(number):
            return (0, number)
        return (1, raw)

    return sorted(rows, key=lambda row: tuple(value(row, column) for column in columns))


def generate_final_artifacts(
    mode: str,
    *,
    csv_dir: Path = CSV_DIR,
    figure_dir: Path = FIGURE_DIR,
    table_dir: Path = TABLE_DIR,
) -> list[Path]:
    for path in (csv_dir, figure_dir, table_dir):
        path.mkdir(parents=True, exist_ok=True)
    nominal = _read(f"nominal_summary_{mode}.csv", csv_dir)
    _line_by_k(nominal, "rmse_median", "Median RMSE (rad/m)", "Nominal RMSE vs SNR", "rmse_vs_snr.png", log=True, figure_dir=figure_dir)
    _line_by_k(nominal, "success_rate", "Success rate", "Nominal Success vs SNR", "success_vs_snr.png", figure_dir=figure_dir)
    _plot_modewise(mode, csv_dir, figure_dir)
    _plot_grid(mode, csv_dir, figure_dir)
    _plot_baseline(mode, csv_dir, figure_dir)
    _plot_fd2_numerov(mode, csv_dir, figure_dir)
    _plot_high_order_physics(csv_dir, figure_dir)
    _plot_environment(mode, csv_dir, figure_dir)
    _plot_identifiability(mode, csv_dir, figure_dir)

    physics_rows = _read("high_order_discretization_summary.csv", csv_dir)
    physics_comparison = []
    for solver, dz_m, role in (
        ("FD2", 1.0, "legacy FD2 reference"),
        ("NUMEROV4", 0.5, "frozen final representation"),
    ):
        for row in physics_rows:
            if row.get("solver") == solver and np.isclose(
                _float(row, "dz_m"), dz_m, rtol=0.0, atol=1e-12
            ):
                physics_comparison.append({**row, "comparison_role": role})
                break

    table_specs = {
        f"nominal_table_{mode}.md": (_sort_rows(nominal, ("K", "SNR_db", "method")), ["method", "K", "SNR_db", "runs", "success_rate", "rmse_median", "rmse_p90", "runtime_seconds_mean"]),
        f"baseline_table_{mode}.md": (_sort_rows(_read(f"baseline_summary_{mode}.csv", csv_dir), ("method", "K", "SNR_db")), ["method", "K", "SNR_db", "runs", "success_rate", "rmse_median", "runtime_seconds_mean"]),
        f"grid_resolution_table_{mode}.md": (_sort_rows(_read(f"grid_resolution_summary_{mode}.csv", csv_dir), ("grid_size", "K", "SNR_db")), ["method", "grid_size", "K", "SNR_db", "runs", "success_rate", "rmse_median", "runtime_seconds_mean"]),
        f"ablation_table_{mode}.md": (_read(f"ablation_summary_{mode}.csv", csv_dir), ["method", "K", "SNR_db", "runs", "success_rate", "rmse_median", "runtime_seconds_mean"]),
        f"environment_table_{mode}.md": (_sort_rows(_read(f"environment_summary_{mode}.csv", csv_dir), ("environment_kind", "environment_offset", "K", "SNR_db")), ["environment_kind", "environment_offset", "K", "SNR_db", "runs", "success_rate", "rmse_median"]),
        "high_order_physics_table.md": (physics_comparison, ["comparison_role", "solver", "dz_m", "mean_mismatch_modes_6_8", "max_mismatch_modes_6_8", "mean_mismatch_modes_1_8"]),
        f"fd_svd_native_reference_table_{mode}.md": (_sort_rows(_read(f"fd_svd_native_reference_summary_{mode}.csv", csv_dir), ("SNR_db",)), ["method", "NR", "K", "SNR_db", "runs", "success_rate", "rmse_median", "rmse_p90", "runtime_seconds_mean"]),
    }
    generated = []
    for name, (rows, columns) in table_specs.items():
        path = table_dir / name
        path.write_text(_markdown_table(rows, columns), encoding="utf-8")
        generated.append(path)
    generated.extend(sorted(figure_dir.glob("*.png")))
    return generated
