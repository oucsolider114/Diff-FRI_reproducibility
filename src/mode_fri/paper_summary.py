"""Generate the final-paper narrative strictly from generated CSV files."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from .paper_io import PAPER_CSV_DIR, PAPER_SUMMARY_PATH


def _read(csv_dir: Path, name: str) -> list[dict[str, str]]:
    path = csv_dir / name
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(value: str | float) -> str:
    number = _number(str(value))
    return f"{number:.6g}" if np.isfinite(number) else str(value)


def _table(rows: list[dict[str, str]], columns: tuple[str, ...]) -> str:
    if not rows:
        return "尚无对应 CSV 结果。\n"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def _sort_rows(rows: list[dict[str, str]], columns: tuple[str, ...]) -> list[dict[str, str]]:
    """Sort report rows numerically where possible, without modifying CSV data."""

    def sort_value(row: dict[str, str], column: str) -> tuple[int, float | str]:
        value = row.get(column, "")
        number = _number(value)
        if np.isfinite(number):
            return (0, number)
        return (1, value)

    return sorted(rows, key=lambda row: tuple(sort_value(row, column) for column in columns))


def _failure_summary(csv_dir: Path, mode: str) -> str:
    counts: Counter[str] = Counter()
    total = 0
    for stem in ("nominal", "baseline", "grid_resolution", "ablation", "environment"):
        for row in _read(csv_dir, f"{stem}_trials_{mode}.csv"):
            total += 1
            counts[row.get("failure_class", "UNCLASSIFIED")] += 1
    if not total:
        return "尚无逐试验 CSV。"
    return "；".join(f"{name}={count}" for name, count in sorted(counts.items())) + f"（总计 {total}）"


def generate_paper_summary(
    mode: str,
    *,
    csv_dir: Path = PAPER_CSV_DIR,
    output_path: Path = PAPER_SUMMARY_PATH,
) -> Path:
    nominal = _read(csv_dir, f"nominal_summary_{mode}.csv")
    baseline = _read(csv_dir, f"baseline_summary_{mode}.csv")
    grid_resolution = _read(csv_dir, f"grid_resolution_summary_{mode}.csv")
    ablation = _read(csv_dir, f"ablation_summary_{mode}.csv")
    environment = _read(csv_dir, f"environment_summary_{mode}.csv")
    environment_trials = _read(csv_dir, f"environment_trials_{mode}.csv")
    physics = _read(csv_dir, "high_order_discretization_summary.csv")
    physics_best = _read(csv_dir, "best_physics_variant_summary.csv")
    fd_svd_native = _read(csv_dir, f"fd_svd_native_reference_summary_{mode}.csv")

    nominal = _sort_rows(nominal, ("K", "SNR_db", "method"))
    baseline = _sort_rows(baseline, ("method", "K", "SNR_db"))
    grid_resolution = _sort_rows(grid_resolution, ("grid_size", "K", "SNR_db"))
    environment = _sort_rows(environment, ("environment_kind", "environment_offset", "K", "SNR_db"))

    negative_nominal = [row for row in nominal if _number(row.get("success_rate", "nan")) < 1.0]
    negative_environment = [row for row in environment if _number(row.get("success_rate", "nan")) < 1.0]
    decision_counts = Counter(row.get("decision_status", "") for row in environment_trials)
    calibration_accepts = sum(int(_number(row.get("calibration_accepted", "0"))) for row in environment_trials)

    physics_comparison: list[dict[str, str]] = []
    for solver, dz_m, role in (
        ("FD2", 1.0, "legacy FD2 reference"),
        ("NUMEROV4", 0.5, "frozen final representation"),
    ):
        match = next(
            (
                row
                for row in physics
                if row.get("solver") == solver
                and np.isclose(_number(row.get("dz_m", "nan")), dz_m, rtol=0.0, atol=1e-12)
            ),
            None,
        )
        if match is not None:
            physics_comparison.append({**match, "comparison_role": role})
    physics_reduction = float("nan")
    if len(physics_comparison) == 2:
        fd2_high = _number(physics_comparison[0].get("mean_mismatch_modes_6_8", "nan"))
        numerov_high = _number(physics_comparison[1].get("mean_mismatch_modes_6_8", "nan"))
        if np.isfinite(fd2_high) and fd2_high > 0 and np.isfinite(numerov_high):
            physics_reduction = (fd2_high - numerov_high) / fd2_high

    baseline_winners: list[str] = []
    for snr in sorted({_number(row.get("SNR_db", "nan")) for row in baseline if np.isfinite(_number(row.get("SNR_db", "nan")))}):
        candidates = [row for row in baseline if _number(row.get("SNR_db", "nan")) == snr]
        finite = [row for row in candidates if np.isfinite(_number(row.get("rmse_median", "nan")))]
        if finite:
            winner = min(finite, key=lambda row: _number(row.get("rmse_median", "nan")))
            baseline_winners.append(
                f"- SNR={snr:g} dB：最低中位 RMSE 为 {winner.get('method')}，"
                f"RMSE={_fmt(winner.get('rmse_median', ''))}。"
            )

    content = f"""# NN_FRI_Mode_Extension 最终结果摘要

> 本文档由 `outputs/final_paper/csv/` 自动生成；表内数字不手工录入。当前结果模式：`{mode}`。

## 冻结方法与评价边界

最终主方法为 HO-PG Mode-FRINET：continuous off-grid FRI/FRINET modal estimation、Numerov4 高阶可微模态表示、真实海底 Robin 正则、mode-order constrained physics-stratified continuous atoms 与 multi-snapshot sparse estimation。固定参数为 `dz=0.5 m`、`lambda_bottom=0.1`、`selector_gamma=0`、`M=24`、`L=20`，最终结果不使用 safe-root refinement。

KRAKEN 模态波数真值只在估计结束后的 CSV 评价字段中使用，不进入 FD-SVD、Grid-GOMP、FD2 或 HO-PG 的估计与选择过程。本工程处理的是 KRAKEN 真实海洋环境仿真数据，不是海上实测数据。

## High-order physics discretization validation

该确定性验证直接复用既有 high-order bottom discretization 后端，以 KRAKEN mode shape 作为估计完成后的物理表示误差评价参考，不修改最终估计器。Modes 6–8 的结果为：

{_table(physics_comparison, ('comparison_role', 'solver', 'dz_m', 'mean_mismatch_modes_6_8', 'max_mismatch_modes_6_8', 'mean_mismatch_modes_1_8'))}

由上述 CSV 计算，FD2 `dz=1.0 m` 到最终 Numerov4 `dz=0.5 m` 的 modes 6–8 平均 mismatch 相对降低为 {_fmt(100.0 * physics_reduction) if np.isfinite(physics_reduction) else '尚无完整数据'}%。完整变体最优行保存在 `best_physics_variant_summary.csv`（共 {len(physics_best)} 行）。可支持的表述是：**high-order differentiable realistic-bottom modal representation significantly reduces high-order KRAKEN/inverse mode-shape mismatch**；不能把 Numerov 本身表述为创新。

## Nominal Monte Carlo

{_table(nominal, ('method', 'K', 'SNR_db', 'runs', 'success_rate', 'rmse_median', 'rmse_p90', 'runtime_seconds_mean'))}

未达到 100% 成功率的 nominal 条目：{len(negative_nominal)}。这类结果保留为噪声条件下的随机优化失败或方法局限，不继续针对单点调参。

## 公平 baseline

所有方法在同一 `(SNR, trial)` 内共用同一噪声数据 realization；比较表如下。

{_table(baseline, ('method', 'K', 'SNR_db', 'runs', 'success_rate', 'rmse_median', 'rmse_p90', 'runtime_seconds_mean'))}

按 CSV 中位 RMSE 逐 SNR 的最低者：

{chr(10).join(baseline_winners) if baseline_winners else '- 尚无完整 baseline CSV。'}

## Paper-native FD-SVD sanity/reference validation

主 baseline 是所有方法使用相同数据预算的 `L=20` 公平比较。下表是额外的 FD-SVD 论文兼容工作点：`NR=64`、`R0=4000 m`、range aperture `=2×849.2 m`。它只用于证明参考方法在其原生观测几何下能够正常运行，不能与 `L=20` 主比较的运行时间或数据预算直接视作完全公平比较。

{_table(_sort_rows(fd_svd_native, ('SNR_db',)), ('method', 'NR', 'K', 'SNR_db', 'runs', 'success_rate', 'rmse_median', 'rmse_p90', 'runtime_seconds_mean'))}

## Grid-resolution comparison

Grid-128/256/512 使用与公平 baseline 相同的 `(SNR, trial)` 数据 realization；真值仍只在估计完成后用于评价。

{_table(grid_resolution, ('method', 'grid_size', 'K', 'SNR_db', 'runs', 'success_rate', 'rmse_median', 'runtime_seconds_mean'))}

## Ablation

{_table(ablation, ('method', 'K', 'SNR_db', 'runs', 'success_rate', 'rmse_median', 'rmse_p90', 'runtime_seconds_mean'))}

该表用于区分 FD2/Numerov 表示、random/stratified atom formation 与固定 `lambda_bottom=0.1` 的实际影响；不把 Numerov、Robin、Adam、NMS 或 KRAKEN 本身表述为创新。

## 环境失配、失败条件与辨识限制

{_table(environment, ('environment_kind', 'environment_offset', 'K', 'SNR_db', 'runs', 'success_rate', 'rmse_median', 'rmse_p90'))}

环境条目中未达到 100% 成功率的条件数：{len(negative_environment)}。环境校准接受次数：{calibration_accepts}。诊断状态计数：{'; '.join(f'{key}={value}' for key, value in sorted(decision_counts.items())) if decision_counts else '尚无数据'}。

单频环境 profile 只用于辨识性诊断与 safe fallback。不可辨识、曲线过平或边界最优时禁止自动校准；不得据此宣称已经解决单频 SSP 或海底参数辨识。SSP、bottom cp、bottom density 的负面结果均保留在逐试验 CSV。

## 失败分类

{_failure_summary(csv_dir, mode)}

## 可支持的最终贡献表述

- continuous off-grid FRI/FRINET 模态波数估计；
- high-order differentiable realistic-bottom modal representation；
- mode-order constrained continuous atom formation；
- 上述组合对高阶模态估计的实际改善，以及由 CSV 明确界定的噪声和环境失配边界。

APR、APR+VP、单频 Joint-SSP 与自动 SSP 校准只作为归档机制实验，不属于最终主方法。结果不支持“真实海上数据已验证”“环境不确定性已解决”或“全部条件 100% 成功”等表述。
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path
