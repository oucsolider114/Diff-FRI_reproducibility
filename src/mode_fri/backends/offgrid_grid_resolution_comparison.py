# -*- coding: utf-8 -*-
"""
Off-Grid / Grid-Resolution Stress Test
======================================================================

目的
----------------------------------------------------------------------
比较：

    Grid-32
    Grid-64
    Grid-128
    Grid-256
    Grid-512

与：

    Mode-FRINET Raw

在扩大搜索区以后，对连续波数估计的：

    1. Success rate
    2. k-RMSE
    3. Max error
    4. Runtime

进行统一比较。

======================================================================
为什么做这个实验
----------------------------------------------------------------------

当前K=3 baseline中，GRID-512表现非常强，因为：

1. 搜索区很窄：
       [1.20, K_MAX]

2. 三个目标模态间隔较大；

3. 512点网格的步长已经约1e-4 rad/m；

因此该问题对dense grid非常友好。

本实验故意扩大搜索区：

    [1.08, K_MAX]

在相同物理参数范围下比较不同网格密度。

如果连续Mode-FRINET确有off-grid优势，则预期：

    coarse grid:
        quantization error明显

    dense grid:
        精度提高，但计算量增加

    Mode-FRINET:
        不存在固定grid quantization floor

======================================================================
注意
----------------------------------------------------------------------

这不是为了“设计一个Grid一定输的实验”。

扩大到1.08的理由是：
    该区间实际包含当前环境前8阶传播模态附近，
    属于合理物理搜索区，而不是人为设置无意义范围。

Forward:
    KRAKEN .MOD

数据：
    与baseline_comparison_final保持相同moving-source结构。

Target:
    K = 3

第一阶段先研究Grid resolution本身。
"""

from __future__ import annotations

import csv
import importlib.util
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# ======================================================================
# 路径
# ======================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]

BASELINE_SCRIPT = (
    Path(__file__).resolve().parent
    / "baseline_comparison_final.py"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "offgrid_grid_resolution_comparison"
)

FIGURE_DIR = (
    OUTPUT_DIR
    / "figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ======================================================================
# 动态加载已经跑通的baseline
# ======================================================================
def load_module(
    path: Path,
    module_name: str,
):

    if not path.exists():
        raise FileNotFoundError(
            f"\n没有找到：\n{path}\n"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"无法加载：{path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        module_name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


BASE = load_module(
    BASELINE_SCRIPT,
    "baseline_for_offgrid_test",
)

MULTI = BASE.MULTI
EXP26 = BASE.EXP26


# ======================================================================
# QUICK / FULL
# ======================================================================
QUICK_TEST = False



if QUICK_TEST:

    SNR_VALUES_DB = [
        5.0,
        15.0,
    ]

    NUMBER_OF_TRIALS = 2

    NUMBER_OF_RESTARTS = 2

    RUN_SUFFIX = "quick"

else:

    SNR_VALUES_DB = [
        -5.0,
        0.0,
        5.0,
        10.0,
        15.0,
        20.0,
        30.0,
    ]

    NUMBER_OF_TRIALS = 10

    NUMBER_OF_RESTARTS = 5

    RUN_SUFFIX = "full"


# ======================================================================
# Grid配置
# ======================================================================
GRID_SIZES = [
    32,
    64,
    128,
    256,
    512,
]

SEARCH_K_MIN = 1.08

SEARCH_K_MAX = float(
    EXP26.K_MAX
)

GRID_MIN_SEPARATION = 0.004


# ======================================================================
# Target
# ======================================================================
NUMBER_OF_TARGET_MODES = 3

SUCCESS_TOLERANCE = 1.5e-3

BASE_RANDOM_SEED = 20260920


# ======================================================================
# Mode-FRINET
#
# 为公平起见：
# FRINET也使用同样扩大的连续搜索范围。
# ======================================================================
FRINET_K_MIN = SEARCH_K_MIN

EXP26.K_MIN = float(
    FRINET_K_MIN
)

EXP26.NUMBER_OF_ATOMS = 24

EXP26.NUMBER_OF_SNAPSHOTS = (
    BASE.NUMBER_OF_RANGE_SAMPLES
)


# ======================================================================
# 输出
# ======================================================================
TRIAL_CSV = (
    OUTPUT_DIR
    / f"offgrid_trials_{RUN_SUFFIX}.csv"
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / f"offgrid_summary_{RUN_SUFFIX}.csv"
)

FRINET_RESTART_CSV = (
    OUTPUT_DIR
    / f"offgrid_frinet_restarts_{RUN_SUFFIX}.csv"
)


# ======================================================================
# CSV
# ======================================================================
def save_csv(
    path: Path,
    rows: list[dict],
):

    if not rows:
        return

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def array_text(
    values,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    ).ravel()

    return ";".join(
        f"{value:.12f}"
        for value
        in values
    )


# ======================================================================
# 评价指标
# ======================================================================
def evaluate_k(
    estimate,
    truth,
):

    estimate = np.sort(
        np.asarray(
            estimate,
            dtype=np.float64,
        )
    )[::-1]

    truth = np.sort(
        np.asarray(
            truth,
            dtype=np.float64,
        )
    )[::-1]

    if estimate.size != truth.size:

        return {
            "success": False,
            "rmse": float("inf"),
            "max_error": float("inf"),
        }

    errors = np.abs(
        estimate
        - truth
    )

    return {

        "success": bool(
            np.max(
                errors
            )
            <= SUCCESS_TOLERANCE
        ),

        "rmse": float(
            np.sqrt(
                np.mean(
                    errors ** 2
                )
            )
        ),

        "max_error": float(
            np.max(
                errors
            )
        ),
    }


# ======================================================================
# Grid Group-OMP
# ======================================================================
def grid_gomp(
    inverse_data: np.ndarray,
    grid_size: int,
):

    start = time.perf_counter()

    grid = np.linspace(
        SEARCH_K_MAX,
        SEARCH_K_MIN,
        grid_size,
        dtype=np.float64,
    )

    dictionary = np.asarray(
        EXP26.build_mode_matrix(
            grid
        ),
        dtype=np.complex128,
    )

    atom_norms = np.linalg.norm(
        dictionary,
        axis=0,
    )

    atom_norms = np.maximum(
        atom_norms,
        1e-15,
    )

    dictionary = (
        dictionary
        /
        atom_norms[
            None,
            :
        ]
    )

    residual = np.asarray(
        inverse_data,
        dtype=np.complex128,
    ).copy()

    selected_indices = []


    for _ in range(
        NUMBER_OF_TARGET_MODES
    ):

        correlations = (
            dictionary.conj().T
            @ residual
        )

        group_scores = np.linalg.norm(
            correlations,
            axis=1,
        )


        # --------------------------------------------------------------
        # 已选附近禁止重复选择
        # --------------------------------------------------------------
        for grid_index in range(
            grid_size
        ):

            candidate_k = (
                grid[
                    grid_index
                ]
            )

            too_close = any(

                abs(
                    candidate_k
                    -
                    grid[
                        old_index
                    ]
                )
                <
                GRID_MIN_SEPARATION

                for old_index
                in selected_indices
            )

            if too_close:

                group_scores[
                    grid_index
                ] = -np.inf


        best_index = int(
            np.argmax(
                group_scores
            )
        )

        selected_indices.append(
            best_index
        )


        selected_dictionary = (
            dictionary[
                :,
                selected_indices
            ]
        )


        coefficients, _, _, _ = (
            np.linalg.lstsq(
                selected_dictionary,
                inverse_data,
                rcond=None,
            )
        )


        residual = (
            inverse_data
            -
            selected_dictionary
            @ coefficients
        )


    estimated_k = np.sort(
        grid[
            selected_indices
        ]
    )[::-1]


    runtime = (
        time.perf_counter()
        - start
    )


    grid_step = float(
        abs(
            grid[1]
            - grid[0]
        )
    )


    return {

        "estimated_k": (
            estimated_k
        ),

        "runtime": float(
            runtime
        ),

        "grid_step": (
            grid_step
        ),
    }


# ======================================================================
# Mode-FRINET
# ======================================================================
def mode_frinet(
    inverse_data: np.ndarray,
    trial_seed: int,
):

    physical_roots = (
        MULTI.calculate_extended_roots()
    )

    restart_results = []

    total_start = (
        time.perf_counter()
    )


    for restart_index in range(
        NUMBER_OF_RESTARTS
    ):

        seed = (
            trial_seed
            +
            restart_index
            * 1009
        )


        restart_start = (
            time.perf_counter()
        )


        trained = (
            EXP26.train_one_restart(
                noisy_data=(
                    inverse_data
                ),
                random_seed=(
                    seed
                ),
            )
        )


        all_k = np.asarray(
            trained[
                "all_wavenumbers"
            ],
            dtype=np.float64,
        )


        activities = np.asarray(
            trained[
                "all_activities"
            ],
            dtype=np.float64,
        )


        selection = (
            MULTI.select_mode_order_beam(

                wavenumbers=(
                    all_k
                ),

                activities=(
                    activities
                ),

                noisy_data=(
                    inverse_data
                ),

                physical_roots=(
                    physical_roots
                ),

                number_of_modes=(
                    NUMBER_OF_TARGET_MODES
                ),

                training_k_min=(
                    FRINET_K_MIN
                ),

                training_k_max=(
                    EXP26.K_MAX
                ),
            )
        )


        restart_runtime = (
            time.perf_counter()
            -
            restart_start
        )


        if selection[
            "success"
        ]:

            raw_k = np.asarray(
                selection[
                    "selected"
                ],
                dtype=np.float64,
            )

            score = float(
                selection[
                    "fit"
                ]
            )

        else:

            raw_k = np.asarray(
                [],
                dtype=np.float64,
            )

            score = float(
                "inf"
            )


        restart_results.append(
            {

                "restart": (
                    restart_index + 1
                ),

                "seed": (
                    seed
                ),

                "selector_success": bool(
                    selection[
                        "success"
                    ]
                ),

                "reason": (
                    selection[
                        "reason"
                    ]
                ),

                "raw_k": (
                    raw_k
                ),

                "score": (
                    score
                ),

                "runtime": (
                    restart_runtime
                ),
            }
        )


    valid = [

        index

        for index, result
        in enumerate(
            restart_results
        )

        if (
            result[
                "selector_success"
            ]
            and
            np.isfinite(
                result[
                    "score"
                ]
            )
        )
    ]


    if not valid:

        return {

            "estimated_k": np.asarray(
                [],
                dtype=np.float64,
            ),

            "runtime": float(
                time.perf_counter()
                -
                total_start
            ),

            "best_restart": -1,

            "restart_results": (
                restart_results
            ),
        }


    best_index = min(

        valid,

        key=lambda index:
        restart_results[
            index
        ][
            "score"
        ],
    )


    estimated_k = np.asarray(
        restart_results[
            best_index
        ][
            "raw_k"
        ],
        dtype=np.float64,
    )


    total_runtime = (
        time.perf_counter()
        -
        total_start
    )


    return {

        "estimated_k": (
            estimated_k
        ),

        "runtime": float(
            total_runtime
        ),

        "best_restart": (
            best_index + 1
        ),

        "restart_results": (
            restart_results
        ),
    }


# ======================================================================
# Summary
# ======================================================================
def summarize(
    trial_rows,
):

    methods = [

        f"GRID_{size}"

        for size in GRID_SIZES

    ] + [

        "MODE_FRINET_RAW"
    ]


    rows = []


    for snr_db in SNR_VALUES_DB:

        for method in methods:

            subset = [

                row

                for row
                in trial_rows

                if (
                    float(
                        row[
                            "snr_db"
                        ]
                    )
                    ==
                    float(
                        snr_db
                    )
                    and
                    row[
                        "method"
                    ]
                    ==
                    method
                )
            ]


            if not subset:
                continue


            rmse = np.asarray(
                [
                    float(
                        row[
                            "rmse"
                        ]
                    )
                    for row
                    in subset
                ],
                dtype=np.float64,
            )


            max_error = np.asarray(
                [
                    float(
                        row[
                            "max_error"
                        ]
                    )
                    for row
                    in subset
                ],
                dtype=np.float64,
            )


            runtime = np.asarray(
                [
                    float(
                        row[
                            "runtime_seconds"
                        ]
                    )
                    for row
                    in subset
                ],
                dtype=np.float64,
            )


            finite_rmse = (
                rmse[
                    np.isfinite(
                        rmse
                    )
                ]
            )


            finite_max = (
                max_error[
                    np.isfinite(
                        max_error
                    )
                ]
            )


            grid_steps = np.asarray(
                [
                    float(
                        row[
                            "grid_step"
                        ]
                    )
                    for row
                    in subset
                    if np.isfinite(
                        float(
                            row[
                                "grid_step"
                            ]
                        )
                    )
                ],
                dtype=np.float64,
            )


            rows.append(
                {

                    "snr_db": (
                        snr_db
                    ),

                    "method": (
                        method
                    ),

                    "trials": (
                        len(
                            subset
                        )
                    ),

                    "success_rate": float(
                        np.mean(
                            [
                                float(
                                    row[
                                        "success"
                                    ]
                                )
                                for row
                                in subset
                            ]
                        )
                    ),

                    "rmse_median": (
                        float(
                            np.median(
                                finite_rmse
                            )
                        )
                        if finite_rmse.size
                        else float("nan")
                    ),

                    "max_error_median": (
                        float(
                            np.median(
                                finite_max
                            )
                        )
                        if finite_max.size
                        else float("nan")
                    ),

                    "runtime_seconds_mean": float(
                        np.mean(
                            runtime
                        )
                    ),

                    "grid_step": (
                        float(
                            np.mean(
                                grid_steps
                            )
                        )
                        if grid_steps.size
                        else float("nan")
                    ),
                }
            )


    return rows


# ======================================================================
# Figures
# ======================================================================
def plot_results(
    summary_rows,
):

    # ==============================================================
    # 1. RMSE vs grid size
    # ==============================================================

    for snr_db in SNR_VALUES_DB:

        grid_rows = [

            row

            for row
            in summary_rows

            if (
                float(
                    row[
                        "snr_db"
                    ]
                )
                ==
                float(
                    snr_db
                )
                and
                row[
                    "method"
                ].startswith(
                    "GRID_"
                )
            )
        ]


        grid_rows.sort(
            key=lambda row:
            int(
                row[
                    "method"
                ].split(
                    "_"
                )[1]
            )
        )


        if not grid_rows:
            continue


        sizes = np.asarray(
            [
                int(
                    row[
                        "method"
                    ].split(
                        "_"
                    )[1]
                )
                for row
                in grid_rows
            ]
        )


        rmse = np.asarray(
            [
                row[
                    "rmse_median"
                ]
                for row
                in grid_rows
            ],
            dtype=np.float64,
        )


        frinet_rows = [

            row

            for row
            in summary_rows

            if (
                float(
                    row[
                        "snr_db"
                    ]
                )
                ==
                float(
                    snr_db
                )
                and
                row[
                    "method"
                ]
                ==
                "MODE_FRINET_RAW"
            )
        ]


        fig, ax = plt.subplots(
            figsize=(
                8.0,
                5.3,
            )
        )


        ax.semilogy(
            sizes,
            rmse,
            marker="o",
            label="Grid-GOMP",
        )


        if frinet_rows:

            frinet_rmse = float(
                frinet_rows[
                    0
                ][
                    "rmse_median"
                ]
            )


            ax.axhline(
                frinet_rmse,
                linestyle="--",
                label="Mode-FRINET",
            )


        ax.set_xlabel(
            "Number of grid points"
        )

        ax.set_ylabel(
            "Median k-RMSE (rad/m)"
        )

        ax.set_title(
            f"Off-Grid Resolution Test, "
            f"SNR={snr_db:.0f} dB"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend()

        fig.tight_layout()


        fig.savefig(
            FIGURE_DIR
            / (
                f"rmse_vs_grid_"
                f"snr_{snr_db:+.0f}_"
                f"{RUN_SUFFIX}.png"
            ),
            dpi=220,
            bbox_inches="tight",
        )


        plt.close(
            fig
        )


    # ==============================================================
    # 2. Runtime vs RMSE
    # ==============================================================

    fig, ax = plt.subplots(
        figsize=(
            8.0,
            5.5,
        )
    )


    for method in (
        [
            f"GRID_{size}"
            for size
            in GRID_SIZES
        ]
        +
        [
            "MODE_FRINET_RAW"
        ]
    ):

        subset = [
            row
            for row
            in summary_rows
            if row[
                "method"
            ]
            ==
            method
        ]


        if not subset:
            continue


        runtime = float(
            np.mean(
                [
                    row[
                        "runtime_seconds_mean"
                    ]
                    for row
                    in subset
                ]
            )
        )


        rmse = float(
            np.median(
                [
                    row[
                        "rmse_median"
                    ]
                    for row
                    in subset
                ]
            )
        )


        ax.scatter(
            runtime,
            rmse,
        )


        ax.annotate(
            method,
            (
                runtime,
                rmse,
            ),
            xytext=(
                5,
                5,
            ),
            textcoords="offset points",
        )


    ax.set_xscale(
        "log"
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        "Mean runtime (s)"
    )

    ax.set_ylabel(
        "Median RMSE (rad/m)"
    )

    ax.set_title(
        "Accuracy-Runtime Tradeoff"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()


    fig.savefig(
        FIGURE_DIR
        / (
            f"runtime_vs_rmse_"
            f"{RUN_SUFFIX}.png"
        ),
        dpi=220,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


# ======================================================================
# Main
# ======================================================================
def main():

    torch.set_default_dtype(
        torch.float64
    )


    print(
        "=" * 170
    )

    print(
        "OFF-GRID / GRID-RESOLUTION COMPARISON"
    )

    print(
        "=" * 170
    )

    print(
        f"mode              = {RUN_SUFFIX}"
    )

    print(
        f"grid sizes        = {GRID_SIZES}"
    )

    print(
        f"search interval   = "
        f"[{SEARCH_K_MIN:.6f}, "
        f"{SEARCH_K_MAX:.6f}]"
    )

    print(
        f"SNR               = {SNR_VALUES_DB}"
    )

    print(
        f"trials            = {NUMBER_OF_TRIALS}"
    )

    print(
        f"FRINET restarts   = {NUMBER_OF_RESTARTS}"
    )

    print(
        "=" * 170
    )


    # ==================================================================
    # KRAKEN moving-source data
    # ==================================================================
    kraken = (
        BASE.load_kraken_environment()
    )


    forward = (
        BASE.generate_clean_moving_source_field(

            receiver_modes=(
                kraken[
                    "receiver_modes"
                ]
            ),

            source_modes=(
                kraken[
                    "source_modes"
                ]
            ),

            complex_k=(
                kraken[
                    "complex_k"
                ]
            ),
        )
    )


    clean_receiver = np.asarray(
        forward[
            "clean_receiver"
        ],
        dtype=np.complex128,
    )


    true_k = np.asarray(
        forward[
            "true_k"
        ],
        dtype=np.float64,
    )


    print(
        "\nKRAKEN truth:"
    )

    print(
        true_k
    )


    # ==================================================================
    # Monte Carlo
    # ==================================================================
    trial_rows = []

    restart_rows = []


    for snr_db in SNR_VALUES_DB:

        print(
            "\n\n"
            + "=" * 160
        )

        print(
            f"SNR = {snr_db:+.1f} dB"
        )

        print(
            "=" * 160
        )


        EXP26.SNR_DB = float(
            snr_db
        )


        for trial in range(
            1,
            NUMBER_OF_TRIALS + 1,
        ):

            noise_seed = (
                BASE_RANDOM_SEED
                +
                trial
                * 1000
                +
                int(
                    (
                        snr_db
                        + 20
                    )
                    * 100
                )
            )


            noisy_receiver, achieved_snr = (
                BASE.add_noise(

                    clean=(
                        clean_receiver
                    ),

                    snr_db=(
                        snr_db
                    ),

                    random_seed=(
                        noise_seed
                    ),
                )
            )


            common_receiver = (
                BASE.normalize_snapshot_columns(
                    noisy_receiver
                )
            )


            inverse_data = (
                BASE.adapt_to_inverse_grid(

                    receiver_data=(
                        common_receiver
                    ),

                    probe_k=(
                        true_k
                    ),
                )
            )


            print(
                f"\nTrial {trial}/"
                f"{NUMBER_OF_TRIALS} | "
                f"achieved SNR="
                f"{achieved_snr:.6f} dB"
            )


            # ==========================================================
            # Grid resolutions
            # ==========================================================
            for grid_size in GRID_SIZES:

                result = (
                    grid_gomp(
                        inverse_data=(
                            inverse_data
                        ),
                        grid_size=(
                            grid_size
                        ),
                    )
                )


                metrics = (
                    evaluate_k(
                        result[
                            "estimated_k"
                        ],
                        true_k,
                    )
                )


                print(
                    f"  GRID-{grid_size:<3d} | "
                    f"step="
                    f"{result['grid_step']:.3e} | "
                    f"success="
                    f"{metrics['success']} | "
                    f"RMSE="
                    f"{metrics['rmse']:.3e} | "
                    f"k="
                    f"{result['estimated_k']}"
                )


                trial_rows.append(
                    {

                        "snr_db": (
                            snr_db
                        ),

                        "trial": (
                            trial
                        ),

                        "method": (
                            f"GRID_{grid_size}"
                        ),

                        "grid_size": (
                            grid_size
                        ),

                        "grid_step": (
                            result[
                                "grid_step"
                            ]
                        ),

                        "success": int(
                            metrics[
                                "success"
                            ]
                        ),

                        "rmse": (
                            metrics[
                                "rmse"
                            ]
                        ),

                        "max_error": (
                            metrics[
                                "max_error"
                            ]
                        ),

                        "runtime_seconds": (
                            result[
                                "runtime"
                            ]
                        ),

                        "estimated_k": (
                            array_text(
                                result[
                                    "estimated_k"
                                ]
                            )
                        ),
                    }
                )


            # ==========================================================
            # Mode-FRINET
            # ==========================================================
            model_seed = (
                BASE_RANDOM_SEED
                +
                500000
                +
                trial
                * 10000
                +
                int(
                    (
                        snr_db
                        + 20
                    )
                    * 1000
                )
            )


            frinet = (
                mode_frinet(
                    inverse_data=(
                        inverse_data
                    ),
                    trial_seed=(
                        model_seed
                    ),
                )
            )


            frinet_metrics = (
                evaluate_k(
                    frinet[
                        "estimated_k"
                    ],
                    true_k,
                )
            )


            print(
                f"  MODE-FRINET | "
                f"success="
                f"{frinet_metrics['success']} | "
                f"RMSE="
                f"{frinet_metrics['rmse']:.3e} | "
                f"k="
                f"{frinet['estimated_k']}"
            )


            trial_rows.append(
                {

                    "snr_db": (
                        snr_db
                    ),

                    "trial": (
                        trial
                    ),

                    "method": (
                        "MODE_FRINET_RAW"
                    ),

                    "grid_size": (
                        0
                    ),

                    "grid_step": (
                        float("nan")
                    ),

                    "success": int(
                        frinet_metrics[
                            "success"
                        ]
                    ),

                    "rmse": (
                        frinet_metrics[
                            "rmse"
                        ]
                    ),

                    "max_error": (
                        frinet_metrics[
                            "max_error"
                        ]
                    ),

                    "runtime_seconds": (
                        frinet[
                            "runtime"
                        ]
                    ),

                    "estimated_k": (
                        array_text(
                            frinet[
                                "estimated_k"
                            ]
                        )
                    ),
                }
            )


            for restart in (
                frinet[
                    "restart_results"
                ]
            ):

                restart_rows.append(
                    {

                        "snr_db": (
                            snr_db
                        ),

                        "trial": (
                            trial
                        ),

                        "restart": (
                            restart[
                                "restart"
                            ]
                        ),

                        "seed": (
                            restart[
                                "seed"
                            ]
                        ),

                        "selector_success": int(
                            restart[
                                "selector_success"
                            ]
                        ),

                        "reason": (
                            restart[
                                "reason"
                            ]
                        ),

                        "score": (
                            restart[
                                "score"
                            ]
                        ),

                        "runtime_seconds": (
                            restart[
                                "runtime"
                            ]
                        ),

                        "raw_k": (
                            array_text(
                                restart[
                                    "raw_k"
                                ]
                            )
                        ),
                    }
                )


            # checkpoint
            save_csv(
                TRIAL_CSV,
                trial_rows,
            )

            save_csv(
                FRINET_RESTART_CSV,
                restart_rows,
            )

            save_csv(
                SUMMARY_CSV,
                summarize(
                    trial_rows
                ),
            )


    # ==================================================================
    # Final
    # ==================================================================
    summary_rows = (
        summarize(
            trial_rows
        )
    )


    save_csv(
        SUMMARY_CSV,
        summary_rows,
    )


    plot_results(
        summary_rows
    )


    print(
        "\n\n"
        + "=" * 180
    )

    print(
        "OFF-GRID SUMMARY"
    )

    print(
        "=" * 180
    )


    for row in summary_rows:

        print(
            f"SNR="
            f"{row['snr_db']:>5.1f} | "
            f"{row['method']:<18} | "
            f"success="
            f"{100*row['success_rate']:6.1f}% | "
            f"RMSE="
            f"{row['rmse_median']:.3e} | "
            f"max="
            f"{row['max_error_median']:.3e} | "
            f"time="
            f"{row['runtime_seconds_mean']:.4f}s | "
            f"step="
            f"{row['grid_step']:.3e}"
        )


    print(
        "\nOutputs:"
    )

    print(
        SUMMARY_CSV
    )

    print(
        TRIAL_CSV
    )

    print(
        FRINET_RESTART_CSV
    )

    print(
        FIGURE_DIR
    )


if __name__ == "__main__":
    main()
