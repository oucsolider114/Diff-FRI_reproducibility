# -*- coding: utf-8 -*-
"""
Physics Weight Ablation for High-Order Mode-FRINET
======================================================================

目的
----------------------------------------------------------------------
回答一个关键问题：

    K=8取得约1e-5精度，

到底主要来自：

A. Numerov高精度连续物理原子 + 数据拟合

还是：

B. 强Robin bottom physics penalty把k直接拉向已知环境本征根？

======================================================================
固定内容
----------------------------------------------------------------------

- K = 8
- KRAKEN .MOD forward
- Numerov4, dz=0.5 m
- physics-stratified continuous trust regions
- M=24
- L=20
- 同一SNR下：
    同一clean data
    同一noise
    同一atom initialization
    同一model seed

因此不同lambda_b之间只有：

    bottom physics penalty weight

发生改变。

======================================================================
重要：干净消融
----------------------------------------------------------------------

本实验主消融将：

    BASE.PHYSICS_SCORE_GAMMA = 0

也就是说selector不再根据Robin residual给候选重新加权。

selector只剩：

    mode-order cell
    activity candidate formation
    joint data-fit

同时：

    不运行safe_refine

防止任何“snap to physical roots”。

======================================================================
扫描
----------------------------------------------------------------------

lambda_b =

    0
    1e-4
    1e-3
    1e-2
    1e-1
    1

另外增加：

    CURRENT_PIPELINE_REF

即：

    lambda_b = 1
    selector gamma = 原来的5000

用于与刚才V2结果做参考。

======================================================================
注意
----------------------------------------------------------------------

lambda_b = 0 不能称为完全data-only。

因为mode-order trust regions仍由物理环境定义。

准确叫法是：

    NO BOTTOM-PENALTY

这一步只消融Robin penalty。
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


# ======================================================================
# PATH
# ======================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]

V2_SCRIPT = (
    Path(__file__).resolve().parent
    / "high_order_physics_mode_frinet_final_v2.py"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "physics_weight_ablation_final"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ======================================================================
# LOAD MODULE
# ======================================================================
def load_module(
    path: Path,
    module_name: str,
):

    if not path.exists():

        raise FileNotFoundError(
            "\n没有找到依赖文件：\n"
            f"{path}\n\n"
            "本实验需要保留：\n"
            "high_order_physics_mode_frinet_final.py\n"
            "high_order_physics_mode_frinet_final_v2.py\n"
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

    sys.modules[module_name] = module

    spec.loader.exec_module(
        module
    )

    return module


V2 = load_module(
    V2_SCRIPT,
    "ho_mode_frinet_v2_for_ablation",
)

BASE = V2.BASE


# ======================================================================
# CONFIG
# ======================================================================
TARGET_MODE_COUNT = 8

SNR_VALUES_DB = [
    15.0,
    30.0,
]

NUMBER_OF_TRIALS = 1

BASE_RANDOM_SEED = BASE.BASE_RANDOM_SEED


# ======================================================================
# PHYSICS WEIGHT SWEEP
# ======================================================================
BOTTOM_LAMBDAS = [
    0.0,
    1.0e-4,
    1.0e-3,
    1.0e-2,
    1.0e-1,
    1.0,
]


# ======================================================================
# Clean selector:
#
# 不使用bottom residual进行selector physics weighting
# ======================================================================
CLEAN_SELECTOR_GAMMA = 0.0


# ======================================================================
# 保存旧配置，用于reference
# ======================================================================
ORIGINAL_BOTTOM_LAMBDA = float(
    BASE.BOTTOM_LAMBDA
)

ORIGINAL_SELECTOR_GAMMA = float(
    BASE.PHYSICS_SCORE_GAMMA
)


# ======================================================================
# OUTPUT
# ======================================================================
RESULT_CSV = (
    OUTPUT_DIR
    / "physics_weight_ablation_results.csv"
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "physics_weight_ablation_summary.csv"
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
    ) as file:

        writer = csv.DictWriter(
            file,
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
# RMSE
# ======================================================================
def calculate_rmse(
    estimate,
    target,
):

    estimate = np.asarray(
        estimate,
        dtype=np.float64,
    )

    target = np.asarray(
        target,
        dtype=np.float64,
    )

    if estimate.size != target.size:

        return float(
            "inf"
        )

    return float(
        np.sqrt(
            np.mean(
                (
                    estimate
                    -
                    target
                ) ** 2
            )
        )
    )


def calculate_max_error(
    estimate,
    target,
):

    estimate = np.asarray(
        estimate,
        dtype=np.float64,
    )

    target = np.asarray(
        target,
        dtype=np.float64,
    )

    if estimate.size != target.size:

        return float(
            "inf"
        )

    return float(
        np.max(
            np.abs(
                estimate
                -
                target
            )
        )
    )


# ======================================================================
# Root distance diagnostic
#
# 如果：
#
#   RMSE(raw, HO roots) << RMSE(raw, KRAKEN truth)
#
# 说明结果明显更贴近inverse physics roots。
# ======================================================================
def calculate_root_lock_ratio(
    truth_rmse,
    root_rmse,
):

    if not np.isfinite(
        truth_rmse
    ):

        return float(
            "inf"
        )

    if not np.isfinite(
        root_rmse
    ):

        return float(
            "inf"
        )

    return float(
        truth_rmse
        /
        max(
            root_rmse,
            1.0e-15,
        )
    )


# ======================================================================
# Selected bottom residual
# ======================================================================
def calculate_mean_bottom_residual(
    wavenumbers,
):

    k_values = np.asarray(
        wavenumbers,
        dtype=np.float64,
    )

    if k_values.size == 0:

        return float(
            "inf"
        )

    residuals = np.asarray(
        [
            BASE.normalized_robin_residual_np(
                float(
                    k
                )
            )
            for k
            in k_values
        ],
        dtype=np.float64,
    )

    return float(
        np.mean(
            residuals
        )
    )


# ======================================================================
# CONFIG LIST
# ======================================================================
def build_experiment_configs():

    configs = []

    for bottom_lambda in BOTTOM_LAMBDAS:

        if bottom_lambda == 0.0:

            name = (
                "NO_BOTTOM_PENALTY"
            )

        else:

            name = (
                f"LAMBDA_{bottom_lambda:.0e}"
            )

        configs.append(
            {
                "name": (
                    name
                ),

                "bottom_lambda": float(
                    bottom_lambda
                ),

                "selector_gamma": float(
                    CLEAN_SELECTOR_GAMMA
                ),

                "type": (
                    "CLEAN_ABLATION"
                ),
            }
        )

    # ------------------------------------------------------------------
    # 当前V2 pipeline参考
    # ------------------------------------------------------------------
    configs.append(
        {
            "name": (
                "CURRENT_PIPELINE_REF"
            ),

            "bottom_lambda": (
                ORIGINAL_BOTTOM_LAMBDA
            ),

            "selector_gamma": (
                ORIGINAL_SELECTOR_GAMMA
            ),

            "type": (
                "REFERENCE"
            ),
        }
    )

    return configs


# ======================================================================
# SUMMARY
# ======================================================================
def summarize(
    rows,
):

    output = []

    config_names = []

    for row in rows:

        name = row[
            "config"
        ]

        if name not in config_names:

            config_names.append(
                name
            )

    for config_name in config_names:

        for snr_db in SNR_VALUES_DB:

            subset = [
                row
                for row
                in rows
                if (
                    row[
                        "config"
                    ]
                    ==
                    config_name
                    and
                    float(
                        row[
                            "snr_db"
                        ]
                    )
                    ==
                    float(
                        snr_db
                    )
                )
            ]

            if not subset:

                continue

            output.append(
                {
                    "config": (
                        config_name
                    ),

                    "snr_db": (
                        snr_db
                    ),

                    "trials": (
                        len(
                            subset
                        )
                    ),

                    "success_rate": float(
                        np.mean(
                            [
                                row[
                                    "raw_success"
                                ]
                                for row
                                in subset
                            ]
                        )
                    ),

                    "truth_rmse_median": float(
                        np.median(
                            [
                                row[
                                    "truth_rmse"
                                ]
                                for row
                                in subset
                            ]
                        )
                    ),

                    "truth_max_error_median": float(
                        np.median(
                            [
                                row[
                                    "truth_max_error"
                                ]
                                for row
                                in subset
                            ]
                        )
                    ),

                    "root_rmse_median": float(
                        np.median(
                            [
                                row[
                                    "root_rmse"
                                ]
                                for row
                                in subset
                            ]
                        )
                    ),

                    "root_lock_ratio_median": float(
                        np.median(
                            [
                                row[
                                    "root_lock_ratio"
                                ]
                                for row
                                in subset
                            ]
                        )
                    ),

                    "data_fit_median": float(
                        np.median(
                            [
                                row[
                                    "selector_fit"
                                ]
                                for row
                                in subset
                            ]
                        )
                    ),

                    "bottom_residual_mean": float(
                        np.mean(
                            [
                                row[
                                    "selected_bottom_residual"
                                ]
                                for row
                                in subset
                            ]
                        )
                    ),

                    "runtime_seconds_mean": float(
                        np.mean(
                            [
                                row[
                                    "runtime_seconds"
                                ]
                                for row
                                in subset
                            ]
                        )
                    ),
                }
            )

    return output


# ======================================================================
# MAIN
# ======================================================================
def main():

    torch.set_default_dtype(
        torch.float64
    )

    print(
        "=" * 190
    )

    print(
        "PHYSICS WEIGHT ABLATION"
    )

    print(
        "HIGH-ORDER NUMEROV MODE-FRINET"
    )

    print(
        "=" * 190
    )

    print(
        f"K                  = {TARGET_MODE_COUNT}"
    )

    print(
        f"SNR                = {SNR_VALUES_DB}"
    )

    print(
        f"Numerov dz         = {BASE.PHYSICS_DZ} m"
    )

    print(
        f"M                  = {BASE.NUMBER_OF_ATOMS}"
    )

    print(
        f"L                  = {BASE.NUMBER_OF_SNAPSHOTS}"
    )

    print(
        f"Clean selector gamma = {CLEAN_SELECTOR_GAMMA}"
    )

    print(
        f"Current gamma ref    = {ORIGINAL_SELECTOR_GAMMA}"
    )

    print(
        "=" * 190
    )


    # ==================================================================
    # Autograd
    # ==================================================================
    BASE.verify_high_order_autograd()


    # ==================================================================
    # Physical roots
    # ==================================================================
    physical_roots = (
        BASE.calculate_physical_roots()
    )

    target_roots = np.asarray(
        physical_roots[
            :TARGET_MODE_COUNT
        ],
        dtype=np.float64,
    )


    # ==================================================================
    # KRAKEN
    # ==================================================================
    kraken = (
        BASE.load_kraken()
    )

    complex_k = np.asarray(
        kraken[
            "complex_k"
        ],
        dtype=np.complex128,
    )

    true_k = np.asarray(
        np.real(
            complex_k[
                :TARGET_MODE_COUNT
            ]
        ),
        dtype=np.float64,
    )

    forward_matrix = np.asarray(
        kraken[
            "mode_matrix"
        ][
            :,
            :TARGET_MODE_COUNT
        ],
        dtype=np.complex128,
    )


    print(
        "\nKRAKEN truth:"
    )

    print(
        true_k
    )

    print(
        "\nHO physical roots:"
    )

    print(
        target_roots
    )

    print(
        "\nTruth-root absolute difference:"
    )

    print(
        np.abs(
            true_k
            -
            target_roots
        )
    )

    print(
        "\nTruth-root RMSE = "
        f"{calculate_rmse(true_k, target_roots):.6e}"
    )


    # ==================================================================
    # Same master data for every lambda
    # ==================================================================
    masters = {}

    for trial in range(
        1,
        NUMBER_OF_TRIALS + 1,
    ):

        seed_base = (

            BASE_RANDOM_SEED

            +

            TARGET_MODE_COUNT
            *
            1_000_000

            +

            trial
            *
            10_000
        )

        masters[
            trial
        ] = BASE.make_master_data(

            forward_matrix=(
                forward_matrix
            ),

            coefficient_seed=(
                seed_base + 1
            ),

            noise_seed=(
                seed_base + 2
            ),
        )


    configs = (
        build_experiment_configs()
    )

    rows = []


    # ==================================================================
    # Run
    # ==================================================================
    try:

        for snr_db in SNR_VALUES_DB:

            print(
                "\n\n"
                + "=" * 190
            )

            print(
                f"SNR = {snr_db:+.1f} dB"
            )

            print(
                "=" * 190
            )


            for trial in range(
                1,
                NUMBER_OF_TRIALS + 1,
            ):

                noisy_data = (
                    BASE.add_noise(
                        masters[
                            trial
                        ],
                        snr_db,
                    )
                )


                # ======================================================
                # IMPORTANT:
                #
                # Same model seed for every lambda.
                #
                # Therefore:
                #   same atom initial positions
                #   same coefficient initialization
                # ======================================================
                model_seed = (

                    BASE_RANDOM_SEED

                    +

                    TARGET_MODE_COUNT
                    *
                    1_000_000

                    +

                    trial
                    *
                    10_000

                    +

                    int(
                        (
                            snr_db
                            +
                            20.0
                        )
                        *
                        1000
                    )
                )


                for config in configs:

                    bottom_lambda = float(
                        config[
                            "bottom_lambda"
                        ]
                    )

                    selector_gamma = float(
                        config[
                            "selector_gamma"
                        ]
                    )


                    # ==================================================
                    # Only these two values change
                    # ==================================================
                    BASE.BOTTOM_LAMBDA = (
                        bottom_lambda
                    )

                    BASE.PHYSICS_SCORE_GAMMA = (
                        selector_gamma
                    )


                    print(
                        "\n"
                        + "-" * 175
                    )

                    print(
                        f"{config['name']} | "
                        f"lambda_b={bottom_lambda:.3e} | "
                        f"selector_gamma={selector_gamma:.3e}"
                    )

                    print(
                        "-" * 175
                    )


                    trained = (
                        V2.train_one_restart_stratified(

                            noisy_data=(
                                noisy_data
                            ),

                            physical_roots=(
                                physical_roots
                            ),

                            number_of_modes=(
                                TARGET_MODE_COUNT
                            ),

                            random_seed=(
                                model_seed
                            ),
                        )
                    )


                    selection = (
                        BASE.select_mode_order_beam(

                            trained=(
                                trained
                            ),

                            noisy_data=(
                                noisy_data
                            ),

                            physical_roots=(
                                physical_roots
                            ),

                            number_of_modes=(
                                TARGET_MODE_COUNT
                            ),
                        )
                    )


                    if selection[
                        "success"
                    ]:

                        raw_k = np.sort(
                            np.asarray(
                                selection[
                                    "selected"
                                ],
                                dtype=np.float64,
                            )
                        )[::-1]


                        evaluation = (
                            BASE.evaluate_k(
                                raw_k,
                                true_k,
                            )
                        )


                        truth_rmse = (
                            calculate_rmse(
                                raw_k,
                                true_k,
                            )
                        )


                        truth_max_error = (
                            calculate_max_error(
                                raw_k,
                                true_k,
                            )
                        )


                        root_rmse = (
                            calculate_rmse(
                                raw_k,
                                target_roots,
                            )
                        )


                        root_max_error = (
                            calculate_max_error(
                                raw_k,
                                target_roots,
                            )
                        )


                        root_lock_ratio = (
                            calculate_root_lock_ratio(
                                truth_rmse=(
                                    truth_rmse
                                ),

                                root_rmse=(
                                    root_rmse
                                ),
                            )
                        )


                        selected_bottom_residual = (
                            calculate_mean_bottom_residual(
                                raw_k
                            )
                        )


                    else:

                        raw_k = np.asarray(
                            [],
                            dtype=np.float64,
                        )

                        evaluation = {
                            "success": False,
                        }

                        truth_rmse = float(
                            "inf"
                        )

                        truth_max_error = float(
                            "inf"
                        )

                        root_rmse = float(
                            "inf"
                        )

                        root_max_error = float(
                            "inf"
                        )

                        root_lock_ratio = float(
                            "inf"
                        )

                        selected_bottom_residual = float(
                            "inf"
                        )


                    print(
                        f"selector success = "
                        f"{selection['success']}"
                    )

                    print(
                        f"selector reason  = "
                        f"{selection['reason']}"
                    )

                    print(
                        f"raw success      = "
                        f"{evaluation['success']}"
                    )

                    print(
                        f"truth RMSE       = "
                        f"{truth_rmse:.6e}"
                    )

                    print(
                        f"truth max error  = "
                        f"{truth_max_error:.6e}"
                    )

                    print(
                        f"root RMSE        = "
                        f"{root_rmse:.6e}"
                    )

                    print(
                        f"root lock ratio  = "
                        f"{root_lock_ratio:.3f}"
                    )

                    print(
                        f"data fit         = "
                        f"{selection['fit']:.6e}"
                    )

                    print(
                        f"bottom residual  = "
                        f"{selected_bottom_residual:.6e}"
                    )

                    print(
                        f"runtime          = "
                        f"{trained['runtime_seconds']:.2f} s"
                    )

                    print(
                        f"k                = "
                        f"{raw_k}"
                    )


                    rows.append(
                        {
                            "config": (
                                config[
                                    "name"
                                ]
                            ),

                            "config_type": (
                                config[
                                    "type"
                                ]
                            ),

                            "K": (
                                TARGET_MODE_COUNT
                            ),

                            "snr_db": (
                                snr_db
                            ),

                            "trial": (
                                trial
                            ),

                            "model_seed": (
                                model_seed
                            ),

                            "bottom_lambda": (
                                bottom_lambda
                            ),

                            "selector_gamma": (
                                selector_gamma
                            ),

                            "selector_success": int(
                                selection[
                                    "success"
                                ]
                            ),

                            "selector_reason": (
                                selection[
                                    "reason"
                                ]
                            ),

                            "raw_success": int(
                                evaluation[
                                    "success"
                                ]
                            ),

                            "truth_rmse": (
                                truth_rmse
                            ),

                            "truth_max_error": (
                                truth_max_error
                            ),

                            "root_rmse": (
                                root_rmse
                            ),

                            "root_max_error": (
                                root_max_error
                            ),

                            "root_lock_ratio": (
                                root_lock_ratio
                            ),

                            "selector_fit": (
                                selection[
                                    "fit"
                                ]
                            ),

                            "selected_bottom_residual": (
                                selected_bottom_residual
                            ),

                            "runtime_seconds": (
                                trained[
                                    "runtime_seconds"
                                ]
                            ),

                            "initial_atoms_per_cell": (
                                ";".join(
                                    str(
                                        int(
                                            value
                                        )
                                    )
                                    for value
                                    in trained[
                                        "initial_atoms_per_cell"
                                    ]
                                )
                            ),

                            "final_atoms_per_cell": (
                                ";".join(
                                    str(
                                        int(
                                            value
                                        )
                                    )
                                    for value
                                    in trained[
                                        "final_atoms_per_cell"
                                    ]
                                )
                            ),

                            "raw_k": (
                                array_text(
                                    raw_k
                                )
                            ),

                            "true_k": (
                                array_text(
                                    true_k
                                )
                            ),

                            "physical_roots": (
                                array_text(
                                    target_roots
                                )
                            ),
                        }
                    )


                    # checkpoint
                    save_csv(
                        RESULT_CSV,
                        rows,
                    )

                    save_csv(
                        SUMMARY_CSV,
                        summarize(
                            rows
                        ),
                    )


    finally:

        # ==============================================================
        # Restore original settings
        # ==============================================================
        BASE.BOTTOM_LAMBDA = (
            ORIGINAL_BOTTOM_LAMBDA
        )

        BASE.PHYSICS_SCORE_GAMMA = (
            ORIGINAL_SELECTOR_GAMMA
        )


    # ==================================================================
    # SUMMARY
    # ==================================================================
    summary_rows = summarize(
        rows
    )

    save_csv(
        SUMMARY_CSV,
        summary_rows,
    )


    print(
        "\n\n"
        + "=" * 195
    )

    print(
        "PHYSICS WEIGHT ABLATION SUMMARY"
    )

    print(
        "=" * 195
    )


    for row in summary_rows:

        print(

            f"{row['config']:<24} | "
            f"SNR={row['snr_db']:>5.1f} | "
            f"succ={100*row['success_rate']:6.1f}% | "
            f"truthRMSE={row['truth_rmse_median']:.3e} | "
            f"rootRMSE={row['root_rmse_median']:.3e} | "
            f"lock={row['root_lock_ratio_median']:.2f} | "
            f"fit={row['data_fit_median']:.3e} | "
            f"time={row['runtime_seconds_mean']:.1f}s"
        )


    # ==================================================================
    # Primary interpretation
    # ==================================================================
    print(
        "\n"
        + "=" * 195
    )

    print(
        "PRIMARY INTERPRETATION"
    )

    print(
        "=" * 195
    )


    for snr_db in SNR_VALUES_DB:

        snr_rows = [
            row
            for row
            in summary_rows
            if float(
                row[
                    "snr_db"
                ]
            )
            ==
            float(
                snr_db
            )
        ]


        zero_row = next(
            (
                row
                for row
                in snr_rows
                if row[
                    "config"
                ]
                ==
                "NO_BOTTOM_PENALTY"
            ),
            None,
        )


        clean_rows = [
            row
            for row
            in snr_rows
            if row[
                "config"
            ]
            !=
            "CURRENT_PIPELINE_REF"
        ]


        successful_clean = [
            row
            for row
            in clean_rows
            if row[
                "success_rate"
            ]
            >
            0.0
        ]


        best_clean = None

        if successful_clean:

            best_clean = min(

                successful_clean,

                key=lambda row:
                row[
                    "truth_rmse_median"
                ],
            )


        print(
            f"\nSNR={snr_db:+.1f} dB"
        )


        if zero_row is not None:

            print(
                "  lambda_b=0:"
            )

            print(
                f"    success = "
                f"{100*zero_row['success_rate']:.1f}%"
            )

            print(
                f"    RMSE    = "
                f"{zero_row['truth_rmse_median']:.6e}"
            )

            print(
                f"    rootLock= "
                f"{zero_row['root_lock_ratio_median']:.3f}"
            )


        if best_clean is not None:

            print(
                f"  best clean ablation = "
                f"{best_clean['config']}"
            )

            print(
                f"    RMSE = "
                f"{best_clean['truth_rmse_median']:.6e}"
            )


    print(
        "\n判定原则："
    )

    print(
        "A. 如果lambda_b=0仍然K=8成功，且RMSE明显优于旧FD2高阶误差，"
    )

    print(
        "   则Numerov连续原子本身已经产生真实的数据反演收益。"
    )

    print(
        "B. 如果1e-4~1e-2已经很好，说明弱物理正则足够，"
    )

    print(
        "   最终算法不需要强Robin root locking。"
    )

    print(
        "C. 如果只有1e-1~1才能成功，而且root RMSE远小于truth RMSE，"
    )

    print(
        "   则当前高精度结果高度依赖已知环境本征根。"
    )

    print(
        "D. 本实验只有1 trial，是机制诊断，不作为最终统计结果。"
    )


    print(
        "\nOutputs:"
    )

    print(
        RESULT_CSV
    )

    print(
        SUMMARY_CSV
    )


if __name__ == "__main__":
    main()
