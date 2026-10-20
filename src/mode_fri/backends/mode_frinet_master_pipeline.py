# -*- coding: utf-8 -*-
"""
=======================================================================
MODE-FRINET MASTER PIPELINE
=======================================================================

统一主程序，集中完成：

A. HO-PG Mode-FRINET (validated numerical backend)
   - K = 3 / 5 / 8
   - Numerov4 high-order differentiable atoms
   - physics-stratified continuous atoms
   - multi-snapshot sparse estimation
   - realistic-bottom Robin regularization
   - lambda_bottom = 0.1 for K=3/5/8
   - one deterministic restart for K=3/5/8
   - selector physics gamma = 0
   - no safe-root refinement

B. Environment Robustness Diagnosis
   - fixed KRAKEN forward data
   - deliberately mismatched inverse SSP
   - environment identifiability profile
   - automatic calibration accept / reject
   - safe fallback to original HO-PG result

=======================================================================
核心原则
=======================================================================

最终模态波数 k 始终是连续变量。

环境校准扫描的只是低维 nuisance parameter：

    delta_c

不是对 k 做网格搜索。

如果环境 profile 太平：
    -> ENVIRONMENT_UNIDENTIFIABLE
    -> 不修改环境
    -> 自动回退到原 HO-PG Mode-FRINET

如果 profile 最优点撞搜索边界：
    -> BOUNDARY_REJECT
    -> 不允许自动校准

当前最终结果不接受单频环境自动校准。profile 只保留为诊断证据，
估计始终使用安全 fallback；历史 calibration 分支不属于最终入口。

=======================================================================
RUN MODE
=======================================================================

QUICK:
    Nominal:
        K = 3,5,8
        SNR = 15 dB
        trial = 1

    Environment:
        K = 8
        SNR = 15 dB
        trial = 1
        inverse SSP offset:
            0,-1,-3,+1,+3 m/s

最终 QUICK/FULL 范围由 configs/final_paper_config.json 管理；本后端中的
旧 RUN_MODE 仅用于历史直接调用兼容。正式入口为 run_final_paper.py。

=======================================================================
输出
=======================================================================

历史后端直出目录：outputs/mode_frinet_master_pipeline/

    master_nominal_results.csv
    master_nominal_summary.csv

    master_environment_results.csv
    master_environment_profile.csv
    master_environment_modewise.csv
    master_environment_decisions.csv
    master_environment_summary.csv

正式输出目录：outputs/final_paper/
"""

from __future__ import annotations

import csv
import importlib.util
import sys
import time
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
    / "mode_frinet_master_pipeline"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ======================================================================
# MODULE LOAD
# ======================================================================

def load_module(
    path: Path,
    module_name: str,
):

    if not path.exists():

        raise FileNotFoundError(
            "\n缺少依赖文件：\n"
            f"{path}\n\n"
            "请确认以下文件存在：\n"
            "experiments/high_order_physics_mode_frinet_final.py\n"
            "experiments/high_order_physics_mode_frinet_final_v2.py\n"
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
            f"无法加载模块：{path}"
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


V2 = load_module(
    V2_SCRIPT,
    "mode_frinet_master_v2",
)

BASE = V2.BASE


# ======================================================================
# RUN MODE
# ======================================================================

RUN_MODE = "QUICK"

# 正式实验再改：
#
# RUN_MODE = "FULL"


if RUN_MODE.upper() == "QUICK":

    NOMINAL_K_VALUES = [
        3,
        5,
        8,
    ]

    NOMINAL_SNR_VALUES = [
        15.0,
    ]

    NOMINAL_TRIALS = 1

    ENVIRONMENT_SNR_VALUES = [
        15.0,
    ]

    ENVIRONMENT_TRIALS = 1

elif RUN_MODE.upper() == "K3_MULTIRESTART_VALIDATE":

    NOMINAL_K_VALUES = [3]

    NOMINAL_SNR_VALUES = [15.0]

    NOMINAL_TRIALS = 5

    ENVIRONMENT_SNR_VALUES = []

    ENVIRONMENT_TRIALS = 0

else:

    NOMINAL_K_VALUES = [
        3,
        5,
        8,
    ]

    NOMINAL_SNR_VALUES = [
        10.0,
        15.0,
        30.0,
    ]

    NOMINAL_TRIALS = 5

    ENVIRONMENT_SNR_VALUES = [
        15.0,
    ]

    ENVIRONMENT_TRIALS = 5


# ======================================================================
# FINAL ALGORITHM CONFIG
# ======================================================================

FINAL_BOTTOM_LAMBDA = 1.0e-1

BOTTOM_LAMBDA_BY_K = {
    3: 1.0e-1,
    5: 1.0e-1,
    8: 1.0e-1,
}

BOTTOM_LAMBDA_CANDIDATES_BY_K = {
    3: [1.0e-1],
    5: [1.0e-1],
    8: [1.0e-1],
}

RESTARTS_BY_K = {
    3: 1,
    5: 1,
    8: 1,
}

# K=3专项冻结：只在该运行模式下使用，不影响已验证的K=8环境模块。
K3_FREEZE_LAMBDAS = [
    0.0,
    1.0e-3,
    1.0e-2,
    1.0e-1,
    1.0,
]

K3_FREEZE_SNR_DB = 15.0

K3_FREEZE_TRIALS = 5

K3_RESTART_DIAGNOSTIC_LAMBDA = 1.0

K3_RESTART_DIAGNOSTIC_TRIAL = 4

K3_RESTART_DIAGNOSTIC_RESTARTS = 5

FINAL_SELECTOR_GAMMA = 0.0

ENVIRONMENT_TARGET_K = 8

BASE_RANDOM_SEED = (
    BASE.BASE_RANDOM_SEED
)


# ======================================================================
# ENVIRONMENT TEST
# ======================================================================

INVERSE_SSP_OFFSETS_MPS = [
    0.0,
    -1.0,
    -3.0,
    1.0,
    3.0,
]


# ======================================================================
# ENVIRONMENT PROFILE SEARCH
# ======================================================================

DELTA_C_MIN = -5.0

DELTA_C_MAX = 5.0

COARSE_STEP = 0.5

FINE_HALF_WIDTH = 0.75

FINE_STEP = 0.05


# ======================================================================
# AUTOMATIC IDENTIFIABILITY PROTECTION
# ======================================================================
#
# 当前实验证明：
#
# profile gain 只有约 1e-5 ~ 2e-4 时完全不可信。
#
# 因此要求：
#
# relative profile depth >= 0.1%
#
# 即：
#
# (J0 - Jbest) / J0 >= 1e-3
#
# 才允许环境校准。
# ======================================================================

MIN_RELATIVE_PROFILE_DEPTH = 1.0e-3

MIN_RELATIVE_PROFILE_SPAN = 1.0e-3

BOUNDARY_MARGIN_MPS = 0.25

ZERO_CORRECTION_TOLERANCE_MPS = 0.25


# ======================================================================
# ORIGINAL BASE CONFIG
# ======================================================================

ORIGINAL_SOUND_SPEED_FUNCTION = (
    BASE.sound_speed_np
)

ORIGINAL_K_MAX = float(
    BASE.K_MAX
)

ORIGINAL_BOTTOM_LAMBDA = float(
    BASE.BOTTOM_LAMBDA
)

ORIGINAL_SELECTOR_GAMMA = float(
    BASE.PHYSICS_SCORE_GAMMA
)


# ======================================================================
# OUTPUT FILES
# ======================================================================

NOMINAL_RESULT_CSV = (
    OUTPUT_DIR
    / "master_nominal_results.csv"
)

NOMINAL_SUMMARY_CSV = (
    OUTPUT_DIR
    / "master_nominal_summary.csv"
)

ENV_RESULT_CSV = (
    OUTPUT_DIR
    / "master_environment_results.csv"
)

ENV_PROFILE_CSV = (
    OUTPUT_DIR
    / "master_environment_profile.csv"
)

ENV_MODEWISE_CSV = (
    OUTPUT_DIR
    / "master_environment_modewise.csv"
)

ENV_DECISION_CSV = (
    OUTPUT_DIR
    / "master_environment_decisions.csv"
)

ENV_SUMMARY_CSV = (
    OUTPUT_DIR
    / "master_environment_summary.csv"
)

K3_FREEZE_RESULT_CSV = (
    OUTPUT_DIR
    / "k3_bottom_lambda_freeze_results.csv"
)

K3_FREEZE_SUMMARY_CSV = (
    OUTPUT_DIR
    / "k3_bottom_lambda_freeze_summary.csv"
)

K3_RESTART_RESULT_CSV = (
    OUTPUT_DIR
    / "k3_restart_diagnostic_results.csv"
)


# ======================================================================
# BASIC HELPERS
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
                )
                ** 2
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
# RESTORE NOMINAL ENVIRONMENT
# ======================================================================

def restore_nominal_environment():

    BASE.sound_speed_np = (
        ORIGINAL_SOUND_SPEED_FUNCTION
    )

    BASE.K_MAX = (
        ORIGINAL_K_MAX
    )

    BASE.BOTTOM_LAMBDA = (
        FINAL_BOTTOM_LAMBDA
    )

    BASE.PHYSICS_SCORE_GAMMA = (
        FINAL_SELECTOR_GAMMA
    )


# ======================================================================
# NOMINAL SSP + OFFSET
# ======================================================================

def set_inverse_ssp_offset(
    offset_mps: float,
):

    offset_mps = float(
        offset_mps
    )


    def current_sound_speed(
        depths,
    ):

        nominal = np.asarray(
            ORIGINAL_SOUND_SPEED_FUNCTION(
                depths
            ),
            dtype=np.float64,
        )

        return (
            nominal
            +
            offset_mps
        )


    BASE.sound_speed_np = (
        current_sound_speed
    )


    reference_depths = np.linspace(
        0.0,
        BASE.WATER_DEPTH_M,
        401,
        dtype=np.float64,
    )


    c_profile = BASE.sound_speed_np(
        reference_depths
    )


    minimum_c = float(
        np.min(
            c_profile
        )
    )


    BASE.K_MAX = (

        BASE.OMEGA

        /

        minimum_c

        *

        0.999999
    )


    BASE.BOTTOM_LAMBDA = (
        FINAL_BOTTOM_LAMBDA
    )

    BASE.PHYSICS_SCORE_GAMMA = (
        FINAL_SELECTOR_GAMMA
    )


    return {

        "offset_mps": (
            offset_mps
        ),

        "minimum_c": (
            minimum_c
        ),

        "k_max": float(
            BASE.K_MAX
        ),
    }


# ======================================================================
# TRAIN ONE HO-PG MODE-FRINET
# ======================================================================

def run_mode_frinet(
    noisy_data,
    physical_roots,
    number_of_modes,
    random_seed,
    bottom_lambda=None,
):

    BASE.BOTTOM_LAMBDA = (
        BOTTOM_LAMBDA_BY_K.get(
            int(number_of_modes),
            FINAL_BOTTOM_LAMBDA,
        )
        if bottom_lambda is None
        else float(bottom_lambda)
    )

    BASE.PHYSICS_SCORE_GAMMA = (
        FINAL_SELECTOR_GAMMA
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
                number_of_modes
            ),

            random_seed=(
                random_seed
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
                number_of_modes
            ),
        )
    )


    if not selection[
        "success"
    ]:

        return {

            "selection_success": False,

            "k": np.asarray(
                [],
                dtype=np.float64,
            ),

            "fit": float(
                "inf"
            ),

            "runtime_seconds": float(
                trained[
                    "runtime_seconds"
                ]
            ),
        }


    estimated_k = np.sort(
        np.asarray(
            selection[
                "selected"
            ],
            dtype=np.float64,
        )
    )[::-1]


    return {

        "selection_success": True,

        "k": (
            estimated_k
        ),

        "fit": float(
            selection[
                "fit"
            ]
        ),

        "runtime_seconds": float(
            trained[
                "runtime_seconds"
            ]
        ),
    }


def run_mode_frinet_robust(
    noisy_data,
    physical_roots,
    number_of_modes,
    random_seed,
):

    restart_count = int(
        RESTARTS_BY_K.get(
            int(number_of_modes),
            1,
        )
    )

    lambda_candidates = BOTTOM_LAMBDA_CANDIDATES_BY_K.get(
        int(number_of_modes),
        [FINAL_BOTTOM_LAMBDA],
    )

    candidates = []

    for bottom_lambda in lambda_candidates:

        for restart in range(restart_count):

            restart_seed = (
                int(random_seed)
                + restart * 1_000_003
            )

            candidate = run_mode_frinet(
                noisy_data=noisy_data,
                physical_roots=physical_roots,
                number_of_modes=number_of_modes,
                random_seed=restart_seed,
                bottom_lambda=bottom_lambda,
            )

            candidate["restart"] = restart
            candidate["random_seed"] = restart_seed
            candidate["bottom_lambda"] = float(bottom_lambda)
            candidates.append(candidate)

    successful = [
        item for item in candidates
        if item["selection_success"]
    ]

    if successful:
        best = min(
            successful,
            key=lambda item: item["fit"],
        ).copy()
    else:
        best = min(
            candidates,
            key=lambda item: item["fit"],
        ).copy()

    ordered_fits = sorted(
        float(item["fit"])
        for item in successful
        if np.isfinite(item["fit"])
    )

    best["restart_count"] = restart_count
    best["candidate_count"] = len(candidates)
    best["selected_restart"] = int(best["restart"])
    best["selected_bottom_lambda"] = float(best["bottom_lambda"])
    best["all_restart_fits"] = array_text(ordered_fits)
    best["restart_fit_gap"] = (
        float(ordered_fits[1] - ordered_fits[0])
        if len(ordered_fits) >= 2
        else float("nan")
    )
    best["runtime_seconds"] = float(
        sum(item["runtime_seconds"] for item in candidates)
    )

    return best


# ======================================================================
# BUILD DATA
# ======================================================================

def create_noisy_dataset(
    forward_matrix,
    number_of_modes,
    snr_db,
    trial,
):

    seed_base = (

        BASE_RANDOM_SEED

        +

        number_of_modes
        *
        1_000_000

        +

        trial
        *
        10_000
    )


    master = (
        BASE.make_master_data(

            forward_matrix=(
                forward_matrix
            ),

            coefficient_seed=(
                seed_base
                +
                1
            ),

            noise_seed=(
                seed_base
                +
                2
            ),
        )
    )


    noisy_data = (
        BASE.add_noise(
            master,
            snr_db,
        )
    )


    model_seed = (

        BASE_RANDOM_SEED

        +

        number_of_modes
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


    return (
        noisy_data,
        model_seed,
    )


# ======================================================================
# K=3 BOTTOM-LAMBDA FREEZE
# ======================================================================

def run_k3_lambda_freeze(
    kraken,
):

    number_of_modes = 3

    true_k = np.asarray(
        np.real(
            np.asarray(
                kraken["complex_k"],
                dtype=np.complex128,
            )[:number_of_modes]
        ),
        dtype=np.float64,
    )

    forward_matrix = np.asarray(
        kraken["mode_matrix"][:, :number_of_modes],
        dtype=np.complex128,
    )

    result_rows = []

    for bottom_lambda in K3_FREEZE_LAMBDAS:

        for trial in range(1, K3_FREEZE_TRIALS + 1):

            restore_nominal_environment()

            physical_roots = BASE.calculate_physical_roots()

            noisy_data, model_seed = create_noisy_dataset(
                forward_matrix=forward_matrix,
                number_of_modes=number_of_modes,
                snr_db=K3_FREEZE_SNR_DB,
                trial=trial,
            )

            result = run_mode_frinet(
                noisy_data=noisy_data,
                physical_roots=physical_roots,
                number_of_modes=number_of_modes,
                random_seed=model_seed,
                bottom_lambda=bottom_lambda,
            )

            estimated_k = result["k"]

            if estimated_k.size == number_of_modes:
                errors = np.abs(estimated_k - true_k)
                truth_rmse = calculate_rmse(estimated_k, true_k)
                max_error = calculate_max_error(estimated_k, true_k)
                success = int(max_error <= BASE.SUCCESS_TOLERANCE)
            else:
                errors = np.full(number_of_modes, np.inf)
                truth_rmse = float("inf")
                max_error = float("inf")
                success = 0

            row = {
                "bottom_lambda": float(bottom_lambda),
                "trial": trial,
                "snr_db": K3_FREEZE_SNR_DB,
                "success": success,
                "truth_rmse": truth_rmse,
                "max_error": max_error,
                "mode1_abs_error": float(errors[0]),
                "mode2_abs_error": float(errors[1]),
                "mode3_abs_error": float(errors[2]),
                "selector_fit": float(result["fit"]),
                "runtime_seconds": float(result["runtime_seconds"]),
                "estimated_k": array_text(estimated_k),
            }

            result_rows.append(row)
            save_csv(K3_FREEZE_RESULT_CSV, result_rows)

            print(
                f"K3 FREEZE | lambda={bottom_lambda:.1e} | "
                f"trial={trial} | RMSE={truth_rmse:.6e} | "
                f"mode1={errors[0]:.6e} | success={success}"
            )

    summary_rows = []

    for bottom_lambda in K3_FREEZE_LAMBDAS:

        group = [
            row for row in result_rows
            if row["bottom_lambda"] == float(bottom_lambda)
        ]

        rmse = np.asarray([row["truth_rmse"] for row in group])
        mode1 = np.asarray([row["mode1_abs_error"] for row in group])

        summary_rows.append(
            {
                "bottom_lambda": float(bottom_lambda),
                "trials": len(group),
                "success_rate": float(np.mean([row["success"] for row in group])),
                "rmse_mean": float(np.mean(rmse)),
                "rmse_median": float(np.median(rmse)),
                "rmse_p90": float(np.quantile(rmse, 0.90)),
                "mode1_error_mean": float(np.mean(mode1)),
                "mode1_error_median": float(np.median(mode1)),
                "mode1_error_p90": float(np.quantile(mode1, 0.90)),
                "runtime_mean": float(np.mean([row["runtime_seconds"] for row in group])),
            }
        )

    save_csv(K3_FREEZE_SUMMARY_CSV, summary_rows)

    eligible = [row for row in summary_rows if row["success_rate"] >= 0.8]

    if not eligible:
        eligible = summary_rows

    best = min(
        eligible,
        key=lambda row: (
            row["rmse_median"],
            row["mode1_error_p90"],
            row["rmse_p90"],
        ),
    )

    print("\nK3 LAMBDA FREEZE COMPLETE")
    print(f"recommended lambda = {best['bottom_lambda']:.1e}")
    print(f"median RMSE        = {best['rmse_median']:.6e}")
    print(f"mode-1 p90 error   = {best['mode1_error_p90']:.6e}")

    return result_rows, summary_rows, best


def run_k3_restart_diagnostic(
    kraken,
):

    number_of_modes = 3
    trial = K3_RESTART_DIAGNOSTIC_TRIAL

    true_k = np.asarray(
        np.real(np.asarray(kraken["complex_k"], dtype=np.complex128)[:number_of_modes]),
        dtype=np.float64,
    )

    forward_matrix = np.asarray(
        kraken["mode_matrix"][:, :number_of_modes],
        dtype=np.complex128,
    )

    restore_nominal_environment()
    physical_roots = BASE.calculate_physical_roots()

    noisy_data, model_seed = create_noisy_dataset(
        forward_matrix=forward_matrix,
        number_of_modes=number_of_modes,
        snr_db=K3_FREEZE_SNR_DB,
        trial=trial,
    )

    rows = []

    for restart in range(K3_RESTART_DIAGNOSTIC_RESTARTS):

        restart_seed = model_seed + restart * 1_000_003

        result = run_mode_frinet(
            noisy_data=noisy_data,
            physical_roots=physical_roots,
            number_of_modes=number_of_modes,
            random_seed=restart_seed,
            bottom_lambda=K3_RESTART_DIAGNOSTIC_LAMBDA,
        )

        estimated_k = result["k"]

        if estimated_k.size == number_of_modes:
            errors = np.abs(estimated_k - true_k)
            rmse = calculate_rmse(estimated_k, true_k)
        else:
            errors = np.full(number_of_modes, np.inf)
            rmse = float("inf")

        rows.append(
            {
                "restart": restart,
                "random_seed": restart_seed,
                "bottom_lambda": K3_RESTART_DIAGNOSTIC_LAMBDA,
                "trial": trial,
                "truth_rmse": rmse,
                "mode1_abs_error": float(errors[0]),
                "mode2_abs_error": float(errors[1]),
                "mode3_abs_error": float(errors[2]),
                "selector_fit": float(result["fit"]),
                "estimated_k": array_text(estimated_k),
                "runtime_seconds": float(result["runtime_seconds"]),
            }
        )

        save_csv(K3_RESTART_RESULT_CSV, rows)

        print(
            f"K3 RESTART | restart={restart} | fit={result['fit']:.9e} | "
            f"RMSE={rmse:.6e} | mode1={errors[0]:.6e}"
        )

    best_by_fit = min(rows, key=lambda row: row["selector_fit"])

    print("\nK3 RESTART DIAGNOSTIC COMPLETE")
    print(f"best-by-fit restart = {best_by_fit['restart']}")
    print(f"best-by-fit RMSE    = {best_by_fit['truth_rmse']:.6e}")

    return rows, best_by_fit


# ======================================================================
# NOMINAL EXPERIMENT
# ======================================================================

def run_nominal_section(
    kraken,
):

    print(
        "\n"
        + "=" * 210
    )

    print(
        "SECTION A — NOMINAL HO-PG MODE-FRINET"
    )

    print(
        "=" * 210
    )


    result_rows = []


    complex_k_all = np.asarray(
        kraken[
            "complex_k"
        ],
        dtype=np.complex128,
    )


    mode_matrix_all = np.asarray(
        kraken[
            "mode_matrix"
        ],
        dtype=np.complex128,
    )


    for number_of_modes in NOMINAL_K_VALUES:

        true_k = np.asarray(
            np.real(
                complex_k_all[
                    :number_of_modes
                ]
            ),
            dtype=np.float64,
        )


        forward_matrix = np.asarray(
            mode_matrix_all[
                :,
                :number_of_modes
            ],
            dtype=np.complex128,
        )


        for snr_db in NOMINAL_SNR_VALUES:

            for trial in range(
                1,
                NOMINAL_TRIALS + 1,
            ):

                print(
                    "\n"
                    + "-" * 185
                )

                print(
                    f"NOMINAL | "
                    f"K={number_of_modes} | "
                    f"SNR={snr_db:+.1f} dB | "
                    f"trial={trial}"
                )

                print(
                    "-" * 185
                )


                restore_nominal_environment()


                physical_roots = (
                    BASE.calculate_physical_roots()
                )


                if (
                    physical_roots.size
                    <
                    number_of_modes
                ):

                    print(
                        "physical roots 数量不足。"
                    )

                    continue


                noisy_data, model_seed = (
                    create_noisy_dataset(

                        forward_matrix=(
                            forward_matrix
                        ),

                        number_of_modes=(
                            number_of_modes
                        ),

                        snr_db=(
                            snr_db
                        ),

                        trial=(
                            trial
                        ),
                    )
                )


                result = (
                    run_mode_frinet_robust(

                        noisy_data=(
                            noisy_data
                        ),

                        physical_roots=(
                            physical_roots
                        ),

                        number_of_modes=(
                            number_of_modes
                        ),

                        random_seed=(
                            model_seed
                        ),
                    )
                )


                estimated_k = result[
                    "k"
                ]


                if (
                    estimated_k.size
                    ==
                    number_of_modes
                ):

                    truth_rmse = (
                        calculate_rmse(
                            estimated_k,
                            true_k,
                        )
                    )


                    max_error = (
                        calculate_max_error(
                            estimated_k,
                            true_k,
                        )
                    )


                    success = int(

                        max_error

                        <=

                        BASE.SUCCESS_TOLERANCE
                    )


                else:

                    truth_rmse = float(
                        "inf"
                    )

                    max_error = float(
                        "inf"
                    )

                    success = 0


                print(
                    f"success = {success}"
                )

                print(
                    f"RMSE    = "
                    f"{truth_rmse:.6e}"
                )

                print(
                    f"max err = "
                    f"{max_error:.6e}"
                )

                print(
                    f"runtime = "
                    f"{result['runtime_seconds']:.2f} s"
                )


                result_rows.append(
                    {
                        "K": (
                            number_of_modes
                        ),

                        "snr_db": (
                            snr_db
                        ),

                        "trial": (
                            trial
                        ),

                        "success": (
                            success
                        ),

                        "truth_rmse": (
                            truth_rmse
                        ),

                        "truth_max_error": (
                            max_error
                        ),

                        "selector_fit": (
                            result[
                                "fit"
                            ]
                        ),

                        "bottom_lambda": result[
                            "selected_bottom_lambda"
                        ],

                        "candidate_count": result[
                            "candidate_count"
                        ],

                        "restart_count": result[
                            "restart_count"
                        ],

                        "selected_restart": result[
                            "selected_restart"
                        ],

                        "restart_fit_gap": result[
                            "restart_fit_gap"
                        ],

                        "all_restart_fits": result[
                            "all_restart_fits"
                        ],

                        "runtime_seconds": (
                            result[
                                "runtime_seconds"
                            ]
                        ),

                        "estimated_k": (
                            array_text(
                                estimated_k
                            )
                        ),

                        "true_k": (
                            array_text(
                                true_k
                            )
                        ),
                    }
                )


                save_csv(
                    NOMINAL_RESULT_CSV,
                    result_rows,
                )


    summary_rows = (
        summarize_nominal(
            result_rows
        )
    )


    save_csv(
        NOMINAL_SUMMARY_CSV,
        summary_rows,
    )


    return (
        result_rows,
        summary_rows,
    )


# ======================================================================
# NOMINAL SUMMARY
# ======================================================================

def summarize_nominal(
    rows,
):

    output = []


    for number_of_modes in NOMINAL_K_VALUES:

        for snr_db in NOMINAL_SNR_VALUES:

            subset = [
                row
                for row
                in rows
                if (
                    int(
                        row[
                            "K"
                        ]
                    )
                    ==
                    number_of_modes

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


            rmse = np.asarray(
                [
                    row[
                        "truth_rmse"
                    ]
                    for row
                    in subset
                ],
                dtype=np.float64,
            )


            output.append(
                {
                    "K": (
                        number_of_modes
                    ),

                    "snr_db": (
                        snr_db
                    ),

                    "runs": (
                        len(
                            subset
                        )
                    ),

                    "success_rate": float(
                        np.mean(
                            [
                                row[
                                    "success"
                                ]
                                for row
                                in subset
                            ]
                        )
                    ),

                    "rmse_mean": float(
                        np.mean(
                            rmse
                        )
                    ),

                    "rmse_median": float(
                        np.median(
                            rmse
                        )
                    ),

                    "rmse_p90": float(
                        np.quantile(
                            rmse,
                            0.90,
                        )
                    ),

                    "runtime_mean": float(
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
# ROOT-CONSTRAINED ENVIRONMENT PROFILE
# ======================================================================

def evaluate_environment_candidate(
    noisy_data,
    inverse_base_offset,
    delta_c,
):

    final_offset = (

        float(
            inverse_base_offset
        )

        +

        float(
            delta_c
        )
    )


    set_inverse_ssp_offset(
        final_offset
    )


    try:

        roots = (
            BASE.calculate_physical_roots()
        )

    except Exception:

        return {
            "valid": False,
            "delta_c": float(
                delta_c
            ),
            "final_offset": float(
                final_offset
            ),
            "fit": float(
                "inf"
            ),
            "condition_number": float(
                "inf"
            ),
        }


    if (
        roots.size
        <
        ENVIRONMENT_TARGET_K
    ):

        return {
            "valid": False,
            "delta_c": float(
                delta_c
            ),
            "final_offset": float(
                final_offset
            ),
            "fit": float(
                "inf"
            ),
            "condition_number": float(
                "inf"
            ),
        }


    target_roots = np.asarray(
        roots[
            :ENVIRONMENT_TARGET_K
        ],
        dtype=np.float64,
    )


    try:

        mode_matrix = np.asarray(
            BASE.build_ho_mode_matrix_np(
                target_roots
            ),
            dtype=np.complex128,
        )


        coefficients, _, _, _ = (
            np.linalg.lstsq(
                mode_matrix,
                noisy_data,
                rcond=None,
            )
        )


        reconstruction = (

            mode_matrix

            @

            coefficients
        )


        fit = (

            np.linalg.norm(
                noisy_data
                -
                reconstruction
            )

            /

            max(
                np.linalg.norm(
                    noisy_data
                ),
                1.0e-15,
            )
        )


        condition_number = float(
            np.linalg.cond(
                mode_matrix
            )
        )


    except Exception:

        return {
            "valid": False,
            "delta_c": float(
                delta_c
            ),
            "final_offset": float(
                final_offset
            ),
            "fit": float(
                "inf"
            ),
            "condition_number": float(
                "inf"
            ),
        }


    return {
        "valid": True,

        "delta_c": float(
            delta_c
        ),

        "final_offset": float(
            final_offset
        ),

        "fit": float(
            fit
        ),

        "condition_number": (
            condition_number
        ),
    }


# ======================================================================
# ENVIRONMENT PROFILE SCAN
# ======================================================================

def scan_environment_profile(
    noisy_data,
    inverse_base_offset,
    scenario_name,
    snr_db,
    trial,
):

    profile_rows = []


    # ==================================================================
    # COARSE
    # ==================================================================

    coarse_grid = np.arange(
        DELTA_C_MIN,
        DELTA_C_MAX
        +
        COARSE_STEP
        *
        0.5,
        COARSE_STEP,
    )


    coarse_results = []


    for delta_c in coarse_grid:

        result = (
            evaluate_environment_candidate(
                noisy_data=(
                    noisy_data
                ),
                inverse_base_offset=(
                    inverse_base_offset
                ),
                delta_c=(
                    delta_c
                ),
            )
        )


        coarse_results.append(
            result
        )


        profile_rows.append(
            {
                "scenario": (
                    scenario_name
                ),

                "snr_db": (
                    snr_db
                ),

                "trial": (
                    trial
                ),

                "stage": (
                    "COARSE"
                ),

                "delta_c_mps": (
                    result[
                        "delta_c"
                    ]
                ),

                "final_ssp_offset_mps": (
                    result[
                        "final_offset"
                    ]
                ),

                "valid": int(
                    result[
                        "valid"
                    ]
                ),

                "fit": (
                    result[
                        "fit"
                    ]
                ),

                "condition_number": (
                    result[
                        "condition_number"
                    ]
                ),
            }
        )


    valid_coarse = [
        item
        for item
        in coarse_results
        if (
            item[
                "valid"
            ]
            and
            np.isfinite(
                item[
                    "fit"
                ]
            )
        )
    ]


    if not valid_coarse:

        return {
            "valid": False,
            "profile_rows": (
                profile_rows
            ),
        }


    best_coarse = min(
        valid_coarse,
        key=lambda x:
        x[
            "fit"
        ],
    )


    # ==================================================================
    # FINE
    # ==================================================================

    fine_lower = max(
        DELTA_C_MIN,
        best_coarse[
            "delta_c"
        ]
        -
        FINE_HALF_WIDTH,
    )


    fine_upper = min(
        DELTA_C_MAX,
        best_coarse[
            "delta_c"
        ]
        +
        FINE_HALF_WIDTH,
    )


    fine_grid = np.arange(
        fine_lower,
        fine_upper
        +
        FINE_STEP
        *
        0.5,
        FINE_STEP,
    )


    fine_results = []


    for delta_c in fine_grid:

        result = (
            evaluate_environment_candidate(
                noisy_data=(
                    noisy_data
                ),
                inverse_base_offset=(
                    inverse_base_offset
                ),
                delta_c=(
                    delta_c
                ),
            )
        )


        fine_results.append(
            result
        )


        profile_rows.append(
            {
                "scenario": (
                    scenario_name
                ),

                "snr_db": (
                    snr_db
                ),

                "trial": (
                    trial
                ),

                "stage": (
                    "FINE"
                ),

                "delta_c_mps": (
                    result[
                        "delta_c"
                    ]
                ),

                "final_ssp_offset_mps": (
                    result[
                        "final_offset"
                    ]
                ),

                "valid": int(
                    result[
                        "valid"
                    ]
                ),

                "fit": (
                    result[
                        "fit"
                    ]
                ),

                "condition_number": (
                    result[
                        "condition_number"
                    ]
                ),
            }
        )


    valid_fine = [
        item
        for item
        in fine_results
        if (
            item[
                "valid"
            ]
            and
            np.isfinite(
                item[
                    "fit"
                ]
            )
        )
    ]


    if not valid_fine:

        return {
            "valid": False,
            "profile_rows": (
                profile_rows
            ),
        }


    best = min(
        valid_fine,
        key=lambda x:
        x[
            "fit"
        ],
    )


    # ==================================================================
    # ZERO-CORRECTION REFERENCE
    # ==================================================================

    zero_result = (
        evaluate_environment_candidate(
            noisy_data=(
                noisy_data
            ),
            inverse_base_offset=(
                inverse_base_offset
            ),
            delta_c=(
                0.0
            ),
        )
    )


    if (
        not zero_result[
            "valid"
        ]
        or
        not np.isfinite(
            zero_result[
                "fit"
            ]
        )
    ):

        return {
            "valid": False,
            "profile_rows": (
                profile_rows
            ),
        }


    all_valid_fits = np.asarray(
        [
            item[
                "fit"
            ]
            for item
            in (
                valid_coarse
                +
                valid_fine
            )
        ],
        dtype=np.float64,
    )


    zero_fit = float(
        zero_result[
            "fit"
        ]
    )


    best_fit = float(
        best[
            "fit"
        ]
    )


    relative_depth = (

        zero_fit
        -
        best_fit

    ) / max(
        zero_fit,
        1.0e-15,
    )


    profile_span = (

        np.max(
            all_valid_fits
        )

        -

        np.min(
            all_valid_fits
        )

    ) / max(
        np.min(
            all_valid_fits
        ),
        1.0e-15,
    )


    best_delta = float(
        best[
            "delta_c"
        ]
    )


    boundary_distance = min(

        abs(
            best_delta
            -
            DELTA_C_MIN
        ),

        abs(
            DELTA_C_MAX
            -
            best_delta
        ),
    )


    boundary_hit = bool(

        boundary_distance

        <=

        BOUNDARY_MARGIN_MPS
    )


    return {
        "valid": True,

        "best_delta": (
            best_delta
        ),

        "best_fit": (
            best_fit
        ),

        "zero_fit": (
            zero_fit
        ),

        "relative_depth": float(
            relative_depth
        ),

        "relative_span": float(
            profile_span
        ),

        "boundary_hit": (
            boundary_hit
        ),

        "profile_rows": (
            profile_rows
        ),
    }


# ======================================================================
# AUTOMATIC ENVIRONMENT DECISION
# ======================================================================

def decide_environment_calibration(
    profile,
):

    if not profile.get(
        "valid",
        False,
    ):

        return {
            "accepted": False,

            "delta_to_use": 0.0,

            "status": (
                "PROFILE_INVALID_FALLBACK"
            ),
        }


    best_delta = float(
        profile[
            "best_delta"
        ]
    )


    relative_depth = float(
        profile[
            "relative_depth"
        ]
    )


    relative_span = float(
        profile[
            "relative_span"
        ]
    )


    boundary_hit = bool(
        profile[
            "boundary_hit"
        ]
    )


    # ------------------------------------------------------------------
    # If optimum is essentially zero:
    #
    # do not unnecessarily alter environment.
    # ------------------------------------------------------------------

    if (
        abs(
            best_delta
        )
        <=
        ZERO_CORRECTION_TOLERANCE_MPS
    ):

        return {
            "accepted": False,

            "delta_to_use": 0.0,

            "status": (
                "NO_CORRECTION_NEEDED"
            ),
        }


    # ------------------------------------------------------------------
    # Boundary optimum:
    #
    # cannot be trusted.
    # ------------------------------------------------------------------

    if boundary_hit:

        return {
            "accepted": False,

            "delta_to_use": 0.0,

            "status": (
                "BOUNDARY_REJECT_FALLBACK"
            ),
        }


    # ------------------------------------------------------------------
    # Flat profile:
    #
    # environment is unidentifiable.
    # ------------------------------------------------------------------

    if (
        relative_depth
        <
        MIN_RELATIVE_PROFILE_DEPTH
    ):

        return {
            "accepted": False,

            "delta_to_use": 0.0,

            "status": (
                "ENVIRONMENT_UNIDENTIFIABLE_DEPTH"
            ),
        }


    if (
        relative_span
        <
        MIN_RELATIVE_PROFILE_SPAN
    ):

        return {
            "accepted": False,

            "delta_to_use": 0.0,

            "status": (
                "ENVIRONMENT_UNIDENTIFIABLE_SPAN"
            ),
        }


    # ------------------------------------------------------------------
    # Strong enough evidence
    # ------------------------------------------------------------------

    return {
        "accepted": True,

        "delta_to_use": (
            best_delta
        ),

        "status": (
            "CALIBRATION_ACCEPTED"
        ),
    }


# ======================================================================
# ENVIRONMENT SECTION
# ======================================================================

def run_environment_section(
    kraken,
):

    print(
        "\n\n"
        + "=" * 210
    )

    print(
        "SECTION B — ENVIRONMENT IDENTIFIABILITY + SAFE FALLBACK"
    )

    print(
        "=" * 210
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
                :ENVIRONMENT_TARGET_K
            ]
        ),
        dtype=np.float64,
    )


    forward_matrix = np.asarray(
        kraken[
            "mode_matrix"
        ][
            :,
            :ENVIRONMENT_TARGET_K
        ],
        dtype=np.complex128,
    )


    result_rows = []

    profile_rows = []

    decision_rows = []

    modewise_rows = []


    for snr_db in ENVIRONMENT_SNR_VALUES:

        for trial in range(
            1,
            ENVIRONMENT_TRIALS + 1,
        ):

            # ==========================================================
            # Same data for every SSP mismatch
            # ==========================================================

            noisy_data, model_seed = (
                create_noisy_dataset(

                    forward_matrix=(
                        forward_matrix
                    ),

                    number_of_modes=(
                        ENVIRONMENT_TARGET_K
                    ),

                    snr_db=(
                        snr_db
                    ),

                    trial=(
                        trial
                    ),
                )
            )


            for inverse_offset in INVERSE_SSP_OFFSETS_MPS:

                scenario_name = (
                    f"INV_SSP_{inverse_offset:+.1f}"
                )


                print(
                    "\n"
                    + "-" * 200
                )

                print(
                    f"{scenario_name} | "
                    f"SNR={snr_db:+.1f} | "
                    f"trial={trial}"
                )

                print(
                    "-" * 200
                )


                # ======================================================
                # 1. BASELINE / UNCALIBRATED
                # ======================================================

                set_inverse_ssp_offset(
                    inverse_offset
                )


                wrong_roots = (
                    BASE.calculate_physical_roots()
                )


                if (
                    wrong_roots.size
                    >=
                    ENVIRONMENT_TARGET_K
                ):

                    baseline = (
                        run_mode_frinet(

                            noisy_data=(
                                noisy_data
                            ),

                            physical_roots=(
                                wrong_roots
                            ),

                            number_of_modes=(
                                ENVIRONMENT_TARGET_K
                            ),

                            random_seed=(
                                model_seed
                            ),
                        )
                    )

                else:

                    baseline = {
                        "selection_success": False,

                        "k": np.asarray(
                            [],
                            dtype=np.float64,
                        ),

                        "fit": float(
                            "inf"
                        ),

                        "runtime_seconds": 0.0,
                    }


                baseline_k = baseline[
                    "k"
                ]


                if (
                    baseline_k.size
                    ==
                    ENVIRONMENT_TARGET_K
                ):

                    baseline_rmse = (
                        calculate_rmse(
                            baseline_k,
                            true_k,
                        )
                    )


                    baseline_max_error = (
                        calculate_max_error(
                            baseline_k,
                            true_k,
                        )
                    )


                    baseline_success = int(

                        baseline_max_error

                        <=

                        BASE.SUCCESS_TOLERANCE
                    )

                else:

                    baseline_rmse = float(
                        "inf"
                    )

                    baseline_max_error = float(
                        "inf"
                    )

                    baseline_success = 0


                print(
                    f"Baseline RMSE = "
                    f"{baseline_rmse:.6e}"
                )


                # ======================================================
                # 2. ENVIRONMENT PROFILE
                # ======================================================

                profile_start = (
                    time.perf_counter()
                )


                profile = (
                    scan_environment_profile(

                        noisy_data=(
                            noisy_data
                        ),

                        inverse_base_offset=(
                            inverse_offset
                        ),

                        scenario_name=(
                            scenario_name
                        ),

                        snr_db=(
                            snr_db
                        ),

                        trial=(
                            trial
                        ),
                    )
                )


                profile_runtime = (

                    time.perf_counter()

                    -

                    profile_start
                )


                profile_rows.extend(
                    profile.get(
                        "profile_rows",
                        [],
                    )
                )


                # ======================================================
                # 3. SAFE DECISION
                # ======================================================

                decision = (
                    decide_environment_calibration(
                        profile
                    )
                )


                accepted = bool(
                    decision[
                        "accepted"
                    ]
                )


                delta_to_use = float(
                    decision[
                        "delta_to_use"
                    ]
                )


                final_offset = (

                    inverse_offset

                    +

                    delta_to_use
                )


                print(
                    "Environment diagnostic:"
                )

                if profile.get(
                    "valid",
                    False,
                ):

                    print(
                        f"  best delta      = "
                        f"{profile['best_delta']:+.4f}"
                    )

                    print(
                        f"  relative depth  = "
                        f"{profile['relative_depth']:.6e}"
                    )

                    print(
                        f"  relative span   = "
                        f"{profile['relative_span']:.6e}"
                    )

                    print(
                        f"  boundary hit    = "
                        f"{profile['boundary_hit']}"
                    )


                print(
                    f"  decision        = "
                    f"{decision['status']}"
                )

                print(
                    f"  accepted        = "
                    f"{accepted}"
                )

                print(
                    f"  delta used      = "
                    f"{delta_to_use:+.4f}"
                )


                # ======================================================
                # 4. CALIBRATED RUN — ONLY IF IDENTIFIABLE
                # ======================================================

                calibrated_result = None


                if accepted:

                    set_inverse_ssp_offset(
                        final_offset
                    )


                    corrected_roots = (
                        BASE.calculate_physical_roots()
                    )


                    if (
                        corrected_roots.size
                        >=
                        ENVIRONMENT_TARGET_K
                    ):

                        calibrated_result = (
                            run_mode_frinet(

                                noisy_data=(
                                    noisy_data
                                ),

                                physical_roots=(
                                    corrected_roots
                                ),

                                number_of_modes=(
                                    ENVIRONMENT_TARGET_K
                                ),

                                random_seed=(
                                    model_seed
                                ),
                            )
                        )


                # ======================================================
                # 5. FINAL SAFE OUTPUT
                # ======================================================

                if (
                    accepted
                    and
                    calibrated_result is not None
                    and
                    calibrated_result[
                        "k"
                    ].size
                    ==
                    ENVIRONMENT_TARGET_K
                ):

                    final_method = (
                        "CALIBRATED"
                    )

                    final_k = (
                        calibrated_result[
                            "k"
                        ]
                    )

                    final_fit = (
                        calibrated_result[
                            "fit"
                        ]
                    )

                    total_runtime = (

                        baseline[
                            "runtime_seconds"
                        ]

                        +

                        profile_runtime

                        +

                        calibrated_result[
                            "runtime_seconds"
                        ]
                    )


                else:

                    final_method = (
                        "FALLBACK_BASELINE"
                    )

                    final_k = (
                        baseline_k
                    )

                    final_fit = (
                        baseline[
                            "fit"
                        ]
                    )

                    total_runtime = (

                        baseline[
                            "runtime_seconds"
                        ]

                        +

                        profile_runtime
                    )


                if (
                    final_k.size
                    ==
                    ENVIRONMENT_TARGET_K
                ):

                    final_rmse = (
                        calculate_rmse(
                            final_k,
                            true_k,
                        )
                    )


                    final_max_error = (
                        calculate_max_error(
                            final_k,
                            true_k,
                        )
                    )


                    final_success = int(

                        final_max_error

                        <=

                        BASE.SUCCESS_TOLERANCE
                    )

                else:

                    final_rmse = float(
                        "inf"
                    )

                    final_max_error = float(
                        "inf"
                    )

                    final_success = 0


                print(
                    f"Final method  = "
                    f"{final_method}"
                )

                print(
                    f"Final RMSE    = "
                    f"{final_rmse:.6e}"
                )

                print(
                    f"Final success = "
                    f"{final_success}"
                )


                # ======================================================
                # SAVE MAIN RESULT
                # ======================================================

                result_rows.append(
                    {
                        "scenario": (
                            scenario_name
                        ),

                        "snr_db": (
                            snr_db
                        ),

                        "trial": (
                            trial
                        ),

                        "inverse_offset_mps": (
                            inverse_offset
                        ),

                        "profile_status": (
                            decision[
                                "status"
                            ]
                        ),

                        "calibration_accepted": int(
                            accepted
                        ),

                        "delta_used_mps": (
                            delta_to_use
                        ),

                        "final_ssp_offset_mps": (
                            final_offset
                        ),

                        "baseline_success": (
                            baseline_success
                        ),

                        "baseline_rmse": (
                            baseline_rmse
                        ),

                        "baseline_max_error": (
                            baseline_max_error
                        ),

                        "final_method": (
                            final_method
                        ),

                        "final_success": (
                            final_success
                        ),

                        "final_rmse": (
                            final_rmse
                        ),

                        "final_max_error": (
                            final_max_error
                        ),

                        "selector_fit": (
                            final_fit
                        ),

                        "runtime_seconds": (
                            total_runtime
                        ),

                        "final_k": (
                            array_text(
                                final_k
                            )
                        ),

                        "true_k": (
                            array_text(
                                true_k
                            )
                        ),
                    }
                )


                # ======================================================
                # DECISION OUTPUT
                # ======================================================

                decision_rows.append(
                    {
                        "scenario": (
                            scenario_name
                        ),

                        "snr_db": (
                            snr_db
                        ),

                        "trial": (
                            trial
                        ),

                        "inverse_offset_mps": (
                            inverse_offset
                        ),

                        "profile_valid": int(
                            profile.get(
                                "valid",
                                False,
                            )
                        ),

                        "best_delta_mps": (
                            profile.get(
                                "best_delta",
                                float(
                                    "nan"
                                ),
                            )
                        ),

                        "zero_fit": (
                            profile.get(
                                "zero_fit",
                                float(
                                    "nan"
                                ),
                            )
                        ),

                        "best_fit": (
                            profile.get(
                                "best_fit",
                                float(
                                    "nan"
                                ),
                            )
                        ),

                        "relative_profile_depth": (
                            profile.get(
                                "relative_depth",
                                float(
                                    "nan"
                                ),
                            )
                        ),

                        "relative_profile_span": (
                            profile.get(
                                "relative_span",
                                float(
                                    "nan"
                                ),
                            )
                        ),

                        "boundary_hit": int(
                            profile.get(
                                "boundary_hit",
                                False,
                            )
                        ),

                        "decision": (
                            decision[
                                "status"
                            ]
                        ),

                        "calibration_accepted": int(
                            accepted
                        ),

                        "delta_used_mps": (
                            delta_to_use
                        ),
                    }
                )


                # ======================================================
                # MODEWISE
                # ======================================================

                for mode_index in range(
                    ENVIRONMENT_TARGET_K
                ):

                    if (
                        baseline_k.size
                        ==
                        ENVIRONMENT_TARGET_K
                    ):

                        baseline_mode_error = abs(

                            baseline_k[
                                mode_index
                            ]

                            -

                            true_k[
                                mode_index
                            ]
                        )

                    else:

                        baseline_mode_error = float(
                            "nan"
                        )


                    if (
                        final_k.size
                        ==
                        ENVIRONMENT_TARGET_K
                    ):

                        final_mode_error = abs(

                            final_k[
                                mode_index
                            ]

                            -

                            true_k[
                                mode_index
                            ]
                        )

                    else:

                        final_mode_error = float(
                            "nan"
                        )


                    modewise_rows.append(
                        {
                            "scenario": (
                                scenario_name
                            ),

                            "snr_db": (
                                snr_db
                            ),

                            "trial": (
                                trial
                            ),

                            "mode": (
                                mode_index + 1
                            ),

                            "true_k": float(
                                true_k[
                                    mode_index
                                ]
                            ),

                            "baseline_k": float(
                                baseline_k[
                                    mode_index
                                ]
                            )
                            if (
                                baseline_k.size
                                ==
                                ENVIRONMENT_TARGET_K
                            )
                            else float(
                                "nan"
                            ),

                            "baseline_error": (
                                baseline_mode_error
                            ),

                            "final_k": float(
                                final_k[
                                    mode_index
                                ]
                            )
                            if (
                                final_k.size
                                ==
                                ENVIRONMENT_TARGET_K
                            )
                            else float(
                                "nan"
                            ),

                            "final_error": (
                                final_mode_error
                            ),

                            "final_method": (
                                final_method
                            ),
                        }
                    )


                # ======================================================
                # CHECKPOINT
                # ======================================================

                save_csv(
                    ENV_RESULT_CSV,
                    result_rows,
                )

                save_csv(
                    ENV_PROFILE_CSV,
                    profile_rows,
                )

                save_csv(
                    ENV_DECISION_CSV,
                    decision_rows,
                )

                save_csv(
                    ENV_MODEWISE_CSV,
                    modewise_rows,
                )

                save_csv(
                    ENV_SUMMARY_CSV,
                    summarize_environment(
                        result_rows
                    ),
                )


    summary_rows = (
        summarize_environment(
            result_rows
        )
    )


    save_csv(
        ENV_SUMMARY_CSV,
        summary_rows,
    )


    return (
        result_rows,
        decision_rows,
        summary_rows,
    )


# ======================================================================
# ENVIRONMENT SUMMARY
# ======================================================================

def summarize_environment(
    rows,
):

    output = []


    scenario_names = sorted(
        {
            row[
                "scenario"
            ]
            for row
            in rows
        }
    )


    for scenario in scenario_names:

        subset = [
            row
            for row
            in rows
            if row[
                "scenario"
            ]
            ==
            scenario
        ]


        if not subset:
            continue


        baseline_rmse = np.asarray(
            [
                row[
                    "baseline_rmse"
                ]
                for row
                in subset
            ],
            dtype=np.float64,
        )


        final_rmse = np.asarray(
            [
                row[
                    "final_rmse"
                ]
                for row
                in subset
            ],
            dtype=np.float64,
        )


        output.append(
            {
                "scenario": (
                    scenario
                ),

                "runs": (
                    len(
                        subset
                    )
                ),

                "calibration_accept_rate": float(
                    np.mean(
                        [
                            row[
                                "calibration_accepted"
                            ]
                            for row
                            in subset
                        ]
                    )
                ),

                "baseline_success_rate": float(
                    np.mean(
                        [
                            row[
                                "baseline_success"
                            ]
                            for row
                            in subset
                        ]
                    )
                ),

                "final_success_rate": float(
                    np.mean(
                        [
                            row[
                                "final_success"
                            ]
                            for row
                            in subset
                        ]
                    )
                ),

                "baseline_rmse_median": float(
                    np.median(
                        baseline_rmse
                    )
                ),

                "final_rmse_median": float(
                    np.median(
                        final_rmse
                    )
                ),

                "delta_used_mean": float(
                    np.mean(
                        [
                            row[
                                "delta_used_mps"
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
# FINAL CONSOLE REPORT
# ======================================================================

def print_final_report(
    nominal_summary,
    environment_summary,
):

    print(
        "\n\n"
        + "=" * 220
    )

    print(
        "MASTER PIPELINE FINAL REPORT"
    )

    print(
        "=" * 220
    )


    print(
        "\n[A] NOMINAL"
    )


    for row in nominal_summary:

        print(

            f"K={row['K']} | "
            f"SNR={row['snr_db']:>5.1f} | "
            f"success="
            f"{100*row['success_rate']:6.1f}% | "
            f"RMSEmed="
            f"{row['rmse_median']:.3e} | "
            f"P90="
            f"{row['rmse_p90']:.3e}"
        )


    print(
        "\n[B] ENVIRONMENT ROBUSTNESS"
    )


    for row in environment_summary:

        print(

            f"{row['scenario']:<15} | "
            f"accept="
            f"{100*row['calibration_accept_rate']:5.1f}% | "
            f"baseSucc="
            f"{100*row['baseline_success_rate']:5.1f}% | "
            f"finalSucc="
            f"{100*row['final_success_rate']:5.1f}% | "
            f"baseRMSE="
            f"{row['baseline_rmse_median']:.3e} | "
            f"finalRMSE="
            f"{row['final_rmse_median']:.3e}"
        )


    print(
        "\n"
        + "=" * 220
    )

    print(
        "INTERPRETATION"
    )

    print(
        "=" * 220
    )


    print(
        "1. 环境profile太平时，程序会自动拒绝校准。"
    )

    print(
        "2. 环境最优点撞±5 m/s边界时，程序会自动拒绝。"
    )

    print(
        "3. 被拒绝后直接返回原HO-PG Mode-FRINET结果。"
    )

    print(
        "4. 因此不会再因为极弱profile差异破坏nominal结果。"
    )

    print(
        "5. 如果未来多频数据让profile形成清晰内部最低点，"
        "同一套框架可以直接允许环境校准。"
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    torch.set_default_dtype(
        torch.float64
    )


    print(
        "=" * 220
    )

    print(
        "MODE-FRINET MASTER PIPELINE"
    )

    print(
        "=" * 220
    )

    print(
        f"RUN_MODE               = {RUN_MODE}"
    )

    print(
        f"Nominal K              = {NOMINAL_K_VALUES}"
    )

    print(
        f"Nominal SNR            = {NOMINAL_SNR_VALUES}"
    )

    print(
        f"Nominal trials         = {NOMINAL_TRIALS}"
    )

    print(
        f"Environment K          = {ENVIRONMENT_TARGET_K}"
    )

    print(
        f"Environment SNR        = {ENVIRONMENT_SNR_VALUES}"
    )

    print(
        f"Environment trials     = {ENVIRONMENT_TRIALS}"
    )

    print(
        f"SSP inverse offsets    = "
        f"{INVERSE_SSP_OFFSETS_MPS}"
    )

    print(
        f"lambda candidates by K = "
        f"{BOTTOM_LAMBDA_CANDIDATES_BY_K}"
    )

    print(
        f"restarts by K          = "
        f"{RESTARTS_BY_K}"
    )

    print(
        f"selector gamma         = "
        f"{FINAL_SELECTOR_GAMMA}"
    )

    print(
        f"profile depth minimum = "
        f"{MIN_RELATIVE_PROFILE_DEPTH:.3e}"
    )

    print(
        f"profile span minimum  = "
        f"{MIN_RELATIVE_PROFILE_SPAN:.3e}"
    )

    print(
        "=" * 220
    )


    # ==================================================================
    # HIGH-ORDER AUTOGRAD SANITY
    # ==================================================================

    BASE.verify_high_order_autograd()


    # ==================================================================
    # FIXED KRAKEN REFERENCE
    # ==================================================================

    restore_nominal_environment()


    kraken = (
        BASE.load_kraken()
    )


    print(
        "\nKRAKEN reference loaded."
    )


    if RUN_MODE.upper() == "K3_LAMBDA_FREEZE":

        try:
            run_k3_lambda_freeze(kraken)
        finally:
            restore_nominal_environment()

        print("\nOutputs:")
        print(K3_FREEZE_RESULT_CSV)
        print(K3_FREEZE_SUMMARY_CSV)
        return


    if RUN_MODE.upper() == "K3_RESTART_DIAGNOSTIC":

        try:
            run_k3_restart_diagnostic(kraken)
        finally:
            restore_nominal_environment()

        print("\nOutput:")
        print(K3_RESTART_RESULT_CSV)
        return


    try:

        # ==============================================================
        # SECTION A
        # ==============================================================

        (
            nominal_results,
            nominal_summary,
        ) = run_nominal_section(
            kraken
        )


        if RUN_MODE.upper() == "K3_MULTIRESTART_VALIDATE":

            print("\nK3 MULTI-RESTART VALIDATION COMPLETE")
            print(NOMINAL_RESULT_CSV)
            print(NOMINAL_SUMMARY_CSV)
            return


        # ==============================================================
        # SECTION B
        # ==============================================================

        (
            environment_results,
            environment_decisions,
            environment_summary,
        ) = run_environment_section(
            kraken
        )


    finally:

        restore_nominal_environment()


    # ==================================================================
    # REPORT
    # ==================================================================

    print_final_report(
        nominal_summary=(
            nominal_summary
        ),
        environment_summary=(
            environment_summary
        ),
    )


    print(
        "\nOutputs:"
    )

    print(
        NOMINAL_RESULT_CSV
    )

    print(
        NOMINAL_SUMMARY_CSV
    )

    print(
        ENV_RESULT_CSV
    )

    print(
        ENV_PROFILE_CSV
    )

    print(
        ENV_DECISION_CSV
    )

    print(
        ENV_MODEWISE_CSV
    )

    print(
        ENV_SUMMARY_CSV
    )


if __name__ == "__main__":
    main()

