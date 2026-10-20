# -*- coding: utf-8 -*-
"""
Unified Multi-Mode Mode-FRINET Validation
======================================================================

目标
----------------------------------------------------------------------
在同一套框架中验证：

    K = 3
    K = 5
    K = 8

Forward data:
    KRAKEN .MOD mode shapes

Inverse model:
    differentiable FD / Robin Mode-FRINET

Selector:
    Mode-Order Cell + Beam Search

Refinement:
    Safe physics refinement

======================================================================
研究逻辑
----------------------------------------------------------------------

此前K=3最终方法已经解决：

1. 非物理候选；
2. 合法高阶模态跨阶竞争；
3. 搜索边界附近高阶根截断；
4. KRAKEN独立forward data验证。

现在关注：

    方法能否从3模态扩展到5/8模态？

不能继续使用K=3的完整组合穷举：

    4^3 = 64
    4^5 = 1024
    4^8 = 65536

因此使用逐阶Beam Search。

对于第m阶模态，Robin根决定mode-order cell，
但cell内部具体选择哪个连续k仍由：

    observation data fit
    +
    physics/activity candidate quality

共同决定。

Beam Search在每加入一个cell后，只保留前BEAM_WIDTH
个部分组合。

======================================================================
重要研究口径
----------------------------------------------------------------------

1. KRAKEN mode shape只用于forward data。
2. inverse side不使用KRAKEN mode shape。
3. KRAKEN实波数只用于事后truth evaluation。
4. selector使用FD/Robin物理根划分mode-order区域。
5. refined结果必须和raw结果分开报告。
6. K=5/8需要扩大training k range，因为旧K_MIN=1.20
   本身不包含k5~k8。
7. QUICK阶段只做有限SNR和trial，用于验证可扩展性。
8. QUICK通过后再做正式FULL。
"""

from __future__ import annotations

import csv
import importlib.util
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# ======================================================================
# 工程路径
# ======================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]

EXP26_PATH = (
    Path(__file__).resolve().parent
    / "26_realistic_bottom_mode_frinet.py"
)

KRAKEN_VALIDATION_PATH = (
    Path(__file__).resolve().parent
    / "kraken_mod_independent_validation.py"
)

KRAKEN_MOD_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "24_kraken_reference_environment"
    / "kraken_case"
    / "teacher_paper_grad05.mod"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "multimode_mode_frinet_final"
)

FIGURE_DIR = RESULT_DIR / "figures"

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ======================================================================
# 动态加载模块
# ======================================================================
def load_python_module(
    path: Path,
    module_name: str,
):

    if not path.exists():

        raise FileNotFoundError(
            f"\n没有找到文件：\n{path}\n"
        )

    spec = (
        importlib.util.spec_from_file_location(
            module_name,
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):

        raise RuntimeError(
            f"无法加载：{path}"
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    sys.modules[
        module_name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


EXP26 = load_python_module(
    EXP26_PATH,
    "exp26_for_multimode",
)

KRAKEN_IO = load_python_module(
    KRAKEN_VALIDATION_PATH,
    "kraken_validation_for_multimode",
)


# ======================================================================
# QUICK / FULL
# ======================================================================
QUICK_TEST = False


if QUICK_TEST:

    TARGET_MODE_COUNTS = [
        3,
        5,
        8,
    ]

    SNR_VALUES_DB = [
        5.0,
        15.0,
    ]

    NUMBER_OF_TRIALS = 2

    NUMBER_OF_RESTARTS = 2

    RUN_SUFFIX = "quick"

else:

    TARGET_MODE_COUNTS = [
        3,
        5,
        8,
    ]

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
# 基本配置
# ======================================================================
NUMBER_OF_SNAPSHOTS = 20

NUMBER_OF_ATOMS = 24

BASE_RANDOM_SEED = 20260903

SUCCESS_TOLERANCE = 1.5e-3


# ======================================================================
# 不同K的训练搜索范围
#
# 关键：
# 旧K_MIN=1.20无法包含k5~k8。
# ======================================================================
K_MIN_BY_MODE_COUNT = {

    3: 1.20,

    5: 1.18,

    8: 1.08,
}


# ======================================================================
# Beam Search参数
# ======================================================================

# 每一个mode-order cell中最多保留多少个局部候选
TOP_CANDIDATES_PER_CELL = 4

# 每完成一个模态阶次扩展以后，
# 最多保留多少个部分组合。
BEAM_WIDTH = 32

# cell内部去除几乎重复的连续原子
CELL_INTERNAL_MIN_SEPARATION = 1.0e-3


# ======================================================================
# Safe refinement
# ======================================================================
REFINEMENT_CAPTURE_RADIUS = 0.006

MAXIMUM_FIT_DEGRADATION = 1.10


# ======================================================================
# 输出
# ======================================================================
RESTART_CSV = (
    RESULT_DIR
    / f"multimode_restart_{RUN_SUFFIX}.csv"
)

TRIAL_CSV = (
    RESULT_DIR
    / f"multimode_trials_{RUN_SUFFIX}.csv"
)

SUMMARY_CSV = (
    RESULT_DIR
    / f"multimode_summary_{RUN_SUFFIX}.csv"
)

MODE_SHAPE_CSV = (
    RESULT_DIR
    / "multimode_kraken_fd_shape_mismatch.csv"
)


# ======================================================================
# 通用工具
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


def array_to_text(
    values,
):

    return ";".join(
        f"{float(x):.12f}"
        for x
        in np.asarray(
            values
        ).ravel()
    )


# ======================================================================
# 数据拟合
# ======================================================================
def calculate_data_fit(
    wavenumbers: np.ndarray,
    noisy_data: np.ndarray,
):

    k = np.asarray(
        wavenumbers,
        dtype=np.float64,
    )

    if k.size == 0:

        return float("inf")


    mode_matrix = np.asarray(
        EXP26.build_mode_matrix(
            k
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
        @ coefficients
    )


    residual = float(
        np.linalg.norm(
            noisy_data
            - reconstruction
        )
        /
        max(
            np.linalg.norm(
                noisy_data
            ),
            1.0e-14,
        )
    )


    return residual


# ======================================================================
# 误差指标
# ======================================================================
def calculate_k_metrics(
    estimated: np.ndarray,
    truth: np.ndarray,
):

    estimated = np.sort(
        np.asarray(
            estimated,
            dtype=np.float64,
        )
    )[::-1]

    truth = np.sort(
        np.asarray(
            truth,
            dtype=np.float64,
        )
    )[::-1]


    if estimated.size != truth.size:

        return {
            "success": False,
            "rmse": float("inf"),
            "max_error": float("inf"),
            "mean_abs_error": float("inf"),
        }


    errors = np.abs(
        estimated
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

        "mean_abs_error": float(
            np.mean(
                errors
            )
        ),
    }


# ======================================================================
# 扩展Robin根
#
# 对K=8，最低目标根约1.105，
# 因此1.08足够覆盖前8个trapped modes。
# ======================================================================
def calculate_extended_roots():

    physics_k_min = 1.08

    k_max = float(
        EXP26.K_MAX
    )


    roots05 = np.asarray(
        EXP26.find_robin_roots(
            k_min=physics_k_min,
            k_max=k_max,
            dz=0.5,
        ),
        dtype=np.float64,
    )


    roots025 = np.asarray(
        EXP26.find_robin_roots(
            k_min=physics_k_min,
            k_max=k_max,
            dz=0.25,
        ),
        dtype=np.float64,
    )


    roots05 = np.sort(
        roots05
    )[::-1]

    roots025 = np.sort(
        roots025
    )[::-1]


    n = min(
        roots05.size,
        roots025.size,
    )


    if n < 8:

        raise RuntimeError(
            "\n扩展Robin根不足8个。\n"
            f"dz0.5 = {roots05.size}\n"
            f"dz0.25 = {roots025.size}\n"
        )


    roots = (
        roots025[:n]
        +
        (
            roots025[:n]
            - roots05[:n]
        )
        / 3.0
    )


    return np.sort(
        roots
    )[::-1]


# ======================================================================
# mode-order cell
#
# 与K=3版本的重要区别：
#
# 最后一阶不要求k_(K+1)。
#
# 例如K=8：
#
# cell8 =
#     [training_K_MIN,
#      midpoint(k7,k8)]
#
# 因此不需要一个可能已经进入leaky regime的k9。
# ======================================================================
def build_mode_order_cells(
    physical_roots: np.ndarray,
    number_of_modes: int,
    training_k_min: float,
    training_k_max: float,
):

    roots = np.sort(
        np.asarray(
            physical_roots,
            dtype=np.float64,
        )
    )[::-1]


    if roots.size < number_of_modes:

        raise RuntimeError(
            f"物理根只有{roots.size}个，"
            f"不足以支持K={number_of_modes}。"
        )


    target_roots = (
        roots[
            :number_of_modes
        ]
    )


    boundaries = []

    for index in range(
        number_of_modes - 1
    ):

        midpoint = (
            target_roots[
                index
            ]
            +
            target_roots[
                index + 1
            ]
        ) / 2.0

        boundaries.append(
            float(
                midpoint
            )
        )


    cells = []


    for mode_index in range(
        number_of_modes
    ):

        # ----------------------------------------------
        # 第一阶：
        # midpoint(k1,k2) ~ K_MAX
        # ----------------------------------------------
        if mode_index == 0:

            lower = (
                boundaries[0]
            )

            upper = (
                training_k_max
                + 1e-12
            )


        # ----------------------------------------------
        # 最后一阶：
        # K_MIN ~ midpoint(k_(K-1),k_K)
        # ----------------------------------------------
        elif (
            mode_index
            ==
            number_of_modes - 1
        ):

            lower = (
                training_k_min
                - 1e-12
            )

            upper = (
                boundaries[
                    mode_index - 1
                ]
            )


        # ----------------------------------------------
        # 中间阶
        # ----------------------------------------------
        else:

            lower = (
                boundaries[
                    mode_index
                ]
            )

            upper = (
                boundaries[
                    mode_index - 1
                ]
            )


        cells.append(
            {
                "mode": (
                    mode_index + 1
                ),

                "lower": float(
                    lower
                ),

                "upper": float(
                    upper
                ),

                "physical_root": float(
                    target_roots[
                        mode_index
                    ]
                ),
            }
        )


    return {
        "roots": (
            target_roots
        ),

        "boundaries": (
            boundaries
        ),

        "cells": (
            cells
        ),
    }


# ======================================================================
# physics/activity score
# ======================================================================
def obtain_physics_scores(
    wavenumbers: np.ndarray,
    activities: np.ndarray,
):

    scores, residuals = (
        EXP26.calculate_physics_scores(
            wavenumbers=(
                wavenumbers
            ),
            activities=(
                activities
            ),
        )
    )


    return (
        np.asarray(
            scores,
            dtype=np.float64,
        ),
        np.asarray(
            residuals,
            dtype=np.float64,
        ),
    )


# ======================================================================
# 每个cell中筛局部候选
# ======================================================================
def get_cell_candidates(
    wavenumbers: np.ndarray,
    scores: np.ndarray,
    lower: float,
    upper: float,
):

    mask_indices = np.where(
        (
            wavenumbers
            > lower
        )
        &
        (
            wavenumbers
            <= upper
        )
    )[0]


    if mask_indices.size == 0:

        return []


    ordered = (
        mask_indices[
            np.argsort(
                scores[
                    mask_indices
                ]
            )[::-1]
        ]
    )


    selected = []


    for index in ordered:

        index = int(
            index
        )

        k = float(
            wavenumbers[
                index
            ]
        )


        separated = all(
            abs(
                k
                -
                float(
                    wavenumbers[
                        old_index
                    ]
                )
            )
            >= CELL_INTERNAL_MIN_SEPARATION

            for old_index
            in selected
        )


        if separated:

            selected.append(
                index
            )


        if (
            len(
                selected
            )
            >= TOP_CANDIDATES_PER_CELL
        ):

            break


    return selected


# ======================================================================
# Beam Search selector
# ======================================================================
def select_mode_order_beam(
    wavenumbers: np.ndarray,
    activities: np.ndarray,
    noisy_data: np.ndarray,
    physical_roots: np.ndarray,
    number_of_modes: int,
    training_k_min: float,
    training_k_max: float,
):

    wavenumbers = np.asarray(
        wavenumbers,
        dtype=np.float64,
    )

    activities = np.asarray(
        activities,
        dtype=np.float64,
    )


    scores, _ = obtain_physics_scores(
        wavenumbers,
        activities,
    )


    cell_info = (
        build_mode_order_cells(
            physical_roots=(
                physical_roots
            ),
            number_of_modes=(
                number_of_modes
            ),
            training_k_min=(
                training_k_min
            ),
            training_k_max=(
                training_k_max
            ),
        )
    )


    cell_candidates = []


    for cell in cell_info[
        "cells"
    ]:

        candidates = (
            get_cell_candidates(
                wavenumbers=(
                    wavenumbers
                ),
                scores=(
                    scores
                ),
                lower=(
                    cell[
                        "lower"
                    ]
                ),
                upper=(
                    cell[
                        "upper"
                    ]
                ),
            )
        )


        if not candidates:

            return {

                "success": False,

                "reason": (
                    f"EMPTY_CELL_MODE_"
                    f"{cell['mode']}"
                ),

                "selected": np.asarray(
                    [],
                    dtype=np.float64,
                ),

                "fit": float("inf"),

                "cell_candidate_counts": [
                    len(x)
                    for x in cell_candidates
                ] + [0],
            }


        cell_candidates.append(
            candidates
        )


    # ==============================================================
    # Beam:
    #
    # 每个元素：
    #
    # {
    #   indices,
    #   fit,
    #   physics_score
    # }
    #
    # ranking:
    #   1. data fit
    #   2. physics score
    # ==============================================================

    beam = [
        {
            "indices": tuple(),
            "fit": 1.0,
            "physics_score": 0.0,
        }
    ]


    for mode_index in range(
        number_of_modes
    ):

        expanded = []


        for state in beam:

            for new_index in (
                cell_candidates[
                    mode_index
                ]
            ):

                indices = (
                    state[
                        "indices"
                    ]
                    +
                    (
                        int(
                            new_index
                        ),
                    )
                )


                k_values = np.asarray(
                    [
                        wavenumbers[
                            index
                        ]
                        for index
                        in indices
                    ],
                    dtype=np.float64,
                )


                fit = (
                    calculate_data_fit(
                        wavenumbers=(
                            k_values
                        ),
                        noisy_data=(
                            noisy_data
                        ),
                    )
                )


                physics_score = float(
                    np.sum(
                        [
                            scores[
                                index
                            ]
                            for index
                            in indices
                        ]
                    )
                )


                expanded.append(
                    {
                        "indices": (
                            indices
                        ),

                        "fit": float(
                            fit
                        ),

                        "physics_score": (
                            physics_score
                        ),
                    }
                )


        if not expanded:

            return {

                "success": False,

                "reason": (
                    "BEAM_EMPTY"
                ),

                "selected": np.asarray(
                    [],
                    dtype=np.float64,
                ),

                "fit": float("inf"),

                "cell_candidate_counts": [
                    len(x)
                    for x
                    in cell_candidates
                ],
            }


        # ----------------------------------------------------------
        # lexicographic：
        #
        # fit越低优先；
        # fit相同则physics score越高优先。
        # ----------------------------------------------------------
        expanded.sort(
            key=lambda state: (
                state[
                    "fit"
                ],
                -state[
                    "physics_score"
                ],
            )
        )


        beam = (
            expanded[
                :BEAM_WIDTH
            ]
        )


    best = (
        beam[0]
    )


    selected = np.asarray(
        [
            wavenumbers[
                index
            ]
            for index
            in best[
                "indices"
            ]
        ],
        dtype=np.float64,
    )


    return {

        "success": True,

        "reason": (
            "BEAM_SELECTED"
        ),

        "selected": (
            selected
        ),

        "fit": float(
            best[
                "fit"
            ]
        ),

        "physics_score": float(
            best[
                "physics_score"
            ]
        ),

        "cell_candidate_counts": [
            len(x)
            for x
            in cell_candidates
        ],
    }


# ======================================================================
# Safe refinement
# ======================================================================
def safe_refine(
    estimated: np.ndarray,
    physical_roots: np.ndarray,
    noisy_data: np.ndarray,
):

    raw = np.sort(
        np.asarray(
            estimated,
            dtype=np.float64,
        )
    )[::-1]


    roots = np.sort(
        np.asarray(
            physical_roots,
            dtype=np.float64,
        )
    )[::-1]


    if raw.size != roots.size:

        return {

            "accepted": False,

            "refined": raw,

            "raw_fit": float("inf"),

            "refined_fit": float("inf"),
        }


    refined = (
        raw.copy()
    )


    for index in range(
        raw.size
    ):

        if (
            abs(
                raw[
                    index
                ]
                -
                roots[
                    index
                ]
            )
            <= REFINEMENT_CAPTURE_RADIUS
        ):

            refined[
                index
            ] = roots[
                index
            ]


    raw_fit = (
        calculate_data_fit(
            raw,
            noisy_data,
        )
    )


    refined_fit = (
        calculate_data_fit(
            refined,
            noisy_data,
        )
    )


    accepted = bool(
        refined_fit
        <=
        raw_fit
        * MAXIMUM_FIT_DEGRADATION
    )


    return {

        "accepted": (
            accepted
        ),

        "refined": (
            refined
            if accepted
            else raw
        ),

        "raw_fit": (
            raw_fit
        ),

        "refined_fit": (
            refined_fit
        ),
    }


# ======================================================================
# KRAKEN数据
# ======================================================================
def prepare_kraken_data():

    mod_data = (
        KRAKEN_IO.parse_kraken_mod(
            KRAKEN_MOD_PATH
        )
    )


    receiver = (
        KRAKEN_IO.extract_integer_receiver_modes(
            mod_data
        )
    )


    all_receiver_modes = np.asarray(
        receiver[
            "receiver_mode_matrix"
        ],
        dtype=np.complex128,
    )


    complex_k = np.asarray(
        mod_data[
            "complex_wavenumbers"
        ],
        dtype=np.complex128,
    )


    return {

        "mod_data": mod_data,

        "receiver_modes": (
            all_receiver_modes
        ),

        "complex_k": (
            complex_k
        ),
    }


# ======================================================================
# 针对任意K适配KRAKEN mode matrix
# ======================================================================
def prepare_forward_matrix(
    receiver_modes: np.ndarray,
    true_k: np.ndarray,
):

    receiver_modes = np.asarray(
        receiver_modes,
        dtype=np.complex128,
    )


    target_receiver_modes = (
        receiver_modes[
            :,
            :true_k.size
        ]
    )


    inverse_probe = np.asarray(
        EXP26.build_mode_matrix(
            true_k
        ),
        dtype=np.complex128,
    )


    if (
        inverse_probe.shape[0]
        ==
        target_receiver_modes.shape[0]
    ):

        kraken_modes = (
            target_receiver_modes.copy()
        )


    elif (
        inverse_probe.shape[0]
        ==
        target_receiver_modes.shape[0]
        + 1
    ):

        # pressure-release z=0
        zero_surface = np.zeros(
            (
                1,
                true_k.size
            ),
            dtype=np.complex128,
        )


        kraken_modes = np.vstack(
            [
                zero_surface,
                target_receiver_modes,
            ]
        )


    else:

        raise RuntimeError(
            "\nKRAKEN / inverse深度维不兼容：\n"
            f"KRAKEN={target_receiver_modes.shape}\n"
            f"inverse={inverse_probe.shape}\n"
        )


    # column normalization
    for matrix in [
        kraken_modes,
        inverse_probe,
    ]:

        for column in range(
            matrix.shape[1]
        ):

            norm = np.linalg.norm(
                matrix[
                    :,
                    column
                ]
            )

            matrix[
                :,
                column
            ] /= max(
                norm,
                1e-14,
            )


    return (
        kraken_modes,
        inverse_probe,
    )


# ======================================================================
# mode mismatch
# ======================================================================
def shape_mismatch(
    phi_k: np.ndarray,
    phi_fd: np.ndarray,
):

    denominator = np.vdot(
        phi_fd,
        phi_fd,
    )


    alpha = (
        np.vdot(
            phi_fd,
            phi_k,
        )
        /
        denominator
    )


    aligned = (
        alpha
        * phi_fd
    )


    mismatch = float(
        np.linalg.norm(
            phi_k
            - aligned
        )
        /
        max(
            np.linalg.norm(
                phi_k
            ),
            1e-14,
        )
    )


    correlation = float(
        abs(
            np.vdot(
                phi_k,
                phi_fd,
            )
        )
        /
        max(
            np.linalg.norm(
                phi_k
            )
            *
            np.linalg.norm(
                phi_fd
            ),
            1e-14,
        )
    )


    return (
        mismatch,
        correlation,
    )


# ======================================================================
# KRAKEN synthetic snapshots
# ======================================================================
def make_master_data(
    forward_matrix: np.ndarray,
    coefficient_seed: int,
    noise_seed: int,
):

    K = (
        forward_matrix.shape[1]
    )


    rng_c = np.random.default_rng(
        coefficient_seed
    )


    coefficients = (
        rng_c.standard_normal(
            (
                K,
                NUMBER_OF_SNAPSHOTS,
            )
        )
        +
        1j
        *
        rng_c.standard_normal(
            (
                K,
                NUMBER_OF_SNAPSHOTS,
            )
        )
    ) / np.sqrt(2.0)


    clean = (
        forward_matrix
        @ coefficients
    )


    rng_n = np.random.default_rng(
        noise_seed
    )


    noise = (
        rng_n.standard_normal(
            clean.shape
        )
        +
        1j
        *
        rng_n.standard_normal(
            clean.shape
        )
    ) / np.sqrt(2.0)


    noise /= max(
        np.linalg.norm(
            noise
        ),
        1e-14,
    )


    return {
        "clean": clean,
        "unit_noise": noise,
    }


def add_noise(
    master: dict,
    snr_db: float,
):

    clean = np.asarray(
        master[
            "clean"
        ],
        dtype=np.complex128,
    )


    unit_noise = np.asarray(
        master[
            "unit_noise"
        ],
        dtype=np.complex128,
    )


    clean_norm = float(
        np.linalg.norm(
            clean
        )
    )


    noise_norm = (
        clean_norm
        /
        (
            10.0
            **
            (
                snr_db
                / 20.0
            )
        )
    )


    noisy = (
        clean
        +
        unit_noise
        * noise_norm
    )


    return noisy


# ======================================================================
# best restart
# ======================================================================
def choose_best_restart(
    results,
):

    valid = [
        index
        for index, result
        in enumerate(
            results
        )
        if result[
            "selector_success"
        ]
        and np.isfinite(
            result[
                "restart_score"
            ]
        )
    ]


    if not valid:

        return None


    return min(
        valid,
        key=lambda index:
        results[
            index
        ][
            "restart_score"
        ],
    )


# ======================================================================
# 汇总
# ======================================================================
def summarize_trials(
    trial_rows,
):

    output = []


    for K in TARGET_MODE_COUNTS:

        for snr in SNR_VALUES_DB:

            subset = [
                row
                for row
                in trial_rows
                if int(
                    row[
                        "K"
                    ]
                )
                == K
                and float(
                    row[
                        "snr_db"
                    ]
                )
                == snr
            ]


            if not subset:
                continue


            def vals(
                key
            ):

                return np.asarray(
                    [
                        float(
                            row[
                                key
                            ]
                        )
                        for row
                        in subset
                    ]
                )


            raw_rmse = vals(
                "raw_rmse"
            )

            ref_rmse = vals(
                "refined_rmse"
            )


            output.append(
                {
                    "K": K,

                    "snr_db": (
                        snr
                    ),

                    "trials": (
                        len(
                            subset
                        )
                    ),

                    "raw_success_rate": float(
                        np.mean(
                            vals(
                                "raw_success"
                            )
                        )
                    ),

                    "refined_success_rate": float(
                        np.mean(
                            vals(
                                "refined_success"
                            )
                        )
                    ),

                    "any_raw_success_rate": float(
                        np.mean(
                            vals(
                                "any_raw_success"
                            )
                        )
                    ),

                    "restart_success_fraction_mean": float(
                        np.mean(
                            vals(
                                "restart_success_fraction"
                            )
                        )
                    ),

                    "raw_rmse_median": float(
                        np.median(
                            raw_rmse[
                                np.isfinite(
                                    raw_rmse
                                )
                            ]
                        )
                    ),

                    "refined_rmse_median": float(
                        np.median(
                            ref_rmse[
                                np.isfinite(
                                    ref_rmse
                                )
                            ]
                        )
                    ),

                    "runtime_seconds_mean": float(
                        np.mean(
                            vals(
                                "runtime_seconds"
                            )
                        )
                    ),
                }
            )


    return output


# ======================================================================
# 图
# ======================================================================
def plot_results(
    summary_rows,
):

    # --------------------------------------------------------------
    # Raw success
    # --------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(
            8.5,
            5.5,
        )
    )


    for K in TARGET_MODE_COUNTS:

        subset = [
            row
            for row
            in summary_rows
            if row[
                "K"
            ]
            == K
        ]


        subset.sort(
            key=lambda x:
            x[
                "snr_db"
            ]
        )


        ax.plot(
            [
                row[
                    "snr_db"
                ]
                for row
                in subset
            ],

            [
                100
                *
                row[
                    "raw_success_rate"
                ]
                for row
                in subset
            ],

            marker="o",

            label=f"K={K}",
        )


    ax.set_ylim(
        -5,
        105,
    )

    ax.set_xlabel(
        "SNR (dB)"
    )

    ax.set_ylabel(
        "Raw success rate (%)"
    )

    ax.set_title(
        "Multi-Mode KRAKEN Independent Validation"
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
            f"multimode_raw_success_"
            f"{RUN_SUFFIX}.png"
        ),
        dpi=220,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


# ======================================================================
# MAIN
# ======================================================================
def main():

    torch.set_default_dtype(
        torch.float64
    )


    EXP26.NUMBER_OF_SNAPSHOTS = (
        NUMBER_OF_SNAPSHOTS
    )

    EXP26.NUMBER_OF_ATOMS = (
        NUMBER_OF_ATOMS
    )


    print(
        "=" * 160
    )

    print(
        "UNIFIED MULTI-MODE MODE-FRINET"
    )

    print(
        "=" * 160
    )

    print(
        f"Mode              = "
        f"{RUN_SUFFIX}"
    )

    print(
        f"K                 = "
        f"{TARGET_MODE_COUNTS}"
    )

    print(
        f"SNR               = "
        f"{SNR_VALUES_DB}"
    )

    print(
        f"Trials            = "
        f"{NUMBER_OF_TRIALS}"
    )

    print(
        f"Restarts          = "
        f"{NUMBER_OF_RESTARTS}"
    )

    print(
        f"L                 = "
        f"{NUMBER_OF_SNAPSHOTS}"
    )

    print(
        f"M                 = "
        f"{NUMBER_OF_ATOMS}"
    )

    print(
        f"Beam width        = "
        f"{BEAM_WIDTH}"
    )

    print(
        "=" * 160
    )


    # ==================================================================
    # KRAKEN
    # ==================================================================
    kraken = (
        prepare_kraken_data()
    )


    complex_k = (
        kraken[
            "complex_k"
        ]
    )


    receiver_modes = (
        kraken[
            "receiver_modes"
        ]
    )


    print(
        "\nKRAKEN前8阶："
    )


    for index in range(
        8
    ):

        value = (
            complex_k[
                index
            ]
        )

        print(
            f"  k{index + 1} = "
            f"{value.real:.12f} "
            f"{value.imag:+.5e}j"
        )


    # ==================================================================
    # Robin roots
    # ==================================================================
    physical_roots = (
        calculate_extended_roots()
    )


    print(
        "\nRobin前8阶："
    )


    for index in range(
        8
    ):

        print(
            f"  r{index + 1} = "
            f"{physical_roots[index]:.12f}"
        )


    # ==================================================================
    # shape mismatch
    # ==================================================================
    shape_rows = []


    for K in TARGET_MODE_COUNTS:

        true_k = np.real(
            complex_k[
                :K
            ]
        ).astype(
            np.float64
        )


        forward_matrix, fd_matrix = (
            prepare_forward_matrix(
                receiver_modes=(
                    receiver_modes
                ),
                true_k=(
                    true_k
                ),
            )
        )


        for mode_index in range(
            K
        ):

            mismatch, corr = (
                shape_mismatch(
                    forward_matrix[
                        :,
                        mode_index
                    ],
                    fd_matrix[
                        :,
                        mode_index
                    ],
                )
            )


            shape_rows.append(
                {
                    "K": K,

                    "mode": (
                        mode_index + 1
                    ),

                    "relative_shape_mismatch": (
                        mismatch
                    ),

                    "correlation": (
                        corr
                    ),
                }
            )


    save_csv(
        MODE_SHAPE_CSV,
        shape_rows,
    )


    # ==================================================================
    # Monte Carlo
    # ==================================================================
    restart_rows = []

    trial_rows = []


    for K in TARGET_MODE_COUNTS:

        print(
            "\n\n"
            + "=" * 160
        )

        print(
            f"K = {K}"
        )

        print(
            "=" * 160
        )


        training_k_min = (
            K_MIN_BY_MODE_COUNT[
                K
            ]
        )


        # --------------------------------------------------------------
        # 修改训练搜索下限
        # --------------------------------------------------------------
        EXP26.K_MIN = float(
            training_k_min
        )


        true_k = np.asarray(
            np.real(
                complex_k[
                    :K
                ]
            ),
            dtype=np.float64,
        )


        target_roots = np.asarray(
            physical_roots[
                :K
            ],
            dtype=np.float64,
        )


        forward_matrix, _ = (
            prepare_forward_matrix(
                receiver_modes=(
                    receiver_modes
                ),
                true_k=(
                    true_k
                ),
            )
        )


        print(
            f"\nTraining range: "
            f"[{training_k_min:.6f}, "
            f"{EXP26.K_MAX:.6f}]"
        )


        print(
            "\nTruth:"
        )

        print(
            true_k
        )


        print(
            "\nRobin roots:"
        )

        print(
            target_roots
        )


        cell_info = (
            build_mode_order_cells(
                physical_roots=(
                    physical_roots
                ),
                number_of_modes=(
                    K
                ),
                training_k_min=(
                    training_k_min
                ),
                training_k_max=(
                    EXP26.K_MAX
                ),
            )
        )


        print(
            "\nOrder cells:"
        )


        for cell in (
            cell_info[
                "cells"
            ]
        ):

            print(
                f"  mode "
                f"{cell['mode']}: "
                f"({cell['lower']:.6f}, "
                f"{cell['upper']:.6f}]"
            )


        # --------------------------------------------------------------
        # Trial master data
        # --------------------------------------------------------------
        masters = {}


        for trial in range(
            1,
            NUMBER_OF_TRIALS + 1,
        ):

            seed_base = (
                BASE_RANDOM_SEED
                +
                K * 1000000
                +
                trial * 10000
            )


            masters[
                trial
            ] = (
                make_master_data(
                    forward_matrix=(
                        forward_matrix
                    ),
                    coefficient_seed=(
                        seed_base
                        + 1
                    ),
                    noise_seed=(
                        seed_base
                        + 2
                    ),
                )
            )


        # --------------------------------------------------------------
        # SNR
        # --------------------------------------------------------------
        for snr_db in SNR_VALUES_DB:

            print(
                "\n"
                + "-" * 140
            )

            print(
                f"K={K} | "
                f"SNR={snr_db:+.1f} dB"
            )

            print(
                "-" * 140
            )


            EXP26.SNR_DB = float(
                snr_db
            )


            for trial in range(
                1,
                NUMBER_OF_TRIALS + 1,
            ):

                noisy_data = (
                    add_noise(
                        masters[
                            trial
                        ],
                        snr_db,
                    )
                )


                print(
                    f"\nTrial "
                    f"{trial}/"
                    f"{NUMBER_OF_TRIALS}"
                )


                restart_results = []

                total_runtime = 0.0


                for restart_index in range(
                    NUMBER_OF_RESTARTS
                ):

                    seed = (
                        BASE_RANDOM_SEED
                        +
                        K * 1000000
                        +
                        trial * 10000
                        +
                        int(
                            (
                                snr_db
                                + 20
                            )
                            * 1000
                        )
                        +
                        restart_index
                        * 1009
                    )


                    print(
                        f"  restart "
                        f"{restart_index + 1}/"
                        f"{NUMBER_OF_RESTARTS}",
                        flush=True,
                    )


                    start = (
                        time.time()
                    )


                    trained = (
                        EXP26.train_one_restart(
                            noisy_data=(
                                noisy_data
                            ),
                            random_seed=(
                                seed
                            ),
                        )
                    )


                    runtime = (
                        time.time()
                        - start
                    )

                    total_runtime += (
                        runtime
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
                        select_mode_order_beam(
                            wavenumbers=(
                                all_k
                            ),
                            activities=(
                                activities
                            ),
                            noisy_data=(
                                noisy_data
                            ),
                            physical_roots=(
                                physical_roots
                            ),
                            number_of_modes=(
                                K
                            ),
                            training_k_min=(
                                training_k_min
                            ),
                            training_k_max=(
                                EXP26.K_MAX
                            ),
                        )
                    )


                    if (
                        selection[
                            "success"
                        ]
                    ):

                        raw_k = (
                            selection[
                                "selected"
                            ]
                        )


                        raw_metrics = (
                            calculate_k_metrics(
                                raw_k,
                                true_k,
                            )
                        )


                        refinement = (
                            safe_refine(
                                estimated=(
                                    raw_k
                                ),
                                physical_roots=(
                                    target_roots
                                ),
                                noisy_data=(
                                    noisy_data
                                ),
                            )
                        )


                        refined_k = (
                            refinement[
                                "refined"
                            ]
                        )


                        refined_metrics = (
                            calculate_k_metrics(
                                refined_k,
                                true_k,
                            )
                        )


                        restart_score_info = (
                            EXP26.calculate_restart_score(
                                selected_wavenumbers=(
                                    raw_k
                                ),
                                all_activities=(
                                    activities
                                ),
                                noisy_data=(
                                    noisy_data
                                ),
                            )
                        )


                        restart_score = float(
                            restart_score_info[
                                "score"
                            ]
                        )


                    else:

                        raw_k = np.asarray(
                            [],
                            dtype=np.float64,
                        )

                        refined_k = (
                            raw_k
                        )

                        raw_metrics = {
                            "success": False,
                            "rmse": float("inf"),
                            "max_error": float("inf"),
                        }

                        refined_metrics = (
                            raw_metrics.copy()
                        )

                        refinement = {
                            "accepted": False,
                        }

                        restart_score = (
                            float("inf")
                        )


                    result = {

                        "selector_success": (
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

                        "refined_k": (
                            refined_k
                        ),

                        "raw_success": (
                            raw_metrics[
                                "success"
                            ]
                        ),

                        "refined_success": (
                            refined_metrics[
                                "success"
                            ]
                        ),

                        "raw_rmse": (
                            raw_metrics[
                                "rmse"
                            ]
                        ),

                        "refined_rmse": (
                            refined_metrics[
                                "rmse"
                            ]
                        ),

                        "restart_score": (
                            restart_score
                        ),

                        "runtime": (
                            runtime
                        ),
                    }


                    restart_results.append(
                        result
                    )


                    restart_rows.append(
                        {
                            "K": K,

                            "snr_db": (
                                snr_db
                            ),

                            "trial": (
                                trial
                            ),

                            "restart": (
                                restart_index
                                + 1
                            ),

                            "seed": (
                                seed
                            ),

                            "selector_success": int(
                                result[
                                    "selector_success"
                                ]
                            ),

                            "reason": (
                                result[
                                    "reason"
                                ]
                            ),

                            "raw_success": int(
                                result[
                                    "raw_success"
                                ]
                            ),

                            "refined_success": int(
                                result[
                                    "refined_success"
                                ]
                            ),

                            "raw_rmse": (
                                result[
                                    "raw_rmse"
                                ]
                            ),

                            "refined_rmse": (
                                result[
                                    "refined_rmse"
                                ]
                            ),

                            "runtime_seconds": (
                                runtime
                            ),

                            "raw_k": (
                                array_to_text(
                                    raw_k
                                )
                            ),

                            "refined_k": (
                                array_to_text(
                                    refined_k
                                )
                            ),
                        }
                    )


                # ------------------------------------------------------
                # trial-level
                # ------------------------------------------------------
                best_index = (
                    choose_best_restart(
                        restart_results
                    )
                )


                any_raw_success = bool(
                    any(
                        item[
                            "raw_success"
                        ]
                        for item
                        in restart_results
                    )
                )


                restart_success_fraction = float(
                    np.mean(
                        [
                            float(
                                item[
                                    "raw_success"
                                ]
                            )
                            for item
                            in restart_results
                        ]
                    )
                )


                if best_index is None:

                    chosen_raw_success = False

                    chosen_refined_success = False

                    chosen_raw_rmse = (
                        float("inf")
                    )

                    chosen_refined_rmse = (
                        float("inf")
                    )

                    chosen_raw_k = []

                    chosen_refined_k = []


                else:

                    chosen = (
                        restart_results[
                            best_index
                        ]
                    )


                    chosen_raw_success = (
                        chosen[
                            "raw_success"
                        ]
                    )

                    chosen_refined_success = (
                        chosen[
                            "refined_success"
                        ]
                    )

                    chosen_raw_rmse = (
                        chosen[
                            "raw_rmse"
                        ]
                    )

                    chosen_refined_rmse = (
                        chosen[
                            "refined_rmse"
                        ]
                    )

                    chosen_raw_k = (
                        chosen[
                            "raw_k"
                        ]
                    )

                    chosen_refined_k = (
                        chosen[
                            "refined_k"
                        ]
                    )


                print(
                    f"    raw success     = "
                    f"{chosen_raw_success}"
                )

                print(
                    f"    refined success = "
                    f"{chosen_refined_success}"
                )

                print(
                    f"    raw RMSE        = "
                    f"{chosen_raw_rmse:.3e}"
                )

                print(
                    f"    raw k           = "
                    f"{np.asarray(chosen_raw_k)}"
                )


                trial_rows.append(
                    {
                        "K": K,

                        "snr_db": (
                            snr_db
                        ),

                        "trial": (
                            trial
                        ),

                        "raw_success": int(
                            chosen_raw_success
                        ),

                        "refined_success": int(
                            chosen_refined_success
                        ),

                        "any_raw_success": int(
                            any_raw_success
                        ),

                        "restart_success_fraction": (
                            restart_success_fraction
                        ),

                        "raw_rmse": (
                            chosen_raw_rmse
                        ),

                        "refined_rmse": (
                            chosen_refined_rmse
                        ),

                        "runtime_seconds": (
                            total_runtime
                        ),

                        "raw_k": (
                            array_to_text(
                                chosen_raw_k
                            )
                        ),

                        "refined_k": (
                            array_to_text(
                                chosen_refined_k
                            )
                        ),
                    }
                )


                # checkpoint
                save_csv(
                    RESTART_CSV,
                    restart_rows,
                )

                save_csv(
                    TRIAL_CSV,
                    trial_rows,
                )

                save_csv(
                    SUMMARY_CSV,
                    summarize_trials(
                        trial_rows
                    ),
                )


    # ==================================================================
    # Final
    # ==================================================================
    summary_rows = (
        summarize_trials(
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
        "MULTI-MODE SUMMARY"
    )

    print(
        "=" * 180
    )


    for row in summary_rows:

        print(
            f"K={row['K']} | "
            f"SNR={row['snr_db']:>5.1f} | "
            f"raw="
            f"{100*row['raw_success_rate']:6.1f}% | "
            f"ref="
            f"{100*row['refined_success_rate']:6.1f}% | "
            f"any="
            f"{100*row['any_raw_success_rate']:6.1f}% | "
            f"restart="
            f"{100*row['restart_success_fraction_mean']:6.1f}% | "
            f"rawRMSE="
            f"{row['raw_rmse_median']:.3e} | "
            f"refRMSE="
            f"{row['refined_rmse_median']:.3e} | "
            f"time="
            f"{row['runtime_seconds_mean']:.1f}s"
        )


    print(
        "\n输出："
    )

    print(
        SUMMARY_CSV
    )

    print(
        TRIAL_CSV
    )

    print(
        RESTART_CSV
    )

    print(
        MODE_SHAPE_CSV
    )

    print(
        FIGURE_DIR
    )


if __name__ == "__main__":
    main()
