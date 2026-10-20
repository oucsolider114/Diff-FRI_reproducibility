# -*- coding: utf-8 -*-
"""
Low-SNR Robustness Final
======================================================================

用途
----------------------------------------------------------------------
把实验27~30发现的低SNR问题集中到一个最终脚本中处理。

已经确认的问题链：

1. 低SNR下会出现合法高阶模态竞争；
2. 搜索下界 K_MIN=1.20 附近的候选与搜索区外 k5 物理根有关；
3. 5 dB 下 k1/k2/k3 正确候选实际上已经形成；
4. 主要失败机制是：
       全局候选竞争允许高阶物理候选跨模态阶次占位。

本脚本最终比较：

    OOB_ROOT_ORDER
    ORDER_CELL_JOINT

最终方法：
    ORDER_CELL_JOINT

核心思想：
----------------------------------------------------------------------
不使用 k1±固定半径、k2±固定半径、k3±固定半径的强硬槽位。

只根据相邻Robin物理根的中点划分：

    Mode-1 Cell
    Mode-2 Cell
    Mode-3 Cell

例如：

    r1 > r2 > r3 > r4

定义：

    b12 = (r1+r2)/2
    b23 = (r2+r3)/2
    b34 = (r3+r4)/2

则：

    C1 : k > b12
    C2 : b23 < k <= b12
    C3 : b34 < k <= b23

每个cell内部首先根据：

    activity + Robin physics score

保留若干个候选。

随后对三个cell的候选进行联合组合搜索，
对每个组合重新最小二乘估计复系数C，并计算：

        ||Y - Psi(k) C_hat||_F / ||Y||_F

最终选择数据重构残差最低的组合。

因此：

    物理先验决定“阶次区域”
    数据决定“区域中的具体波数”

避免直接把精确Robin根作为答案。


实验模式
----------------------------------------------------------------------
QUICK_TEST = True

    SNR = [5, 10, 15] dB
    2 trials / SNR
    2 restarts / trial

主要用于验证代码和机制。

QUICK通过以后：

QUICK_TEST = False

    SNR = [-5,-3,0,5,10,15,20,30] dB
    10 trials / SNR
    5 restarts / trial


依赖
----------------------------------------------------------------------
只依赖：

    experiments/26_realistic_bottom_mode_frinet.py

不依赖实验27、28、29、30。

因此本脚本FULL验证完成以后，
实验27~30可以移动到archive目录。


重要研究口径
----------------------------------------------------------------------
1. 当前仍是K=3。
2. 当前观测数据仍使用实验26的可微FD模态矩阵生成。
3. 真值波数来自KRAKEN。
4. 后续仍需要KRAKEN .MOD独立数据验证。
5. ORDER_CELL_JOINT利用“目标为前三阶模态”的阶次先验。
6. 它不是完全未知模态数/未知阶次情况下的通用盲估计器。
"""

from __future__ import annotations

import csv
import importlib.util
import itertools
import math
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

EXPERIMENT26_PATH = (
    Path(__file__).resolve().parent
    / "26_realistic_bottom_mode_frinet.py"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "low_snr_robustness_final"
)

FIGURE_DIR = (
    RESULT_DIR
    / "figures"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ======================================================================
# 动态加载实验26
# ======================================================================
def load_experiment26():

    if not EXPERIMENT26_PATH.exists():

        raise FileNotFoundError(
            "\n没有找到实验26：\n"
            f"{EXPERIMENT26_PATH}\n"
        )

    module_name = (
        "experiment26_for_low_snr_final"
    )

    spec = (
        importlib.util.spec_from_file_location(
            module_name,
            EXPERIMENT26_PATH,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):

        raise RuntimeError(
            "实验26模块加载失败。"
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


EXP26 = load_experiment26()


# ======================================================================
# QUICK / FULL
# ======================================================================
QUICK_TEST = False


if QUICK_TEST:

    SNR_VALUES_DB = [
        5.0,
        10.0,
        15.0,
    ]

    NUMBER_OF_TRIALS = 2

    NUMBER_OF_RESTARTS = 2

    RUN_SUFFIX = "quick"

else:

    SNR_VALUES_DB = [
        -5.0,
        -3.0,
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
# 当前最终模型配置
#
# 保持实验26/27/28的主配置：
#     K=3
#     L=20
#     M=24
#
# 不在本实验继续扫描L/M/R。
# ======================================================================
NUMBER_OF_SNAPSHOTS = 20

NUMBER_OF_ATOMS = 24

NUMBER_OF_TARGET_MODES = (
    EXP26.NUMBER_OF_TRUE_MODES
)


# ======================================================================
# 随机种子
# ======================================================================
BASE_RANDOM_SEED = 20260831


# ======================================================================
# 搜索范围
# ======================================================================
TRAINING_K_MIN = float(
    EXP26.K_MIN
)

TRAINING_K_MAX = float(
    EXP26.K_MAX
)


# ======================================================================
# 扩展物理根搜索范围
#
# 必须保持：
#
#     PHYSICS_K_MIN > omega / c_bottom
#
# 因为当前实数Robin公式针对trapped mode。
#
# 之前实验中1.08已经验证可以覆盖k5~k8。
# ======================================================================
PHYSICS_K_MIN = 1.08


# ======================================================================
# OOB selector参数
#
# 延续已经验证过的机制参数，
# 不针对最终结果重新调。
# ======================================================================
ROOT_CAPTURE_RADIUS = float(
    EXP26.ROOT_CAPTURE_RADIUS
)

ROOT_ORDER_PENALTY = 0.25

MINIMUM_SELECTION_SEPARATION = 0.004


# ======================================================================
# ORDER_CELL_JOINT参数
# ======================================================================

# 每个阶次区域最多保留多少个高分候选参加联合组合搜索。
#
# K=3时若每格4个：
#
#     最大组合数 = 4^3 = 64
#
# 很小。
TOP_CANDIDATES_PER_CELL = 4


# 为避免两个几乎重复的训练原子都进入同一个cell候选池，
# cell内部先做一次轻量NMS。
CELL_INTERNAL_MIN_SEPARATION = 0.0010


# ======================================================================
# Safe Refinement
#
# 只在估计已经进入正确物理根邻域时才允许精修。
#
# 它不负责修复错误阶次。
# ======================================================================
REFINEMENT_CAPTURE_RADIUS = 0.006

MAXIMUM_FIT_DEGRADATION = 1.10


# ======================================================================
# 成功阈值
# ======================================================================
SUCCESS_TOLERANCE = 1.5e-3


# ======================================================================
# 方法
# ======================================================================
METHODS = [
    "OOB_ROOT_ORDER",
    "ORDER_CELL_JOINT",
]


# ======================================================================
# 输出文件
# ======================================================================
RESTART_CSV = (
    RESULT_DIR
    / (
        "low_snr_restart_results_"
        f"{RUN_SUFFIX}.csv"
    )
)

TRIAL_CSV = (
    RESULT_DIR
    / (
        "low_snr_trial_results_"
        f"{RUN_SUFFIX}.csv"
    )
)

SUMMARY_CSV = (
    RESULT_DIR
    / (
        "low_snr_summary_"
        f"{RUN_SUFFIX}.csv"
    )
)


# ======================================================================
# 工具函数
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
    values: np.ndarray,
) -> str:

    values = np.asarray(
        values
    ).ravel()

    return ";".join(
        f"{float(value):.12f}"
        for value
        in values
    )


# ======================================================================
# 实验26配置
# ======================================================================
def configure_exp26():

    EXP26.NUMBER_OF_SNAPSHOTS = (
        NUMBER_OF_SNAPSHOTS
    )

    EXP26.NUMBER_OF_ATOMS = (
        NUMBER_OF_ATOMS
    )


# ======================================================================
# 扩展Robin物理根
#
# 暂时扩大实验26的K_MIN，
# 计算结束后恢复。
# ======================================================================
# ======================================================================
# 扩展Robin物理根
#
# 训练搜索区：
#     K_MIN = 1.20
#
# 但为了识别训练搜索区之外的高阶物理模态，
# 物理根目录额外向下扩展到：
#
#     PHYSICS_K_MIN = 1.08
#
# 这里不会改变FRINET训练范围，
# 只扩大selector所使用的物理根知识范围。
# ======================================================================
def calculate_extended_physical_roots():

    print(
        "\n计算扩展 Robin 物理根..."
    )

    print(
        f"  physics k range = "
        f"[{PHYSICS_K_MIN:.6f}, "
        f"{TRAINING_K_MAX:.6f}]"
    )


    # ------------------------------------------------------------------
    # dz = 0.5
    # ------------------------------------------------------------------
    roots_dz05 = np.asarray(
        EXP26.find_robin_roots(
            k_min=PHYSICS_K_MIN,
            k_max=TRAINING_K_MAX,
            dz=0.5,
        ),
        dtype=np.float64,
    )


    # ------------------------------------------------------------------
    # dz = 0.25
    # ------------------------------------------------------------------
    roots_dz025 = np.asarray(
        EXP26.find_robin_roots(
            k_min=PHYSICS_K_MIN,
            k_max=TRAINING_K_MAX,
            dz=0.25,
        ),
        dtype=np.float64,
    )


    # ------------------------------------------------------------------
    # 从大到小排序：
    #
    # k1 > k2 > k3 > ...
    # ------------------------------------------------------------------
    roots_dz05 = np.sort(
        roots_dz05
    )[::-1]

    roots_dz025 = np.sort(
        roots_dz025
    )[::-1]


    print(
        "\ndz=0.5 extended roots:"
    )

    print(
        roots_dz05
    )


    print(
        "\ndz=0.25 extended roots:"
    )

    print(
        roots_dz025
    )


    # ------------------------------------------------------------------
    # 两个网格必须至少有：
    #
    # k1, k2, k3, k4, k5
    #
    # 否则不能诊断搜索区外k5机制。
    # ------------------------------------------------------------------
    number_of_roots = min(
        roots_dz05.size,
        roots_dz025.size,
    )


    if number_of_roots < 5:

        raise RuntimeError(
            "\n扩展Robin根不足5个。\n"
            f"dz=0.5 roots = {roots_dz05.size}\n"
            f"dz=0.25 roots = {roots_dz025.size}\n"
            "\n当前物理根搜索范围没有覆盖到k5，"
            "不能继续LOW-SNR最终实验。"
        )


    # ------------------------------------------------------------------
    # 为避免两个网格在极高阶处根数略有不同，
    # 只对共同存在的前N阶做Richardson外推。
    # ------------------------------------------------------------------
    roots_dz05 = (
        roots_dz05[
            :number_of_roots
        ]
    )

    roots_dz025 = (
        roots_dz025[
            :number_of_roots
        ]
    )


    # ------------------------------------------------------------------
    # 二阶有限差分 Richardson 外推
    #
    # e(h) = C h^2
    #
    # h_coarse = 0.5
    # h_fine   = 0.25
    #
    # k_R =
    # k_fine + (k_fine-k_coarse)/(2^2-1)
    # ------------------------------------------------------------------
    richardson_roots = (
        roots_dz025
        +
        (
            roots_dz025
            - roots_dz05
        )
        / 3.0
    )


    richardson_roots = np.sort(
        richardson_roots
    )[::-1]


    print(
        "\nExtended Richardson roots:"
    )

    for index, value in enumerate(
        richardson_roots,
        start=1,
    ):

        if index <= NUMBER_OF_TARGET_MODES:

            mode_type = "TARGET"

        else:

            mode_type = "HIGHER"


        if value >= TRAINING_K_MIN:

            range_type = "IN-BAND"

        else:

            range_type = "OUT-OF-BAND"


        print(
            f"  k{index:02d} = "
            f"{value:.12f} | "
            f"{mode_type:6s} | "
            f"{range_type}"
        )


    # ------------------------------------------------------------------
    # 关键机制检查：
    #
    # 我们之前已经发现：
    #
    # k5 ≈ 1.19956
    # K_MIN = 1.20
    #
    # 二者仅相差约4e-4。
    #
    # 因此训练原子被物理约束吸引向k5时，
    # 会在搜索下界1.20附近形成clipped candidate。
    # ------------------------------------------------------------------
    k5 = float(
        richardson_roots[4]
    )

    k5_boundary_distance = abs(
        TRAINING_K_MIN
        - k5
    )


    print(
        "\nOut-of-band k5 mechanism check:"
    )

    print(
        f"  TRAINING_K_MIN = "
        f"{TRAINING_K_MIN:.12f}"
    )

    print(
        f"  k5             = "
        f"{k5:.12f}"
    )

    print(
        f"  |K_MIN-k5|     = "
        f"{k5_boundary_distance:.6e}"
    )

    print(
        f"  capture radius = "
        f"{ROOT_CAPTURE_RADIUS:.6e}"
    )


    if (
        k5
        >= TRAINING_K_MIN
    ):

        print(
            "\nWARNING："
            "k5并没有位于训练搜索区之外。"
        )

    elif (
        k5_boundary_distance
        <= ROOT_CAPTURE_RADIUS
    ):

        print(
            "\nPASS："
            "k5位于训练区外，且足够接近K_MIN，"
            "能够解释lower-bound clipped candidate。"
        )

    else:

        print(
            "\nWARNING："
            "k5虽然位于训练区外，"
            "但与K_MIN距离超过ROOT_CAPTURE_RADIUS。"
        )


    return (
        richardson_roots
    )


# ======================================================================
# 数据生成
#
# 固定同一个trial的clean signal和noise direction。
#
# 不同SNR只改变noise scale，
# 因此不同SNR之间是paired comparison。
# ======================================================================
def generate_trial_master_data(
    true_wavenumbers: np.ndarray,
    coefficient_seed: int,
    noise_seed: int,
):

    mode_matrix = np.asarray(
        EXP26.build_mode_matrix(
            true_wavenumbers
        ),
        dtype=np.complex128,
    )


    coefficient_rng = (
        np.random.default_rng(
            coefficient_seed
        )
    )

    coefficients = (
        coefficient_rng.standard_normal(
            (
                NUMBER_OF_TARGET_MODES,
                NUMBER_OF_SNAPSHOTS,
            )
        )
        +
        1j
        * coefficient_rng.standard_normal(
            (
                NUMBER_OF_TARGET_MODES,
                NUMBER_OF_SNAPSHOTS,
            )
        )
    ) / np.sqrt(2.0)


    clean_data = (
        mode_matrix
        @ coefficients
    )


    noise_rng = (
        np.random.default_rng(
            noise_seed
        )
    )

    raw_noise = (
        noise_rng.standard_normal(
            clean_data.shape
        )
        +
        1j
        * noise_rng.standard_normal(
            clean_data.shape
        )
    ) / np.sqrt(2.0)


    raw_noise = (
        raw_noise
        /
        max(
            np.linalg.norm(
                raw_noise
            ),
            1.0e-14,
        )
    )


    return {
        "clean_data": clean_data,
        "unit_noise": raw_noise,
    }


def build_noisy_data(
    master: dict,
    snr_db: float,
):

    clean_data = np.asarray(
        master[
            "clean_data"
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
            clean_data
        )
    )


    target_noise_norm = (
        clean_norm
        /
        (
            10.0
            ** (
                snr_db
                / 20.0
            )
        )
    )


    noise = (
        unit_noise
        * target_noise_norm
    )


    noisy_data = (
        clean_data
        + noise
    )


    achieved_snr = (
        20.0
        * np.log10(
            clean_norm
            /
            max(
                float(
                    np.linalg.norm(
                        noise
                    )
                ),
                1.0e-14,
            )
        )
    )


    return {
        "clean_data": (
            clean_data
        ),

        "noisy_data": (
            noisy_data
        ),

        "achieved_snr_db": float(
            achieved_snr
        ),
    }


# ======================================================================
# 数据重构残差
# ======================================================================
def calculate_data_fit(
    wavenumbers: np.ndarray,
    noisy_data: np.ndarray,
):

    wavenumbers = np.asarray(
        wavenumbers,
        dtype=np.float64,
    )


    mode_matrix = np.asarray(
        EXP26.build_mode_matrix(
            wavenumbers
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


    return {
        "residual": (
            residual
        ),

        "coefficients": (
            coefficients
        ),

        "reconstruction": (
            reconstruction
        ),
    }


# ======================================================================
# 简单score NMS
# ======================================================================
def scored_nms(
    wavenumbers: np.ndarray,
    scores: np.ndarray,
    number_to_select: int,
    minimum_separation: float,
):

    wavenumbers = np.asarray(
        wavenumbers,
        dtype=np.float64,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )


    order = np.argsort(
        scores
    )[::-1]


    selected_indices = []


    for index in order:

        index = int(
            index
        )

        k = float(
            wavenumbers[
                index
            ]
        )


        if all(
            abs(
                k
                - float(
                    wavenumbers[
                        previous_index
                    ]
                )
            )
            >= minimum_separation
            for previous_index
            in selected_indices
        ):

            selected_indices.append(
                index
            )


        if len(
            selected_indices
        ) >= number_to_select:

            break


    # ==============================================================
    # 极端情况下NMS数量不足：
    # 按score补齐，但仍不重复index。
    # ==============================================================
    if len(
        selected_indices
    ) < number_to_select:

        for index in order:

            index = int(
                index
            )

            if (
                index
                not in selected_indices
            ):

                selected_indices.append(
                    index
                )

            if len(
                selected_indices
            ) >= number_to_select:

                break


    return np.asarray(
        selected_indices[
            :number_to_select
        ],
        dtype=np.int64,
    )


# ======================================================================
# OOB ROOT ORDER
#
# 作为最终方法的直接旧基线。
# ======================================================================
def select_oob_root_order(
    wavenumbers: np.ndarray,
    activities: np.ndarray,
    extended_roots: np.ndarray,
):

    wavenumbers = np.asarray(
        wavenumbers,
        dtype=np.float64,
    )

    activities = np.asarray(
        activities,
        dtype=np.float64,
    )


    physics_scores, _ = (
        EXP26.calculate_physics_scores(
            wavenumbers=wavenumbers,
            activities=activities,
        )
    )


    corrected_scores = np.asarray(
        physics_scores,
        dtype=np.float64,
    ).copy()


    for atom_index, k in enumerate(
        wavenumbers
    ):

        distances = np.abs(
            extended_roots
            - k
        )

        nearest_root_index = int(
            np.argmin(
                distances
            )
        )

        nearest_distance = float(
            distances[
                nearest_root_index
            ]
        )


        # root index:
        #
        # 0 -> k1
        # 1 -> k2
        # 2 -> k3
        # 3 -> k4
        # 4 -> k5
        #
        # k4及以后都属于高阶非目标模态。
        if (
            nearest_root_index
            >= NUMBER_OF_TARGET_MODES
            and
            nearest_distance
            <= ROOT_CAPTURE_RADIUS
        ):

            corrected_scores[
                atom_index
            ] *= ROOT_ORDER_PENALTY


    selected_indices = (
        scored_nms(
            wavenumbers=wavenumbers,
            scores=corrected_scores,
            number_to_select=(
                NUMBER_OF_TARGET_MODES
            ),
            minimum_separation=(
                MINIMUM_SELECTION_SEPARATION
            ),
        )
    )


    return np.asarray(
        wavenumbers[
            selected_indices
        ],
        dtype=np.float64,
    )


# ======================================================================
# ORDER CELL边界
# ======================================================================
def calculate_order_cells(
    extended_roots: np.ndarray,
):

    roots = np.sort(
        np.asarray(
            extended_roots,
            dtype=np.float64,
        )
    )[::-1]


    if roots.size < (
        NUMBER_OF_TARGET_MODES
        + 1
    ):

        raise RuntimeError(
            "物理根数量不足，无法建立阶次cell。"
        )


    target_plus_one = roots[
        :NUMBER_OF_TARGET_MODES + 1
    ]


    boundaries = []


    for index in range(
        NUMBER_OF_TARGET_MODES
    ):

        boundary = (
            target_plus_one[
                index
            ]
            +
            target_plus_one[
                index + 1
            ]
        ) / 2.0

        boundaries.append(
            float(
                boundary
            )
        )


    # 对K=3：
    #
    # boundaries[0] = b12
    # boundaries[1] = b23
    # boundaries[2] = b34
    #
    # cell 1:
    #       b12 < k <= K_MAX
    #
    # cell 2:
    #       b23 < k <= b12
    #
    # cell 3:
    #       b34 < k <= b23
    cells = []


    for mode_index in range(
        NUMBER_OF_TARGET_MODES
    ):

        if mode_index == 0:

            upper = (
                TRAINING_K_MAX
                + 1.0e-12
            )

            lower = (
                boundaries[0]
            )

        else:

            upper = (
                boundaries[
                    mode_index - 1
                ]
            )

            lower = (
                boundaries[
                    mode_index
                ]
            )


        cells.append(
            {
                "mode_index": (
                    mode_index
                ),

                "lower": float(
                    lower
                ),

                "upper": float(
                    upper
                ),
            }
        )


    return {
        "roots": (
            roots
        ),

        "boundaries": (
            boundaries
        ),

        "cells": (
            cells
        ),
    }


# ======================================================================
# cell内部候选筛选
# ======================================================================
def get_cell_candidates(
    wavenumbers: np.ndarray,
    physics_scores: np.ndarray,
    lower: float,
    upper: float,
):

    indices = np.where(
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


    if indices.size == 0:

        return np.asarray(
            [],
            dtype=np.int64,
        )


    local_order = indices[
        np.argsort(
            physics_scores[
                indices
            ]
        )[::-1]
    ]


    kept = []


    for index in local_order:

        index = int(
            index
        )

        k = float(
            wavenumbers[
                index
            ]
        )


        if all(
            abs(
                k
                - float(
                    wavenumbers[
                        previous_index
                    ]
                )
            )
            >= CELL_INTERNAL_MIN_SEPARATION
            for previous_index
            in kept
        ):

            kept.append(
                index
            )


        if len(
            kept
        ) >= TOP_CANDIDATES_PER_CELL:

            break


    return np.asarray(
        kept,
        dtype=np.int64,
    )


# ======================================================================
# ORDER_CELL_JOINT最终selector
# ======================================================================
def select_order_cell_joint(
    wavenumbers: np.ndarray,
    activities: np.ndarray,
    extended_roots: np.ndarray,
    noisy_data: np.ndarray,
):

    wavenumbers = np.asarray(
        wavenumbers,
        dtype=np.float64,
    )

    activities = np.asarray(
        activities,
        dtype=np.float64,
    )


    physics_scores, physics_residuals = (
        EXP26.calculate_physics_scores(
            wavenumbers=wavenumbers,
            activities=activities,
        )
    )


    cell_info = (
        calculate_order_cells(
            extended_roots
        )
    )


    cell_candidate_indices = []


    for cell in (
        cell_info[
            "cells"
        ]
    ):

        indices = (
            get_cell_candidates(
                wavenumbers=wavenumbers,
                physics_scores=physics_scores,
                lower=cell[
                    "lower"
                ],
                upper=cell[
                    "upper"
                ],
            )
        )

        cell_candidate_indices.append(
            indices
        )


    # ==============================================================
    # 如果任何目标阶次区域没有候选：
    # selector失败。
    #
    # 不进行强制补齐，
    # 因为这本身就是重要失败信息。
    # ==============================================================
    if any(
        indices.size == 0
        for indices
        in cell_candidate_indices
    ):

        return {
            "success": False,

            "reason": (
                "EMPTY_ORDER_CELL"
            ),

            "selected": np.asarray(
                [],
                dtype=np.float64,
            ),

            "joint_fit": (
                float("inf")
            ),

            "cell_candidate_counts": [
                int(
                    indices.size
                )
                for indices
                in cell_candidate_indices
            ],

            "cell_info": (
                cell_info
            ),
        }


    # ==============================================================
    # 穷举每个cell的候选组合。
    #
    # 每个cell最多4个，K=3时最多64组。
    #
    # 主目标：
    #       最小化观测数据重构残差
    #
    # tie-break：
    #       physics score总和越大越好
    #
    # 不引入额外可调权重。
    # ==============================================================
    best = None


    candidate_lists = [
        list(
            map(
                int,
                indices
            )
        )
        for indices
        in cell_candidate_indices
    ]


    for combination in itertools.product(
        *candidate_lists
    ):

        # 每个cell不重叠，
        # 正常情况下不可能重复原子。
        if len(
            set(
                combination
            )
        ) < NUMBER_OF_TARGET_MODES:

            continue


        selected_k = np.asarray(
            [
                wavenumbers[
                    index
                ]
                for index
                in combination
            ],
            dtype=np.float64,
        )


        fit_info = (
            calculate_data_fit(
                wavenumbers=selected_k,
                noisy_data=noisy_data,
            )
        )


        fit_residual = float(
            fit_info[
                "residual"
            ]
        )


        total_physics_score = float(
            np.sum(
                [
                    physics_scores[
                        index
                    ]
                    for index
                    in combination
                ]
            )
        )


        # lexicographic：
        #
        # 1. data fit越小越好
        # 2. physics score越大越好
        ranking_key = (
            fit_residual,
            -total_physics_score,
        )


        if (
            best is None
            or
            ranking_key
            <
            best[
                "ranking_key"
            ]
        ):

            best = {
                "ranking_key": (
                    ranking_key
                ),

                "selected": (
                    selected_k
                ),

                "indices": (
                    combination
                ),

                "joint_fit": (
                    fit_residual
                ),

                "total_physics_score": (
                    total_physics_score
                ),
            }


    if best is None:

        return {
            "success": False,

            "reason": (
                "NO_VALID_COMBINATION"
            ),

            "selected": np.asarray(
                [],
                dtype=np.float64,
            ),

            "joint_fit": (
                float("inf")
            ),

            "cell_candidate_counts": [
                int(
                    indices.size
                )
                for indices
                in cell_candidate_indices
            ],

            "cell_info": (
                cell_info
            ),
        }


    return {
        "success": True,

        "reason": (
            "ORDER_CELL_JOINT_SELECTED"
        ),

        "selected": np.asarray(
            best[
                "selected"
            ],
            dtype=np.float64,
        ),

        "joint_fit": float(
            best[
                "joint_fit"
            ]
        ),

        "cell_candidate_counts": [
            int(
                indices.size
            )
            for indices
            in cell_candidate_indices
        ],

        "cell_info": (
            cell_info
        ),

        "physics_residuals": (
            physics_residuals
        ),
    }


# ======================================================================
# k误差评价
# ======================================================================
def calculate_k_metrics(
    estimated_wavenumbers: np.ndarray,
    true_wavenumbers: np.ndarray,
):

    estimated = np.sort(
        np.asarray(
            estimated_wavenumbers,
            dtype=np.float64,
        )
    )[::-1]

    truth = np.sort(
        np.asarray(
            true_wavenumbers,
            dtype=np.float64,
        )
    )[::-1]


    if (
        estimated.size
        != truth.size
    ):

        return {
            "success": False,

            "k_rmse": (
                float("inf")
            ),

            "maximum_error": (
                float("inf")
            ),

            "errors": np.full(
                truth.size,
                np.inf,
            ),
        }


    errors = np.abs(
        estimated
        - truth
    )


    k_rmse = float(
        np.sqrt(
            np.mean(
                errors ** 2
            )
        )
    )


    maximum_error = float(
        np.max(
            errors
        )
    )


    success = bool(
        maximum_error
        <= SUCCESS_TOLERANCE
    )


    return {
        "success": (
            success
        ),

        "k_rmse": (
            k_rmse
        ),

        "maximum_error": (
            maximum_error
        ),

        "errors": (
            errors
        ),
    }


# ======================================================================
# Safe Refinement
# ======================================================================
def safe_refine(
    selected_wavenumbers: np.ndarray,
    target_roots: np.ndarray,
    noisy_data: np.ndarray,
):

    raw = np.sort(
        np.asarray(
            selected_wavenumbers,
            dtype=np.float64,
        )
    )[::-1]


    target_roots = np.sort(
        np.asarray(
            target_roots,
            dtype=np.float64,
        )
    )[::-1]


    if raw.size != NUMBER_OF_TARGET_MODES:

        return {
            "accepted": False,

            "reason": (
                "WRONG_NUMBER_OF_MODES"
            ),

            "refined": (
                raw
            ),

            "raw_fit": (
                float("inf")
            ),

            "refined_fit": (
                float("inf")
            ),
        }


    refined = raw.copy()


    # ==============================================================
    # 因为已经按降序对应前三阶，
    # 每个估计只允许向对应阶次Robin根靠拢。
    #
    # 不允许k3跳到k4。
    # ==============================================================
    for index in range(
        NUMBER_OF_TARGET_MODES
    ):

        distance = abs(
            raw[
                index
            ]
            -
            target_roots[
                index
            ]
        )


        if (
            distance
            <= REFINEMENT_CAPTURE_RADIUS
        ):

            refined[
                index
            ] = (
                target_roots[
                    index
                ]
            )


    raw_fit = (
        calculate_data_fit(
            wavenumbers=raw,
            noisy_data=noisy_data,
        )[
            "residual"
        ]
    )


    refined_fit = (
        calculate_data_fit(
            wavenumbers=refined,
            noisy_data=noisy_data,
        )[
            "residual"
        ]
    )


    accepted = bool(
        refined_fit
        <=
        raw_fit
        * MAXIMUM_FIT_DEGRADATION
    )


    if accepted:

        output = (
            refined
        )

        reason = (
            "ACCEPTED"
        )

    else:

        output = (
            raw
        )

        reason = (
            "FIT_DEGRADATION"
        )


    return {
        "accepted": (
            accepted
        ),

        "reason": (
            reason
        ),

        "refined": np.asarray(
            output,
            dtype=np.float64,
        ),

        "raw_fit": float(
            raw_fit
        ),

        "refined_fit": float(
            refined_fit
        ),
    }


# ======================================================================
# 诊断：
# 被选结果是否接近k4/k5
# ======================================================================
def selected_root_diagnostics(
    selected: np.ndarray,
    extended_roots: np.ndarray,
):

    selected = np.asarray(
        selected,
        dtype=np.float64,
    )


    def root_selected(
        root_index: int,
    ):

        if (
            extended_roots.size
            <= root_index
            or
            selected.size == 0
        ):

            return False


        root = float(
            extended_roots[
                root_index
            ]
        )


        return bool(
            np.min(
                np.abs(
                    selected
                    - root
                )
            )
            <= ROOT_CAPTURE_RADIUS
        )


    lower_bound_selected = bool(
        selected.size > 0
        and
        np.min(
            np.abs(
                selected
                - TRAINING_K_MIN
            )
        )
        <= ROOT_CAPTURE_RADIUS
    )


    return {
        "k4_selected": int(
            root_selected(
                3
            )
        ),

        "k5_selected": int(
            root_selected(
                4
            )
        ),

        "lower_bound_selected": int(
            lower_bound_selected
        ),
    }


# ======================================================================
# restart结果
# ======================================================================
def evaluate_restart_method(
    trained: dict,
    method: str,
    extended_roots: np.ndarray,
    target_roots: np.ndarray,
    true_wavenumbers: np.ndarray,
    noisy_data: np.ndarray,
):

    all_k = np.asarray(
        trained[
            "all_wavenumbers"
        ],
        dtype=np.float64,
    )

    all_activities = np.asarray(
        trained[
            "all_activities"
        ],
        dtype=np.float64,
    )


    if method == "OOB_ROOT_ORDER":

        selected = (
            select_oob_root_order(
                wavenumbers=all_k,
                activities=all_activities,
                extended_roots=(
                    extended_roots
                ),
            )
        )

        selector_success = True

        selector_reason = (
            "OOB_SELECTED"
        )

        cell_counts = [
            -1,
            -1,
            -1,
        ]

        joint_fit = (
            calculate_data_fit(
                selected,
                noisy_data,
            )[
                "residual"
            ]
        )


    elif method == "ORDER_CELL_JOINT":

        selection = (
            select_order_cell_joint(
                wavenumbers=all_k,
                activities=all_activities,
                extended_roots=(
                    extended_roots
                ),
                noisy_data=(
                    noisy_data
                ),
            )
        )


        selector_success = bool(
            selection[
                "success"
            ]
        )

        selector_reason = str(
            selection[
                "reason"
            ]
        )

        selected = np.asarray(
            selection[
                "selected"
            ],
            dtype=np.float64,
        )

        cell_counts = list(
            selection[
                "cell_candidate_counts"
            ]
        )

        joint_fit = float(
            selection[
                "joint_fit"
            ]
        )


    else:

        raise ValueError(
            f"未知method：{method}"
        )


    if (
        not selector_success
        or
        selected.size
        != NUMBER_OF_TARGET_MODES
    ):

        return {
            "selector_success": False,

            "selector_reason": (
                selector_reason
            ),

            "selected_raw": (
                selected
            ),

            "selected_refined": (
                selected
            ),

            "raw_success": False,

            "refined_success": False,

            "raw_k_rmse": (
                float("inf")
            ),

            "refined_k_rmse": (
                float("inf")
            ),

            "raw_max_error": (
                float("inf")
            ),

            "refined_max_error": (
                float("inf")
            ),

            "restart_score": (
                float("inf")
            ),

            "joint_fit": (
                joint_fit
            ),

            "safe_refine_accepted": (
                False
            ),

            "cell_counts": (
                cell_counts
            ),

            "k4_selected": 0,

            "k5_selected": 0,

            "lower_bound_selected": 0,
        }


    raw_metrics = (
        calculate_k_metrics(
            estimated_wavenumbers=(
                selected
            ),
            true_wavenumbers=(
                true_wavenumbers
            ),
        )
    )


    refine_info = (
        safe_refine(
            selected_wavenumbers=(
                selected
            ),
            target_roots=(
                target_roots
            ),
            noisy_data=(
                noisy_data
            ),
        )
    )


    refined = np.asarray(
        refine_info[
            "refined"
        ],
        dtype=np.float64,
    )


    refined_metrics = (
        calculate_k_metrics(
            estimated_wavenumbers=(
                refined
            ),
            true_wavenumbers=(
                true_wavenumbers
            ),
        )
    )


    restart_score_info = (
        EXP26.calculate_restart_score(
            selected_wavenumbers=(
                selected
            ),
            all_activities=(
                all_activities
            ),
            noisy_data=(
                noisy_data
            ),
        )
    )


    diagnostics = (
        selected_root_diagnostics(
            selected=(
                selected
            ),
            extended_roots=(
                extended_roots
            ),
        )
    )


    return {
        "selector_success": True,

        "selector_reason": (
            selector_reason
        ),

        "selected_raw": (
            selected
        ),

        "selected_refined": (
            refined
        ),

        "raw_success": bool(
            raw_metrics[
                "success"
            ]
        ),

        "refined_success": bool(
            refined_metrics[
                "success"
            ]
        ),

        "raw_k_rmse": float(
            raw_metrics[
                "k_rmse"
            ]
        ),

        "refined_k_rmse": float(
            refined_metrics[
                "k_rmse"
            ]
        ),

        "raw_max_error": float(
            raw_metrics[
                "maximum_error"
            ]
        ),

        "refined_max_error": float(
            refined_metrics[
                "maximum_error"
            ]
        ),

        "restart_score": float(
            restart_score_info[
                "score"
            ]
        ),

        "joint_fit": float(
            joint_fit
        ),

        "safe_refine_accepted": bool(
            refine_info[
                "accepted"
            ]
        ),

        "cell_counts": (
            cell_counts
        ),

        **diagnostics,
    }


# ======================================================================
# trial中选择最佳restart
#
# 保持沿用实验26 restart score。
#
# 注意：
# restart score本身也会在最终CSV中保留，
# 如果出现anySuccess高但chosenSuccess低，
# 后续才说明restart选择有问题。
# ======================================================================
def choose_best_restart(
    results: list[dict],
):

    valid_indices = [
        index
        for index, result
        in enumerate(
            results
        )
        if bool(
            result[
                "selector_success"
            ]
        )
        and
        np.isfinite(
            float(
                result[
                    "restart_score"
                ]
            )
        )
    ]


    if not valid_indices:

        return None


    best_index = min(
        valid_indices,
        key=lambda index:
        float(
            results[
                index
            ][
                "restart_score"
            ]
        ),
    )


    return int(
        best_index
    )


# ======================================================================
# 汇总
# ======================================================================
def summarize_trials(
    trial_rows: list[dict],
):

    summary_rows = []


    for snr_db in (
        SNR_VALUES_DB
    ):

        for method in (
            METHODS
        ):

            subset = [
                row
                for row
                in trial_rows
                if float(
                    row[
                        "snr_db"
                    ]
                )
                == float(
                    snr_db
                )
                and
                row[
                    "method"
                ]
                == method
            ]


            if not subset:

                continue


            def values(
                key: str,
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
                    ],
                    dtype=np.float64,
                )


            raw_rmse = values(
                "chosen_raw_k_rmse"
            )

            refined_rmse = values(
                "chosen_refined_k_rmse"
            )


            finite_raw = raw_rmse[
                np.isfinite(
                    raw_rmse
                )
            ]

            finite_refined = refined_rmse[
                np.isfinite(
                    refined_rmse
                )
            ]


            summary_rows.append(
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

                    "raw_success_rate": float(
                        np.mean(
                            values(
                                "chosen_raw_success"
                            )
                        )
                    ),

                    "refined_success_rate": float(
                        np.mean(
                            values(
                                "chosen_refined_success"
                            )
                        )
                    ),

                    "any_restart_raw_success_rate": float(
                        np.mean(
                            values(
                                "any_restart_raw_success"
                            )
                        )
                    ),

                    "any_restart_refined_success_rate": float(
                        np.mean(
                            values(
                                "any_restart_refined_success"
                            )
                        )
                    ),

                    "successful_restart_fraction_mean": float(
                        np.mean(
                            values(
                                "successful_restart_fraction"
                            )
                        )
                    ),

                    "k4_selection_rate": float(
                        np.mean(
                            values(
                                "chosen_k4_selected"
                            )
                        )
                    ),

                    "k5_selection_rate": float(
                        np.mean(
                            values(
                                "chosen_k5_selected"
                            )
                        )
                    ),

                    "lower_bound_selection_rate": float(
                        np.mean(
                            values(
                                "chosen_lower_bound_selected"
                            )
                        )
                    ),

                    "raw_k_rmse_median": (
                        float(
                            np.median(
                                finite_raw
                            )
                        )
                        if finite_raw.size > 0
                        else float("nan")
                    ),

                    "refined_k_rmse_median": (
                        float(
                            np.median(
                                finite_refined
                            )
                        )
                        if finite_refined.size > 0
                        else float("nan")
                    ),
                }
            )


    return summary_rows


# ======================================================================
# 绘图
# ======================================================================
def make_figures(
    summary_rows: list[dict],
):

    # ==============================================================
    # Figure 1: Raw success
    # ==============================================================
    fig, ax = plt.subplots(
        figsize=(
            9.0,
            5.5,
        )
    )


    for method in METHODS:

        subset = [
            row
            for row
            in summary_rows
            if row[
                "method"
            ]
            == method
        ]


        subset = sorted(
            subset,
            key=lambda row:
            float(
                row[
                    "snr_db"
                ]
            ),
        )


        x = np.asarray(
            [
                row[
                    "snr_db"
                ]
                for row
                in subset
            ],
            dtype=np.float64,
        )


        y = (
            100.0
            *
            np.asarray(
                [
                    row[
                        "raw_success_rate"
                    ]
                    for row
                    in subset
                ],
                dtype=np.float64,
            )
        )


        ax.plot(
            x,
            y,
            marker="o",
            label=method,
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
        "Low-SNR Raw Recovery"
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
            "low_snr_raw_success_"
            f"{RUN_SUFFIX}.png"
        ),
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


    # ==============================================================
    # Figure 2: refined success
    # ==============================================================
    fig, ax = plt.subplots(
        figsize=(
            9.0,
            5.5,
        )
    )


    for method in METHODS:

        subset = [
            row
            for row
            in summary_rows
            if row[
                "method"
            ]
            == method
        ]


        subset = sorted(
            subset,
            key=lambda row:
            float(
                row[
                    "snr_db"
                ]
            ),
        )


        x = np.asarray(
            [
                row[
                    "snr_db"
                ]
                for row
                in subset
            ],
            dtype=np.float64,
        )


        y = (
            100.0
            *
            np.asarray(
                [
                    row[
                        "refined_success_rate"
                    ]
                    for row
                    in subset
                ],
                dtype=np.float64,
            )
        )


        ax.plot(
            x,
            y,
            marker="o",
            label=method,
        )


    ax.set_ylim(
        -5,
        105,
    )

    ax.set_xlabel(
        "SNR (dB)"
    )

    ax.set_ylabel(
        "Refined success rate (%)"
    )

    ax.set_title(
        "Low-SNR Refined Recovery"
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
            "low_snr_refined_success_"
            f"{RUN_SUFFIX}.png"
        ),
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


    # ==============================================================
    # Figure 3: k5 / lower-bound failure
    # ==============================================================
    fig, ax = plt.subplots(
        figsize=(
            9.0,
            5.5,
        )
    )


    for method in METHODS:

        subset = [
            row
            for row
            in summary_rows
            if row[
                "method"
            ]
            == method
        ]


        subset = sorted(
            subset,
            key=lambda row:
            float(
                row[
                    "snr_db"
                ]
            ),
        )


        x = np.asarray(
            [
                row[
                    "snr_db"
                ]
                for row
                in subset
            ],
            dtype=np.float64,
        )


        y = (
            100.0
            *
            np.asarray(
                [
                    row[
                        "k5_selection_rate"
                    ]
                    for row
                    in subset
                ],
                dtype=np.float64,
            )
        )


        ax.plot(
            x,
            y,
            marker="o",
            label=method,
        )


    ax.set_ylim(
        -5,
        105,
    )

    ax.set_xlabel(
        "SNR (dB)"
    )

    ax.set_ylabel(
        "k5 selection rate (%)"
    )

    ax.set_title(
        "Out-of-Band Higher-Order Selection"
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
            "low_snr_k5_selection_"
            f"{RUN_SUFFIX}.png"
        ),
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ======================================================================
# 主程序
# ======================================================================
def main():

    torch.set_default_dtype(
        torch.float64
    )


    configure_exp26()


    print(
        "=" * 150
    )

    print(
        "LOW-SNR ROBUSTNESS FINAL"
    )

    print(
        "=" * 150
    )

    print(
        f"Mode              = "
        f"{RUN_SUFFIX}"
    )

    print(
        f"SNR values        = "
        f"{SNR_VALUES_DB}"
    )

    print(
        f"Trials / SNR      = "
        f"{NUMBER_OF_TRIALS}"
    )

    print(
        f"Restarts / trial  = "
        f"{NUMBER_OF_RESTARTS}"
    )

    print(
        f"K                 = "
        f"{NUMBER_OF_TARGET_MODES}"
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
        f"Training range    = "
        f"[{TRAINING_K_MIN:.6f}, "
        f"{TRAINING_K_MAX:.6f}]"
    )

    print(
        f"Physics range min = "
        f"{PHYSICS_K_MIN:.6f}"
    )

    print(
        "=" * 150
    )


    # ==================================================================
    # KRAKEN真值
    # ==================================================================
    kraken_all = np.asarray(
        EXP26.load_kraken_reference(),
        dtype=np.float64,
    )


    kraken_all = np.sort(
        kraken_all
    )[::-1]


    true_wavenumbers = (
        kraken_all[
            :NUMBER_OF_TARGET_MODES
        ]
    )


    print(
        "\nKRAKEN target truth:"
    )


    for index, value in enumerate(
        true_wavenumbers,
        start=1,
    ):

        print(
            f"  k{index} = "
            f"{value:.12f}"
        )


    # ==================================================================
    # in-band Robin roots
    # ==================================================================
    inband_roots = np.asarray(
        EXP26.calculate_richardson_roots(),
        dtype=np.float64,
    )


    inband_roots = np.sort(
        inband_roots
    )[::-1]


    target_roots = (
        inband_roots[
            :NUMBER_OF_TARGET_MODES
        ]
    )


    print(
        "\nRobin target roots:"
    )


    for index, value in enumerate(
        target_roots,
        start=1,
    ):

        print(
            f"  r{index} = "
            f"{value:.12f}"
        )


    # ==================================================================
    # extended roots
    # ==================================================================
    extended_roots = (
        calculate_extended_physical_roots()
    )


    print(
        "\nExtended Robin root catalogue:"
    )


    for index, value in enumerate(
        extended_roots,
        start=1,
    ):

        status = (
            "TARGET"
            if index
            <= NUMBER_OF_TARGET_MODES
            else
            "HIGHER"
        )


        if (
            value
            < TRAINING_K_MIN
        ):

            band_status = (
                "OUT-OF-BAND"
            )

        else:

            band_status = (
                "IN-BAND"
            )


        print(
            f"  r{index:02d} = "
            f"{value:.12f} | "
            f"{status:6s} | "
            f"{band_status}"
        )


    if (
        extended_roots.size
        >= 5
    ):

        print(
            "\nBoundary mechanism check:"
        )

        print(
            f"  K_MIN = "
            f"{TRAINING_K_MIN:.12f}"
        )

        print(
            f"  k5    = "
            f"{extended_roots[4]:.12f}"
        )

        print(
            f"  |K_MIN-k5| = "
            f"{abs(TRAINING_K_MIN-extended_roots[4]):.6e}"
        )


    # ==================================================================
    # order cells
    # ==================================================================
    order_cell_info = (
        calculate_order_cells(
            extended_roots
        )
    )


    print(
        "\nMode-order cells:"
    )


    for cell in (
        order_cell_info[
            "cells"
        ]
    ):

        print(
            f"  Mode "
            f"{cell['mode_index'] + 1}: "
            f"({cell['lower']:.9f}, "
            f"{cell['upper']:.9f}]"
        )


    print(
        "\n注意："
    )

    print(
        "ORDER_CELL_JOINT只使用这些阶次区域，"
        "不使用k_r±固定窄半径进行选模。"
    )


    restart_rows = []

    trial_rows = []


    # ==================================================================
    # Trial master data
    #
    # 同一个trial跨SNR共享clean/noise direction。
    # ==================================================================
    trial_master_data = {}


    for trial in range(
        1,
        NUMBER_OF_TRIALS + 1,
    ):

        trial_seed = (
            BASE_RANDOM_SEED
            + trial * 10000
        )


        trial_master_data[
            trial
        ] = (
            generate_trial_master_data(
                true_wavenumbers=(
                    true_wavenumbers
                ),
                coefficient_seed=(
                    trial_seed
                    + 1
                ),
                noise_seed=(
                    trial_seed
                    + 2
                ),
            )
        )


    # ==================================================================
    # SNR sweep
    # ==================================================================
    for snr_db in (
        SNR_VALUES_DB
    ):

        print(
            "\n\n"
            + "=" * 150
        )

        print(
            f"SNR = "
            f"{snr_db:+.1f} dB"
        )

        print(
            "=" * 150
        )


        EXP26.SNR_DB = float(
            snr_db
        )


        for trial in range(
            1,
            NUMBER_OF_TRIALS + 1,
        ):

            print(
                "\n"
                + "#" * 120
            )

            print(
                f"SNR "
                f"{snr_db:+.1f} dB | "
                f"Trial "
                f"{trial}/"
                f"{NUMBER_OF_TRIALS}"
            )

            print(
                "#" * 120
            )


            dataset = (
                build_noisy_data(
                    master=(
                        trial_master_data[
                            trial
                        ]
                    ),
                    snr_db=(
                        snr_db
                    ),
                )
            )


            noisy_data = np.asarray(
                dataset[
                    "noisy_data"
                ],
                dtype=np.complex128,
            )


            achieved_snr = float(
                dataset[
                    "achieved_snr_db"
                ]
            )


            print(
                f"Achieved SNR = "
                f"{achieved_snr:.6f} dB"
            )


            # ==========================================================
            # train restart pool once
            #
            # 所有selector共享相同trained models。
            # ==========================================================
            trained_pool = []


            trial_seed = (
                BASE_RANDOM_SEED
                + trial * 10000
            )


            restart_seed_base = (
                trial_seed
                + 100000
                +
                int(
                    round(
                        (
                            snr_db
                            + 20.0
                        )
                        * 1000.0
                    )
                )
            )


            for restart_index in range(
                NUMBER_OF_RESTARTS
            ):

                restart_seed = (
                    restart_seed_base
                    +
                    restart_index
                    * 1009
                )


                print(
                    f"\n  restart "
                    f"{restart_index + 1}/"
                    f"{NUMBER_OF_RESTARTS} | "
                    f"seed="
                    f"{restart_seed}",
                    flush=True,
                )


                start_time = (
                    time.time()
                )


                trained = (
                    EXP26.train_one_restart(
                        noisy_data=noisy_data,
                        random_seed=(
                            restart_seed
                        ),
                    )
                )


                runtime = float(
                    time.time()
                    - start_time
                )


                trained_pool.append(
                    trained
                )


                print(
                    f"    runtime = "
                    f"{runtime:.1f}s"
                )


            # ==========================================================
            # methods
            # ==========================================================
            for method in (
                METHODS
            ):

                method_restart_results = []


                for restart_index, trained in enumerate(
                    trained_pool
                ):

                    result = (
                        evaluate_restart_method(
                            trained=trained,
                            method=method,
                            extended_roots=(
                                extended_roots
                            ),
                            target_roots=(
                                target_roots
                            ),
                            true_wavenumbers=(
                                true_wavenumbers
                            ),
                            noisy_data=(
                                noisy_data
                            ),
                        )
                    )


                    method_restart_results.append(
                        result
                    )


                    cell_counts = (
                        result[
                            "cell_counts"
                        ]
                    )


                    restart_rows.append(
                        {
                            "snr_db": (
                                snr_db
                            ),

                            "trial": (
                                trial
                            ),

                            "restart": (
                                restart_index + 1
                            ),

                            "method": (
                                method
                            ),

                            "selector_success": int(
                                result[
                                    "selector_success"
                                ]
                            ),

                            "selector_reason": (
                                result[
                                    "selector_reason"
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

                            "raw_k_rmse": (
                                result[
                                    "raw_k_rmse"
                                ]
                            ),

                            "refined_k_rmse": (
                                result[
                                    "refined_k_rmse"
                                ]
                            ),

                            "raw_max_error": (
                                result[
                                    "raw_max_error"
                                ]
                            ),

                            "refined_max_error": (
                                result[
                                    "refined_max_error"
                                ]
                            ),

                            "joint_fit": (
                                result[
                                    "joint_fit"
                                ]
                            ),

                            "restart_score": (
                                result[
                                    "restart_score"
                                ]
                            ),

                            "safe_refine_accepted": int(
                                result[
                                    "safe_refine_accepted"
                                ]
                            ),

                            "k4_selected": (
                                result[
                                    "k4_selected"
                                ]
                            ),

                            "k5_selected": (
                                result[
                                    "k5_selected"
                                ]
                            ),

                            "lower_bound_selected": (
                                result[
                                    "lower_bound_selected"
                                ]
                            ),

                            "cell1_count": (
                                cell_counts[0]
                                if len(
                                    cell_counts
                                ) > 0
                                else -1
                            ),

                            "cell2_count": (
                                cell_counts[1]
                                if len(
                                    cell_counts
                                ) > 1
                                else -1
                            ),

                            "cell3_count": (
                                cell_counts[2]
                                if len(
                                    cell_counts
                                ) > 2
                                else -1
                            ),

                            "selected_raw": (
                                array_to_text(
                                    result[
                                        "selected_raw"
                                    ]
                                )
                            ),

                            "selected_refined": (
                                array_to_text(
                                    result[
                                        "selected_refined"
                                    ]
                                )
                            ),
                        }
                    )


                # ======================================================
                # choose restart
                # ======================================================
                best_index = (
                    choose_best_restart(
                        method_restart_results
                    )
                )


                any_raw_success = bool(
                    any(
                        result[
                            "raw_success"
                        ]
                        for result
                        in method_restart_results
                    )
                )


                any_refined_success = bool(
                    any(
                        result[
                            "refined_success"
                        ]
                        for result
                        in method_restart_results
                    )
                )


                successful_restart_fraction = float(
                    np.mean(
                        [
                            float(
                                result[
                                    "raw_success"
                                ]
                            )
                            for result
                            in method_restart_results
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

                    chosen_raw = np.asarray(
                        [],
                        dtype=np.float64,
                    )

                    chosen_refined = np.asarray(
                        [],
                        dtype=np.float64,
                    )

                    chosen_k4 = 0

                    chosen_k5 = 0

                    chosen_lower = 0


                else:

                    chosen = (
                        method_restart_results[
                            best_index
                        ]
                    )


                    chosen_raw_success = bool(
                        chosen[
                            "raw_success"
                        ]
                    )

                    chosen_refined_success = bool(
                        chosen[
                            "refined_success"
                        ]
                    )

                    chosen_raw_rmse = float(
                        chosen[
                            "raw_k_rmse"
                        ]
                    )

                    chosen_refined_rmse = float(
                        chosen[
                            "refined_k_rmse"
                        ]
                    )

                    chosen_raw = np.asarray(
                        chosen[
                            "selected_raw"
                        ],
                        dtype=np.float64,
                    )

                    chosen_refined = np.asarray(
                        chosen[
                            "selected_refined"
                        ],
                        dtype=np.float64,
                    )

                    chosen_k4 = int(
                        chosen[
                            "k4_selected"
                        ]
                    )

                    chosen_k5 = int(
                        chosen[
                            "k5_selected"
                        ]
                    )

                    chosen_lower = int(
                        chosen[
                            "lower_bound_selected"
                        ]
                    )


                print(
                    f"\n  [{method}]"
                )

                print(
                    f"    raw success      = "
                    f"{chosen_raw_success}"
                )

                print(
                    f"    refined success  = "
                    f"{chosen_refined_success}"
                )

                print(
                    f"    any raw success  = "
                    f"{any_raw_success}"
                )

                print(
                    f"    restart success  = "
                    f"{successful_restart_fraction*100:.1f}%"
                )

                print(
                    f"    raw kRMSE        = "
                    f"{chosen_raw_rmse:.6e}"
                )

                print(
                    f"    refined kRMSE    = "
                    f"{chosen_refined_rmse:.6e}"
                )

                print(
                    f"    raw k            = "
                    f"{chosen_raw}"
                )

                print(
                    f"    refined k        = "
                    f"{chosen_refined}"
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
                            method
                        ),

                        "achieved_snr_db": (
                            achieved_snr
                        ),

                        "chosen_raw_success": int(
                            chosen_raw_success
                        ),

                        "chosen_refined_success": int(
                            chosen_refined_success
                        ),

                        "any_restart_raw_success": int(
                            any_raw_success
                        ),

                        "any_restart_refined_success": int(
                            any_refined_success
                        ),

                        "successful_restart_fraction": (
                            successful_restart_fraction
                        ),

                        "chosen_raw_k_rmse": (
                            chosen_raw_rmse
                        ),

                        "chosen_refined_k_rmse": (
                            chosen_refined_rmse
                        ),

                        "chosen_k4_selected": (
                            chosen_k4
                        ),

                        "chosen_k5_selected": (
                            chosen_k5
                        ),

                        "chosen_lower_bound_selected": (
                            chosen_lower
                        ),

                        "chosen_raw": (
                            array_to_text(
                                chosen_raw
                            )
                        ),

                        "chosen_refined": (
                            array_to_text(
                                chosen_refined
                            )
                        ),
                    }
                )


            # ==========================================================
            # incremental checkpoint
            # ==========================================================
            save_csv(
                RESTART_CSV,
                restart_rows,
            )

            save_csv(
                TRIAL_CSV,
                trial_rows,
            )


            current_summary = (
                summarize_trials(
                    trial_rows
                )
            )


            save_csv(
                SUMMARY_CSV,
                current_summary,
            )


    # ==================================================================
    # final summary
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


    make_figures(
        summary_rows
    )


    # ==================================================================
    # console table
    # ==================================================================
    print(
        "\n\n"
        + "=" * 180
    )

    print(
        "FINAL LOW-SNR SUMMARY"
    )

    print(
        "=" * 180
    )


    for row in (
        summary_rows
    ):

        print(
            f"SNR={row['snr_db']:>5.1f} dB | "
            f"{row['method']:18s} | "
            f"raw="
            f"{100*row['raw_success_rate']:6.1f}% | "
            f"ref="
            f"{100*row['refined_success_rate']:6.1f}% | "
            f"any="
            f"{100*row['any_restart_raw_success_rate']:6.1f}% | "
            f"restart="
            f"{100*row['successful_restart_fraction_mean']:6.1f}% | "
            f"k4="
            f"{100*row['k4_selection_rate']:6.1f}% | "
            f"k5="
            f"{100*row['k5_selection_rate']:6.1f}% | "
            f"boundary="
            f"{100*row['lower_bound_selection_rate']:6.1f}% | "
            f"rawRMSE="
            f"{row['raw_k_rmse_median']:.3e} | "
            f"refRMSE="
            f"{row['refined_k_rmse_median']:.3e}"
        )


    print(
        "\n"
        + "=" * 180
    )

    print(
        "输出文件"
    )

    print(
        "=" * 180
    )

    print(
        RESTART_CSV
    )

    print(
        TRIAL_CSV
    )

    print(
        SUMMARY_CSV
    )

    print(
        FIGURE_DIR
    )


    print(
        "\n"
        + "=" * 180
    )

    print(
        "最终判定重点"
    )

    print(
        "=" * 180
    )

    print(
        "1. 5 dB："
    )

    print(
        "   ORDER_CELL_JOINT应明显高于OOB_ROOT_ORDER。"
    )

    print(
        "\n2. 10 dB："
    )

    print(
        "   ORDER_CELL_JOINT至少不能破坏实验28已解决的性能。"
    )

    print(
        "\n3. 15 dB及以上："
    )

    print(
        "   不应损失原本已经稳定的高SNR性能。"
    )

    print(
        "\n4. k5/lower-bound selection："
    )

    print(
        "   ORDER_CELL_JOINT理论上应接近0。"
    )

    print(
        "\n5. 如果-5~-3 dB仍失败："
    )

    print(
        "   不继续无限调selector。"
    )

    print(
        "   直接记录当前低SNR性能边界，"
        "进入多模态和KRAKEN独立数据阶段。"
    )

    print(
        "=" * 180
    )


if __name__ == "__main__":
    main()
