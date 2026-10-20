# -*- coding: utf-8 -*-
"""
High-Order Physics Mode-FRINET
======================================================================

核心目标
----------------------------------------------------------------------
在不改变最初 FRI / FRINET 核心思想的前提下，将原来的：

    continuous k
        ↓
    FD2 differentiable modal atom
        ↓
    Y_hat = Phi(k) C

升级为：

    continuous k
        ↓
    Numerov4 high-order differentiable modal atom
        ↓
    realistic-bottom Robin physics
        ↓
    Y_hat = Phi_HO(k) C

保留：
----------------------------------------------------------------------
1. M=24 过参数化连续波数原子
2. k 直接作为连续可训练参数
3. 多快照复模态系数 C
4. 跨快照 group sparsity
5. bottom physics loss
6. 三阶段退火
7. Multi-start
8. mode-order cell
9. joint data-fit selection
10. Safe Physics Refinement

变化：
----------------------------------------------------------------------
仅升级核心物理解码器：

    FD2, dz=1.0

        ->

    Numerov4, dz=0.5

并采用四阶 backward derivative 计算真实海底 Robin residual。

======================================================================
为什么这样做
----------------------------------------------------------------------

前面的独立物理验证已经发现：

旧：
    FD2, dz=1.0

在 modes 6-8 上存在显著 KRAKEN mode-shape mismatch。

Numerov4 + dz=0.5 显著降低该高阶模型失配。

现在必须验证：

    mode-shape improvement

是否真正转化成：

    K=5 / K=8 wavenumber estimation improvement.

======================================================================
数据
----------------------------------------------------------------------

Forward:
    KRAKEN .MOD mode shapes

Inverse:
    High-Order Numerov Mode-FRINET

因此仍然满足：

    forward solver != inverse solver

不是自生成、自反演的闭环。

======================================================================
重要
----------------------------------------------------------------------

QUICK 第一轮只做：

    K = 3, 5, 8
    SNR = 15, 30 dB
    1 trial
    1 restart

主要看：

    K=8 raw 是否从旧版的 high-SNR 0% 出现明显改善。

不要一上来跑 FULL。
"""

from __future__ import annotations

import csv
import importlib.util
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


# ======================================================================
# 路径
# ======================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]

KRAKEN_IO_SCRIPT = (
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

OLD_MULTIMODE_SUMMARY = (
    PROJECT_ROOT
    / "outputs"
    / "multimode_mode_frinet_final"
    / "multimode_summary_full.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "high_order_physics_mode_frinet_final"
)

FIGURE_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ======================================================================
# 动态加载 KRAKEN parser
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

    sys.modules[module_name] = module

    spec.loader.exec_module(
        module
    )

    return module


KRAKEN_IO = load_module(
    KRAKEN_IO_SCRIPT,
    "kraken_io_for_ho_mode_frinet",
)


# ======================================================================
# QUICK / FULL
# ======================================================================
QUICK_TEST = True


if QUICK_TEST:

    TARGET_MODE_COUNTS = [
        3,
        5,
        8,
    ]

    SNR_VALUES_DB = [
        15.0,
        30.0,
    ]

    NUMBER_OF_TRIALS = 1

    NUMBER_OF_RESTARTS = 1

    RUN_SUFFIX = "quick"

else:

    # 暂时不要直接跑这个。
    # QUICK成功以后我们再决定正式FULL预算。
    TARGET_MODE_COUNTS = [
        3,
        5,
        8,
    ]

    SNR_VALUES_DB = [
        0.0,
        5.0,
        10.0,
        15.0,
        20.0,
        30.0,
    ]

    NUMBER_OF_TRIALS = 5

    NUMBER_OF_RESTARTS = 3

    RUN_SUFFIX = "full"


# ======================================================================
# 环境
# ======================================================================
FREQUENCY_HZ = 300.0

OMEGA = (
    2.0
    * np.pi
    * FREQUENCY_HZ
)

WATER_DEPTH_M = 40.0

WATER_DENSITY = 1.0

BOTTOM_CP = 1753.0

BOTTOM_DENSITY = 1.957


# ======================================================================
# High-order physics
# ======================================================================
PHYSICS_DZ = 0.5

BOTTOM_DERIVATIVE_ORDER = 4

BOTTOM_BULK_K = (
    OMEGA
    / BOTTOM_CP
)

# 水柱最大传播参考波数
K_MAX = (
    OMEGA
    / 1500.0
    * 0.999999
)

PHYSICS_ROOT_K_MIN = 1.08


# ======================================================================
# 根据目标模态数使用合理训练区间
# ======================================================================
K_MIN_BY_MODE_COUNT = {

    3: 1.20,

    5: 1.18,

    8: 1.08,
}


# ======================================================================
# Mode-FRINET core
# ======================================================================
NUMBER_OF_ATOMS = 24

NUMBER_OF_SNAPSHOTS = 20


# ======================================================================
# 原 V2 三阶段骨架
# ======================================================================
WARMUP_STEPS = 250

STAGE1_STEPS = 500

STAGE2_STEPS = 1000

STAGE3_STEPS = 500


COEFFICIENT_LR = 1.0e-2

STAGE1_K_LR = 5.0e-3

STAGE2_K_LR_START = 5.0e-3

STAGE2_K_LR_END = 1.0e-3

STAGE3_K_LR = 5.0e-4


GROUP_LAMBDA_STAGE1 = 1.0e-3

GROUP_LAMBDA_STAGE2_START = 1.0e-3

GROUP_LAMBDA_STAGE2_END = 5.0e-3

GROUP_LAMBDA_STAGE3 = 5.0e-3


BOTTOM_LAMBDA = 1.0

STAGE3_TOP_ATOMS = 8

GRADIENT_CLIP = 5.0


# ======================================================================
# Selector
# ======================================================================
TOP_CANDIDATES_PER_CELL = 4

CELL_INTERNAL_MIN_SEPARATION = 1.0e-3

BEAM_WIDTH = 64

PHYSICS_SCORE_GAMMA = 5000.0


# ======================================================================
# Refinement / metrics
# ======================================================================
SUCCESS_TOLERANCE = 1.5e-3

REFINEMENT_CAPTURE_RADIUS = 0.006

MAXIMUM_FIT_DEGRADATION = 1.10


BASE_RANDOM_SEED = 20260903


# ======================================================================
# OUTPUT
# ======================================================================
ROOT_CSV = (
    OUTPUT_DIR
    / "ho_physical_roots.csv"
)

RESTART_CSV = (
    OUTPUT_DIR
    / f"ho_mode_frinet_restarts_{RUN_SUFFIX}.csv"
)

TRIAL_CSV = (
    OUTPUT_DIR
    / f"ho_mode_frinet_trials_{RUN_SUFFIX}.csv"
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / f"ho_mode_frinet_summary_{RUN_SUFFIX}.csv"
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
# SSP
# ======================================================================
def sound_speed_np(
    depths,
):

    z = np.asarray(
        depths,
        dtype=np.float64,
    )

    c = np.empty_like(
        z
    )

    top = (
        z <= 10.0
    )

    gradient = (
        (z > 10.0)
        &
        (z < 20.0)
    )

    deep = (
        z >= 20.0
    )

    c[top] = 1505.0

    c[gradient] = (
        1505.0
        -
        0.5
        *
        (
            z[gradient]
            - 10.0
        )
    )

    c[deep] = 1500.0

    return c


# ======================================================================
# NumPy Numerov
# ======================================================================
def numerov_mode_np(
    horizontal_k: float,
    dz: float = PHYSICS_DZ,
):

    number_of_steps = int(
        round(
            WATER_DEPTH_M
            / dz
        )
    )

    depths = np.linspace(
        0.0,
        WATER_DEPTH_M,
        number_of_steps + 1,
        dtype=np.float64,
    )

    c = sound_speed_np(
        depths
    )

    q = (
        OMEGA ** 2
        /
        c ** 2
        -
        horizontal_k ** 2
    )

    psi = np.zeros(
        depths.size,
        dtype=np.float64,
    )

    psi[0] = 0.0

    psi[1] = 1.0

    h2 = dz ** 2

    for j in range(
        1,
        depths.size - 1,
    ):

        numerator = (

            2.0
            *
            (
                1.0
                -
                5.0
                *
                h2
                *
                q[j]
                /
                12.0
            )
            *
            psi[j]

            -

            (
                1.0
                +
                h2
                *
                q[j - 1]
                /
                12.0
            )
            *
            psi[j - 1]
        )

        denominator = (
            1.0
            +
            h2
            *
            q[j + 1]
            /
            12.0
        )

        psi[j + 1] = (
            numerator
            /
            denominator
        )

    return (
        depths,
        psi,
    )


# ======================================================================
# 4th-order bottom derivative
# ======================================================================
def bottom_derivative_np(
    psi,
    dz: float,
):

    return (

        25.0 * psi[-1]
        - 48.0 * psi[-2]
        + 36.0 * psi[-3]
        - 16.0 * psi[-4]
        + 3.0 * psi[-5]

    ) / (
        12.0
        * dz
    )


# ======================================================================
# Robin residual
# ======================================================================
def robin_residual_np(
    horizontal_k: float,
):

    _, psi = numerov_mode_np(
        horizontal_k
    )

    derivative = bottom_derivative_np(
        psi,
        PHYSICS_DZ,
    )

    q_bottom_squared = (
        horizontal_k ** 2
        -
        BOTTOM_BULK_K ** 2
    )

    if q_bottom_squared <= 0.0:

        return float(
            "nan"
        )

    q_bottom = math.sqrt(
        q_bottom_squared
    )

    boundary_term = (
        WATER_DENSITY
        /
        BOTTOM_DENSITY
        *
        q_bottom
        *
        psi[-1]
    )

    return float(
        derivative
        +
        boundary_term
    )


def normalized_robin_residual_np(
    horizontal_k: float,
):

    _, psi = numerov_mode_np(
        horizontal_k
    )

    derivative = bottom_derivative_np(
        psi,
        PHYSICS_DZ,
    )

    q2 = (
        horizontal_k ** 2
        -
        BOTTOM_BULK_K ** 2
    )

    if q2 <= 0:

        return 1.0

    q_bottom = math.sqrt(
        q2
    )

    boundary_term = (
        WATER_DENSITY
        /
        BOTTOM_DENSITY
        *
        q_bottom
        *
        psi[-1]
    )

    residual = (
        derivative
        +
        boundary_term
    )

    return float(
        abs(
            residual
        )
        /
        (
            abs(
                derivative
            )
            +
            abs(
                boundary_term
            )
            +
            1.0e-15
        )
    )


# ======================================================================
# Bisection
# ======================================================================
def bisect_root(
    left,
    right,
    maximum_iterations=80,
):

    f_left = robin_residual_np(
        left
    )

    f_right = robin_residual_np(
        right
    )

    if (
        not np.isfinite(
            f_left
        )
        or
        not np.isfinite(
            f_right
        )
    ):
        raise FloatingPointError(
            "root bracket residual 非有限"
        )

    if f_left == 0.0:
        return float(
            left
        )

    if f_right == 0.0:
        return float(
            right
        )

    if (
        f_left
        *
        f_right
        >
        0.0
    ):
        raise ValueError(
            "不是有效的root bracket"
        )

    for _ in range(
        maximum_iterations
    ):

        middle = (
            left
            +
            right
        ) / 2.0

        f_middle = robin_residual_np(
            middle
        )

        if (
            f_left
            *
            f_middle
            <= 0.0
        ):

            right = middle

            f_right = f_middle

        else:

            left = middle

            f_left = f_middle

    return float(
        (
            left
            +
            right
        )
        / 2.0
    )


# ======================================================================
# Find physical roots
# ======================================================================
def calculate_physical_roots():

    scan_k = np.linspace(
        PHYSICS_ROOT_K_MIN,
        K_MAX,
        5000,
        dtype=np.float64,
    )

    residuals = np.asarray(
        [
            robin_residual_np(
                float(k)
            )
            for k
            in scan_k
        ],
        dtype=np.float64,
    )

    roots = []

    for index in range(
        scan_k.size - 1
    ):

        left_value = residuals[
            index
        ]

        right_value = residuals[
            index + 1
        ]

        if (
            not np.isfinite(
                left_value
            )
            or
            not np.isfinite(
                right_value
            )
        ):
            continue

        if (
            left_value
            ==
            0.0
        ):

            roots.append(
                float(
                    scan_k[index]
                )
            )

        elif (
            left_value
            *
            right_value
            <
            0.0
        ):

            root = bisect_root(
                float(
                    scan_k[
                        index
                    ]
                ),
                float(
                    scan_k[
                        index + 1
                    ]
                ),
            )

            roots.append(
                root
            )

    roots = np.asarray(
        roots,
        dtype=np.float64,
    )

    roots = np.unique(
        np.round(
            roots,
            12,
        )
    )

    roots = np.sort(
        roots
    )[::-1]

    if roots.size < 8:

        raise RuntimeError(
            f"只找到 {roots.size} 个物理根，"
            "不足8阶。"
        )

    return roots


# ======================================================================
# NumPy high-order normalized mode matrix
# ======================================================================
def build_ho_mode_matrix_np(
    wavenumbers,
):

    wavenumbers = np.asarray(
        wavenumbers,
        dtype=np.float64,
    )

    columns = []

    for k in wavenumbers:

        depths, psi = numerov_mode_np(
            float(
                k
            )
        )

        sample_step = int(
            round(
                1.0
                /
                PHYSICS_DZ
            )
        )

        indices = np.arange(
            0,
            psi.size,
            sample_step,
            dtype=np.int64,
        )

        sampled = psi[
            indices
        ]

        if sampled.size != 41:

            raise RuntimeError(
                "Numerov integer-depth采样不是41点"
            )

        norm = np.linalg.norm(
            sampled
        )

        if (
            not np.isfinite(
                norm
            )
            or
            norm <= 1.0e-15
        ):

            raise FloatingPointError(
                f"k={k} mode norm异常"
            )

        columns.append(
            sampled
            /
            norm
        )

    return np.column_stack(
        columns
    )


# ======================================================================
# Torch differentiable Numerov
# ======================================================================
class HighOrderModeFRINET(
    nn.Module
):

    def __init__(
        self,
        number_of_atoms: int,
        number_of_snapshots: int,
        k_min: float,
        k_max: float,
        random_seed: int,
    ):

        super().__init__()

        self.number_of_atoms = int(
            number_of_atoms
        )

        self.number_of_snapshots = int(
            number_of_snapshots
        )

        self.k_min = float(
            k_min
        )

        self.k_max = float(
            k_max
        )

        self.dz = float(
            PHYSICS_DZ
        )


        # --------------------------------------------------------------
        # Fine depth grid
        # --------------------------------------------------------------
        number_of_steps = int(
            round(
                WATER_DEPTH_M
                /
                self.dz
            )
        )

        depths = np.linspace(
            0.0,
            WATER_DEPTH_M,
            number_of_steps + 1,
            dtype=np.float64,
        )

        c_profile = sound_speed_np(
            depths
        )

        self.register_buffer(
            "depths",
            torch.as_tensor(
                depths,
                dtype=torch.float64,
            ),
        )

        self.register_buffer(
            "sound_speed",
            torch.as_tensor(
                c_profile,
                dtype=torch.float64,
            ),
        )


        sample_step = int(
            round(
                1.0
                /
                self.dz
            )
        )

        sample_indices = np.arange(
            0,
            depths.size,
            sample_step,
            dtype=np.int64,
        )

        if sample_indices.size != 41:

            raise RuntimeError(
                "HighOrderModeFRINET需要41个整数深度点"
            )

        self.register_buffer(
            "sample_indices",
            torch.as_tensor(
                sample_indices,
                dtype=torch.long,
            ),
        )


        # --------------------------------------------------------------
        # Continuous k initialization
        #
        # 均匀随机分布在连续区间，
        # 不是固定 grid。
        # --------------------------------------------------------------
        rng = np.random.default_rng(
            random_seed
        )

        u = rng.uniform(
            0.03,
            0.97,
            size=(
                self.number_of_atoms,
            ),
        )

        raw_k = np.log(
            u
            /
            (
                1.0
                -
                u
            )
        )

        self.k_raw = nn.Parameter(
            torch.as_tensor(
                raw_k,
                dtype=torch.float64,
            )
        )


        # --------------------------------------------------------------
        # Complex multi-snapshot coefficients
        # --------------------------------------------------------------
        coefficient_real = (
            0.02
            *
            rng.standard_normal(
                (
                    self.number_of_atoms,
                    self.number_of_snapshots,
                )
            )
        )

        coefficient_imag = (
            0.02
            *
            rng.standard_normal(
                (
                    self.number_of_atoms,
                    self.number_of_snapshots,
                )
            )
        )

        self.coefficient_real = nn.Parameter(
            torch.as_tensor(
                coefficient_real,
                dtype=torch.float64,
            )
        )

        self.coefficient_imag = nn.Parameter(
            torch.as_tensor(
                coefficient_imag,
                dtype=torch.float64,
            )
        )


    # ------------------------------------------------------------------
    # Continuous k
    # ------------------------------------------------------------------
    def get_wavenumbers(
        self,
    ):

        sigmoid = torch.sigmoid(
            self.k_raw
        )

        return (
            self.k_min
            +
            (
                self.k_max
                -
                self.k_min
            )
            *
            sigmoid
        )


    # ------------------------------------------------------------------
    # Vectorized Numerov
    #
    # output:
    #     full_modes [M, 81]
    # ------------------------------------------------------------------
    def numerov_modes(
        self,
        wavenumbers,
    ):

        k = wavenumbers

        q = (
            OMEGA ** 2
            /
            self.sound_speed[
                None,
                :
            ] ** 2
            -
            k[
                :,
                None
            ] ** 2
        )

        h2 = (
            self.dz ** 2
        )

        psi_previous = torch.zeros_like(
            k
        )

        psi_current = torch.ones_like(
            k
        )

        values = [
            psi_previous,
            psi_current,
        ]


        for j in range(
            1,
            self.depths.numel() - 1,
        ):

            numerator = (

                2.0
                *
                (
                    1.0
                    -
                    5.0
                    *
                    h2
                    *
                    q[
                        :,
                        j
                    ]
                    /
                    12.0
                )
                *
                psi_current

                -

                (
                    1.0
                    +
                    h2
                    *
                    q[
                        :,
                        j - 1
                    ]
                    /
                    12.0
                )
                *
                psi_previous
            )

            denominator = (
                1.0
                +
                h2
                *
                q[
                    :,
                    j + 1
                ]
                /
                12.0
            )

            psi_next = (
                numerator
                /
                denominator
            )

            values.append(
                psi_next
            )

            psi_previous = (
                psi_current
            )

            psi_current = (
                psi_next
            )


        full_modes = torch.stack(
            values,
            dim=1,
        )

        return full_modes


    # ------------------------------------------------------------------
    # 4th-order realistic-bottom residual
    # ------------------------------------------------------------------
    def bottom_residuals(
        self,
        full_modes,
        wavenumbers,
    ):

        derivative = (

            25.0
            *
            full_modes[
                :,
                -1
            ]

            -

            48.0
            *
            full_modes[
                :,
                -2
            ]

            +

            36.0
            *
            full_modes[
                :,
                -3
            ]

            -

            16.0
            *
            full_modes[
                :,
                -4
            ]

            +

            3.0
            *
            full_modes[
                :,
                -5
            ]

        ) / (
            12.0
            *
            self.dz
        )


        q_bottom_squared = (
            wavenumbers ** 2
            -
            BOTTOM_BULK_K ** 2
        )

        q_bottom = torch.sqrt(
            torch.clamp(
                q_bottom_squared,
                min=1.0e-12,
            )
        )


        boundary_term = (

            WATER_DENSITY
            /
            BOTTOM_DENSITY

            *

            q_bottom

            *

            full_modes[
                :,
                -1
            ]
        )


        residual = (
            derivative
            +
            boundary_term
        )


        normalized = (

            torch.abs(
                residual
            )

            /

            (
                torch.abs(
                    derivative
                )

                +

                torch.abs(
                    boundary_term
                )

                +

                1.0e-12
            )
        )


        return normalized


    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
    ):

        wavenumbers = (
            self.get_wavenumbers()
        )


        full_modes = (
            self.numerov_modes(
                wavenumbers
            )
        )


        sampled_modes = (
            full_modes[
                :,
                self.sample_indices
            ]
        )


        # [M,41] -> [41,M]
        mode_matrix = (
            sampled_modes.transpose(
                0,
                1
            )
        )


        mode_norms = torch.sqrt(
            torch.sum(
                mode_matrix ** 2,
                dim=0,
            )
            +
            1.0e-18
        )


        mode_matrix = (
            mode_matrix
            /
            mode_norms[
                None,
                :
            ]
        )


        coefficients = torch.complex(
            self.coefficient_real,
            self.coefficient_imag,
        )


        reconstruction = (

            mode_matrix.to(
                torch.complex128
            )

            @

            coefficients
        )


        activities = torch.sqrt(

            torch.sum(
                torch.abs(
                    coefficients
                ) ** 2,
                dim=1,
            )

            +

            1.0e-18
        )


        bottom_residuals = (
            self.bottom_residuals(
                full_modes,
                wavenumbers,
            )
        )


        return SimpleNamespace(

            reconstruction=(
                reconstruction
            ),

            wavenumbers=(
                wavenumbers
            ),

            activities=(
                activities
            ),

            bottom_residuals=(
                bottom_residuals
            ),

            mode_matrix=(
                mode_matrix
            ),
        )


# ======================================================================
# Torch autograd sanity check
# ======================================================================
def verify_high_order_autograd():

    device = torch.device(
        "cpu"
    )

    test_k = torch.tensor(
        1.2356,
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )


    # 单原子临时Numerov
    dz = PHYSICS_DZ

    number_of_steps = int(
        round(
            WATER_DEPTH_M
            /
            dz
        )
    )

    depths = np.linspace(
        0.0,
        WATER_DEPTH_M,
        number_of_steps + 1,
    )

    c = torch.as_tensor(
        sound_speed_np(
            depths
        ),
        dtype=torch.float64,
    )


    def scalar_function(
        k,
    ):

        q = (
            OMEGA ** 2
            /
            c ** 2
            -
            k ** 2
        )

        previous = torch.zeros_like(
            k
        )

        current = torch.ones_like(
            k
        )

        values = [
            previous,
            current,
        ]

        h2 = dz ** 2

        for j in range(
            1,
            len(
                depths
            ) - 1,
        ):

            next_value = (

                2.0
                *
                (
                    1.0
                    -
                    5.0
                    *
                    h2
                    *
                    q[j]
                    /
                    12.0
                )
                *
                current

                -

                (
                    1.0
                    +
                    h2
                    *
                    q[j - 1]
                    /
                    12.0
                )
                *
                previous

            ) / (

                1.0
                +
                h2
                *
                q[j + 1]
                /
                12.0
            )

            values.append(
                next_value
            )

            previous = current

            current = next_value


        psi = torch.stack(
            values
        )


        # 用一个平滑scalar进行梯度检查
        return torch.sum(
            psi ** 2
        ) / (
            psi.numel()
        )


    value = scalar_function(
        test_k
    )

    value.backward()

    autograd_gradient = float(
        test_k.grad
    )


    epsilon = 1.0e-6

    with torch.no_grad():

        plus = float(
            scalar_function(
                torch.tensor(
                    1.2356 + epsilon,
                    dtype=torch.float64,
                )
            )
        )

        minus = float(
            scalar_function(
                torch.tensor(
                    1.2356 - epsilon,
                    dtype=torch.float64,
                )
            )
        )


    finite_difference_gradient = (
        plus
        -
        minus
    ) / (
        2.0
        *
        epsilon
    )


    relative_error = (

        abs(
            autograd_gradient
            -
            finite_difference_gradient
        )

        /

        max(
            abs(
                finite_difference_gradient
            ),
            1.0e-12,
        )
    )


    print(
        "\nHigh-order autograd check:"
    )

    print(
        f"  autograd = "
        f"{autograd_gradient:.9e}"
    )

    print(
        f"  finite   = "
        f"{finite_difference_gradient:.9e}"
    )

    print(
        f"  rel err  = "
        f"{relative_error:.3e}"
    )


    if relative_error > 1.0e-4:

        raise RuntimeError(
            "Numerov autograd验证失败"
        )


# ======================================================================
# Loss
# ======================================================================
def calculate_training_loss(
    output,
    target,
    group_lambda,
):

    fit_loss = torch.mean(
        torch.abs(
            output.reconstruction
            -
            target
        ) ** 2
    )


    group_loss = (

        torch.mean(
            output.activities
        )

        /

        math.sqrt(
            target.shape[1]
        )
    )


    activity_weights = (

        output.activities.detach()

        +

        0.02
        *
        torch.mean(
            output.activities.detach()
        )
    )


    bottom_loss = (

        torch.sum(

            activity_weights

            *

            output.bottom_residuals ** 2
        )

        /

        torch.sum(
            activity_weights
        )
    )


    total_loss = (

        fit_loss

        +

        group_lambda
        *
        group_loss

        +

        BOTTOM_LAMBDA
        *
        bottom_loss
    )


    return {
        "total": (
            total_loss
        ),

        "fit": (
            fit_loss
        ),

        "group": (
            group_loss
        ),

        "bottom": (
            bottom_loss
        ),
    }


# ======================================================================
# Train one restart
# ======================================================================
def train_one_restart(
    noisy_data,
    number_of_modes,
    random_seed,
):

    torch.manual_seed(
        random_seed
    )

    np.random.seed(
        random_seed
    )

    device = torch.device(
        "cpu"
    )


    data = torch.as_tensor(
        noisy_data,
        dtype=torch.complex128,
        device=device,
    )


    data_scale = torch.sqrt(
        torch.mean(
            torch.abs(
                data
            ) ** 2
        )
    )


    normalized_data = (
        data
        /
        data_scale
    )


    k_min = (
        K_MIN_BY_MODE_COUNT[
            number_of_modes
        ]
    )


    model = HighOrderModeFRINET(

        number_of_atoms=(
            NUMBER_OF_ATOMS
        ),

        number_of_snapshots=(
            noisy_data.shape[1]
        ),

        k_min=(
            k_min
        ),

        k_max=(
            K_MAX
        ),

        random_seed=(
            random_seed
        ),

    ).to(
        device
    )


    total_start = (
        time.perf_counter()
    )


    # ==================================================================
    # Warmup coefficients
    # ==================================================================
    model.k_raw.requires_grad_(
        False
    )


    optimizer = torch.optim.Adam(

        [
            model.coefficient_real,
            model.coefficient_imag,
        ],

        lr=(
            COEFFICIENT_LR
        ),
    )


    for _ in range(
        WARMUP_STEPS
    ):

        optimizer.zero_grad()

        output = model()

        losses = (
            calculate_training_loss(
                output=(
                    output
                ),

                target=(
                    normalized_data
                ),

                group_lambda=(
                    GROUP_LAMBDA_STAGE1
                ),
            )
        )


        # warmup不使用bottom拉动k，因为k被冻结
        warmup_loss = (

            losses[
                "fit"
            ]

            +

            GROUP_LAMBDA_STAGE1
            *
            losses[
                "group"
            ]
        )


        warmup_loss.backward()


        torch.nn.utils.clip_grad_norm_(
            [
                model.coefficient_real,
                model.coefficient_imag,
            ],
            GRADIENT_CLIP,
        )


        optimizer.step()


    # ==================================================================
    # Stage 1
    # ==================================================================
    model.k_raw.requires_grad_(
        True
    )


    optimizer = torch.optim.Adam(

        [

            {
                "params": [
                    model.k_raw
                ],

                "lr": (
                    STAGE1_K_LR
                ),
            },

            {
                "params": [
                    model.coefficient_real,
                    model.coefficient_imag,
                ],

                "lr": (
                    COEFFICIENT_LR
                ),
            },

        ]
    )


    for _ in range(
        STAGE1_STEPS
    ):

        optimizer.zero_grad()

        output = model()

        losses = calculate_training_loss(

            output=(
                output
            ),

            target=(
                normalized_data
            ),

            group_lambda=(
                GROUP_LAMBDA_STAGE1
            ),
        )


        if not torch.isfinite(
            losses[
                "total"
            ]
        ):

            raise FloatingPointError(
                "Stage1 loss非有限"
            )


        losses[
            "total"
        ].backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP,
        )


        optimizer.step()


    # ==================================================================
    # Stage 2 annealing
    # ==================================================================
    optimizer = torch.optim.Adam(

        [

            {
                "params": [
                    model.k_raw
                ],

                "lr": (
                    STAGE2_K_LR_START
                ),
            },

            {
                "params": [
                    model.coefficient_real,
                    model.coefficient_imag,
                ],

                "lr": (
                    COEFFICIENT_LR
                ),
            },

        ]
    )


    for local_step in range(
        STAGE2_STEPS
    ):

        fraction = (

            local_step

            /

            max(
                STAGE2_STEPS - 1,
                1,
            )
        )


        k_lr = (

            STAGE2_K_LR_START

            +

            fraction
            *
            (
                STAGE2_K_LR_END
                -
                STAGE2_K_LR_START
            )
        )


        group_lambda = (

            GROUP_LAMBDA_STAGE2_START

            +

            fraction
            *
            (
                GROUP_LAMBDA_STAGE2_END
                -
                GROUP_LAMBDA_STAGE2_START
            )
        )


        optimizer.param_groups[
            0
        ][
            "lr"
        ] = (
            k_lr
        )


        optimizer.zero_grad()


        output = model()


        losses = calculate_training_loss(

            output=(
                output
            ),

            target=(
                normalized_data
            ),

            group_lambda=(
                group_lambda
            ),
        )


        if not torch.isfinite(
            losses[
                "total"
            ]
        ):

            raise FloatingPointError(
                "Stage2 loss非有限"
            )


        losses[
            "total"
        ].backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP,
        )


        optimizer.step()


    # ==================================================================
    # Stage 3
    #
    # 只允许Top-8原子的k继续移动。
    # ==================================================================
    with torch.no_grad():

        stage2_output = model()

        top_count = min(
            STAGE3_TOP_ATOMS,
            NUMBER_OF_ATOMS,
        )

        top_indices = torch.topk(

            stage2_output.activities,

            k=(
                top_count
            ),

        ).indices


        active_mask = torch.zeros(
            NUMBER_OF_ATOMS,
            dtype=torch.float64,
            device=device,
        )

        active_mask[
            top_indices
        ] = 1.0


    optimizer = torch.optim.Adam(

        [

            {
                "params": [
                    model.k_raw
                ],

                "lr": (
                    STAGE3_K_LR
                ),
            },

            {
                "params": [
                    model.coefficient_real,
                    model.coefficient_imag,
                ],

                "lr": (
                    COEFFICIENT_LR
                ),
            },

        ]
    )


    for _ in range(
        STAGE3_STEPS
    ):

        optimizer.zero_grad()

        output = model()


        losses = calculate_training_loss(

            output=(
                output
            ),

            target=(
                normalized_data
            ),

            group_lambda=(
                GROUP_LAMBDA_STAGE3
            ),
        )


        if not torch.isfinite(
            losses[
                "total"
            ]
        ):

            raise FloatingPointError(
                "Stage3 loss非有限"
            )


        losses[
            "total"
        ].backward()


        # --------------------------------------------------------------
        # 仅Top-8 k允许更新
        # --------------------------------------------------------------
        if model.k_raw.grad is not None:

            model.k_raw.grad.mul_(
                active_mask
            )


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP,
        )


        optimizer.step()


    # ==================================================================
    # Final
    # ==================================================================
    model.eval()


    with torch.no_grad():

        final_output = model()


    runtime = (
        time.perf_counter()
        -
        total_start
    )


    return {

        "all_wavenumbers":
            final_output.wavenumbers
            .cpu()
            .numpy()
            .copy(),

        "all_activities":
            final_output.activities
            .cpu()
            .numpy()
            .copy(),

        "all_bottom_residuals":
            final_output.bottom_residuals
            .cpu()
            .numpy()
            .copy(),

        "runtime_seconds":
            float(
                runtime
            ),

        "final_fit_loss":
            float(
                torch.mean(
                    torch.abs(
                        final_output.reconstruction
                        -
                        normalized_data
                    ) ** 2
                ).cpu()
            ),
    }


# ======================================================================
# Data-fit
# ======================================================================
def calculate_data_fit(
    wavenumbers,
    noisy_data,
):

    k = np.asarray(
        wavenumbers,
        dtype=np.float64,
    )


    if k.size == 0:

        return float(
            "inf"
        )


    mode_matrix = (
        build_ho_mode_matrix_np(
            k
        )
    )


    coefficients, _, _, _ = (
        np.linalg.lstsq(

            mode_matrix.astype(
                np.complex128
            ),

            noisy_data,

            rcond=None,
        )
    )


    reconstruction = (

        mode_matrix

        @

        coefficients
    )


    residual = (

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


    return float(
        residual
    )


# ======================================================================
# Correct mode-order cells
#
# 对K=3/5：
# 最后一阶cell下界使用“目标根与下一物理根的中点”，
# 不允许k4/k6等更高阶根混入目标cell。
#
# 对K=8：
# 当前只有8个trapped roots，因此最后cell下界为training k_min。
# ======================================================================
def build_mode_cells(
    physical_roots,
    number_of_modes,
    training_k_min,
):

    roots = np.sort(
        np.asarray(
            physical_roots,
            dtype=np.float64,
        )
    )[::-1]


    if roots.size < number_of_modes:

        raise RuntimeError(
            "物理根数量不足"
        )


    cells = []


    for mode_index in range(
        number_of_modes
    ):

        root = roots[
            mode_index
        ]


        if mode_index == 0:

            upper = K_MAX


        else:

            upper = (

                roots[
                    mode_index - 1
                ]

                +

                roots[
                    mode_index
                ]

            ) / 2.0


        if (
            mode_index + 1
            <
            roots.size
        ):

            lower = (

                roots[
                    mode_index
                ]

                +

                roots[
                    mode_index + 1
                ]

            ) / 2.0


        else:

            lower = (
                training_k_min
            )


        lower = max(
            lower,
            training_k_min,
        )


        cells.append(
            {
                "mode": (
                    mode_index + 1
                ),

                "root": float(
                    root
                ),

                "lower": float(
                    lower
                ),

                "upper": float(
                    upper
                ),
            }
        )


    return cells


# ======================================================================
# Cell candidates
# ======================================================================
def get_cell_candidates(
    wavenumbers,
    activities,
    bottom_residuals,
    cell,
):

    wavenumbers = np.asarray(
        wavenumbers,
        dtype=np.float64,
    )

    activities = np.asarray(
        activities,
        dtype=np.float64,
    )

    bottom_residuals = np.asarray(
        bottom_residuals,
        dtype=np.float64,
    )


    physics_scores = (

        activities

        /

        (
            1.0

            +

            PHYSICS_SCORE_GAMMA
            *
            bottom_residuals ** 2
        )
    )


    indices = np.where(

        (
            wavenumbers
            >
            cell[
                "lower"
            ]
        )

        &

        (
            wavenumbers
            <=
            cell[
                "upper"
            ]
        )

    )[0]


    if indices.size == 0:

        return []


    ordered = indices[
        np.argsort(
            physics_scores[
                indices
            ]
        )[::-1]
    ]


    selected = []


    for index in ordered:

        candidate_k = float(
            wavenumbers[
                index
            ]
        )


        if all(

            abs(
                candidate_k
                -
                float(
                    wavenumbers[
                        old_index
                    ]
                )
            )
            >=
            CELL_INTERNAL_MIN_SEPARATION

            for old_index
            in selected

        ):

            selected.append(
                int(
                    index
                )
            )


        if (
            len(
                selected
            )
            >=
            TOP_CANDIDATES_PER_CELL
        ):

            break


    return selected


# ======================================================================
# Beam selector
# ======================================================================
def select_mode_order_beam(
    trained,
    noisy_data,
    physical_roots,
    number_of_modes,
):

    wavenumbers = np.asarray(
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

    bottom_residuals = np.asarray(
        trained[
            "all_bottom_residuals"
        ],
        dtype=np.float64,
    )


    training_k_min = (
        K_MIN_BY_MODE_COUNT[
            number_of_modes
        ]
    )


    cells = build_mode_cells(

        physical_roots=(
            physical_roots
        ),

        number_of_modes=(
            number_of_modes
        ),

        training_k_min=(
            training_k_min
        ),
    )


    cell_candidates = []


    for cell in cells:

        candidates = (
            get_cell_candidates(

                wavenumbers=(
                    wavenumbers
                ),

                activities=(
                    activities
                ),

                bottom_residuals=(
                    bottom_residuals
                ),

                cell=(
                    cell
                ),
            )
        )


        if not candidates:

            return {

                "success": False,

                "reason": (
                    f"EMPTY_CELL_MODE_{cell['mode']}"
                ),

                "selected": np.asarray(
                    [],
                    dtype=np.float64,
                ),

                "fit": float(
                    "inf"
                ),

                "cell_counts": [
                    len(
                        old
                    )
                    for old
                    in cell_candidates
                ] + [0],
            }


        cell_candidates.append(
            candidates
        )


    physics_scores = (

        activities

        /

        (
            1.0

            +

            PHYSICS_SCORE_GAMMA
            *
            bottom_residuals ** 2
        )
    )


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
                        k_values,
                        noisy_data,
                    )
                )


                physics_score = float(

                    np.sum(

                        [
                            physics_scores[
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

                        "fit": (
                            fit
                        ),

                        "physics_score": (
                            physics_score
                        ),
                    }
                )


        expanded.sort(

            key=lambda state: (

                state[
                    "fit"
                ],

                -
                state[
                    "physics_score"
                ],
            )
        )


        beam = (
            expanded[
                :BEAM_WIDTH
            ]
        )


        if not beam:

            return {

                "success": False,

                "reason": (
                    "BEAM_EMPTY"
                ),

                "selected": np.asarray(
                    [],
                    dtype=np.float64,
                ),

                "fit": float(
                    "inf"
                ),

                "cell_counts": [
                    len(
                        item
                    )
                    for item
                    in cell_candidates
                ],
            }


    best = beam[
        0
    ]


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
            "HO_MODE_ORDER_BEAM"
        ),

        "selected": (
            selected
        ),

        "fit": float(
            best[
                "fit"
            ]
        ),

        "cell_counts": [
            len(
                item
            )
            for item
            in cell_candidates
        ],
    }


# ======================================================================
# Safe refinement
# ======================================================================
def safe_refine(
    raw_k,
    target_roots,
    noisy_data,
):

    raw = np.sort(
        np.asarray(
            raw_k,
            dtype=np.float64,
        )
    )[::-1]


    roots = np.sort(
        np.asarray(
            target_roots,
            dtype=np.float64,
        )
    )[::-1]


    if raw.size != roots.size:

        return {

            "accepted": False,

            "refined": (
                raw
            ),
        }


    refined = raw.copy()


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

            <=

            REFINEMENT_CAPTURE_RADIUS
        ):

            refined[
                index
            ] = (
                roots[
                    index
                ]
            )


    raw_fit = calculate_data_fit(
        raw,
        noisy_data,
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
        *
        MAXIMUM_FIT_DEGRADATION
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
# Metrics
# ======================================================================
def evaluate_k(
    estimated,
    truth,
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

            "rmse": float(
                "inf"
            ),

            "max_error": float(
                "inf"
            ),

            "errors": np.full(
                truth.shape,
                np.inf,
            ),
        }


    errors = np.abs(
        estimated
        -
        truth
    )


    return {

        "success": bool(

            np.max(
                errors
            )

            <=

            SUCCESS_TOLERANCE
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

        "errors": (
            errors
        ),
    }


# ======================================================================
# KRAKEN forward
# ======================================================================
def load_kraken():

    mod = KRAKEN_IO.parse_kraken_mod(
        KRAKEN_MOD_PATH
    )


    receiver = (
        KRAKEN_IO.extract_integer_receiver_modes(
            mod
        )
    )


    receiver_modes = np.asarray(
        receiver[
            "receiver_mode_matrix"
        ],
        dtype=np.complex128,
    )


    complex_k = np.asarray(
        mod[
            "complex_wavenumbers"
        ],
        dtype=np.complex128,
    )


    surface = np.zeros(
        (
            1,
            receiver_modes.shape[1],
        ),
        dtype=np.complex128,
    )


    mode_matrix = np.vstack(
        [
            surface,
            receiver_modes,
        ]
    )


    # normalize each KRAKEN mode
    for column in range(
        mode_matrix.shape[1]
    ):

        mode_matrix[
            :,
            column
        ] /= max(

            np.linalg.norm(
                mode_matrix[
                    :,
                    column
                ]
            ),

            1.0e-15,
        )


    return {
        "mode_matrix": (
            mode_matrix
        ),

        "complex_k": (
            complex_k
        ),
    }


# ======================================================================
# Dataset
#
# 与旧multimode实验保持相同seed结构。
# ======================================================================
def make_master_data(
    forward_matrix,
    coefficient_seed,
    noise_seed,
):

    K = (
        forward_matrix.shape[
            1
        ]
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

    ) / np.sqrt(
        2.0
    )


    clean = (
        forward_matrix
        @
        coefficients
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

    ) / np.sqrt(
        2.0
    )


    noise /= max(
        np.linalg.norm(
            noise
        ),
        1.0e-15,
    )


    return {
        "clean": (
            clean
        ),

        "unit_noise": (
            noise
        ),
    }


def add_noise(
    master,
    snr_db,
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


    clean_norm = np.linalg.norm(
        clean
    )


    noise_norm = (

        clean_norm

        /

        (
            10.0
            **
            (
                snr_db
                /
                20.0
            )
        )
    )


    noisy = (

        clean

        +

        noise_norm
        *
        unit_noise
    )


    return noisy


# ======================================================================
# Summary
# ======================================================================
def summarize(
    trial_rows,
):

    rows = []


    for K in TARGET_MODE_COUNTS:

        for snr_db in SNR_VALUES_DB:

            subset = [

                row

                for row
                in trial_rows

                if (
                    int(
                        row[
                            "K"
                        ]
                    )
                    ==
                    K

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


            raw_rmse = np.asarray(
                [
                    row[
                        "raw_rmse"
                    ]
                    for row
                    in subset
                ],
                dtype=np.float64,
            )


            ref_rmse = np.asarray(
                [
                    row[
                        "refined_rmse"
                    ]
                    for row
                    in subset
                ],
                dtype=np.float64,
            )


            rows.append(
                {

                    "K": (
                        K
                    ),

                    "snr_db": (
                        snr_db
                    ),

                    "trials": (
                        len(
                            subset
                        )
                    ),

                    "raw_success_rate": float(

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

                    "refined_success_rate": float(

                        np.mean(

                            [
                                row[
                                    "refined_success"
                                ]

                                for row
                                in subset
                            ]
                        )
                    ),

                    "any_restart_raw_success_rate": float(

                        np.mean(

                            [
                                row[
                                    "any_restart_raw_success"
                                ]

                                for row
                                in subset
                            ]
                        )
                    ),

                    "raw_rmse_median": float(
                        np.median(
                            raw_rmse
                        )
                    ),

                    "refined_rmse_median": float(
                        np.median(
                            ref_rmse
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


    return rows


# ======================================================================
# Plot
# ======================================================================
def plot_summary(
    summary_rows,
):

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
            ==
            K
        ]


        subset.sort(
            key=lambda row:
            row[
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
                100.0
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
        "High-Order Physics Mode-FRINET"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()


    fig.savefig(
        FIGURE_DIR
        /
        f"ho_raw_success_{RUN_SUFFIX}.png",
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


    print(
        "=" * 180
    )

    print(
        "HIGH-ORDER PHYSICS MODE-FRINET"
    )

    print(
        "=" * 180
    )

    print(
        f"Mode              = {RUN_SUFFIX}"
    )

    print(
        f"Physics solver    = Numerov4"
    )

    print(
        f"Physics dz        = {PHYSICS_DZ} m"
    )

    print(
        f"Bottom derivative = {BOTTOM_DERIVATIVE_ORDER}th-order"
    )

    print(
        f"M                 = {NUMBER_OF_ATOMS}"
    )

    print(
        f"L                 = {NUMBER_OF_SNAPSHOTS}"
    )

    print(
        f"K                 = {TARGET_MODE_COUNTS}"
    )

    print(
        f"SNR               = {SNR_VALUES_DB}"
    )

    print(
        f"trials            = {NUMBER_OF_TRIALS}"
    )

    print(
        f"restarts          = {NUMBER_OF_RESTARTS}"
    )

    print(
        "=" * 180
    )


    # ==================================================================
    # Autograd
    # ==================================================================
    verify_high_order_autograd()


    # ==================================================================
    # High-order roots
    # ==================================================================
    physical_roots = (
        calculate_physical_roots()
    )


    print(
        "\nHigh-order Robin roots:"
    )


    root_rows = []


    for index, root in enumerate(
        physical_roots,
        start=1,
    ):

        print(
            f"  r{index} = "
            f"{root:.12f}"
        )

        root_rows.append(
            {
                "mode": (
                    index
                ),

                "ho_robin_root": (
                    root
                ),
            }
        )


    save_csv(
        ROOT_CSV,
        root_rows,
    )


    # ==================================================================
    # KRAKEN
    # ==================================================================
    kraken = (
        load_kraken()
    )


    kraken_modes = (
        kraken[
            "mode_matrix"
        ]
    )


    complex_k = (
        kraken[
            "complex_k"
        ]
    )


    print(
        "\nKRAKEN truth vs HO root:"
    )


    for index in range(
        8
    ):

        truth = float(
            np.real(
                complex_k[
                    index
                ]
            )
        )

        root = float(
            physical_roots[
                index
            ]
        )

        print(
            f"  Mode {index + 1}: "
            f"KRAKEN={truth:.12f} | "
            f"HO={root:.12f} | "
            f"|dk|={abs(truth-root):.3e}"
        )


    # ==================================================================
    # Monte Carlo
    # ==================================================================
    restart_rows = []

    trial_rows = []


    for K in TARGET_MODE_COUNTS:

        true_k = np.asarray(
            np.real(
                complex_k[
                    :K
                ]
            ),
            dtype=np.float64,
        )


        forward_matrix = np.asarray(
            kraken_modes[
                :,
                :K
            ],
            dtype=np.complex128,
        )


        target_roots = np.asarray(
            physical_roots[
                :K
            ],
            dtype=np.float64,
        )


        print(
            "\n\n"
            + "=" * 180
        )

        print(
            f"K = {K}"
        )

        print(
            "=" * 180
        )

        print(
            "Truth:"
        )

        print(
            true_k
        )

        print(
            "HO roots:"
        )

        print(
            target_roots
        )


        masters = {}


        for trial in range(
            1,
            NUMBER_OF_TRIALS + 1,
        ):

            seed_base = (

                BASE_RANDOM_SEED

                +

                K
                *
                1_000_000

                +

                trial
                *
                10_000
            )


            masters[
                trial
            ] = (
                make_master_data(

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
            )


        for snr_db in SNR_VALUES_DB:

            print(
                "\n"
                + "-" * 170
            )

            print(
                f"K={K} | "
                f"SNR={snr_db:+.1f} dB"
            )

            print(
                "-" * 170
            )


            for trial in range(
                1,
                NUMBER_OF_TRIALS + 1,
            ):

                noisy_data = add_noise(
                    masters[
                        trial
                    ],
                    snr_db,
                )


                restart_results = []

                total_runtime = 0.0


                for restart_index in range(
                    NUMBER_OF_RESTARTS
                ):

                    model_seed = (

                        BASE_RANDOM_SEED

                        +

                        K
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

                        +

                        restart_index
                        *
                        1009
                    )


                    print(
                        f"\n  Trial "
                        f"{trial}/"
                        f"{NUMBER_OF_TRIALS} | "
                        f"restart "
                        f"{restart_index + 1}/"
                        f"{NUMBER_OF_RESTARTS}",
                        flush=True,
                    )


                    trained = (
                        train_one_restart(

                            noisy_data=(
                                noisy_data
                            ),

                            number_of_modes=(
                                K
                            ),

                            random_seed=(
                                model_seed
                            ),
                        )
                    )


                    total_runtime += (
                        trained[
                            "runtime_seconds"
                        ]
                    )


                    selection = (
                        select_mode_order_beam(

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
                                K
                            ),
                        )
                    )


                    if selection[
                        "success"
                    ]:

                        raw_k = np.sort(
                            selection[
                                "selected"
                            ]
                        )[::-1]


                        raw_eval = evaluate_k(
                            raw_k,
                            true_k,
                        )


                        refinement = (
                            safe_refine(

                                raw_k=(
                                    raw_k
                                ),

                                target_roots=(
                                    target_roots
                                ),

                                noisy_data=(
                                    noisy_data
                                ),
                            )
                        )


                        refined_k = np.asarray(
                            refinement[
                                "refined"
                            ],
                            dtype=np.float64,
                        )


                        refined_eval = (
                            evaluate_k(
                                refined_k,
                                true_k,
                            )
                        )


                    else:

                        raw_k = np.asarray(
                            [],
                            dtype=np.float64,
                        )

                        refined_k = (
                            raw_k
                        )

                        raw_eval = {
                            "success": False,
                            "rmse": float("inf"),
                            "max_error": float("inf"),
                            "errors": np.full(
                                K,
                                np.inf,
                            ),
                        }

                        refined_eval = (
                            raw_eval.copy()
                        )

                        refinement = {
                            "accepted": False,
                        }


                    restart_result = {

                        "selector_success": (
                            selection[
                                "success"
                            ]
                        ),

                        "selector_reason": (
                            selection[
                                "reason"
                            ]
                        ),

                        "selector_fit": (
                            selection[
                                "fit"
                            ]
                        ),

                        "raw_k": (
                            raw_k
                        ),

                        "refined_k": (
                            refined_k
                        ),

                        "raw_eval": (
                            raw_eval
                        ),

                        "refined_eval": (
                            refined_eval
                        ),

                        "refine_accepted": bool(
                            refinement[
                                "accepted"
                            ]
                        ),

                        "runtime": (
                            trained[
                                "runtime_seconds"
                            ]
                        ),
                    }


                    restart_results.append(
                        restart_result
                    )


                    restart_rows.append(
                        {

                            "K": (
                                K
                            ),

                            "snr_db": (
                                snr_db
                            ),

                            "trial": (
                                trial
                            ),

                            "restart": (
                                restart_index + 1
                            ),

                            "seed": (
                                model_seed
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

                            "selector_fit": (
                                selection[
                                    "fit"
                                ]
                            ),

                            "raw_success": int(
                                raw_eval[
                                    "success"
                                ]
                            ),

                            "refined_success": int(
                                refined_eval[
                                    "success"
                                ]
                            ),

                            "raw_rmse": (
                                raw_eval[
                                    "rmse"
                                ]
                            ),

                            "refined_rmse": (
                                refined_eval[
                                    "rmse"
                                ]
                            ),

                            "raw_max_error": (
                                raw_eval[
                                    "max_error"
                                ]
                            ),

                            "refined_max_error": (
                                refined_eval[
                                    "max_error"
                                ]
                            ),

                            "runtime_seconds": (
                                trained[
                                    "runtime_seconds"
                                ]
                            ),

                            "raw_k": (
                                array_text(
                                    raw_k
                                )
                            ),

                            "refined_k": (
                                array_text(
                                    refined_k
                                )
                            ),
                        }
                    )


                # ======================================================
                # Choose best restart WITHOUT truth
                # ======================================================
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
                                "selector_fit"
                            ]
                        )
                    )
                ]


                if valid:

                    best_index = min(

                        valid,

                        key=lambda index:
                        restart_results[
                            index
                        ][
                            "selector_fit"
                        ],
                    )


                    best = restart_results[
                        best_index
                    ]


                    raw_eval = (
                        best[
                            "raw_eval"
                        ]
                    )

                    refined_eval = (
                        best[
                            "refined_eval"
                        ]
                    )

                    raw_k = (
                        best[
                            "raw_k"
                        ]
                    )

                    refined_k = (
                        best[
                            "refined_k"
                        ]
                    )


                else:

                    best_index = -1

                    raw_eval = {
                        "success": False,
                        "rmse": float("inf"),
                        "max_error": float("inf"),
                    }

                    refined_eval = (
                        raw_eval.copy()
                    )

                    raw_k = np.asarray(
                        [],
                        dtype=np.float64,
                    )

                    refined_k = (
                        raw_k
                    )


                any_raw_success = bool(

                    any(

                        result[
                            "raw_eval"
                        ][
                            "success"
                        ]

                        for result
                        in restart_results
                    )
                )


                print(
                    "\n  RESULT"
                )

                print(
                    f"    raw success     = "
                    f"{raw_eval['success']}"
                )

                print(
                    f"    raw RMSE        = "
                    f"{raw_eval['rmse']:.6e}"
                )

                print(
                    f"    raw max error   = "
                    f"{raw_eval['max_error']:.6e}"
                )

                print(
                    f"    raw k           = "
                    f"{np.asarray(raw_k)}"
                )

                print(
                    f"    refined success = "
                    f"{refined_eval['success']}"
                )

                print(
                    f"    refined RMSE    = "
                    f"{refined_eval['rmse']:.6e}"
                )


                trial_rows.append(
                    {

                        "K": (
                            K
                        ),

                        "snr_db": (
                            snr_db
                        ),

                        "trial": (
                            trial
                        ),

                        "best_restart": (
                            best_index + 1
                            if best_index
                            >= 0
                            else -1
                        ),

                        "raw_success": int(
                            raw_eval[
                                "success"
                            ]
                        ),

                        "refined_success": int(
                            refined_eval[
                                "success"
                            ]
                        ),

                        "any_restart_raw_success": int(
                            any_raw_success
                        ),

                        "raw_rmse": (
                            raw_eval[
                                "rmse"
                            ]
                        ),

                        "refined_rmse": (
                            refined_eval[
                                "rmse"
                            ]
                        ),

                        "raw_max_error": (
                            raw_eval[
                                "max_error"
                            ]
                        ),

                        "refined_max_error": (
                            refined_eval[
                                "max_error"
                            ]
                        ),

                        "runtime_seconds": (
                            total_runtime
                        ),

                        "raw_k": (
                            array_text(
                                raw_k
                            )
                        ),

                        "refined_k": (
                            array_text(
                                refined_k
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
                    summarize(
                        trial_rows
                    ),
                )


    # ==================================================================
    # Final
    # ==================================================================
    summary_rows = summarize(
        trial_rows
    )


    save_csv(
        SUMMARY_CSV,
        summary_rows,
    )


    plot_summary(
        summary_rows
    )


    print(
        "\n\n"
        + "=" * 185
    )

    print(
        "HIGH-ORDER MODE-FRINET SUMMARY"
    )

    print(
        "=" * 185
    )


    for row in summary_rows:

        print(

            f"K={row['K']} | "
            f"SNR={row['snr_db']:>5.1f} | "
            f"raw="
            f"{100*row['raw_success_rate']:6.1f}% | "
            f"ref="
            f"{100*row['refined_success_rate']:6.1f}% | "
            f"rawRMSE="
            f"{row['raw_rmse_median']:.3e} | "
            f"refRMSE="
            f"{row['refined_rmse_median']:.3e} | "
            f"time="
            f"{row['runtime_seconds_mean']:.1f}s"
        )


    print(
        "\n"
        + "=" * 185
    )

    print(
        "PRIMARY DECISION"
    )

    print(
        "=" * 185
    )


    k8_high = [

        row

        for row
        in summary_rows

        if (
            row[
                "K"
            ]
            ==
            8

            and

            row[
                "snr_db"
            ]
            >=
            15.0
        )
    ]


    if k8_high:

        mean_k8_success = float(

            np.mean(

                [
                    row[
                        "raw_success_rate"
                    ]

                    for row
                    in k8_high
                ]
            )
        )


        print(
            f"K=8 high-SNR raw success mean = "
            f"{100*mean_k8_success:.1f}%"
        )


        if mean_k8_success > 0.0:

            print(
                "PROMISING:"
            )

            print(
                "旧FD2版本K=8 high-SNR raw success曾为0%。"
            )

            print(
                "High-order Numerov physics已经产生了可观测的估计收益。"
            )

            print(
                "下一步应进行受控Monte-Carlo确认，而不是继续修改模型。"
            )


        else:

            print(
                "NO TRANSFER YET:"
            )

            print(
                "虽然物理mode-shape mismatch显著降低，"
                "但当前训练/选择层尚未把它转化为K=8恢复成功率。"
            )

            print(
                "下一步检查candidate formation，而不是否定Numerov物理。"
            )


    print(
        "\nOutputs:"
    )

    print(
        ROOT_CSV
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


if __name__ == "__main__":
    main()
