# -*- coding: utf-8 -*-
"""
High-Order Bottom / Interior Discretization Validation
======================================================================

目的
----------------------------------------------------------------------
诊断KRAKEN真实海底条件下，高阶模态m=6~8的mode-shape mismatch来源。

比较两种水柱模态计算：

1. FD2
   当前Mode-FRINET使用的二阶中心有限差分递推。

2. NUMEROV4
   针对
       psi'' + Q(z,k) psi = 0
   的高阶Numerov递推。

同时测试：

    dz = 1.0
    dz = 0.5
    dz = 0.25
    dz = 0.125 m

底部Robin残差同时使用：

    2nd-order backward derivative
    4th-order backward derivative

======================================================================
关键目的
----------------------------------------------------------------------

不是为了让某个结果“必须变好”，而是区分：

A. 如果减小dz明显降低KRAKEN mode-shape mismatch：
       -> 水柱离散误差是重要来源。

B. 如果Numerov4显著优于FD2：
       -> 高阶可微模态原子值得加入最终Mode-FRINET。

C. 如果shape几乎不变，但4阶bottom derivative显著降低残差：
       -> 底部导数离散是重要来源。

D. 如果以上都不能明显改善m=6~8：
       -> 问题不只是数值阶数，需要重新检查真实半空间边界/
          KRAKEN模态表示/环境模型差异。

======================================================================
注意
----------------------------------------------------------------------

本脚本：

    不训练神经网络；
    不修改selector；
    不做参数调优。

只做物理原子本身的独立验证。
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ======================================================================
# PATH
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "high_order_bottom_discretization_validation"
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
# OUTPUT FILES
# ======================================================================
MODEWISE_CSV = (
    OUTPUT_DIR
    / "high_order_discretization_modewise.csv"
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "high_order_discretization_summary.csv"
)

BOTTOM_CSV = (
    OUTPUT_DIR
    / "bottom_derivative_residual_comparison.csv"
)

BEST_CSV = (
    OUTPUT_DIR
    / "best_physics_variant_summary.csv"
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


KRAKEN_IO = load_module(
    KRAKEN_IO_SCRIPT,
    "kraken_io_for_high_order_validation",
)


# ======================================================================
# ENVIRONMENT
# ======================================================================
FREQUENCY_HZ = 300.0

OMEGA = (
    2.0
    * np.pi
    * FREQUENCY_HZ
)

WATER_DEPTH_M = 40.0

BOTTOM_CP = 1753.0

WATER_DENSITY = 1.0

BOTTOM_DENSITY = 1.957

NUMBER_OF_MODES = 8


# ======================================================================
# EXPERIMENT CONFIG
# ======================================================================
DZ_VALUES = [
    1.0,
    0.5,
    0.25,
    0.125,
]

SOLVERS = [
    "FD2",
    "NUMEROV4",
]


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


# ======================================================================
# SSP
#
# teacher environment:
#
# 0~10 m:
#     c = 1505
#
# 10~20 m:
#     c decreases linearly:
#     1505 -> 1500
#
# 20~40 m:
#     c = 1500
# ======================================================================
def sound_speed(
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


    c[
        top
    ] = 1505.0


    c[
        gradient
    ] = (
        1505.0
        -
        0.5
        *
        (
            z[
                gradient
            ]
            - 10.0
        )
    )


    c[
        deep
    ] = 1500.0


    return c


# ======================================================================
# Q(z,k)
#
# psi'' + Q psi = 0
#
# Q = omega^2/c^2 - k^2
# ======================================================================
def calculate_q(
    depths,
    horizontal_k,
):

    c = sound_speed(
        depths
    )

    return (
        OMEGA ** 2
        /
        c ** 2
        -
        horizontal_k ** 2
    )


# ======================================================================
# FD2
#
# psi_{j+1}
# =
# [2-h^2 Q_j] psi_j
# -
# psi_{j-1}
# ======================================================================
def solve_fd2(
    horizontal_k: float,
    dz: float,
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


    q = calculate_q(
        depths,
        horizontal_k,
    )


    psi = np.zeros(
        depths.size,
        dtype=np.complex128,
    )


    # pressure-release surface
    psi[0] = 0.0


    # arbitrary nonzero derivative / amplitude
    #
    # overall amplitude is removed later by complex alignment
    psi[1] = 1.0


    for j in range(
        1,
        depths.size - 1,
    ):

        psi[
            j + 1
        ] = (
            (
                2.0
                -
                dz ** 2
                *
                q[j]
            )
            *
            psi[j]
            -
            psi[
                j - 1
            ]
        )


    return (
        depths,
        psi,
    )


# ======================================================================
# NUMEROV4
#
# For:
#
#     psi'' + Q(z) psi = 0
#
# Numerov:
#
# psi_{n+1}
# =
# [
#   2(1 - 5h^2 Q_n/12) psi_n
#   -
#   (1 + h^2 Q_{n-1}/12) psi_{n-1}
# ]
# /
# [
#   1 + h^2 Q_{n+1}/12
# ]
#
# Higher-order interior propagation.
# ======================================================================
def solve_numerov4(
    horizontal_k: float,
    dz: float,
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


    q = calculate_q(
        depths,
        horizontal_k,
    )


    psi = np.zeros(
        depths.size,
        dtype=np.complex128,
    )


    psi[0] = 0.0


    # arbitrary initial amplitude.
    # Since psi(0)=0, any nonzero psi(h) only determines scale
    # for this homogeneous linear equation.
    psi[1] = 1.0


    h2 = (
        dz ** 2
    )


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
                q[
                    j - 1
                ]
                /
                12.0
            )
            *
            psi[
                j - 1
            ]
        )


        denominator = (
            1.0
            +
            h2
            *
            q[
                j + 1
            ]
            /
            12.0
        )


        psi[
            j + 1
        ] = (
            numerator
            /
            denominator
        )


    return (
        depths,
        psi,
    )


# ======================================================================
# Unified solver
# ======================================================================
def solve_mode(
    solver_name: str,
    horizontal_k: float,
    dz: float,
):

    if solver_name == "FD2":

        return solve_fd2(
            horizontal_k,
            dz,
        )


    if solver_name == "NUMEROV4":

        return solve_numerov4(
            horizontal_k,
            dz,
        )


    raise ValueError(
        f"Unknown solver: {solver_name}"
    )


# ======================================================================
# Sample fine-grid mode at integer depths:
#
# z = 0,1,...,40
#
# No interpolation error because all tested dz divide 1 m exactly.
# ======================================================================
def sample_integer_depths(
    depths: np.ndarray,
    psi: np.ndarray,
):

    target_depths = np.arange(
        0.0,
        WATER_DEPTH_M + 1.0,
        1.0,
        dtype=np.float64,
    )


    indices = np.asarray(
        np.round(
            target_depths
            /
            (
                depths[1]
                -
                depths[0]
            )
        ),
        dtype=np.int64,
    )


    sampled = (
        psi[
            indices
        ]
    )


    return (
        target_depths,
        sampled,
    )


# ======================================================================
# Best complex-scalar alignment
#
# Mode function has arbitrary:
#
# amplitude
# sign
# complex phase
#
# Therefore:
#
# alpha =
# (candidate^H reference)
# /
# (candidate^H candidate)
# ======================================================================
def compare_shapes(
    reference,
    candidate,
):

    reference = np.asarray(
        reference,
        dtype=np.complex128,
    )

    candidate = np.asarray(
        candidate,
        dtype=np.complex128,
    )


    denominator = np.vdot(
        candidate,
        candidate,
    )


    if (
        abs(
            denominator
        )
        <= 1e-15
    ):

        raise FloatingPointError(
            "Candidate mode energy = 0"
        )


    alpha = (
        np.vdot(
            candidate,
            reference,
        )
        /
        denominator
    )


    aligned = (
        alpha
        *
        candidate
    )


    mismatch = float(
        np.linalg.norm(
            reference
            -
            aligned
        )
        /
        max(
            np.linalg.norm(
                reference
            ),
            1e-15,
        )
    )


    correlation = float(
        abs(
            np.vdot(
                reference,
                candidate,
            )
        )
        /
        max(
            np.linalg.norm(
                reference
            )
            *
            np.linalg.norm(
                candidate
            ),
            1e-15,
        )
    )


    return {
        "alpha": (
            alpha
        ),

        "aligned": (
            aligned
        ),

        "mismatch": (
            mismatch
        ),

        "correlation": (
            correlation
        ),
    }


# ======================================================================
# Lossless acoustic half-space
#
# q_b =
#
# sqrt(k^2 - (omega/cb)^2)
# ======================================================================
def bottom_decay(
    horizontal_k,
):

    bottom_bulk_k = (
        OMEGA
        /
        BOTTOM_CP
    )


    q = np.sqrt(
        horizontal_k ** 2
        -
        bottom_bulk_k ** 2
        +
        0j
    )


    if np.real(q) < 0:

        q = -q


    return q


# ======================================================================
# 2nd-order backward derivative
# ======================================================================
def derivative_bottom_2nd(
    psi,
    dz,
):

    return (

        3.0
        *
        psi[-1]

        -

        4.0
        *
        psi[-2]

        +

        psi[-3]

    ) / (
        2.0
        *
        dz
    )


# ======================================================================
# 4th-order backward derivative
#
# f'(x_N)
# =
# (25f_N -48f_{N-1} +36f_{N-2}
#  -16f_{N-3} +3f_{N-4})/(12h)
# ======================================================================
def derivative_bottom_4th(
    psi,
    dz,
):

    return (

        25.0
        *
        psi[-1]

        -

        48.0
        *
        psi[-2]

        +

        36.0
        *
        psi[-3]

        -

        16.0
        *
        psi[-4]

        +

        3.0
        *
        psi[-5]

    ) / (
        12.0
        *
        dz
    )


# ======================================================================
# Robin residual
#
# dpsi/dz
# +
# rho_w/rho_b * q_b * psi(H)
# =
# 0
# ======================================================================
def calculate_bottom_residual(
    psi,
    horizontal_k,
    dz,
    derivative_order,
):

    if derivative_order == 2:

        derivative = (
            derivative_bottom_2nd(
                psi,
                dz,
            )
        )


    elif derivative_order == 4:

        derivative = (
            derivative_bottom_4th(
                psi,
                dz,
            )
        )


    else:

        raise ValueError(
            derivative_order
        )


    q = bottom_decay(
        horizontal_k
    )


    bottom_term = (

        WATER_DENSITY
        /
        BOTTOM_DENSITY

        *

        q

        *

        psi[-1]
    )


    residual = (
        derivative
        +
        bottom_term
    )


    normalized = float(

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
                bottom_term
            )

            +

            1e-15
        )
    )


    return {
        "derivative": (
            derivative
        ),

        "bottom_term": (
            bottom_term
        ),

        "residual": (
            residual
        ),

        "normalized": (
            normalized
        ),
    }


# ======================================================================
# MAIN
# ======================================================================
def main():

    print(
        "=" * 170
    )

    print(
        "HIGH-ORDER BOTTOM / INTERIOR DISCRETIZATION VALIDATION"
    )

    print(
        "=" * 170
    )


    # ==================================================================
    # Read KRAKEN MOD
    # ==================================================================
    mod = (
        KRAKEN_IO.parse_kraken_mod(
            KRAKEN_MOD_PATH
        )
    )


    receiver = (
        KRAKEN_IO.extract_integer_receiver_modes(
            mod
        )
    )


    receiver_modes = np.asarray(
        receiver[
            "receiver_mode_matrix"
        ][
            :,
            :NUMBER_OF_MODES
        ],
        dtype=np.complex128,
    )


    complex_k = np.asarray(
        mod[
            "complex_wavenumbers"
        ][
            :NUMBER_OF_MODES
        ],
        dtype=np.complex128,
    )


    true_k = np.asarray(
        np.real(
            complex_k
        ),
        dtype=np.float64,
    )


    # ==================================================================
    # KRAKEN z = 0,...,40
    # ==================================================================
    surface_row = np.zeros(
        (
            1,
            NUMBER_OF_MODES,
        ),
        dtype=np.complex128,
    )


    kraken_modes = np.vstack(
        [
            surface_row,
            receiver_modes,
        ]
    )


    # normalize each column
    for mode_index in range(
        NUMBER_OF_MODES
    ):

        norm = np.linalg.norm(
            kraken_modes[
                :,
                mode_index
            ]
        )

        kraken_modes[
            :,
            mode_index
        ] /= max(
            norm,
            1e-15,
        )


    print(
        "\nKRAKEN real horizontal wavenumbers:"
    )


    for mode_index in range(
        NUMBER_OF_MODES
    ):

        print(
            f"  k{mode_index + 1} = "
            f"{true_k[mode_index]:.12f}"
        )


    # ==================================================================
    # Run all variants
    # ==================================================================
    mode_rows = []

    bottom_rows = []


    for solver_name in SOLVERS:

        for dz in DZ_VALUES:

            print(
                "\n"
                + "-" * 150
            )

            print(
                f"Solver={solver_name} | "
                f"dz={dz:.3f} m"
            )

            print(
                "-" * 150
            )


            for mode_index in range(
                NUMBER_OF_MODES
            ):

                mode_number = (
                    mode_index + 1
                )


                k = (
                    true_k[
                        mode_index
                    ]
                )


                fine_depths, fine_psi = (
                    solve_mode(
                        solver_name=(
                            solver_name
                        ),
                        horizontal_k=(
                            k
                        ),
                        dz=(
                            dz
                        ),
                    )
                )


                sampled_depths, sampled_psi = (
                    sample_integer_depths(
                        fine_depths,
                        fine_psi,
                    )
                )


                comparison = (
                    compare_shapes(
                        reference=(
                            kraken_modes[
                                :,
                                mode_index
                            ]
                        ),
                        candidate=(
                            sampled_psi
                        ),
                    )
                )


                mode_rows.append(
                    {
                        "solver": (
                            solver_name
                        ),

                        "dz_m": (
                            dz
                        ),

                        "mode": (
                            mode_number
                        ),

                        "kraken_k": (
                            k
                        ),

                        "shape_mismatch": (
                            comparison[
                                "mismatch"
                            ]
                        ),

                        "shape_correlation": (
                            comparison[
                                "correlation"
                            ]
                        ),
                    }
                )


                for derivative_order in [
                    2,
                    4,
                ]:

                    bottom = (
                        calculate_bottom_residual(
                            psi=(
                                fine_psi
                            ),
                            horizontal_k=(
                                k
                            ),
                            dz=(
                                dz
                            ),
                            derivative_order=(
                                derivative_order
                            ),
                        )
                    )


                    bottom_rows.append(
                        {
                            "solver": (
                                solver_name
                            ),

                            "dz_m": (
                                dz
                            ),

                            "mode": (
                                mode_number
                            ),

                            "bottom_derivative_order": (
                                derivative_order
                            ),

                            "normalized_bottom_residual": (
                                bottom[
                                    "normalized"
                                ]
                            ),

                            "residual_real": float(
                                np.real(
                                    bottom[
                                        "residual"
                                    ]
                                )
                            ),

                            "residual_imag": float(
                                np.imag(
                                    bottom[
                                        "residual"
                                    ]
                                )
                            ),
                        }
                    )


                print(
                    f"  Mode {mode_number} | "
                    f"shape mismatch="
                    f"{comparison['mismatch']:.6e} | "
                    f"corr="
                    f"{comparison['correlation']:.9f}"
                )


    # ==================================================================
    # Save raw modewise
    # ==================================================================
    save_csv(
        MODEWISE_CSV,
        mode_rows,
    )


    save_csv(
        BOTTOM_CSV,
        bottom_rows,
    )


    # ==================================================================
    # Summary per solver/dz
    # ==================================================================
    summary_rows = []


    for solver_name in SOLVERS:

        for dz in DZ_VALUES:

            subset = [
                row
                for row
                in mode_rows
                if (
                    row[
                        "solver"
                    ]
                    ==
                    solver_name
                    and
                    float(
                        row[
                            "dz_m"
                        ]
                    )
                    ==
                    float(
                        dz
                    )
                )
            ]


            low = np.asarray(
                [
                    row[
                        "shape_mismatch"
                    ]
                    for row
                    in subset
                    if row[
                        "mode"
                    ]
                    <= 3
                ],
                dtype=np.float64,
            )


            middle = np.asarray(
                [
                    row[
                        "shape_mismatch"
                    ]
                    for row
                    in subset
                    if (
                        row[
                            "mode"
                        ]
                        >= 4
                        and
                        row[
                            "mode"
                        ]
                        <= 5
                    )
                ],
                dtype=np.float64,
            )


            high = np.asarray(
                [
                    row[
                        "shape_mismatch"
                    ]
                    for row
                    in subset
                    if row[
                        "mode"
                    ]
                    >= 6
                ],
                dtype=np.float64,
            )


            all_values = np.asarray(
                [
                    row[
                        "shape_mismatch"
                    ]
                    for row
                    in subset
                ],
                dtype=np.float64,
            )


            residual2 = np.asarray(
                [
                    row[
                        "normalized_bottom_residual"
                    ]
                    for row
                    in bottom_rows
                    if (
                        row[
                            "solver"
                        ]
                        ==
                        solver_name
                        and
                        float(
                            row[
                                "dz_m"
                            ]
                        )
                        ==
                        float(
                            dz
                        )
                        and
                        row[
                            "bottom_derivative_order"
                        ]
                        ==
                        2
                        and
                        row[
                            "mode"
                        ]
                        >= 6
                    )
                ],
                dtype=np.float64,
            )


            residual4 = np.asarray(
                [
                    row[
                        "normalized_bottom_residual"
                    ]
                    for row
                    in bottom_rows
                    if (
                        row[
                            "solver"
                        ]
                        ==
                        solver_name
                        and
                        float(
                            row[
                                "dz_m"
                            ]
                        )
                        ==
                        float(
                            dz
                        )
                        and
                        row[
                            "bottom_derivative_order"
                        ]
                        ==
                        4
                        and
                        row[
                            "mode"
                        ]
                        >= 6
                    )
                ],
                dtype=np.float64,
            )


            summary_rows.append(
                {
                    "solver": (
                        solver_name
                    ),

                    "dz_m": (
                        dz
                    ),

                    "mean_mismatch_modes_1_3": float(
                        np.mean(
                            low
                        )
                    ),

                    "mean_mismatch_modes_4_5": float(
                        np.mean(
                            middle
                        )
                    ),

                    "mean_mismatch_modes_6_8": float(
                        np.mean(
                            high
                        )
                    ),

                    "max_mismatch_modes_6_8": float(
                        np.max(
                            high
                        )
                    ),

                    "mean_mismatch_modes_1_8": float(
                        np.mean(
                            all_values
                        )
                    ),

                    "high_order_bottom_residual_2nd_mean": float(
                        np.mean(
                            residual2
                        )
                    ),

                    "high_order_bottom_residual_4th_mean": float(
                        np.mean(
                            residual4
                        )
                    ),
                }
            )


    save_csv(
        SUMMARY_CSV,
        summary_rows,
    )


    # ==================================================================
    # Baseline:
    #
    # current physics approximately FD2 dz=1
    # ==================================================================
    baseline_row = next(
        row
        for row
        in summary_rows
        if (
            row[
                "solver"
            ]
            ==
            "FD2"
            and
            float(
                row[
                    "dz_m"
                ]
            )
            ==
            1.0
        )
    )


    baseline_high = float(
        baseline_row[
            "mean_mismatch_modes_6_8"
        ]
    )


    # ==================================================================
    # Best high-order variant
    # ==================================================================
    best_row = min(
        summary_rows,
        key=lambda row:
        row[
            "mean_mismatch_modes_6_8"
        ],
    )


    best_high = float(
        best_row[
            "mean_mismatch_modes_6_8"
        ]
    )


    improvement = (
        baseline_high
        -
        best_high
    ) / max(
        baseline_high,
        1e-15,
    )


    best_rows = [
        {
            "baseline_solver": (
                "FD2"
            ),

            "baseline_dz_m": (
                1.0
            ),

            "baseline_high_order_mean_mismatch": (
                baseline_high
            ),

            "best_solver": (
                best_row[
                    "solver"
                ]
            ),

            "best_dz_m": (
                best_row[
                    "dz_m"
                ]
            ),

            "best_high_order_mean_mismatch": (
                best_high
            ),

            "relative_high_order_improvement": (
                improvement
            ),

            "best_high_order_max_mismatch": (
                best_row[
                    "max_mismatch_modes_6_8"
                ]
            ),

            "best_bottom_residual_2nd_mean": (
                best_row[
                    "high_order_bottom_residual_2nd_mean"
                ]
            ),

            "best_bottom_residual_4th_mean": (
                best_row[
                    "high_order_bottom_residual_4th_mean"
                ]
            ),
        }
    ]


    save_csv(
        BEST_CSV,
        best_rows,
    )


    # ==================================================================
    # Figures
    # ==================================================================

    # --------------------------------------------------------------
    # Figure 1:
    # mode index -> mismatch
    # --------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(
            9.0,
            5.8,
        )
    )


    for solver_name in SOLVERS:

        for dz in DZ_VALUES:

            subset = [
                row
                for row
                in mode_rows
                if (
                    row[
                        "solver"
                    ]
                    ==
                    solver_name
                    and
                    float(
                        row[
                            "dz_m"
                        ]
                    )
                    ==
                    float(
                        dz
                    )
                )
            ]


            subset.sort(
                key=lambda row:
                row[
                    "mode"
                ]
            )


            ax.plot(
                [
                    row[
                        "mode"
                    ]
                    for row
                    in subset
                ],
                [
                    row[
                        "shape_mismatch"
                    ]
                    for row
                    in subset
                ],
                marker="o",
                label=(
                    f"{solver_name}, "
                    f"dz={dz}"
                ),
            )


    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        "Mode index"
    )

    ax.set_ylabel(
        "Relative mode-shape mismatch"
    )

    ax.set_title(
        "KRAKEN vs High-Order Differentiable Modal Models"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        ncol=2
    )

    fig.tight_layout()


    fig.savefig(
        FIGURE_DIR
        / "mode_shape_mismatch_all_variants.png",
        dpi=220,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


    # --------------------------------------------------------------
    # Figure 2:
    # dz -> mean high-order mismatch
    # --------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(
            7.5,
            5.3,
        )
    )


    for solver_name in SOLVERS:

        subset = [
            row
            for row
            in summary_rows
            if row[
                "solver"
            ]
            ==
            solver_name
        ]


        subset.sort(
            key=lambda row:
            row[
                "dz_m"
            ],
            reverse=True,
        )


        ax.plot(
            [
                row[
                    "dz_m"
                ]
                for row
                in subset
            ],
            [
                row[
                    "mean_mismatch_modes_6_8"
                ]
                for row
                in subset
            ],
            marker="o",
            label=(
                solver_name
            ),
        )


    ax.set_xscale(
        "log"
    )

    ax.set_yscale(
        "log"
    )

    ax.invert_xaxis()

    ax.set_xlabel(
        "Depth step dz (m)"
    )

    ax.set_ylabel(
        "Mean mismatch, modes 6-8"
    )

    ax.set_title(
        "High-Order Mode Accuracy vs Discretization"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()


    fig.savefig(
        FIGURE_DIR
        / "high_order_mismatch_vs_dz.png",
        dpi=220,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


    # --------------------------------------------------------------
    # Figure 3:
    # bottom derivative residual
    # --------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(
            8.0,
            5.5,
        )
    )


    for solver_name in SOLVERS:

        subset = [
            row
            for row
            in summary_rows
            if row[
                "solver"
            ]
            ==
            solver_name
        ]


        subset.sort(
            key=lambda row:
            row[
                "dz_m"
            ],
            reverse=True,
        )


        ax.plot(
            [
                row[
                    "dz_m"
                ]
                for row
                in subset
            ],
            [
                row[
                    "high_order_bottom_residual_2nd_mean"
                ]
                for row
                in subset
            ],
            marker="o",
            label=(
                f"{solver_name} + "
                "2nd bottom derivative"
            ),
        )


        ax.plot(
            [
                row[
                    "dz_m"
                ]
                for row
                in subset
            ],
            [
                row[
                    "high_order_bottom_residual_4th_mean"
                ]
                for row
                in subset
            ],
            marker="o",
            linestyle="--",
            label=(
                f"{solver_name} + "
                "4th bottom derivative"
            ),
        )


    ax.set_xscale(
        "log"
    )

    ax.set_yscale(
        "log"
    )

    ax.invert_xaxis()

    ax.set_xlabel(
        "Depth step dz (m)"
    )

    ax.set_ylabel(
        "Mean normalized bottom residual, modes 6-8"
    )

    ax.set_title(
        "Bottom Boundary Accuracy"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        ncol=2
    )

    fig.tight_layout()


    fig.savefig(
        FIGURE_DIR
        / "bottom_residual_vs_dz.png",
        dpi=220,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


    # ==================================================================
    # Console summary
    # ==================================================================
    print(
        "\n\n"
        + "=" * 180
    )

    print(
        "DISCRETIZATION SUMMARY"
    )

    print(
        "=" * 180
    )


    for row in summary_rows:

        print(
            f"{row['solver']:<9} | "
            f"dz={row['dz_m']:<6.3f} | "
            f"M1-3="
            f"{row['mean_mismatch_modes_1_3']:.6e} | "
            f"M4-5="
            f"{row['mean_mismatch_modes_4_5']:.6e} | "
            f"M6-8="
            f"{row['mean_mismatch_modes_6_8']:.6e} | "
            f"max6-8="
            f"{row['max_mismatch_modes_6_8']:.6e} | "
            f"R2="
            f"{row['high_order_bottom_residual_2nd_mean']:.6e} | "
            f"R4="
            f"{row['high_order_bottom_residual_4th_mean']:.6e}"
        )


    print(
        "\n"
        + "=" * 180
    )

    print(
        "BEST VARIANT"
    )

    print(
        "=" * 180
    )


    print(
        f"Baseline: "
        f"FD2 dz=1.0 | "
        f"high-order mismatch="
        f"{baseline_high:.6e}"
    )


    print(
        f"Best: "
        f"{best_row['solver']} "
        f"dz={best_row['dz_m']} | "
        f"high-order mismatch="
        f"{best_high:.6e}"
    )


    print(
        f"Relative improvement = "
        f"{100.0 * improvement:+.2f}%"
    )


    print(
        "\n"
        + "=" * 180
    )

    print(
        "DECISION"
    )

    print(
        "=" * 180
    )


    if improvement >= 0.30:

        print(
            "STRONG PASS:"
        )

        print(
            "High-order differentiable modal representation "
            "reduces modes 6-8 mismatch by >=30%."
        )

        print(
            "This is worth integrating into the final "
            "realistic-bottom Mode-FRINET."
        )


    elif improvement >= 0.10:

        print(
            "PASS / PARTIAL:"
        )

        print(
            "Higher-order numerical physics provides a meaningful "
            "10-30% improvement."
        )

        print(
            "Next test whether the reduction transfers to K=5/K=8 "
            "wavenumber estimation."
        )


    else:

        print(
            "NO STRONG DISCRETIZATION BENEFIT:"
        )

        print(
            "Finer/higher-order numerical integration does not "
            "substantially remove KRAKEN high-order mismatch."
        )

        print(
            "Do not add expensive high-order discretization to "
            "Mode-FRINET yet."
        )

        print(
            "The next target should be the actual half-space/interface "
            "modal physics rather than numerical order alone."
        )


    print(
        "\nOutputs:"
    )

    print(
        MODEWISE_CSV
    )

    print(
        SUMMARY_CSV
    )

    print(
        BOTTOM_CSV
    )

    print(
        BEST_CSV
    )

    print(
        FIGURE_DIR
    )


if __name__ == "__main__":
    main()
