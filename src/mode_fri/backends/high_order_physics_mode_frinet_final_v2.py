# -*- coding: utf-8 -*-
"""
High-Order Physics Mode-FRINET V2
======================================================================

核心改进
----------------------------------------------------------------------
在High-Order Numerov Mode-FRINET基础上，引入：

    Physics-Stratified Continuous Trust Regions

不再：

    M个连续原子
        -> 整个[k_min, k_max]随机初始化
        -> 训练过程中可跨越任意模态阶次

而是：

    mode-order physical cells
        ↓
    每个cell分配若干连续原子
        ↓
    每个原子始终在所属cell内部连续优化
        ↓
    Numerov4 differentiable physics
        ↓
    group sparse multi-snapshot fitting
        ↓
    mode-order beam selection

重要：
----------------------------------------------------------------------
这不是Grid。

每个k仍然是连续变量：

    k_j = lower_j
          + (upper_j-lower_j) * sigmoid(theta_j)

theta_j通过梯度训练。

物理cell只限定“模态阶次可信域”，
并没有指定cell中的具体波数答案。

======================================================================
本次Smoke Test
----------------------------------------------------------------------
只测：

    K = 8
    SNR = 15, 30 dB
    trial = 1
    restart = 1

旧Numerov QUICK在这两组上失败原因：

    EMPTY_CELL_MODE_1

本次第一目标：

    彻底消除EMPTY_CELL。

第二目标：

    看K=8 high-SNR raw estimation是否真正受益于
    Numerov high-order physics。

======================================================================
依赖
----------------------------------------------------------------------

experiments/high_order_physics_mode_frinet_final.py

该文件继续提供已经验证过的：

    Numerov solver
    Robin roots
    KRAKEN parser
    loss
    selector
    safe refinement
    data generation
"""

from __future__ import annotations

import csv
import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch


# ======================================================================
# PATH
# ======================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]

BASE_SCRIPT = (
    Path(__file__).resolve().parent
    / "high_order_physics_mode_frinet_final.py"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "high_order_physics_mode_frinet_final_v2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ======================================================================
# LOAD BASE
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


BASE = load_module(
    BASE_SCRIPT,
    "ho_mode_frinet_base_for_v2",
)


# ======================================================================
# SMOKE TEST
# ======================================================================
TARGET_MODE_COUNT = 8

SNR_VALUES_DB = [
    15.0,
    30.0,
]

NUMBER_OF_TRIALS = 1

NUMBER_OF_RESTARTS = 1

RUN_SUFFIX = "stratified_smoke"


# ======================================================================
# 保持原Mode-FRINET配置
# ======================================================================
NUMBER_OF_ATOMS = BASE.NUMBER_OF_ATOMS

NUMBER_OF_SNAPSHOTS = BASE.NUMBER_OF_SNAPSHOTS

BASE_RANDOM_SEED = BASE.BASE_RANDOM_SEED


# ======================================================================
# 分层连续原子配置
# ======================================================================

# 初始化时不放在sigmoid极端位置。
#
# 每个cell内部仍然覆盖90%的宽度。
INITIAL_CELL_MARGIN_FRACTION = 0.05


# ======================================================================
# OUTPUT
# ======================================================================
ATOM_PLAN_CSV = (
    OUTPUT_DIR
    / "stratified_atom_plan.csv"
)

RESTART_CSV = (
    OUTPUT_DIR
    / f"ho_v2_restarts_{RUN_SUFFIX}.csv"
)

TRIAL_CSV = (
    OUTPUT_DIR
    / f"ho_v2_trials_{RUN_SUFFIX}.csv"
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / f"ho_v2_summary_{RUN_SUFFIX}.csv"
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
# 为每个模态cell分配连续原子
# ======================================================================
def allocate_atoms_to_cells(
    physical_roots: np.ndarray,
    number_of_modes: int,
    random_seed: int,
):

    if (
        NUMBER_OF_ATOMS
        <
        number_of_modes
    ):

        raise RuntimeError(
            "\nNUMBER_OF_ATOMS必须>=目标模态数。\n"
            f"M={NUMBER_OF_ATOMS}, K={number_of_modes}"
        )


    training_k_min = (
        BASE.K_MIN_BY_MODE_COUNT[
            number_of_modes
        ]
    )


    cells = (
        BASE.build_mode_cells(

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
    )


    # ------------------------------------------------------------------
    # 先保证每个cell均匀获得原子。
    #
    # K=8,M=24：
    #
    #     3 atoms / cell
    # ------------------------------------------------------------------
    base_count = (
        NUMBER_OF_ATOMS
        //
        number_of_modes
    )

    remainder = (
        NUMBER_OF_ATOMS
        %
        number_of_modes
    )


    counts = np.full(
        number_of_modes,
        base_count,
        dtype=np.int64,
    )


    # ------------------------------------------------------------------
    # 如果不能整除：
    # 额外原子优先给更宽的cell。
    #
    # 这只增加搜索覆盖，不使用truth。
    # ------------------------------------------------------------------
    widths = np.asarray(

        [
            cell[
                "upper"
            ]
            -
            cell[
                "lower"
            ]

            for cell
            in cells
        ],

        dtype=np.float64,
    )


    if remainder > 0:

        extra_order = np.argsort(
            widths
        )[::-1]

        for index in (
            extra_order[
                :remainder
            ]
        ):

            counts[
                index
            ] += 1


    rng = np.random.default_rng(
        random_seed
    )


    lower_bounds = []

    upper_bounds = []

    initial_wavenumbers = []

    cell_ids = []

    plan_rows = []


    atom_global_index = 0


    for cell_index, cell in enumerate(
        cells
    ):

        lower = float(
            cell[
                "lower"
            ]
        )

        upper = float(
            cell[
                "upper"
            ]
        )

        width = (
            upper
            -
            lower
        )


        if width <= 0.0:

            raise RuntimeError(
                f"Mode {cell_index + 1} cell宽度异常"
            )


        number_of_cell_atoms = int(
            counts[
                cell_index
            ]
        )


        # --------------------------------------------------------------
        # continuous random positions
        #
        # 仍然不是grid。
        # --------------------------------------------------------------
        u = rng.uniform(

            INITIAL_CELL_MARGIN_FRACTION,

            1.0
            -
            INITIAL_CELL_MARGIN_FRACTION,

            size=(
                number_of_cell_atoms,
            ),
        )


        initial_k = (
            lower
            +
            width
            *
            u
        )


        for local_index in range(
            number_of_cell_atoms
        ):

            atom_global_index += 1


            lower_bounds.append(
                lower
            )

            upper_bounds.append(
                upper
            )

            initial_wavenumbers.append(
                float(
                    initial_k[
                        local_index
                    ]
                )
            )

            cell_ids.append(
                cell_index
            )


            plan_rows.append(
                {
                    "atom": (
                        atom_global_index
                    ),

                    "mode_cell": (
                        cell_index + 1
                    ),

                    "cell_lower": (
                        lower
                    ),

                    "cell_upper": (
                        upper
                    ),

                    "physical_root": (
                        cell[
                            "root"
                        ]
                    ),

                    "initial_k": float(
                        initial_k[
                            local_index
                        ]
                    ),
                }
            )


    lower_bounds = np.asarray(
        lower_bounds,
        dtype=np.float64,
    )

    upper_bounds = np.asarray(
        upper_bounds,
        dtype=np.float64,
    )

    initial_wavenumbers = np.asarray(
        initial_wavenumbers,
        dtype=np.float64,
    )

    cell_ids = np.asarray(
        cell_ids,
        dtype=np.int64,
    )


    if (
        initial_wavenumbers.size
        != NUMBER_OF_ATOMS
    ):

        raise RuntimeError(
            "原子分配总数异常"
        )


    return {
        "cells": (
            cells
        ),

        "counts": (
            counts
        ),

        "lower_bounds": (
            lower_bounds
        ),

        "upper_bounds": (
            upper_bounds
        ),

        "initial_wavenumbers": (
            initial_wavenumbers
        ),

        "cell_ids": (
            cell_ids
        ),

        "plan_rows": (
            plan_rows
        ),
    }


# ======================================================================
# Stratified High-Order Mode-FRINET
#
# 继承原Numerov模型，只改变k参数化。
# ======================================================================
class StratifiedHighOrderModeFRINET(
    BASE.HighOrderModeFRINET
):

    def __init__(
        self,
        number_of_atoms: int,
        number_of_snapshots: int,
        atom_lower_bounds: np.ndarray,
        atom_upper_bounds: np.ndarray,
        initial_wavenumbers: np.ndarray,
        atom_cell_ids: np.ndarray,
        random_seed: int,
    ):

        atom_lower_bounds = np.asarray(
            atom_lower_bounds,
            dtype=np.float64,
        )

        atom_upper_bounds = np.asarray(
            atom_upper_bounds,
            dtype=np.float64,
        )

        initial_wavenumbers = np.asarray(
            initial_wavenumbers,
            dtype=np.float64,
        )

        atom_cell_ids = np.asarray(
            atom_cell_ids,
            dtype=np.int64,
        )


        if not (
            atom_lower_bounds.size
            ==
            atom_upper_bounds.size
            ==
            initial_wavenumbers.size
            ==
            atom_cell_ids.size
            ==
            number_of_atoms
        ):

            raise ValueError(
                "Stratified atom arrays维度不一致"
            )


        # --------------------------------------------------------------
        # 先初始化父类Numerov物理结构。
        #
        # 父类global k bounds之后不会再被get_wavenumbers使用。
        # --------------------------------------------------------------
        super().__init__(

            number_of_atoms=(
                number_of_atoms
            ),

            number_of_snapshots=(
                number_of_snapshots
            ),

            k_min=float(
                np.min(
                    atom_lower_bounds
                )
            ),

            k_max=float(
                np.max(
                    atom_upper_bounds
                )
            ),

            random_seed=(
                random_seed
            ),
        )


        self.register_buffer(

            "atom_lower_bounds",

            torch.as_tensor(
                atom_lower_bounds,
                dtype=torch.float64,
            ),
        )


        self.register_buffer(

            "atom_upper_bounds",

            torch.as_tensor(
                atom_upper_bounds,
                dtype=torch.float64,
            ),
        )


        self.register_buffer(

            "atom_cell_ids",

            torch.as_tensor(
                atom_cell_ids,
                dtype=torch.long,
            ),
        )


        # --------------------------------------------------------------
        # 将initial k映射回每个cell自己的logit坐标。
        # --------------------------------------------------------------
        normalized_position = (

            (
                initial_wavenumbers
                -
                atom_lower_bounds
            )

            /

            (
                atom_upper_bounds
                -
                atom_lower_bounds
            )
        )


        normalized_position = np.clip(
            normalized_position,
            1.0e-5,
            1.0
            -
            1.0e-5,
        )


        initial_raw = np.log(

            normalized_position

            /

            (
                1.0
                -
                normalized_position
            )
        )


        with torch.no_grad():

            self.k_raw.copy_(

                torch.as_tensor(
                    initial_raw,
                    dtype=torch.float64,
                )
            )


    # ------------------------------------------------------------------
    # 每个原子自己的continuous trust region
    # ------------------------------------------------------------------
    def get_wavenumbers(
        self,
    ):

        sigmoid = torch.sigmoid(
            self.k_raw
        )


        return (

            self.atom_lower_bounds

            +

            (
                self.atom_upper_bounds
                -
                self.atom_lower_bounds
            )

            *

            sigmoid
        )


# ======================================================================
# Stage-3 active mask
#
# 旧版：
#     全局Top-8。
#
# K=8时可能导致某个cell没有可继续移动的原子。
#
# V2：
#     每个cell至少保留1个最高activity原子，
#     剩余名额再全局按activity补充。
# ======================================================================
def build_cell_aware_stage3_mask(
    activities: torch.Tensor,
    cell_ids: torch.Tensor,
    number_of_modes: int,
):

    number_of_atoms = int(
        activities.numel()
    )


    top_count = min(

        number_of_atoms,

        max(
            BASE.STAGE3_TOP_ATOMS,
            number_of_modes,
        ),
    )


    mandatory = []


    for cell_index in range(
        number_of_modes
    ):

        indices = torch.where(
            cell_ids
            ==
            cell_index
        )[0]


        if indices.numel() == 0:

            raise RuntimeError(
                f"Stage3发现空cell {cell_index + 1}"
            )


        local_activity = (
            activities[
                indices
            ]
        )


        best_local = indices[

            torch.argmax(
                local_activity
            )
        ]


        mandatory.append(
            int(
                best_local.item()
            )
        )


    selected = list(
        dict.fromkeys(
            mandatory
        )
    )


    global_order = torch.argsort(
        activities,
        descending=True,
    )


    for index in global_order:

        value = int(
            index.item()
        )


        if value not in selected:

            selected.append(
                value
            )


        if (
            len(
                selected
            )
            >=
            top_count
        ):

            break


    mask = torch.zeros(
        number_of_atoms,
        dtype=torch.float64,
        device=activities.device,
    )


    mask[
        selected
    ] = 1.0


    return (
        mask,
        selected,
    )


# ======================================================================
# Train
# ======================================================================
def train_one_restart_stratified(
    noisy_data: np.ndarray,
    physical_roots: np.ndarray,
    number_of_modes: int,
    random_seed: int,
):

    torch.manual_seed(
        random_seed
    )

    np.random.seed(
        random_seed
    )


    # ==================================================================
    # Continuous stratified atom plan
    # ==================================================================
    atom_plan = (
        allocate_atoms_to_cells(

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


    model = StratifiedHighOrderModeFRINET(

        number_of_atoms=(
            NUMBER_OF_ATOMS
        ),

        number_of_snapshots=(
            noisy_data.shape[1]
        ),

        atom_lower_bounds=(
            atom_plan[
                "lower_bounds"
            ]
        ),

        atom_upper_bounds=(
            atom_plan[
                "upper_bounds"
            ]
        ),

        initial_wavenumbers=(
            atom_plan[
                "initial_wavenumbers"
            ]
        ),

        atom_cell_ids=(
            atom_plan[
                "cell_ids"
            ]
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
    # Warmup: coefficient only
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
            BASE.COEFFICIENT_LR
        ),
    )


    for _ in range(
        BASE.WARMUP_STEPS
    ):

        optimizer.zero_grad()

        output = model()


        losses = BASE.calculate_training_loss(

            output=(
                output
            ),

            target=(
                normalized_data
            ),

            group_lambda=(
                BASE.GROUP_LAMBDA_STAGE1
            ),
        )


        warmup_loss = (

            losses[
                "fit"
            ]

            +

            BASE.GROUP_LAMBDA_STAGE1

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

            BASE.GRADIENT_CLIP,
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
                    BASE.STAGE1_K_LR
                ),
            },

            {
                "params": [
                    model.coefficient_real,
                    model.coefficient_imag,
                ],

                "lr": (
                    BASE.COEFFICIENT_LR
                ),
            },
        ]
    )


    for _ in range(
        BASE.STAGE1_STEPS
    ):

        optimizer.zero_grad()

        output = model()


        losses = BASE.calculate_training_loss(

            output=(
                output
            ),

            target=(
                normalized_data
            ),

            group_lambda=(
                BASE.GROUP_LAMBDA_STAGE1
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
            BASE.GRADIENT_CLIP,
        )


        optimizer.step()


    # ==================================================================
    # Stage 2
    # ==================================================================
    optimizer = torch.optim.Adam(

        [

            {
                "params": [
                    model.k_raw
                ],

                "lr": (
                    BASE.STAGE2_K_LR_START
                ),
            },

            {
                "params": [
                    model.coefficient_real,
                    model.coefficient_imag,
                ],

                "lr": (
                    BASE.COEFFICIENT_LR
                ),
            },
        ]
    )


    for local_step in range(
        BASE.STAGE2_STEPS
    ):

        fraction = (

            local_step

            /

            max(
                BASE.STAGE2_STEPS - 1,
                1,
            )
        )


        k_lr = (

            BASE.STAGE2_K_LR_START

            +

            fraction

            *

            (
                BASE.STAGE2_K_LR_END
                -
                BASE.STAGE2_K_LR_START
            )
        )


        group_lambda = (

            BASE.GROUP_LAMBDA_STAGE2_START

            +

            fraction

            *

            (
                BASE.GROUP_LAMBDA_STAGE2_END
                -
                BASE.GROUP_LAMBDA_STAGE2_START
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


        losses = BASE.calculate_training_loss(

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
            BASE.GRADIENT_CLIP,
        )


        optimizer.step()


    # ==================================================================
    # Stage 3
    # ==================================================================
    with torch.no_grad():

        stage2_output = model()


        active_mask, stage3_indices = (
            build_cell_aware_stage3_mask(

                activities=(
                    stage2_output.activities
                ),

                cell_ids=(
                    model.atom_cell_ids
                ),

                number_of_modes=(
                    number_of_modes
                ),
            )
        )


    optimizer = torch.optim.Adam(

        [

            {
                "params": [
                    model.k_raw
                ],

                "lr": (
                    BASE.STAGE3_K_LR
                ),
            },

            {
                "params": [
                    model.coefficient_real,
                    model.coefficient_imag,
                ],

                "lr": (
                    BASE.COEFFICIENT_LR
                ),
            },
        ]
    )


    for _ in range(
        BASE.STAGE3_STEPS
    ):

        optimizer.zero_grad()

        output = model()


        losses = BASE.calculate_training_loss(

            output=(
                output
            ),

            target=(
                normalized_data
            ),

            group_lambda=(
                BASE.GROUP_LAMBDA_STAGE3
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
        # cell-aware active k mask
        # --------------------------------------------------------------
        if model.k_raw.grad is not None:

            model.k_raw.grad.mul_(
                active_mask
            )


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            BASE.GRADIENT_CLIP,
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


    final_wavenumbers = (
        final_output.wavenumbers
        .cpu()
        .numpy()
        .copy()
    )


    final_activities = (
        final_output.activities
        .cpu()
        .numpy()
        .copy()
    )


    final_bottom_residuals = (
        final_output.bottom_residuals
        .cpu()
        .numpy()
        .copy()
    )


    # ==================================================================
    # Verify final cell coverage
    # ==================================================================
    final_counts = []


    for cell in (
        atom_plan[
            "cells"
        ]
    ):

        count = int(
            np.sum(

                (
                    final_wavenumbers
                    >
                    cell[
                        "lower"
                    ]
                )

                &

                (
                    final_wavenumbers
                    <=
                    cell[
                        "upper"
                    ]
                )
            )
        )


        final_counts.append(
            count
        )


    return {

        "all_wavenumbers": (
            final_wavenumbers
        ),

        "all_activities": (
            final_activities
        ),

        "all_bottom_residuals": (
            final_bottom_residuals
        ),

        "runtime_seconds": float(
            runtime
        ),

        "initial_atoms_per_cell": (
            atom_plan[
                "counts"
            ].copy()
        ),

        "final_atoms_per_cell": np.asarray(
            final_counts,
            dtype=np.int64,
        ),

        "stage3_active_indices": np.asarray(
            stage3_indices,
            dtype=np.int64,
        ),

        "atom_plan": (
            atom_plan
        ),
    }


# ======================================================================
# Summary
# ======================================================================
def summarize(
    rows,
):

    output = []


    for snr_db in (
        SNR_VALUES_DB
    ):

        subset = [

            row

            for row
            in rows

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


        if not subset:
            continue


        output.append(
            {
                "K": (
                    TARGET_MODE_COUNT
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

                "raw_rmse_median": float(
                    np.median(
                        [
                            row[
                                "raw_rmse"
                            ]
                            for row
                            in subset
                        ]
                    )
                ),

                "refined_rmse_median": float(
                    np.median(
                        [
                            row[
                                "refined_rmse"
                            ]
                            for row
                            in subset
                        ]
                    )
                ),

                "raw_max_error_median": float(
                    np.median(
                        [
                            row[
                                "raw_max_error"
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
        "=" * 180
    )

    print(
        "HIGH-ORDER PHYSICS MODE-FRINET V2"
    )

    print(
        "PHYSICS-STRATIFIED CONTINUOUS TRUST REGIONS"
    )

    print(
        "=" * 180
    )

    print(
        f"K                 = {TARGET_MODE_COUNT}"
    )

    print(
        f"M                 = {NUMBER_OF_ATOMS}"
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
        f"Numerov dz        = {BASE.PHYSICS_DZ} m"
    )

    print(
        "=" * 180
    )


    # ==================================================================
    # Numerov autograd依然先验证
    # ==================================================================
    BASE.verify_high_order_autograd()


    # ==================================================================
    # Physical roots
    # ==================================================================
    physical_roots = (
        BASE.calculate_physical_roots()
    )


    print(
        "\nHigh-order physical roots:"
    )


    for index, root in enumerate(
        physical_roots[
            :TARGET_MODE_COUNT
        ],
        start=1,
    ):

        print(
            f"  r{index} = "
            f"{root:.12f}"
        )


    # ==================================================================
    # Print stratified plan
    # ==================================================================
    preview_plan = (
        allocate_atoms_to_cells(

            physical_roots=(
                physical_roots
            ),

            number_of_modes=(
                TARGET_MODE_COUNT
            ),

            random_seed=(
                BASE_RANDOM_SEED
            ),
        )
    )


    print(
        "\nPhysics-stratified atom allocation:"
    )


    for index, cell in enumerate(
        preview_plan[
            "cells"
        ]
    ):

        print(

            f"  Mode {index + 1}: "
            f"cell=("
            f"{cell['lower']:.9f}, "
            f"{cell['upper']:.9f}] | "
            f"atoms="
            f"{preview_plan['counts'][index]}"
        )


    save_csv(
        ATOM_PLAN_CSV,
        preview_plan[
            "plan_rows"
        ],
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


    target_roots = np.asarray(
        physical_roots[
            :TARGET_MODE_COUNT
        ],
        dtype=np.float64,
    )


    print(
        "\nKRAKEN truth:"
    )

    print(
        true_k
    )


    print(
        "\nHO roots:"
    )

    print(
        target_roots
    )


    print(
        "\nTruth-root errors:"
    )

    print(
        np.abs(
            true_k
            -
            target_roots
        )
    )


    # ==================================================================
    # Same K=8 trial as previous QUICK
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


    restart_rows = []

    trial_rows = []


    # ==================================================================
    # SNR
    # ==================================================================
    for snr_db in SNR_VALUES_DB:

        print(
            "\n\n"
            + "=" * 180
        )

        print(
            f"K={TARGET_MODE_COUNT} | "
            f"SNR={snr_db:+.1f} dB"
        )

        print(
            "=" * 180
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


            restart_results = []


            for restart_index in range(
                NUMBER_OF_RESTARTS
            ):

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

                    +

                    restart_index
                    *
                    1009
                )


                print(
                    f"\nTrial {trial} | "
                    f"restart {restart_index + 1} | "
                    f"seed={model_seed}",
                    flush=True,
                )


                trained = (
                    train_one_restart_stratified(

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


                print(
                    "  initial atoms/cell = "
                    f"{trained['initial_atoms_per_cell']}"
                )

                print(
                    "  final atoms/cell   = "
                    f"{trained['final_atoms_per_cell']}"
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


                if (
                    selection[
                        "success"
                    ]
                ):

                    raw_k = np.sort(
                        np.asarray(
                            selection[
                                "selected"
                            ],
                            dtype=np.float64,
                        )
                    )[::-1]


                    raw_eval = (
                        BASE.evaluate_k(
                            raw_k,
                            true_k,
                        )
                    )


                    refinement = (
                        BASE.safe_refine(

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
                        BASE.evaluate_k(
                            refined_k,
                            true_k,
                        )
                    )


                else:

                    raw_k = np.asarray(
                        [],
                        dtype=np.float64,
                    )

                    refined_k = np.asarray(
                        [],
                        dtype=np.float64,
                    )


                    raw_eval = {
                        "success": False,
                        "rmse": float("inf"),
                        "max_error": float("inf"),
                    }


                    refined_eval = (
                        raw_eval.copy()
                    )


                    refinement = {
                        "accepted": False,
                    }


                result = {

                    "selection": (
                        selection
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

                    "runtime": (
                        trained[
                            "runtime_seconds"
                        ]
                    ),
                }


                restart_results.append(
                    result
                )


                restart_rows.append(
                    {
                        "K": (
                            TARGET_MODE_COUNT
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

                        "cell_counts": (
                            ";".join(
                                str(
                                    value
                                )
                                for value
                                in selection[
                                    "cell_counts"
                                ]
                            )
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


            # ==========================================================
            # best restart，不使用truth
            # ==========================================================
            valid = [

                index

                for index, result
                in enumerate(
                    restart_results
                )

                if (
                    result[
                        "selection"
                    ][
                        "success"
                    ]

                    and

                    np.isfinite(
                        result[
                            "selection"
                        ][
                            "fit"
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
                        "selection"
                    ][
                        "fit"
                    ],
                )


                best = (
                    restart_results[
                        best_index
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

                runtime = float(
                    sum(
                        item[
                            "runtime"
                        ]
                        for item
                        in restart_results
                    )
                )


            else:

                best_index = -1

                raw_k = np.asarray(
                    [],
                    dtype=np.float64,
                )

                refined_k = np.asarray(
                    [],
                    dtype=np.float64,
                )

                raw_eval = {
                    "success": False,
                    "rmse": float("inf"),
                    "max_error": float("inf"),
                }

                refined_eval = (
                    raw_eval.copy()
                )

                runtime = float(
                    sum(
                        item[
                            "runtime"
                        ]
                        for item
                        in restart_results
                    )
                )


            print(
                "\nRESULT"
            )

            print(
                f"  raw success     = "
                f"{raw_eval['success']}"
            )

            print(
                f"  raw RMSE        = "
                f"{raw_eval['rmse']:.6e}"
            )

            print(
                f"  raw max error   = "
                f"{raw_eval['max_error']:.6e}"
            )

            print(
                f"  raw k           = "
                f"{raw_k}"
            )

            print(
                f"  raw abs errors  = "
                f"{np.abs(raw_k - true_k) if raw_k.size else []}"
            )

            print(
                f"  refined success = "
                f"{refined_eval['success']}"
            )

            print(
                f"  refined RMSE    = "
                f"{refined_eval['rmse']:.6e}"
            )

            print(
                f"  runtime         = "
                f"{runtime:.2f} s"
            )


            trial_rows.append(
                {
                    "K": (
                        TARGET_MODE_COUNT
                    ),

                    "snr_db": (
                        snr_db
                    ),

                    "trial": (
                        trial
                    ),

                    "best_restart": (
                        best_index + 1
                        if best_index >= 0
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
                        runtime
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
    # FINAL
    # ==================================================================
    summary_rows = summarize(
        trial_rows
    )


    save_csv(
        SUMMARY_CSV,
        summary_rows,
    )


    print(
        "\n\n"
        + "=" * 180
    )

    print(
        "V2 STRATIFIED SMOKE SUMMARY"
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
            f"rawRMSE="
            f"{row['raw_rmse_median']:.3e} | "
            f"max="
            f"{row['raw_max_error_median']:.3e} | "
            f"time="
            f"{row['runtime_seconds_mean']:.2f}s"
        )


    print(
        "\n"
        + "=" * 180
    )

    print(
        "判定重点"
    )

    print(
        "=" * 180
    )

    print(
        "1. initial/final atoms per cell必须全部>0。"
    )

    print(
        "2. 不应该再出现EMPTY_CELL_MODE_x。"
    )

    print(
        "3. 重点看K=8 raw max error是否<=1.5e-3。"
    )

    print(
        "4. 若K=8成功，说明Numerov高阶物理改善已经真正传递到8模态估计。"
    )

    print(
        "\nOutputs:"
    )

    print(
        ATOM_PLAN_CSV
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


if __name__ == "__main__":
    main()
