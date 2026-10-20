# -*- coding: utf-8 -*-
"""
KRAKEN .MOD Independent-Data Validation
======================================================================

目的
----------------------------------------------------------------------
验证当前最终 Mode-FRINET 在“独立 forward model”下是否仍能工作。

以前的数据链：

    KRAKEN k truth
        ↓
    自己的 FD/Robin mode matrix
        ↓
    synthetic Y
        ↓
    同一个 FD/Robin inverse model

存在 forward / inverse model 高度一致的问题。

本实验改成：

    KRAKEN .MOD
        ↓
    KRAKEN mode depth functions
        ↓
    Y_KRAKEN = Phi_KRAKEN C + N
        ↓
    Mode-FRINET (FD/Robin inverse model)
        ↓
    ORDER_CELL_JOINT
        ↓
    Safe Refinement

即：

    Forward model  = KRAKEN
    Inverse model  = differentiable FD/Robin

两者不再相同。


当前文件
----------------------------------------------------------------------
输入：

outputs/24_kraken_reference_environment/kraken_case/
    teacher_paper_grad05.mod

已确认该文件：

    record length = 328 bytes
    records       = 21
    modes         = 13
    stored depths = 41

stored depths：

    1, 2, ..., 32, 32.5, 33, ..., 40 m

其中32.5 m为source depth额外写入点。

VLA receiver depths：

    1, 2, ..., 40 m

若inverse model使用0~40 m共41点，则自动增加：

    phi(z=0) = 0

因为水面采用pressure-release boundary。


重要研究口径
----------------------------------------------------------------------
1. KRAKEN复波数的实部作为待估水平波数truth。
2. 当前Mode-FRINET只估计实水平波数。
3. KRAKEN波数虚部对应衰减，不直接作为估计目标。
4. 固定距离上的传播相位和幅度可吸收到多快照复系数中。
5. 模态函数整体幅度/符号/复相位任意，因此比较mode shape时
   使用最佳复比例因子消除尺度和相位歧义。
6. 当前验证仍然是已知环境下前三阶模态 K=3。
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

FINAL_SCRIPT_PATH = (
    Path(__file__).resolve().parent
    / "low_snr_robustness_final.py"
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
    / "kraken_mod_independent_validation"
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
# 加载最终低SNR脚本
#
# 它内部再加载实验26。
# ======================================================================
def load_final_module():

    if not FINAL_SCRIPT_PATH.exists():

        raise FileNotFoundError(
            "\n没有找到最终低SNR脚本：\n"
            f"{FINAL_SCRIPT_PATH}\n"
        )

    module_name = (
        "low_snr_final_for_kraken_mod"
    )

    spec = (
        importlib.util.spec_from_file_location(
            module_name,
            FINAL_SCRIPT_PATH,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):

        raise RuntimeError(
            "low_snr_robustness_final.py 加载失败。"
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


FINAL = load_final_module()

EXP26 = FINAL.EXP26


# ======================================================================
# QUICK / FULL
# ======================================================================
QUICK_TEST = False


if QUICK_TEST:

    SNR_VALUES_DB = [
        0.0,
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
# 模型配置
# ======================================================================
NUMBER_OF_TARGET_MODES = 3

NUMBER_OF_SNAPSHOTS = 20

NUMBER_OF_ATOMS = 24

BASE_RANDOM_SEED = 20260901


# ======================================================================
# KRAKEN .MOD格式
#
# 当前teacher_paper_grad05.mod实际结构已确认。
# ======================================================================
EXPECTED_RECORD_LENGTH_WORDS = 82

BYTES_PER_WORD = 4

RECORD_LENGTH_BYTES = (
    EXPECTED_RECORD_LENGTH_WORDS
    * BYTES_PER_WORD
)

EXPECTED_NUMBER_OF_RECORDS = 21

EXPECTED_NUMBER_OF_MODES = 13

EXPECTED_STORED_DEPTHS = 41


# ======================================================================
# 成功阈值
# ======================================================================
SUCCESS_TOLERANCE = 1.5e-3


# ======================================================================
# 输出文件
# ======================================================================
MOD_INFO_CSV = (
    RESULT_DIR
    / "kraken_mod_information.csv"
)

MODE_MISMATCH_CSV = (
    RESULT_DIR
    / "kraken_fd_mode_shape_mismatch.csv"
)

RESTART_CSV = (
    RESULT_DIR
    / (
        "kraken_mod_restart_results_"
        f"{RUN_SUFFIX}.csv"
    )
)

TRIAL_CSV = (
    RESULT_DIR
    / (
        "kraken_mod_trial_results_"
        f"{RUN_SUFFIX}.csv"
    )
)

SUMMARY_CSV = (
    RESULT_DIR
    / (
        "kraken_mod_summary_"
        f"{RUN_SUFFIX}.csv"
    )
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


def array_to_text(
    values: np.ndarray,
) -> str:

    return ";".join(
        f"{float(value):.12f}"
        for value
        in np.asarray(
            values
        ).ravel()
    )


# ======================================================================
# KRAKEN .MOD解析
# ======================================================================
def parse_kraken_mod(
    path: Path,
):

    if not path.exists():

        raise FileNotFoundError(
            "\n没有找到KRAKEN .MOD文件：\n"
            f"{path}\n"
        )


    raw = path.read_bytes()

    file_size = len(
        raw
    )


    if (
        file_size
        % RECORD_LENGTH_BYTES
        != 0
    ):

        raise RuntimeError(
            "\n.MOD文件大小不能被328整除。\n"
            f"file size = {file_size}\n"
            f"record    = {RECORD_LENGTH_BYTES}\n"
            "\n当前解析器只针对已经验证过的"
            "teacher_paper_grad05.mod格式。"
        )


    number_of_records = (
        file_size
        // RECORD_LENGTH_BYTES
    )


    if (
        number_of_records
        != EXPECTED_NUMBER_OF_RECORDS
    ):

        raise RuntimeError(
            "\n.MOD record数量与当前已验证文件不一致。\n"
            f"实际 = {number_of_records}\n"
            f"预期 = {EXPECTED_NUMBER_OF_RECORDS}\n"
        )


    def record(
        one_based_index: int,
    ) -> bytes:

        start = (
            one_based_index - 1
        ) * RECORD_LENGTH_BYTES

        end = (
            start
            + RECORD_LENGTH_BYTES
        )

        return raw[
            start:end
        ]


    # ------------------------------------------------------------------
    # Record 1
    #
    # 前4字节 = record length words = 82
    # 后面包含title。
    # ------------------------------------------------------------------
    record1 = record(
        1
    )

    lrecl = int(
        np.frombuffer(
            record1[
                :4
            ],
            dtype="<i4",
            count=1,
        )[0]
    )


    if (
        lrecl
        != EXPECTED_RECORD_LENGTH_WORDS
    ):

        raise RuntimeError(
            "\n.MOD内部record length异常。\n"
            f"实际 = {lrecl}\n"
            f"预期 = {EXPECTED_RECORD_LENGTH_WORDS}\n"
        )


    title_bytes = (
        record1[
            4:
        ]
        .split(
            b"\x00",
            1,
        )[0]
    )

    title = (
        title_bytes
        .decode(
            "ascii",
            errors="ignore",
        )
        .strip()
    )


    # ------------------------------------------------------------------
    # Record 5
    #
    # 存储KRAKEN source/receiver合并后的depth grid。
    #
    # 当前文件前41个float32有效。
    # ------------------------------------------------------------------
    stored_depths = np.frombuffer(
        record(
            5
        ),
        dtype="<f4",
        count=EXPECTED_STORED_DEPTHS,
    ).astype(
        np.float64
    )


    # ------------------------------------------------------------------
    # Record 6
    #
    # 第一个int32 = mode count。
    # ------------------------------------------------------------------
    number_of_modes = int(
        np.frombuffer(
            record(
                6
            ),
            dtype="<i4",
            count=1,
        )[0]
    )


    if (
        number_of_modes
        != EXPECTED_NUMBER_OF_MODES
    ):

        raise RuntimeError(
            "\n.MOD模态数异常。\n"
            f"实际 = {number_of_modes}\n"
            f"预期 = {EXPECTED_NUMBER_OF_MODES}\n"
        )


    # ------------------------------------------------------------------
    # Records 8 ~ 20
    #
    # 13个mode depth functions。
    #
    # 每条记录正好：
    #
    # 41 × complex64
    # =
    # 41 × 8
    # =
    # 328 bytes
    #
    # shape:
    #
    #     [stored_depth, mode]
    # ------------------------------------------------------------------
    mode_columns = []


    for mode_index in range(
        number_of_modes
    ):

        record_index = (
            8
            + mode_index
        )

        mode_values = np.frombuffer(
            record(
                record_index
            ),
            dtype="<c8",
            count=EXPECTED_STORED_DEPTHS,
        ).astype(
            np.complex128
        )

        mode_columns.append(
            mode_values
        )


    stored_mode_matrix = np.column_stack(
        mode_columns
    )


    # ------------------------------------------------------------------
    # Record 21
    #
    # 前13个complex64 = complex horizontal wavenumbers。
    # ------------------------------------------------------------------
    complex_wavenumbers = np.frombuffer(
        record(
            21
        ),
        dtype="<c8",
        count=number_of_modes,
    ).astype(
        np.complex128
    )


    return {
        "title": (
            title
        ),

        "file_size": (
            file_size
        ),

        "record_length_words": (
            lrecl
        ),

        "record_length_bytes": (
            RECORD_LENGTH_BYTES
        ),

        "number_of_records": (
            number_of_records
        ),

        "number_of_modes": (
            number_of_modes
        ),

        "stored_depths": (
            stored_depths
        ),

        "stored_mode_matrix": (
            stored_mode_matrix
        ),

        "complex_wavenumbers": (
            complex_wavenumbers
        ),
    }


# ======================================================================
# 从KRAKEN stored grid提取真正VLA receiver grid
#
# ENV：
#
#     NRD = 40
#     1.0 40.0 /
#
# 即整数深度1~40。
#
# 32.5仅来自source depth，不属于receiver VLA。
# ======================================================================
def extract_integer_receiver_modes(
    mod_data: dict,
):

    stored_depths = np.asarray(
        mod_data[
            "stored_depths"
        ],
        dtype=np.float64,
    )

    stored_modes = np.asarray(
        mod_data[
            "stored_mode_matrix"
        ],
        dtype=np.complex128,
    )


    receiver_depths = np.arange(
        1.0,
        41.0,
        1.0,
        dtype=np.float64,
    )


    receiver_rows = []


    for depth in receiver_depths:

        matches = np.where(
            np.isclose(
                stored_depths,
                depth,
                atol=1.0e-6,
                rtol=0.0,
            )
        )[0]


        if matches.size != 1:

            raise RuntimeError(
                "\n无法唯一找到KRAKEN receiver depth："
                f"{depth:.3f} m\n"
            )


        receiver_rows.append(
            int(
                matches[0]
            )
        )


    receiver_modes = (
        stored_modes[
            receiver_rows,
            :
        ]
    )


    return {
        "receiver_depths": (
            receiver_depths
        ),

        "receiver_mode_matrix": (
            receiver_modes
        ),

        "receiver_rows": np.asarray(
            receiver_rows,
            dtype=np.int64,
        ),
    }


# ======================================================================
# 将KRAKEN VLA适配到Mode-FRINET inverse-model深度维
#
# 正常预期：
#
#     inverse FD model = z=0,1,...,40 -> 41 rows
#
# KRAKEN VLA：
#
#     z=1,...,40 -> 40 rows
#
# 因为surface pressure-release：
#
#     phi(0)=0
#
# 所以补一行0。
#
# 同时保留一个40-row兼容分支，以防实验26内部只使用receiver rows。
# ======================================================================
def adapt_kraken_modes_to_inverse_grid(
    receiver_modes: np.ndarray,
    true_wavenumbers: np.ndarray,
):

    receiver_modes = np.asarray(
        receiver_modes,
        dtype=np.complex128,
    )


    inverse_probe = np.asarray(
        EXP26.build_mode_matrix(
            true_wavenumbers
        ),
        dtype=np.complex128,
    )


    inverse_rows = int(
        inverse_probe.shape[0]
    )


    if inverse_rows == 41:

        surface_row = np.zeros(
            (
                1,
                receiver_modes.shape[1],
            ),
            dtype=np.complex128,
        )

        adapted = np.vstack(
            [
                surface_row,
                receiver_modes,
            ]
        )

        adapted_depths = np.arange(
            0.0,
            41.0,
            1.0,
            dtype=np.float64,
        )

        adaptation = (
            "PREPEND_PRESSURE_RELEASE_SURFACE"
        )


    elif inverse_rows == 40:

        adapted = (
            receiver_modes.copy()
        )

        adapted_depths = np.arange(
            1.0,
            41.0,
            1.0,
            dtype=np.float64,
        )

        adaptation = (
            "RECEIVER_GRID_DIRECT"
        )


    else:

        raise RuntimeError(
            "\nKRAKEN数据与实验26 inverse model深度维不兼容。\n"
            f"inverse build_mode_matrix rows = {inverse_rows}\n"
            f"KRAKEN receiver rows           = {receiver_modes.shape[0]}\n"
            "\n当前只支持inverse rows=40或41。"
        )


    if (
        adapted.shape[0]
        != inverse_probe.shape[0]
    ):

        raise RuntimeError(
            "适配后KRAKEN矩阵仍与inverse矩阵行数不一致。"
        )


    return {
        "adapted_depths": (
            adapted_depths
        ),

        "adapted_mode_matrix": (
            adapted
        ),

        "inverse_probe": (
            inverse_probe
        ),

        "adaptation": (
            adaptation
        ),
    }


# ======================================================================
# column normalization
#
# 模态绝对幅度可以被复系数C吸收。
# ======================================================================
def normalize_columns(
    matrix: np.ndarray,
):

    matrix = np.asarray(
        matrix,
        dtype=np.complex128,
    )


    output = np.zeros_like(
        matrix
    )


    norms = []


    for column_index in range(
        matrix.shape[1]
    ):

        column = matrix[
            :,
            column_index
        ]

        norm = float(
            np.linalg.norm(
                column
            )
        )


        if (
            not np.isfinite(
                norm
            )
            or
            norm <= 1.0e-14
        ):

            raise FloatingPointError(
                "KRAKEN mode column norm异常。"
            )


        output[
            :,
            column_index
        ] = (
            column
            / norm
        )

        norms.append(
            norm
        )


    return (
        output,
        np.asarray(
            norms,
            dtype=np.float64,
        ),
    )


# ======================================================================
# 模态函数结构失配
#
# 由于mode shape存在任意complex scalar：
#
#     phi_K ~= alpha phi_FD
#
# 先求最佳alpha：
#
#     alpha =
#     (phi_FD^H phi_K) /
#     (phi_FD^H phi_FD)
#
# 再计算relative mismatch。
# ======================================================================
def calculate_mode_shape_mismatch(
    kraken_modes: np.ndarray,
    fd_modes: np.ndarray,
):

    kraken_modes = np.asarray(
        kraken_modes,
        dtype=np.complex128,
    )

    fd_modes = np.asarray(
        fd_modes,
        dtype=np.complex128,
    )


    if (
        kraken_modes.shape
        != fd_modes.shape
    ):

        raise ValueError(
            "KRAKEN/FD mode matrix shape不一致。"
        )


    rows = []


    for mode_index in range(
        kraken_modes.shape[1]
    ):

        phi_k = (
            kraken_modes[
                :,
                mode_index
            ]
        )

        phi_fd = (
            fd_modes[
                :,
                mode_index
            ]
        )


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
            max(
                abs(
                    denominator
                ),
                1.0e-14,
            )
        )


        aligned_fd = (
            alpha
            * phi_fd
        )


        relative_error = float(
            np.linalg.norm(
                phi_k
                - aligned_fd
            )
            /
            max(
                np.linalg.norm(
                    phi_k
                ),
                1.0e-14,
            )
        )


        normalized_correlation = float(
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
                1.0e-14,
            )
        )


        rows.append(
            {
                "mode": (
                    mode_index + 1
                ),

                "alpha_real": (
                    float(
                        np.real(
                            alpha
                        )
                    )
                ),

                "alpha_imag": (
                    float(
                        np.imag(
                            alpha
                        )
                    )
                ),

                "relative_shape_mismatch": (
                    relative_error
                ),

                "normalized_correlation": (
                    normalized_correlation
                ),
            }
        )


    return rows


# ======================================================================
# 生成一组clean master data
#
# Forward matrix明确来自KRAKEN。
# ======================================================================
def generate_kraken_master_data(
    kraken_mode_matrix: np.ndarray,
    coefficient_seed: int,
    noise_seed: int,
):

    kraken_mode_matrix = np.asarray(
        kraken_mode_matrix,
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
        kraken_mode_matrix
        @ coefficients
    )


    noise_rng = (
        np.random.default_rng(
            noise_seed
        )
    )


    unit_noise = (
        noise_rng.standard_normal(
            clean_data.shape
        )
        +
        1j
        * noise_rng.standard_normal(
            clean_data.shape
        )
    ) / np.sqrt(2.0)


    unit_noise = (
        unit_noise
        /
        max(
            np.linalg.norm(
                unit_noise
            ),
            1.0e-14,
        )
    )


    return {
        "coefficients": (
            coefficients
        ),

        "clean_data": (
            clean_data
        ),

        "unit_noise": (
            unit_noise
        ),
    }


def add_noise_at_snr(
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
            **
            (
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
        *
        np.log10(
            clean_norm
            /
            max(
                np.linalg.norm(
                    noise
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
# 选择best restart
# ======================================================================
def choose_best_restart(
    results: list[dict],
):

    valid = [
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


    if not valid:

        return None


    return min(
        valid,
        key=lambda index:
        float(
            results[
                index
            ][
                "restart_score"
            ]
        ),
    )


# ======================================================================
# summary
# ======================================================================
def summarize_trials(
    trial_rows: list[dict],
):

    output = []


    for snr_db in SNR_VALUES_DB:

        subset = [
            row
            for row
            in trial_rows
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


        output.append(
            {
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
                    if finite_raw.size
                    > 0
                    else float("nan")
                ),

                "refined_k_rmse_median": (
                    float(
                        np.median(
                            finite_refined
                        )
                    )
                    if finite_refined.size
                    > 0
                    else float("nan")
                ),
            }
        )


    return output


# ======================================================================
# 绘图
# ======================================================================
def plot_mode_shapes(
    depths: np.ndarray,
    kraken_modes: np.ndarray,
    fd_modes: np.ndarray,
):

    for mode_index in range(
        NUMBER_OF_TARGET_MODES
    ):

        phi_k = (
            kraken_modes[
                :,
                mode_index
            ]
        )

        phi_fd = (
            fd_modes[
                :,
                mode_index
            ]
        )


        alpha = (
            np.vdot(
                phi_fd,
                phi_k,
            )
            /
            max(
                abs(
                    np.vdot(
                        phi_fd,
                        phi_fd,
                    )
                ),
                1.0e-14,
            )
        )


        aligned_fd = (
            alpha
            * phi_fd
        )


        fig, ax = plt.subplots(
            figsize=(
                6.0,
                7.0,
            )
        )


        ax.plot(
            np.real(
                phi_k
            ),
            depths,
            marker="o",
            markersize=3,
            label="KRAKEN",
        )


        ax.plot(
            np.real(
                aligned_fd
            ),
            depths,
            linestyle="--",
            label="FD/Robin aligned",
        )


        ax.invert_yaxis()

        ax.set_xlabel(
            "Normalized modal amplitude"
        )

        ax.set_ylabel(
            "Depth (m)"
        )

        ax.set_title(
            f"Mode {mode_index + 1}: "
            "KRAKEN vs FD/Robin"
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
                f"mode_{mode_index + 1}_"
                "kraken_vs_fd.png"
            ),
            dpi=220,
            bbox_inches="tight",
        )


        plt.close(
            fig
        )


def plot_success(
    summary_rows: list[dict],
):

    x = np.asarray(
        [
            row[
                "snr_db"
            ]
            for row
            in summary_rows
        ],
        dtype=np.float64,
    )


    raw = (
        100.0
        *
        np.asarray(
            [
                row[
                    "raw_success_rate"
                ]
                for row
                in summary_rows
            ],
            dtype=np.float64,
        )
    )


    refined = (
        100.0
        *
        np.asarray(
            [
                row[
                    "refined_success_rate"
                ]
                for row
                in summary_rows
            ],
            dtype=np.float64,
        )
    )


    fig, ax = plt.subplots(
        figsize=(
            8.5,
            5.5,
        )
    )


    ax.plot(
        x,
        raw,
        marker="o",
        label="Raw",
    )


    ax.plot(
        x,
        refined,
        marker="o",
        label="Physics-refined",
    )


    ax.set_ylim(
        -5,
        105,
    )

    ax.set_xlabel(
        "SNR (dB)"
    )

    ax.set_ylabel(
        "Success rate (%)"
    )

    ax.set_title(
        "KRAKEN Independent-Data Validation"
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
            "kraken_independent_success_"
            f"{RUN_SUFFIX}.png"
        ),
        dpi=220,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


# ======================================================================
# main
# ======================================================================
def main():

    torch.set_default_dtype(
        torch.float64
    )


    # ------------------------------------------------------------------
    # 保持最终算法配置
    # ------------------------------------------------------------------
    EXP26.NUMBER_OF_SNAPSHOTS = (
        NUMBER_OF_SNAPSHOTS
    )

    EXP26.NUMBER_OF_ATOMS = (
        NUMBER_OF_ATOMS
    )


    print(
        "=" * 150
    )

    print(
        "KRAKEN .MOD INDEPENDENT-DATA VALIDATION"
    )

    print(
        "=" * 150
    )

    print(
        f"Mode               = "
        f"{RUN_SUFFIX}"
    )

    print(
        f"SNR values         = "
        f"{SNR_VALUES_DB}"
    )

    print(
        f"Trials / SNR       = "
        f"{NUMBER_OF_TRIALS}"
    )

    print(
        f"Restarts / trial   = "
        f"{NUMBER_OF_RESTARTS}"
    )

    print(
        f"Snapshots L        = "
        f"{NUMBER_OF_SNAPSHOTS}"
    )

    print(
        f"Atoms M            = "
        f"{NUMBER_OF_ATOMS}"
    )

    print(
        f"KRAKEN MOD         = "
        f"{KRAKEN_MOD_PATH}"
    )

    print(
        "=" * 150
    )


    # ==================================================================
    # 读取MOD
    # ==================================================================
    mod_data = (
        parse_kraken_mod(
            KRAKEN_MOD_PATH
        )
    )


    stored_depths = (
        mod_data[
            "stored_depths"
        ]
    )

    stored_modes = (
        mod_data[
            "stored_mode_matrix"
        ]
    )

    complex_wavenumbers = (
        mod_data[
            "complex_wavenumbers"
        ]
    )


    print(
        "\nMOD information:"
    )

    print(
        f"  title          = "
        f"{mod_data['title']}"
    )

    print(
        f"  file size      = "
        f"{mod_data['file_size']} bytes"
    )

    print(
        f"  record length  = "
        f"{mod_data['record_length_bytes']} bytes"
    )

    print(
        f"  records        = "
        f"{mod_data['number_of_records']}"
    )

    print(
        f"  modes          = "
        f"{mod_data['number_of_modes']}"
    )

    print(
        f"  stored depths  = "
        f"{stored_depths.size}"
    )

    print(
        f"  stored matrix  = "
        f"{stored_modes.shape}"
    )


    print(
        "\nStored depths:"
    )

    print(
        stored_depths
    )


    print(
        "\nFirst 8 complex KRAKEN wavenumbers:"
    )


    for index in range(
        min(
            8,
            complex_wavenumbers.size,
        )
    ):

        value = (
            complex_wavenumbers[
                index
            ]
        )

        print(
            f"  k{index + 1} = "
            f"{value.real:.12f} "
            f"{value.imag:+.6e}j"
        )


    # ==================================================================
    # 真值
    # ==================================================================
    true_wavenumbers = np.asarray(
        np.real(
            complex_wavenumbers[
                :NUMBER_OF_TARGET_MODES
            ]
        ),
        dtype=np.float64,
    )


    print(
        "\nIndependent truth from .MOD:"
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
    # receiver-only mode matrix
    # ==================================================================
    receiver = (
        extract_integer_receiver_modes(
            mod_data
        )
    )


    receiver_modes = np.asarray(
        receiver[
            "receiver_mode_matrix"
        ][
            :,
            :NUMBER_OF_TARGET_MODES
        ],
        dtype=np.complex128,
    )


    print(
        "\nKRAKEN receiver mode matrix:"
    )

    print(
        f"  depths = "
        f"{receiver['receiver_depths'][0]:.1f}"
        f" ... "
        f"{receiver['receiver_depths'][-1]:.1f} m"
    )

    print(
        f"  shape  = "
        f"{receiver_modes.shape}"
    )


    # ==================================================================
    # inverse grid适配
    # ==================================================================
    adapted = (
        adapt_kraken_modes_to_inverse_grid(
            receiver_modes=(
                receiver_modes
            ),
            true_wavenumbers=(
                true_wavenumbers
            ),
        )
    )


    kraken_modes = np.asarray(
        adapted[
            "adapted_mode_matrix"
        ],
        dtype=np.complex128,
    )


    fd_modes = np.asarray(
        adapted[
            "inverse_probe"
        ],
        dtype=np.complex128,
    )


    adapted_depths = np.asarray(
        adapted[
            "adapted_depths"
        ],
        dtype=np.float64,
    )


    print(
        "\nGrid adaptation:"
    )

    print(
        f"  method          = "
        f"{adapted['adaptation']}"
    )

    print(
        f"  KRAKEN shape    = "
        f"{kraken_modes.shape}"
    )

    print(
        f"  inverse shape   = "
        f"{fd_modes.shape}"
    )


    # ==================================================================
    # normalize
    # ==================================================================
    kraken_modes, kraken_norms = (
        normalize_columns(
            kraken_modes
        )
    )


    fd_modes, fd_norms = (
        normalize_columns(
            fd_modes
        )
    )


    # ==================================================================
    # 模态失配
    # ==================================================================
    mismatch_rows = (
        calculate_mode_shape_mismatch(
            kraken_modes=(
                kraken_modes
            ),
            fd_modes=(
                fd_modes
            ),
        )
    )


    save_csv(
        MODE_MISMATCH_CSV,
        mismatch_rows,
    )


    print(
        "\nKRAKEN vs FD/Robin mode-shape mismatch:"
    )


    for row in (
        mismatch_rows
    ):

        print(
            f"  Mode {row['mode']} | "
            f"rel mismatch = "
            f"{row['relative_shape_mismatch']:.6e} | "
            f"correlation = "
            f"{row['normalized_correlation']:.9f}"
        )


    plot_mode_shapes(
        depths=(
            adapted_depths
        ),
        kraken_modes=(
            kraken_modes
        ),
        fd_modes=(
            fd_modes
        ),
    )


    # ==================================================================
    # 保存MOD信息
    # ==================================================================
    info_rows = []


    for index in range(
        mod_data[
            "number_of_modes"
        ]
    ):

        value = (
            complex_wavenumbers[
                index
            ]
        )


        info_rows.append(
            {
                "mode": (
                    index + 1
                ),

                "wavenumber_real": float(
                    np.real(
                        value
                    )
                ),

                "wavenumber_imag": float(
                    np.imag(
                        value
                    )
                ),

                "is_target": int(
                    index
                    <
                    NUMBER_OF_TARGET_MODES
                ),
            }
        )


    save_csv(
        MOD_INFO_CSV,
        info_rows,
    )


    # ==================================================================
    # Robin root catalogue
    #
    # selector/inverse side仍使用FD/Robin physics，
    # 不使用MOD mode shapes。
    # ==================================================================
    extended_roots = (
        FINAL.calculate_extended_physical_roots()
    )


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
        "\nInverse-side Robin roots:"
    )


    for index, root in enumerate(
        target_roots,
        start=1,
    ):

        print(
            f"  r{index} = "
            f"{root:.12f}"
        )


    print(
        "\nKRAKEN truth vs inverse Robin root:"
    )


    for index in range(
        NUMBER_OF_TARGET_MODES
    ):

        difference = abs(
            true_wavenumbers[
                index
            ]
            -
            target_roots[
                index
            ]
        )


        print(
            f"  Mode {index + 1}: "
            f"|Δk| = "
            f"{difference:.6e}"
        )


    # ==================================================================
    # Monte Carlo
    # ==================================================================
    restart_rows = []

    trial_rows = []


    # 同一个trial跨SNR共享clean data/noise direction。
    trial_master = {}


    for trial in range(
        1,
        NUMBER_OF_TRIALS + 1,
    ):

        base_seed = (
            BASE_RANDOM_SEED
            + trial
            * 10000
        )


        trial_master[
            trial
        ] = (
            generate_kraken_master_data(
                kraken_mode_matrix=(
                    kraken_modes
                ),
                coefficient_seed=(
                    base_seed
                    + 1
                ),
                noise_seed=(
                    base_seed
                    + 2
                ),
            )
        )


    for snr_db in (
        SNR_VALUES_DB
    ):

        print(
            "\n\n"
            + "=" * 150
        )

        print(
            f"SNR = {snr_db:+.1f} dB"
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
                f"SNR {snr_db:+.1f} dB | "
                f"Trial {trial}/"
                f"{NUMBER_OF_TRIALS}"
            )

            print(
                "#" * 120
            )


            dataset = (
                add_noise_at_snr(
                    trial_master[
                        trial
                    ],
                    snr_db,
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
            # restart pool
            # ==========================================================
            method_results = []


            base_seed = (
                BASE_RANDOM_SEED
                + trial
                * 10000
                + int(
                    (
                        snr_db
                        + 20.0
                    )
                    * 1000
                )
                + 100000
            )


            for restart_index in range(
                NUMBER_OF_RESTARTS
            ):

                seed = (
                    base_seed
                    + restart_index
                    * 1009
                )


                print(
                    f"\n  restart "
                    f"{restart_index + 1}/"
                    f"{NUMBER_OF_RESTARTS} | "
                    f"seed={seed}",
                    flush=True,
                )


                start_time = (
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


                result = (
                    FINAL.evaluate_restart_method(
                        trained=(
                            trained
                        ),

                        method=(
                            "ORDER_CELL_JOINT"
                        ),

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


                runtime = float(
                    time.time()
                    - start_time
                )


                method_results.append(
                    result
                )


                counts = (
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
                            restart_index
                            + 1
                        ),

                        "seed": (
                            seed
                        ),

                        "runtime_seconds": (
                            runtime
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

                        "restart_score": (
                            result[
                                "restart_score"
                            ]
                        ),

                        "joint_fit": (
                            result[
                                "joint_fit"
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
                            counts[0]
                            if len(
                                counts
                            ) > 0
                            else -1
                        ),

                        "cell2_count": (
                            counts[1]
                            if len(
                                counts
                            ) > 1
                            else -1
                        ),

                        "cell3_count": (
                            counts[2]
                            if len(
                                counts
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


            # ==========================================================
            # best restart
            # ==========================================================
            best_index = (
                choose_best_restart(
                    method_results
                )
            )


            any_raw_success = bool(
                any(
                    result[
                        "raw_success"
                    ]
                    for result
                    in method_results
                )
            )


            any_refined_success = bool(
                any(
                    result[
                        "refined_success"
                    ]
                    for result
                    in method_results
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
                        in method_results
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

                chosen_k4 = 0
                chosen_k5 = 0
                chosen_lower = 0

                chosen_raw = np.asarray(
                    [],
                    dtype=np.float64,
                )

                chosen_refined = np.asarray(
                    [],
                    dtype=np.float64,
                )


            else:

                chosen = (
                    method_results[
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


            print(
                "\n  [ORDER_CELL_JOINT]"
            )

            print(
                f"    raw success       = "
                f"{chosen_raw_success}"
            )

            print(
                f"    refined success   = "
                f"{chosen_refined_success}"
            )

            print(
                f"    any raw success   = "
                f"{any_raw_success}"
            )

            print(
                f"    restart success   = "
                f"{100.0 * successful_restart_fraction:.1f}%"
            )

            print(
                f"    raw kRMSE         = "
                f"{chosen_raw_rmse:.6e}"
            )

            print(
                f"    refined kRMSE     = "
                f"{chosen_refined_rmse:.6e}"
            )

            print(
                f"    raw k             = "
                f"{chosen_raw}"
            )

            print(
                f"    refined k         = "
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
            # checkpoint
            # ==========================================================
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
    # final
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


    plot_success(
        summary_rows
    )


    print(
        "\n\n"
        + "=" * 160
    )

    print(
        "KRAKEN INDEPENDENT-DATA SUMMARY"
    )

    print(
        "=" * 160
    )


    for row in (
        summary_rows
    ):

        print(
            f"SNR={row['snr_db']:>5.1f} dB | "
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
        + "=" * 160
    )

    print(
        "输出文件"
    )

    print(
        "=" * 160
    )

    print(
        MOD_INFO_CSV
    )

    print(
        MODE_MISMATCH_CSV
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
        + "=" * 160
    )

    print(
        "判定"
    )

    print(
        "=" * 160
    )

    print(
        "1. 首先检查KRAKEN / FD mode correlation。"
    )

    print(
        "2. QUICK重点看10/15 dB是否仍能稳定恢复。"
    )

    print(
        "3. 若独立数据相较自生成数据性能下降，这是合理且重要的结果，"
        "不能为了恢复原来的100%继续调参。"
    )

    print(
        "4. 若完全失效，则优先检查mode-shape mismatch和深度网格，"
        "而不是直接修改训练超参数。"
    )

    print(
        "5. Safe Refinement结果必须继续和Raw结果分开报告。"
    )

    print(
        "=" * 160
    )


if __name__ == "__main__":
    main()
