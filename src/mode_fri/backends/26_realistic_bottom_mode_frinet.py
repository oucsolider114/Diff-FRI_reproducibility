# -*- coding: utf-8 -*-
"""
第26阶段：真实海底 Robin Mode-FRINET

目标
------------------------------------------------------------
1. 采用老师论文对应的非均匀 SSP + 真实声学半空间海底；
2. KRAKEN 前三阶实部波数作为数据生成参考真值；
3. Mode-FRINET 连续原子训练保持原 V2 三阶段退火框架；
4. 将原压力释放底边界约束替换为可微 Robin 海底残差；
5. 比较：
       NMS
       PHYS_NMS
       ROOT_ORDER
6. Safe Physics Refinement 改为 Robin 物理根精修；
7. 首先在 15 dB / K=3 / L=20 / M=24 下验证主链路。

重要说明
------------------------------------------------------------
本阶段的“数据”仍采用：

    Y = Psi(k) C + N

其中真实 k 使用 KRAKEN 参考波数，
水中模态深度函数使用同一非均匀 SSP 的有限差分模型生成。

因此本阶段属于：

    KRAKEN-k-referenced realistic-bottom simulation

还不是直接读取 KRAKEN .MOD 模态场作为观测数据。

此外，本阶段 Robin 模型只拟合水平波数实部，
暂不把 KRAKEN 海底衰减对应的复波数虚部 alpha
纳入 Mode-FRINET 的训练参数。
"""

from __future__ import annotations

import csv
import math
import sys
import time
from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# ============================================================
# 工程路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

RESULT_DIR = OUTPUT_DIR / "26_realistic_bottom_mode_frinet"
FIGURE_DIR = RESULT_DIR / "figures"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mode_fri.mode_frinet import ModeFRINET  # noqa: E402


# ============================================================
# 运行模式
# ------------------------------------------------------------
# 第一次务必保持 True。
# QUICK 跑通、结果合理后，再改 False 跑正式 10×5。
# ============================================================
QUICK_TEST = False

SNR_DB = 15.0

NUMBER_OF_TRUE_MODES = 3
NUMBER_OF_SNAPSHOTS = 20
NUMBER_OF_ATOMS = 24

if QUICK_TEST:
    NUMBER_OF_TRIALS = 2
    NUMBER_OF_RESTARTS = 2
else:
    NUMBER_OF_TRIALS = 10
    NUMBER_OF_RESTARTS = 5

BASE_RANDOM_SEED = 20260826


# ============================================================
# 老师论文 / KRAKEN 环境
# ============================================================
WATER_DEPTH = 40.0
FREQUENCY = 300.0

WATER_DENSITY = 1.0
BOTTOM_SOUND_SPEED = 1753.0
BOTTOM_DENSITY = 1.957

# KRAKEN 中已经设置 0.5 dB / wavelength。
# 当前 Robin 实值模型暂不显式建模 alpha。
BOTTOM_ATTENUATION = 0.5

TRAINING_DZ = 1.0

OMEGA = 2.0 * math.pi * FREQUENCY

# 水柱最低声速为 1500 m/s
MAXIMUM_WATER_WAVENUMBER = OMEGA / 1500.0

# 海底体波波数
BOTTOM_BULK_WAVENUMBER = OMEGA / BOTTOM_SOUND_SPEED


# ============================================================
# Mode-FRINET 搜索区间
# ------------------------------------------------------------
# 前三阶 KRAKEN:
# 1.253758...
# 1.246358...
# 1.235612...
#
# 第四阶:
# 1.220074...
#
# 取 1.20 可把第4阶物理根保留在搜索区域中，
# 这样可以真正验证 ROOT_ORDER 是否能抑制合法但错误的高阶根。
# ============================================================
K_MIN = 1.20
K_MAX = MAXIMUM_WATER_WAVENUMBER * 0.999999


# ============================================================
# V2 三阶段训练参数
# ============================================================
COEFFICIENT_LEARNING_RATE = 1.0e-2

GROUP_LAMBDA_START = 0.001
GROUP_LAMBDA_FINAL = 0.005

ROBIN_LAMBDA = 1.0

WARMUP_STEPS = 300

STAGE1_STEPS = 500
STAGE1_K_LR = 0.005

STAGE2_STEPS = 1000
STAGE2_K_LR_START = 0.005
STAGE2_K_LR_END = 0.001

STAGE3_STEPS = 500
STAGE3_K_LR = 5.0e-4

STAGE3_ACTIVE_ATOMS = 8

GRADIENT_CLIP_NORM = 5.0


# ============================================================
# 选模 / ROOT_ORDER 参数
# ============================================================
CLUSTER_RADIUS = 0.0015
MINIMUM_MODE_SEPARATION = 0.004

PHYSICS_SCORE_GAMMA = 5000.0
ROOT_CAPTURE_RADIUS = 0.006
ROOT_ORDER_PENALTY = 0.25

REFINE_CAPTURE_RADIUS = 0.006
REFINE_FIT_DEGRADATION = 0.10

SUCCESS_TOLERANCE = 1.5e-3

METHODS = [
    "NMS",
    "PHYS_NMS",
    "ROOT_ORDER",
]


# ============================================================
# 已知 KRAKEN 参考值
# ------------------------------------------------------------
# 仅作为 Stage24 CSV 缺失时的后备。
# 正常情况下程序优先自动读取 Stage24 输出。
# ============================================================
KRAKEN_FALLBACK = np.asarray(
    [
        1.253758206000,
        1.246358878000,
        1.235612498000,
        1.220074263000,
        1.199563873000,
        1.173776536000,
        1.142472525000,
        1.105332087000,
    ],
    dtype=np.float64,
)


# ============================================================
# SSP
# ============================================================
def build_grid_and_ssp(
    dz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    老师论文当前复现实验使用的 SSP：

        0 ~ 10 m  : 1505 m/s
        10 ~ 20 m : 1505 -> 1500 m/s
        20 ~ 40 m : 1500 m/s
    """
    number_of_intervals = int(round(WATER_DEPTH / dz))

    z = np.linspace(
        0.0,
        WATER_DEPTH,
        number_of_intervals + 1,
        dtype=np.float64,
    )

    c = np.full_like(z, 1500.0)

    top_mask = z <= 10.0
    c[top_mask] = 1505.0

    thermocline_mask = (z > 10.0) & (z < 20.0)
    c[thermocline_mask] = (
        1505.0
        - 0.5 * (z[thermocline_mask] - 10.0)
    )

    lower_mask = z >= 20.0
    c[lower_mask] = 1500.0

    return z, c


# ============================================================
# NumPy 水中模态递推
# ============================================================
def solve_mode_numpy(
    horizontal_wavenumber: float,
    dz: float,
) -> np.ndarray:
    """
    表面压力释放：
        psi(0) = 0

    数值归一化：
        psi(dz) = 1

    水柱内部：
        psi'' + [omega^2/c^2(z) - k^2] psi = 0
    """
    _, c = build_grid_and_ssp(dz)

    psi = np.zeros(c.size, dtype=np.float64)

    psi[0] = 0.0
    psi[1] = 1.0

    k = float(horizontal_wavenumber)

    for j in range(1, c.size - 1):
        coefficient = (
            2.0
            - dz**2
            * (
                OMEGA**2 / c[j] ** 2
                - k**2
            )
        )

        psi[j + 1] = (
            coefficient * psi[j]
            - psi[j - 1]
        )

    return psi


# ============================================================
# Robin 海底边界
# ============================================================
def bottom_decay_constant_numpy(
    horizontal_wavenumber: float,
) -> float:
    """
    对 trapped mode:

        q_b = sqrt(k^2 - (omega/c_b)^2)

    当前搜索区域 K_MIN=1.20 > omega/c_b，
    因此均位于实值衰减区。
    """
    value = (
        float(horizontal_wavenumber) ** 2
        - BOTTOM_BULK_WAVENUMBER**2
    )

    if value <= 0.0:
        return 0.0

    return float(math.sqrt(value))


def robin_boundary_residual_numpy(
    horizontal_wavenumber: float,
    dz: float,
) -> float:
    """
    真实声学半空间海底 Robin 条件：

        psi'(H)
        + (rho_w/rho_b) q_b psi(H)
        = 0

    底部导数采用二阶向后差分：

        psi'(H)
        ~= [3 psi_N - 4 psi_(N-1) + psi_(N-2)] / (2 dz)
    """
    psi = solve_mode_numpy(
        horizontal_wavenumber=horizontal_wavenumber,
        dz=dz,
    )

    derivative = (
        3.0 * psi[-1]
        - 4.0 * psi[-2]
        + psi[-3]
    ) / (2.0 * dz)

    q_bottom = bottom_decay_constant_numpy(
        horizontal_wavenumber
    )

    residual = (
        derivative
        + (WATER_DENSITY / BOTTOM_DENSITY)
        * q_bottom
        * psi[-1]
    )

    return float(residual)


def normalized_robin_residual_numpy(
    horizontal_wavenumber: float,
    dz: float = TRAINING_DZ,
) -> float:
    """
    构造无量纲 Robin 残差。

    与旧版：

        |psi(H)| / ||psi||

    类似，现在使用：

        |B_Robin| * dz / ||psi||
    """
    psi = solve_mode_numpy(
        horizontal_wavenumber=horizontal_wavenumber,
        dz=dz,
    )

    raw_residual = robin_boundary_residual_numpy(
        horizontal_wavenumber=horizontal_wavenumber,
        dz=dz,
    )

    norm = float(np.linalg.norm(psi))

    if norm <= 1.0e-14:
        return float("inf")

    return float(
        abs(raw_residual)
        * dz
        / norm
    )


# ============================================================
# Torch 批量 Robin 残差
# ============================================================
def normalized_robin_residual_torch(
    wavenumbers: torch.Tensor,
    sound_speed_profile: torch.Tensor,
    dz: float,
) -> torch.Tensor:
    """
    输入：
        wavenumbers: [M]

    输出：
        normalized Robin residual: [M]

    全过程保持自动微分。
    """
    k = wavenumbers

    psi_previous = torch.zeros_like(k)
    psi_current = torch.ones_like(k)

    psi_values = [
        psi_previous,
        psi_current,
    ]

    for j in range(1, sound_speed_profile.numel() - 1):
        coefficient = (
            2.0
            - dz**2
            * (
                OMEGA**2
                / sound_speed_profile[j] ** 2
                - k**2
            )
        )

        psi_next = (
            coefficient * psi_current
            - psi_previous
        )

        psi_values.append(psi_next)

        psi_previous = psi_current
        psi_current = psi_next

    psi = torch.stack(
        psi_values,
        dim=0,
    )  # [Nz, M]

    derivative = (
        3.0 * psi[-1]
        - 4.0 * psi[-2]
        + psi[-3]
    ) / (2.0 * dz)

    bottom_bulk_k = torch.as_tensor(
        BOTTOM_BULK_WAVENUMBER,
        dtype=k.dtype,
        device=k.device,
    )

    decay_square = (
        k**2
        - bottom_bulk_k**2
    )

    # 当前搜索区间本来就 > 海底体波波数。
    # clamp 只是避免数值舍入导致 sqrt 非有限。
    q_bottom = torch.sqrt(
        torch.clamp(
            decay_square,
            min=1.0e-14,
        )
    )

    residual = (
        derivative
        + (WATER_DENSITY / BOTTOM_DENSITY)
        * q_bottom
        * psi[-1]
    )

    mode_norm = torch.linalg.vector_norm(
        psi,
        dim=0,
    ).clamp_min(1.0e-14)

    normalized = (
        residual
        * dz
        / mode_norm
    )

    return normalized


# ============================================================
# Robin 根扫描 + 二分
# ============================================================
def bisect_robin_root(
    left: float,
    right: float,
    dz: float,
    iterations: int = 70,
) -> float:
    f_left = robin_boundary_residual_numpy(
        left,
        dz,
    )
    f_right = robin_boundary_residual_numpy(
        right,
        dz,
    )

    if f_left == 0.0:
        return float(left)

    if f_right == 0.0:
        return float(right)

    if f_left * f_right > 0.0:
        raise ValueError(
            "二分区间内 Robin 残差没有符号变化"
        )

    a = float(left)
    b = float(right)
    fa = float(f_left)

    for _ in range(iterations):
        middle = 0.5 * (a + b)

        fm = robin_boundary_residual_numpy(
            middle,
            dz,
        )

        if fa * fm <= 0.0:
            b = middle
        else:
            a = middle
            fa = fm

    return float(0.5 * (a + b))


def find_robin_roots(
    dz: float,
    k_min: float,
    k_max: float,
    scan_points: int = 6001,
) -> np.ndarray:
    """
    在当前 Mode-FRINET 搜索区间内扫描全部 Robin 物理根。
    """
    grid = np.linspace(
        k_min,
        k_max,
        scan_points,
        dtype=np.float64,
    )

    residuals = np.asarray(
        [
            robin_boundary_residual_numpy(
                float(k),
                dz,
            )
            for k in grid
        ],
        dtype=np.float64,
    )

    roots = []

    for i in range(grid.size - 1):
        f1 = residuals[i]
        f2 = residuals[i + 1]

        if not (
            np.isfinite(f1)
            and np.isfinite(f2)
        ):
            continue

        if f1 == 0.0:
            root = float(grid[i])

        elif f1 * f2 < 0.0:
            root = bisect_robin_root(
                left=float(grid[i]),
                right=float(grid[i + 1]),
                dz=dz,
            )

        else:
            continue

        if not roots:
            roots.append(root)

        elif abs(root - roots[-1]) > 1.0e-6:
            roots.append(root)

    roots = np.asarray(
        roots,
        dtype=np.float64,
    )

    return np.sort(roots)[::-1]


def calculate_richardson_roots() -> np.ndarray:
    """
    实验25验证了 Robin 根具有明显二阶收敛。

    用 dz=0.5 与 0.25 做二阶 Richardson：

        k_inf ~= k_h/2 + (k_h/2-k_h)/3
    """
    print("\n计算当前环境 Robin 物理根...")

    roots_050 = find_robin_roots(
        dz=0.5,
        k_min=K_MIN,
        k_max=K_MAX,
    )

    roots_025 = find_robin_roots(
        dz=0.25,
        k_min=K_MIN,
        k_max=K_MAX,
    )

    number = min(
        roots_050.size,
        roots_025.size,
    )

    if number < NUMBER_OF_TRUE_MODES:
        raise RuntimeError(
            "Robin 物理根数量不足，无法继续实验26"
        )

    roots_050 = roots_050[:number]
    roots_025 = roots_025[:number]

    richardson = (
        roots_025
        + (
            roots_025
            - roots_050
        ) / 3.0
    )

    richardson = np.sort(
        richardson
    )[::-1]

    print("dz=0.5 roots:")
    print(roots_050)

    print("dz=0.25 roots:")
    print(roots_025)

    print("Richardson roots:")
    print(richardson)

    return richardson


# ============================================================
# KRAKEN 真值读取
# ============================================================
def try_read_kraken_csv(
    csv_path: Path,
) -> np.ndarray | None:
    if not csv_path.exists():
        return None

    try:
        with csv_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            if reader.fieldnames is None:
                return None

            fieldnames = reader.fieldnames

            preferred_fields = [
                "k",
                "k_real",
                "real_k",
                "wavenumber",
                "wavenumber_real",
                "horizontal_wavenumber",
                "k_rad_m",
                "k_rad_per_m",
            ]

            chosen_field = None

            lower_map = {
                name.lower().strip(): name
                for name in fieldnames
            }

            for preferred in preferred_fields:
                if preferred in lower_map:
                    chosen_field = lower_map[preferred]
                    break

            # 如果没有标准列名，则自动找包含 k/wavenumber
            if chosen_field is None:
                for name in fieldnames:
                    lower = name.lower()

                    if (
                        "wavenumber" in lower
                        or lower.strip() == "k"
                    ):
                        if "alpha" not in lower:
                            chosen_field = name
                            break

            rows = list(reader)

            if chosen_field is not None:
                values = []

                for row in rows:
                    try:
                        value = float(
                            row[chosen_field]
                        )
                    except Exception:
                        continue

                    if 0.5 < value < 2.0:
                        values.append(value)

                if len(values) >= 3:
                    return np.sort(
                        np.asarray(
                            values,
                            dtype=np.float64,
                        )
                    )[::-1]

            # 最后后备：
            # 在每一行寻找 1~1.4 的浮点数，
            # 排除 mode index / phase speed 等。
            values = []

            for row in rows:
                candidates = []

                for value_text in row.values():
                    try:
                        value = float(value_text)
                    except Exception:
                        continue

                    if 1.0 < value < 1.4:
                        candidates.append(value)

                if candidates:
                    values.append(candidates[0])

            if len(values) >= 3:
                return np.sort(
                    np.asarray(
                        values,
                        dtype=np.float64,
                    )
                )[::-1]

    except Exception:
        return None

    return None


def load_kraken_reference() -> np.ndarray:
    candidates = [
        RESULT_DIR.parent
        / "24_kraken_reference_environment"
        / "24_kraken_reference_wavenumbers.csv",

        RESULT_DIR.parent
        / "24_kraken_reference_environment"
        / "24_kraken_wavenumbers.csv",

        RESULT_DIR.parent
        / "24_kraken_reference_environment"
        / "kraken_reference_wavenumbers.csv",
    ]

    for path in candidates:
        values = try_read_kraken_csv(path)

        if (
            values is not None
            and values.size >= NUMBER_OF_TRUE_MODES
        ):
            print(
                "\nKRAKEN参考波数读取自："
            )
            print(path)
            return values

    print(
        "\n警告：未自动找到实验24的KRAKEN CSV，"
        "使用实验24已经确认的参考波数后备值。"
    )

    return KRAKEN_FALLBACK.copy()


# ============================================================
# 模态矩阵
# ------------------------------------------------------------
# receiver rows 使用 z=1...40 m。
# z=0 是压力释放表面，不作为观测阵元。
# ============================================================
def build_mode_matrix(
    wavenumbers: np.ndarray,
    dz: float = TRAINING_DZ,
) -> np.ndarray:
    columns = []

    for k in np.asarray(
        wavenumbers,
        dtype=np.float64,
    ):
        psi = solve_mode_numpy(
            horizontal_wavenumber=float(k),
            dz=dz,
        )

        # 去除 z=0 表面点
        receiver_mode = psi[1:].copy()

        norm = float(
            np.linalg.norm(receiver_mode)
        )

        if (
            not np.isfinite(norm)
            or norm <= 1.0e-14
        ):
            raise FloatingPointError(
                f"k={k:.12f} 模态范数异常"
            )

        columns.append(
            receiver_mode / norm
        )

    return np.column_stack(columns)


# ============================================================
# 数据生成
# ============================================================
def generate_dataset(
    true_wavenumbers: np.ndarray,
    coefficient_seed: int,
    noise_seed: int,
) -> dict[str, np.ndarray | float]:
    true_modes = build_mode_matrix(
        true_wavenumbers
    )

    coefficient_rng = np.random.default_rng(
        coefficient_seed
    )

    coefficients = (
        coefficient_rng.standard_normal(
            (
                NUMBER_OF_TRUE_MODES,
                NUMBER_OF_SNAPSHOTS,
            )
        )
        + 1j
        * coefficient_rng.standard_normal(
            (
                NUMBER_OF_TRUE_MODES,
                NUMBER_OF_SNAPSHOTS,
            )
        )
    ) / math.sqrt(2.0)

    clean_data = (
        true_modes.astype(np.complex128)
        @ coefficients
    )

    noise_rng = np.random.default_rng(
        noise_seed
    )

    noise = (
        noise_rng.standard_normal(
            clean_data.shape
        )
        + 1j
        * noise_rng.standard_normal(
            clean_data.shape
        )
    ) / math.sqrt(2.0)

    clean_norm = float(
        np.linalg.norm(clean_data)
    )

    noise_norm = float(
        np.linalg.norm(noise)
    )

    target_noise_norm = (
        clean_norm
        / (
            10.0
            ** (SNR_DB / 20.0)
        )
    )

    noise *= (
        target_noise_norm
        / max(noise_norm, 1.0e-14)
    )

    noisy_data = clean_data + noise

    achieved_snr = (
        20.0
        * math.log10(
            clean_norm
            / max(
                float(np.linalg.norm(noise)),
                1.0e-14,
            )
        )
    )

    return {
        "true_modes": true_modes,
        "coefficients": coefficients,
        "clean_data": clean_data,
        "noisy_data": noisy_data,
        "achieved_snr_db": achieved_snr,
    }


# ============================================================
# 损失函数
# ============================================================
def calculate_training_losses(
    model: ModeFRINET,
    normalized_data: torch.Tensor,
    sound_speed_tensor: torch.Tensor,
    group_lambda: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    object,
]:
    output = model()

    # output.reconstruction:
    # [Nz, L]
    #
    # z=0 是压力释放表面，数据只用 z=1...40m。
    reconstruction = output.reconstruction[1:, :]

    fit_loss = torch.mean(
        torch.abs(
            reconstruction
            - normalized_data
        ) ** 2
    )

    group_loss = (
        torch.mean(
            output.activities
        )
        / math.sqrt(NUMBER_OF_SNAPSHOTS)
    )

    robin_residuals = (
        normalized_robin_residual_torch(
            wavenumbers=output.wavenumbers,
            sound_speed_profile=sound_speed_tensor,
            dz=TRAINING_DZ,
        )
    )

    activity_weights = (
        output.activities.detach()
        + 0.02
        * torch.mean(
            output.activities.detach()
        )
    )

    robin_loss = (
        torch.sum(
            activity_weights
            * robin_residuals**2
        )
        / torch.sum(
            activity_weights
        ).clamp_min(1.0e-14)
    )

    total_loss = (
        fit_loss
        + float(group_lambda) * group_loss
        + ROBIN_LAMBDA * robin_loss
    )

    return (
        total_loss,
        fit_loss,
        group_loss,
        robin_loss,
        output,
    )


# ============================================================
# 单次 restart：真实海底 V2
# ============================================================
def train_one_restart(
    noisy_data: np.ndarray,
    random_seed: int,
) -> dict[str, object]:
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    device = torch.device("cpu")

    _, c_profile = build_grid_and_ssp(
        TRAINING_DZ
    )

    sound_speed_tensor = torch.as_tensor(
        c_profile,
        dtype=torch.float64,
        device=device,
    )

    noisy_tensor = torch.as_tensor(
        noisy_data,
        dtype=torch.complex128,
        device=device,
    )

    data_scale = torch.sqrt(
        torch.mean(
            torch.abs(noisy_tensor) ** 2
        )
    ).clamp_min(1.0e-14)

    normalized_data = (
        noisy_tensor / data_scale
    )

    model = ModeFRINET(
        number_of_atoms=NUMBER_OF_ATOMS,
        number_of_snapshots=NUMBER_OF_SNAPSHOTS,
        k_min=K_MIN,
        k_max=K_MAX,
        omega=OMEGA,
        sound_speed_profile=sound_speed_tensor,
        dz=TRAINING_DZ,
        random_seed=random_seed,
    ).to(device)

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Warmup
    # 只训练复系数，不移动波数
    # --------------------------------------------------------
    model.k_raw.requires_grad_(False)

    warmup_optimizer = torch.optim.Adam(
        [
            model.coefficient_real,
            model.coefficient_imag,
        ],
        lr=COEFFICIENT_LEARNING_RATE,
    )

    for _ in range(WARMUP_STEPS):
        warmup_optimizer.zero_grad()

        output = model()

        reconstruction = (
            output.reconstruction[1:, :]
        )

        fit_loss = torch.mean(
            torch.abs(
                reconstruction
                - normalized_data
            ) ** 2
        )

        group_loss = (
            torch.mean(
                output.activities
            )
            / math.sqrt(
                NUMBER_OF_SNAPSHOTS
            )
        )

        loss = (
            fit_loss
            + GROUP_LAMBDA_START
            * group_loss
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Warmup loss 非有限"
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [
                model.coefficient_real,
                model.coefficient_imag,
            ],
            max_norm=GRADIENT_CLIP_NORM,
        )

        warmup_optimizer.step()

    # --------------------------------------------------------
    # 联合优化
    # --------------------------------------------------------
    model.k_raw.requires_grad_(True)

    optimizer = torch.optim.Adam(
        [
            {
                "params": [model.k_raw],
                "lr": STAGE1_K_LR,
            },
            {
                "params": [
                    model.coefficient_real,
                    model.coefficient_imag,
                ],
                "lr": COEFFICIENT_LEARNING_RATE,
            },
        ]
    )

    # --------------------------------------------------------
    # Stage 1
    # --------------------------------------------------------
    for _ in range(STAGE1_STEPS):
        optimizer.param_groups[0][
            "lr"
        ] = STAGE1_K_LR

        optimizer.zero_grad()

        (
            total_loss,
            _,
            _,
            _,
            _,
        ) = calculate_training_losses(
            model=model,
            normalized_data=normalized_data,
            sound_speed_tensor=sound_speed_tensor,
            group_lambda=GROUP_LAMBDA_START,
        )

        if not torch.isfinite(
            total_loss
        ):
            raise FloatingPointError(
                "Stage1 loss 非有限"
            )

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP_NORM,
        )

        optimizer.step()

    # --------------------------------------------------------
    # Stage 2：连续退火
    # --------------------------------------------------------
    for local_step in range(
        STAGE2_STEPS
    ):
        if STAGE2_STEPS <= 1:
            progress = 1.0
        else:
            progress = (
                local_step
                / (STAGE2_STEPS - 1)
            )

        current_group_lambda = (
            GROUP_LAMBDA_START
            + progress
            * (
                GROUP_LAMBDA_FINAL
                - GROUP_LAMBDA_START
            )
        )

        current_k_lr = (
            STAGE2_K_LR_START
            + progress
            * (
                STAGE2_K_LR_END
                - STAGE2_K_LR_START
            )
        )

        optimizer.param_groups[0][
            "lr"
        ] = current_k_lr

        optimizer.zero_grad()

        (
            total_loss,
            _,
            _,
            _,
            _,
        ) = calculate_training_losses(
            model=model,
            normalized_data=normalized_data,
            sound_speed_tensor=sound_speed_tensor,
            group_lambda=current_group_lambda,
        )

        if not torch.isfinite(
            total_loss
        ):
            raise FloatingPointError(
                "Stage2 loss 非有限"
            )

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP_NORM,
        )

        optimizer.step()

    # --------------------------------------------------------
    # Stage 3：
    # 固定当前 activity Top-8，
    # 仅允许这8个 k 原子继续移动
    # --------------------------------------------------------
    model.eval()

    with torch.no_grad():
        stage3_output = model()

        number_active = min(
            STAGE3_ACTIVE_ATOMS,
            stage3_output.activities.numel(),
        )

        active_indices = torch.topk(
            stage3_output.activities,
            k=number_active,
        ).indices

    model.train()

    k_gradient_mask = torch.zeros_like(
        model.k_raw
    )

    k_gradient_mask[
        active_indices
    ] = 1.0

    optimizer.param_groups[0][
        "lr"
    ] = STAGE3_K_LR

    for _ in range(STAGE3_STEPS):
        optimizer.zero_grad()

        (
            total_loss,
            _,
            _,
            _,
            _,
        ) = calculate_training_losses(
            model=model,
            normalized_data=normalized_data,
            sound_speed_tensor=sound_speed_tensor,
            group_lambda=GROUP_LAMBDA_FINAL,
        )

        if not torch.isfinite(
            total_loss
        ):
            raise FloatingPointError(
                "Stage3 loss 非有限"
            )

        total_loss.backward()

        if model.k_raw.grad is not None:
            model.k_raw.grad.mul_(
                k_gradient_mask
            )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP_NORM,
        )

        optimizer.step()

    # --------------------------------------------------------
    # 最终状态
    # --------------------------------------------------------
    model.eval()

    with torch.no_grad():
        (
            total_loss,
            fit_loss,
            group_loss,
            robin_loss,
            final_output,
        ) = calculate_training_losses(
            model=model,
            normalized_data=normalized_data,
            sound_speed_tensor=sound_speed_tensor,
            group_lambda=GROUP_LAMBDA_FINAL,
        )

    runtime = (
        time.perf_counter()
        - start_time
    )

    return {
        "random_seed": int(
            random_seed
        ),
        "all_wavenumbers": (
            final_output.wavenumbers
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
            .copy()
        ),
        "all_activities": (
            final_output.activities
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
            .copy()
        ),
        "fit_loss": float(
            fit_loss.cpu()
        ),
        "group_loss": float(
            group_loss.cpu()
        ),
        "robin_loss": float(
            robin_loss.cpu()
        ),
        "total_loss": float(
            total_loss.cpu()
        ),
        "runtime_seconds": float(
            runtime
        ),
    }


# ============================================================
# NMS
# ============================================================
def select_by_scored_nms(
    wavenumbers: np.ndarray,
    activities: np.ndarray,
    scores: np.ndarray,
    number_to_select: int,
) -> np.ndarray:
    wavenumbers = np.asarray(
        wavenumbers,
        dtype=np.float64,
    )

    activities = np.asarray(
        activities,
        dtype=np.float64,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    if not (
        wavenumbers.shape
        == activities.shape
        == scores.shape
    ):
        raise ValueError(
            "NMS输入形状不一致"
        )

    order = np.argsort(
        scores
    )[::-1]

    suppressed = np.zeros(
        wavenumbers.size,
        dtype=bool,
    )

    selected = []

    for index in order:
        if suppressed[index]:
            continue

        center_k = float(
            wavenumbers[index]
        )

        local_mask = (
            np.abs(
                wavenumbers
                - center_k
            )
            <= CLUSTER_RADIUS
        )

        local_k = (
            wavenumbers[
                local_mask
            ]
        )

        local_weights = (
            activities[
                local_mask
            ]
        )

        weight_sum = float(
            np.sum(local_weights)
        )

        if weight_sum <= 1.0e-30:
            selected_k = center_k
        else:
            selected_k = float(
                np.sum(
                    local_k
                    * local_weights
                )
                / weight_sum
            )

        selected.append(
            selected_k
        )

        suppressed[
            np.abs(
                wavenumbers
                - center_k
            )
            < MINIMUM_MODE_SEPARATION
        ] = True

        if (
            len(selected)
            >= number_to_select
        ):
            break

    if (
        len(selected)
        < number_to_select
    ):
        raise RuntimeError(
            "NMS没有选出足够模态"
        )

    return np.asarray(
        selected,
        dtype=np.float64,
    )


# ============================================================
# Robin 物理评分
# ============================================================
def calculate_physics_scores(
    wavenumbers: np.ndarray,
    activities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    residuals = np.asarray(
        [
            normalized_robin_residual_numpy(
                float(k),
                TRAINING_DZ,
            )
            for k in wavenumbers
        ],
        dtype=np.float64,
    )

    scores = (
        activities
        / (
            1.0
            + PHYSICS_SCORE_GAMMA
            * residuals**2
        )
    )

    return (
        scores,
        residuals,
    )


# ============================================================
# 三种选模方法
# ============================================================
def select_modes(
    method: str,
    wavenumbers: np.ndarray,
    activities: np.ndarray,
    physical_roots: np.ndarray,
) -> np.ndarray:
    if method == "NMS":
        return select_by_scored_nms(
            wavenumbers=wavenumbers,
            activities=activities,
            scores=activities,
            number_to_select=NUMBER_OF_TRUE_MODES,
        )

    physics_scores, _ = (
        calculate_physics_scores(
            wavenumbers=wavenumbers,
            activities=activities,
        )
    )

    if method == "PHYS_NMS":
        return select_by_scored_nms(
            wavenumbers=wavenumbers,
            activities=activities,
            scores=physics_scores,
            number_to_select=NUMBER_OF_TRUE_MODES,
        )

    if method == "ROOT_ORDER":
        roots = np.sort(
            np.asarray(
                physical_roots,
                dtype=np.float64,
            )
        )[::-1]

        distances = np.abs(
            wavenumbers[:, None]
            - roots[None, :]
        )

        nearest_root_index = (
            np.argmin(
                distances,
                axis=1,
            )
        )

        nearest_distance = (
            distances[
                np.arange(
                    wavenumbers.size
                ),
                nearest_root_index,
            ]
        )

        corrected_scores = (
            physics_scores.copy()
        )

        # 第1/2/3阶是目标。
        # 第4阶及之后如果已经进入合法物理根吸引域，
        # 仍进行软惩罚，而不是直接删除。
        high_order_mask = (
            (
                nearest_distance
                <= ROOT_CAPTURE_RADIUS
            )
            & (
                nearest_root_index
                >= NUMBER_OF_TRUE_MODES
            )
        )

        corrected_scores[
            high_order_mask
        ] *= ROOT_ORDER_PENALTY

        return select_by_scored_nms(
            wavenumbers=wavenumbers,
            activities=activities,
            scores=corrected_scores,
            number_to_select=NUMBER_OF_TRUE_MODES,
        )

    raise ValueError(
        f"未知选模方法：{method}"
    )


# ============================================================
# LS 拟合
# ============================================================
def calculate_ls_fit_error(
    wavenumbers: np.ndarray,
    fitting_data: np.ndarray,
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
]:
    mode_matrix = build_mode_matrix(
        wavenumbers
    ).astype(
        np.complex128
    )

    coefficients = np.linalg.lstsq(
        mode_matrix,
        fitting_data,
        rcond=None,
    )[0]

    reconstruction = (
        mode_matrix
        @ coefficients
    )

    relative_error = float(
        np.linalg.norm(
            reconstruction
            - fitting_data
        )
        / max(
            np.linalg.norm(
                fitting_data
            ),
            1.0e-14,
        )
    )

    return (
        relative_error,
        coefficients,
        reconstruction,
    )


# ============================================================
# Restart 评分
# ------------------------------------------------------------
# 不使用 KRAKEN 真值。
# ============================================================
def calculate_restart_score(
    selected_wavenumbers: np.ndarray,
    all_activities: np.ndarray,
    noisy_data: np.ndarray,
) -> dict[str, float]:
    fit_error, _, _ = (
        calculate_ls_fit_error(
            selected_wavenumbers,
            noisy_data,
        )
    )

    robin_residuals = np.asarray(
        [
            normalized_robin_residual_numpy(
                float(k),
                TRAINING_DZ,
            )
            for k in selected_wavenumbers
        ],
        dtype=np.float64,
    )

    robin_penalty = float(
        np.mean(
            robin_residuals**2
        )
    )

    sorted_activity = np.sort(
        np.asarray(
            all_activities,
            dtype=np.float64,
        )
    )[::-1]

    concentration = float(
        np.sum(
            sorted_activity[
                :NUMBER_OF_TRUE_MODES
            ]
        )
        / max(
            np.sum(
                sorted_activity
            ),
            1.0e-14,
        )
    )

    concentration_penalty = (
        1.0 - concentration
    )

    separation_penalty = 0.0

    for i in range(
        selected_wavenumbers.size
    ):
        for j in range(
            i + 1,
            selected_wavenumbers.size,
        ):
            distance = abs(
                selected_wavenumbers[i]
                - selected_wavenumbers[j]
            )

            if (
                distance
                < MINIMUM_MODE_SEPARATION
            ):
                separation_penalty += (
                    (
                        MINIMUM_MODE_SEPARATION
                        - distance
                    )
                    / MINIMUM_MODE_SEPARATION
                ) ** 2

    score = (
        fit_error
        + 0.20 * robin_penalty
        + 0.05
        * concentration_penalty
        + 10.0
        * separation_penalty
    )

    return {
        "score": float(
            score
        ),
        "fit_error": float(
            fit_error
        ),
        "robin_penalty": float(
            robin_penalty
        ),
        "concentration": float(
            concentration
        ),
        "separation_penalty": float(
            separation_penalty
        ),
    }


# ============================================================
# 一对一波数匹配
# ============================================================
def best_one_to_one_match(
    estimated: np.ndarray,
    truth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    estimated = np.asarray(
        estimated,
        dtype=np.float64,
    )

    truth = np.asarray(
        truth,
        dtype=np.float64,
    )

    if estimated.size != truth.size:
        raise ValueError(
            "估计模态数与真值模态数不同"
        )

    best_match = None
    best_cost = float("inf")

    for order in permutations(
        range(estimated.size)
    ):
        candidate = estimated[
            list(order)
        ]

        cost = float(
            np.sum(
                (
                    candidate
                    - truth
                ) ** 2
            )
        )

        if cost < best_cost:
            best_cost = cost
            best_match = candidate.copy()

    if best_match is None:
        raise RuntimeError(
            "波数匹配失败"
        )

    return (
        best_match,
        best_match - truth,
    )


# ============================================================
# 估计评价
# ============================================================
def evaluate_estimate(
    estimated_wavenumbers: np.ndarray,
    true_wavenumbers: np.ndarray,
    noisy_data: np.ndarray,
    clean_data: np.ndarray,
) -> dict[str, object]:
    matched, errors = (
        best_one_to_one_match(
            estimated_wavenumbers,
            true_wavenumbers,
        )
    )

    k_rmse = float(
        np.sqrt(
            np.mean(
                errors**2
            )
        )
    )

    maximum_error = float(
        np.max(
            np.abs(errors)
        )
    )

    mode_matrix = build_mode_matrix(
        matched
    ).astype(
        np.complex128
    )

    coefficients = np.linalg.lstsq(
        mode_matrix,
        noisy_data,
        rcond=None,
    )[0]

    reconstruction = (
        mode_matrix
        @ coefficients
    )

    field_rmse = float(
        np.linalg.norm(
            reconstruction
            - clean_data
        )
        / max(
            np.linalg.norm(
                clean_data
            ),
            1.0e-14,
        )
    )

    return {
        "matched": matched,
        "errors": errors,
        "k_rmse": k_rmse,
        "maximum_error": maximum_error,
        "success": bool(
            maximum_error
            <= SUCCESS_TOLERANCE
        ),
        "field_rmse": field_rmse,
    }


# ============================================================
# Safe Robin-Richardson Refinement
# ============================================================
def safe_robin_refinement(
    raw_wavenumbers: np.ndarray,
    physical_roots: np.ndarray,
    fitting_data: np.ndarray,
) -> dict[str, object]:
    raw = np.asarray(
        raw_wavenumbers,
        dtype=np.float64,
    )

    roots = np.sort(
        np.asarray(
            physical_roots,
            dtype=np.float64,
        )
    )[::-1]

    distance_matrix = np.abs(
        raw[:, None]
        - roots[None, :]
    )

    nearest_indices = np.argmin(
        distance_matrix,
        axis=1,
    )

    nearest_distances = (
        distance_matrix[
            np.arange(raw.size),
            nearest_indices,
        ]
    )

    if np.any(
        nearest_distances
        > REFINE_CAPTURE_RADIUS
    ):
        return {
            "executed": False,
            "reason": (
                "RAW_OUTSIDE_ROBIN_ROOT_BASIN"
            ),
            "refined_wavenumbers": raw.copy(),
        }

    # 不允许两个估计同时吸到同一物理根
    if (
        len(
            set(
                nearest_indices.tolist()
            )
        )
        < raw.size
    ):
        return {
            "executed": False,
            "reason": "ROOT_COLLISION",
            "refined_wavenumbers": raw.copy(),
        }

    refined = roots[
        nearest_indices
    ].copy()

    raw_fit, _, _ = (
        calculate_ls_fit_error(
            raw,
            fitting_data,
        )
    )

    refined_fit, _, _ = (
        calculate_ls_fit_error(
            refined,
            fitting_data,
        )
    )

    allowed_fit = (
        raw_fit
        * (
            1.0
            + REFINE_FIT_DEGRADATION
        )
        + 1.0e-12
    )

    if refined_fit > allowed_fit:
        return {
            "executed": False,
            "reason": "FIT_DEGRADATION",
            "refined_wavenumbers": raw.copy(),
        }

    return {
        "executed": True,
        "reason": "ACCEPTED",
        "refined_wavenumbers": refined,
    }


# ============================================================
# 选某方法的最佳 restart
# ============================================================
def evaluate_restart_candidate(
    method: str,
    trained: dict[str, object],
    physical_roots: np.ndarray,
    true_wavenumbers: np.ndarray,
    noisy_data: np.ndarray,
    clean_data: np.ndarray,
) -> dict[str, object]:
    all_wavenumbers = np.asarray(
        trained["all_wavenumbers"],
        dtype=np.float64,
    )

    all_activities = np.asarray(
        trained["all_activities"],
        dtype=np.float64,
    )

    selected = select_modes(
        method=method,
        wavenumbers=all_wavenumbers,
        activities=all_activities,
        physical_roots=physical_roots,
    )

    restart_score = (
        calculate_restart_score(
            selected_wavenumbers=selected,
            all_activities=all_activities,
            noisy_data=noisy_data,
        )
    )

    raw_eval = evaluate_estimate(
        estimated_wavenumbers=selected,
        true_wavenumbers=true_wavenumbers,
        noisy_data=noisy_data,
        clean_data=clean_data,
    )

    return {
        "selected": selected,
        "restart_score": float(
            restart_score["score"]
        ),
        "fit_error": float(
            restart_score["fit_error"]
        ),
        "robin_penalty": float(
            restart_score["robin_penalty"]
        ),
        "concentration": float(
            restart_score["concentration"]
        ),
        "separation_penalty": float(
            restart_score[
                "separation_penalty"
            ]
        ),
        "raw_eval": raw_eval,
        "training_runtime": float(
            trained["runtime_seconds"]
        ),
    }


def choose_best_restart_and_refine(
    method: str,
    candidates: list[dict[str, object]],
    physical_roots: np.ndarray,
    true_wavenumbers: np.ndarray,
    noisy_data: np.ndarray,
    clean_data: np.ndarray,
) -> dict[str, object]:
    best_index = int(
        np.argmin(
            [
                float(
                    item[
                        "restart_score"
                    ]
                )
                for item in candidates
            ]
        )
    )

    best = candidates[
        best_index
    ]

    raw_selected = np.asarray(
        best["selected"],
        dtype=np.float64,
    )

    refinement = (
        safe_robin_refinement(
            raw_wavenumbers=raw_selected,
            physical_roots=physical_roots,
            fitting_data=noisy_data,
        )
    )

    refined_wavenumbers = np.asarray(
        refinement[
            "refined_wavenumbers"
        ],
        dtype=np.float64,
    )

    refined_eval = evaluate_estimate(
        estimated_wavenumbers=refined_wavenumbers,
        true_wavenumbers=true_wavenumbers,
        noisy_data=noisy_data,
        clean_data=clean_data,
    )

    successful_restarts = sum(
        int(
            candidate[
                "raw_eval"
            ]["success"]
        )
        for candidate in candidates
    )

    # 第三阶是否捕获
    third_root = float(
        true_wavenumbers[2]
    )

    third_captured = bool(
        np.min(
            np.abs(
                raw_selected
                - third_root
            )
        )
        <= ROOT_CAPTURE_RADIUS
    )

    # 第四阶是否合法但被误选
    fourth_selected = False

    if physical_roots.size >= 4:
        fourth_root = float(
            physical_roots[3]
        )

        fourth_selected = bool(
            np.min(
                np.abs(
                    raw_selected
                    - fourth_root
                )
            )
            <= ROOT_CAPTURE_RADIUS
        )

    return {
        "method": method,
        "best_restart": best_index + 1,
        "restart_success_fraction": (
            successful_restarts
            / len(candidates)
        ),
        "raw_selected": raw_selected,
        "raw_eval": best["raw_eval"],
        "refine_executed": bool(
            refinement["executed"]
        ),
        "refine_reason": str(
            refinement["reason"]
        ),
        "refined_wavenumbers": (
            refined_wavenumbers
        ),
        "refined_eval": refined_eval,
        "third_captured": third_captured,
        "fourth_selected": fourth_selected,
        "best_restart_score": float(
            best["restart_score"]
        ),
        "best_fit_error": float(
            best["fit_error"]
        ),
        "best_robin_penalty": float(
            best["robin_penalty"]
        ),
    }


# ============================================================
# CSV
# ============================================================
def array_to_text(
    values: np.ndarray,
) -> str:
    return ";".join(
        f"{float(value):.12f}"
        for value in np.asarray(
            values
        ).ravel()
    )


def save_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
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
        writer.writerows(rows)


# ============================================================
# 汇总
# ============================================================
def summarize_results(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    summary_rows = []

    for method in METHODS:
        subset = [
            row
            for row in rows
            if row["method"] == method
        ]

        if not subset:
            continue

        summary_rows.append(
            {
                "method": method,
                "trials": len(subset),

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

                "refine_execution_rate": float(
                    np.mean(
                        [
                            row[
                                "refine_executed"
                            ]
                            for row
                            in subset
                        ]
                    )
                ),

                "restart_success_fraction_mean": float(
                    np.mean(
                        [
                            row[
                                "restart_success_fraction"
                            ]
                            for row
                            in subset
                        ]
                    )
                ),

                "third_capture_rate": float(
                    np.mean(
                        [
                            row[
                                "third_captured"
                            ]
                            for row
                            in subset
                        ]
                    )
                ),

                "fourth_false_selection_rate": float(
                    np.mean(
                        [
                            row[
                                "fourth_selected"
                            ]
                            for row
                            in subset
                        ]
                    )
                ),

                "raw_k_rmse_median": float(
                    np.median(
                        [
                            row[
                                "raw_k_rmse"
                            ]
                            for row
                            in subset
                        ]
                    )
                ),

                "refined_k_rmse_median": float(
                    np.median(
                        [
                            row[
                                "refined_k_rmse"
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

                "refined_max_error_median": float(
                    np.median(
                        [
                            row[
                                "refined_max_error"
                            ]
                            for row
                            in subset
                        ]
                    )
                ),

                "raw_field_rmse_median": float(
                    np.median(
                        [
                            row[
                                "raw_field_rmse"
                            ]
                            for row
                            in subset
                        ]
                    )
                ),

                "refined_field_rmse_median": float(
                    np.median(
                        [
                            row[
                                "refined_field_rmse"
                            ]
                            for row
                            in subset
                        ]
                    )
                ),

                "trial_runtime_seconds_median": float(
                    np.median(
                        [
                            row[
                                "trial_runtime_seconds"
                            ]
                            for row
                            in subset
                        ]
                    )
                ),
            }
        )

    return summary_rows


# ============================================================
# 绘图
# ============================================================
def make_figures(
    summary_rows: list[dict[str, object]],
    suffix: str,
) -> None:
    methods = [
        str(row["method"])
        for row in summary_rows
    ]

    x = np.arange(
        len(methods)
    )

    # --------------------------------------------------------
    # 1. 成功率
    # --------------------------------------------------------
    raw_success = (
        np.asarray(
            [
                float(
                    row[
                        "raw_success_rate"
                    ]
                )
                for row
                in summary_rows
            ]
        )
        * 100.0
    )

    refined_success = (
        np.asarray(
            [
                float(
                    row[
                        "refined_success_rate"
                    ]
                )
                for row
                in summary_rows
            ]
        )
        * 100.0
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(9.5, 5.5)
    )

    ax.bar(
        x - width / 2.0,
        raw_success,
        width,
        label="Raw",
    )

    ax.bar(
        x + width / 2.0,
        refined_success,
        width,
        label="Robin refined",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0.0, 105.0)
    ax.set_ylabel("Success rate (%)")
    ax.set_title(
        "15 dB realistic-bottom Mode-FRINET"
    )
    ax.grid(
        axis="y",
        alpha=0.3,
    )
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / (
            f"105_realistic_bottom_success_"
            f"{suffix}.png"
        ),
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # 2. k-RMSE
    # --------------------------------------------------------
    raw_rmse = np.asarray(
        [
            float(
                row[
                    "raw_k_rmse_median"
                ]
            )
            for row
            in summary_rows
        ]
    )

    refined_rmse = np.asarray(
        [
            float(
                row[
                    "refined_k_rmse_median"
                ]
            )
            for row
            in summary_rows
        ]
    )

    fig, ax = plt.subplots(
        figsize=(9.5, 5.5)
    )

    ax.semilogy(
        methods,
        raw_rmse,
        marker="o",
        label="Raw",
    )

    ax.semilogy(
        methods,
        refined_rmse,
        marker="s",
        label="Robin refined",
    )

    ax.axhline(
        SUCCESS_TOLERANCE,
        linestyle="--",
        label="Success tolerance",
    )

    ax.set_ylabel(
        "Median k-RMSE (rad/m)"
    )

    ax.set_title(
        "Realistic-bottom wavenumber error"
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
            f"106_realistic_bottom_rmse_"
            f"{suffix}.png"
        ),
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # 3. 第三阶捕获 / 第四阶误选
    # --------------------------------------------------------
    third_capture = (
        np.asarray(
            [
                float(
                    row[
                        "third_capture_rate"
                    ]
                )
                for row
                in summary_rows
            ]
        )
        * 100.0
    )

    fourth_false = (
        np.asarray(
            [
                float(
                    row[
                        "fourth_false_selection_rate"
                    ]
                )
                for row
                in summary_rows
            ]
        )
        * 100.0
    )

    fig, ax = plt.subplots(
        figsize=(9.5, 5.5)
    )

    ax.bar(
        x - width / 2.0,
        third_capture,
        width,
        label="3rd-mode capture",
    )

    ax.bar(
        x + width / 2.0,
        fourth_false,
        width,
        label="4th-root false selection",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0.0, 105.0)
    ax.set_ylabel("Rate (%)")
    ax.set_title(
        "Realistic-bottom mode-order selection"
    )
    ax.grid(
        axis="y",
        alpha=0.3,
    )
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / (
            f"107_realistic_bottom_selection_"
            f"{suffix}.png"
        ),
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 主程序
# ============================================================
def main() -> None:
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.set_default_dtype(
        torch.float64
    )

    suffix = (
        "quick"
        if QUICK_TEST
        else "full"
    )

    print("=" * 112)
    print(
        "第26阶段：真实海底 Robin Mode-FRINET"
    )
    print("=" * 112)

    print(
        f"模式                 = {suffix}"
    )
    print(
        f"Frequency            = {FREQUENCY:.1f} Hz"
    )
    print(
        f"Water depth          = {WATER_DEPTH:.1f} m"
    )
    print(
        "SSP                  = "
        "1505 -> 1500 m/s"
    )
    print(
        f"Bottom c             = {BOTTOM_SOUND_SPEED:.1f} m/s"
    )
    print(
        f"Bottom density       = {BOTTOM_DENSITY:.3f} g/cm^3"
    )
    print(
        f"Bottom attenuation   = {BOTTOM_ATTENUATION:.3f} dB/wavelength"
    )
    print(
        f"Training dz          = {TRAINING_DZ:.3f} m"
    )
    print(
        f"SNR                  = {SNR_DB:+.1f} dB"
    )
    print(
        f"K / L / M            = "
        f"{NUMBER_OF_TRUE_MODES} / "
        f"{NUMBER_OF_SNAPSHOTS} / "
        f"{NUMBER_OF_ATOMS}"
    )
    print(
        f"Trials               = {NUMBER_OF_TRIALS}"
    )
    print(
        f"Restarts/trial       = {NUMBER_OF_RESTARTS}"
    )
    print(
        f"k search             = "
        f"[{K_MIN:.6f}, {K_MAX:.6f}]"
    )
    print("=" * 112)

    # --------------------------------------------------------
    # 1. KRAKEN 参考真值
    # --------------------------------------------------------
    kraken_all = (
        load_kraken_reference()
    )

    true_wavenumbers = (
        np.sort(
            kraken_all[
                :NUMBER_OF_TRUE_MODES
            ]
        )[::-1]
    )

    print(
        "\nKRAKEN前三阶参考真值："
    )

    for index, value in enumerate(
        true_wavenumbers,
        start=1,
    ):
        print(
            f"  mode {index}: "
            f"{value:.12f} rad/m"
        )

    # --------------------------------------------------------
    # 2. 当前环境自身的 Robin 物理根
    # --------------------------------------------------------
    physical_roots = (
        calculate_richardson_roots()
    )

    print(
        "\n当前搜索区间 Robin-Richardson 物理根："
    )

    for index, value in enumerate(
        physical_roots,
        start=1,
    ):
        print(
            f"  root {index}: "
            f"{value:.12f} rad/m"
        )

    print(
        "\n前三阶 Robin-Richardson vs KRAKEN："
    )

    for index in range(
        NUMBER_OF_TRUE_MODES
    ):
        error = abs(
            physical_roots[index]
            - true_wavenumbers[index]
        )

        print(
            f"  mode {index + 1}: "
            f"|Delta k| = {error:.6e}"
        )

    print(
        "\n注意：这里的 Safe Refinement "
        "使用 Robin-Richardson 物理根，"
        "不是直接把 KRAKEN 真值作为答案。"
    )

    # --------------------------------------------------------
    # 3. Monte Carlo
    # --------------------------------------------------------
    rows = []

    for trial in range(
        1,
        NUMBER_OF_TRIALS + 1,
    ):
        print(
            "\n"
            + "-" * 112
        )
        print(
            f"[Trial {trial}/"
            f"{NUMBER_OF_TRIALS}]"
        )
        print(
            "-" * 112
        )

        base_seed = (
            BASE_RANDOM_SEED
            + trial * 1000
        )

        dataset = generate_dataset(
            true_wavenumbers=true_wavenumbers,
            coefficient_seed=base_seed + 1,
            noise_seed=base_seed + 2,
        )

        noisy_data = np.asarray(
            dataset["noisy_data"]
        )

        clean_data = np.asarray(
            dataset["clean_data"]
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

        trained_restarts = []

        trial_start = (
            time.perf_counter()
        )

        # 同一批训练原子同时给三种选模方法使用
        for restart in range(
            NUMBER_OF_RESTARTS
        ):
            seed = (
                base_seed
                + 100
                + restart * 1009
            )

            print(
                f"  restart "
                f"{restart + 1}/"
                f"{NUMBER_OF_RESTARTS} "
                f"(seed={seed}) ...",
                flush=True,
            )

            trained = train_one_restart(
                noisy_data=noisy_data,
                random_seed=seed,
            )

            trained_restarts.append(
                trained
            )

            print(
                "    "
                f"time="
                f"{trained['runtime_seconds']:.1f}s  "
                f"fit="
                f"{trained['fit_loss']:.4e}  "
                f"robin="
                f"{trained['robin_loss']:.4e}"
            )

        trial_runtime = (
            time.perf_counter()
            - trial_start
        )

        # ----------------------------------------------------
        # 同一训练结果比较三个 selector
        # ----------------------------------------------------
        for method in METHODS:
            candidates = [
                evaluate_restart_candidate(
                    method=method,
                    trained=trained,
                    physical_roots=physical_roots,
                    true_wavenumbers=true_wavenumbers,
                    noisy_data=noisy_data,
                    clean_data=clean_data,
                )
                for trained
                in trained_restarts
            ]

            final = (
                choose_best_restart_and_refine(
                    method=method,
                    candidates=candidates,
                    physical_roots=physical_roots,
                    true_wavenumbers=true_wavenumbers,
                    noisy_data=noisy_data,
                    clean_data=clean_data,
                )
            )

            raw_eval = final[
                "raw_eval"
            ]

            refined_eval = final[
                "refined_eval"
            ]

            print(
                f"\n  [{method}]"
            )

            print(
                "    raw      =",
                np.asarray(
                    final[
                        "raw_selected"
                    ]
                ),
            )

            print(
                "    refined  =",
                np.asarray(
                    final[
                        "refined_wavenumbers"
                    ]
                ),
            )

            print(
                f"    raw kRMSE      = "
                f"{raw_eval['k_rmse']:.6e}"
            )

            print(
                f"    raw max error  = "
                f"{raw_eval['maximum_error']:.6e}"
            )

            print(
                f"    raw success    = "
                f"{bool(raw_eval['success'])}"
            )

            print(
                f"    refine         = "
                f"{final['refine_executed']} "
                f"({final['refine_reason']})"
            )

            print(
                f"    refined kRMSE  = "
                f"{refined_eval['k_rmse']:.6e}"
            )

            print(
                f"    refined success= "
                f"{bool(refined_eval['success'])}"
            )

            print(
                f"    third captured = "
                f"{final['third_captured']}"
            )

            print(
                f"    fourth selected= "
                f"{final['fourth_selected']}"
            )

            row = {
                "trial": trial,
                "method": method,
                "snr_db": SNR_DB,
                "achieved_snr_db": achieved_snr,

                "best_restart": final[
                    "best_restart"
                ],

                "restart_success_fraction": final[
                    "restart_success_fraction"
                ],

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

                "refine_executed": int(
                    final[
                        "refine_executed"
                    ]
                ),

                "refine_reason": final[
                    "refine_reason"
                ],

                "third_captured": int(
                    final[
                        "third_captured"
                    ]
                ),

                "fourth_selected": int(
                    final[
                        "fourth_selected"
                    ]
                ),

                "raw_k_rmse": raw_eval[
                    "k_rmse"
                ],

                "raw_max_error": raw_eval[
                    "maximum_error"
                ],

                "refined_k_rmse": (
                    refined_eval[
                        "k_rmse"
                    ]
                ),

                "refined_max_error": (
                    refined_eval[
                        "maximum_error"
                    ]
                ),

                "raw_field_rmse": (
                    raw_eval[
                        "field_rmse"
                    ]
                ),

                "refined_field_rmse": (
                    refined_eval[
                        "field_rmse"
                    ]
                ),

                "best_restart_score": final[
                    "best_restart_score"
                ],

                "best_fit_error": final[
                    "best_fit_error"
                ],

                "best_robin_penalty": final[
                    "best_robin_penalty"
                ],

                "trial_runtime_seconds": (
                    trial_runtime
                ),

                "raw_selected": (
                    array_to_text(
                        final[
                            "raw_selected"
                        ]
                    )
                ),

                "raw_matched": (
                    array_to_text(
                        raw_eval[
                            "matched"
                        ]
                    )
                ),

                "raw_errors": (
                    array_to_text(
                        raw_eval[
                            "errors"
                        ]
                    )
                ),

                "refined_wavenumbers": (
                    array_to_text(
                        final[
                            "refined_wavenumbers"
                        ]
                    )
                ),

                "refined_matched": (
                    array_to_text(
                        refined_eval[
                            "matched"
                        ]
                    )
                ),

                "refined_errors": (
                    array_to_text(
                        refined_eval[
                            "errors"
                        ]
                    )
                ),
            }

            rows.append(
                row
            )

    # --------------------------------------------------------
    # 4. 保存
    # --------------------------------------------------------
    trial_csv = (
        RESULT_DIR
        / (
            "26_realistic_bottom_trials_"
            f"{suffix}.csv"
        )
    )

    save_csv(
        trial_csv,
        rows,
    )

    summary_rows = (
        summarize_results(
            rows
        )
    )

    summary_csv = (
        RESULT_DIR
        / (
            "26_realistic_bottom_summary_"
            f"{suffix}.csv"
        )
    )

    save_csv(
        summary_csv,
        summary_rows,
    )

    make_figures(
        summary_rows=summary_rows,
        suffix=suffix,
    )

    # --------------------------------------------------------
    # 5. 控制台总表
    # --------------------------------------------------------
    print(
        "\n"
        + "=" * 112
    )

    print(
        "第26阶段汇总"
    )

    print(
        "=" * 112
    )

    for row in summary_rows:
        print(
            f"{row['method']:12s} | "
            f"raw="
            f"{100.0 * row['raw_success_rate']:6.1f}% | "
            f"ref="
            f"{100.0 * row['refined_success_rate']:6.1f}% | "
            f"kRMSE(raw)="
            f"{row['raw_k_rmse_median']:.3e} | "
            f"kRMSE(ref)="
            f"{row['refined_k_rmse_median']:.3e} | "
            f"k3 capture="
            f"{100.0 * row['third_capture_rate']:6.1f}% | "
            f"k4 false="
            f"{100.0 * row['fourth_false_selection_rate']:6.1f}%"
        )

    print(
        "\n结果文件："
    )
    print(
        trial_csv
    )
    print(
        summary_csv
    )
    print(
        FIGURE_DIR
    )

    print(
        "\n解释口径："
    )

    print(
        "1. raw 是含噪数据驱动的连续波数估计；"
    )

    print(
        "2. refined 是粗估计进入正确 Robin 根吸引域后，"
        "利用已知环境先验执行的物理精修；"
    )

    print(
        "3. refined 精度不应再表述为 1e-14，"
        "因为这里以 KRAKEN 为参考，而 Robin-Richardson "
        "模型本身与 KRAKEN 仍存在约 1e-6 量级模型差异；"
    )

    print(
        "4. 本阶段尚未把 KRAKEN attenuation 对应的复波数虚部 "
        "直接纳入 Mode-FRINET。"
    )

    print(
        "=" * 112
    )


if __name__ == "__main__":
    main()
