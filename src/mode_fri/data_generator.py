from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np

from mode_fri.mode_solver import solve_mode
from mode_fri.scenario import ShallowWaterScenario


@dataclass
class VerticalArrayDataset:
    """固定距离、单频、多组垂直阵复声压数据集。"""

    z: np.ndarray
    true_wavenumbers: np.ndarray
    true_modes: np.ndarray
    true_coefficients: np.ndarray
    clean_data: np.ndarray
    noisy_data: np.ndarray
    requested_snr_db: float
    achieved_snr_db: float


def load_refined_wavenumbers(
    csv_path: Path,
    number_of_modes: int,
) -> np.ndarray:
    """
    从第二阶段生成的 CSV 中读取离散差分方程的精确本征波数。
    """
    if number_of_modes <= 0:
        raise ValueError("number_of_modes 必须大于 0")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"没有找到本征波数文件：{csv_path}\n"
            "请先运行 experiments/02_refine_eigenwavenumbers.py"
        )

    values: list[float] = []

    with csv_path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            values.append(
                float(row["estimated_k_rad_per_m"])
            )

    if len(values) < number_of_modes:
        raise ValueError(
            f"CSV 中只有 {len(values)} 个模态，"
            f"无法读取前 {number_of_modes} 个模态"
        )

    return np.asarray(
        values[:number_of_modes],
        dtype=float,
    )


def build_mode_matrix(
    scenario: ShallowWaterScenario,
    wavenumbers: np.ndarray,
) -> np.ndarray:
    """
    根据本征波数生成真实模态矩阵。

    返回形状：
        [Nz, K]

    每一列是一个满足：
        psi(0)=0
        psi(dz)=1
    的简正波模态。
    """
    wavenumbers = np.asarray(
        wavenumbers,
        dtype=float,
    )

    if wavenumbers.ndim != 1:
        raise ValueError("wavenumbers 必须是一维数组")

    c_profile = scenario.sound_speed_profile()

    mode_columns = []

    for candidate_k in wavenumbers:
        psi = solve_mode(
            horizontal_wavenumber=float(candidate_k),
            omega=scenario.omega,
            sound_speed_profile=c_profile,
            dz=scenario.dz,
        )

        if not np.isclose(psi[0], 0.0, atol=1e-12):
            raise RuntimeError("模态不满足 psi(0)=0")

        if not np.isclose(psi[1], 1.0, atol=1e-12):
            raise RuntimeError("模态不满足 psi(dz)=1")

        mode_columns.append(psi)

    return np.column_stack(mode_columns)


def generate_complex_coefficients(
    number_of_modes: int,
    number_of_snapshots: int,
    random_seed: int,
) -> np.ndarray:
    """
    生成多组测量中的复模态展开系数。

    每一行对应一个真实模态；
    每一列对应一组垂直阵测量。

    所有快照共享相同的活跃模态集合，但幅度和相位允许变化。
    """
    if number_of_modes <= 0:
        raise ValueError("number_of_modes 必须大于 0")

    if number_of_snapshots <= 0:
        raise ValueError("number_of_snapshots 必须大于 0")

    rng = np.random.default_rng(random_seed)

    # 不同模态的基础幅度，使三个模态强度有所区别
    base_amplitudes = np.linspace(
        1.0,
        0.55,
        number_of_modes,
    )

    # 每组数据存在轻微幅度起伏
    amplitude_jitter = rng.uniform(
        low=0.85,
        high=1.15,
        size=(number_of_modes, number_of_snapshots),
    )

    phases = rng.uniform(
        low=0.0,
        high=2.0 * np.pi,
        size=(number_of_modes, number_of_snapshots),
    )

    coefficients = (
        base_amplitudes[:, None]
        * amplitude_jitter
        * np.exp(1j * phases)
    )

    return coefficients.astype(np.complex128)


def add_complex_white_noise(
    clean_data: np.ndarray,
    snr_db: float,
    random_seed: int,
) -> tuple[np.ndarray, float]:
    """
    按指定 SNR 添加复高斯白噪声。

    复噪声实部和虚部相互独立，并各承担一半噪声功率。
    """
    clean_data = np.asarray(
        clean_data,
        dtype=np.complex128,
    )

    if clean_data.size == 0:
        raise ValueError("clean_data 不能为空")

    signal_power = float(
        np.mean(np.abs(clean_data) ** 2)
    )

    if signal_power <= 0.0:
        raise ValueError("信号功率必须大于 0")

    target_noise_power = (
        signal_power / (10.0 ** (snr_db / 10.0))
    )

    rng = np.random.default_rng(random_seed)

    noise_scale = np.sqrt(
        target_noise_power / 2.0
    )

    noise = noise_scale * (
        rng.standard_normal(clean_data.shape)
        + 1j * rng.standard_normal(clean_data.shape)
    )

    noisy_data = clean_data + noise

    measured_noise_power = float(
        np.mean(np.abs(noise) ** 2)
    )

    achieved_snr_db = float(
        10.0
        * np.log10(signal_power / measured_noise_power)
    )

    return noisy_data, achieved_snr_db


def generate_vertical_array_dataset(
    scenario: ShallowWaterScenario,
    true_wavenumbers: np.ndarray,
    number_of_snapshots: int = 20,
    snr_db: float = 10.0,
    coefficient_seed: int = 20260731,
    noise_seed: int = 20260732,
) -> VerticalArrayDataset:
    """
    生成固定距离、单频、多组垂直阵数据：

        Y = Psi C + N

    Psi:
        [Nz, K]，真实模态矩阵。

    C:
        [K, L]，多组复展开系数。

    Y:
        [Nz, L]，多组垂直阵复声压。
    """
    true_wavenumbers = np.asarray(
        true_wavenumbers,
        dtype=float,
    )

    true_modes = build_mode_matrix(
        scenario=scenario,
        wavenumbers=true_wavenumbers,
    )

    true_coefficients = generate_complex_coefficients(
        number_of_modes=true_wavenumbers.size,
        number_of_snapshots=number_of_snapshots,
        random_seed=coefficient_seed,
    )

    clean_data = (
        true_modes.astype(np.complex128)
        @ true_coefficients
    )

    noisy_data, achieved_snr_db = add_complex_white_noise(
        clean_data=clean_data,
        snr_db=snr_db,
        random_seed=noise_seed,
    )

    return VerticalArrayDataset(
        z=scenario.z,
        true_wavenumbers=true_wavenumbers,
        true_modes=true_modes,
        true_coefficients=true_coefficients,
        clean_data=clean_data,
        noisy_data=noisy_data,
        requested_snr_db=float(snr_db),
        achieved_snr_db=achieved_snr_db,
    )
