"""
M4a：水平波数环境扰动一阶修正（JASA Express Lett. 1, 126002 (2021), Eq.(20)）。

背景：
    算法假设声速 c0（可能与实际环境不符）时，估计的模态波数会系统性地
    偏离真实波数——精修会确定性吸附到"假设环境的物理根"上（实验 13：
    1% 声速误差即产生约 2.1e-3 的波数偏移）。

修正：
    若已知真实声速剖面相对假设值的偏差 Δc(z) = c(z) − c0，可用一阶扰动
    公式把估计波数修正回真实环境：

        δk_n = −(k0² / k_n) · ∫₀^H (Δc(z') / c0) · φ_n²(z') dz'

    其中 k0 = ω/c0 为假设环境的参考波数，φ_n 为（归一化）模态深度函数。
    对常声速偏差 Δc 为常数时，∫Δc·φ_n² dz = Δc·∫φ_n² dz = Δc。

    注意：该修正需要知道 Δc（例如来自实测/气候学/数据估计），
    属于"已知环境先验下的偏差补偿"，将失配从"硬失败"转为"软敏感"。
"""

from __future__ import annotations

import numpy as np

from mode_fri.mode_solver import solve_mode


def wavenumber_perturbation(
    wavenumber: float,
    omega: float,
    assumed_sound_speed: float,
    mode_function: np.ndarray,
    delta_c_profile: np.ndarray,
    dz: float,
) -> float:
    """
    对单个波数应用 JASA Eq.(20) 一阶扰动修正。

    参数
    ----
    wavenumber:
        估计波数 k_n（rad/m）。
    omega:
        角频率（rad/s）。
    assumed_sound_speed:
        算法假设声速 c0（m/s）。
    mode_function:
        假设环境下求解器生成的模态深度函数 ψ_n(z)，长度 Nz。
    delta_c_profile:
        真实声速相对假设值的偏差剖面 Δc(z) = c(z) − c0，长度 Nz。
    dz:
        深度间隔（m）。

    返回
    ----
    delta_k:
        一阶修正量 δk_n（rad/m），修正后波数为 k_n + δk_n。
    """
    mode = np.asarray(mode_function, dtype=float)

    if mode.ndim != 1:
        raise ValueError("mode_function 必须是一维数组")

    delta_c = np.asarray(delta_c_profile, dtype=float)

    if delta_c.shape != mode.shape:
        raise ValueError("delta_c_profile 与 mode_function 长度必须一致")

    mode_norm = float(np.linalg.norm(mode))

    if mode_norm <= 1e-12:
        raise FloatingPointError("模态函数范数过小")

    normalized_mode = mode / mode_norm

    reference_wavenumber = omega / assumed_sound_speed

    if reference_wavenumber <= 0.0 or wavenumber <= 0.0:
        raise ValueError("波数与参考波数必须大于 0")

    # ∫ (Δc / c0) · φ² dz，用梯形/矩形数值积分（dz 恒定）
    integrand = (
        delta_c / assumed_sound_speed
    ) * normalized_mode**2

    integral = float(np.sum(integrand) * dz)

    delta_k = (
        -(reference_wavenumber**2 / wavenumber)
        * integral
    )

    return delta_k


def apply_perturbation_correction(
    wavenumbers: np.ndarray,
    scenario,
    mode_functions: np.ndarray | None = None,
    delta_c_profile: np.ndarray | None = None,
    true_sound_speed: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    对一组估计波数施加扰动修正。

    参数
    ----
    wavenumbers:
        估计波数数组（rad/m）。
    scenario:
        算法使用的环境对象（须提供 omega / dz / sound_speed_profile()）。
    mode_functions:
        与 wavenumbers 对应的模态函数矩阵 [Nz, K]；
        为 None 时用 scenario 自行生成。
    delta_c_profile:
        真实声速偏差剖面 Δc(z)；为 None 时由 true_sound_speed 生成常偏差。
    true_sound_speed:
        真实（恒定）声速，用于生成 Δc(z) = true − assumed 常数剖面。

    返回
    ----
    corrected:
        修正后的波数数组。
    deltas:
        各模态的修正量 δk_n。
    """
    wavenumbers = np.asarray(wavenumbers, dtype=float)

    if wavenumbers.ndim != 1:
        raise ValueError("wavenumbers 必须是一维数组")

    omega = float(scenario.omega)
    dz = float(scenario.dz)

    assumed_speed = float(
        np.mean(
            np.asarray(
                scenario.sound_speed_profile(),
                dtype=float,
            )
        )
    )

    if delta_c_profile is None:
        if true_sound_speed is None:
            raise ValueError(
                "必须提供 delta_c_profile 或 true_sound_speed"
            )

        delta_c_profile = np.full(
            int(scenario.sound_speed_profile().size),
            float(true_sound_speed) - assumed_speed,
            dtype=float,
        )

    if mode_functions is None:
        c_profile = np.asarray(
            scenario.sound_speed_profile(),
            dtype=float,
        )

        mode_columns = [
            solve_mode(
                horizontal_wavenumber=float(k),
                omega=omega,
                sound_speed_profile=c_profile,
                dz=dz,
            )
            for k in wavenumbers
        ]

        mode_functions = np.column_stack(mode_columns)

    mode_functions = np.asarray(mode_functions, dtype=float)

    if mode_functions.ndim != 2:
        raise ValueError("mode_functions 必须是二维矩阵 [Nz, K]")

    if mode_functions.shape[1] != wavenumbers.size:
        raise ValueError("mode_functions 列数必须与 wavenumbers 一致")

    deltas = np.asarray(
        [
            wavenumber_perturbation(
                wavenumber=float(k),
                omega=omega,
                assumed_sound_speed=assumed_speed,
                mode_function=mode_functions[:, index],
                delta_c_profile=delta_c_profile,
                dz=dz,
            )
            for index, k in enumerate(wavenumbers)
        ],
        dtype=float,
    )

    return wavenumbers + deltas, deltas
