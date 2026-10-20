import numpy as np


def solve_mode(
    horizontal_wavenumber: float,
    omega: float,
    sound_speed_profile: np.ndarray,
    dz: float,
) -> np.ndarray:
    """
    根据给定的水平候选波数 k，通过有限差分递推生成模态函数。

    简正波方程：
        d²ψ/dz² + [ω²/c²(z) - k²]ψ = 0

    差分递推（余弦差分，JASA Express Lett. 1, 126002 (2021), Eq.(12)）：
        ψ[j+1] =
        2·cos(dz·√[ω²/c²(z_j) - k²])·ψ[j] - ψ[j-1]

    该格式在 c(z) 恒定区间内对任意有限步长 dz 精确；
    dz→0 时 2·cos(dz·√q) ≈ 2 - dz²·q 退化为经典差分格式。
    倏逝区（q<0）时 cos(i·x)=cosh(x)，取 cosh 分支保证数值稳定。

    边界与归一化条件：
        ψ(0) = 0
        ψ(dz) = 1

    参数
    ----
    horizontal_wavenumber:
        候选水平波数 k，单位 rad/m。
    omega:
        角频率 ω，单位 rad/s。
    sound_speed_profile:
        深度方向声速剖面 c(z)。
    dz:
        深度离散间隔。

    返回
    ----
    psi:
        对应候选波数的离散模态函数。
    """
    c_profile = np.asarray(
        sound_speed_profile,
        dtype=float,
    )

    if c_profile.ndim != 1:
        raise ValueError("sound_speed_profile 必须是一维数组")

    if c_profile.size < 3:
        raise ValueError("深度网格点数不能少于 3")

    if horizontal_wavenumber < 0:
        raise ValueError("horizontal_wavenumber 不能小于 0")

    if omega <= 0:
        raise ValueError("omega 必须大于 0")

    if dz <= 0:
        raise ValueError("dz 必须大于 0")

    if np.any(c_profile <= 0):
        raise ValueError("声速剖面中的所有值必须大于 0")

    psi = np.zeros(
        shape=c_profile.size,
        dtype=float,
    )

    # 海面压力释放边界
    psi[0] = 0.0

    # 归一化条件
    psi[1] = 1.0

    for depth_index in range(1, c_profile.size - 1):
        local_sound_speed = c_profile[depth_index]

        q_value = (
            omega**2 / local_sound_speed**2
            - horizontal_wavenumber**2
        )

        # 余弦差分系数（JASA Express Lett. 1, 126002 (2021), Eq.(12)）：
        #     ψ(z+h) + ψ(z−h) − 2·cos(h·√(ω²/c² − k²))·ψ(z) = 0
        # 在 c(z) 恒定区间内对任意有限步长 h 精确；
        # h→0 时 2·cos(h·√q) ≈ 2 − h²·q 退化为经典差分格式。
        # q<0（倏逝区）时 cos(i·x) = cosh(x)，取 cosh 分支保证数值稳定。
        if q_value >= 0.0:
            difference_coefficient = (
                2.0 * np.cos(dz * np.sqrt(q_value))
            )
        else:
            difference_coefficient = (
                2.0 * np.cosh(dz * np.sqrt(-q_value))
            )

        psi[depth_index + 1] = (
            difference_coefficient * psi[depth_index]
            - psi[depth_index - 1]
        )

        if not np.isfinite(psi[depth_index + 1]):
            raise FloatingPointError(
                "模态递推过程中出现 NaN 或无穷大，"
                f"位置索引为 {depth_index + 1}"
            )

    return psi


def pressure_release_bottom_residual(
    psi: np.ndarray,
) -> float:
    """
    计算海底压力释放边界残差。

    当前临时采用：
        ψ(H)=0

    因此残差为：
        |ψ(H)|

    后续获得真实海底条件后，需要修改这一部分。
    """
    mode = np.asarray(psi)

    if mode.ndim != 1:
        raise ValueError("psi 必须是一维数组")

    if mode.size == 0:
        raise ValueError("psi 不能为空")

    return float(np.abs(mode[-1]))


def verify_surface_conditions(
    psi: np.ndarray,
    tolerance: float = 1e-12,
) -> bool:
    """
    检查海面边界和归一化条件是否满足。

    ψ(0)=0
    ψ(dz)=1
    """
    mode = np.asarray(psi)

    if mode.size < 2:
        return False

    return bool(
        np.isclose(mode[0], 0.0, atol=tolerance)
        and np.isclose(mode[1], 1.0, atol=tolerance)
    )


# =========================================================
# M6：可渗透海底（流体半空间阻抗边界）
# =========================================================
def solve_mode_impedance(
    horizontal_wavenumber: float,
    omega: float,
    sound_speed_profile: np.ndarray,
    dz: float,
    water_density: float = 1000.0,
    bottom_sound_speed: float = 1753.0,
    bottom_density: float = 1957.0,
) -> np.ndarray:
    """
    可渗透海底（流体半空间）下生成水层内的模态函数。

    与 `solve_mode`（压力释放海底）的区别仅在于底部边界：
    压力释放要求 ψ(H)=0，而可渗透海底要求水层解与海底半空间解在
    z=H 处满足压力与法向质点速度连续（阻抗边界）：

        ψ'(H)/ψ(H) = −(ρ_w/ρ_b)·√(k² − ω²/c_b²)

    对捕获模态（k > ω/c_b），海底内的解为衰减指数 exp(−γ(z−H))，
    γ = √(k² − ω²/c_b²)。水层递推仍用余弦差分（M1）。

    参数
    ----
    horizontal_wavenumber:
        候选水平波数 k（rad/m）。
    omega:
        角频率（rad/s）。
    sound_speed_profile:
        水层声速剖面 c(z)，长度 Nz。
    dz:
        深度间隔（m）。
    water_density / bottom_sound_speed / bottom_density:
        水与海底密度（kg/m³）、海底声速（m/s）。

    返回
    ----
    psi:
        水层内的离散模态函数（ψ(0)=0, ψ(dz)=1 归一化），长度 Nz。
    """
    c_profile = np.asarray(sound_speed_profile, dtype=float)

    if c_profile.ndim != 1:
        raise ValueError("sound_speed_profile 必须是一维数组")

    if c_profile.size < 3:
        raise ValueError("深度网格点数不能少于 3")

    if horizontal_wavenumber < 0:
        raise ValueError("horizontal_wavenumber 不能小于 0")

    if omega <= 0:
        raise ValueError("omega 必须大于 0")

    if dz <= 0:
        raise ValueError("dz 必须大于 0")

    if np.any(c_profile <= 0):
        raise ValueError("声速剖面中的所有值必须大于 0")

    if water_density <= 0 or bottom_density <= 0:
        raise ValueError("密度必须大于 0")

    if bottom_sound_speed <= 0:
        raise ValueError("海底声速必须大于 0")

    psi = np.zeros(
        shape=c_profile.size,
        dtype=float,
    )

    psi[0] = 0.0
    psi[1] = 1.0

    for depth_index in range(1, c_profile.size - 1):
        local_sound_speed = c_profile[depth_index]

        q_value = (
            omega**2 / local_sound_speed**2
            - horizontal_wavenumber**2
        )

        if q_value >= 0.0:
            difference_coefficient = (
                2.0 * np.cos(dz * np.sqrt(q_value))
            )
        else:
            difference_coefficient = (
                2.0 * np.cosh(dz * np.sqrt(-q_value))
            )

        psi[depth_index + 1] = (
            difference_coefficient * psi[depth_index]
            - psi[depth_index - 1]
        )

        if not np.isfinite(psi[depth_index + 1]):
            raise FloatingPointError(
                "模态递推过程中出现 NaN 或无穷大，"
                f"位置索引为 {depth_index + 1}"
            )

    return psi


def impedance_bottom_residual(
    psi: np.ndarray,
    horizontal_wavenumber: float,
    omega: float,
    sound_speed_profile: np.ndarray,
    dz: float,
    water_density: float = 1000.0,
    bottom_sound_speed: float = 1753.0,
    bottom_density: float = 1957.0,
) -> float:
    """
    可渗透海底阻抗边界残差（射法求根的判定函数）。

    对捕获模态（k > ω/c_b），真实本征波数应满足边界条件：

        ψ'(H) + (ρ_w/ρ_b)·γ·ψ(H) = 0，
        γ = √(k² − ω²/c_b²)

    采用**乘积形式**（而非 ψ'/ψ 比值形式），避免 ψ(H)=0 处的极点，
    残差是 k 的连续函数，变号即对应本征波数。ψ'(H) 用三点单侧差分
    （二阶精度）：ψ'(H) ≈ (3ψ[N] − 4ψ[N−1] + ψ[N−2]) / (2·dz)。
    结果按 ||ψ||₂ 归一化，量纲与压力释放残差一致。

    极限：
        ρ_b→0   → 残差 ∝ ψ(H)，即压力释放；
        ρ_b→∞  → 残差 → ψ'(H)，即刚性海底。

    返回
    ----
    residual:
        F(k) 的值；F(k) 变号过零即对应本征波数。
    """
    mode = np.asarray(psi, dtype=float)

    if mode.ndim != 1 or mode.size < 3:
        raise ValueError("psi 必须是一维数组且长度不小于 3")

    bottom_wavenumber = omega / bottom_sound_speed

    gamma_squared = (
        horizontal_wavenumber**2 - bottom_wavenumber**2
    )

    if gamma_squared < 0.0:
        gamma = np.sqrt(np.abs(gamma_squared))
    else:
        gamma = np.sqrt(gamma_squared)

    psi_derivative = (
        3.0 * mode[-1]
        - 4.0 * mode[-2]
        + mode[-3]
    ) / (2.0 * dz)

    psi_bottom = mode[-1]

    mode_norm = max(float(np.linalg.norm(mode)), 1e-12)

    residual = (
        psi_derivative
        + (water_density / bottom_density)
        * gamma
        * psi_bottom
    )

    return float(residual / mode_norm)