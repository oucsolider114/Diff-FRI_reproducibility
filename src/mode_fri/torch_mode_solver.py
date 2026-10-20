import torch


def solve_modes_torch(
    horizontal_wavenumbers: torch.Tensor,
    omega: float,
    sound_speed_profile: torch.Tensor,
    dz: float,
) -> torch.Tensor:
    """
    使用 PyTorch 实现可微的浅海简正波有限差分递推。

    参数
    ----
    horizontal_wavenumbers:
        候选水平波数，可以是标量，也可以是一维张量 [M]。
    omega:
        角频率，单位 rad/s。
    sound_speed_profile:
        深度方向声速剖面，一维张量 [Nz]。
    dz:
        深度网格间隔，单位 m。

    返回
    ----
    psi:
        模态函数矩阵，形状为 [Nz, M]。

        每一列对应一个候选水平波数生成的模态函数。

    边界和归一化条件
    ----------------
    psi(0) = 0
    psi(dz) = 1

    差分格式
    --------
    余弦差分（JASA Express Lett. 1, 126002 (2021), Eq.(12)）：
        ψ(z+h) + ψ(z−h) − 2·cos(h·√(ω²/c² − k²))·ψ(z) = 0
    在 c(z) 恒定区间内对任意有限步长 h 精确；
    h→0 时退化为经典差分 2 − h²·q。q<0（倏逝区）时
    cos(i·x) = cosh(x)，取 cosh 分支保证数值稳定且保持可微。
    """
    if not isinstance(horizontal_wavenumbers, torch.Tensor):
        horizontal_wavenumbers = torch.as_tensor(
            horizontal_wavenumbers,
            dtype=torch.float64,
        )

    # 统一变成一维 [M]
    k = horizontal_wavenumbers.reshape(-1)

    if k.numel() == 0:
        raise ValueError("horizontal_wavenumbers 不能为空")

    if torch.any(k < 0):
        raise ValueError("水平波数不能小于 0")

    if dz <= 0:
        raise ValueError("dz 必须大于 0")

    c_profile = torch.as_tensor(
        sound_speed_profile,
        dtype=k.dtype,
        device=k.device,
    )

    if c_profile.ndim != 1:
        raise ValueError("sound_speed_profile 必须是一维张量")

    if c_profile.numel() < 3:
        raise ValueError("深度网格点数不能少于 3")

    if torch.any(c_profile <= 0):
        raise ValueError("声速剖面中的所有值必须大于 0")

    omega_tensor = torch.as_tensor(
        omega,
        dtype=k.dtype,
        device=k.device,
    )

    dz_tensor = torch.as_tensor(
        dz,
        dtype=k.dtype,
        device=k.device,
    )

    # 海面边界 psi(0)=0
    psi_previous = torch.zeros_like(k)

    # 归一化条件 psi(dz)=1
    psi_current = torch.ones_like(k)

    psi_values = [
        psi_previous,
        psi_current,
    ]

    # 注意：这里不用原位赋值，保证自动微分计算图完整
    for depth_index in range(1, c_profile.numel() - 1):
        local_sound_speed = c_profile[depth_index]

        q_value = (
            omega_tensor**2 / local_sound_speed**2
            - k**2
        )

        # 余弦差分系数（JASA Express Lett. 1, 126002 (2021), Eq.(12)）：
        #     2·cos(h·√(ω²/c² − k²))
        # 对任意有限步长 h 精确；q<0（倏逝区）时 cos(i·x)=cosh(x)，
        # 用 torch.where 分支出 cosh，保证数值稳定且保持可微。
        sqrt_abs_q = torch.sqrt(
            torch.abs(q_value)
        )

        difference_coefficient = torch.where(
            q_value >= 0.0,
            2.0 * torch.cos(
                dz_tensor * sqrt_abs_q
            ),
            2.0 * torch.cosh(
                dz_tensor * sqrt_abs_q
            ),
        )

        psi_next = (
            difference_coefficient * psi_current
            - psi_previous
        )

        psi_values.append(psi_next)

        psi_previous = psi_current
        psi_current = psi_next

    psi = torch.stack(
        psi_values,
        dim=0,
    )

    return psi


def normalized_bottom_residual_torch(
    psi: torch.Tensor,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """
    计算归一化海底边界残差。

    直接使用 psi(H) 时，不同模态的整体幅值差异较大，
    会导致梯度尺度不稳定。

    因此使用：

        residual = psi(H) / ||psi||_2

    返回形状为 [M]。
    """
    if psi.ndim != 2:
        raise ValueError("psi 必须是形状 [Nz, M] 的二维张量")

    mode_norm = torch.linalg.vector_norm(
        psi,
        dim=0,
    )

    mode_norm = torch.clamp(
        mode_norm,
        min=epsilon,
    )

    return psi[-1, :] / mode_norm


def bottom_residual_loss(
    psi: torch.Tensor,
) -> torch.Tensor:
    """
    海底压力释放边界损失：

        L_bottom = mean[(psi(H)/||psi||)^2]
    """
    residual = normalized_bottom_residual_torch(psi)

    return torch.mean(residual**2)


# =========================================================
# M6：可渗透海底（流体半空间阻抗边界，可微）
# =========================================================
def solve_modes_torch_impedance(
    horizontal_wavenumbers: torch.Tensor,
    omega: float,
    sound_speed_profile: torch.Tensor,
    dz: float,
    water_density: torch.Tensor | float = 1000.0,
    bottom_sound_speed: torch.Tensor | float = 1753.0,
    bottom_density: torch.Tensor | float = 1957.0,
) -> torch.Tensor:
    """
    可渗透海底下可微的水层模态求解。

    与 `solve_modes_torch`（压力释放）共用水层余弦差分递推，
    仅底部边界条件不同（由 `impedance_bottom_residual_torch` 表达）。

    参数
    ----
    horizontal_wavenumbers:
        候选水平波数 [M]。
    omega:
        角频率（rad/s）。
    sound_speed_profile:
        水层声速剖面 [Nz]。
    dz:
        深度间隔（m）。
    water_density / bottom_sound_speed / bottom_density:
        水/海底密度（kg/m³）与海底声速（m/s）；
        可为可学习张量（requires_grad）。

    返回
    ----
    psi:
        模态函数矩阵 [Nz, M]（ψ(0)=0, ψ(dz)=1 归一化）。
    """
    if not isinstance(horizontal_wavenumbers, torch.Tensor):
        horizontal_wavenumbers = torch.as_tensor(
            horizontal_wavenumbers,
            dtype=torch.float64,
        )

    k = horizontal_wavenumbers.reshape(-1)

    if k.numel() == 0:
        raise ValueError("horizontal_wavenumbers 不能为空")

    if torch.any(k < 0):
        raise ValueError("水平波数不能小于 0")

    if dz <= 0:
        raise ValueError("dz 必须大于 0")

    c_profile = torch.as_tensor(
        sound_speed_profile,
        dtype=k.dtype,
        device=k.device,
    )

    if c_profile.ndim != 1:
        raise ValueError("sound_speed_profile 必须是一维张量")

    if c_profile.numel() < 3:
        raise ValueError("深度网格点数不能少于 3")

    omega_tensor = torch.as_tensor(
        omega,
        dtype=k.dtype,
        device=k.device,
    )

    dz_tensor = torch.as_tensor(
        dz,
        dtype=k.dtype,
        device=k.device,
    )

    water_rho = torch.as_tensor(
        water_density,
        dtype=k.dtype,
        device=k.device,
    )

    bottom_c = torch.as_tensor(
        bottom_sound_speed,
        dtype=k.dtype,
        device=k.device,
    )

    bottom_rho = torch.as_tensor(
        bottom_density,
        dtype=k.dtype,
        device=k.device,
    )

    psi_previous = torch.zeros_like(k)
    psi_current = torch.ones_like(k)

    psi_values = [
        psi_previous,
        psi_current,
    ]

    for depth_index in range(1, c_profile.numel() - 1):
        local_sound_speed = c_profile[depth_index]

        q_value = (
            omega_tensor**2 / local_sound_speed**2
            - k**2
        )

        sqrt_abs_q = torch.sqrt(
            torch.abs(q_value)
        )

        difference_coefficient = torch.where(
            q_value >= 0.0,
            2.0 * torch.cos(dz_tensor * sqrt_abs_q),
            2.0 * torch.cosh(dz_tensor * sqrt_abs_q),
        )

        psi_next = (
            difference_coefficient * psi_current
            - psi_previous
        )

        psi_values.append(psi_next)

        psi_previous = psi_current
        psi_current = psi_next

    return torch.stack(psi_values, dim=0)


def impedance_bottom_residual_torch(
    psi: torch.Tensor,
    horizontal_wavenumbers: torch.Tensor,
    omega: float,
    sound_speed_profile: torch.Tensor,
    dz: float,
    water_density: torch.Tensor | float = 1000.0,
    bottom_sound_speed: torch.Tensor | float = 1753.0,
    bottom_density: torch.Tensor | float = 1957.0,
) -> torch.Tensor:
    """
    可渗透海底阻抗边界残差（可微、平滑），返回形状 [M]：

        F_m = |ψ'(H) + (ρ_w/ρ_b)·√(k_m² − (ω/c_b)²)·ψ(H)| / ||ψ_m||₂

    使用**复数平方根**（主分支 Re≥0）：捕获模态（k > ω/c_b）时 √(·) 为实，
    与经典阻抗条件一致；泄漏模态（k < ω/c_b）时 √(·) 为纯虚数，|F| 平滑，
    梯度不会因硬 clamp 而消失——c_b 可学习参数可跨泄漏/捕获区收敛。

    ψ'(H) 用三点单侧差分（二阶精度）。c_b / ρ_b 可学习（requires_grad）。
    """
    if psi.ndim != 2:
        raise ValueError("psi 必须是形状 [Nz, M] 的二维张量")

    if psi.shape[0] < 3:
        raise ValueError("psi 深度维度不能小于 3")

    k = torch.as_tensor(
        horizontal_wavenumbers,
        dtype=psi.dtype,
        device=psi.device,
    ).reshape(-1)

    if k.numel() != psi.shape[1]:
        raise ValueError("k 长度必须与 psi 列数一致")

    omega_tensor = torch.as_tensor(
        omega,
        dtype=psi.dtype,
        device=psi.device,
    )

    dz_tensor = torch.as_tensor(
        dz,
        dtype=psi.dtype,
        device=psi.device,
    )

    water_rho = torch.as_tensor(
        water_density,
        dtype=psi.dtype,
        device=psi.device,
    )

    bottom_c = torch.as_tensor(
        bottom_sound_speed,
        dtype=psi.dtype,
        device=psi.device,
    )

    bottom_rho = torch.as_tensor(
        bottom_density,
        dtype=psi.dtype,
        device=psi.device,
    )

    bottom_wavenumber = omega_tensor / bottom_c

    gamma_squared = k**2 - bottom_wavenumber**2

    # 复数平方根（主分支 Re>=0），跨捕获/泄漏区平滑可微
    gamma_complex = torch.sqrt(
        gamma_squared.to(torch.complex128)
    )

    psi_derivative = (
        3.0 * psi[-1, :]
        - 4.0 * psi[-2, :]
        + psi[-3, :]
    ) / (2.0 * dz_tensor)

    psi_bottom = psi[-1, :]

    mode_norm = torch.clamp(
        torch.linalg.vector_norm(psi, dim=0),
        min=1e-12,
    )

    boundary_value = (
        psi_derivative.to(torch.complex128)
        + (water_rho / bottom_rho)
        * gamma_complex
        * psi_bottom.to(torch.complex128)
    )

    # 取模使残差为实值（|F| 在根处呈 V 形，梯度指向根，适合优化）
    residual = torch.abs(boundary_value)

    return residual / mode_norm


def impedance_bottom_residual_loss(
    psi: torch.Tensor,
    horizontal_wavenumbers: torch.Tensor,
    omega: float,
    sound_speed_profile: torch.Tensor,
    dz: float,
    water_density: torch.Tensor | float = 1000.0,
    bottom_sound_speed: torch.Tensor | float = 1753.0,
    bottom_density: torch.Tensor | float = 1957.0,
) -> torch.Tensor:
    """
    可渗透海底阻抗边界损失：

        L_bottom = mean[R_m²]
    """
    residual = impedance_bottom_residual_torch(
        psi=psi,
        horizontal_wavenumbers=horizontal_wavenumbers,
        omega=omega,
        sound_speed_profile=sound_speed_profile,
        dz=dz,
        water_density=water_density,
        bottom_sound_speed=bottom_sound_speed,
        bottom_density=bottom_density,
    )

    return torch.mean(residual**2)