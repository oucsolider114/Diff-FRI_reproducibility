"""
M3：子空间匹配射法初始化（JASA Express Lett. 1, 126002 (2021), Eq.(23)
适配快照模型）。

思路：
    观测 Y = ΨC + N 的（降噪后）左奇异向量 Û 近似模态深度函数子空间；
    对每个 Û 列，在候选水平波数网格上，用余弦差分递推生成 ψ(z; k)
    （ψ(0)=0, ψ(dz)=1），最大化与 Û 的归一化相关，得到 K 个粗估波数；
    再以这些粗估波数为中心生成窄带原子初始化区间，替代均匀盲铺。

预期收益：
    - 初始化从"均匀盲铺 + 5 次重启"变为"数据引导 + 更少重启"；
    - 原子初始即落在物理根吸引域附近，raw 成功率提升、计算量下降。
"""

from __future__ import annotations

import numpy as np

from mode_fri.mode_solver import solve_mode


def subspace_shooting_initialization(
    data: np.ndarray,
    scenario,
    number_of_modes: int,
    k_min: float | None = None,
    k_max: float | None = None,
    scan_points: int = 3000,
    mode_solver=None,
) -> np.ndarray:
    """
    用数据左奇异向量 + 余弦差分射法匹配估计 K 个粗波数。

    参数
    ----
    data:
        复声压数据矩阵 [Nz, L]（建议先经 Cadzow 降噪）。
    scenario:
        ShallowWaterScenario 场景。
    number_of_modes:
        目标模态数 K。
    k_min / k_max:
        候选波数扫描区间，默认 [1e-3, 0.9999·k0]。
    scan_points:
        扫描网格点数，分辨率约 (k_max-k_min)/scan_points。
    mode_solver:
        生成模态函数的可调用对象，签名 (k, omega, c_profile, dz) -> psi；
        默认使用压力释放 solve_mode；可渗透海底（M6/M9）时传入
        solve_mode_impedance 的包装（如 lambda k, o, c, dz: solve_mode_impedance(k, o, c, dz, ...)）。

    返回
    ----
    estimates:
        按波数降序排列的 K 个粗估波数。
    """
    if mode_solver is None:
        mode_solver = solve_mode

    data = np.asarray(data)

    if data.ndim != 2:
        raise ValueError("data 必须是二维矩阵 [Nz, L]")

    if number_of_modes <= 0:
        raise ValueError("number_of_modes 必须大于 0")

    if number_of_modes > min(data.shape):
        raise ValueError("number_of_modes 不能超过数据矩阵的秩上界")

    reference_k = float(scenario.reference_wavenumber)

    lower = 1e-3 if k_min is None else float(k_min)
    upper = (
        reference_k * 0.9999
        if k_max is None
        else float(k_max)
    )

    if not 0.0 <= lower < upper:
        raise ValueError("必须满足 0 <= k_min < k_max")

    u, _, _ = np.linalg.svd(
        data,
        full_matrices=False,
    )

    grid = np.linspace(lower, upper, int(scan_points))
    grid_spacing = grid[1] - grid[0]

    c_profile = scenario.sound_speed_profile()
    omega = scenario.omega
    dz = scenario.dz

    # 预计算网格上全部归一化模态函数 [Nz, scan_points]
    psi_grid = np.column_stack(
        [
            mode_solver(
                horizontal_wavenumber=float(k),
                omega=omega,
                sound_speed_profile=c_profile,
                dz=dz,
            )
            for k in grid
        ]
    )

    psi_norms = np.linalg.norm(
        psi_grid,
        axis=0,
    )

    psi_norms = np.maximum(psi_norms, 1e-12)

    normalized_psi = psi_grid / psi_norms

    estimates: list[float] = []
    used_positions: list[float] = []

    for mode_index in range(number_of_modes):
        column = u[:, mode_index]
        column_norm = max(
            float(np.linalg.norm(column)),
            1e-12,
        )

        normalized_column = column / column_norm

        correlation = np.abs(
            normalized_column.conj() @ normalized_psi
        )

        order = np.argsort(correlation)[::-1]

        selected_k: float | None = None

        for index in order:
            candidate = float(grid[index])

            # 去重：与已选波数太近（5 倍网格间距）则跳过
            if any(
                abs(candidate - used) < 5.0 * grid_spacing
                for used in used_positions
            ):
                continue

            selected_k = candidate
            break

        if selected_k is None:
            selected_k = float(grid[int(order[0])])

        estimates.append(selected_k)
        used_positions.append(selected_k)

    estimates_array = np.asarray(estimates, dtype=float)

    return np.sort(estimates_array)[::-1]


def build_narrowband_initialization(
    centers: np.ndarray,
    number_of_atoms: int,
    k_min: float,
    k_max: float,
    bandwidth: float = 0.006,
) -> np.ndarray:
    """
    以若干粗估波数为中心生成窄带原子初始化。

    每个中心生成 number_of_atoms // K 个原子，均匀铺在
    [center - bandwidth, center + bandwidth] 内；数量不足时用
    [k_min, k_max] 均匀网格补齐，最后裁剪到合法区间并排序。

    参数
    ----
    centers:
        粗估波数中心，形状 [K]。
    number_of_atoms:
        原子总数 M。
    k_min / k_max:
        波数合法区间。
    bandwidth:
        单边带宽，默认 0.006（与 ROOT_ORDER 捕获半径同量级）。

    返回
    ----
    initial_k:
        长度 number_of_atoms 的初始化波数（升序）。
    """
    centers = np.asarray(centers, dtype=float)

    if centers.ndim != 1 or centers.size == 0:
        raise ValueError("centers 必须是非空一维数组")

    if number_of_atoms <= 0:
        raise ValueError("number_of_atoms 必须大于 0")

    if not 0.0 <= k_min < k_max:
        raise ValueError("必须满足 0 <= k_min < k_max")

    if bandwidth < 0.0:
        raise ValueError("bandwidth 不能为负")

    number_of_centers = centers.size
    atoms_per_center = max(
        1,
        number_of_atoms // number_of_centers,
    )

    atom_list: list[np.ndarray] = []

    for center in centers:
        if atoms_per_center == 1:
            positions = np.asarray([center], dtype=float)
        else:
            positions = np.linspace(
                center - bandwidth,
                center + bandwidth,
                atoms_per_center,
            )

        atom_list.append(positions)

    initial = np.concatenate(atom_list)

    if initial.size < number_of_atoms:
        extra = np.linspace(
            k_min,
            k_max,
            number_of_atoms - initial.size,
        )
        initial = np.concatenate([initial, extra])

    initial = np.clip(initial, k_min, k_max)

    return np.sort(initial)[:number_of_atoms]
