from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from mode_fri.mode_init import build_narrowband_initialization
from mode_fri.torch_mode_solver import (
    impedance_bottom_residual_torch,
    solve_modes_torch,
    solve_modes_torch_impedance,
)


@dataclass
class ModeFRINETOutput:
    """Mode-FRINET 一次前向传播的输出。"""

    reconstruction: torch.Tensor
    wavenumbers: torch.Tensor
    coefficients: torch.Tensor
    raw_modes: torch.Tensor
    normalized_modes: torch.Tensor
    bottom_residuals: torch.Tensor
    activities: torch.Tensor


class ModeFRINET(nn.Module):
    """
    过参数化连续简正波模型。

    给定 M 个连续候选水平波数：

        k_1, k_2, ..., k_M

    利用可微差分方程生成模态矩阵：

        Phi(k) ∈ R^[Nz × M]

    多组垂直阵数据重构为：

        Y_hat = Phi(k) C

    其中：
        C ∈ C^[M × L]

    每一行对应一个候选模态在全部快照中的复展开系数。
    """

    def __init__(
        self,
        number_of_atoms: int,
        number_of_snapshots: int,
        k_min: float,
        k_max: float,
        omega: float,
        sound_speed_profile: torch.Tensor,
        dz: float,
        random_seed: int = 20260731,
        initial_wavenumbers: np.ndarray | None = None,
        initialization_bandwidth: float = 0.006,
        bottom_config: dict | None = None,
    ) -> None:
        super().__init__()

        if number_of_atoms <= 0:
            raise ValueError("number_of_atoms 必须大于 0")

        if number_of_snapshots <= 0:
            raise ValueError("number_of_snapshots 必须大于 0")

        if not 0.0 <= k_min < k_max:
            raise ValueError("必须满足 0 <= k_min < k_max")

        if dz <= 0:
            raise ValueError("dz 必须大于 0")

        self.number_of_atoms = number_of_atoms
        self.number_of_snapshots = number_of_snapshots
        self.k_min = float(k_min)
        self.k_max = float(k_max)
        self.omega = float(omega)
        self.dz = float(dz)

        # M6：可渗透海底（可选）。None = 压力释放（默认，行为与之前完全一致）
        self.bottom_config = None

        if bottom_config is not None:
            water_density = float(
                bottom_config.get("water_density", 1000.0)
            )
            bottom_speed_value = float(
                bottom_config["bottom_sound_speed"]
            )
            bottom_density_value = float(
                bottom_config["bottom_density"]
            )
            learnable = bool(
                bottom_config.get("learnable", False)
            )

            self.register_buffer(
                "water_density",
                torch.as_tensor(
                    water_density,
                    dtype=torch.float64,
                ),
            )

            if learnable:
                self.bottom_sound_speed = nn.Parameter(
                    torch.tensor(
                        bottom_speed_value,
                        dtype=torch.float64,
                    )
                )
                self.bottom_density = nn.Parameter(
                    torch.tensor(
                        bottom_density_value,
                        dtype=torch.float64,
                    )
                )
            else:
                self.register_buffer(
                    "bottom_sound_speed",
                    torch.as_tensor(
                        bottom_speed_value,
                        dtype=torch.float64,
                    ),
                )
                self.register_buffer(
                    "bottom_density",
                    torch.as_tensor(
                        bottom_density_value,
                        dtype=torch.float64,
                    ),
                )

            self.bottom_config = {
                "learnable": learnable,
            }

        c_profile = torch.as_tensor(
            sound_speed_profile,
            dtype=torch.float64,
        )

        if c_profile.ndim != 1:
            raise ValueError("sound_speed_profile 必须是一维张量")

        self.register_buffer(
            "sound_speed_profile",
            c_profile,
        )

        # -------------------------------------------------
        # 候选波数初始化：
        #   默认：均匀分布在 [k_min, k_max] 内（均匀盲铺）；
        #   M3：以数据引导的粗估波数（initial_wavenumbers）为中心
        #       生成窄带初始化，替代均匀盲铺。
        # 使用 sigmoid 将无约束参数映射到物理区间
        # -------------------------------------------------
        if initial_wavenumbers is None:
            initial_k = torch.linspace(
                self.k_min,
                self.k_max,
                number_of_atoms,
                dtype=torch.float64,
            )
        else:
            initial_k = torch.as_tensor(
                build_narrowband_initialization(
                    centers=np.asarray(
                        initial_wavenumbers,
                        dtype=float,
                    ),
                    number_of_atoms=number_of_atoms,
                    k_min=self.k_min,
                    k_max=self.k_max,
                    bandwidth=initialization_bandwidth,
                ),
                dtype=torch.float64,
            )

        normalized_position = (
            (initial_k - self.k_min)
            / (self.k_max - self.k_min)
        )

        normalized_position = torch.clamp(
            normalized_position,
            min=1e-4,
            max=1.0 - 1e-4,
        )

        initial_k_raw = torch.logit(
            normalized_position
        )

        self.k_raw = nn.Parameter(initial_k_raw)

        # -------------------------------------------------
        # 复系数使用实部和虚部分开表示
        # -------------------------------------------------
        generator = torch.Generator()
        generator.manual_seed(random_seed)

        coefficient_real = (
            0.01
            * torch.randn(
                number_of_atoms,
                number_of_snapshots,
                dtype=torch.float64,
                generator=generator,
            )
        )

        coefficient_imag = (
            0.01
            * torch.randn(
                number_of_atoms,
                number_of_snapshots,
                dtype=torch.float64,
                generator=generator,
            )
        )

        self.coefficient_real = nn.Parameter(
            coefficient_real
        )

        self.coefficient_imag = nn.Parameter(
            coefficient_imag
        )

    def get_wavenumbers(self) -> torch.Tensor:
        """将无约束参数映射到合法波数区间。"""
        return (
            self.k_min
            + (self.k_max - self.k_min)
            * torch.sigmoid(self.k_raw)
        )

    def get_coefficients(self) -> torch.Tensor:
        """返回复数展开系数矩阵 C。"""
        return torch.complex(
            self.coefficient_real,
            self.coefficient_imag,
        )

    def forward(self) -> ModeFRINETOutput:
        wavenumbers = self.get_wavenumbers()

        if self.bottom_config is None:
            # 压力释放海底（默认，M1 余弦差分）
            raw_modes = solve_modes_torch(
                horizontal_wavenumbers=wavenumbers,
                omega=self.omega,
                sound_speed_profile=self.sound_speed_profile,
                dz=self.dz,
            )
        else:
            # M6：可渗透海底（流体半空间阻抗边界，c_b/ρ_b 可学习）
            raw_modes = solve_modes_torch_impedance(
                horizontal_wavenumbers=wavenumbers,
                omega=self.omega,
                sound_speed_profile=self.sound_speed_profile,
                dz=self.dz,
                water_density=self.water_density,
                bottom_sound_speed=self.bottom_sound_speed,
                bottom_density=self.bottom_density,
            )

        # -------------------------------------------------
        # 不同候选波数生成的模态幅值尺度差异很大。
        # 为了提高训练稳定性，对每列做 L2 归一化。
        # 模态缩放由复系数 C 自动吸收。
        # -------------------------------------------------
        mode_norms = torch.linalg.vector_norm(
            raw_modes,
            dim=0,
        )

        mode_norms = torch.clamp(
            mode_norms,
            min=1e-12,
        )

        normalized_modes = (
            raw_modes / mode_norms
        )

        coefficients = self.get_coefficients()

        complex_modes = torch.complex(
            normalized_modes,
            torch.zeros_like(normalized_modes),
        )

        reconstruction = (
            complex_modes @ coefficients
        )

        # 归一化海底边界残差
        if self.bottom_config is None:
            bottom_residuals = (
                raw_modes[-1, :] / mode_norms
            )
        else:
            bottom_residuals = impedance_bottom_residual_torch(
                psi=raw_modes,
                horizontal_wavenumbers=wavenumbers,
                omega=self.omega,
                sound_speed_profile=self.sound_speed_profile,
                dz=self.dz,
                water_density=self.water_density,
                bottom_sound_speed=self.bottom_sound_speed,
                bottom_density=self.bottom_density,
            )

        # 每个候选模态在全部快照上的总活动强度
        activities = torch.sqrt(
            torch.sum(
                torch.abs(coefficients) ** 2,
                dim=1,
            )
            + 1e-12
        )

        return ModeFRINETOutput(
            reconstruction=reconstruction,
            wavenumbers=wavenumbers,
            coefficients=coefficients,
            raw_modes=raw_modes,
            normalized_modes=normalized_modes,
            bottom_residuals=bottom_residuals,
            activities=activities,
        )