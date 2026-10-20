from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ShallowWaterScenario:
    """
    浅海简正波测试场景。

    当前采用最简单的常声速、海面和海底压力释放模型，
    主要用于验证差分递推程序是否正确。
    """

    water_depth: float = 100.0   # 水深，单位：m
    dz: float = 1.0              # 深度离散间隔，单位：m
    frequency: float = 50.0      # 单频信号频率，单位：Hz
    sound_speed: float = 1500.0  # 常声速，单位：m/s

    def __post_init__(self) -> None:
        if self.water_depth <= 0:
            raise ValueError("water_depth 必须大于 0")

        if self.dz <= 0:
            raise ValueError("dz 必须大于 0")

        if self.frequency <= 0:
            raise ValueError("frequency 必须大于 0")

        if self.sound_speed <= 0:
            raise ValueError("sound_speed 必须大于 0")

        steps = self.water_depth / self.dz

        if not np.isclose(steps, round(steps)):
            raise ValueError(
                "water_depth 必须能够被 dz 整除，"
                f"当前 water_depth/dz={steps}"
            )

    @property
    def omega(self) -> float:
        """角频率 ω=2πf。"""
        return 2.0 * np.pi * self.frequency

    @property
    def reference_wavenumber(self) -> float:
        """介质中的参考总波数 k0=ω/c。"""
        return self.omega / self.sound_speed

    @property
    def z(self) -> np.ndarray:
        """深度离散网格，z=0 表示海面。"""
        number_of_steps = int(round(self.water_depth / self.dz))

        return np.arange(
            number_of_steps + 1,
            dtype=float,
        ) * self.dz

    def sound_speed_profile(self) -> np.ndarray:
        """
        生成常声速剖面。

        后续可将这里替换为随深度变化的真实声速剖面 c(z)。
        """
        return np.full(
            shape=self.z.shape,
            fill_value=self.sound_speed,
            dtype=float,
        )