"""模拟一个做匀加速直线运动的物体，以及一台带高斯噪声的位置传感器。"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SimulationConfig:
    """模拟参数（物理量采用国际单位制：米、秒）。"""

    dt: float = 0.02          # 采样间隔（秒）
    duration: float = 5.0     # 总时长（秒）
    x0: float = 1.0           # 初始位置（米）
    v0: float = 0.6           # 初速度（米/秒）
    acceleration: float = -0.4  # 加速度（米/秒²）
    noise_std: float = 0.08   # 传感器噪声标准差（米）
    seed: int = 42            # 随机种子，保证结果可复现


@dataclass
class SimulationResult:
    """一次模拟的全部数据。"""

    times: np.ndarray
    true_position: np.ndarray
    true_velocity: np.ndarray
    true_acceleration: np.ndarray
    measured_position: np.ndarray
    noise_std: float


def simulate_position_sensor(config: SimulationConfig) -> SimulationResult:
    """生成真实轨迹，再叠加高斯噪声作为传感器测量值。"""

    rng = np.random.default_rng(config.seed)
    times = np.arange(0.0, config.duration, config.dt)

    true_position = (
        config.x0
        + config.v0 * times
        + 0.5 * config.acceleration * times**2
    )
    true_velocity = config.v0 + config.acceleration * times
    true_acceleration = np.full_like(times, config.acceleration)

    measured_position = true_position + rng.normal(
        0.0, config.noise_std, size=times.shape
    )

    return SimulationResult(
        times=times,
        true_position=true_position,
        true_velocity=true_velocity,
        true_acceleration=true_acceleration,
        measured_position=measured_position,
        noise_std=config.noise_std,
    )
