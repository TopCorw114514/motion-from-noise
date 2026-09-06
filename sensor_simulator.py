"""传感器与运动模拟模块。

这个文件回答一个问题：
“既然我们要演示‘如何从噪声里还原真实运动’，那真实运动和传感器噪声
从哪里来？”

答案：先用物理公式生成一条完全干净的“真实轨迹”（Ground Truth），
再往上面叠加随机噪声，模拟传感器读数。

这样做的好处是：
1. 我们手里有“标准答案”，可以定量评价算法好坏；
2. 结果可以用随机种子固定，保证每次运行可复现；
3. 以后接入真实传感器时，只需要把本文件的输出替换成实测数据，
   下游的卡尔曼滤波代码完全不用改。
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SimulationConfig:
    """模拟参数（物理量全部使用国际单位制：米、秒）。

    使用 dataclass 的好处是：参数集中放在一起，调用方一目了然，
    以后想改实验条件（比如加大噪声、换一条轨迹）时不用到处找数字。
    """

    dt: float = 0.02            # 采样间隔（秒）：每 0.02 秒传感器读一次数
    duration: float = 5.0       # 总时长（秒）：本次一共模拟 5 秒
    x0: float = 1.0             # 初始位置（米）：t=0 时物体在哪
    v0: float = 0.6             # 初速度（米/秒）：t=0 时物体运动多快
    acceleration: float = -0.4  # 加速度（米/秒²）：负号表示在减速/反向加速
    noise_std: float = 0.08     # 传感器噪声标准差（米）：读数随机波动幅度
    seed: int = 42              # 随机种子：固定后每次生成的噪声序列完全一样


@dataclass
class SimulationResult:
    """一次模拟返回的全部数据。

    字段说明：
    - times: 每个采样时刻 t（秒）
    - true_position: 真实位置，即没有噪声的理想轨迹
    - true_velocity: 真实速度（卡尔曼要估计的目标之一）
    - true_acceleration: 真实加速度
    - measured_position: 传感器读数 = 真实位置 + 噪声
    - noise_std: 本次使用的噪声标准差，方便下游算法自动设置参数
    """

    times: np.ndarray
    true_position: np.ndarray
    true_velocity: np.ndarray
    true_acceleration: np.ndarray
    measured_position: np.ndarray
    noise_std: float


def simulate_position_sensor(config: SimulationConfig) -> SimulationResult:
    """生成“真实轨迹 + 噪声测量”的完整模拟数据。"""

    # 用指定的随机种子创建一个随机数生成器。
    # 只要 seed 相同，后续 random 调用产生的噪声就完全相同，
    # 这样别人运行同一个文件能得到一模一样的实验报告。
    rng = np.random.default_rng(config.seed)

    # 生成采样时刻 t = 0, dt, 2*dt, 3*dt, ...
    # 例如 dt=0.02、duration=5 秒时，会得到 250 个采样点。
    times = np.arange(0.0, config.duration, config.dt)

    # ---------- 1. 真实轨迹（物理模型） ----------
    # 匀加速直线运动的高中物理公式：
    #   x(t) = x0 + v0·t + ½·a·t²
    # NumPy 的向量化写法：times 是数组时，公式会对每一个 t 同时计算。
    true_position = (
        config.x0
        + config.v0 * times
        + 0.5 * config.acceleration * times**2
    )

    # 速度是位置的导数：v(t) = v0 + a·t
    # 这个量传感器通常“测不到”，只能靠卡尔曼滤波估计出来。
    true_velocity = config.v0 + config.acceleration * times

    # 匀加速运动中加速度是常数，所以每个时刻的真实加速度都相同。
    true_acceleration = np.full_like(times, config.acceleration)

    # ---------- 2. 传感器读数（真实轨迹 + 噪声） ----------
    # 用正态分布生成均值为 0、标准差为 noise_std 的随机噪声。
    # 为什么用高斯噪声？因为卡尔曼滤波的基本假设之一就是
    # 测量噪声近似服从高斯分布，这也是现实中许多传感器的常见模型。
    measured_position = true_position + rng.normal(
        0.0, config.noise_std, size=times.shape
    )

    # 把生成好的所有数据打包返回。
    return SimulationResult(
        times=times,
        true_position=true_position,
        true_velocity=true_velocity,
        true_acceleration=true_acceleration,
        measured_position=measured_position,
        noise_std=config.noise_std,
    )