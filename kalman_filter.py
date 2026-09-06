"""从零实现的一维匀加速运动卡尔曼滤波。

状态向量 X = [位置, 速度, 加速度]

本项目不使用任何现成的卡尔曼库，矩阵运算均由 NumPy 手工完成，
因此每一步都能在代码里和数学公式一一对应。
"""

import numpy as np


class ConstantAccelerationKalman:
    """对匀加速直线运动建模的卡尔曼滤波器。"""

    def __init__(
        self,
        dt: float,
        measurement_noise_std: float,
        process_noise: float = 0.0001,
    ):
        self.dt = dt

        # 状态转移矩阵 F：把上一拍的状态推到这一拍
        self.F = np.array(
            [
                [1.0, dt, 0.5 * dt * dt],
                [0.0, 1.0, dt],
                [0.0, 0.0, 1.0],
            ]
        )

        # 测量矩阵 H：传感器只能读到位置
        self.H = np.array([[1.0, 0.0, 0.0]])

        # 测量噪声协方差 R
        self.R = np.array([[measurement_noise_std**2]])

        # 过程噪声协方差 Q：加速度并不是绝对恒定，
        # 留一点“模型不确定性”让滤波器保持警觉
        g = np.array([[0.5 * dt * dt], [dt], [1.0]])
        self.Q = process_noise * (g @ g.T)

        # 初始状态：位置用第一个读数，速度/加速度先未知设为 0
        self.state = None
        self.covariance = None

    def _initialize(self, first_measurement: float):
        self.state = np.array([first_measurement, 0.0, 0.0])
        # 初始协方差：位置基本可信，速度/加速度置信度较低
        self.covariance = np.diag(
            [
                max(self.R[0, 0], 1e-6),
                1.0,
                1.0,
            ]
        )

    def step(self, measurement: float):
        """处理一个新的测量值，并返回更新后的状态。"""

        # 第一步：预测（用运动模型推算当前状态）
        self.state = self.F @ self.state
        self.covariance = (
            self.F @ self.covariance @ self.F.T + self.Q
        )

        # 第二步：更新（结合传感器读数）
        innovation = measurement - (self.H @ self.state)[0]
        innovation_covariance = (
            self.H @ self.covariance @ self.H.T + self.R
        )[0, 0]

        # 卡尔曼增益：模型与传感器各信多少
        gain = (
            self.covariance @ self.H.T
        )[:, 0] / innovation_covariance

        self.state = self.state + gain * innovation
        self.covariance = (
            np.eye(3) - np.outer(gain, self.H[0])
        ) @ self.covariance

        return self.state.copy()

    def filter_all(self, measurements: np.ndarray) -> dict:
        """对整段测量序列滤波，返回位置/速度/加速度估计。"""

        positions = np.empty_like(measurements)
        velocities = np.empty_like(measurements)
        accelerations = np.empty_like(measurements)

        # 用第一个读数完成初始化
        self._initialize(measurements[0])
        positions[0] = measurements[0]
        velocities[0] = 0.0
        accelerations[0] = 0.0

        for i in range(1, len(measurements)):
            state = self.step(measurements[i])
            positions[i], velocities[i], accelerations[i] = state

        return {
            "position": positions,
            "velocity": velocities,
            "acceleration": accelerations,
        }
