"""从零实现的一维匀加速运动卡尔曼滤波（详细注释版）。

状态向量 X = [位置, 速度, 加速度]

为什么不用现成的卡尔曼库？
因为本项目的意义就是让人看懂“卡尔曼到底在算什么”。
用 NumPy 手工写矩阵运算，代码里的每一行都能和教材公式一一对应，
面试时也能把每一步讲清楚。

卡尔曼滤波的总体思路可以浓缩成两句话：
1. 预测（Predict）：根据运动模型，猜测物体现在大概在哪；
2. 更新（Update）：拿传感器读数修正这个猜测，修正多少由“可信度”决定。
"""

import numpy as np


class ConstantAccelerationKalman:
    """对匀加速直线运动建模的卡尔曼滤波器。

    状态空间模型：
        X = [x, v, a]              # 位置、速度、加速度
        测量值 Z = H·X + 噪声      # 传感器只观测位置
    """

    def __init__(
        self,
        dt: float,                # 采样间隔（秒）
        measurement_noise_std: float,  # 传感器噪声标准差
        process_noise: float = 0.0001, # 过程噪声，见下方解释
    ):
        """初始化滤波器所需的全部矩阵。"""

        # 保存采样间隔，预测下一步位置时要用到。
        self.dt = dt

        # ---------- 1. 状态转移矩阵 F ----------
        # F 描述“如果没有任何新测量，根据运动模型，状态会如何变化”。
        #
        # 匀加速运动的一步递推公式：
        #   x_new = x + v·dt + ½·a·dt²
        #   v_new = v + a·dt
        #   a_new = a          （假设加速度本拍不变）
        #
        # 把上面三个公式写成矩阵乘法 X_new = F·X：
        #   [x_new]   [1   dt   ½dt²] [x]
        #   [v_new] = [0    1    dt ] [v]
        #   [a_new]   [0    0     1 ] [a]
        self.F = np.array(
            [
                [1.0, dt, 0.5 * dt * dt],   # 位置行
                [0.0, 1.0, dt],             # 速度行
                [0.0, 0.0, 1.0],            # 加速度行
            ]
        )

        # ---------- 2. 测量矩阵 H ----------
        # H 描述“状态 X 中，哪些量能被传感器直接看到”。
        # 本项目中传感器只测位置，所以 H = [1, 0, 0]：
        #   预测测量值 = H·X = 位置
        self.H = np.array([[1.0, 0.0, 0.0]])

        # ---------- 3. 测量噪声协方差 R ----------
        # R 表示“我对传感器读数的信任程度”。
        # 如果噪声标准差是 σ，那么方差就是 σ²（方差 = 标准差的平方）。
        # R 越大 = 越不相信传感器，卡尔曼增益就会越小。
        self.R = np.array([[measurement_noise_std**2]])

        # ---------- 4. 过程噪声协方差 Q ----------
        # 现实中没有哪个模型 100% 准确：真实加速度可能略有变化，
        # 或者存在我们没建模的小扰动。Q 就是给这些“模型误差”留的余地。
        #
        # g 表示一个“加速度扰动”会如何影响三个状态：
        #   对位置的影响 = ½·dt²
        #   对速度的影响 = dt
        #   对加速度的影响 = 1
        # Q = process_noise · (g·gᵀ) 把一维扰动展开成 3×3 协方差矩阵。
        g = np.array([[0.5 * dt * dt], [dt], [1.0]])
        self.Q = process_noise * (g @ g.T)

        # 状态与协方差先设为 None，等收到第一个测量值后再初始化。
        self.state = None        # 当前状态估计 X = [位置, 速度, 加速度]
        self.covariance = None   # 当前估计的不确定性矩阵 P

    def _initialize(self, first_measurement: float):
        """用第一个测量值初始化滤波器。

        卡尔曼滤波必须从一个初始状态出发。第一个读数给了位置，
        所以位置直接取它；速度和加速度还没有任何信息，先设为 0，
        让滤波器在后续几步中自动收敛。
        """

        self.state = np.array([first_measurement, 0.0, 0.0])

        # 初始协方差 P 表示“我对初始状态的信心”。
        # 对角线越大 = 越不确定。
        # - 位置：我们确实“量”了它，所以给较小值；
        # - 速度/加速度：完全未知，给 1.0，表示允许滤波器大幅修正。
        self.covariance = np.diag(
            [
                max(self.R[0, 0], 1e-6),  # 位置不确定性 ≈ 测量方差
                1.0,                        # 速度不确定性
                1.0,                        # 加速度不确定性
            ]
        )

    def step(self, measurement: float):
        """处理一个新的测量值，并返回更新后的状态。

        这是卡尔曼滤波最核心的一步，包含“预测 + 更新”。
        """

        # ========== 第一步：预测（Predict） ==========
        # 用运动模型 F 把上一拍状态推到当前时刻：
        #   X_pred = F·X
        # 即使这一拍还没读到传感器，我们也知道物体大约该在哪了。
        self.state = self.F @ self.state

        # 不确定性也会随时间增长：
        #   P_pred = F·P·Fᵀ + Q
        # F·P·Fᵀ 表示原来的不确定性经过运动模型被放大/变换，
        # + Q 表示模型本身的不完美又额外增加了一点不确定性。
        self.covariance = (
            self.F @ self.covariance @ self.F.T + self.Q
        )

        # ========== 第二步：更新（Update） ==========
        # 现在读到传感器数值 measurement。
        #
        # 1) 计算“新息/残差” innovation：
        #    innovation = 实际测量值 - 模型预测的测量值
        #    H·state 就是把状态里的位置“取出来”和测量值比较。
        innovation = measurement - (self.H @ self.state)[0]

        # 2) 计算这个残差的不确定性 S：
        #    S = H·P·Hᵀ + R
        #    它同时包含状态不确定性和测量噪声。
        innovation_covariance = (
            self.H @ self.covariance @ self.H.T + self.R
        )[0, 0]

        # 3) 计算卡尔曼增益 K（本项目最核心的一个数）：
        #    K = P·Hᵀ / S
        #    K 决定“模型预测”和“传感器读数”各信多少：
        #    - 测量噪声 R 很小 → S 小 → K 大 → 更相信传感器；
        #    - 测量噪声 R 很大 → S 大 → K 小 → 更相信模型。
        #    因为我们只测一维位置，增益是长度为 3 的向量，
        #    分别表示位置/速度/加速度各要修正多少。
        gain = (
            self.covariance @ self.H.T
        )[:, 0] / innovation_covariance

        # 4) 用增益加权更新状态：
        #    X_new = X_pred + K · innovation
        #    innovation 是“传感器比模型多知道的信息”，
        #    K 决定把其中多少信息吸收进状态。
        self.state = self.state + gain * innovation

        # 5) 更新协方差：采用数值更稳定的 Joseph 形式。
        #    A = I - K·H
        #    P_new = A·P·Aᵀ + K·R·Kᵀ
        # 它和常见的简化式 (I-K·H)·P 在数学上等价，
        # 但长期迭代时能更好地保持协方差矩阵的对称正定性。
        update_matrix = np.eye(3) - np.outer(gain, self.H[0])
        self.covariance = (
            update_matrix @ self.covariance @ update_matrix.T
            + np.outer(gain, gain) * self.R[0, 0]
        )

        # 返回副本而不是内部数组，防止调用方意外修改内部状态。
        return self.state.copy()

    def filter_all(self, measurements: np.ndarray) -> dict:
        """对整段测量序列逐点滤波，返回完整的状态估计序列。

        参数 measurements：传感器测到的一串位置。
        返回值是一个字典，包含估计出的位置、速度、加速度数组。
        """

        # 预先分配三个与测量序列等长的数组，避免循环里反复扩容。
        positions = np.empty_like(measurements)
        velocities = np.empty_like(measurements)
        accelerations = np.empty_like(measurements)

        # 用第一个读数初始化；第一个时刻只能“相信”测量值本身。
        self._initialize(measurements[0])
        positions[0] = measurements[0]
        velocities[0] = 0.0     # 尚无速度信息，先填 0
        accelerations[0] = 0.0  # 尚无加速度信息，先填 0

        # 从第二个读数开始，每个新测量都走一遍“预测+更新”。
        # 这就是“在线滤波”：边收到数据边输出估计，
        # 不需要等整段数据都到齐，符合真实传感器的使用场景。
        for i in range(1, len(measurements)):
            state = self.step(measurements[i])
            positions[i], velocities[i], accelerations[i] = state

        # 把三类状态分别返回，方便上层画图、算误差。
        return {
            "position": positions,
            "velocity": velocities,
            "acceleration": accelerations,
        }