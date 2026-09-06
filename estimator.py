"""误差指标计算与运动参数辨识。

这一层回答两个问题：
1. “算法到底好多少？”——用 RMSE（均方根误差）量化；
2. “能不能从轨迹里反推出运动参数？”——用最小二乘拟合实现。

有了这两个工具，作品就不只是“画一条好看的曲线”，
而是能给出可以写进报告、经得起追问的数字结论。
"""

import numpy as np


def rmse(estimated: np.ndarray, truth: np.ndarray) -> float:
    """均方根误差（Root Mean Square Error）。

    计算步骤（对每个采样点）：
    1. 求误差：estimated - truth
    2. 平方：把正负误差都变成正数，同时放大大的误差
    3. 求平均：得到平均平方误差
    4. 开根号：把单位还原成米

    为什么用它而不是“平均绝对误差”？
    因为平方会让较大的误差受到更重惩罚，
    能更敏锐地反映滤波曲线是否出现明显偏离。
    """

    # 逐点误差取平方 -> 平均 -> 开根号。
    return float(np.sqrt(np.mean((estimated - truth) ** 2)))


def fit_motion_parameters(
    times: np.ndarray, positions: np.ndarray
) -> dict:
    """用二次多项式拟合 x(t)，反推出初始位置、初速度、加速度。

    物理背景：
    匀加速直线运动的位移公式是
        x(t) = x0 + v0·t + ½·a·t²

    如果把它看成一个关于 t 的二次多项式
        x(t) = p2·t² + p1·t + p0

    对比系数立刻得到：
        p0 = x0（初始位置）
        p1 = v0（初速度）
        p2 = ½·a，所以 a = 2·p2（加速度）

    np.polyfit(times, positions, 2) 做的事情就是“最小二乘拟合”：
    找一条二次曲线，使它与数据点的垂直误差平方和最小。

    为什么在“滤波后的数据”上拟合比在“原始噪声数据”上拟合更好？
    因为卡尔曼已经把大部分随机噪声去掉了，
    曲线形状更接近真实的二次轨迹，拟合出的参数自然更准。
    """

    # polyfit 返回最高次到最低次的系数：[p2, p1, p0]。
    p2, p1, p0 = np.polyfit(times, positions, 2)

    # 按上面的系数换算关系返回三个物理量。
    return {
        "x0": float(p0),                      # 初始位置（米）
        "v0": float(p1),                      # 初速度（米/秒）
        "acceleration": float(2.0 * p2),      # 加速度（米/秒²）
    }