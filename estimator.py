"""误差指标计算与运动参数辨识。"""

import numpy as np


def rmse(estimated: np.ndarray, truth: np.ndarray) -> float:
    """均方根误差。"""
    return float(np.sqrt(np.mean((estimated - truth) ** 2)))


def fit_motion_parameters(
    times: np.ndarray, positions: np.ndarray
) -> dict:
    """用二次多项式拟合 x(t) = p2·t² + p1·t + p0。

    物理关系：
      x(t) = x0 + v0·t + ½·a·t²

    因此：
      p0 = x0
      p1 = v0
      p2 = ½·a   =>   a = 2·p2
    """

    p2, p1, p0 = np.polyfit(times, positions, 2)
    return {
        "x0": float(p0),
        "v0": float(p1),
        "acceleration": float(2.0 * p2),
    }
