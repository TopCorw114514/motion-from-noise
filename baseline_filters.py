"""供对比的简单滤波算法。"""

import numpy as np


def causal_moving_average(
    signal: np.ndarray, window_size: int = 21
) -> np.ndarray:
    """因果滑动平均：每个点只取“当前及过去”若干点的平均值。

    真实传感器必须实时处理，不能等待未来的数据，
    因此这个对照算法和卡尔曼滤波一样只能用历史信息。
    """

    padded = np.pad(signal, (window_size - 1, 0), mode="edge")
    kernel = np.ones(window_size) / window_size
    return np.convolve(padded, kernel, mode="valid")
