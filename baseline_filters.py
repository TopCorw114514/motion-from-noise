"""供对比的简单滤波算法。

卡尔曼滤波到底“好”在哪？要证明它好，不能只自说自话，
必须放一个朴素的对照算法一起比。本文件实现的就是对照组：
因果滑动平均（moving average）。

滑动平均的思路非常直观：
“既然每个点都带噪声，那就把最近 N 个点平均一下，噪声就会被摊平。”

但它的缺点也很明显：平均必然引入滞后。窗口越长越平滑，
也越“跟不上”真实运动。卡尔曼滤波因为有运动模型，
可以在去噪的同时尽量不滞后。
"""

import numpy as np


def causal_moving_average(
    signal: np.ndarray, window_size: int = 21
) -> np.ndarray:
    """因果滑动平均：每个点只取“当前及过去”若干点的平均值。

    为什么叫“因果”（causal）？
    因为真实传感器必须实时处理，处理第 k 个读数时，
    未来的第 k+1 个读数还没有出现，绝不能偷看未来数据。

    与之相对的是“中心平均”——每个点取前后各一半，
    它虽然效果更好，但需要等整段数据到齐才能算，不是实时算法。
    为了让对照公平，这里必须用因果版本，和卡尔曼一样只用历史信息。

    实现细节：
    1. 在信号开头补 window_size-1 个“边缘值”，
       保证第 0 个点也能算出平均值；
    2. 用长度为 window_size、权值都是 1/window_size 的卷积核
       与信号做卷积；
    3. 这样第 i 个输出 = (x[i-window+1] + ... + x[i]) / window_size。
    """

    # np.pad(signal, (left, right), mode="edge")：
    # 在数组开头复制边界值 padding，尾部不补。
    # 例如信号 [1,2,3]、窗口 3，则变成 [1,1,2,3,3]。
    padded = np.pad(signal, (window_size - 1, 0), mode="edge")

    # 卷积核：window_size 个 1，每个除以 window_size，
    # 等价于“滑动窗口内所有数相加后再除以窗口大小”。
    kernel = np.ones(window_size) / window_size

    # mode="valid" 表示只输出卷积核完全覆盖数据的部分；
    # 因为前面手动补了边，输出长度恰好等于原始信号长度。
    return np.convolve(padded, kernel, mode="valid")