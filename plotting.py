"""把实验结果画成一张清晰的对比图。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 自动保存图片，不依赖桌面窗口
import matplotlib.pyplot as plt
import numpy as np


def prepare_chinese_fonts():
    """优先使用常见中文字体，避免图中中文变成方块。"""
    from matplotlib import font_manager

    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "PingFang SC",
        "WenQuanYi Micro Hei",
    ]:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def save_comparison_figure(
    result,
    kalman_estimate: dict,
    moving_average: np.ndarray,
    output_path: Path,
    warmup_time: float = 2.0,
) -> Path:
    """生成并保存 2×2 对比图。"""

    prepare_chinese_fonts()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t = result.times
    warmup_mask = t >= warmup_time

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 8), constrained_layout=True
    )

    # 左上：整体看噪声与滤波
    ax = axes[0, 0]
    ax.plot(
        t,
        result.measured_position,
        ".", color="#d62728", markersize=2, alpha=0.55,
        label="原始传感器数据",
    )
    ax.plot(
        t,
        result.true_position,
        "-", color="black", linewidth=2,
        label="真实轨迹（Ground Truth）",
    )
    ax.plot(
        t,
        kalman_estimate["position"],
        "-", color="#2ca02c", linewidth=2,
        label="卡尔曼滤波",
    )
    ax.plot(
        t,
        moving_average,
        "-", color="#1f77b4", linewidth=1.5, alpha=0.9,
        label="移动平均（对照）",
    )
    ax.axvline(warmup_time, color="gray", linestyle="--", alpha=0.8)
    ax.text(
        warmup_time + 0.02,
        ax.get_ylim()[1] * 0.95,
        "滤波预热段",
        fontsize=9,
        color="gray",
    )
    ax.set_title("位置估计：噪声中的还原")
    ax.set_xlabel("时间 t（秒）")
    ax.set_ylabel("位置 x（米）")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # 右上：放大看细节
    ax = axes[0, 1]
    slice_ = t >= 2.0
    ax.plot(
        t[slice_],
        result.measured_position[slice_],
        ".", color="#d62728", markersize=2.5, alpha=0.5,
        label="原始数据",
    )
    ax.plot(
        t[slice_],
        result.true_position[slice_],
        "-", color="black", linewidth=2,
        label="真实轨迹",
    )
    ax.plot(
        t[slice_],
        kalman_estimate["position"][slice_],
        "-", color="#2ca02c", linewidth=2.2,
        label="卡尔曼滤波",
    )
    ax.plot(
        t[slice_],
        moving_average[slice_],
        "-", color="#1f77b4", linewidth=1.6,
        label="移动平均",
    )
    ax.set_title("位置估计（2 秒后放大）")
    ax.set_xlabel("时间 t（秒）")
    ax.set_ylabel("位置 x（米）")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # 左下：速度估计
    ax = axes[1, 0]
    ax.plot(
        t,
        result.true_velocity,
        "-", color="black", linewidth=2,
        label="真实速度",
    )
    ax.plot(
        t,
        kalman_estimate["velocity"],
        "-", color="#2ca02c", linewidth=2,
        label="卡尔曼估计速度",
    )
    ax.axvline(warmup_time, color="gray", linestyle="--", alpha=0.8)
    ax.set_title("速度估计（传感器没有直接测速度）")
    ax.set_xlabel("时间 t（秒）")
    ax.set_ylabel("速度 v（米/秒）")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # 右下：加速度估计
    ax = axes[1, 1]
    ax.plot(
        t,
        result.true_acceleration,
        "-", color="black", linewidth=2,
        label="真实加速度",
    )
    ax.plot(
        t,
        kalman_estimate["acceleration"],
        "-", color="#2ca02c", linewidth=2,
        label="卡尔曼估计加速度",
    )
    ax.axvline(warmup_time, color="gray", linestyle="--", alpha=0.8)
    ax.set_ylim(
        result.true_acceleration.min() - 1.0,
        result.true_acceleration.max() + 1.0,
    )
    ax.set_title("加速度估计")
    ax.set_xlabel("时间 t（秒）")
    ax.set_ylabel("加速度 a（米/秒²）")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_measurement_figure(
    times: np.ndarray,
    measured: np.ndarray,
    kalman_estimate: dict,
    moving_average: np.ndarray,
    output_path: Path,
    truth_position: np.ndarray | None = None,
    truth_velocity: np.ndarray | None = None,
) -> Path:
    """处理外部 CSV 数据时使用的出图函数。

    外部数据通常没有“真实值”，所以真值曲线是可选的。
    """

    prepare_chinese_fonts()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 8), constrained_layout=True
    )

    ax = axes[0, 0]
    ax.plot(
        times, measured,
        ".", color="#d62728", markersize=2, alpha=0.55,
        label="原始传感器数据",
    )
    if truth_position is not None:
        ax.plot(
            times, truth_position,
            "-", color="black", linewidth=2,
            label="真实轨迹（若有）",
        )
    ax.plot(
        times, kalman_estimate["position"],
        "-", color="#2ca02c", linewidth=2,
        label="卡尔曼滤波",
    )
    ax.plot(
        times, moving_average,
        "-", color="#1f77b4", linewidth=1.5, alpha=0.9,
        label="滑动平均（对照）",
    )
    ax.set_title("位置估计")
    ax.set_xlabel("时间 t（秒）")
    ax.set_ylabel("位置 x（米）")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    # 右侧上半格展示滤波前后残差/噪声特征，直观看到去噪效果
    residual_raw = measured - (
        truth_position if truth_position is not None else np.convolve(
            measured, np.ones(21) / 21, mode="same"
        )
    )
    ax.hist(
        residual_raw, bins=30, alpha=0.5,
        color="#d62728", label="原始数据相对基准的偏差",
    )
    ax.axvline(0, color="black", linewidth=1, alpha=0.6)
    ax.set_title("数据偏差分布")
    ax.set_xlabel("偏差（米）")
    ax.set_ylabel("样本数")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    if truth_velocity is not None:
        ax.plot(
            times, truth_velocity,
            "-", color="black", linewidth=2,
            label="真实速度",
        )
    ax.plot(
        times, kalman_estimate["velocity"],
        "-", color="#2ca02c", linewidth=2,
        label="卡尔曼估计速度",
    )
    ax.set_title("速度估计")
    ax.set_xlabel("时间 t（秒）")
    ax.set_ylabel("速度 v（米/秒）")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(
        times, kalman_estimate["acceleration"],
        "-", color="#2ca02c", linewidth=2,
        label="卡尔曼估计加速度",
    )
    ax.set_title("加速度估计")
    ax.set_xlabel("时间 t（秒）")
    ax.set_ylabel("加速度 a（米/秒²）")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
