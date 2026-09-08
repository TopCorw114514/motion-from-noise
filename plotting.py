"""可视化模块（详细注释版）。

作用：把实验数据变成一张 2×2 的对比图。
评审看项目时，第一眼看到的往往是图而不是代码，
所以图的安排原则是“一眼能看懂结论”：
- 左上：整体看原始噪声 vs 卡尔曼
- 右上：放大看细节，检查有没有明显滞后
- 左下：速度估计（传感器没直接测速度）
- 右下：加速度估计（滤波器“推导”出的高阶状态）
"""

from pathlib import Path

from baseline_filters import causal_moving_average

# 先把 matplotlib 切换成 Agg 后端再导入 pyplot。
# Agg 是“只负责把图画成文件”的后端，不依赖桌面窗口，
# 因此在任何环境（包括没有图形界面的服务器）都能保存 PNG。
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np


def prepare_chinese_fonts():
    """优先使用电脑上常见的中文字体，避免图里中文变成方块。

    Matplotlib 默认字体不含中文。这里先扫描系统已安装字体，
    找到第一个支持中文的字体就设成默认无衬线字体。
    不同系统字体名不同，所以按优先级逐个尝试。
    """

    from matplotlib import font_manager

    # 已安装字体名集合（只取名称，不涉及字体文件细节）。
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in [
        "Microsoft YaHei",       # Windows 常见
        "SimHei",                # Windows 黑体
        "Noto Sans CJK SC",      # Linux/macOS 常见
        "PingFang SC",           # macOS 常见
        "WenQuanYi Micro Hei",   # Linux 备选
    ]:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            break

    # 让负号正常显示（避免把 -1 显示成方块）。
    plt.rcParams["axes.unicode_minus"] = False


def save_comparison_figure(
    result,
    kalman_estimate: dict,
    moving_average: np.ndarray,
    output_path: Path,
    warmup_time: float = 2.0,
) -> Path:
    """生成并保存 2×2 对比图（用于仿真演示，有真实轨迹）。"""

    prepare_chinese_fonts()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t = result.times                # 时间轴
    warmup_mask = t >= warmup_time  # 只用于区分评估区，不参与裁剪

    # 创建一个 2 行 2 列的画布；constrained_layout 自动防止子图重叠。
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 8), constrained_layout=True
    )

    # ===== 左上：整体位置对比 =====
    # 四组数据叠在一起：
    # 红色小点 = 原始传感器读数（噪声很大、很散）
    # 黑色粗线 = 真实轨迹（标准答案）
    # 绿色线   = 卡尔曼滤波输出（应贴近黑色）
    # 蓝色线   = 滑动平均（对照组）
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
    # 用竖直虚线标出“预热期”结束位置，提醒看图人前 2 秒不算。
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

    # ===== 右上：放大细节 =====
    # 左边整图数据点太多，看不出滤波是否“跟手”，
    # 所以只截取 warmup_time（默认 2 秒）之后的区间放大，
    # 看绿线是否贴住黑线、蓝线是否明显滞后。
    ax = axes[0, 1]
    slice_ = t >= warmup_time
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

    # ===== 左下：速度估计 =====
    # 这是卡尔曼的“额外能力”：传感器只测位置，
    # 但状态向量里有速度，所以滤波器能把速度也估出来。
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

    # ===== 右下：加速度估计 =====
    # 加速度是位置的二阶信息，收敛会比位置慢一些，
    # 所以这张图最能说明“滤波器需要预热期”。
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
    # 给纵轴留一点余量，避免曲线紧贴边框。
    ax.set_ylim(
        result.true_acceleration.min() - 1.0,
        result.true_acceleration.max() + 1.0,
    )
    ax.set_title("加速度估计")
    ax.set_xlabel("时间 t（秒）")
    ax.set_ylabel("加速度 a（米/秒²）")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # 统一保存为 PNG（150 dpi 足够清晰），随后关闭画布释放内存。
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
    有真值就画出真值方便对照；没有真值就只展示滤波结果。
    """

    prepare_chinese_fonts()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 8), constrained_layout=True
    )

    # 左上：位置曲线。
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

    # 右上：数据偏差分布直方图。
    # 如果知道真值，偏差 = 测量 - 真值；
    # 如果不知道真值，就用一段滑动平均当近似基准。
    # 画直方图能直观看到“误差是否集中在 0 附近”。
    ax = axes[0, 1]
    # 若没有真值，用“因果滑动平均”当近似基准：
    # 它和项目其他地方一样只使用当前及过去的数据，
    # 不会因为偷看未来数据而显得过于平滑。
    residual_raw = measured - (
        truth_position
        if truth_position is not None
        else causal_moving_average(measured, window_size=21)
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

    # 左下：速度估计。
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

    # 右下：加速度估计。
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