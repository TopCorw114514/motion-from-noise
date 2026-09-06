"""多噪声强度实验（详细注释版）。

作用：改变传感器噪声，观察卡尔曼滤波效果是否稳定。

这个实验回答评审很可能问的问题：
“你的算法只在一种噪声下有效，还是普遍有效？”
如果换了噪声效果就崩，说明算法是“调参调出来的偶然结果”；
如果从弱噪声到强噪声都保持稳定，才能证明算法本身有效。
"""

import argparse

from pathlib import Path

# matplotlib：绘图库。Agg 是无窗口后端，适合在脚本里直接存图，
# 不会因为当前电脑没有图形界面而报错。
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

from baseline_filters import causal_moving_average
from estimator import rmse
from kalman_filter import ConstantAccelerationKalman
from plotting import prepare_chinese_fonts
from sensor_simulator import SimulationConfig, simulate_position_sensor


def run_single_case(
    noise_std: float, seed: int = 42
) -> dict:
    """在指定噪声强度下完整跑一遍“模拟->滤波->评估”，返回关键指标。

    这个函数把“跑一次实验”封装成独立单元，
    主程序只要循环调用它，就能批量测试不同噪声。
    """

    # 只改噪声强度，其余参数（轨迹、时长）保持相同，保证公平对照。
    config = SimulationConfig(noise_std=noise_std, seed=seed)
    data = simulate_position_sensor(config)

    # 卡尔曼滤波：与单次演示使用完全相同的核心代码。
    kf = ConstantAccelerationKalman(
        dt=config.dt,
        measurement_noise_std=config.noise_std,
        process_noise=0.0001,
    )
    kalman = kf.filter_all(data.measured_position)

    # 滑动平均对照组。
    moving_average = causal_moving_average(
        data.measured_position, window_size=11
    )

    # 同样跳过前 2 秒预热期，只评估滤波器稳定后的表现。
    warm_mask = data.times >= 2.0

    # 三种误差：原始数据、滑动平均、卡尔曼滤波。
    raw = rmse(
        data.measured_position[warm_mask],
        data.true_position[warm_mask],
    )
    kalman_rmse = rmse(
        kalman["position"][warm_mask],
        data.true_position[warm_mask],
    )
    moving_rmse = rmse(
        moving_average[warm_mask],
        data.true_position[warm_mask],
    )

    # 返回结构化结果，方便上层绘制表格/图表。
    return {
        "noise_std": noise_std,
        "raw_rmse": raw,
        "kalman_rmse": kalman_rmse,
        "moving_avg_rmse": moving_rmse,
        "improvement_pct": (1.0 - kalman_rmse / raw) * 100.0,
    }


def save_sweep_figure(results, output_path: Path) -> Path:
    """绘制不同噪声下的 RMSE 对比图（左：误差，右：降幅）。"""

    prepare_chinese_fonts()  # 让中文标签正常显示

    # 从实验结果字典里拆出要画的几条序列。
    noise_levels = [r["noise_std"] for r in results]
    raw = [r["raw_rmse"] for r in results]
    moving = [r["moving_avg_rmse"] for r in results]
    kalman = [r["kalman_rmse"] for r in results]

    # 一张图分成左右两个子图。
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5), constrained_layout=True
    )

    # ---- 左图：误差随噪声变化 ----
    ax = axes[0]
    ax.plot(
        noise_levels, raw, "o-", color="#d62728",
        label="原始数据 RMSE",
    )
    ax.plot(
        noise_levels, moving, "s-", color="#1f77b4",
        label="滑动平均 RMSE",
    )
    ax.plot(
        noise_levels, kalman, "^-", color="#2ca02c",
        label="卡尔曼滤波 RMSE",
    )
    # 灰色虚线是“理论下界”：即使完全不滤波，
    # 误差也不可能小于噪声本身的统计水平，用来当参照。
    ax.plot(
        noise_levels, noise_levels, "--", color="gray",
        label="理论下界（噪声本身）",
    )
    ax.set_title("不同噪声强度下的位置误差")
    ax.set_xlabel("传感器噪声标准差（米）")
    ax.set_ylabel("RMSE（米）")
    ax.legend()
    ax.grid(alpha=0.3)

    # ---- 右图：误差降低比例 ----
    ax = axes[1]
    improvement = [r["improvement_pct"] for r in results]
    ax.bar(
        [str(n) for n in noise_levels],  # 横轴直接显示噪声值
        improvement,
        color="#2ca02c",
        alpha=0.85,
    )
    # 在每个柱子上方标出百分比数字，方便直接读取。
    for x, value in zip(range(len(improvement)), improvement):
        ax.text(x, value + 1, f"{value:.1f}%", ha="center")
    ax.set_title("卡尔曼滤波相对原始数据的误差降低比例")
    ax.set_xlabel("传感器噪声标准差（米）")
    ax.set_ylabel("误差降低比例（%）")
    ax.set_ylim(0, max(improvement) * 1.25)
    ax.grid(alpha=0.3, axis="y")

    # 保存并关闭，避免占用内存/弹出无关窗口。
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main():
    # ---------- 命令行参数 ----------
    parser = argparse.ArgumentParser(description="多噪声强度实验")
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",   # 允许一次传多个值：--noise-levels 0.02 0.05 0.08
        default=[0.02, 0.05, 0.08, 0.12, 0.2, 0.35],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--outdir", type=Path, default=Path("output") / "sweep"
    )
    args = parser.parse_args()

    # ---------- 对每一档噪声都跑一遍实验 ----------
    results = [
        run_single_case(noise, seed=args.seed)
        for noise in args.noise_levels
    ]

    # ---------- 保存图表与 CSV ----------
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    figure_path = save_sweep_figure(results, outdir / "noise_sweep.png")

    # 汇总表：既给人看，也方便后续用 Excel/脚本继续分析。
    summary_path = outdir / "summary.csv"
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(
            "noise_std_m,raw_rmse_m,moving_avg_rmse_m,"
            "kalman_rmse_m,improvement_pct\n"
        )
        for r in results:
            fh.write(
                f"{r['noise_std']:.4f},{r['raw_rmse']:.6f},"
                f"{r['moving_avg_rmse']:.6f},{r['kalman_rmse']:.6f},"
                f"{r['improvement_pct']:.2f}\n"
            )

    # ---------- 终端表格输出 ----------
    print("\n" + "=" * 76)
    print("多噪声强度实验完成")
    print("=" * 76)
    print(f"{'噪声(m)':<10}{'原始RMSE':>12}{'滑动平均':>12}"
          f"{'卡尔曼':>12}{'降幅':>10}")
    for r in results:
        print(
            f"{r['noise_std']:<10.3f}{r['raw_rmse']:>12.4f}"
            f"{r['moving_avg_rmse']:>12.4f}{r['kalman_rmse']:>12.4f}"
            f"{r['improvement_pct']:>9.1f}%"
        )

    print(f"\n对比图已保存：{figure_path.resolve()}")
    print(f"汇总表已保存：{summary_path.resolve()}")


if __name__ == "__main__":
    main()