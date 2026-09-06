"""改变噪声强度，观察卡尔曼滤波的效果是否稳定。

这个实验回答评审很可能问的问题：
“你的算法只在一种噪声下有效，还是普遍有效？”
"""

import argparse
from pathlib import Path

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
    """在指定噪声下跑一遍完整流程，返回关键指标。"""

    config = SimulationConfig(noise_std=noise_std, seed=seed)
    data = simulate_position_sensor(config)

    kf = ConstantAccelerationKalman(
        dt=config.dt,
        measurement_noise_std=config.noise_std,
        process_noise=0.0001,
    )
    kalman = kf.filter_all(data.measured_position)
    moving_average = causal_moving_average(
        data.measured_position, window_size=11
    )

    warm_mask = data.times >= 2.0
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

    return {
        "noise_std": noise_std,
        "raw_rmse": raw,
        "kalman_rmse": kalman_rmse,
        "moving_avg_rmse": moving_rmse,
        "improvement_pct": (1.0 - kalman_rmse / raw) * 100.0,
    }


def save_sweep_figure(results, output_path: Path) -> Path:
    """绘制不同噪声下的 RMSE 对比。"""

    prepare_chinese_fonts()
    noise_levels = [r["noise_std"] for r in results]
    raw = [r["raw_rmse"] for r in results]
    moving = [r["moving_avg_rmse"] for r in results]
    kalman = [r["kalman_rmse"] for r in results]

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5), constrained_layout=True
    )

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
    ax.plot(
        noise_levels, noise_levels, "--", color="gray",
        label="理论下界（噪声本身）",
    )
    ax.set_title("不同噪声强度下的位置误差")
    ax.set_xlabel("传感器噪声标准差（米）")
    ax.set_ylabel("RMSE（米）")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    improvement = [r["improvement_pct"] for r in results]
    ax.bar(
        [str(n) for n in noise_levels],
        improvement,
        color="#2ca02c",
        alpha=0.85,
    )
    for x, value in zip(range(len(improvement)), improvement):
        ax.text(x, value + 1, f"{value:.1f}%", ha="center")
    ax.set_title("卡尔曼滤波相对原始数据的误差降低比例")
    ax.set_xlabel("传感器噪声标准差（米）")
    ax.set_ylabel("误差降低比例（%）")
    ax.set_ylim(0, max(improvement) * 1.25)
    ax.grid(alpha=0.3, axis="y")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="多噪声强度实验")
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.08, 0.12, 0.2, 0.35],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--outdir", type=Path, default=Path("output") / "sweep"
    )
    args = parser.parse_args()

    results = [
        run_single_case(noise, seed=args.seed)
        for noise in args.noise_levels
    ]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    figure_path = save_sweep_figure(results, outdir / "noise_sweep.png")

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
