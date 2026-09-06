"""一键运行：模拟 -> 卡尔曼滤波 -> 对比 -> 参数辨识 -> 出图出报告。"""

import argparse
from pathlib import Path

import numpy as np

from baseline_filters import causal_moving_average
from estimator import fit_motion_parameters, rmse
from kalman_filter import ConstantAccelerationKalman
from plotting import save_comparison_figure
from sensor_simulator import (
    SimulationConfig,
    simulate_position_sensor,
)


def main():
    parser = argparse.ArgumentParser(description="传感器数据处理演示")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("output") / "demo",
    )
    args = parser.parse_args()

    config = SimulationConfig(
        duration=args.duration,
        noise_std=args.noise,
        seed=args.seed,
    )
    data = simulate_position_sensor(config)

    # 1. 卡尔曼滤波
    kf = ConstantAccelerationKalman(
        dt=config.dt,
        measurement_noise_std=config.noise_std,
        process_noise=0.0001,
    )
    kalman = kf.filter_all(data.measured_position)

    # 2. 移动平均对照
    moving_average = causal_moving_average(
        data.measured_position, window_size=11
    )

    # 3. 定量指标（前 2 秒是滤波预热期，评估时剔除）
    warmup_time = 2.0
    warm_mask = data.times >= warmup_time

    raw_rmse = rmse(
        data.measured_position[warm_mask],
        data.true_position[warm_mask],
    )
    kalman_rmse = rmse(
        kalman["position"][warm_mask],
        data.true_position[warm_mask],
    )
    moving_avg_rmse = rmse(
        moving_average[warm_mask],
        data.true_position[warm_mask],
    )
    noise_reduction = (1.0 - kalman_rmse / raw_rmse) * 100.0

    # 4. 运动参数辨识
    fitted_raw = fit_motion_parameters(
        data.times, data.measured_position
    )
    fitted_kalman = fit_motion_parameters(
        data.times, kalman["position"]
    )

    # 5. 保存图片、CSV 与文字报告
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    figure_path = save_comparison_figure(
        data,
        kalman,
        moving_average,
        outdir / "comparison.png",
        warmup_time=warmup_time,
    )

    np.savetxt(
        outdir / "processed_data.csv",
        np.column_stack(
            [
                data.times,
                data.true_position,
                data.measured_position,
                kalman["position"],
                kalman["velocity"],
                kalman["acceleration"],
                moving_average,
            ]
        ),
        delimiter=",",
        header=(
            "time_s,true_position_m,measured_position_m,"
            "kalman_position_m,kalman_velocity_m_per_s,"
            "kalman_acceleration_m_per_s2,moving_average_m"
        ),
        comments="",
        fmt="%.6f",
    )

    report_path = outdir / "experiment_report.txt"
    lines = [
        "=" * 64,
        "《从噪声中还原运动》实验报告",
        "=" * 64,
        "",
        f"模拟配置：采样间隔 {config.dt} 秒，总时长 {config.duration} 秒",
        f"传感器噪声标准差：{config.noise_std} 米",
        f"真实运动参数：x0={config.x0} m，"
        f"v0={config.v0} m/s，a={config.acceleration} m/s²",
        "",
        "一、去噪效果（评估区间：2 秒之后）",
        f"  原始传感器数据 RMSE：{raw_rmse:.4f} m",
        f"  移动平均 RMSE：       {moving_avg_rmse:.4f} m",
        f"  卡尔曼滤波 RMSE：     {kalman_rmse:.4f} m",
        f"  卡尔曼相对原始数据误差降低：{noise_reduction:.1f}%",
        "",
        "二、运动参数辨识",
        f"  {'参数':<12}{'真实值':>14}{'原始拟合':>14}{'滤波后拟合':>14}",
        f"  {'x0 (m)':<12}{config.x0:>14.3f}"
        f"{fitted_raw['x0']:>14.3f}{fitted_kalman['x0']:>14.3f}",
        f"  {'v0 (m/s)':<12}{config.v0:>14.3f}"
        f"{fitted_raw['v0']:>14.3f}{fitted_kalman['v0']:>14.3f}",
        f"  {'a (m/s²)':<12}{config.acceleration:>14.3f}"
        f"{fitted_raw['acceleration']:>14.3f}"
        f"{fitted_kalman['acceleration']:>14.3f}",
        "",
        "说明：滤波预热期约 2 秒，之后估计值才会收敛。",
        f"图片已保存到：{figure_path}",
        f"详细数据已保存到：{outdir / 'processed_data.csv'}",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # 6. 终端输出
    print("\n" + "=" * 64)
    print("《从噪声中还原运动》运行完成")
    print("=" * 64)
    print(f"原始数据 RMSE：    {raw_rmse:.4f} m")
    print(f"移动平均 RMSE：    {moving_avg_rmse:.4f} m")
    print(f"卡尔曼滤波 RMSE：  {kalman_rmse:.4f} m")
    print(f"误差降低：         {noise_reduction:.1f}%")
    print()
    print("真实运动参数 vs 原始拟合 vs 卡尔曼滤波后拟合")
    print(f"  x0：{config.x0:.3f}  vs {fitted_raw['x0']:.3f}"
          f"  vs {fitted_kalman['x0']:.3f} m")
    print(f"  v0：{config.v0:.3f}  vs {fitted_raw['v0']:.3f}"
          f"  vs {fitted_kalman['v0']:.3f} m/s")
    print(f"  a ：{config.acceleration:.3f}  vs "
          f"{fitted_raw['acceleration']:.3f}  vs "
          f"{fitted_kalman['acceleration']:.3f} m/s²")
    print()
    print(f"对比图：   {figure_path.resolve()}")
    print(f"数据表：   {(outdir / 'processed_data.csv').resolve()}")
    print(f"实验报告： {report_path.resolve()}")


if __name__ == "__main__":
    main()
