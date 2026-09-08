"""一键运行入口（详细注释版）。

整条流水线：
    模拟生成真实+噪声数据
        -> 卡尔曼滤波
        -> 滑动平均对照
        -> 计算误差指标
        -> 反推运动参数
        -> 保存对比图 / CSV / 实验报告

运行方法：
    python run_demo.py
"""

# argparse 用于解析命令行参数，例如 --seed 42 可以固定随机种子。
import argparse

# Path 用来跨平台地拼接文件路径（Windows/Linux 都适用）。
from pathlib import Path

# NumPy 是我们唯一的数值计算底层库。
import numpy as np

# 以下都是本项目自己写的模块：
# - causal_moving_average：滑动平均（对照组）
# - fit_motion_parameters / rmse：参数辨识和误差指标
# - ConstantAccelerationKalman：核心算法
# - save_comparison_figure：画图
# - SimulationConfig / simulate_position_sensor：模拟传感器数据
from baseline_filters import causal_moving_average
from estimator import fit_motion_parameters, rmse
from kalman_filter import ConstantAccelerationKalman
from plotting import save_comparison_figure
from sensor_simulator import (
    SimulationConfig,
    simulate_position_sensor,
)


def main():
    # ---------- 0. 读取命令行参数 ----------
    # argparse 会生成一个命令行小工具，
    # 用户可以不传任何参数直接运行，也可以用参数做对照实验。
    parser = argparse.ArgumentParser(description="传感器数据处理演示")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子：相同种子产生完全相同的结果",
    )
    parser.add_argument(
        "--duration", type=float, default=5.0,
        help="模拟总时长（秒）",
    )
    parser.add_argument(
        "--noise", type=float, default=0.08,
        help="传感器噪声标准差（米）",
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("output") / "demo",
        help="输出目录",
    )
    args = parser.parse_args()

    # ---------- 1. 生成实验数据 ----------
    # 把命令行参数打包成 SimulationConfig，
    # 然后调用 simulate_position_sensor 生成：
    #   - 真实位置/速度/加速度（Ground Truth）
    #   - 带噪声的传感器读数
    config = SimulationConfig(
        duration=args.duration,
        noise_std=args.noise,
        seed=args.seed,
    )
    data = simulate_position_sensor(config)

    # ---------- 2. 卡尔曼滤波 ----------
    # 创建滤波器时告诉它：采样间隔是多少、传感器噪声有多大、
    # 模型本身留多少不确定性（process_noise）。
    # 然后一次处理整串测量值，得到逐点的位置/速度/加速度估计。
    kf = ConstantAccelerationKalman(
        dt=config.dt,
        measurement_noise_std=config.noise_std,
        process_noise=0.0001,
    )
    kalman = kf.filter_all(data.measured_position)

    # ---------- 3. 滑动平均（对照组） ----------
    # 同样只用历史数据，窗口取 11 个采样点。
    # 后面会把它的误差和卡尔曼放在一起比较。
    moving_average = causal_moving_average(
        data.measured_position, window_size=11
    )

    # ---------- 4. 计算定量误差 ----------
    # 卡尔曼启动后的前 ~2 秒是“预热期”：初始速度/加速度未知，
    # 滤波器需要几个采样点才能收敛。如果把这 2 秒也算进误差，
    # 会对算法不公平。因此用 warm_mask 只评估 t >= 2 秒的数据。
    warmup_time = 2.0
    warm_mask = data.times >= warmup_time

    # 原始噪声数据的误差：作为“不做任何处理”的基准。
    raw_rmse = rmse(
        data.measured_position[warm_mask],
        data.true_position[warm_mask],
    )

    # 卡尔曼滤波后的误差：应当明显小于原始误差。
    kalman_rmse = rmse(
        kalman["position"][warm_mask],
        data.true_position[warm_mask],
    )

    # 滑动平均后的误差：用于和卡尔曼对比。
    moving_avg_rmse = rmse(
        moving_average[warm_mask],
        data.true_position[warm_mask],
    )

    # 误差降低百分比 = (原始误差 - 卡尔曼误差) / 原始误差。
    # 例如 0.08 -> 0.019，大约是降低了 76% 左右。
    noise_reduction = (1.0 - kalman_rmse / raw_rmse) * 100.0

    # ---------- 5. 运动参数辨识 ----------
    # 用二次多项式分别拟合“原始噪声数据”和“卡尔曼滤波后数据”，
    # 再换算成物理参数 x0/v0/a。
    # 注意：这里同时展示原始拟合与滤波后拟合，是把参数辨识当作功能演示；
    # 滤波的卖点是实时逐点输出位置/速度/加速度，而不是让离线拟合更准。
    fitted_raw = fit_motion_parameters(
        data.times, data.measured_position
    )
    fitted_kalman = fit_motion_parameters(
        data.times, kalman["position"]
    )

    # ---------- 6. 保存结果 ----------
    # 创建输出目录（不存在时自动创建，不会报错）。
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 6a. 画一张 2×2 对比图并保存为 PNG。
    figure_path = save_comparison_figure(
        data,             # 包含真实轨迹和原始测量
        kalman,           # 卡尔曼滤波估计结果
        moving_average,   # 滑动平均结果
        outdir / "comparison.png",
        warmup_time=warmup_time,  # 图中用虚线标出预热区
    )

    # 6b. 把所有关键曲线保存成 CSV。
    # np.column_stack 把多个一维数组按列拼成一个大表格，
    # 这样用户可以用 Excel 打开，逐行查看每一个时刻的数据。
    np.savetxt(
        outdir / "processed_data.csv",
        np.column_stack(
            [
                data.times,               # 时间
                data.true_position,       # 真实位置
                data.measured_position,   # 原始测量
                kalman["position"],       # 滤波后的位置
                kalman["velocity"],       # 估计的速度
                kalman["acceleration"],   # 估计的加速度
                moving_average,           # 滑动平均位置
            ]
        ),
        delimiter=",",   # 用逗号分隔，Excel 可直接打开
        header=(
            "time_s,true_position_m,measured_position_m,"
            "kalman_position_m,kalman_velocity_m_per_s,"
            "kalman_acceleration_m_per_s2,moving_average_m"
        ),
        comments="",     # 不让 NumPy 在表头前加 '#'
        fmt="%.6f",      # 保留 6 位小数
    )

    # 6c. 生成一份文字版实验报告（同样方便截图放进演示文稿）。
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
        "参数辨识说明：滤波的价值是实时逐点输出位置/速度/加速度；",
        "离线拟合仅为功能演示，原始拟合与滤波后拟合并不用来证明谁更准。",
        "",
        "说明：滤波预热期约 2 秒，之后估计值才会收敛。",
        f"图片已保存到：{figure_path}",
        f"详细数据已保存到：{outdir / 'processed_data.csv'}",
    ]
    # 用 UTF-8 写文件，避免中文在 Windows 记事本里乱码。
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # ---------- 7. 终端输出 ----------
    # 让用户不打开文件也能立刻看到最重要的几个数字。
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


# 只有直接运行本文件时才执行 main()；
# 如果被其他文件 import，则不会自动运行，方便复用。
if __name__ == "__main__":
    main()