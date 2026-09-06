"""把外部 CSV 传感器数据送入同一套卡尔曼滤波流程。

CSV 格式约定（方便直接用 Excel 导出）：
  第一列：时间（秒）
  第二列：传感器测得的位置（米）
  可选第三列：真实值（米），用于评分，没有也没关系
第一行可以是表头，程序会自动跳过。

用法：
  python process_csv.py --input 你的文件.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from baseline_filters import causal_moving_average
from estimator import fit_motion_parameters, rmse
from kalman_filter import ConstantAccelerationKalman
from plotting import save_measurement_figure


def load_sensor_csv(path: Path) -> dict:
    """读取 CSV 并返回 times / measured / truth(可选)。"""

    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.reader(fh):
            cells = [cell.strip() for cell in row]
            if any(cells):
                rows.append(cells)

    # 自动跳过表头：只要第一列不是数字就视为表头
    while rows:
        try:
            float(rows[0][0])
            break
        except ValueError:
            rows.pop(0)

    if len(rows) < 3:
        raise ValueError("有效数据不足，至少需要 3 行数据。")

    array = np.array(rows, dtype=float)
    if array.shape[1] < 2:
        raise ValueError("CSV 至少需要两列：时间、测量位置。")

    times = array[:, 0]
    measured = array[:, 1]
    truth = array[:, 2] if array.shape[1] >= 3 else None

    # 如果时间不是递增的，先按时间排序
    order = np.argsort(times)
    times = times[order]
    measured = measured[order]
    if truth is not None:
        truth = truth[order]

    if np.any(np.diff(times) <= 0):
        raise ValueError("时间列存在重复或非递增数据，请检查 CSV。")

    return {
        "times": times,
        "measured": measured,
        "truth": truth,
    }


def estimate_noise_std(times: np.ndarray, measured: np.ndarray) -> float:
    """用二阶差分自动估计噪声标准差。

    一阶差分会把“真实速度变化”也混进来，而二阶差分能基本消掉
    匀速/匀加速运动本身，只留下噪声的离散二阶差分。
    若噪声为高斯白噪声，二阶差分的标准差等于 sqrt(6)·σ。
    """

    second_diff = np.diff(measured, n=2)
    center = np.median(second_diff)
    mad = np.median(np.abs(second_diff - center))
    estimated = 1.4826 * mad / np.sqrt(6.0)
    return float(max(estimated, 1e-9))


def main():
    parser = argparse.ArgumentParser(
        description="处理你自己的 CSV 传感器数据"
    )
    parser.add_argument("--input", "-i", type=Path, default=None)
    parser.add_argument(
        "--noise",
        type=float,
        default=None,
        help="测量噪声标准差（米）；不填则自动估计",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("output") / "csv",
    )
    parser.add_argument(
        "--generate-sample",
        type=Path,
        default=None,
        help="生成一份示例传感器 CSV（用于快速体验，无需真实数据）",
    )
    args = parser.parse_args()

    if args.generate_sample is not None:
        from sensor_simulator import (
            SimulationConfig,
            simulate_position_sensor,
        )

        sample_path = Path(args.generate_sample)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        data = simulate_position_sensor(SimulationConfig(seed=42))
        np.savetxt(
            sample_path,
            np.column_stack(
                [
                    data.times,
                    data.measured_position,
                    data.true_position,
                ]
            ),
            delimiter=",",
            header="time_s,measured_position_m,true_position_m",
            comments="",
            fmt="%.6f",
        )
        print(f"示例数据已生成：{sample_path.resolve()}")
        return

    if args.input is None:
        parser.error(
            "请提供 --input 数据文件，或用 --generate-sample 生成示例数据。"
        )

    data = load_sensor_csv(args.input)
    times = data["times"]
    measured = data["measured"]
    truth = data["truth"]

    noise_std = args.noise
    if noise_std is None:
        noise_std = estimate_noise_std(times, measured)

    dt = float(np.median(np.diff(times)))

    kf = ConstantAccelerationKalman(
        dt=dt,
        measurement_noise_std=noise_std,
        process_noise=0.0001,
    )
    kalman = kf.filter_all(measured)
    moving_average = causal_moving_average(measured, window_size=11)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    figure_path = save_measurement_figure(
        times=times,
        measured=measured,
        kalman_estimate=kalman,
        moving_average=moving_average,
        output_path=outdir / "comparison.png",
        truth_position=truth,
        truth_velocity=None,
    )

    np.savetxt(
        outdir / "processed_data.csv",
        np.column_stack(
            [
                times,
                measured,
                kalman["position"],
                kalman["velocity"],
                kalman["acceleration"],
                moving_average,
            ]
        ),
        delimiter=",",
        header=(
            "time_s,measured_position_m,kalman_position_m,"
            "kalman_velocity_m_per_s,kalman_acceleration_m_per_s2,"
            "moving_average_m"
        ),
        comments="",
        fmt="%.6f",
    )

    fitted = fit_motion_parameters(times, kalman["position"])

    print("\n" + "=" * 64)
    print("CSV 数据处理完成")
    print("=" * 64)
    print(f"数据点数：{len(times)}，采样间隔约 {dt:.4f} 秒")
    print(f"自动估计的测量噪声：{noise_std:.5f} 米")
    print()
    print("滤波后的运动参数拟合结果：")
    print(f"  x0 ≈ {fitted['x0']:.3f} m")
    print(f"  v0 ≈ {fitted['v0']:.3f} m/s")
    print(f"  a  ≈ {fitted['acceleration']:.3f} m/s²")
    print()

    if truth is not None:
        truth_fitted = fit_motion_parameters(times, truth)
        print("真实值对照（CSV 第三列提供真值）：")
        print(f"  x0 = {truth_fitted['x0']:.3f} m")
        print(f"  v0 = {truth_fitted['v0']:.3f} m/s")
        print(f"  a  = {truth_fitted['acceleration']:.3f} m/s²")
        kf_rmse = rmse(kalman["position"], truth)
        raw_rmse = rmse(measured, truth)
        print(f"\n位置 RMSE：原始 {raw_rmse:.5f} m，"
              f"滤波后 {kf_rmse:.5f} m")

    print(f"\n对比图已保存：{figure_path.resolve()}")
    print(f"处理后数据：{(outdir / 'processed_data.csv').resolve()}")


if __name__ == "__main__":
    main()
