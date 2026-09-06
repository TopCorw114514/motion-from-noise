"""把外部 CSV 传感器数据送入同一套卡尔曼滤波流程（详细注释版）。

为什么需要这个文件？
因为招新评审很可能会问：“你的代码是不是只能跑你自己造的数据？”
这个入口就是为了证明：只要数据格式符合约定，真实传感器导出的 CSV
也能直接复用同一套卡尔曼滤波。

CSV 格式约定（方便直接用 Excel 导出）：
  第一列：时间（秒）
  第二列：传感器测得的位置（米）
  可选第三列：真实值（米），用于评分，没有也没关系
第一行可以是表头，程序会自动跳过。

用法：
  python process_csv.py --input 你的文件.csv
"""

# argparse：解析命令行参数；csv：读取 CSV 表格。
import argparse
import csv

# Path：跨平台处理文件路径。
from pathlib import Path

# numpy 提供数组运算，是整个项目的数值基础。
import numpy as np

# 复用项目内已有模块：
# - causal_moving_average：滑动平均对照组
# - fit_motion_parameters / rmse：参数辨识和误差
# - ConstantAccelerationKalman：核心算法
# - save_measurement_figure：给外部数据画图
from baseline_filters import causal_moving_average
from estimator import fit_motion_parameters, rmse
from kalman_filter import ConstantAccelerationKalman
from plotting import save_measurement_figure


def load_sensor_csv(path: Path) -> dict:
    """读取 CSV 并返回 times / measured / truth(可选)。

    返回字典包含：
    - times：时间数组（秒）
    - measured：传感器读数数组（米）
    - truth：真实值数组（米）；CSV 没有第三列时为 None
    """

    # 用 csv.reader 逐行读取，比手工 split(",") 更稳健：
    # 它能正确处理引号、空格等边界情况。
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.reader(fh):
            # 去掉每个单元格首尾空格，过滤完全空白的行。
            cells = [cell.strip() for cell in row]
            if any(cells):
                rows.append(cells)

    # 自动跳过表头：只要第一行第一列无法转换成数字，
    # 就认为它是表头，删掉再看下一行。
    while rows:
        try:
            float(rows[0][0])
            break
        except ValueError:
            rows.pop(0)

    # 至少需要 3 行有效数据，否则无法体现滤波趋势。
    if len(rows) < 3:
        raise ValueError("有效数据不足，至少需要 3 行数据。")

    # 把“字符串组成的行”整体转成浮点数二维数组。
    array = np.array(rows, dtype=float)

    # 至少要“时间 + 测量值”两列。
    if array.shape[1] < 2:
        raise ValueError("CSV 至少需要两列：时间、测量位置。")

    # 按列取出数据：第 0 列时间，第 1 列测量，第 2 列真值（可选）。
    times = array[:, 0]
    measured = array[:, 1]
    truth = array[:, 2] if array.shape[1] >= 3 else None

    # 有些 Excel 导出的数据时间不是严格递增的。
    # argsort 返回“从小到大排序后的下标”，用它统一重排三列，
    # 保证卡尔曼滤波按时间顺序处理。
    order = np.argsort(times)
    times = times[order]
    measured = measured[order]
    if truth is not None:
        truth = truth[order]

    # 排序后若仍有重复/相同时间，说明数据本身有问题，直接报错。
    if np.any(np.diff(times) <= 0):
        raise ValueError("时间列存在重复或非递增数据，请检查 CSV。")

    return {
        "times": times,
        "measured": measured,
        "truth": truth,
    }


def estimate_noise_std(times: np.ndarray, measured: np.ndarray) -> float:
    """用二阶差分自动估计噪声标准差。

    为什么要自动估计？
    卡尔曼滤波需要知道测量噪声方差 R。处理用户自己的 CSV 时，
    我们不知道对方传感器的噪声大小，所以要先从数据本身估一个。

    为什么用二阶差分？
    一阶差分（相邻读数相减）里还混着“真实运动速度变化”；
    而匀加速/匀速运动的二阶差分基本只剩噪声项。
    对高斯白噪声做二阶差分，结果的标准差是真实噪声的 sqrt(6) 倍，
    所以最后要除以 sqrt(6)。

    这里用 MAD（绝对中位差）而不是普通标准差，
    因为 MAD 对个别异常点更稳健，不会因为一两个跳变把噪声估得过大。
    """

    # np.diff(x, n=2) 表示对数组连续求两次差分。
    second_diff = np.diff(measured, n=2)

    # MAD 稳健统计：先找二阶差分的中间值，再找“离中间值”的中位距离。
    center = np.median(second_diff)
    mad = np.median(np.abs(second_diff - center))

    # 1.4826 是把 MAD 换算成标准差的常数（高斯分布下），
    # sqrt(6) 是二阶差分引入的放大倍数。
    estimated = 1.4826 * mad / np.sqrt(6.0)

    # 防止极小值或 0 导致后续矩阵运算出问题。
    return float(max(estimated, 1e-9))


def main():
    # ---------- 1. 解析命令行参数 ----------
    parser = argparse.ArgumentParser(
        description="处理你自己的 CSV 传感器数据"
    )
    parser.add_argument("--input", "-i", type=Path, default=None,
                        help="CSV 文件路径")
    parser.add_argument(
        "--noise", type=float, default=None,
        help="测量噪声标准差（米）；不填则自动估计",
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("output") / "csv",
        help="结果输出目录",
    )
    parser.add_argument(
        "--generate-sample", type=Path, default=None,
        help="生成一份示例传感器 CSV（用于快速体验，无需真实数据）",
    )
    args = parser.parse_args()

    # ---------- 2. 生成示例数据（可选） ----------
    # 如果用户只想快速体验、手边没有真实 CSV，可以用这个选项。
    if args.generate_sample is not None:
        from sensor_simulator import (
            SimulationConfig,
            simulate_position_sensor,
        )

        sample_path = Path(args.generate_sample)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        # 生成一份“真实+测量+真值”三列的标准示例。
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

    # ---------- 3. 读取数据 ----------
    if args.input is None:
        parser.error(
            "请提供 --input 数据文件，或用 --generate-sample 生成示例数据。"
        )

    data = load_sensor_csv(args.input)
    times = data["times"]
    measured = data["measured"]
    truth = data["truth"]

    # 决定测量噪声：用户显式给了就听用户的，否则自动估计。
    noise_std = args.noise
    if noise_std is None:
        noise_std = estimate_noise_std(times, measured)

    # 采样间隔可能不是整数，用所有相邻时间差的中位数代表 dt，
    # 对偶发的不均匀采样更稳健。
    dt = float(np.median(np.diff(times)))

    # ---------- 4. 卡尔曼滤波 + 滑动平均 ----------
    kf = ConstantAccelerationKalman(
        dt=dt,
        measurement_noise_std=noise_std,
        process_noise=0.0001,
    )
    kalman = kf.filter_all(measured)

    # 滑动平均窗口固定 11 个点；如果采样率差异很大，可自行调整。
    moving_average = causal_moving_average(measured, window_size=11)

    # ---------- 5. 保存结果 ----------
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 5a. 出图：外部数据可能没有真值，所以真值设为可选。
    figure_path = save_measurement_figure(
        times=times,
        measured=measured,
        kalman_estimate=kalman,
        moving_average=moving_average,
        output_path=outdir / "comparison.png",
        truth_position=truth,      # 没有真值时自动忽略
        truth_velocity=None,
    )

    # 5b. 把滤波结果存成新的 CSV，方便用户用 Excel 进一步分析。
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

    # 5c. 用二次拟合反推运动参数（假设数据接近匀加速运动）。
    fitted = fit_motion_parameters(times, kalman["position"])

    # ---------- 6. 终端输出 ----------
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

    # 如果 CSV 有第三列真值，就额外输出误差对照。
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