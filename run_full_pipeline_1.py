import argparse
import json
import os
import datetime
import subprocess
import sys
import pandas as pd

# 导入绘图函数
from plot_heatmap import plot_heatmaps_from_csv

def main():
    parser = argparse.ArgumentParser(description="自动运行数据生成和热力图绘制全流程")
    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,
        help='实验参数配置文件路径（JSON 格式）'
    )
    parser.add_argument(
        '--simulator',
        type=str,
        default='single_channel',
        choices=['single_channel', 'dual_channel', 'duel_channel'],
        help='通信通道类型: single_channel 或 dual/duel_channel'
    )
    parser.add_argument(
        '--pipeline',
        type=str,
        default='1f1b',
        choices=['1f1b', 'gpipe'],
        help='选择流水线调度策略: 1f1b 或 gpipe'
    )
    parser.add_argument(
        '--priority_step',
        type=float,
        default=0.1,
        help='优先级扫描步长（默认: 0.1）'
    )
    parser.add_argument(
        '--enable-recv-congestion',
        action='store_true',
        help='开启 1f1b 单通道的接收端拥塞控制 (适用于 1.3 阶段)'
    )
    args = parser.parse_args()

    # 验证配置文件存在
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        sys.exit(1)

    # 读取配置 JSON
    try:
        with open(args.config, 'r') as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ 配置文件解析错误: {e}")
        sys.exit(1)

    # 准备输出文件名
    config_basename = os.path.splitext(os.path.basename(args.config))[0]
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_prefix = f"{config_basename}_{timestamp}"

    os.makedirs("outputs/data", exist_ok=True)
    os.makedirs("outputs/heatmaps", exist_ok=True)

    output_csv = f"outputs/data/{output_prefix}.csv"
    temp_param_file = f"temp_param_{timestamp}.json"

    # 写入临时 JSON 文件供 data_generator 使用
    with open(temp_param_file, 'w') as f:
        json.dump(config_data, f, indent=2)

    # 调用 data_generator.py
    print(f"\n[1/2] 🔧 正在生成数据: {output_csv}")
    try:
        # 统一 dual/duel 名称传递给 data_generator，并追加 pipeline 和接收端拥塞控制参数
        sim_name = args.simulator
        if sim_name == 'dual_channel':
            sim_name = 'duel_channel'
        cmd = [sys.executable, 'data_generator.py',
               '--config', temp_param_file,
               '--simulator', sim_name,
               '--pipeline', args.pipeline,
               '--step', str(args.priority_step),
               '--output', output_csv]
        if args.enable_recv_congestion:
            cmd.append('--enable-recv-congestion')
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("❌ data_generator.py 执行失败！")
        if os.path.exists(temp_param_file):
            os.remove(temp_param_file)
        sys.exit(1)

    # 生成热力图
    print(f"\n[2/2] 🎨 正在生成热力图...")
    try:
        plot_heatmaps_from_csv(output_csv, output_dir="outputs/heatmaps")
    except Exception as e:
        print(f"❌ 绘图出错: {e}")
        sys.exit(1)

    # 根据生成的数据提取最优组合并保存
    try:
        df = pd.read_csv(output_csv)
        df['total_time'] = pd.to_numeric(df['total_time'], errors='coerce')
        df = df.dropna(subset=['total_time'])
        # 定义环境参数列以分组
        config_cols = [
            'n', 'sync_freq', 'fwd_len', 'bwd_len', 'prop_len',
            'sync_base_len', 'fwd_impact', 'bwd_impact'
        ]
        if all(col in df.columns for col in config_cols):
            idx = df.groupby(config_cols)['total_time'].idxmin()
            best_df = df.loc[idx].reset_index(drop=True)
        else:
            best_df = df.nsmallest(1, 'total_time')
        best_output = f"outputs/data/{config_basename}_best.csv"
        best_df.to_csv(best_output, index=False)
        print(f"\n📝 已生成最优结果文件: {best_output}")
    except Exception as e:
        print(f"❌ 生成最优结果文件时出错: {e}")

    # 删除临时文件
    if os.path.exists(temp_param_file):
        os.remove(temp_param_file)

    print(f"\n✅ 全部完成！")
    print(f"📄 数据文件: {output_csv}")
    print(f"🖼️ 热力图目录: outputs/heatmaps/")

if __name__ == '__main__':
    main()