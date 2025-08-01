import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import sys

def format_priority(val):
    """将优先级值格式化为字符串，保留 'a'"""
    if str(val).strip().lower() == 'a':
        return 'a'
    try:
        return f"{float(val):.1f}"
    except:
        return str(val)

def sort_key(val):
    """排序规则：'a' 在最前，然后按数值升序，其他无效值在最后"""
    if val == 'a':
        return (-1, None)
    try:
        return (0, float(val))
    except:
        return (1, val)

def plot_single_heatmap(df_group, config_dict, output_dir, show_plot=True):
    df_group['x_coord'] = df_group['p_grad_prop'].apply(format_priority)
    df_group['y_coord'] = df_group['p_fwd_prop'].apply(format_priority)

    x_labels = sorted(df_group['x_coord'].unique(), key=sort_key)
    y_labels = sorted(df_group['y_coord'].unique(), key=sort_key, reverse=True)

    heatmap_data = pd.DataFrame(index=y_labels, columns=x_labels, dtype=np.float64)

    for _, row in df_group.iterrows():
        x = format_priority(row['p_grad_prop'])
        y = format_priority(row['p_fwd_prop'])
        try:
            val = float(row['total_time'])
            heatmap_data.loc[y, x] = val
        except:
            continue

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cmap='YlGnBu',             # 原配色方案
        vmin=50, vmax=90,          # 增强对比：更宽色阶范围
        cbar_kws={'label': 'Total Time'},
        square=True
    )

    title = ', '.join([f"{k}={v}" for k, v in config_dict.items()])
    plt.title(f"Heatmap for config: {title}", fontsize=14)
    plt.xlabel("p_grad_prop")
    plt.ylabel("p_fwd_prop")

    filename = '_'.join([f"{k}-{v}" for k, v in config_dict.items()])
    filepath = os.path.join(output_dir, f"heatmap_{filename}.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(filepath)
    print(f"[✓] Saved heatmap to: {filepath}")

    if show_plot:
        plt.show()
    plt.close()

def plot_heatmaps_from_csv(csv_path, output_dir="./heatmaps"):
    df = pd.read_csv(csv_path)

    # 保留 'a'，但过滤掉 total_time 非数字行
    df['total_time'] = pd.to_numeric(df['total_time'], errors='coerce')
    df = df.dropna(subset=['total_time'])

    config_columns = [
        'n', 'sync_freq', 'fwd_len', 'bwd_len', 'prop_len',
        'sync_base_len', 'fwd_impact', 'bwd_impact'
    ]

    grouped = df.groupby(config_columns)
    for config_values, group in grouped:
        config_dict = dict(zip(config_columns, config_values))
        plot_single_heatmap(group.copy(), config_dict, output_dir)

def plot_heatmaps(csv_path, output_dir="./heatmaps"):
    plot_heatmaps_from_csv(csv_path, output_dir)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_heatmap.py <your_file.csv>")
    else:
        csv_file = sys.argv[1]
        plot_heatmaps_from_csv(csv_file)
