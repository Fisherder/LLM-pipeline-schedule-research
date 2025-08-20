import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Use larger Times New Roman fonts globally
import os
import numpy as np
import sys

def format_priority(val):
    """
    将优先级值格式化为字符串，保留 'a'。对于数值，使用最高四位小数（数据生成时已四舍五入），并去掉多余的尾随零。

    例如：0.0500 -> '0.05'，0.1000 -> '0.1'，0.3333 -> '0.3333'。
    这样在步长较小（如 0.05、0.01）时也不会被错误地合并到 0.1 的格子中。
    """
    # 判断字母优先级（如 'a'）
    if isinstance(val, str) and val.strip().lower() == 'a':
        return 'a'
    try:
        fval = float(val)
        # 四舍五入到 4 位小数（数据生成时就是 4 位），然后移除尾随零和小数点
        formatted = f"{fval:.4f}".rstrip('0').rstrip('.')
        return formatted
    except Exception:
        # 对于无法解析的值，直接返回字符串
        return str(val)

def sort_key(val):
    """
    排序规则：数值从小到大；'a'视为大于 1.0，因此排在数值之后。
    其他无法转换为数值的值排在最后。
    """
    if val == 'a':
        # 将 'a' 视为大于 1.0 的特殊值，例如 2.0
        return (0, 2.0)
    try:
        return (0, float(val))
    except:
        return (1, val)

def plot_single_heatmap(df_group, config_dict, output_dir, show_plot=True):
    df_group['x_coord'] = df_group['p_grad_prop'].apply(format_priority)
    df_group['y_coord'] = df_group['p_fwd_prop'].apply(format_priority)

    x_labels = sorted(df_group['x_coord'].unique(), key=sort_key)
    # y 轴反转，使得数值大的排在图顶部；'a' 由于被视为 2.0，会位于最顶部
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
    # 动态计算颜色上下限：分别取最大值向上取整到 10 的倍数，最小值向下取整到 10 的倍数
    import math
    try:
        numeric_values = heatmap_data.values.astype(float)
        valid = numeric_values[~np.isnan(numeric_values)]
        if valid.size > 0:
            max_val = np.max(valid)
            min_val = np.min(valid)
            # 若所有值相等，仍取最近 10 的倍数区间
            vmax = math.ceil(max_val / 10.0) * 10.0
            vmin = math.floor(min_val / 10.0) * 10.0
            # 若 vmax == vmin，则扩大区间
            if abs(vmax - vmin) < 1e-9:
                vmin = max(0.0, vmin - 10.0)
                vmax = vmax + 10.0
        else:
            vmax, vmin = 1.0, 0.0
    except Exception:
        # 回退默认范围
        vmin, vmax = 0.0, 1.0
    # 绘制热力图并获取轴对象
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cmap='YlGnBu',
        vmin=vmin, vmax=vmax,
        cbar_kws={'label': 'Total Time'},
        square=True
    )

    # 标题和坐标轴标签
    title = ', '.join([f"{k}={v}" for k, v in config_dict.items()])
    plt.title(f"Heatmap for config: {title}", fontsize=18)
    plt.xlabel("p_grad_prop", fontsize=18)
    plt.ylabel("p_fwd_prop", fontsize=18)

    # 寻找最优点（总时间最低）并标记
    try:
        # 使用 stack() 展平成 Series 以便找出最小值
        min_val = heatmap_data.stack().astype(float).min()
        # 获取最小值所在的行列索引
        best_positions = []
        for y_idx, y_label in enumerate(heatmap_data.index):
            for x_idx, x_label in enumerate(heatmap_data.columns):
                cell_val = heatmap_data.loc[y_label, x_label]
                try:
                    cell_float = float(cell_val)
                except (TypeError, ValueError):
                    continue
                if abs(cell_float - min_val) < 1e-9:
                    best_positions.append((y_idx, x_idx))
        # 在这些位置绘制红色星标
        for y_idx, x_idx in best_positions:
            ax.scatter(x_idx + 0.5, y_idx + 0.5, marker='*', s=150, color='red', edgecolors='white', linewidths=0.5)
    except Exception:
        pass

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

    # 动态获取配置参数列：排除共享优先级列和 total_time。这样可以适配任意配置 JSON 中定义的参数。
    priority_cols = {'p_fwd_prop', 'p_grad_prop', 'p_grad_sync', 'total_time'}
    config_columns = [c for c in df.columns if c not in priority_cols]
    if not config_columns:
        # 如果无法识别配置参数，则直接绘制一个整体热力图
        plot_single_heatmap(df.copy(), {}, output_dir)
        return

    grouped = df.groupby(config_columns)
    for config_values, group in grouped:
        config_dict = dict(zip(config_columns, config_values))
        plot_single_heatmap(group.copy(), config_dict, output_dir)

def plot_heatmaps(csv_path, output_dir="./heatmaps"):
    plot_heatmaps_from_csv(csv_path, output_dir)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_heatmap.py <csv_file>")
    else:
        csv_file = sys.argv[1]
        plot_heatmaps_from_csv(csv_file)
