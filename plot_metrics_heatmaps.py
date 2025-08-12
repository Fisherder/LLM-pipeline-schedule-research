#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 要绘制的指标；CSV缺哪个就自动跳过
DEFAULT_METRICS = [
    "total_time",
    "compute_ratio",
    "avg_throughput",
    "avg_latency",
    "compute_utilization",
]

# 与你的 CSV 一致的配置分组列
CONFIG_COLUMNS = [
    "n", "sync_freq", "fwd_len", "bwd_len", "prop_len",
    "sync_base_len", "fwd_impact", "bwd_impact"
]

# 统一配色方案（最小->最大同一渐变）
UNIFIED_CMAP = "YlGnBu"

# ---------- 保留 'a' 的格式化与排序规则 ----------

def format_priority(val):
    """将优先级值格式化为字符串，保留 'a'；其余数字保留两位小数。"""
    s = str(val).strip()
    if s.lower() == "a":
        return "a"
    try:
        return f"{float(s):.2f}"
    except:
        return s  # 其它无效值保留原样

def sort_key(val):
    """排序：数值升序，其它无效值其次，'a' 最大（最后）。"""
    if isinstance(val, str) and val.lower() == "a":
        return (2, float("inf"))  # 'a' 最大
    try:
        return (0, float(val))     # 正常数值
    except:
        return (1, str(val))       # 其它无效值排在数值之后、'a'之前

# ---------- 绘图核心 ----------

def _make_heatmap_table(df_group: pd.DataFrame, metric: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """把一组相同配置的数据，整理成以 p_fwd_prop / p_grad_prop 为轴的表"""
    # 坐标标签（保留 'a'）
    x_col = df_group["p_grad_prop"].apply(format_priority)
    y_col = df_group["p_fwd_prop"].apply(format_priority)

    x_labels = sorted(x_col.unique(), key=sort_key)
    # y 从大到小，且我们把 'a' 视为最大，因此反转后 'a' 会在最上方
    y_labels = sorted(y_col.unique(), key=sort_key, reverse=True)

    table = pd.DataFrame(index=y_labels, columns=x_labels, dtype=np.float64)

    # 填值（只对可数值化的该指标赋值）
    for _, row in df_group.iterrows():
        x = format_priority(row["p_grad_prop"])
        y = format_priority(row["p_fwd_prop"])
        val = pd.to_numeric(row.get(metric, np.nan), errors="coerce")
        if pd.notna(val):
            table.loc[y, x] = float(val)

    return table, x_labels, y_labels

def _best_cell(table: pd.DataFrame, metric: str):
    """找最优点：total_time/avg_latency 取最小，其它取最大"""
    arr = table.values.astype(float)
    if np.all(np.isnan(arr)):
        return None
    if metric in ("total_time", "avg_latency"):
        idx = np.nanargmin(arr)
    else:
        idx = np.nanargmax(arr)
    r, c = np.unravel_index(idx, arr.shape)
    y = table.index[r]
    x = table.columns[c]
    v = arr[r, c]
    return x, y, v

def plot_single_heatmap(df_group: pd.DataFrame,
                        config_dict: Dict[str, object],
                        output_dir: Path,
                        metric: str,
                        cmap: str = UNIFIED_CMAP,
                        vmin: float = None,
                        vmax: float = None,
                        show_plot: bool = False):
    table, x_labels, y_labels = _make_heatmap_table(df_group, metric)

    if table.isna().all().all():
        print(f"[!] 跳过：该配置下 '{metric}' 全为空。")
        return

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        table,
        annot=True,
        fmt=".2f",                 # 两位小数
        linewidths=0.5,
        cmap=cmap,                 # 统一配色方案
        vmin=vmin, vmax=vmax,      # 不指定则自适应
        cbar_kws={'label': metric.replace("_", " ").title()},
        square=True
    )

    title = ", ".join([f"{k}={v}" for k, v in config_dict.items()])
    plt.title(f"{metric} | config: {title}", fontsize=14)
    plt.xlabel("p_grad_prop")
    plt.ylabel("p_fwd_prop")

    # 用红色标记最优格（红色边框矩形）
    best = _best_cell(table, metric)
    if best:
        bx, by, bval = best
        try:
            yi = list(table.index).index(by)
            xi = list(table.columns).index(bx)
            # 在该单元格外圈画红色边框
            rect = Rectangle((xi, yi), 1, 1, fill=False, edgecolor='red', linewidth=2.5)
            ax.add_patch(rect)
            # 也可在中心加红色小点，增强可见性
            ax.scatter(xi + 0.5, yi + 0.5, s=120, c='red', marker='o')
        except Exception:
            pass

    # 文件名
    filename = "_".join([f"{k}-{v}" for k, v in config_dict.items()])
    outname = f"heatmap_{metric}_{filename}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    outpath = output_dir / outname
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    print(f"[✓] Saved: {outpath}")
    if show_plot:
        plt.show()
    plt.close()

def plot_heatmaps_from_csv(csv_path: Path,
                           output_dir: Path,
                           metrics: List[str],
                           vmin: float = None,
                           vmax: float = None,
                           show_plot: bool = False):
    df = pd.read_csv(csv_path)

    # total_time 非数值的行过滤（其它指标在绘制阶段各自再做 numeric 过滤）
    if "total_time" in df.columns:
        df = df[pd.to_numeric(df["total_time"], errors="coerce").notna()].copy()

    # 分组
    missing_cfg = [c for c in CONFIG_COLUMNS if c not in df.columns]
    if missing_cfg:
        raise ValueError(f"CSV 缺少配置列: {missing_cfg}")

    grouped = df.groupby(CONFIG_COLUMNS, dropna=False)

    for config_values, group in grouped:
        config_dict = dict(zip(CONFIG_COLUMNS, config_values))
        for metric in metrics:
            if metric not in group.columns:
                continue
            plot_single_heatmap(
                group.copy(),
                config_dict,
                output_dir,
                metric=metric,
                cmap=UNIFIED_CMAP,  # 统一配色
                vmin=vmin, vmax=vmax,
                show_plot=show_plot
            )

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser("Plot heatmaps from metrics CSV（保留 'a'，两位小数，统一配色；'a' 最大；红色标注最优）")
    ap.add_argument("--csv", nargs="+", required=True, help="一个或多个 CSV 路径")
    # --outdir 改为可选；若不提供，则默认按每个 CSV 单独目录 ./outputs/metrics_<csv名>/
    ap.add_argument("--outdir", default=None, help="输出目录（可选）。不填则按每个CSV生成 ./outputs/metrics_<csv名>/")
    ap.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS,
                    help=f"要绘制的指标，默认：{DEFAULT_METRICS}")
    ap.add_argument("--vmin", type=float, default=None, help="色阶下限（可选）")
    ap.add_argument("--vmax", type=float, default=None, help="色阶上限（可选）")
    ap.add_argument("--show", action="store_true", help="显示窗口（调试用）")
    args = ap.parse_args()

    csv_list = [Path(c) for c in args.csv]

    for c in csv_list:
        # 默认输出目录：./outputs/metrics_<csv名>/
        if args.outdir:
            outdir_for_this = Path(args.outdir)
        else:
            outdir_for_this = Path("./outputs") / f"metrics_{c.stem}"
        plot_heatmaps_from_csv(
            c,
            outdir_for_this,
            metrics=args.metrics,
            vmin=args.vmin,
            vmax=args.vmax,
            show_plot=args.show
        )

    print("✅ Done.")

if __name__ == "__main__":
    main()
