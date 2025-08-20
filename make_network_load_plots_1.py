#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成单通道 GPipe 与 1F1B 的网络流量图（统一调用主代码：schedule_visualizer_1f1b_single_channel.py）：
- GPipe:  通过 mode='gpipe' 指定
- 1F1B:   通过 mode='1f1b' 指定（默认开启接收端拥塞控制）

可指定主代码中的关键参数：gpus、n（micro-batches）、sync_freq、impact、sync_solo_w、是否 ideal、各算子长度与优先级等。

输出（改为 PDF）：
- [<prefix>_]network_load_gpipe_single_{bar|line}.pdf
- [<prefix>_]network_load_1f1b_single_{with_congestion|no_congestion}_{bar|line}.pdf

用法示例：
python make_network_load_plots.py --mode both -n 8 --sync-freq 2 --gpus 4 --dpi 300 --prefix Default
"""

import argparse
import re
from typing import Dict, List, Tuple
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator

# 全局字体：加粗并进一步增大（坐标轴 & 刻度更大）
rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 26,              # 基础字号（略增）
    'font.weight': 'bold',
    'axes.titlesize': 28,
    'axes.titleweight': 'bold',
    'axes.labelsize': 30,         # 坐标轴标题更大
    'axes.labelweight': 'bold',
    'xtick.labelsize': 28,        # 刻度字号更大
    'ytick.labelsize': 28,
    'legend.fontsize': 26,
    'savefig.format': 'pdf',      # 默认保存为 PDF（也由文件后缀决定）
})


def sanitize_prefix(s: str) -> str:
    """将前缀清理为安全的文件名片段：去首尾、把所有空白压成下划线、非法字符替换成 '-'。"""
    s = s.strip()
    if not s:
        return ""
    s = "_".join(s.split())
    s = re.sub("[^A-Za-z0-9._-]", "-", s)
    return s


def compute_network_load(blocks: List[Dict]) -> Tuple[List[float], List[float], Dict[str, List[float]]]:
    """从调度 blocks 统计通信带宽随时间的占用（单通道场景）。
    返回：左端时间数组、对应时间片宽度数组、各类型占用。
    """
    comm_types = {"fwd_prop", "grad_prop", "grad_sync"}
    times = sorted({b["start"] for b in blocks} | {b["end"] for b in blocks})
    loads = {"total": [], "fwd_prop": [], "grad_prop": [], "grad_sync": []}
    widths: List[float] = []
    for i in range(len(times) - 1):
        t0, t1 = times[i], times[i + 1]
        widths.append(t1 - t0)
        total = fwd = grad = sync = 0.0
        for b in blocks:
            if b["type"] in comm_types and b["start"] <= t0 and b["end"] >= t1:
                w = float(b.get("width", 0.0))
                total += w
                if b["type"] == "fwd_prop":
                    fwd += w
                elif b["type"] == "grad_prop":
                    grad += w
                elif b["type"] == "grad_sync":
                    sync += w
        loads["total"].append(total)
        loads["fwd_prop"].append(fwd)
        loads["grad_prop"].append(grad)
        loads["grad_sync"].append(sync)
    return times[:-1], widths, loads


def plot_network_load(
    times: List[float],
    widths: List[float],
    loads: Dict[str, List[float]],
    title: str,
    outfile: str,
    dpi: int = 300,
    style: str = 'bar',
    line_width: float = 4.0,
) -> None:
    """绘图：支持折线/柱状图；图例单行且位于标题与主图之间；梯度同步使用更醒目的黄色；提高导出分辨率。"""
    fig, ax = plt.subplots(figsize=(12.5, 5.0))  # 尺寸保持；PDF 矢量导出

    # 颜色（与主调度图一致）
    col_fwd = "#87ceeb"     # 天蓝
    col_grad = "#c8a2c8"    # 紫
    col_sync = "#ffcc33"    # 更明显的黄

    if style == 'bar':
        # 堆叠柱（每个时间片一根柱，宽度=时间片长度，左对齐）
        ax.bar(times, loads["fwd_prop"], width=widths, align='edge', label="Fwd Prop", color=col_fwd)
        ax.bar(times, loads["grad_prop"], width=widths, align='edge', bottom=loads["fwd_prop"], label="Grad Prop", color=col_grad)
        bottom_sync = [f + g for f, g in zip(loads["fwd_prop"], loads["grad_prop"])]
        ax.bar(times, loads["grad_sync"], width=widths, align='edge', bottom=bottom_sync, label="Grad Sync", color=col_sync)
    else:
        # 折线（保留总占用）
        ax.plot(times, loads["total"], label="Total Comm", color="black", linewidth=line_width)
        ax.plot(times, loads["fwd_prop"], label="Fwd Prop", color=col_fwd, linewidth=line_width)
        ax.plot(times, loads["grad_prop"], label="Grad Prop", color=col_grad, linewidth=line_width)
        ax.plot(times, loads["grad_sync"], label="Grad Sync", color=col_sync, linewidth=line_width)

    # 轴标题加粗（字号进一步增大）
    ax.set_xlabel("Time", fontsize=30, fontweight='bold')
    ax.set_ylabel("Bandwidth Usage", fontsize=30, fontweight='bold')

    # X 轴主刻度固定为 5 的整数倍，并把右侧上限对齐到 5 的整数倍
    if times:
        xmax = times[-1]
        xmax_round = int(math.ceil(xmax / 5.0) * 5)
        ax.set_xlim(0, xmax_round)
    ax.xaxis.set_major_locator(MultipleLocator(5))

    # 使用 suptitle 把标题放在图最上方，为下方图例留出空间
    fig.suptitle(title, fontsize=28, fontweight='bold', y=0.98)

    # 在标题与主图之间放置单行图例（不遮挡主图）
    fig.subplots_adjust(top=0.82, bottom=0.16, left=0.11, right=0.98)
    ncols = 3 if style == 'bar' else 4
    fig.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 0.92),   # 介于 suptitle(0.98) 与 axes 顶部(0.82) 之间
        ncol=ncols,                   # 单行
        frameon=False,
        handlelength=1.5,
        handleheight=1.0,
        handletextpad=0.4,
        columnspacing=0.8,
        borderaxespad=0.1,
        prop={'weight': 'bold', 'size': 22}
    )

    # 刻度也加粗 + 更大字号
    ax.tick_params(axis='both', labelsize=28)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('bold')

    # 保存为 PDF（矢量）。dpi 仅影响可能的栅格化元素。
    plt.savefig(outfile, dpi=dpi)
    plt.close()


# ---------------------------
# 统一从“主代码文件”导入与构造参数
# ---------------------------

def build_lengths_from_args(pipeline_mod, args):
    """基于命令行覆盖默认的算子时长。"""
    lengths = dict(pipeline_mod.default_lengths)
    if args.len_forward is not None:
        lengths['forward'] = args.len_forward
    if args.len_backward is not None:
        lengths['backward'] = args.len_backward
    if args.len_fwd_prop is not None:
        lengths['fwd_prop'] = args.len_fwd_prop
    if args.len_grad_prop is not None:
        lengths['grad_prop'] = args.len_grad_prop
    if args.len_grad_sync is not None:
        lengths['grad_sync'] = args.len_grad_sync
    if 'fwd_prop' not in lengths:
        lengths['fwd_prop'] = lengths['grad_prop']
    return lengths


def build_priorities_from_args(pipeline_mod, args):
    """基于命令行覆盖默认的优先级。"""
    priorities = dict(pipeline_mod.default_priorities)
    if args.prio_fwd_prop is not None:
        priorities['fwd_prop'] = args.prio_fwd_prop
    if args.prio_grad_prop is not None:
        priorities['grad_prop'] = args.prio_grad_prop
    if args.prio_grad_sync is not None:
        priorities['grad_sync'] = args.prio_grad_sync
    if args.prio_forward is not None:
        priorities['forward'] = args.prio_forward
    if args.prio_backward is not None:
        priorities['backward'] = args.prio_backward
    return priorities


def run_mode(
    mode: str,
    n: int,
    gpus: int,
    is_ideal: bool,
    lengths,
    priorities,
    exclusive_tiers,
    impact: float,
    sync_solo_w: float,
    sync_freq: int,
    use_recv_congestion: bool,
) -> List[Dict]:
    """用统一主文件的 calculate_full_pipeline_schedule 计算给定模式（'1f1b' 或 'gpipe'）的 blocks。"""
    import schedule_visualizer_1f1b_single_channel as pipeline
    blocks, total_time, _throttled = pipeline.calculate_full_pipeline_schedule(
        n=n,
        num_gpus=max(2, gpus),
        is_ideal=is_ideal,
        lengths=lengths,
        priorities=priorities,
        exclusive_tiers=exclusive_tiers,
        impact=impact,
        sync_solo_w=sync_solo_w,
        sync_freq=sync_freq,
        use_recv_congestion=use_recv_congestion,
        mode=mode,
    )
    return blocks


def run_gpipe(args) -> str:
    import schedule_visualizer_1f1b_single_channel as pipeline
    lengths = build_lengths_from_args(pipeline, args)
    priorities = build_priorities_from_args(pipeline, args)
    blocks = run_mode(
        mode='gpipe',
        n=args.num_microbatches,
        gpus=args.gpus,
        is_ideal=args.ideal,
        lengths=lengths,
        priorities=priorities,
        exclusive_tiers=pipeline.default_exclusive_tiers,
        impact=args.impact_gpipe,
        sync_solo_w=args.sync_solo_w,
        sync_freq=args.sync_freq,
        use_recv_congestion=False,  # GPipe 下该开关不影响，但明确设为 False
    )
    times, widths, loads = compute_network_load(blocks)
    prefix_slug = (sanitize_prefix(args.prefix) + "_") if args.prefix else ""
    out = f"{prefix_slug}network_load_gpipe_single_{args.style}.pdf"
    base = "GPipe Network Load"
    title = f"{args.prefix.strip()} {base}" if args.prefix else base
    plot_network_load(times, widths, loads, title, out, dpi=args.dpi, style=args.style, line_width=args.line_width)
    return out


def run_1f1b(args) -> str:
    import schedule_visualizer_1f1b_single_channel as pipeline
    lengths = build_lengths_from_args(pipeline, args)
    priorities = build_priorities_from_args(pipeline, args)
    # 默认开启接收端拥塞控制（可用 --disable-recv-congestion 关闭）
    use_cong = (not args.disable_recv_congestion)
    blocks = run_mode(
        mode='1f1b',
        n=args.num_microbatches,
        gpus=args.gpus,
        is_ideal=args.ideal,
        lengths=lengths,
        priorities=priorities,
        exclusive_tiers=pipeline.default_exclusive_tiers,
        impact=args.impact_1f1b,
        sync_solo_w=args.sync_solo_w,
        sync_freq=args.sync_freq,
        use_recv_congestion=use_cong,
    )
    times, widths, loads = compute_network_load(blocks)
    suffix = "with_congestion" if use_cong else "no_congestion"
    prefix_slug = (sanitize_prefix(args.prefix) + "_") if args.prefix else ""
    out = f"{prefix_slug}network_load_1f1b_single_{suffix}_{args.style}.pdf"
    base = "1F1B Network Load"
    title = f"{args.prefix.strip()} {base}" if args.prefix else base
    plot_network_load(times, widths, loads, title, out, dpi=args.dpi, style=args.style, line_width=args.line_width)
    return out


def main():
    ap = argparse.ArgumentParser(description="基于统一主代码的单通道网络流量图生成器（支持折线/柱状）")
    ap.add_argument("--mode", choices=["gpipe", "1f1b", "both"], default="both", help="选择生成哪种模式的图")
    # 主代码核心参数
    ap.add_argument("-n", "--num_microbatches", type=int, default=8, help="微批数量 n")
    ap.add_argument("--gpus", type=int, default=4, help="GPU 数量 (>=2)")
    ap.add_argument("--sync-freq", type=int, default=2, help="grad_sync 频率")
    ap.add_argument("--impact-1f1b", dest="impact_1f1b", type=float, default=0.2, help="1F1B 下 grad_sync 对计算的影响系数（默认 0.2）")
    ap.add_argument("--impact-gpipe", dest="impact_gpipe", type=float, default=0.0, help="GPipe 下的影响系数（默认 0.0）")
    ap.add_argument("--sync-solo-w", type=float, default=1.0, help="单独执行的 sync 任务占用的资源份额（与主代码一致）")
    ap.add_argument("--ideal", action="store_true", help="使用 ideal 渠道（主代码 is_ideal=True）")

    # 1F1B 的接收端拥塞控制：默认开启，可通过该开关禁用
    ap.add_argument("--disable-recv-congestion", action="store_true", help="关闭 1F1B 的接收端拥塞控制（默认开启）")

    # （可选）覆盖算子时长
    ap.add_argument("--len-forward", type=float, help="前向计算时长")
    ap.add_argument("--len-backward", type=float, help="反向计算时长")
    ap.add_argument("--len-fwd-prop", type=float, help="前向传播通信时长")
    ap.add_argument("--len-grad-prop", type=float, help="梯度传播通信时长")
    ap.add_argument("--len-grad-sync", type=float, help="梯度同步通信时长")

    # （可选）覆盖优先级
    ap.add_argument("--prio-forward", type=float, help="前向计算优先级")
    ap.add_argument("--prio-backward", type=float, help="反向计算优先级")
    ap.add_argument("--prio-fwd-prop", type=float, help="前向传播通信优先级")
    ap.add_argument("--prio-grad-prop", type=float, help="梯度传播通信优先级")
    ap.add_argument("--prio-grad-sync", type=float, help="梯度同步通信优先级")

    # 导出分辨率（DPI）与风格
    ap.add_argument("--dpi", type=int, default=400, help="保存图片的 DPI（默认 400，PDF 矢量为主，仅影响栅格化元素）")
    ap.add_argument("--style", choices=["bar", "line"], default="bar", help="绘图风格：柱状(bar)/折线(line)，默认柱状")
    ap.add_argument("--line-width", type=float, default=2.5, help="折线图线宽（单位：Pt），仅在 --style line 时生效")
    ap.add_argument("--prefix", type=str, default="", help="标题前缀，例如 --prefix Default => 'Default <原标题>'")

    args = ap.parse_args()

    outs: List[str] = []
    if args.mode in ("gpipe", "both"):
        outs.append(run_gpipe(args))
    if args.mode in ("1f1b", "both"):
        outs.append(run_1f1b(args))

    print("Saved:", ", ".join(outs))


if __name__ == "__main__":
    main()
