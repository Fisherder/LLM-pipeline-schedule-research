#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成单通道 GPipe 与 1F1B 的网络流量折线图：
- GPipe:  schedule_visualizer_gpipe_single_channel.py
- 1F1B:   schedule_visualizer_1f1b_single_channel.py  (支持 use_recv_congestion 参数)

输出：
- network_load_gpipe_single.png
- network_load_1f1b_single_no_congestion.png  (默认关闭接收端拥塞控制)

用法示例：
python make_network_load_plots.py --mode both -n 8 --sync-freq 2
"""
import argparse
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Use larger Times New Roman fonts globally
rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 14,
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})

def compute_network_load(blocks: List[Dict]) -> Tuple[List[float], Dict[str, List[float]]]:
    """从调度 blocks 统计通信带宽随时间的占用（单通道场景）。"""
    # 仅统计通信任务
    comm_types = {"fwd_prop", "grad_prop", "grad_sync"}
    # 收集所有事件边界
    times = sorted({b["start"] for b in blocks} | {b["end"] for b in blocks})
    loads = {"total": [], "fwd_prop": [], "grad_prop": [], "grad_sync": []}
    # 逐时间片累计“带宽份额”（width）。在单通道模型里它可视为瞬时占比。
    for i in range(len(times) - 1):
        t0, t1 = times[i], times[i + 1]
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
    return times[:-1], loads

def plot_network_load(times: List[float], loads: Dict[str, List[float]], title: str, outfile: str) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(times, loads["total"], label="Total Comm", color="black")
    plt.plot(times, loads["fwd_prop"], label="Fwd Prop",  color="#87ceeb")  # 天蓝
    plt.plot(times, loads["grad_prop"], label="Grad Prop", color="#c8a2c8")  # 紫
    plt.plot(times, loads["grad_sync"], label="Grad Sync", color="#fffacd")  # 浅黄
    plt.xlabel("Time", fontsize=16)
    plt.ylabel("Aggregated Bandwidth Usage", fontsize=16)
    plt.title(title, fontsize=20)
    plt.legend(loc="upper right", fontsize=14)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

def run_gpipe_single(n: int, sync_freq: int) -> str:
    import schedule_visualizer_gpipe_single_channel as gpipe
    # 兼容不同返回值版本（2或3个）
    result = gpipe.calculate_full_pipeline_schedule(
        n, 4, False,
        gpipe.default_lengths,
        gpipe.default_priorities,
        gpipe.default_exclusive_tiers,
        0.0, 0.0, 1.0, sync_freq
    )
    blocks = result[0] if isinstance(result, tuple) else result
    times, loads = compute_network_load(blocks)
    out = "network_load_gpipe_single.png"
    plot_network_load(times, loads, "Network Load (Single Channel, GPipe)", out)
    return out

def run_1f1b_single(n: int, sync_freq: int, use_recv_congestion: bool) -> str:
    import schedule_visualizer_1f1b_single_channel as onef1b
    # 新版支持 use_recv_congestion（你已修改过）
    result = onef1b.calculate_full_pipeline_schedule(
        n=n, num_gpus=4, is_ideal=False,
        lengths=onef1b.default_lengths,
        priorities=onef1b.default_priorities,
        exclusive_tiers=onef1b.default_exclusive_tiers,
        fwd_impact=0.2, bwd_impact=0.2,
        sync_solo_w=1.0, sync_freq=sync_freq,
        use_recv_congestion=use_recv_congestion
    )
    blocks = result[0]
    times, loads = compute_network_load(blocks)
    suffix = "with_congestion" if use_recv_congestion else "no_congestion"
    out = f"network_load_1f1b_single_{suffix}.png"
    title = f"Network Load (Single Channel, 1F1B, recv_cc={'ON' if use_recv_congestion else 'OFF'})"
    plot_network_load(times, loads, title, out)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gpipe", "1f1b", "both"], default="both")
    ap.add_argument("-n", "--num_microbatches", type=int, default=8)
    ap.add_argument("--sync-freq", type=int, default=2)
    ap.add_argument("--enable-recv-congestion", action="store_true",
                    help="开启1F1B的接收端拥塞控制；不加该参数则关闭")
    args = ap.parse_args()

    outs = []
    if args.mode in ("gpipe", "both"):
        outs.append(run_gpipe_single(args.num_microbatches, args.sync_freq))
    if args.mode in ("1f1b", "both"):
        outs.append(run_1f1b_single(args.num_microbatches, args.sync_freq,
                                    use_recv_congestion=args.enable_recv_congestion))

    print("Saved:", ", ".join(outs))

if __name__ == "__main__":
    main()
