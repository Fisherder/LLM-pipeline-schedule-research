"""
文件名: schedule_visualizer_gpipe_ideal.py

描述:
    本脚本模拟基于 GPipe 的流水线并行训练的理想场景调度，在所有 micro‑batch
    完成前向阶段之后再执行反向阶段（F‑then‑B 模式）。与常规模拟不同之处在于，
    理想场景下各通信行为彼此独占各自的通道，并且计算不受梯度同步影响。

    具体区别包括：
      1. 每个 GPU 拥有 4 条资源通道，其中 1 条用于计算任务（前向和反向），其余 3
         条分别用于前向激活传递 (fwd_prop)、反向梯度传递 (grad_prop) 以及梯度同步
         (grad_sync)。每种通信任务只在其对应的通道上排队和运行，彼此之间不存在
         带宽竞争，也不受接收端拥塞控制影响。
      2. 取消梯度同步对前向/反向计算的影响（即无 ``fwd_impact`` 和 ``bwd_impact`` 调节）。
         计算任务仅按照依赖顺序串行执行，不会因通信而降低速度。

    绘图时按照通道类别从上到下依次为：梯度同步、梯度传递、计算、前向传递，直观展示
    各 GPU 在理想场景下的调度情况。

用法:
    python schedule_visualizer_gpipe_ideal.py [-n NUM_GROUPS] [--sync-freq FREQ]
                                              [--no-show] [--show-completion-time]

    与其他调度可视化脚本类似，``-n`` 指定 micro‑batch 数量，``--sync-freq`` 控制
    每多少个 micro‑batch 聚合一次梯度同步。默认每 2 个 micro‑batch 做一次同步。
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
from fractions import Fraction
from copy import deepcopy
from collections import defaultdict
import numpy as np
from itertools import groupby

# --- 全局定义 ---
W_UNIT = 1.0

# 颜色方案，与其他脚本保持一致
colors = {
    'forward':   '#a7c7e7',  # 前向计算
    'backward':  '#c1e1c1',  # 反向计算
    'fwd_prop':  '#87ceeb',  # 激活传递
    'grad_prop': '#d1b3e2',  # 梯度传递
    'grad_sync': '#fffacd',  # 梯度同步
    'param_sync': '#ffb3ba', # 参数同步（未使用）
    'bubble':    '#f0f0f0'   # 闲置时间（绘图用）
}

# 默认任务长度，可通过命令行覆盖
default_lengths = {
    'forward': 1,
    'backward': 2,
    'fwd_prop': 1,
    'grad_prop': 1,
    'grad_sync': 2,
    'param_sync': 1
}

# 通信任务的优先级权重，理想场景下由于各通信分离，权重仅在相同通道内起作用
default_priorities = {
    'forward': 1,
    'backward': 1,
    'fwd_prop': 1,
    'grad_prop': 1,
    'grad_sync': 1,
    'param_sync': 1
}

# 独占任务等级定义，理想场景下不启用主动独占
default_exclusive_tiers = {
    'fwd_prop': None,
    'grad_prop': None,
    'grad_sync': None,
}


def calculate_full_pipeline_schedule(n, num_gpus, lengths, priorities, exclusive_tiers, sync_freq=1):
    """模拟理想场景下 GPipe F‑then‑B 流水线的完整调度。

    参数说明:
        n (int): micro‑batch 数量。
        num_gpus (int): GPU/阶段数量。
        lengths (dict): 各任务类型所需的工作量。
        priorities (dict): 各类型通信任务的权重，用于分配带宽。
        exclusive_tiers (dict): 主动独占任务定义，理想场景未使用。
        sync_freq (int): 每多少个 micro‑batch 执行一次梯度同步。

    返回:
        all_blocks (list): 记录调度结果的时间段，用于绘图。
        total_time (float): 模拟完成的总时间。
    """
    all_tasks = []
    # 定义每个 GPU 的通道布局：
    #   ch_offset + 0: 计算 (forward/backward)
    #   ch_offset + 1: 前向激活传递 (fwd_prop)
    #   ch_offset + 2: 反向梯度传递 (grad_prop)
    #   ch_offset + 3: 梯度同步 (grad_sync)
    for j in range(num_gpus):
        base_ch = j * 4
        compute_ch = base_ch + 0
        fwd_ch = base_ch + 1
        grad_ch = base_ch + 2
        sync_ch = base_ch + 3
        for i in range(n):
            # 前向计算
            all_tasks.append({
                'id': ('forward', i, j),
                'type': 'forward',
                'work_needed': lengths['forward'],
                'dependencies': [],
                'status': 'pending',
                'channel': compute_ch,
                'label': f'{i + 1}'
            })
            # 反向计算
            all_tasks.append({
                'id': ('backward', i, j),
                'type': 'backward',
                'work_needed': lengths['backward'],
                'dependencies': [],
                'status': 'pending',
                'channel': compute_ch,
                'label': f'{i + 1}'
            })
            # 前向激活传递（除最后一个阶段）
            if j < num_gpus - 1:
                all_tasks.append({
                    'id': ('fwd_prop', i, j),
                    'type': 'fwd_prop',
                    'work_needed': lengths['fwd_prop'],
                    'dependencies': [],
                    'status': 'pending',
                    'channel': fwd_ch,
                    'label': f'{i + 1}'
                })
            # 反向梯度传递（除第一个阶段）
            if j > 0:
                all_tasks.append({
                    'id': ('grad_prop', i, j),
                    'type': 'grad_prop',
                    'work_needed': lengths['grad_prop'],
                    'dependencies': [],
                    'status': 'pending',
                    'channel': grad_ch,
                    'label': f'{i + 1}'
                })
            # 梯度同步：根据 sync_freq 合并多个 micro‑batch
            is_sync_point = (i + 1) % sync_freq == 0
            is_last_batch = (i == n - 1)
            is_leftover_sync = is_last_batch and ((i + 1) % sync_freq != 0)
            if is_sync_point or is_leftover_sync:
                num_grads = (i + 1) % sync_freq if is_leftover_sync else sync_freq
                sync_work = lengths['grad_sync'] * num_grads
                all_tasks.append({
                    'id': ('grad_sync', i, j),
                    'type': 'grad_sync',
                    'work_needed': sync_work,
                    'dependencies': [],
                    'status': 'pending',
                    'channel': sync_ch,
                    'label': f'{(i // sync_freq) + 1}'
                })
    # 建立依赖关系，与单通道 GPipe F‑then‑B 相同
    barrier_id = ('forward', n - 1, num_gpus - 1)
    for task in all_tasks:
        if len(task['id']) < 3:
            continue
        task_type, i, j = task['id']
        if task_type == 'forward':
            # 阶段 0：前向串行；其它阶段依赖上一阶段激活传递
            if j == 0:
                if i > 0:
                    task['dependencies'].append(('forward', i - 1, j))
            else:
                task['dependencies'].append(('fwd_prop', i, j - 1))
        elif task_type == 'backward':
            # 反向依赖对应前向计算和下一阶段梯度传递，且需等待全局屏障
            task['dependencies'].append(('forward', i, j))
            if j < num_gpus - 1:
                task['dependencies'].append(('grad_prop', i, j + 1))
            task['dependencies'].append(barrier_id)
        elif task_type == 'fwd_prop':
            task['dependencies'].append(('forward', i, j))
        elif task_type == 'grad_prop':
            task['dependencies'].append(('backward', i, j))
        elif task_type == 'grad_sync':
            task['dependencies'].append(('backward', i, j))
            # 链式依赖上一同步任务
            prev_sync_idx = i - sync_freq
            if prev_sync_idx >= 0:
                prev_sync_id = ('grad_sync', prev_sync_idx, j)
                if any(t['id'] == prev_sync_id for t in all_tasks):
                    task['dependencies'].append(prev_sync_id)
    # 模拟事件循环
    current_time = 0.0
    blocks = []
    finished_task_ids = set()
    while len(finished_task_ids) < len(all_tasks):
        # 标记就绪任务
        for task in all_tasks:
            if task['status'] == 'pending' and all(dep in finished_task_ids for dep in task['dependencies']):
                task['status'] = 'ready'
        tasks_to_set_running = []
        shares_now = {}
        # 收集 ready 或 running 任务
        active_tasks = [t for t in all_tasks if t['status'] in ['ready', 'running']]
        tasks_by_channel = defaultdict(list)
        for task in active_tasks:
            tasks_by_channel[task['channel']].append(task)
        # 对每个通道分配带宽
        for ch, tasks_in_channel in tasks_by_channel.items():
            # 计算通道：每个 GPU 同一时间只能执行一个计算任务
            # 我们通过 channel % 4 == 0 判断是否为计算通道（见上面的定义）
            is_compute_channel = (ch % 4 == 0)
            eligible_tasks = [t for t in tasks_in_channel if t['status'] in ['ready', 'running']]
            if not eligible_tasks:
                continue
            if is_compute_channel:
                if any(t['status'] == 'running' for t in eligible_tasks):
                    tasks_for_allocation = [t for t in eligible_tasks if t['status'] == 'running']
                else:
                    # 优先前向，再反向；同类按 micro‑batch 顺序
                    def sort_key(task):
                        type_prio = 0 if task['type'] == 'forward' else 1
                        return (type_prio, task['id'][1])
                    eligible_tasks.sort(key=sort_key)
                    tasks_for_allocation = [eligible_tasks[0]]
                tasks_to_set_running.extend(tasks_for_allocation)
                for task in tasks_for_allocation:
                    shares_now[task['id']] = W_UNIT
            else:
                # 通信通道：理想场景下，每个通道只有对应的一种通信任务，不存在跨类竞争
                # 但可能有同类不同微批的任务同时 ready，按权重公平分配
                # 忽略独占机制，因为 exclusive_tiers 均为 None
                tasks_for_allocation = eligible_tasks
                tasks_by_type = defaultdict(list)
                for task in tasks_for_allocation:
                    tasks_by_type[task['type']].append(task)
                # 在理想场景中，一个通道只会有一个任务类型，但代码仍泛化处理
                type_priorities = {t: priorities.get(t, 0) for t in tasks_by_type.keys()}
                total_p = sum(type_priorities.values())
                if total_p > 1e-9:
                    for task_type, tasks_same_type in tasks_by_type.items():
                        share_for_type = W_UNIT * (type_priorities[task_type] / total_p)
                        share_per_task = share_for_type / len(tasks_same_type)
                        for task in tasks_same_type:
                            shares_now[task['id']] = share_per_task
                else:
                    # 没有优先级定义的情况，平分带宽
                    share_per_task = W_UNIT / len(tasks_for_allocation)
                    for task in tasks_for_allocation:
                        shares_now[task['id']] = share_per_task
                tasks_to_set_running.extend(tasks_for_allocation)
        # 更新 ready 任务状态为 running
        unique_tasks_to_run = {t['id']: t for t in tasks_to_set_running}.values()
        for task in unique_tasks_to_run:
            if task['status'] == 'ready':
                task['status'] = 'running'
        # 如果没有任务正在使用带宽，则跳过时间
        tasks_with_share = [t for t in active_tasks if shares_now.get(t['id'], 0) > 1e-9]
        if not tasks_with_share:
            if len(finished_task_ids) == len(all_tasks):
                break
            else:
                continue
        # 计算下一事件时间：所有运行任务的剩余时间中最小的
        min_time_to_finish = float('inf')
        for task in tasks_with_share:
            share = shares_now.get(task['id'], 0)
            if share > 1e-9:
                remaining = task.get('work_needed', 0) - task.get('work_done', 0)
                time_needed = remaining / share
                if time_needed < min_time_to_finish:
                    min_time_to_finish = time_needed
        slice_duration = min_time_to_finish
        if slice_duration == float('inf') or slice_duration <= 1e-9:
            slice_duration = 1e-9
        next_event_time = current_time + slice_duration
        # 更新 running 任务的已完成工作量并记录绘图块
        for task in tasks_with_share:
            if 'work_done' not in task:
                task['work_done'] = 0
            final_share = shares_now.get(task['id'], 0)
            task['work_done'] += final_share * slice_duration
            if final_share > 1e-9:
                blocks.append({
                    'id': task['id'],
                    'label': task['label'],
                    'type': task['type'],
                    'start': current_time,
                    'end': next_event_time,
                    'color': colors[task['type']],
                    'width': final_share,
                    'channel': task['channel']
                })
        current_time = next_event_time
        # 检查是否有任务完成
        for task in all_tasks:
            if task['status'] == 'running' and task.get('work_done', 0) >= task['work_needed'] - 1e-9:
                task['status'] = 'finished'
                finished_task_ids.add(task['id'])
    return blocks, current_time


# --- 绘图函数 ---
def plot_single_gpu_chart(gpu_rank, all_blocks, total_time, show_plot):
    """绘制单个 GPU 在理想场景下的任务调度时间轴。"""
    blocks_for_gpu = [b for b in all_blocks if b['id'][2] == gpu_rank]
    if not blocks_for_gpu:
        print(f"GPU {gpu_rank + 1} 没有任务，跳过绘图。")
        return
    # 通道顺序：最上层为梯度传递，表示向上游传播；其次为梯度同步，其次为计算通道，最下方为前向传递，表示向下游传播。
    VISUAL_FWD = 0
    VISUAL_COMP = 1
    VISUAL_SYNC = 2
    VISUAL_GRAD = 3
    # y_map 对应 Matplotlib 中 y 轴由下向上递增，数值越大越靠上。
    y_map = {
        VISUAL_FWD: 0,    # 前向传递在最下方
        VISUAL_COMP: 1,   # 计算通道在下第二层
        VISUAL_SYNC: 2,   # 梯度同步在下第三层
        VISUAL_GRAD: 3    # 梯度传递在最上方
    }
    # 标记从下到上的通道名称：底部为前向传递，其上是计算，再上是梯度同步，最上为梯度传递。
    visual_layout_labels = [
        "Fwd Prop", "Compute", "Grad Sync", "Grad Prop"
    ]
    num_channels = 4
    fig, ax = plt.subplots(1, 1, figsize=(22, 2.5 * num_channels))
    fig.suptitle(f"GPU {gpu_rank + 1} - GPipe F-then-B Schedule (Ideal)", fontsize=16, y=0.98)
    ax.set_ylabel("Resource Channels", fontsize=12)
    ax.set_ylim(0, num_channels)
    ax.set_yticks([i + 0.5 for i in range(num_channels)])
    ax.set_yticklabels(visual_layout_labels, fontsize=10)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    drawable_blocks = []
    time_slices = sorted(list(set([b['start'] for b in blocks_for_gpu] + [b['end'] for b in blocks_for_gpu])))
    for i in range(len(time_slices) - 1):
        start, end = time_slices[i], time_slices[i + 1]
        blocks_in_slice = [b for b in blocks_for_gpu if b['start'] <= start and b['end'] >= end]
        blocks_by_virtual_channel = defaultdict(list)
        for b in blocks_in_slice:
            # 根据任务类型映射到虚拟通道
            if b['type'] == 'grad_prop':
                ch = VISUAL_GRAD
            elif b['type'] == 'grad_sync':
                ch = VISUAL_SYNC
            elif b['type'] in ['forward', 'backward']:
                ch = VISUAL_COMP
            elif b['type'] == 'fwd_prop':
                ch = VISUAL_FWD
            else:
                ch = VISUAL_COMP
            blocks_by_virtual_channel[ch].append(b)
        for ch, blocks_on_ch in blocks_by_virtual_channel.items():
            y_base = y_map[ch]
            y_stack_offset = 0
            for block in sorted(blocks_on_ch, key=lambda b: b['type']):
                new_block = block.copy()
                new_block['start'], new_block['end'] = start, end
                new_block['y_pos'] = y_base + y_stack_offset
                new_block['height'] = block['width']
                drawable_blocks.append(new_block)
                y_stack_offset += block['width']
    key_func = lambda b: b['id']
    drawable_blocks.sort(key=key_func)
    for task_id, group_iter in groupby(drawable_blocks, key=key_func):
        group = sorted(list(group_iter), key=lambda b: b['start'])
        if not group:
            continue
        current_segment = []
        for block in group:
            if not current_segment or abs(block['start'] - current_segment[-1]['end']) < 1e-9:
                current_segment.append(block)
            else:
                draw_polygon_for_segment(ax, current_segment)
                current_segment = [block]
        if current_segment:
            draw_polygon_for_segment(ax, current_segment)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_xlim(0, total_time * 1.05)
    plt.xticks(np.arange(0, int(total_time * 1.05) + 2, 2), fontsize=10)
    legend_patches = [mpatches.Patch(color=c, label=t.replace('_', ' ').title()) for t, c in colors.items()]
    fig.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.15 / num_channels if num_channels > 1 else -0.2),
               ncol=len(colors), fontsize=12)
    plt.tight_layout(rect=[0, 0.1, 1, 0.9])
    output_filename = f'pipeline_schedule_gpu_{gpu_rank + 1}_gpipe_ideal_viz.png'
    plt.savefig(output_filename, dpi=150)
    print(f"\n为 GPU {gpu_rank + 1} 生成的理想场景 GPipe 可视化图表已保存为: {output_filename}")
    if show_plot:
        plt.show()
    plt.close(fig)


def draw_polygon_for_segment(ax, segment):
    """绘制相同任务在多个相邻时间片上的堆叠多边形，并标注 micro‑batch 编号。"""
    if not segment:
        return
    task_id = segment[0]['id']
    # 使用斜纹表示前向激活传递
    hatch = '\\' if segment[0]['type'] == 'fwd_prop' else None
    top_verts, bottom_verts = [], []
    for block in segment:
        if not top_verts or top_verts[-1] != (block['start'], block['y_pos'] + block['height']):
            top_verts.append((block['start'], block['y_pos'] + block['height']))
        top_verts.append((block['end'], block['y_pos'] + block['height']))
        if not bottom_verts or bottom_verts[-1] != (block['start'], block['y_pos']):
            bottom_verts.append((block['start'], block['y_pos']))
        bottom_verts.append((block['end'], block['y_pos']))
    all_verts = top_verts + bottom_verts[::-1]
    polygon = mpatches.Polygon(all_verts, closed=True, facecolor=segment[0]['color'], edgecolor='black',
                               alpha=0.9, linewidth=0.5, hatch=hatch, zorder=1)
    ax.add_patch(polygon)
    # 计算文本位置
    min_start = segment[0]['start']
    max_end = segment[-1]['end']
    total_area = sum((b['end'] - b['start']) * b['height'] for b in segment)
    if total_area > 1e-9:
        avg_y_center = sum((b['end'] - b['start']) * b['height'] * (b['y_pos'] + b['height'] / 2) for b in segment) / total_area
    else:
        avg_y_center = segment[0]['y_pos'] + segment[0]['height'] / 2
    ax.text(min_start + (max_end - min_start) / 2, avg_y_center, segment[0]['label'], ha='center', va='center',
            color='black', fontsize=9, weight='bold', zorder=2)


def plot_total_pipeline_chart(all_blocks, total_time, num_gpus, show_completion_time, show_plot):
    """绘制所有 GPU 的整体时间轴，用于观察理想场景下的流水线全貌。"""
    fig, axes = plt.subplots(num_gpus, 1, figsize=(22, 3.5 * num_gpus), sharex=True, gridspec_kw={'hspace': 0})
    fig.suptitle("Total GPipe F-then-B Pipeline Schedule (Ideal)", fontsize=20, y=0.95)
    for gpu_rank in range(num_gpus):
        ax = axes[gpu_rank]
        blocks_for_gpu = [b for b in all_blocks if b['id'][2] == gpu_rank]
        # 虚拟通道顺序与单个 GPU 绘图一致：从下到上依次为前向传递、计算、梯度同步、梯度传递。
        VISUAL_FWD = 0
        VISUAL_COMP = 1
        VISUAL_SYNC = 2
        VISUAL_GRAD = 3
        y_map = {
            VISUAL_FWD: 0,    # 前向传递在最下方
            VISUAL_COMP: 1,   # 计算通道
            VISUAL_SYNC: 2,   # 梯度同步
            VISUAL_GRAD: 3    # 梯度传递在最上方
        }
        visual_layout_labels = ["Fwd Prop", "Compute", "Grad Sync", "Grad Prop"]
        num_channels = 4
        ax.set_ylabel(f"GPU {gpu_rank + 1}", fontsize=14, rotation=0, labelpad=40, va='center')
        ax.set_ylim(0, num_channels)
        ax.set_yticks([i + 0.5 for i in range(num_channels)])
        ax.set_yticklabels(visual_layout_labels, fontsize=10)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        if not blocks_for_gpu:
            continue
        drawable_blocks = []
        time_slices = sorted(list(set([b['start'] for b in blocks_for_gpu] + [b['end'] for b in blocks_for_gpu])))
        for i in range(len(time_slices) - 1):
            start, end = time_slices[i], time_slices[i + 1]
            blocks_in_slice = [b for b in blocks_for_gpu if b['start'] <= start and b['end'] >= end]
            blocks_by_virtual_channel = defaultdict(list)
            for b in blocks_in_slice:
                if b['type'] == 'grad_prop':
                    ch = VISUAL_GRAD
                elif b['type'] == 'grad_sync':
                    ch = VISUAL_SYNC
                elif b['type'] in ['forward', 'backward']:
                    ch = VISUAL_COMP
                elif b['type'] == 'fwd_prop':
                    ch = VISUAL_FWD
                else:
                    ch = VISUAL_COMP
                blocks_by_virtual_channel[ch].append(b)
            for ch, blocks_on_ch in blocks_by_virtual_channel.items():
                y_base = y_map[ch]
                y_stack_offset = 0
                for block in sorted(blocks_on_ch, key=lambda b: b['type']):
                    new_block = block.copy()
                    new_block['start'], new_block['end'] = start, end
                    new_block['y_pos'] = y_base + y_stack_offset
                    new_block['height'] = block['width']
                    drawable_blocks.append(new_block)
                    y_stack_offset += block['width']
        key_func = lambda b: b['id']
        drawable_blocks.sort(key=key_func)
        for task_id, group_iter in groupby(drawable_blocks, key=key_func):
            group = sorted(list(group_iter), key=lambda b: b['start'])
            if not group:
                continue
            current_segment = []
            for block in group:
                if not current_segment or abs(block['start'] - current_segment[-1]['end']) < 1e-9:
                    current_segment.append(block)
                else:
                    draw_polygon_for_segment(ax, current_segment)
                    current_segment = [block]
            if current_segment:
                draw_polygon_for_segment(ax, current_segment)
    axes[-1].set_xlabel("Time", fontsize=14)
    plt.xlim(0, total_time * 1.05)
    plt.xticks(np.arange(0, int(total_time * 1.05) + 2, 2), fontsize=10)
    if show_completion_time:
        ax0 = axes[0]
        ax0.axvline(x=total_time, color='r', linestyle='--', linewidth=2, zorder=5)
        ax0.annotate(f'{total_time:.2f}',
                     xy=(total_time, ax0.get_ylim()[1]),
                     xytext=(total_time, ax0.get_ylim()[1] + 0.5),
                     arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=8),
                     ha='center', va='bottom', fontsize=12, color='red', zorder=6)
    legend_patches = [mpatches.Patch(color=c, label=t.replace('_', ' ').title()) for t, c in colors.items()]
    fig.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=len(colors), fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    output_filename = 'pipeline_schedule_total_gpipe_ideal_viz.png'
    plt.savefig(output_filename, dpi=150)
    print(f"\n总览图已保存为: {output_filename}")
    if show_plot:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate a full 4-GPU GPipe (F-then-B) pipeline in an ideal scenario.')
    parser.add_argument('-n', '--num_groups', type=int, default=8, help='Number of micro-batches.')
    parser.add_argument('--sync-freq', type=int, default=2, help='Frequency of grad_sync tasks.')
    parser.add_argument('--no-show', action='store_true', help='Save chart without displaying it.')
    parser.add_argument('--show-completion-time', action='store_true', help='Show the completion time annotation on the total chart.')
    args = parser.parse_args()
    lengths = default_lengths.copy()
    # 若 fwd_prop 长度未定义，则默认与 grad_prop 相同
    if 'fwd_prop' not in lengths:
        lengths['fwd_prop'] = lengths['grad_prop']
    # 调度
    all_blocks, total_time = calculate_full_pipeline_schedule(
        n=args.num_groups,
        num_gpus=4,
        lengths=lengths,
        priorities=default_priorities,
        exclusive_tiers=default_exclusive_tiers,
        sync_freq=args.sync_freq
    )
    # 绘制各 GPU 图表
    for gpu_rank in range(4):
        plot_single_gpu_chart(gpu_rank, all_blocks, total_time, show_plot=False)
    # 绘制总览图
    plot_total_pipeline_chart(all_blocks, total_time, 4, args.show_completion_time, show_plot=(not args.no_show))