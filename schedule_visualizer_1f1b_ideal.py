"""
文件名: schedule_visualizer_1f1b_ideal.py

描述:
    本脚本在理想场景下模拟传统 1F1B 流水线并行训练的调度过程。
    与仓库中的 1F1B 调度模拟相比，本脚本为不同通信行为分别提供独立的
    通道，并取消梯度同步对计算的影响，从而模拟出资源无争用的理想情况。

    理想场景配置如下：
      * 每个 GPU 拥有 4 条通道：梯度传递 (grad_prop) 通道位于最上方，梯度
        同步 (grad_sync) 通道次之，计算通道再次之，前向激活传递 (fwd_prop)
        通道在最下方。各通道之间相互独立，互不影响。
      * 采用经典 1F1B 调度：在 pipeline 填满后，正向和反向交替执行。反向
        任务无需等待全部前向完成，而是在依赖满足后即可启动。
      * 取消 ``fwd_impact`` 和 ``bwd_impact`` 机制，计算任务速度不受梯度同步
        或其他通信的干扰。

    输出包括每个 GPU 的时间轴图以及整条流水线的总览图。

用法:
    python schedule_visualizer_1f1b_ideal.py [-n NUM_GROUPS] [--sync-freq FREQ]
                                             [--no-show] [--show-completion-time]

    ``-n`` 指定 micro‑batch 的数量，默认 8；``--sync-freq`` 控制梯度同步的频率，
    默认每 2 个 micro‑batch 同步一次；``--no-show`` 表示仅保存图表不展示。
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
from fractions import Fraction
from copy import deepcopy
from collections import defaultdict
import numpy as np
from itertools import groupby

W_UNIT = 1.0

colors = {
    'forward':   '#a7c7e7',
    'backward':  '#c1e1c1',
    'fwd_prop':  '#87ceeb',
    'grad_prop': '#d1b3e2',
    'grad_sync': '#fffacd',
    'param_sync': '#ffb3ba',
    'bubble':    '#f0f0f0'
}

default_lengths = {
    'forward': 1,
    'backward': 2,
    'fwd_prop': 1,
    'grad_prop': 1,
    'grad_sync': 2,
    'param_sync': 1
}

default_priorities = {
    'forward': 1,
    'backward': 1,
    'fwd_prop': 1,
    'grad_prop': 1,
    'grad_sync': 1,
    'param_sync': 1
}

default_exclusive_tiers = {
    'fwd_prop': None,
    'grad_prop': None,
    'grad_sync': None,
}


def calculate_full_pipeline_schedule(n, num_gpus, lengths, priorities, exclusive_tiers, sync_freq=1):
    """模拟理想场景下 1F1B 调度的完整流水线。"""
    all_tasks = []
    for j in range(num_gpus):
        base_ch = j * 4
        compute_ch = base_ch + 0
        fwd_ch = base_ch + 1
        grad_ch = base_ch + 2
        sync_ch = base_ch + 3
        for i in range(n):
            all_tasks.append({
                'id': ('forward', i, j),
                'type': 'forward',
                'work_needed': lengths['forward'],
                'dependencies': [],
                'status': 'pending',
                'channel': compute_ch,
                'label': f'{i + 1}'
            })
            all_tasks.append({
                'id': ('backward', i, j),
                'type': 'backward',
                'work_needed': lengths['backward'],
                'dependencies': [],
                'status': 'pending',
                'channel': compute_ch,
                'label': f'{i + 1}'
            })
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
            # 梯度同步
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
    # 建立依赖关系（1F1B，不添加全局屏障）
    for task in all_tasks:
        if len(task['id']) < 3:
            continue
        task_type, i, j = task['id']
        if task_type == 'forward':
            # 阶段 0: 前向串行；其它阶段需等待来自上一阶段的激活传递
            if j == 0:
                if i > 0:
                    task['dependencies'].append(('forward', i - 1, j))
            else:
                task['dependencies'].append(('fwd_prop', i, j - 1))
            # 1F1B 模式下，前向任务不依赖于本阶段的反向任务；前向与反向交替逻辑在
            # 计算调度器中实现，无需在依赖关系中体现。
        elif task_type == 'backward':
            # 反向必须等待对应前向完成和来自下一个阶段的梯度传递完成。
            task['dependencies'].append(('forward', i, j))
            if j < num_gpus - 1:
                task['dependencies'].append(('grad_prop', i, j + 1))
        elif task_type == 'fwd_prop':
            task['dependencies'].append(('forward', i, j))
        elif task_type == 'grad_prop':
            task['dependencies'].append(('backward', i, j))
        elif task_type == 'grad_sync':
            task['dependencies'].append(('backward', i, j))
            # 链式依赖上一同步
            prev_sync_idx = i - sync_freq
            if prev_sync_idx >= 0:
                prev_sync_id = ('grad_sync', prev_sync_idx, j)
                if any(t['id'] == prev_sync_id for t in all_tasks):
                    task['dependencies'].append(prev_sync_id)
    current_time = 0.0
    blocks = []
    finished_task_ids = set()
    while len(finished_task_ids) < len(all_tasks):
        for task in all_tasks:
            if task['status'] == 'pending' and all(dep in finished_task_ids for dep in task['dependencies']):
                task['status'] = 'ready'
                # 记录任务变为 ready 的时间，用于非抢占式调度
                # 如果任务之前已经标记 ready，则不覆盖，确保保留最早 ready 时间
                if 'ready_time' not in task:
                    task['ready_time'] = current_time
        tasks_to_set_running = []
        shares_now = {}
        active_tasks = [t for t in all_tasks if t['status'] in ['ready', 'running']]
        tasks_by_channel = defaultdict(list)
        for task in active_tasks:
            tasks_by_channel[task['channel']].append(task)
        for ch, tasks_in_channel in tasks_by_channel.items():
            is_compute_channel = (ch % 4 == 0)
            eligible_tasks = [t for t in tasks_in_channel if t['status'] in ['ready', 'running']]
            if not eligible_tasks:
                continue
            if is_compute_channel:
                # 计算通道：每个 GPU (stage) 在任意时间只能执行一个计算任务。
                # 如果该通道已有任务在运行，则继续运行该任务，不做切换。
                if any(t['status'] == 'running' for t in eligible_tasks):
                    tasks_for_allocation = [t for t in eligible_tasks if t['status'] == 'running']
                else:
                    stage_idx = ch // 4
                    # 根据 stage 采用统一的 1F1B 调度优先级策略：
                    # 根据阶段选择调度策略
                    if stage_idx == 0:
                        # 顶层 GPU：先执行前 num_gpus 个前向，后续前向在对应反向完成后才能执行
                        g = num_gpus
                        allowed_forward = []
                        for t in eligible_tasks:
                            if t['type'] == 'forward':
                                mb = t['id'][1]
                                if mb < g:
                                    allowed_forward.append(t)
                                else:
                                    prev_b_id = ('backward', mb - g, stage_idx)
                                    if prev_b_id in finished_task_ids:
                                        allowed_forward.append(t)
                        allowed_backward = [t for t in eligible_tasks if t['type'] == 'backward']
                        if allowed_forward:
                            allowed_forward.sort(key=lambda t: t['id'][1])
                            tasks_for_allocation = [allowed_forward[0]]
                        elif allowed_backward:
                            allowed_backward.sort(key=lambda t: t['id'][1])
                            tasks_for_allocation = [allowed_backward[0]]
                        else:
                            tasks_for_allocation = []
                    elif stage_idx == num_gpus - 1:
                        # 底层 GPU：每完成一个前向立即执行对应反向
                        g = 1
                        allowed_forward = []
                        for t in eligible_tasks:
                            if t['type'] == 'forward':
                                mb = t['id'][1]
                                if mb < g:
                                    allowed_forward.append(t)
                                else:
                                    prev_b_id = ('backward', mb - g, stage_idx)
                                    if prev_b_id in finished_task_ids:
                                        allowed_forward.append(t)
                        allowed_backward = [t for t in eligible_tasks if t['type'] == 'backward']
                        if allowed_forward:
                            allowed_forward.sort(key=lambda t: t['id'][1])
                            tasks_for_allocation = [allowed_forward[0]]
                        elif allowed_backward:
                            allowed_backward.sort(key=lambda t: t['id'][1])
                            tasks_for_allocation = [allowed_backward[0]]
                        else:
                            tasks_for_allocation = []
                    else:
                        # 中间 GPU：根据到来的通信自然启动计算，前向与反向不施加额外 gating
                        # 使用 ready_time 排序，使任务按照最早满足依赖的顺序执行
                        compute_ready = [t for t in eligible_tasks if t['type'] in ['forward', 'backward']]
                        if compute_ready:
                            # 以任务 ready 的时间作为主要排序键，若无该字段则认为当前时间
                            def sort_key(task):
                                return (task.get('ready_time', current_time), task['id'][1])
                            compute_ready.sort(key=sort_key)
                            tasks_for_allocation = [compute_ready[0]]
                        else:
                            tasks_for_allocation = []
                # 将选中的任务标记为分配带宽
                for task in tasks_for_allocation:
                    tasks_to_set_running.append(task)
                    shares_now[task['id']] = W_UNIT
            else:
                tasks_for_allocation = eligible_tasks
                tasks_by_type = defaultdict(list)
                for task in tasks_for_allocation:
                    tasks_by_type[task['type']].append(task)
                type_priorities = {t: priorities.get(t, 0) for t in tasks_by_type.keys()}
                total_p = sum(type_priorities.values())
                if total_p > 1e-9:
                    for task_type, tasks_same_type in tasks_by_type.items():
                        share_for_type = W_UNIT * (type_priorities[task_type] / total_p)
                        share_per_task = share_for_type / len(tasks_same_type)
                        for task in tasks_same_type:
                            shares_now[task['id']] = share_per_task
                else:
                    share_per_task = W_UNIT / len(tasks_for_allocation)
                    for task in tasks_for_allocation:
                        shares_now[task['id']] = share_per_task
                tasks_to_set_running.extend(tasks_for_allocation)
        unique_tasks_to_run = {t['id']: t for t in tasks_to_set_running}.values()
        for task in unique_tasks_to_run:
            if task['status'] == 'ready':
                task['status'] = 'running'
        tasks_with_share = [t for t in active_tasks if shares_now.get(t['id'], 0) > 1e-9]
        if not tasks_with_share:
            if len(finished_task_ids) == len(all_tasks):
                break
            else:
                continue
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
        for task in all_tasks:
            if task['status'] == 'running' and task.get('work_done', 0) >= task['work_needed'] - 1e-9:
                task['status'] = 'finished'
                finished_task_ids.add(task['id'])
    return blocks, current_time


def plot_single_gpu_chart(gpu_rank, all_blocks, total_time, show_plot):
    blocks_for_gpu = [b for b in all_blocks if b['id'][2] == gpu_rank]
    if not blocks_for_gpu:
        print(f"GPU {gpu_rank + 1} 没有任务，跳过绘图。")
        return
    # 通道顺序：从下到上依次为前向传递（Fwd Prop）、计算（Compute）、梯度同步（Grad Sync）和梯度传递（Grad Prop）。
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
    # 设置 y 轴标签列表，索引对应于下->上
    visual_layout_labels = ["Fwd Prop", "Compute", "Grad Sync", "Grad Prop"]
    num_channels = 4
    fig, ax = plt.subplots(1, 1, figsize=(22, 2.5 * num_channels))
    fig.suptitle(f"GPU {gpu_rank + 1} - 1F1B Schedule (Ideal)", fontsize=16, y=0.98)
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
    output_filename = f'pipeline_schedule_gpu_{gpu_rank + 1}_1f1b_ideal_viz.png'
    plt.savefig(output_filename, dpi=150)
    print(f"\n为 GPU {gpu_rank + 1} 生成的理想场景 1F1B 可视化图表已保存为: {output_filename}")
    if show_plot:
        plt.show()
    plt.close(fig)


def draw_polygon_for_segment(ax, segment):
    if not segment:
        return
    task_id = segment[0]['id']
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
    fig, axes = plt.subplots(num_gpus, 1, figsize=(22, 3.5 * num_gpus), sharex=True, gridspec_kw={'hspace': 0})
    fig.suptitle("Total 1F1B Pipeline Schedule (Ideal)", fontsize=20, y=0.95)
    for gpu_rank in range(num_gpus):
        ax = axes[gpu_rank]
        blocks_for_gpu = [b for b in all_blocks if b['id'][2] == gpu_rank]
        # 调整通道顺序，与单 GPU 图保持一致：从下到上依次为前向传递、计算、梯度同步、梯度传递。
        VISUAL_FWD = 0
        VISUAL_COMP = 1
        VISUAL_SYNC = 2
        VISUAL_GRAD = 3
        y_map = {
            VISUAL_FWD: 0,    # 前向传递位于最下方
            VISUAL_COMP: 1,   # 计算通道
            VISUAL_SYNC: 2,   # 梯度同步
            VISUAL_GRAD: 3    # 梯度传递位于最上方
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
    output_filename = 'pipeline_schedule_total_1f1b_ideal_viz.png'
    plt.savefig(output_filename, dpi=150)
    print(f"\n总览图已保存为: {output_filename}")
    if show_plot:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate a full 4-GPU 1F1B pipeline in an ideal scenario.')
    parser.add_argument('-n', '--num_groups', type=int, default=8, help='Number of micro-batches.')
    parser.add_argument('--sync-freq', type=int, default=2, help='Frequency of grad_sync tasks.')
    parser.add_argument('--no-show', action='store_true', help='Save chart without displaying it.')
    parser.add_argument('--show-completion-time', action='store_true', help='Show the completion time annotation on the total chart.')
    args = parser.parse_args()
    lengths = default_lengths.copy()
    if 'fwd_prop' not in lengths:
        lengths['fwd_prop'] = lengths['grad_prop']
    all_blocks, total_time = calculate_full_pipeline_schedule(
        n=args.num_groups,
        num_gpus=4,
        lengths=lengths,
        priorities=default_priorities,
        exclusive_tiers=default_exclusive_tiers,
        sync_freq=args.sync_freq
    )
    for gpu_rank in range(4):
        plot_single_gpu_chart(gpu_rank, all_blocks, total_time, show_plot=False)
    plot_total_pipeline_chart(all_blocks, total_time, 4, args.show_completion_time, show_plot=(not args.no_show))