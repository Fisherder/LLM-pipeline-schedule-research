"""
文件名: schedule_visualizer_gpipe_single_channel.py

描述:
    该脚本模拟基于 GPipe 的流水线并行训练调度（前向阶段之后执行反向阶段），
    并使用单一通信通道进行可视化。与仓库中的 ``schedule_visualizer_single_channel.py``
    不同之处在于，这里的计算顺序遵循先完成所有 micro‑batch 的前向计算，然后再执行
    所有反向计算（F‑then‑B），而非正反交替的 1F1B 调度。

    计算和绘图逻辑保持与原脚本一致：每个 GPU 拥有一个计算通道和一个通信通道，
    所有通信任务（前向激活传递、反向梯度传递及梯度同步）均在该单一通信通道上竞争
    资源，从而更真实地反映物理链路争用情况。支持不同任务类型之间的带宽共享权重、
    主动独占优先级、接收端拥塞控制等。

用法:
    python schedule_visualizer_gpipe_single_channel.py [-n NUM_GROUPS] [--fwd-impact F] [--bwd-impact B]
                                                      [--sync-solo-w W] [--sync-freq FREQ]
                                                      [--no-show] [--show-throttling] [--show-completion-time]

    与 ``schedule_visualizer_single_channel.py`` 类似，``-n`` 指定 micro‑batch 的数量，
    默认 8；``--sync-freq`` 控制多少个 micro‑batch 聚合一次梯度同步；``--fwd-impact``
    和 ``--bwd-impact`` 用于模拟梯度同步占用带宽对前向/反向计算的影响；其它参数
    用于控制绘图显示。
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
W_UNIT = 1.0  # 单位带宽，可分配给同一通道上的任务

# 颜色方案，不同任务使用不同颜色显示
colors = {
    'forward':   '#a7c7e7',  # 前向计算
    'backward':  '#c1e1c1',  # 反向计算
    'fwd_prop':  '#87ceeb',  # 激活传递
    'grad_prop': '#d1b3e2',  # 梯度传递
    'grad_sync': '#fffacd',  # 梯度同步
    'param_sync': '#ffb3ba', # 参数同步（本脚本未使用）
    'bubble':    '#f0f0f0'   # 气泡或闲置时间（绘图用）
}

# 默认任务长度，每个任务需要多少单位“工作量”。可以通过命令行覆盖部分值。
default_lengths = {
    'forward': 1,
    'backward': 2,
    'fwd_prop': 1,
    'grad_prop': 1,
    'grad_sync': 2,
    'param_sync': 1
}

# 定义通信任务的共享权重，用于带宽分配。数值越大表示越高优先级。
default_priorities = {
    'forward': 1,
    'backward': 1,
    'fwd_prop': 0.5,
    'grad_prop': 0.5,
    'grad_sync': 0.5,
    'param_sync': 1
}

# 主动独占任务的优先级等级；字母序越靠前优先级越高。None 表示普通共享任务。
default_exclusive_tiers = {
    'fwd_prop': None,
    'grad_prop': None,
    'grad_sync': None,
}


def calculate_full_pipeline_schedule(n, num_gpus, is_ideal, lengths, priorities,
                                     exclusive_tiers, fwd_impact, bwd_impact,
                                     sync_solo_w=1.0, sync_freq=2):
    """模拟 GPipe F‑then‑B 流水线的完整调度。

    参数说明:
        n (int): micro‑batch 数量。
        num_gpus (int): 阶段（GPU）数量。
        is_ideal (bool): 若为 True，则梯度同步使用理想独立通道；否则与其他通信共享带宽。
        lengths (dict): 各类型任务的工作量。
        priorities (dict): 各类型通信任务的带宽权重。
        exclusive_tiers (dict): 主动独占任务的等级定义。
        fwd_impact (float): 梯度同步对前向计算效率的影响系数。
        bwd_impact (float): 梯度同步对反向计算效率的影响系数。
        sync_solo_w (float): 当某个 GPU 独占执行梯度同步时的带宽占比。
        sync_freq (int): 多久（多少个 micro‑batch）进行一次梯度同步。

    返回:
        all_blocks (list): 调度过程中记录的时间段块，用于绘图。
        total_time (float): 总训练时间（模拟时钟）。
        throttled_task_ids (set): 因接收端拥塞被调节过带宽的通信任务集合。
    """
    all_tasks = []
    # 1. 创建所有任务
    for j in range(num_gpus):
        compute_ch = j * 2 + 0
        comm_ch = j * 2 + 1
        for i in range(n):
            # 前向和反向计算任务
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
            # 前向激活传递
            if j < num_gpus - 1:
                all_tasks.append({
                    'id': ('fwd_prop', i, j),
                    'type': 'fwd_prop',
                    'work_needed': lengths['fwd_prop'],
                    'dependencies': [],
                    'status': 'pending',
                    'channel': comm_ch,
                    'label': f'{i + 1}'
                })
            # 反向梯度传递
            if j > 0:
                all_tasks.append({
                    'id': ('grad_prop', i, j),
                    'type': 'grad_prop',
                    'work_needed': lengths['grad_prop'],
                    'dependencies': [],
                    'status': 'pending',
                    'channel': comm_ch,
                    'label': f'{i + 1}'
                })
            # 梯度同步任务
            is_sync_point = (i + 1) % sync_freq == 0
            is_last_batch = (i == n - 1)
            is_leftover_sync = is_last_batch and ((i + 1) % sync_freq != 0)
            if is_sync_point or is_leftover_sync:
                num_grads = (i + 1) % sync_freq if is_leftover_sync else sync_freq
                sync_work = lengths['grad_sync'] * num_grads
                sync_channel = (comm_ch) + num_gpus * 2 if is_ideal else comm_ch
                all_tasks.append({
                    'id': ('grad_sync', i, j),
                    'type': 'grad_sync',
                    'work_needed': sync_work,
                    'dependencies': [],
                    'status': 'pending',
                    'channel': sync_channel,
                    'label': f'{(i // sync_freq) + 1}'
                })

    # 2. 建立依赖关系
    # 全局屏障：最后一次前向通过最后一个阶段时刻
    barrier_id = ('forward', n - 1, num_gpus - 1)
    for task in all_tasks:
        if len(task['id']) < 3:
            continue
        task_type, i, j = task['id']
        if task_type == 'forward':
            # 阶段 0：前向按 micro‑batch 顺序串行启动
            if j == 0:
                if i > 0:
                    task['dependencies'].append(('forward', i - 1, j))
            else:
                # 其他阶段的前向依赖上一阶段的激活传递
                task['dependencies'].append(('fwd_prop', i, j - 1))
        elif task_type == 'backward':
            # 反向必须等待对应的前向完成
            task['dependencies'].append(('forward', i, j))
            # 如果不是最末阶段，还需等待来自下一阶段的梯度
            if j < num_gpus - 1:
                task['dependencies'].append(('grad_prop', i, j + 1))
            # GPipe 模式：所有反向在完成全部前向后才能启动
            task['dependencies'].append(barrier_id)
        elif task_type == 'fwd_prop':
            # 激活传递依赖本阶段前向
            task['dependencies'].append(('forward', i, j))
        elif task_type == 'grad_prop':
            # 梯度传递依赖本阶段反向
            task['dependencies'].append(('backward', i, j))
        elif task_type == 'grad_sync':
            # 梯度同步依赖本阶段反向
            task['dependencies'].append(('backward', i, j))
            # 同一 GPU 上前一次同步任务的链式依赖
            prev_sync_idx = i - sync_freq
            if prev_sync_idx >= 0:
                prev_sync_id = ('grad_sync', prev_sync_idx, j)
                if any(t['id'] == prev_sync_id for t in all_tasks):
                    task['dependencies'].append(prev_sync_id)

    # 3. 事件循环模拟
    current_time = 0.0
    blocks = []  # 记录各任务在时间轴上的活动区间，用于绘图
    throttled_task_ids = set()  # 记录因接收端拥塞被缩减带宽的通信任务 id
    finished_task_ids = set()

    while len(finished_task_ids) < len(all_tasks):
        # 标记已就绪的任务
        for task in all_tasks:
            if task['status'] == 'pending' and all(dep in finished_task_ids for dep in task['dependencies']):
                task['status'] = 'ready'

        tasks_to_set_running = []
        shares_now = {}

        # 收集所有 ready 或 running 状态的任务，按通道划分
        active_tasks = [t for t in all_tasks if t['status'] in ['ready', 'running']]
        tasks_by_channel = defaultdict(list)
        for task in active_tasks:
            tasks_by_channel[task['channel']].append(task)

        # 通道调度：对每个通道独立分配带宽
        for ch, tasks_in_channel in tasks_by_channel.items():
            is_compute_channel = (ch % 2 == 0)
            eligible_tasks = [t for t in tasks_in_channel if t['status'] in ['ready', 'running']]
            if not eligible_tasks:
                continue

            if is_compute_channel:
                # 计算通道: 同一时间每个 GPU 只能执行一个计算任务（forward 或 backward）。
                if any(t['status'] == 'running' for t in eligible_tasks):
                    # 若已有任务运行，则保持运行
                    tasks_for_allocation = [t for t in eligible_tasks if t['status'] == 'running']
                else:
                    # 否则优先选择前向任务，其次是反向任务；同一类任务按 micro‑batch 顺序排序
                    def sort_key(task):
                        type_prio = 0 if task['type'] == 'forward' else 1
                        return (type_prio, task['id'][1])
                    eligible_tasks.sort(key=sort_key)
                    tasks_for_allocation = [eligible_tasks[0]]
                tasks_to_set_running.extend(tasks_for_allocation)
                for task in tasks_for_allocation:
                    shares_now[task['id']] = W_UNIT
            else:
                # 通信通道: 可同时发送多个任务，根据优先级和独占规则分配带宽
                exclusive_tasks = []
                sharing_tasks = []
                zero_prio_tasks = []
                for t in eligible_tasks:
                    if exclusive_tiers.get(t['type']) is not None:
                        exclusive_tasks.append(t)
                    elif priorities.get(t['type'], 0) > 1e-9:
                        sharing_tasks.append(t)
                    else:
                        zero_prio_tasks.append(t)
                tasks_for_allocation = []
                if exclusive_tasks:
                    # 独占任务：选择等级最高者独占带宽
                    exclusive_tasks.sort(key=lambda t: exclusive_tiers[t['type']])
                    highest_tier_task = exclusive_tasks[0]
                    tasks_for_allocation = [highest_tier_task]
                    shares_now[highest_tier_task['id']] = W_UNIT
                elif sharing_tasks:
                    # 共享任务：按照权重分配
                    tasks_for_allocation = sharing_tasks
                    tasks_by_type = defaultdict(list)
                    for task in tasks_for_allocation:
                        tasks_by_type[task['type']].append(task)
                    type_priorities = {t: priorities.get(t, 0) for t in tasks_by_type.keys()}
                    total_p = sum(type_priorities.values())
                    if total_p > 1e-9:
                        for task_type, tasks in tasks_by_type.items():
                            share_for_type = W_UNIT * (type_priorities[task_type] / total_p)
                            share_per_task = share_for_type / len(tasks)
                            for task in tasks:
                                shares_now[task['id']] = share_per_task
                elif zero_prio_tasks:
                    # 零优先级任务：平分剩余带宽
                    tasks_for_allocation = zero_prio_tasks
                    share_per_task = W_UNIT / len(tasks_for_allocation)
                    for task in tasks_for_allocation:
                        shares_now[task['id']] = share_per_task
                if tasks_for_allocation:
                    tasks_to_set_running.extend(tasks_for_allocation)

        # --- 拥塞控制: 接收端带宽限制 ---
        # 类似原脚本，实现接收端瓶颈，先固定接收端允许的带宽，后重新分配
        for _ in range(num_gpus * 2):
            # 针对每个接收 GPU，限制来自上游和下游的流入总带宽不超过 W_UNIT
            for receiver_j in range(1, num_gpus - 1):
                incoming_fwd_tasks = [t for t in active_tasks if t['type'] == 'fwd_prop' and t['id'][2] == receiver_j - 1]
                incoming_grad_tasks = [t for t in active_tasks if t['type'] == 'grad_prop' and t['id'][2] == receiver_j + 1]
                total_incoming_w = sum(shares_now.get(t['id'], 0) for t in incoming_fwd_tasks + incoming_grad_tasks)
                if total_incoming_w > W_UNIT:
                    scale = W_UNIT / total_incoming_w
                    for t in incoming_fwd_tasks + incoming_grad_tasks:
                        if t['id'] in shares_now:
                            shares_now[t['id']] *= scale
                            t['bw_fixed'] = True
                            throttled_task_ids.add(t['id'])
            # 对于每个发送者，若某些任务的带宽被固定，重新分配剩余带宽
            for sender_j in range(num_gpus):
                comm_ch = sender_j * 2 + 1
                sending_tasks = [t for t in active_tasks if t['channel'] == comm_ch and t['id'][2] == sender_j]
                if not sending_tasks:
                    continue
                fixed_w = sum(shares_now.get(t['id'], 0) for t in sending_tasks if t.get('bw_fixed'))
                unallocated_w = W_UNIT - fixed_w
                unfixed_tasks = [t for t in sending_tasks if not t.get('bw_fixed')]
                if not unfixed_tasks or unallocated_w < 1e-9:
                    continue
                exclusive_unfixed = []
                sharing_unfixed = []
                zero_prio_unfixed = []
                for t in unfixed_tasks:
                    if exclusive_tiers.get(t['type']) is not None:
                        exclusive_unfixed.append(t)
                    elif priorities.get(t['type'], 0) > 1e-9:
                        sharing_unfixed.append(t)
                    else:
                        zero_prio_unfixed.append(t)
                if exclusive_unfixed:
                    exclusive_unfixed.sort(key=lambda t: exclusive_tiers[t['type']])
                    winner = exclusive_unfixed[0]
                    for t in unfixed_tasks:
                        shares_now[t['id']] = unallocated_w if t == winner else 0
                elif sharing_unfixed:
                    tasks_by_type = defaultdict(list)
                    for task in sharing_unfixed:
                        tasks_by_type[task['type']].append(task)
                    type_priorities = {t: priorities.get(t, 0) for t in tasks_by_type.keys()}
                    total_p = sum(type_priorities.values())
                    if total_p > 0:
                        for task_type, tasks in tasks_by_type.items():
                            share_for_type = unallocated_w * (type_priorities[task_type] / total_p)
                            share_per_task = share_for_type / len(tasks)
                            for task in tasks:
                                shares_now[task['id']] = share_per_task
                elif zero_prio_unfixed:
                    share_per_task = unallocated_w / len(zero_prio_unfixed)
                    for task in zero_prio_unfixed:
                        shares_now[task['id']] = share_per_task
        # 清除临时标记
        for task in all_tasks:
            if 'bw_fixed' in task:
                del task['bw_fixed']

        # 更新 ready 状态的任务至 running
        unique_tasks_to_run = {t['id']: t for t in tasks_to_set_running}.values()
        for task in unique_tasks_to_run:
            if task['status'] == 'ready':
                task['status'] = 'running'

        # 计算本时间片内所有运行任务的最小完成时间
        tasks_with_share = [t for t in active_tasks if shares_now.get(t['id'], 0) > 1e-9]
        if not tasks_with_share:
            if len(finished_task_ids) == len(all_tasks):
                break
            else:
                continue
        min_time_to_finish = float('inf')
        for task in tasks_with_share:
            share = shares_now.get(task['id'], 0)
            if not is_ideal:
                gpu_idx = task['id'][2]
                if task['type'] == 'forward':
                    share *= (1.0 - fwd_impact * sum(shares_now.get(t['id'], 0) for t in tasks_with_share if t['id'][2] == gpu_idx and t['type'] == 'grad_sync'))
                elif task['type'] == 'backward':
                    share *= (1.0 - bwd_impact * sum(shares_now.get(t['id'], 0) for t in tasks_with_share if t['id'][2] == gpu_idx and t['type'] == 'grad_sync'))
            if share > 1e-9:
                remaining = task.get('work_needed', 0) - task.get('work_done', 0)
                time_needed = remaining / share
                if time_needed < min_time_to_finish:
                    min_time_to_finish = time_needed
        slice_duration = min_time_to_finish
        if slice_duration == float('inf') or slice_duration <= 1e-9:
            slice_duration = 1e-9
        next_event_time = current_time + slice_duration
        # 更新每个运行任务完成的工作量并记录时间片
        for task in tasks_with_share:
            if 'work_done' not in task:
                task['work_done'] = 0
            final_share = shares_now.get(task['id'], 0)
            if not is_ideal:
                gpu_idx = task['id'][2]
                if task['type'] == 'forward':
                    final_share *= (1.0 - fwd_impact * sum(shares_now.get(t['id'], 0) for t in tasks_with_share if t['id'][2] == gpu_idx and t['type'] == 'grad_sync'))
                elif task['type'] == 'backward':
                    final_share *= (1.0 - bwd_impact * sum(shares_now.get(t['id'], 0) for t in tasks_with_share if t['id'][2] == gpu_idx and t['type'] == 'grad_sync'))
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
        # 将完成任务标记为 finished
        for task in all_tasks:
            if task['status'] == 'running' and task.get('work_done', 0) >= task['work_needed'] - 1e-9:
                task['status'] = 'finished'
                finished_task_ids.add(task['id'])
    return blocks, current_time, throttled_task_ids


# --- 绘图函数 (单通信通道版本) ---
def plot_single_gpu_chart(gpu_rank, all_blocks, total_time, throttled_task_ids, show_throttling, show_plot):
    """为单个 GPU 绘制任务执行时间轴。"""
    blocks_for_gpu = [b for b in all_blocks if b['id'][2] == gpu_rank]
    if not blocks_for_gpu:
        print(f"GPU {gpu_rank+1} 没有任务，跳过绘图。")
        return
    num_channels = 2
    visual_layout_labels = ["Compute", "Communication"]
    fig, ax = plt.subplots(1, 1, figsize=(22, 2.5 * num_channels))
    fig.suptitle(f"GPU {gpu_rank + 1} - GPipe F-then-B Schedule (Single Comm Visualization)", fontsize=16, y=0.98)
    ax.set_ylabel("Resource Channels", fontsize=12)
    ax.set_ylim(0, num_channels)
    ax.set_yticks([i + 0.5 for i in range(num_channels)])
    ax.set_yticklabels(visual_layout_labels, fontsize=10)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    y_map = {
        gpu_rank * 2 + 0: 0,  # Compute
        gpu_rank * 2 + 1: 1   # Communication
    }
    # 切分时间轴用于堆叠并绘制带宽面积
    drawable_blocks = []
    time_slices = sorted(list(set([b['start'] for b in blocks_for_gpu] + [b['end'] for b in blocks_for_gpu])))
    for i in range(len(time_slices) - 1):
        start, end = time_slices[i], time_slices[i + 1]
        blocks_in_slice = [b for b in blocks_for_gpu if b['start'] <= start and b['end'] >= end]
        blocks_by_channel = defaultdict(list)
        for b in blocks_in_slice:
            blocks_by_channel[b['channel']].append(b)
        for ch, blocks_on_ch in blocks_by_channel.items():
            if ch not in y_map:
                continue
            y_base = y_map[ch]
            y_stack_offset = 0
            for block in sorted(blocks_on_ch, key=lambda b: b['type']):
                new_block = block.copy()
                new_block['start'], new_block['end'] = start, end
                new_block['y_pos'] = y_base + y_stack_offset
                new_block['height'] = block['width']
                drawable_blocks.append(new_block)
                y_stack_offset += block['width']
    # 按任务 id 合并连续时间段
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
                draw_polygon_for_segment(ax, current_segment, throttled_task_ids, show_throttling)
                current_segment = [block]
        if current_segment:
            draw_polygon_for_segment(ax, current_segment, throttled_task_ids, show_throttling)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_xlim(0, total_time * 1.05)
    plt.xticks(np.arange(0, int(total_time * 1.05) + 2, 2), fontsize=10)
    legend_patches = [mpatches.Patch(color=c, label=t.replace('_', ' ').title()) for t, c in colors.items()]
    fig.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.15 / num_channels if num_channels > 1 else -0.2),
               ncol=len(colors), fontsize=12)
    plt.tight_layout(rect=[0, 0.1, 1, 0.9])
    output_filename = f'pipeline_schedule_gpu_{gpu_rank + 1}_gpipe_single_channel_viz.png'
    plt.savefig(output_filename, dpi=150)
    print(f"\n为 GPU {gpu_rank + 1} 生成的单通道 GPipe 可视化图表已保存为: {output_filename}")
    if show_plot:
        plt.show()
    plt.close(fig)


def draw_polygon_for_segment(ax, segment, throttled_task_ids, show_throttling):
    """绘制同一任务在相邻时间片上的堆叠多边形并标注 micro‑batch 标签。"""
    if not segment:
        return
    task_id = segment[0]['id']
    # GPipe 模式中仍使用同样的斜纹表示前向激活传递
    hatch = '\\' if segment[0]['type'] == 'fwd_prop' else None
    is_throttled = task_id in throttled_task_ids
    edge_color = 'red' if is_throttled and show_throttling else 'black'
    line_width = 1.5 if is_throttled and show_throttling else 0.5
    top_verts, bottom_verts = [], []
    for block in segment:
        if not top_verts or top_verts[-1] != (block['start'], block['y_pos'] + block['height']):
            top_verts.append((block['start'], block['y_pos'] + block['height']))
        top_verts.append((block['end'], block['y_pos'] + block['height']))
        if not bottom_verts or bottom_verts[-1] != (block['start'], block['y_pos']):
            bottom_verts.append((block['start'], block['y_pos']))
        bottom_verts.append((block['end'], block['y_pos']))
    all_verts = top_verts + bottom_verts[::-1]
    polygon = mpatches.Polygon(all_verts, closed=True, facecolor=segment[0]['color'], edgecolor=edge_color,
                               alpha=0.9, linewidth=line_width, hatch=hatch, zorder=1)
    ax.add_patch(polygon)
    # 在多边形中央标注 micro‑batch 编号
    min_start = segment[0]['start']
    max_end = segment[-1]['end']
    total_area = sum((b['end'] - b['start']) * b['height'] for b in segment)
    if total_area > 1e-9:
        avg_y_center = sum((b['end'] - b['start']) * b['height'] * (b['y_pos'] + b['height'] / 2) for b in segment) / total_area
    else:
        avg_y_center = segment[0]['y_pos'] + segment[0]['height'] / 2
    ax.text(min_start + (max_end - min_start) / 2, avg_y_center, segment[0]['label'],
            ha='center', va='center', color='black', fontsize=9, weight='bold', zorder=2)


def plot_total_pipeline_chart(all_blocks, total_time, num_gpus, throttled_task_ids, show_throttling, show_completion_time, show_plot):
    """绘制所有 GPU 的整体时间轴，用于观察流水线全貌。"""
    fig, axes = plt.subplots(num_gpus, 1, figsize=(22, 2.5 * num_gpus), sharex=True, gridspec_kw={'hspace': 0})
    fig.suptitle("Total GPipe F-then-B Pipeline Schedule (Single Comm Visualization)", fontsize=20, y=0.95)
    for gpu_rank in range(num_gpus):
        ax = axes[gpu_rank]
        blocks_for_gpu = [b for b in all_blocks if b['id'][2] == gpu_rank]
        y_map = {
            gpu_rank * 2 + 0: 0,
            gpu_rank * 2 + 1: 1
        }
        visual_layout_labels = ["Compute", "Communication"]
        num_channels = 2
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
            blocks_by_channel = defaultdict(list)
            for b in blocks_in_slice:
                blocks_by_channel[b['channel']].append(b)
            for ch, blocks_on_ch in blocks_by_channel.items():
                if ch not in y_map:
                    continue
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
                    draw_polygon_for_segment(ax, current_segment, throttled_task_ids, show_throttling)
                    current_segment = [block]
            if current_segment:
                draw_polygon_for_segment(ax, current_segment, throttled_task_ids, show_throttling)
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
    output_filename = 'pipeline_schedule_total_gpipe_single_channel_viz.png'
    plt.savefig(output_filename, dpi=150)
    print(f"\n总览图已保存为: {output_filename}")
    if show_plot:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate a full 4-GPU GPipe (F-then-B) pipeline.')
    parser.add_argument('-n', '--num_groups', type=int, default=8, help='Number of micro-batches.')
    parser.add_argument('--fwd-impact', type=float, default=0.2, help='Impact of sync on forward compute.')
    parser.add_argument('--bwd-impact', type=float, default=0.2, help='Impact of sync on backward compute.')
    parser.add_argument('--sync-solo-w', type=float, default=1.0, help='Resource share for a solo sync task.')
    parser.add_argument('--sync-freq', type=int, default=2, help='Frequency of grad_sync tasks.')
    parser.add_argument('--no-show', action='store_true', help='Save chart without displaying it.')
    parser.add_argument('--show-throttling', action='store_true', help='Highlight communication tasks throttled by receiver congestion.')
    parser.add_argument('--show-completion-time', action='store_true', help='Show the completion time annotation on the total chart.')
    args = parser.parse_args()
    lengths = default_lengths.copy()
    # 在某些配置下 fwd_prop 长度可能缺失，默认等同于 grad_prop
    if 'fwd_prop' not in lengths:
        lengths['fwd_prop'] = lengths['grad_prop']
    # 调用核心调度函数
    all_blocks, total_time, throttled_task_ids = calculate_full_pipeline_schedule(
        n=args.num_groups,
        num_gpus=4,
        is_ideal=False,
        lengths=lengths,
        priorities=default_priorities,
        exclusive_tiers=default_exclusive_tiers,
        fwd_impact=args.fwd_impact,
        bwd_impact=args.bwd_impact,
        sync_solo_w=args.sync_solo_w,
        sync_freq=args.sync_freq
    )
    # 为每个 GPU 绘制单独的时间轴
    for gpu_rank in range(4):
        plot_single_gpu_chart(gpu_rank, all_blocks, total_time, throttled_task_ids, args.show_throttling, show_plot=False)
    # 绘制整体流水线图
    plot_total_pipeline_chart(all_blocks, total_time, 4, throttled_task_ids, args.show_throttling,
                              show_completion_time=args.show_completion_time, show_plot=(not args.no_show))
