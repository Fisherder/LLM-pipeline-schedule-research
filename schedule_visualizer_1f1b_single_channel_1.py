# 文件名: schedule_visualizer_1f1b_single_channel.py
# 描述: 这是一个为了对比可视化方案而创建的版本。
#       它的核心计算逻辑与双通道版本完全相同（包含接收端拥塞控制），
#       但绘图逻辑被修改，将所有通信任务绘制在单一的“通信”通道中，以最真实地反映物理资源竞争。
# V9  更新: 实现“主动独占”强优先级逻辑。
# V10 更新: 将颜色图例移动到“标题与主图之间”，并保留底部空白。
# V12 更新: 图例用轴坐标锚定到主图上沿（距离更小可控）；去掉 tight_layout，改用 subplots_adjust；
#           明显增大主图高度（figsize 更高）。
# V13 更新: （此前）将“总览图”中的所有文字加粗：包括总标题、各 GPU 的 Y 轴标签、X 轴标题、
#           坐标刻度、小示意图文字、图例文字、完成时间标注等。单 GPU 小图保持不变。
# V14 更新: （本次）**进一步增大坐标轴刻度字体**：统一把 x/y 刻度字号提升到 34pt，
#           并在具体绘图处通过 tick_params 强制生效（覆盖默认值）。

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator

# === Enlarged font settings (auto-added) ===
# 不回退已有字号，只在更小时才抬高最小值。
try:
    rcParams['font.size'] = max(rcParams.get('font.size', 10), 16)
    rcParams['axes.titlesize'] = max(rcParams.get('axes.titlesize', 12), 18)
    rcParams['axes.labelsize'] = max(rcParams.get('axes.labelsize', 10), 16)
    rcParams['xtick.labelsize'] = max(rcParams.get('xtick.labelsize', 10), 14)
    rcParams['ytick.labelsize'] = max(rcParams.get('ytick.labelsize', 10), 14)
    rcParams['legend.fontsize'] = max(rcParams.get('legend.fontsize', 10), 14)
except Exception:
    pass

# 全局字体（本次把刻度进一步放大为 34pt）
rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 28,
    'axes.titlesize': 28,
    'axes.labelsize': 28,
    'xtick.labelsize': 34,   # 原 28 -> 34
    'ytick.labelsize': 34,   # 原 28 -> 34
    'legend.fontsize': 28,
})

import argparse
from fractions import Fraction
from copy import deepcopy
from collections import defaultdict
import numpy as np
from itertools import groupby
import os

# --- 全局定义 ---
W_UNIT = 1.0

colors = {
    'forward': '#a7c7e7', 'backward': '#c1e1c1',
    'fwd_prop': '#87ceeb', 'grad_prop': '#d1b3e2',
    'grad_sync': '#fffacd'
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
    'forward':    1,
    'backward':   1,
    'fwd_prop':   0.5,
    'grad_prop':  0.5,
    'grad_sync':  0.5,
    'param_sync': 1
}
default_exclusive_tiers = {
    'fwd_prop': None,
    'grad_prop': None,
    'grad_sync': None,
}

# --- 核心模拟与计算函数 ---

def _to_pdf_filename(name):
    """确保输出为 .pdf 文件。"""
    base, ext = os.path.splitext(str(name))
    return base + '.pdf' if ext.lower() != '.pdf' else str(name)


def calculate_full_pipeline_schedule(n, num_gpus, is_ideal, lengths, priorities, exclusive_tiers,
                                     impact, sync_solo_w=1.0, sync_freq=1, use_recv_congestion=True, mode="1f1b"):
    """
    Compute the full pipeline schedule and return block records, total time and throttled tasks.

    Parameters
    ----------
    n : int
        Number of micro-batches.
    num_gpus : int
        Number of GPUs to simulate.
    is_ideal : bool
        If True, place all grad_sync tasks on a dedicated channel.
    lengths : dict
        Base work units for each task type (forward/backward/prop/sync).
    priorities : dict
        Relative priorities for sharing tasks. These override the module's
        ``default_priorities`` when provided.
    exclusive_tiers : dict
        Exclusive tier assignments for communication tasks. These override
        ``default_exclusive_tiers`` when provided.
    impact : float
        Reduction factor applied to forward/backward compute when grad_sync
        communication is occurring on the same GPU.
    sync_solo_w : float, optional
        Weight allocated to a solo sync task (unused in current implementation).
    sync_freq : int, optional
        The frequency (in micro-batches) at which gradient synchronization occurs.
    use_recv_congestion : bool, optional
        Enable receiver-side congestion control.
    mode : str, optional
        Scheduling mode ('1f1b' or 'gpipe').

    Notes
    -----
    Internally, this function temporarily overrides the module-level
    ``default_priorities`` and ``default_exclusive_tiers`` dictionaries with
    the values supplied via the ``priorities`` and ``exclusive_tiers``
    arguments, respectively. This ensures that the scheduling logic honours
    caller-specified priorities and exclusivity tiers. The original defaults
    are restored before the function returns so that subsequent calls are
    unaffected.
    """

    # Temporarily override default priorities and exclusive tiers with user-specified
    # values. We copy the originals up front so that we can restore them later.
    global default_priorities, default_exclusive_tiers
    orig_default_priorities = deepcopy(default_priorities)
    orig_default_exclusive_tiers = deepcopy(default_exclusive_tiers)
    # Override per-call priorities/exclusive tiers if provided
    if priorities:
        for t, p_val in priorities.items():
            # Only update when provided (None indicates no override)
            if p_val is not None:
                default_priorities[t] = p_val
    if exclusive_tiers:
        for t, tier_val in exclusive_tiers.items():
            default_exclusive_tiers[t] = tier_val

    all_tasks = []
    # 1) 创建任务
    for j in range(num_gpus):
        compute_ch = j * 2 + 0
        comm_ch   = j * 2 + 1
        for i in range(n):
            all_tasks.append({'id': ('forward', i, j),  'type': 'forward',  'work_needed': lengths['forward'],
                              'dependencies': [], 'status': 'pending', 'channel': compute_ch, 'label': f'{i + 1}'})
            all_tasks.append({'id': ('backward', i, j), 'type': 'backward', 'work_needed': lengths['backward'],
                              'dependencies': [], 'status': 'pending', 'channel': compute_ch, 'label': f'{i + 1}'})
            if j < num_gpus - 1:
                all_tasks.append({'id': ('fwd_prop',  i, j), 'type': 'fwd_prop',  'work_needed': lengths['fwd_prop'],
                                  'dependencies': [], 'status': 'pending', 'channel': comm_ch, 'label': f'{i + 1}'})
            if j > 0:
                all_tasks.append({'id': ('grad_prop', i, j), 'type': 'grad_prop', 'work_needed': lengths['grad_prop'],
                                  'dependencies': [], 'status': 'pending', 'channel': comm_ch, 'label': f'{i + 1}'})
            is_sync_point   = (i + 1) % sync_freq == 0
            is_last_batch   = (i == n - 1)
            is_leftover_sync = is_last_batch and ((i + 1) % sync_freq != 0)
            if is_sync_point or is_leftover_sync:
                num_grads  = (i + 1) % sync_freq if is_leftover_sync else sync_freq
                sync_work  = lengths['grad_sync'] * num_grads
                sync_ch    = (comm_ch) + num_gpus * 2 if is_ideal else comm_ch
                all_tasks.append({'id': ('grad_sync', i, j), 'type': 'grad_sync', 'work_needed': sync_work,
                                  'dependencies': [], 'status': 'pending', 'channel': sync_ch, 'label': f'{(i // sync_freq) + 1}'})

    # 2) 依赖关系
    if mode.lower() == 'gpipe':
        barrier_id = ('forward', n - 1, num_gpus - 1)
        for task in all_tasks:
            if len(task['id']) < 3: continue
            t, i, j = task['id']
            if t == 'forward':
                if j == 0:
                    if i > 0: task['dependencies'].append(('forward', i - 1, j))
                else:
                    task['dependencies'].append(('fwd_prop', i, j - 1))
            elif t == 'backward':
                task['dependencies'] += [('forward', i, j)]
                if j < num_gpus - 1: task['dependencies'].append(('grad_prop', i, j + 1))
                task['dependencies'].append(barrier_id)
            elif t == 'fwd_prop':
                task['dependencies'].append(('forward', i, j))
            elif t == 'grad_prop':
                task['dependencies'].append(('backward', i, j))
            elif t == 'grad_sync':
                task['dependencies'].append(('backward', i, j))
                prev_sync_idx = i - sync_freq
                if prev_sync_idx >= 0:
                    prev_sync_id = ('grad_sync', prev_sync_idx, j)
                    if any(tk['id'] == prev_sync_id for tk in all_tasks):
                        task['dependencies'].append(prev_sync_id)
    else:
        for task in all_tasks:
            if len(task['id']) < 3: continue
            t, i, j = task['id']
            if t == 'forward':
                if j == 0:
                    if i < num_gpus:
                        if i > 0: task['dependencies'].append(('forward', i - 1, 0))
                    else:
                        task['dependencies'].append(('backward', i - num_gpus, 0))
                else:
                    task['dependencies'].append(('fwd_prop', i, j - 1))
                    if j == num_gpus - 1 and i > 0:
                        task['dependencies'].append(('backward', i - 1, j))
            elif t == 'backward':
                task['dependencies'].append(('forward', i, j))
                if j < num_gpus - 1: task['dependencies'].append(('grad_prop', i, j + 1))
            elif t == 'fwd_prop':
                task['dependencies'].append(('forward', i, j))
            elif t == 'grad_prop':
                task['dependencies'].append(('backward', i, j))
            elif t == 'grad_sync':
                task['dependencies'].append(('backward', i, j))
                prev_sync_idx = i - sync_freq
                if prev_sync_idx >= 0:
                    prev_sync_id = ('grad_sync', prev_sync_idx, j)
                    if any(tk['id'] == prev_sync_id for tk in all_tasks):
                        task['dependencies'].append(prev_sync_id)

    # 3) 事件循环
    current_time = 0.0
    blocks = []
    throttled_task_ids = set()
    finished_task_ids = set()

    while len(finished_task_ids) < len(all_tasks):
        for task in all_tasks:
            if task['status'] == 'pending' and all(dep in finished_task_ids for dep in task['dependencies']):
                task['status'] = 'ready'

        tasks_to_set_running = []
        shares_now = {}
        active_tasks = [t for t in all_tasks if t['status'] in ['ready', 'running']]
        tasks_by_channel = defaultdict(list)
        for task in active_tasks: tasks_by_channel[task['channel']].append(task)

        for ch, tasks_in_channel in tasks_by_channel.items():
            is_compute_channel = (ch % 2 == 0)
            eligible = [t for t in tasks_in_channel if t['status'] in ['ready', 'running']]
            if not eligible: continue

            if is_compute_channel:
                if any(t['status'] == 'running' for t in eligible):
                    chosen = [t for t in eligible if t['status'] == 'running']
                else:
                    def sort_key(task):
                        type_prio = 0 if task['type'] == 'forward' else 1
                        return (type_prio, task['id'][1])
                    eligible.sort(key=sort_key)
                    chosen = [eligible[0]]
                tasks_to_set_running.extend(chosen)
                for t in chosen: shares_now[t['id']] = W_UNIT
            else:
                exclusive_tasks = []
                sharing_tasks   = []
                zero_prio_tasks = []
                for t in eligible:
                    if default_exclusive_tiers.get(t['type']) is not None:
                        exclusive_tasks.append(t)
                    elif default_priorities.get(t['type'], 0) > 1e-9:
                        sharing_tasks.append(t)
                    else:
                        zero_prio_tasks.append(t)
                chosen = []
                if exclusive_tasks:
                    exclusive_tasks.sort(key=lambda t: default_exclusive_tiers[t['type']])
                    ht = exclusive_tasks[0]
                    chosen = [ht]
                    shares_now[ht['id']] = W_UNIT
                elif sharing_tasks:
                    chosen = sharing_tasks
                    by_type = defaultdict(list)
                    for t in chosen: by_type[t['type']].append(t)
                    type_p = {tp: default_priorities.get(tp, 0) for tp in by_type}
                    total_p = sum(type_p.values())
                    if total_p > 1e-9:
                        for tp, tasks in by_type.items():
                            share_type = W_UNIT * (type_p[tp] / total_p)
                            share_each = share_type / len(tasks)
                            for t in tasks: shares_now[t['id']] = share_each
                elif zero_prio_tasks:
                    chosen = zero_prio_tasks
                    share_each = W_UNIT / len(chosen)
                    for t in chosen: shares_now[t['id']] = share_each
                if chosen:
                    tasks_to_set_running.extend(chosen)

        # 拥塞控制（省略：与原版本一致，下方逻辑保持不变）
        if use_recv_congestion:
            for _ in range(num_gpus * 2):
                for receiver_j in range(1, num_gpus - 1):
                    incoming_fwd  = [t for t in active_tasks if t['type'] == 'fwd_prop'  and t['id'][2] == receiver_j - 1]
                    incoming_grad = [t for t in active_tasks if t['type'] == 'grad_prop' and t['id'][2] == receiver_j + 1]
                    total_in = sum(shares_now.get(t['id'], 0) for t in incoming_fwd + incoming_grad)
                    if total_in > W_UNIT:
                        incoming = [t for t in (incoming_fwd + incoming_grad) if t['id'] in shares_now]
                        if not incoming: continue
                        excl_in   = [t for t in incoming if default_exclusive_tiers.get(t['type']) is not None]
                        top_tasks = []
                        if excl_in:
                            highest = min(default_exclusive_tiers[t['type']] for t in excl_in)
                            top_tasks = [t for t in excl_in if default_exclusive_tiers[t['type']] == highest]
                        excl_share = sum(shares_now.get(t['id'], 0) for t in top_tasks)
                        remaining  = W_UNIT
                        if top_tasks:
                            if excl_share >= W_UNIT - 1e-9:
                                scale = W_UNIT / excl_share
                                for t in top_tasks:
                                    shares_now[t['id']] *= scale; t['bw_fixed'] = True; throttled_task_ids.add(t['id'])
                                for t in incoming:
                                    if t not in top_tasks:
                                        shares_now[t['id']] = 0.0; t['bw_fixed'] = True; throttled_task_ids.add(t['id'])
                                continue
                            else:
                                for t in top_tasks:
                                    t['bw_fixed'] = True; throttled_task_ids.add(t['id'])
                                remaining -= excl_share
                        non_excl = [t for t in incoming if t not in top_tasks]
                        if not non_excl: continue
                        current_total = sum(shares_now.get(t['id'], 0) for t in non_excl)
                        if current_total > remaining + 1e-9:
                            groups = defaultdict(lambda: {'tasks': [], 'share': 0.0, 'priority': 0.0})
                            for t in non_excl:
                                p = default_priorities.get(t['type'], 0.0)
                                groups[p]['priority'] = p
                                groups[p]['tasks'].append(t)
                                groups[p]['share'] += shares_now.get(t['id'], 0.0)
                            excess = current_total - remaining
                            scaling = {k: 1.0 for k in groups}
                            remain  = {k: {'share': g['share'], 'priority': g['priority']} for k, g in groups.items()}
                            while excess > 1e-9 and remain:
                                weights, total_w = {}, 0.0
                                for k, g in remain.items():
                                    p = g['priority']; w = (1e12 if p <= 1e-9 else 1.0 / p)
                                    weights[k] = w; total_w += w
                                if total_w <= 1e-9: break
                                fully = []
                                for k, g in remain.items():
                                    w = weights[k]; proposed = excess * (w / total_w)
                                    if proposed >= g['share'] - 1e-12: fully.append(k)
                                if not fully:
                                    for k, g in remain.items():
                                        w = weights[k]; reduction = excess * (w / total_w)
                                        new_share = max(0.0, g['share'] - reduction)
                                        if g['share'] > 1e-12: scaling[k] *= new_share / g['share']
                                        else: scaling[k] = 0.0
                                    excess = 0.0; break
                                else:
                                    for k in fully:
                                        g_share = remain[k]['share']; scaling[k] = 0.0
                                        excess -= g_share; del remain[k]
                                    continue
                            for k, g in groups.items():
                                f = scaling.get(k, 0.0)
                                for t in g['tasks']:
                                    shares_now[t['id']] *= f; t['bw_fixed'] = True; throttled_task_ids.add(t['id'])
                        else:
                            for t in non_excl:
                                t['bw_fixed'] = True; throttled_task_ids.add(t['id'])

                # 发送端再分配
                for sender_j in range(num_gpus):
                    comm_ch = sender_j * 2 + 1
                    sending = [t for t in active_tasks if t['channel'] == comm_ch and t['id'][2] == sender_j]
                    if not sending: continue
                    fixed_w = sum(shares_now.get(t['id'], 0) for t in sending if t.get('bw_fixed'))
                    unalloc = W_UNIT - fixed_w
                    unfixed = [t for t in sending if not t.get('bw_fixed')]
                    if not unfixed or unalloc < 1e-9: continue
                    excl_u, share_u, zero_u = [], [], []
                    for t in unfixed:
                        if default_exclusive_tiers.get(t['type']) is not None: excl_u.append(t)
                        elif default_priorities.get(t['type'], 0) > 1e-9:   share_u.append(t)
                        else: zero_u.append(t)
                    if excl_u:
                        excl_u.sort(key=lambda t: default_exclusive_tiers[t['type']])
                        winner = excl_u[0]
                        for t in unfixed: shares_now[t['id']] = (unalloc if t == winner else 0)
                    elif share_u:
                        by_type = defaultdict(list)
                        for t in share_u: by_type[t['type']].append(t)
                        type_p  = {tp: default_priorities.get(tp, 0) for tp in by_type}
                        total_p = sum(type_p.values())
                        if total_p > 0:
                            for tp, tasks in by_type.items():
                                share_tp = unalloc * (type_p[tp] / total_p)
                                share_each = share_tp / len(tasks)
                                for t in tasks: shares_now[t['id']] = share_each
                    elif zero_u:
                        share_each = unalloc / len(zero_u)
                        for t in zero_u: shares_now[t['id']] = share_each

            for t in all_tasks:
                if 'bw_fixed' in t: del t['bw_fixed']

        # 推进时间
        efficiencies = {}
        for j in range(num_gpus):
            comm_ch = j * 2 + 1
            sync_share = sum(shares_now.get(t['id'], 0) for t in active_tasks
                             if t['channel'] == comm_ch and t['type'] == 'grad_sync')
            efficiencies[j] = {'fwd': 1.0 - (sync_share * impact),
                               'bwd': 1.0 - (sync_share * impact)}
        for t in {t['id']: t for t in tasks_to_set_running}.values():
            if t['status'] == 'ready': t['status'] = 'running'

        tasks_with_share = [t for t in active_tasks if shares_now.get(t['id'], 0) > 1e-9]
        if not tasks_with_share:
            if len(finished_task_ids) == len(all_tasks): break
            else: continue

        min_time = float('inf')
        for t in tasks_with_share:
            share = shares_now.get(t['id'], 0)
            if not is_ideal:
                g = t['id'][2]
                if t['type'] == 'forward':  share *= efficiencies[g]['fwd']
                elif t['type'] == 'backward': share *= efficiencies[g]['bwd']
            if share > 1e-9:
                min_time = min(min_time, (t.get('work_needed', 0) - t.get('work_done', 0)) / share)
        slice_dur = min_time if min_time != float('inf') and min_time > 1e-9 else 1e-9
        next_t = current_time + slice_dur

        for t in tasks_with_share:
            if 'work_done' not in t: t['work_done'] = 0
            share = shares_now.get(t['id'], 0)
            if not is_ideal:
                g = t['id'][2]
                if t['type'] == 'forward': share *= efficiencies[g]['fwd']
                elif t['type'] == 'backward': share *= efficiencies[g]['bwd']
            t['work_done'] += share * slice_dur
            if share > 1e-9:
                blocks.append({'id': t['id'], 'label': t['label'], 'type': t['type'],
                               'start': current_time, 'end': next_t,
                               'color': colors[t['type']], 'width': share, 'channel': t['channel']})
        current_time = next_t
        for t in all_tasks:
            if t['status'] == 'running' and t.get('work_done', 0) >= t['work_needed'] - 1e-9:
                t['status'] = 'finished'; finished_task_ids.add(t['id'])

    # Restore the original default priorities and exclusive tiers to avoid leaking
    # overrides into subsequent invocations. Without this reset, any caller-supplied
    # configuration would persist and incorrectly influence later simulations.
    default_priorities = orig_default_priorities
    default_exclusive_tiers = orig_default_exclusive_tiers
    return blocks, current_time, throttled_task_ids

# --- 单 GPU 图（单通信通道可视化） ---

def plot_single_gpu_chart(gpu_rank, all_blocks, total_time, throttled_task_ids,
                          show_throttling, show_plot, schedule_mode='1f1b', num_gpus=None):
    blocks_for_gpu = [b for b in all_blocks if b['id'][2] == gpu_rank]
    if not blocks_for_gpu:
        print(f"GPU {gpu_rank+1} 没有任务，跳过绘图。")
        return
    visual_layout_labels = ["Compute", "Communication"]
    num_channels = 2

    # 主图显著增高
    fig, ax = plt.subplots(1, 1, figsize=(22, 3.2 * num_channels))
    fig.suptitle(f"GPU {gpu_rank + 1} - {schedule_mode.upper()} Schedule", fontsize=24, y=0.955)
    # 通过 subplots_adjust 控制上下边距：底部留 10% 空白，上部 8% 供标题+图例
    fig.subplots_adjust(left=0.1, right=0.995, bottom=0.40, top=0.92)

    ax.set_ylabel("Resource Channels", fontsize=24)
    ax.set_ylim(0, num_channels)
    ax.set_yticks([i + 0.5 for i in range(num_channels)])
    ax.set_yticklabels(visual_layout_labels, fontsize=34)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    y_map = {gpu_rank * 2 + 0: 0, gpu_rank * 2 + 1: 1}
    drawable_blocks = []
    time_slices = sorted(list(set([b['start'] for b in blocks_for_gpu] + [b['end'] for b in blocks_for_gpu])))
    for i in range(len(time_slices) - 1):
        start, end = time_slices[i], time_slices[i+1]
        blocks_in_slice = [b for b in blocks_for_gpu if b['start'] <= start and b['end'] >= end]
        blocks_by_channel = defaultdict(list)
        for b in blocks_in_slice: blocks_by_channel[b['channel']].append(b)
        for ch, blocks_on_ch in blocks_by_channel.items():
            if ch not in y_map: continue
            y_base = y_map[ch]; y_stack = 0
            for block in sorted(blocks_on_ch, key=lambda b: b['type']):
                nb = block.copy()
                nb['start'], nb['end'] = start, end
                nb['y_pos']  = y_base + y_stack
                nb['height'] = block['width']
                drawable_blocks.append(nb)
                y_stack += block['width']

    key_func = lambda b: b['id']
    drawable_blocks.sort(key=key_func)
    for task_id, group_iter in groupby(drawable_blocks, key=key_func):
        group = sorted(list(group_iter), key=lambda b: b['start'])
        if not group: continue
        cur = []
        for block in group:
            if not cur or abs(block['start'] - cur[-1]['end']) < 1e-9:
                cur.append(block)
            else:
                draw_polygon_for_segment(ax, cur, throttled_task_ids, show_throttling); cur = [block]
        if cur: draw_polygon_for_segment(ax, cur, throttled_task_ids, show_throttling)

    ax.set_xlabel("Time", fontsize=24)
    max_t = float(np.ceil(total_time * 1.05))
    ax.set_xlim(0, max_t)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    # **刻度字体进一步放大：34pt**
    ax.tick_params(axis='x', which='both', labelsize=34)
    ax.tick_params(axis='y', which='both', labelsize=34)

    # 图例锚到“轴坐标”，贴紧主图上沿
    legend_patches = [mpatches.Patch(color=c, label=t.replace('_', ' ').title()) for t, c in colors.items()]
    fig.legend(
        handles=legend_patches,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.36),
        ncol=len(colors),
        prop={'size': 34, 'weight': 'bold'},
        frameon=False,
        borderaxespad=0.10,
        handlelength=0.9,
        handleheight=0.8,
        handletextpad=0.10,
        columnspacing=0.30,
        labelspacing=0.20
    )

    output_filename = f"single_channel_gpu{gpu_rank+1}_{schedule_mode}.pdf"
    pdf_path = _to_pdf_filename(output_filename)
    plt.savefig(pdf_path, dpi=200, format='pdf')
    print(f"\n为 GPU {gpu_rank+1} 生成的单通道可视化图表已保存为: {pdf_path}")
    if show_plot: plt.show()
    plt.close(fig)


def draw_polygon_for_segment(ax, segment, throttled_task_ids, show_throttling):
    if not segment: return
    task_id = segment[0]['id']
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

    polygon = mpatches.Polygon(top_verts + bottom_verts[::-1], closed=True,
                               facecolor=segment[0]['color'], edgecolor=edge_color,
                               alpha=0.9, linewidth=line_width, hatch=hatch, zorder=1)
    ax.add_patch(polygon)

    min_start = segment[0]['start']
    max_end   = segment[-1]['end']
    total_area = sum((b['end'] - b['start']) * b['height'] for b in segment)
    if total_area > 1e-9:
        avg_y_center = sum((b['end'] - b['start']) * b['height'] *
                           (b['y_pos'] + b['height'] / 2) for b in segment) / total_area
    else:
        avg_y_center = segment[0]['y_pos'] + segment[0]['height'] / 2

    ax.text(min_start + (max_end - min_start) / 2, avg_y_center, segment[0]['label'],
            ha='center', va='center', color='black', fontsize=28, weight='bold', zorder=2)


def add_gpu_two_channel_legend(fig, anchor=(0.02, 0.86), size=(0.24, 0.13), face="#ffffff", edge_lw=1.2):
    axL = fig.add_axes([anchor[0], anchor[1], size[0], size[1]], frame_on=False, zorder=50)
    try: axL.set_in_layout(False)
    except Exception: pass
    axL.set_xlim(0, 1); axL.set_ylim(0, 1); axL.axis('off')
    # 文字全部加粗
    axL.text(0.02, 0.50, 'GPU', ha='left', va='center', fontsize=24, fontweight='bold', transform=axL.transAxes)
    box_x, box_y = 0.20, 0.12; box_w, box_h = 0.32, 0.76
    axL.add_patch(mpatches.Rectangle((box_x, box_y + box_h/2), box_w, box_h/2,
                                     facecolor=face, edgecolor='none',
                                     transform=axL.transAxes, zorder=51))
    axL.add_patch(mpatches.Rectangle((box_x, box_y), box_w, box_h/2,
                                     facecolor=face, edgecolor='none',
                                     transform=axL.transAxes, zorder=51))
    axL.add_patch(mpatches.Rectangle((box_x, box_y), box_w, box_h,
                                     facecolor='none', edgecolor='black', lw=edge_lw,
                                     transform=axL.transAxes, zorder=53))
    axL.plot([box_x, box_x + box_w], [box_y + box_h/2, box_y + box_h/2],
             color='black', lw=edge_lw, linestyle='--', dashes=(4, 3),
             transform=axL.transAxes, zorder=54)
    label_x = box_x + box_w + 0.055
    axL.text(label_x, box_y + box_h*0.75, 'Communication', ha='left', va='center',
             fontsize=28, fontweight='bold', transform=axL.transAxes)
    axL.text(label_x, box_y + box_h*0.25, 'Compute', ha='left', va='center',
             fontsize=28, fontweight='bold', transform=axL.transAxes)

# --- 总览图 ---

def plot_total_pipeline_chart(all_blocks, total_time, num_gpus, throttled_task_ids,
                              show_throttling, show_completion_time, show_plot, schedule_mode='1f1b'):
    # 主图显著增高
    fig, axes = plt.subplots(num_gpus, 1, figsize=(24, 2.8 * num_gpus), sharex=True, gridspec_kw={'hspace': 0})
    # 标题加粗
    fig.suptitle(f"Default {schedule_mode.upper()} Pipeline Schedule", fontsize=38, fontweight='bold', y=0.955)
    # 上下边距：底部保留 10% 空白，上部 8–9% 给标题、小示意图和图例
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.20, top=0.80)

    # 左上角两通道示意（内部文本已加粗）
    add_gpu_two_channel_legend(fig, anchor=(0.02, 0.86), size=(0.24, 0.13))

    for gpu_rank in range(num_gpus):
        ax = axes[gpu_rank]
        blocks_for_gpu = [b for b in all_blocks if b['id'][2] == gpu_rank]
        y_map = {gpu_rank * 2 + 0: 0, gpu_rank * 2 + 1: 1}
        visual_layout_labels = ["", ""]
        num_channels = 2

        # Y 轴标签（GPU i）加粗
        ax.set_ylabel(f"GPU {gpu_rank + 1}", fontsize=34, fontweight='bold', rotation=0, labelpad=44, va='center')
        ax.set_ylim(0, num_channels)
        ax.set_yticks([i + 0.5 for i in range(num_channels)])
        ax.set_yticklabels(visual_layout_labels, fontsize=28)
        ax.grid(axis='x', linestyle='--', alpha=0.7)

        if not blocks_for_gpu: continue

        drawable_blocks = []
        time_slices = sorted(list(set([b['start'] for b in blocks_for_gpu] + [b['end'] for b in blocks_for_gpu])))
        for i in range(len(time_slices) - 1):
            start, end = time_slices[i], time_slices[i+1]
            blocks_in_slice = [b for b in blocks_for_gpu if b['start'] <= start and b['end'] >= end]
            blocks_by_channel = defaultdict(list)
            for b in blocks_in_slice: blocks_by_channel[b['channel']].append(b)
            for ch, blocks_on_ch in blocks_by_channel.items():
                if ch not in y_map: continue
                y_base = y_map[ch]; y_stack = 0
                for block in sorted(blocks_on_ch, key=lambda b: b['type']):
                    nb = block.copy()
                    nb['start'], nb['end'] = start, end
                    nb['y_pos']  = y_base + y_stack
                    nb['height'] = block['width']
                    drawable_blocks.append(nb)
                    y_stack += block['width']

        key_func = lambda b: b['id']
        drawable_blocks.sort(key=key_func)
        for task_id, group_iter in groupby(drawable_blocks, key=key_func):
            group = sorted(list(group_iter), key=lambda b: b['start'])
            if not group: continue
            cur = []
            for block in group:
                if not cur or abs(block['start'] - cur[-1]['end']) < 1e-9:
                    cur.append(block)
                else:
                    draw_polygon_for_segment(ax, cur, throttled_task_ids, show_throttling); cur = [block]
            if cur: draw_polygon_for_segment(ax, cur, throttled_task_ids, show_throttling)

    # X 轴标题加粗（只在最后一行轴）
    axes[-1].set_xlabel("Time", fontsize=34, fontweight='bold')

    max_t = float(np.ceil(total_time * 1.05))
    for ax in axes:
        ax.set_xlim(0, max_t)
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        # **统一放大 x/y 刻度字号为 34pt**
        ax.tick_params(axis='x', which='both', labelsize=34)
        ax.tick_params(axis='y', which='both', labelsize=34)
        ax.hlines(1, 0, max_t, colors='black', linestyles=(0, (4, 4)), linewidth=1.0, alpha=0.9, zorder=0.5)

        # 刻度文字加粗
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight('bold')

    if show_completion_time:
        ax0 = axes[0]
        ax0.axvline(x=total_time, color='r', linestyle='--', linewidth=2, zorder=5)
        ax0.annotate(f'{total_time:.2f}', xy=(total_time, ax0.get_ylim()[1]),
                     xytext=(total_time, ax0.get_ylim()[1] + 0.5),
                     arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=8),
                     ha='center', va='bottom', fontsize=34, color='red', fontweight='bold', zorder=6)

    # 图例整体加粗
    legend_patches = [mpatches.Patch(color=c, label=t.replace('_', ' ').title()) for t, c in colors.items()]
    fig.legend(handles=legend_patches,
               loc='upper center',
               bbox_to_anchor=(0.5, 1.45),
               bbox_transform=axes[0].transAxes,
               ncol=len(colors),
               prop={'size': 30, 'weight': 'bold'},
               frameon=False,
               borderaxespad=0.2)

    output_filename = f"single_channel_total_{schedule_mode}.pdf"
    pdf_path = _to_pdf_filename(output_filename)
    plt.savefig(pdf_path, dpi=150, format='pdf')
    print(f"\n总览图已保存为: {pdf_path}")
    if show_plot: plt.show()
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate a 1F1B/GPipe pipeline (single-comm channel).')
    parser.add_argument('-n', '--num_groups', type=int, default=8, help='Number of micro-batches.')
    parser.add_argument('--gpus', type=int, default=4, help='Number of GPUs (>=2).')
    parser.add_argument('--mode', choices=['1f1b','gpipe'], default='1f1b', help='Scheduling mode: 1f1b (default) or gpipe (F-then-B).')
    parser.add_argument('--impact', type=float, default=None, help='Impact of grad_sync on both forward and backward compute.')
    parser.add_argument('--fwd-impact', type=float, default=None, help='[已弃用] 前向影响')
    parser.add_argument('--bwd-impact', type=float, default=None, help='[已弃用] 反向影响')
    parser.add_argument('--sync-solo-w', type=float, default=1.0, help='Resource share for a solo sync task.')
    parser.add_argument('--sync-freq', type=int, default=2, help='Frequency of grad_sync tasks.')
    parser.add_argument('--no-show', action='store_true', help='Save chart without displaying it.')
    parser.add_argument('--show-throttling', action='store_true', help='Highlight communication tasks throttled by receiver congestion.')
    parser.add_argument('--show-completion-time', action='store_true', help='Show the completion time annotation on the total chart.') 
    parser.add_argument('--enable-recv-congestion', action='store_true', help='Enable receiver-side congestion control.')

    args = parser.parse_args()

    lengths = default_lengths.copy()
    if 'fwd_prop' not in lengths:
        lengths['fwd_prop'] = lengths['grad_prop']

    final_impact = args.impact
    if final_impact is None:
        if args.fwd_impact is not None: final_impact = args.fwd_impact
        elif args.bwd_impact is not None: final_impact = args.bwd_impact
        else: final_impact = 0.2

    all_blocks, total_time, throttled_task_ids = calculate_full_pipeline_schedule(
        n=args.num_groups,
        num_gpus=max(2, args.gpus),
        is_ideal=False,
        lengths=lengths,
        priorities=default_priorities,
        exclusive_tiers=default_exclusive_tiers,
        impact=final_impact,
        sync_solo_w=args.sync_solo_w,
        sync_freq=args.sync_freq,
        use_recv_congestion=args.enable_recv_congestion,
        mode=args.mode
    )

    # 每个 GPU 的单图（不显示）
    for gpu_rank in range(max(2, args.gpus)):
        plot_single_gpu_chart(gpu_rank, all_blocks, total_time, throttled_task_ids,
                              args.show_throttling, show_plot=False, schedule_mode=args.mode, num_gpus=max(2, args.gpus))

    # 总览图
    plot_total_pipeline_chart(all_blocks, total_time, max(2, args.gpus), throttled_task_ids,
                              args.show_throttling, show_completion_time=args.show_completion_time,
                              show_plot=(not args.no_show), schedule_mode=args.mode)
