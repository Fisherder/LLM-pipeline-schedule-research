"""
文件名: generate_metrics.py

描述:
    该脚本用于在给定参数配置和通信优先级组合的情况下，
    调用流水线模拟器（1F1B 或 GPipe，单/双通道），
    生成包括训练总时间、计算与通信时间占比、
    通信吞吐量、通信延迟及资源利用率等多项性能指标。

    与 data_generator.py 类似，该脚本遍历给定 JSON 配置中的所有参数组合、
    不同的主动独占场景以及共享优先级排列，通过调用
    schedule_visualizer_* 模块的 calculate_full_pipeline_schedule 函数
    获取模拟调度结果，并从中提取详细指标。

用法示例:
    python generate_metrics.py --config my_experiment_config.json \
        --pipeline 1f1b --simulator single_channel \
        --step 0.1 --output outputs/data/metrics.csv \
        --enable-recv-congestion

    其中:
        --config 指定参数配置文件;
        --pipeline 选择 "1f1b" 或 "gpipe";
        --simulator 选择 "single_channel" 或 "dual/duel_channel";
        --step 设置共享优先级扫描步长;
        --enable-recv-congestion 在 1f1b 单通道场景下启用接收端拥塞控制;
        --output 指定输出 CSV 文件。

输出 CSV 列包含:
    n, sync_freq, fwd_len, bwd_len, prop_len, sync_base_len, fwd_impact, bwd_impact,
    p_fwd_prop, p_grad_prop, p_grad_sync, total_time,
    compute_ratio, comm_ratio, avg_throughput, avg_latency,
    compute_utilization, comm_utilization

作者: ChatGPT
日期: 2025-08-10
"""

import argparse
import importlib
import inspect
import json
import multiprocessing
import numpy as np
import os
import datetime
from itertools import product
from typing import Dict, List, Tuple


# ---------- 工具函数 ----------
def generate_scan_values(config_value):
    """从配置项生成扫描列表，同 data_generator.py"""
    if isinstance(config_value, list):
        return config_value
    elif isinstance(config_value, dict):
        steps = int(round((config_value['stop'] - config_value['start']) / config_value.get('step', 1.0))) + 1
        return np.linspace(config_value['start'], config_value['stop'], steps)
    return [config_value]


def compute_network_load(blocks: List[Dict]) -> Tuple[List[float], Dict[str, List[float]]]:
    """
    根据调度结果 blocks，统计单通道场景下通信带宽随时间的占用。
    返回时间点列表 times 和各通信类型的带宽占用 loads。
    - loads['total'] 为在该时间段内所有通信任务的带宽和
    - loads['fwd_prop'], loads['grad_prop'], loads['grad_sync'] 分别对应不同通信任务
    """
    comm_types = {"fwd_prop", "grad_prop", "grad_sync"}
    # 收集所有通信相关任务的边界时间
    time_points = sorted({float(b['start']) for b in blocks if b['type'] in comm_types} |
                         {float(b['end']) for b in blocks if b['type'] in comm_types})
    if len(time_points) < 2:
        return [], {'total': [], 'fwd_prop': [], 'grad_prop': [], 'grad_sync': []}
    loads = {'total': [], 'fwd_prop': [], 'grad_prop': [], 'grad_sync': []}
    for i in range(len(time_points) - 1):
        t0, t1 = time_points[i], time_points[i + 1]
        total = fwd = grad = sync = 0.0
        for b in blocks:
            if b['type'] in comm_types and b['start'] <= t0 and b['end'] >= t1:
                # b['width'] 表示占用的带宽份额，若不存在则视为1
                w = float(b.get('width', 1.0))
                total += w
                if b['type'] == 'fwd_prop':
                    fwd += w
                elif b['type'] == 'grad_prop':
                    grad += w
                elif b['type'] == 'grad_sync':
                    sync += w
        loads['total'].append(total)
        loads['fwd_prop'].append(fwd)
        loads['grad_prop'].append(grad)
        loads['grad_sync'].append(sync)
    return time_points[:-1], loads


def compute_metrics(blocks: List[Dict]) -> Dict[str, float]:
    """
    根据调度块列表计算各项指标。
    返回:
        dict 包含 total_time, compute_ratio, comm_ratio, avg_throughput, avg_latency,
        compute_utilization, comm_utilization
    """
    if not blocks:
        return {
            'total_time': 0.0,
            'compute_ratio': 0.0,
            'comm_ratio': 0.0,
            'avg_throughput': 0.0,
            'avg_latency': 0.0,
            'compute_utilization': 0.0,
            'comm_utilization': 0.0,
        }
    # 总训练时间为所有任务的结束时间最大值
    total_time = max(float(b['end']) for b in blocks)
    # 计算通道总时长 (所有 GPU) 和通信通道总时长
    compute_time = sum((float(b['end']) - float(b['start'])) for b in blocks if b['type'] in ('forward', 'backward'))
    comm_time = sum((float(b['end']) - float(b['start'])) for b in blocks if b['type'] in ('fwd_prop', 'grad_prop', 'grad_sync', 'param_sync'))
    # 计算与通信时间占比: 相对于所有计算+通信任务的总和
    total_compute_comm = compute_time + comm_time
    if total_compute_comm > 0:
        compute_ratio = compute_time / total_compute_comm
        comm_ratio = comm_time / total_compute_comm
    else:
        compute_ratio = 0.0
        comm_ratio = 0.0
    # 通信吞吐量: 按时间片平均 aggregated bandwidth 使用率
    times, loads = compute_network_load(blocks)
    if loads['total']:
        avg_throughput = float(np.mean(loads['total']))
    else:
        avg_throughput = 0.0
    # 通信延迟: 平均每个通信任务持续时间
    comm_latencies = [float(b['end']) - float(b['start']) for b in blocks if b['type'] in ('fwd_prop', 'grad_prop', 'grad_sync')]
    avg_latency = float(np.mean(comm_latencies)) if comm_latencies else 0.0
    # 推断 GPU 数量
    # 块 id 格式: (task_type, micro_batch_index, stage_index)
    stage_indices = [b['id'][2] for b in blocks if isinstance(b['id'], tuple) and len(b['id']) >= 3]
    num_gpus = max(stage_indices) + 1 if stage_indices else 1
    # 计算计算资源利用率: 所有计算任务持续时间 / (num_gpus * total_time)
    compute_utilization = compute_time / (num_gpus * total_time) if total_time > 0 else 0.0
    # 计算通信资源利用率: 对每个通信任务乘以占用带宽宽度，再除以 (num_gpus * total_time)
    comm_utilization_sum = sum(float(b.get('width', 1.0)) * (float(b['end']) - float(b['start']))
                              for b in blocks if b['type'] in ('fwd_prop', 'grad_prop', 'grad_sync'))
    comm_utilization = comm_utilization_sum / (num_gpus * total_time) if total_time > 0 else 0.0
    return {
        'total_time': total_time,
        'compute_ratio': compute_ratio,
        'comm_ratio': comm_ratio,
        'avg_throughput': avg_throughput,
        'avg_latency': avg_latency,
        'compute_utilization': compute_utilization,
        'comm_utilization': comm_utilization
    }


def run_single_simulation_metrics(args_tuple) -> str:
    """
    核心工作函数，用于并行计算。
    输入元组: (sim_module_name, params, exclusive_config, priority_config, header_params)
    输出: 一行 CSV 字符串
    """
    sim_module_name, current_params, exclusive_config, priority_config, header_params = args_tuple
    sim = importlib.import_module(sim_module_name)

    base_lengths = {
        'forward': current_params['fwd_len'],
        'backward': current_params['bwd_len'],
        'fwd_prop': current_params['prop_len'],
        'grad_prop': current_params['prop_len'],
        'grad_sync': current_params['sync_base_len'],
        'param_sync': 1
    }

    # 根据独占等级更新任务独占配置和共享权重
    final_priorities = dict(sim.default_priorities)
    final_exclusive_tiers = dict(sim.default_exclusive_tiers)
    for task, tier in exclusive_config.items():
        final_exclusive_tiers[task] = tier
    # 根据优先级配置分配共享权重
    epsilon = 1e-9
    sharing_tasks = [t for t, tier in final_exclusive_tiers.items() if tier is None]
    if sharing_tasks:
        safe_priorities = {t: priority_config.get(t, 0) if priority_config.get(t, 0) > 0 else epsilon for t in sharing_tasks}
        total_safe_p = sum(safe_priorities.values())
        if total_safe_p > 1e-9:
            for task in sharing_tasks:
                final_priorities[task] = safe_priorities[task] / total_safe_p

    # 可选的接收端拥塞控制开关
    extra_kwargs = {}
    try:
        sig = inspect.signature(sim.calculate_full_pipeline_schedule)
        enable_cc = os.environ.get('ENABLE_RECV_CONGESTION', '0') == '1'
        if 'use_recv_congestion' in sig.parameters:
            # 仅对 1f1b 单通道有效
            extra_kwargs['use_recv_congestion'] = enable_cc
    except Exception:
        pass

    # 调用模拟器获取调度结果
    result = sim.calculate_full_pipeline_schedule(
        n=current_params['n'],
        num_gpus=4,
        is_ideal=False,
        lengths=base_lengths,
        priorities=final_priorities,
        exclusive_tiers=final_exclusive_tiers,
        fwd_impact=current_params['fwd_impact'],
        bwd_impact=current_params['bwd_impact'],
        sync_solo_w=1.0,
        sync_freq=current_params['sync_freq'],
        **extra_kwargs
    )
    # result 是三元组 (blocks, total_time, throttled_ids)
    blocks = result[0] if isinstance(result, tuple) else result
    metrics = compute_metrics(blocks)

    # 输出用的实际优先级值：如果是独占任务，用字母，否则用浮点数
    output_priorities = {}
    for task in ['fwd_prop', 'grad_prop', 'grad_sync']:
        if exclusive_config.get(task) is not None:
            output_priorities[task] = exclusive_config[task]
        else:
            output_priorities[task] = priority_config.get(task, 0.0)
    # 组合 CSV 行
    values = [current_params.get(h) for h in header_params]
    values.extend([
        output_priorities['fwd_prop'],
        output_priorities['grad_prop'],
        output_priorities['grad_sync'],
        f"{metrics['total_time']:.4f}",
        f"{metrics['compute_ratio']:.4f}",
        f"{metrics['comm_ratio']:.4f}",
        f"{metrics['avg_throughput']:.4f}",
        f"{metrics['avg_latency']:.4f}",
        f"{metrics['compute_utilization']:.4f}",
        f"{metrics['comm_utilization']:.4f}"
    ])
    return ','.join(map(str, values)) + '\n'


def load_param_grid(json_path: str) -> Dict[str, List]:
    """从 JSON 文件加载参数网格，返回字典"""
    with open(json_path, 'r') as f:
        return json.load(f)


def generate_metrics(simulator_module_name: str, output_file: str, param_grid: Dict[str, List], priority_step: float) -> None:
    """
    扫描所有参数组合和优先级组合，计算指标并写入 CSV。
    类似于 data_generator.generate_data_for_fitting，但包含更多指标。
    """
    tasks = ['fwd_prop', 'grad_prop', 'grad_sync']
    # 定义独占场景。仅考虑单独提升某一通信任务为独占。
    exclusive_scenarios = [
        {'fwd_prop': None, 'grad_prop': None, 'grad_sync': None},
        {'fwd_prop': 'a', 'grad_prop': None, 'grad_sync': None},
        {'fwd_prop': None, 'grad_prop': 'a', 'grad_sync': None},
    ]
    param_names = list(param_grid.keys())
    param_values = [generate_scan_values(param_grid[k]) for k in param_names]
    all_param_combos_raw = list(product(*param_values))
    # 过滤掉 sync_freq > n 的组合
    param_combinations = []
    for combo in all_param_combos_raw:
        param_dict = dict(zip(param_names, combo))
        if param_dict['sync_freq'] <= param_dict['n']:
            param_combinations.append(param_dict)
    all_jobs = []
    header_params = param_names
    for params in param_combinations:
        for exclusive_config in exclusive_scenarios:
            sharing_tasks = [t for t, tier in exclusive_config.items() if tier is None]
            num_sharing = len(sharing_tasks)
            priority_configs_for_scenario = []
            if num_sharing > 0:
                step_values = np.arange(0, 1.0 + priority_step, priority_step)
                for p_values in product(step_values, repeat=num_sharing):
                    if abs(sum(p_values) - 1.0) < 1e-9:
                        priority_config = {task: round(p, 4) for task, p in zip(sharing_tasks, p_values)}
                        priority_configs_for_scenario.append(priority_config)
            else:
                priority_configs_for_scenario.append({})
            for priority_config in priority_configs_for_scenario:
                full_priority_config = {t: 0.0 for t in tasks}
                full_priority_config.update(priority_config)
                all_jobs.append((simulator_module_name, params, exclusive_config, full_priority_config, header_params))
    total_runs = len(all_jobs)
    print(f"🔍 预计总指标计算次数: {total_runs}")
    # 执行并行计算
    try:
        results = []
        num_workers = max(1, os.cpu_count() - 1)
        print(f"🧮 使用 {num_workers} 个核心进行指标计算...")
        with multiprocessing.Pool(processes=num_workers) as pool:
            for line in pool.imap_unordered(run_single_simulation_metrics, all_jobs):
                results.append(line)
        # 按总时间升序排序
        results.sort(key=lambda l: float(l.strip().split(',')[len(header_params) + 3]))  # total_time 列索引
        # 写入文件
        with open(output_file, 'w') as f:
            header = header_params + [
                'p_fwd_prop', 'p_grad_prop', 'p_grad_sync',
                'total_time', 'compute_ratio', 'comm_ratio',
                'avg_throughput', 'avg_latency',
                'compute_utilization', 'comm_utilization'
            ]
            f.write(','.join(header) + '\n')
            for line in results:
                f.write(line)
        print(f"✅ 指标数据已保存到: {output_file}")
    except Exception as e:
        print(f"⚠️ 生成指标数据时出错: {e}")


def main():
    parser = argparse.ArgumentParser(description="基于流水线模拟生成通信/计算指标数据")
    parser.add_argument('--config', required=True, help='参数配置文件 (JSON)')
    parser.add_argument('--pipeline', choices=['1f1b', 'gpipe'], default='1f1b', help='选择流水线调度策略')
    parser.add_argument('--simulator', choices=['single_channel', 'dual_channel', 'duel_channel'], default='single_channel', help='通信通道类型')
    parser.add_argument('--step', type=float, default=0.1, help='共享优先级扫描步长')
    parser.add_argument('--output', type=str, default=None, help='输出 CSV 文件')
    parser.add_argument('--enable-recv-congestion', action='store_true', help='在 1f1b 单通道场景下启用接收端拥塞控制')
    args = parser.parse_args()
    # 处理 dual/duel 名称差异
    sim_name = args.simulator
    if sim_name == 'dual_channel':
        sim_name = 'duel_channel'
    sim_module_name = f'schedule_visualizer_{args.pipeline}_{sim_name}'
    # 检查模块是否存在
    try:
        importlib.import_module(sim_module_name)
    except ImportError as e:
        print(f"❌ 无法导入模拟器模块 {sim_module_name}: {e}")
        return
    # 加载参数网格
    param_grid = load_param_grid(args.config)
    # 构造输出文件名
    if args.output:
        output_file = args.output
    else:
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"outputs/data/{config_name}_metrics_{timestamp}.csv"
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # 设置环境变量供多进程访问
    os.environ['PIPELINE_TYPE'] = args.pipeline
    os.environ['ENABLE_RECV_CONGESTION'] = '1' if args.enable_recv_congestion else '0'
    generate_metrics(sim_module_name, output_file, param_grid, args.step)


if __name__ == '__main__':
    main()