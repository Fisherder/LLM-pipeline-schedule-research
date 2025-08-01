# 文件名: data_generator.py
# 描述: 支持从 JSON 配置文件读取参数，生成模拟数据集。

import numpy as np
import argparse
import importlib
from copy import deepcopy
from tqdm import tqdm
import sys
import json
from itertools import product
import multiprocessing
import os
import datetime

DEFAULT_PRIORITY_STEP = 0.1

def generate_scan_values(config):
    if isinstance(config, list):
        return config
    elif isinstance(config, dict):
        steps = int(round((config['stop'] - config['start']) / config.get('step', 1.0))) + 1
        return np.linspace(config['start'], config['stop'], steps)
    return [config]

def run_single_simulation(args_tuple):
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

    final_priorities = deepcopy(sim.default_priorities)
    final_exclusive_tiers = deepcopy(sim.default_exclusive_tiers)

    for task, tier in exclusive_config.items():
        final_exclusive_tiers[task] = tier

    epsilon = 1e-9
    sharing_tasks = [t for t, tier in final_exclusive_tiers.items() if tier is None]
    if sharing_tasks:
        safe_priorities = {t: priority_config.get(t, 0) if priority_config.get(t, 0) > 0 else epsilon for t in sharing_tasks}
        total_safe_p = sum(safe_priorities.values())
        if total_safe_p > 1e-9:
            for task in sharing_tasks:
                final_priorities[task] = safe_priorities[task] / total_safe_p

    _, total_time, _ = sim.calculate_full_pipeline_schedule(
        n=current_params['n'],
        num_gpus=4,
        is_ideal=False,
        lengths=base_lengths,
        priorities=final_priorities,
        exclusive_tiers=final_exclusive_tiers,
        fwd_impact=current_params['fwd_impact'],
        bwd_impact=current_params['bwd_impact'],
        sync_solo_w=1.0,
        sync_freq=current_params['sync_freq']
    )

    output_priorities = {}
    for task in ['fwd_prop', 'grad_prop', 'grad_sync']:
        if exclusive_config.get(task) is not None:
            output_priorities[task] = exclusive_config[task]
        else:
            output_priorities[task] = priority_config.get(task, 0.0)

    data_values = [current_params.get(h) for h in header_params]
    data_values.extend([
        output_priorities['fwd_prop'],
        output_priorities['grad_prop'],
        output_priorities['grad_sync'],
        f"{total_time:.4f}"
    ])
    return ','.join(map(str, data_values)) + '\n'

def load_param_grid_from_json(json_path):
    with open(json_path, 'r') as f:
        config = json.load(f)
    return config

def generate_data_for_fitting(simulator_module_name, output_file, param_grid, priority_step):
    tasks = ['fwd_prop', 'grad_prop', 'grad_sync']

    exclusive_scenarios = [
        {'fwd_prop': None, 'grad_prop': None, 'grad_sync': None},
        {'fwd_prop': 'a', 'grad_prop': None, 'grad_sync': None},
        {'fwd_prop': None, 'grad_prop': 'a', 'grad_sync': None},
    ]

    param_names = list(param_grid.keys())
    param_values = [generate_scan_values(param_grid[k]) for k in param_names]
    all_param_combos_raw = list(product(*param_values))

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
    print(f"总共需要进行 {len(param_combinations)} 组环境参数测试。")
    print(f"总共生成了 {len(exclusive_scenarios)} 种独占场景。")
    print(f"预计总模拟次数: {total_runs}")

    try:
        all_results = []
        num_workers = max(1, os.cpu_count() - 1)
        print(f"使用 {num_workers} 个核心并行计算 (总核心数: {os.cpu_count()})...")
        with multiprocessing.Pool(processes=num_workers) as pool:
            results_iterator = pool.imap_unordered(run_single_simulation, all_jobs)
            for result_line in tqdm(results_iterator, total=total_runs, desc="正在收集数据"):
                all_results.append(result_line)

        print("\n所有模拟已完成，正在排序数据...")
        all_results.sort(key=lambda line: float(line.strip().split(',')[-1]))

        print(f"正在将排序后的数据写入到: {output_file}")
        with open(output_file, 'w') as f:
            header_line = ','.join(header_params + ['p_fwd_prop', 'p_grad_prop', 'p_grad_sync', 'total_time'])
            f.write(header_line + '\n')
            for result_line in tqdm(all_results, desc="正在写入文件"):
                f.write(result_line)

        print(f"\n数据生成完成！结果已保存到: {output_file}")

    except IOError as e:
        print(f"\n错误: 无法写入文件 '{output_file}'。请检查路径和权限。")
        print(f"详细信息: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='为流水线模拟器生成拟合数据。')
    parser.add_argument('--simulator', type=str, choices=['single_channel', 'dual_channel'], default='single_channel')
    parser.add_argument('--step', type=float, default=DEFAULT_PRIORITY_STEP)
    parser.add_argument('--config', type=str, required=True, help="JSON 格式的参数配置文件路径")
    parser.add_argument('--output', type=str, default=None, help='输出CSV文件路径 (默认自动命名)')

    args = parser.parse_args()

    try:
        sim_module_name = f'schedule_visualizer_{args.simulator}'
        importlib.import_module(sim_module_name)
        print(f"# --- 使用模拟器: {sim_module_name}.py ---")
        print(f"# --- 优先级扫描步长: {args.step} ---")
    except ImportError as e:
        print(f"错误: 无法导入模拟器模块 '{e.name}'")
        exit(1)

    param_grid = load_param_grid_from_json(args.config)

    if args.output:
        output_file = args.output
    else:
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"outputs/data/{config_name}_{timestamp}.csv"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    generate_data_for_fitting(sim_module_name, output_file, param_grid, args.step)
