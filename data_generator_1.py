"""
自定义数据生成脚本。

该脚本支持从 JSON 配置文件读取模拟参数并批量运行流水线模拟。与原始实现不同，
这里将前向 (fwd) 和反向 (bwd) 梯度同步对计算的影响统一为单一参数 `impact`。
如果配置文件中依然提供了 `fwd_impact` 或 `bwd_impact`，脚本会将它们合并为 `impact`
的取值列表（取并集）。这样保证在扫描参数时，前向和反向计算的影响始终一致。
"""

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

# 默认共享优先级扫描步长
DEFAULT_PRIORITY_STEP = 0.1

def generate_scan_values(config):
    """
    根据配置生成扫描值列表。
    - 如果 config 是 list，则直接返回该列表；
    - 如果 config 是 dict，则视为区间设置，支持 {'start': x, 'stop': y, 'step': s}；
    - 其他情况则返回单元素列表。
    """
    if isinstance(config, list):
        return config
    elif isinstance(config, dict):
        steps = int(round((config['stop'] - config['start']) / config.get('step', 1.0))) + 1
        return np.linspace(config['start'], config['stop'], steps)
    return [config]

def run_single_simulation(args_tuple):
    """
    执行单次模拟任务。

    参数 args_tuple 包含：
    - sim_module_name: 模拟器模块名
    - current_params: 当前环境参数组合
    - exclusive_config: 独占场景配置
    - priority_config: 共享任务优先级配置
    - header_params: 要输出的参数列顺序
    """
    sim_module_name, current_params, exclusive_config, priority_config, header_params = args_tuple
    sim = importlib.import_module(sim_module_name)

    # 基本任务长度映射
    base_lengths = {
        'forward': current_params['fwd_len'],
        'backward': current_params['bwd_len'],
        'fwd_prop': current_params['prop_len'],
        'grad_prop': current_params['prop_len'],
        'grad_sync': current_params['sync_base_len'],
        'param_sync': 1
    }

    # 初始化优先级和独占等级
    final_priorities = deepcopy(sim.default_priorities)
    final_exclusive_tiers = deepcopy(sim.default_exclusive_tiers)

    for task, tier in exclusive_config.items():
        final_exclusive_tiers[task] = tier

    # 归一化共享优先级使其和为 1
    epsilon = 1e-9
    sharing_tasks = [t for t, tier in final_exclusive_tiers.items() if tier is None]
    if sharing_tasks:
        safe_priorities = {t: priority_config.get(t, 0) if priority_config.get(t, 0) > 0 else epsilon for t in sharing_tasks}
        total_safe_p = sum(safe_priorities.values())
        if total_safe_p > 1e-9:
            for task in sharing_tasks:
                final_priorities[task] = safe_priorities[task] / total_safe_p

    # 动态传递接收端拥塞开关
    extra_kwargs = {}
    try:
        import inspect
        sig = inspect.signature(sim.calculate_full_pipeline_schedule)
        pipeline_type = os.environ.get('PIPELINE_TYPE', '1f1b')
        enable_cc = os.environ.get('ENABLE_RECV_CONGESTION', '0') == '1'
        if 'use_recv_congestion' in sig.parameters:
            extra_kwargs['use_recv_congestion'] = enable_cc
    except Exception:
        pass

    # 从当前参数中获取 impact；若不存在则尝试旧字段 fwd_impact/bwd_impact
    impact_val = current_params.get('impact')
    if impact_val is None:
        if 'fwd_impact' in current_params:
            impact_val = current_params['fwd_impact']
        elif 'bwd_impact' in current_params:
            impact_val = current_params['bwd_impact']
        else:
            impact_val = 0.0

    # 调用模拟器计算调度
    _, total_time, _ = sim.calculate_full_pipeline_schedule(
        n=current_params['n'],
        num_gpus=4,
        is_ideal=False,
        lengths=base_lengths,
        priorities=final_priorities,
        exclusive_tiers=final_exclusive_tiers,
        impact=impact_val,
        sync_solo_w=1.0,
        sync_freq=current_params['sync_freq'],
        **extra_kwargs
    )

    # 构建输出优先级（保留独占等级字母值）
    output_priorities = {}
    for task in ['fwd_prop', 'grad_prop', 'grad_sync']:
        if exclusive_config.get(task) is not None:
            output_priorities[task] = exclusive_config[task]
        else:
            output_priorities[task] = priority_config.get(task, 0.0)

    # 组装输出数据
    data_values = [current_params.get(h) for h in header_params]
    data_values.extend([
        output_priorities['fwd_prop'],
        output_priorities['grad_prop'],
        output_priorities['grad_sync'],
        f"{total_time:.4f}"
    ])
    return ','.join(map(str, data_values)) + '\n'

def load_param_grid_from_json(json_path):
    """读取 JSON 配置并将可能的 fwd/bwd_impact 合并为 impact。"""
    with open(json_path, 'r') as f:
        config = json.load(f)
    # 合并前向和反向影响
    # 如果配置中已经包含 impact，则直接使用 impact；否则检查 fwd_impact 和 bwd_impact
    if 'impact' not in config:
        fwd = config.pop('fwd_impact', None)
        bwd = config.pop('bwd_impact', None)
        unify_values = []
        if fwd is not None:
            unify_values.extend(generate_scan_values(fwd))
        if bwd is not None:
            unify_values.extend(generate_scan_values(bwd))
        # 去重并排序
        if unify_values:
            uniq = sorted(set(float(x) for x in unify_values))
            # 将 unique 值列表直接作为 impact 的扫描列表
            config['impact'] = uniq
    return config

def generate_data_for_fitting(simulator_module_name, output_file, param_grid, priority_step):
    """
    遍历参数网格和独占场景组合，生成模拟数据并输出 CSV。优先级扫描按照给定步长生成等和组合。
    """
    tasks = ['fwd_prop', 'grad_prop', 'grad_sync']

    # 支持三种独占配置：无独占、fwd 独占、grad 独占
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
        # 过滤无效组合：sync_freq 不能大于 n
        if param_dict['sync_freq'] <= param_dict['n']:
            param_combinations.append(param_dict)

    all_jobs = []
    header_params = param_names  # 记录要输出的环境参数列顺序

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
            # header_params 顺序 + 三个优先级列 + total_time
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
    parser.add_argument('--simulator', type=str, choices=['single_channel', 'dual_channel', 'duel_channel'], default='single_channel',
                        help='通信通道类型: single_channel 表示单通道, dual/duel_channel 表示双通道')
    parser.add_argument('--pipeline', type=str, choices=['1f1b', 'gpipe'], default='1f1b',
                        help='流水线调度策略: 1f1b 或 gpipe')
    parser.add_argument('--step', type=float, default=DEFAULT_PRIORITY_STEP,
                        help='共享优先级扫描步长 (默认 0.1)')
    parser.add_argument('--config', type=str, required=True, help="JSON 格式的参数配置文件路径")
    parser.add_argument('--output', type=str, default=None, help='输出CSV文件路径 (默认自动命名)')
    parser.add_argument('--enable-recv-congestion', action='store_true',
                        help='开启接收端拥塞控制 (仅对 1f1b 单通道脚本有效)')

    args = parser.parse_args()

    # 处理 dual/duel 名称差异，统一映射到实际文件名拼写
    simulator_name = args.simulator
    if simulator_name == 'dual_channel':
        simulator_name = 'duel_channel'

    # 构建模拟器模块名，例如 schedule_visualizer_1f1b_single_channel 或 schedule_visualizer_gpipe_duel_channel
    sim_module_name = f'schedule_visualizer_{args.pipeline}_{simulator_name}'
    try:
        importlib.import_module(sim_module_name)
        print(f"# --- 使用模拟器: {sim_module_name}.py ---")
        print(f"# --- 优先级扫描步长: {args.step} ---")
    except ImportError as e:
        print(f"错误: 无法导入模拟器模块 '{e.name}'. 请检查 --pipeline 和 --simulator 参数是否正确。")
        exit(1)

    # 读取配置并合并 impact 参数
    param_grid = load_param_grid_from_json(args.config)

    # 生成输出文件名
    if args.output:
        output_file = args.output
    else:
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"outputs/data/{config_name}_{timestamp}.csv"

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 将接收端拥塞控制选项打包到环境变量中供 run_single_simulation 检索
    os.environ['PIPELINE_TYPE'] = args.pipeline
    os.environ['ENABLE_RECV_CONGESTION'] = '1' if args.enable_recv_congestion else '0'
    generate_data_for_fitting(sim_module_name, output_file, param_grid, args.step)