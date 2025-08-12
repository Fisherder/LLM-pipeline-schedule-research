# 大模型流水线调度模拟器 - 综合使用文档

## 第1章：项目概述

本项目是一个用于模拟、分析与可视化大规模语言模型训练中两类典型流水线调度策略的综合工具链：

- 1F1B（One-Forward-One-Backward，前后交替）
- GPipe（先全部前向，再全部反向，F-then-B）

通过高精度、事件驱动的模拟，项目精确复现多 GPU 环境下的计算与通信调度，揭示性能瓶颈，并为调度策略设计提供量化数据与直观洞察。工具链覆盖从参数配置、批量数据生成、指标提取，到多种可视化（甘特图、热力图、网络负载曲线）的完整流程。

### 1.1 工具链构成

- 核心模拟器（Simulators）
  - 1F1B：
    - `schedule_visualizer_1f1b_single_channel.py`
    - `schedule_visualizer_1f1b_duel_channel.py`（文件名历史拼写为“duel”，命令行参数同时兼容 `dual/duel`）
    - `schedule_visualizer_1f1b_ideal.py`（理想、互不争用的通道模型）
  - GPipe：
    - `schedule_visualizer_gpipe_single_channel.py`
    - `schedule_visualizer_gpipe_duel_channel.py`
    - `schedule_visualizer_gpipe_ideal.py`
  - 说明：single_channel 与 duel/dual_channel 的差异主要体现在“可视化布局”，底层通信仍在同一物理通道上竞争资源；ideal 版本用于对照，不建模资源争用与计算降速影响。

- 数据生成（Generators）
  - `data_generator.py`：扫描共享/独占优先级组合，批量生成“总完成时间”CSV。
  - `generate_metrics.py`：在上述基础上进一步计算通信/计算比、吞吐、延迟、资源利用率等指标，输出更丰富的指标 CSV。

- 可视化（Visualizers）
  - `plot_heatmap.py`：读取 `data_generator.py` 输出的 CSV，按配置分组批量绘制“总完成时间”热力图。
  - `plot_metrics_heatmaps.py`：读取 `generate_metrics.py` 的指标 CSV，为多种指标（如 `total_time`、`avg_latency`、`compute_utilization` 等）分别生成热力图，并以红色标注最优格。
  - `make_network_load_plots.py`：直接调用单通道模拟器，统计并绘制通信带宽随时间的占用曲线（总占用与分任务占用）。

- 自动化编排（Orchestrator）
  - `run_full_pipeline.py`：推荐的主入口脚本，一键完成“数据生成 -> 热力图绘制”。

### 1.2 功能亮点

- 高精度模拟：精确建模计算/通信资源竞争、跨 GPU 数据依赖、网络拥塞反压、主动独占与共享优先级等。
- 高度可配置：任务长度、网络干扰、梯度累积频率以及优先级策略均可通过 JSON 配置灵活调整。
- 高性能计算：多进程并行生成数据，充分利用多核 CPU，大幅加速大规模实验。
- 自动化与批量处理：从参数配置到批量出图的全流程自动化。
- 信息丰富的可视化：甘特图、热力图、网络负载等多视角展示调度行为与性能结果。

## 第2章：核心概念

- 1F1B：流水线填满后，前向与反向在各阶段交替执行，减少“气泡”（空闲时间）。
- 资源通道模型：每个 GPU 拥有计算与通信两个物理通道；计算通道独占、通信通道共享。
- 优先级体系：
  - 主动独占（Active Exclusive）：在 `default_exclusive_tiers` 中以字母分级（'a' > 'b' > ...），就绪即独占带宽。
  - 普通共享（Shared）：在 `default_priorities` 中用 0~1 权重，按比例共享带宽。
  - 零优先级：仅当无独占且无共享任务时才均分带宽。
- 网络效应建模：
  - `fwd_impact` / `bwd_impact`：梯度同步对计算效率的影响系数。
  - 接收端拥塞控制：当入向流量超载时，对发送端带宽进行反压与再分配（1F1B 单通道脚本支持开关）。

## 第3章：标准工作流程（推荐）

### 步骤 1：准备 JSON 配置

示例 `configs/config_1.json` 结构：

```
{
  "n": [8],
  "sync_freq": [2],
  "fwd_len": [1.0],
  "bwd_len": { "start": 2.0, "stop": 2.0, "step": 0.5 },
  "prop_len": { "start": 1.0, "stop": 1.0, "step": 0.5 },
  "sync_base_len": { "start": 2.0, "stop": 2.0, "step": 0.5 },
  "fwd_impact": { "start": 0.2, "stop": 0.2, "step": 0.1 },
  "bwd_impact": { "start": 0.2, "stop": 0.2, "step": 0.1 }
}
```

### 步骤 2：一键执行全流程

```
python run_full_pipeline.py -c configs/config_1.json \
  --pipeline 1f1b --simulator single_channel \
  --priority_step 0.1 [--enable-recv-congestion]
```

- `--pipeline {1f1b,gpipe}`：选择调度策略（默认 1f1b）。
- `--simulator {single_channel,dual_channel}`：选择可视化布局（兼容 `duel_channel`）。
- `--priority_step`：共享优先级扫描步长（默认 0.1）。
- `--enable-recv-congestion`：启用接收端拥塞控制（仅对 1F1B 单通道有效）。

脚本将自动：

1. 并行调用模拟器批量生成 CSV（`outputs/data/*.csv`）。
2. 按配置分组绘制“总完成时间”热力图（`outputs/heatmaps/*.png`）。

### 步骤 3：查看结果

- `outputs/data/*.csv`：排序后的模拟结果表。
- `outputs/heatmaps/*.png`：每个配置一张热力图。

## 第4章：高级用法 & 脚本独立解析

### 4.1 `run_full_pipeline.py`（自动化编排）

```
python run_full_pipeline.py -c <配置文件> \
  [--pipeline {1f1b,gpipe}] \
  [--simulator {single_channel,dual_channel}] \
  [--priority_step 0.1] [--enable-recv-congestion]
```

### 4.2 `data_generator.py`（数据生成：总时间）

```
python data_generator.py --config <配置文件> \
  [--pipeline {1f1b,gpipe}] \
  [--simulator {single_channel,dual_channel}] \
  [--step 0.1] [--output <csv路径>] [--enable-recv-congestion]
```

输出 CSV 列：`n,sync_freq,fwd_len,bwd_len,prop_len,sync_base_len,fwd_impact,bwd_impact,p_fwd_prop,p_grad_prop,p_grad_sync,total_time`

### 4.3 `plot_heatmap.py`（总时间热力图）

```
python plot_heatmap.py <csv路径> [-o <输出目录>]
```

默认将每个配置绘制为一张热力图，保留独占等级 'a'，对数值做格式化排序。

### 4.4 `schedule_visualizer_*.py`（核心模拟器：单次可视化）

各脚本顶部定义默认参数：

- `default_lengths`（任务工作量）
- `default_priorities`（共享权重）
- `default_exclusive_tiers`（主动独占等级，'a' > 'b' > ...，`None` 表示共享）

示例：

- 1F1B 单通道：
  - `python schedule_visualizer_1f1b_single_channel.py -n 8 --sync-freq 2 --show-throttling [--enable-recv-congestion]`
- 1F1B 双通道可视化：
  - `python schedule_visualizer_1f1b_duel_channel.py -n 8 --sync-freq 2`
- 1F1B 理想模型：
  - `python schedule_visualizer_1f1b_ideal.py -n 8 --sync-freq 2`
- GPipe 单通道：
  - `python schedule_visualizer_gpipe_single_channel.py -n 8 --sync-freq 2`
- GPipe 双通道可视化：
  - `python schedule_visualizer_gpipe_duel_channel.py -n 8 --sync-freq 2`
- GPipe 理想模型：
  - `python schedule_visualizer_gpipe_ideal.py -n 8 --sync-freq 2`

提示：单/双通道脚本的差异主要在可视化布局；带宽分配与拥塞控制的核心逻辑保持一致。

### 4.5 `generate_metrics.py`（多指标数据）

```
python generate_metrics.py --config <配置文件> \
  [--pipeline {1f1b,gpipe}] \
  [--simulator {single_channel,dual_channel}] \
  [--step 0.1] [--output <csv路径>] [--enable-recv-congestion]
```

新增指标列：`compute_ratio,comm_ratio,avg_throughput,avg_latency,compute_utilization,comm_utilization`

### 4.6 `plot_metrics_heatmaps.py`（多指标热力图）

```
python plot_metrics_heatmaps.py --csv <一个或多个csv> \
  [--metrics total_time avg_latency compute_utilization ...] \
  [--outdir <输出目录>] [--vmin <下限>] [--vmax <上限>] [--show]
```

未显式指定 `--outdir` 时，默认按照 CSV 名称生成 `./outputs/metrics_<csv名>/` 目录保存热力图。

### 4.7 `make_network_load_plots.py`（网络负载曲线）

```
python make_network_load_plots.py --mode {gpipe,1f1b,both} \
  -n 8 --sync-freq 2 [--enable-recv-congestion]
```

输出：`network_load_gpipe_single.png`、`network_load_1f1b_single_no_congestion.png`/`_with_congestion.png`

## 第5章：联动方式与代码结构

典型联动流程：

run_full_pipeline.py（起点）

-> 调用 data_generator.py（并行生成 CSV）

-> 导入并调用 schedule_visualizer_{1f1b|gpipe}_{single|duel}.py 的 `calculate_full_pipeline_schedule`

-> 输出 `outputs/data/<config>_<timestamp>.csv`

-> 调用 plot_heatmap.py（按配置分组绘制 total_time 热力图）

-> 输出 `outputs/heatmaps/heatmap_*.png`

可选链路：

- generate_metrics.py -> 输出多指标 CSV -> plot_metrics_heatmaps.py -> 输出指标热力图
- make_network_load_plots.py -> 输出网络负载曲线 PNG

命名说明：仓库中 duel/dual 的差异仅为历史拼写问题；CLI 参数与脚本均已兼容。
