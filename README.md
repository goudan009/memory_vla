# VLA Memory

**π0.5 + 记忆模块**研究项目。代码基座 = SERF 的 OpenPI/π0.5 实现，加入动作子空间条件记忆，先单任务跑通，再扩到 3-5 个任务。

训练期记忆写入（JIT 外 episode/timestep 传递）尚未闭环；`src/vlamemory` 已有核心实现和 OpenPI 集成快照，可做 CPU 单测。

## 目录

```
src/vlamemory/          记忆核心包（已实现）
  memory/               记忆存储、均匀检索、动作子空间条件检索
  modulation.py         检索结果注入 prefix 的接口
  data/                 历史帧采样、episode sampler、dataloader
  config.py             MemoryConfig 配置
  integrations/openpi/  Pi0 / Pi0Config 接入快照（依赖外部 openpi）
src/references/         第三方参考源码（memoryvla / mem / serf），只读
src/simulation/         OmniGibson / joylo 入口（外部软链）
configs/                实验配置（目前仅 evaluation/base_config.yaml）
scripts/                执行入口：b1k_python.sh / evaluate_b1k.sh / check_b1k_sim.py 等
evaluation/             serf_rgb 评估器（eval.py / eval_metrics.py）
baselines/              对照方法适配代码（目前无实现）
checkpoints/            权重入口，behavior_50t_checkpoint 软链指向 SERF-VLA 原件
datasets/               数据入口，通过软链引用外部存储（无真实数据提交）
env/                    运行环境说明（见 env/README.md）
paper/                  论文稿（paper.md）和参考论文 PDF
tests/                  测试用例（待补充）
train/                  训练脚本（待实现）
outputs/                运行日志与 agent 产物（gitignored）
```

## 安装（开发模式）

```bash
# 在 /nvme3/envs/b1k2025-policy 环境里
pip install -e . --no-deps   # 仅装 vlamemory 包本体
# 集成 OpenPI：
pip install -e /nvme3/projects/vla_memory/code/openpi --no-deps
```

`pyproject.toml` 的 `[tool.setuptools.packages.find]` 只打包 `src/vlamemory`，
`src/references` 和 `src/simulation` 不会被装入发行包。

## 两套运行环境

详见 `env/README.md`。

- **环境 1（4090）**：仿真 + 推理，Conda `b1k2025-sim`（Isaac Sim 4.5 + OmniGibson 3.7.1）
- **环境 2（A100）**：模型训练，`/home/1024201092WYH/.venvs/lerobot_pi05`（LeRobot 0.4.3 + torch 2.7.1+cu126）；备选 `/nvme3/projects/vla_memory/code/openpi/.venv`（JAX/flax/openpi）

## 资产路径速查

| 资产 | 实际位置 |
|---|---|
| 50 任务底座权重 | `checkpoints/behavior_50t_checkpoint` → `/nvme3/projects/SERF-VLA/checkpoints/behavior-1k-solution/behavior_50t_checkpoint` |
| 2025 演示数据（parquet） | `datasets/b1k2025/parquet` → `/nvme3/projects/SERF-VLA/datasets/2025-BEHAVIOR-1K-CHALLENGE/data` |
| 仿真资产 | `datasets/simulation/behavior-1k-assets` → `/nvme3/projects/SERF-VLA/datasets/behavior-1k-assets` |
| OmniGibson 源码 | `src/simulation/omnigibson` → `/nvme3/shared/b1k2025/BEHAVIOR-1K/OmniGibson` |

## 状态

- CPU 单测：`src/vlamemory` 核心包已通过（数据管线、记忆模块、前向 smoke）
- 评估入口：`sh scripts/evaluate_b1k.sh` → `evaluation/serf_rgb/eval.py`（serf_rgb）
- 仿真 smoke：`turning_on_radio` 200 步 RGB 正常（输出 `outputs/b1k2025_smoke_20261104/`）
- 训练端到端：**尚未跑通**（JIT 外记忆写入、数据配方与权重路径需对齐）
