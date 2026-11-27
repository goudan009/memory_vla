# VLA Memory

基于 **GR00T N1.6** 加入 MemoryVLA 的历史记忆方法，在 **B1K-2025** 上训练和评估。v1 首先验证普通记忆是否提供可重复收益，或能通过历史干预确认它在特定决策中的作用。

从 [实验进展](docs/research_progress.md) 可查当前研究状态和对应源码；各实验的任务书和归档报告见[实验文档索引](Output/README.md)。

## 文档导航

| 要做什么 | 从这里开始 |
| --- | --- |
| 启动或查询训练、推理、评估 | [工作台执行手册](docs/workbench/agent.md)、[CLI](workbench-cli/README.md)、[设备说明](train/DEVICES.md) |
| 配环境、跑评估、看结果 | [运行环境](env/README.md)、[评估使用说明](evaluation/README.md)、[正式评估协议](evaluation/evaluation.md) |
| 查基线和当前实现 | [GR00T 基线](baselines/groot_n1d6/README.md)、[V1](src/v1_groot_memoryvla/README.md)、[记忆融合源码](src/v1_groot_memoryvla/model/memory/fusion.py) |
| 看架构和设计依据 | [文档索引](docs/README.md)、[V1 架构与设计依据](docs/v1/README.md)、[V2_1 方案](docs/v2/2.1.html)、[工作台设计](docs/workbench/README.md) |
| V2_1数据、机器人拓扑与Task40状态标签 | [开发前调查](v2_1_preflight/README.md)、[最新标签扩展](v2_1_preflight/label_extension/README.md) |
| 查论文稿、旧诊断和命令 | [论文与研究笔记](paper/README.md)、[历史归档](docs/archive/README.md) |

版本专用说明随源码放在 `src/<版本>/`；当前协作规则见 [AGENTS.md](AGENTS.md)。实时作业和设备状态以工作台为准。

## 项目架构

采用“版本内维护模型与训练，版本间共用评估”的布局。v1 直接沿用 GR00T 的模块划分，后续记忆代码在对应模块内增加；不再额外嵌套一层 `gr00t/`。

```text
memory_vla/
├── README.md                             整体架构、职责与入口
├── pyproject.toml                        统一源码包发现及资源打包
├── src/
│   ├── references_code/                  本地上游参考，不纳入 Git
│   │   ├── Isaac-GR00T/                  官方 N1.6，固定提交，保留 Git
│   │   ├── memoryvla/                    MemoryVLA 方法参考
│   │   ├── mem/
│   │   └── serf/
│   └── v1_groot_memoryvla/               v1 的完整开发代码
│       ├── README.md / handoff.md / LICENSE
│       ├── train.py                      本版训练入口
│       ├── serve.py                      本版模型服务入口
│       ├── configs/                      模型、数据与训练配置定义
│       │   └── experiments/              本版具体实验参数的存放约定
│       ├── model/                        GR00T 模型及双路记忆融合
│       ├── data/                         真实 B1K 转换、历史组 Dataset / Collator
│       ├── experiment/                   Trainer、训练、保存和加载
│       ├── policy/                       推理、独立历史状态与 Episode Reset
│       ├── utils/                        本版内部工具
│       ├── env/                          本版依赖清单、环境说明与上游留档
│       └── tests/                        本版单元及模型集成测试
├── baselines/
│   └── groot_n1d6/                       原版基线的训练/服务入口与环境说明
├── evaluation/                           基线与各版本共用
│   ├── README.md                         正式评估使用说明
│   ├── evaluation.md                     固定评估协议、指标与配对规则
│   ├── configs/                          公共评估配置的职责和现有参数说明
│   ├── targets/                          源码版本、checkpoint、服务和结果映射约定
│   ├── scripts/                          正式启动、续跑、监控和结果查看
│   ├── sim/                              共用仿真适配、Gym 注册、录像 wrapper
│   ├── rollout_policy.py                 上游通用 rollout，已集中到此
│   ├── open_loop_eval.py                 上游离线动作评估
│   ├── real_robot/                       上游真机评估示例
│   └── V_*/                             基线与实验结果，不纳入 Git
├── tests/                                公共评估、协议和工作台测试
├── env/                                  公共仿真环境说明及已有环境软链接
├── scripts/                              公共资源工具与旧命令兼容入口
├── assets/                               共用资源入口，不纳入 Git
│   ├── simulation/                       场景、物体、机器人、任务实例和密钥
│   └── datasets/b1k2025/                  演示数据：meta、data、videos
├── checkpoints/                          按版本/运行组织模型权重
├── outputs/                              按版本/运行组织训练日志等产物
├── third_party/Isaac-GR00T/               已有运行快照，含两处延迟导入修改
├── train/                                训练网页后端、源码同步与公共训练入口
├── workbench-cli/                        工作台命令行工具
├── docs/                                 工作台手册、架构与历史归档
├── configs/                              旧实验/旧仿真配置，保留兼容
└── paper/                                论文草稿与研究笔记
```

## 哪些归版本，哪些共用

| 内容 | 存放位置 | 划分依据 |
| --- | --- | --- |
| 模型、记忆、历史采样、processor、Trainer、在线 reset | `src/<版本>/` | 会改变模型行为，随版本演进 |
| 训练入口、训练参数、消融参数、依赖版本、模型测试 | `src/<版本>/{train.py,configs,env,tests}` | 需要独立训练与复现 |
| 模型服务 | `src/<版本>/serve.py` | 加载本版代码与权重，对外遵守同一策略协议 |
| 原版基线 | `baselines/groot_n1d6/` + 官方参考目录 | 使用固定的原版代码，独立进程运行 |
| 仿真、观测/动作接入、任务、实例、seed、评分与录像 | `evaluation/` | 各版本必须遵守同一评估条件 |
| 被评模型及服务地址记录 | `evaluation/targets/` 的约定 | 只描述评估对象，不包含模型实现 |
| 仿真环境、资产/数据、初始预训练权重 | 根目录 `env/`、`assets/`、`checkpoints/` | 共用大资源，避免复制 |
| 训练产物 | `checkpoints/<版本>/<运行>/`、`outputs/<版本>/<运行>/` | 共用存储入口，按版本隔离内容 |
| 评估产物 | 如 `evaluation/V_2/memory/task22_eval03/` | 共用结果格式、汇总和查看工具；命名见 [规则](evaluation/NAMING.md) |

根 `pyproject.toml` 统一发现源码包；版本专用依赖见各版 README。公共仿真进程与各版模型服务使用各自 Python 环境，通过 Policy Server 协议连接。模型加载涉及同名 Transformers 注册，原版与 v1 使用独立服务进程。

训练和依赖清单使用真实文件，放在对应版本内；无需给根目录再加训练或依赖软链接。软链接用于大资源和外部运行环境，例如 `env/b1k2025-sim`。当前 v1 独立模型环境为 `src/v1_groot_memoryvla/.venv/`。

根 `train/` 和 `configs/` 维护公共调度、设备配置和兼容入口；版本专用训练配置留在各版。不要直接移动已有评估结果。

## 资产与数据路径

所有资源从 `assets/` 进入，不再保留顶层 `datasets/` 或重复的 `simulation/b1k2025` 入口：

```text
assets/
├── simulation/
│   ├── behavior-1k-assets/             场景、物体，固定版本 3.7.2rc1
│   ├── omnigibson-robot-assets/        机器人资产
│   ├── 2025-challenge-task-instances/  任务实例及 metadata
│   ├── 2025-challenge-hidden-instances/ 新补齐46项任务的官方隐藏测试状态
│   └── omnigibson.key                 已有密钥入口，不提交 Git
└── datasets/
    └── b1k2025/
        ├── meta/                      演示元数据
        ├── data/                      演示 parquet，原入口名为 parquet
        └── videos/                    演示图像视频
```

这些入口链接到外部大资源目录，不复制大文件。当前演示数据链接保留原目标；仿真场景与机器人资产分别指向 `/nvme3/projects/SERF-VLA/datasets/behavior-1k-assets`、`/nvme3/projects/SERF-VLA/datasets/omnigibson-robot-assets`，任务实例指向 `/nvme3/shared/b1k2025/datasets/2025-challenge-task-instances`。2026-11-09 恢复后已确认场景版本为 `3.7.2rc1`，Task21/22/26 元数据和各 10 个隐藏测试实例可读取；Task21 单环境实际完成 256 步调试窗口，双环境显存限制见 [评测说明](evaluation/README.md)。正式评测仍依赖固定 GR00T 源码树中的隐藏测试实例，保持原路径。

2026-11-10 补齐其余46项 B1K-2025 任务的460个官方隐藏测试状态，存于 `/data/datasets/b1k2025/2025-challenge-hidden-instances`，通过上面的 simulation 入口链接；固定 GR00T 的缺失 `test_instances/<task>` 目录再链接到该入口。已有 Task0/21/22/26 四项目录保留原样。当前全部50项任务均有0～9号测试实例，用 `./evaluation/scripts/run_groot_eval.sh --list-tasks` 查看；实际新任务短窗验收范围见评测说明。

仿真默认读取本项目的 `assets/simulation/`。评测启动器仅通过显式设置 `BEHAVIOR_DATA` 覆盖资产根目录，忽略其他项目遗留的 `OMNIGIBSON_DATA_PATH`；单独调用 `scripts/b1k_python.sh` 时仍可通过 `OMNIGIBSON_DATA_PATH` 指定资源。资源根目录应直接包含上述三个资产目录及密钥。下载时可放在项目外的大容量目录，完成后将对应入口链接到实际解压目录。

## V1.2 / GR00T N1.7 无记忆基线

独立迁移架构与本地验收见 [N1.7 Baseline](docs/v1_2/n17-baseline.md)。当前仅建立源码骨架，尚未开放正式训练入口；不影响 V2.1 的原有运行。

## 安装与训练入口

命令均从项目根目录执行。在已具备依赖的独立模型环境中安装源码：

```bash
python -m pip install -e . --no-deps

# v1 原生训练和模型服务的参数
python -m v1_groot_memoryvla.train --help
python -m v1_groot_memoryvla.serve --help

# 原版基线入口：使用 baseline 自己的模型环境
python baselines/groot_n1d6/train.py --help
python baselines/groot_n1d6/serve.py --help
```

v1 依赖、CUDA wheel 来源、固定上游版本、模型结构和训练命令见 [v1 README](src/v1_groot_memoryvla/README.md)，进度、验收与待办见 [handoff.md](src/v1_groot_memoryvla/handoff.md)。v1 的单 GPU 环境不依赖 DeepSpeed；原有模型和仿真环境保持独立。

## 多版本共用评估

新训练按 `V_2_memory_task22_train02` 命名，新评估按 `evaluation/V_2/memory/task22_eval03/` 命名。完整规则、源码/权重关联和后端脚本职责见 [命名与版本管理](evaluation/NAMING.md)。下方旧分组命令仍兼容；新实验推荐使用 `--group memory` 或具体 `ablation_说明`。

先在模型侧启动所需版本及 checkpoint，再用同一套评估脚本指定其服务地址：

```bash
# 原有 baseline 默认入口
./evaluation/scripts/run_groot_eval.sh

# V_1 的 experment_1 分组；服务必须已加载本版 checkpoint
./evaluation/scripts/run_groot_eval.sh --version V_1 --group experment_1 \
  --model-name grootmeory --policy-host MODEL_HOST --policy-port 5557 \
  --task 22 --seed 0 --run-tag v1_memory

# 查看全部结果
./evaluation/scripts/view_results.sh --once
```

`--version` 选择版本目录（默认 `V_1`），`--group` 选择版本内的分组（默认 `baseline`，实验用 `experment_N`），两项也支持自定义单层名称。它们不是源码加载开关；`--model-name` 也只影响结果命名。每版可分别放 baseline 和多个实验，每次仍需记录准确的代码、配置、checkpoint 和服务地址。映射说明见 [评估对象约定](evaluation/targets/README.md)。

正式评估沿用当前外部固定 runner，原有任务、计分、运行路径和结果不变。新集中的 `evaluation/sim/` 及通用 rollout 尚未替换正式 runner，也未完成仿真对齐验收。使用方法见 [evaluation/README.md](evaluation/README.md)，公平比较规则见 [evaluation/evaluation.md](evaluation/evaluation.md)。未微调的官方权重用于原版参照；检验记忆收益还需要同数据、同预算的无记忆微调对照。
