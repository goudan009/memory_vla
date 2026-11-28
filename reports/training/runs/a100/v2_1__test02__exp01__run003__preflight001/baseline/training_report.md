# 训练分析报告

由工作台归档程序生成；不含 AI 推断。缺失字段为未记录，不以零替代。

## 1. 实验身份与权重

| 字段 | 值 |
| --- | --- |
| run | v2_1__test02__exp01__run003__preflight001 |
| device | a100 |
| arm | baseline |
| status | completed |
| step | 4 |
| 目标步数 | 4 |
| 归档事件 | 31e20d304097534b9bfb-79677fd3d4d3 |
| 归档时间 | 2026-11-27T11:36:56.986585+00:00 |
| 训练备注 | User-authorized plan-v005 native_random_v2 Stage1 gate: 6000-step scheduler, pause at2 and strict same-run resume to4; no Stage2 |
| 源码摘要 | 53a2c0ab6c747c0264988ca0b6c53e12c433cbc7354d6c542bfb95158d818615 |
| 初始权重/续训 | {"checkpoint": "/nvme3/huggingface/train_model/GR00T/GR00T-N1.6-3B", "label": "V2_1 原始起点", "mode": "fresh"} |

## 2. 实际训练配置

配置来源：["/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2_1__test02__exp01__run003__preflight001/config/web_launch.json"]

| 参数 | 值 |
| --- | --- |
| accumulation | 8 |
| arm | baseline |
| asynchronous_validation | False |
| category | engineering_preflight |
| checkpoint | /nvme3/huggingface/train_model/GR00T/GR00T-N1.6-3B |
| checkpoint_interval | 1000 |
| checkpoint_root | /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2_1__test02__exp01__run003__preflight001/run/checkpoints |
| created_at | 2026-11-27T19:26:37.323272 |
| dataset | /nvme3/data_wyh/datasets/b1k2025_groot_task40_v1_200 |
| enqueue | True |
| existing_pids | [3857420, 4127800] |
| experiment_id | exp01 |
| global_decisions | 64 |
| gpus | [0, 1, 2, 3] |
| learning_rate | 0.0001 |
| method | baseline |
| microbatch | 2 |
| model_version | V_2_1 |
| naming_schema | groot-runs-v3 |
| normalization | guarded |
| notes | User-authorized plan-v005 native_random_v2 Stage1 gate: 6000-step scheduler, pause at2 and strict same-run resume to4; no Stage2 |
| official_root | /nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1 |
| output | /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2_1__test02__exp01__run003__preflight001 |
| preflight_only | True |
| preflight_steps | 4 |
| recipe | native_random_v2 |
| run_id | v2_1__test02__exp01__run003__preflight001 |
| run_name | v2_1__test02__exp01__run003__preflight001 |
| scheduler | cosine |
| seed | 42 |
| share_existing | True |
| source | /nvme3/projects/memory_vla/deployments/v2_1__test02__exp01__run003__preflight001 |
| split | /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2.1__test01__exp14__run001/run/data_split.json |
| stage | stage1 |
| steps | 6000 |
| task | 40 |
| test_id | test02 |
| train_episodes | 180 |
| train_number | 3 |
| training_id | v2_1__test02__exp01__run003__preflight001 |
| training_run | run003 |
| validation_enabled | True |
| validation_episodes | 20 |
| validation_gpu | 未记录 |
| version | V_2_1 |
| video_backend | torchcodec |
| warmup_ratio | 0.05 |
| weight_decay | 1e-05 |
| workers | 2 |

## 3. 实际运行与数据

| 字段 | 值 |
| --- | --- |
| prepared.mode | 未记录 |
| prepared.train_frames | 未记录 |
| prepared.val_frames | 未记录 |
| prepared.train_valid_anchors | 未记录 |
| prepared.accepted_cells | 未记录 |
| prepared.hashes | 未记录 |
| split | 未记录 |
| 初始权重分片 SHA-256 | 未记录 |
| 统计量来源 | 未记录 |

## 4. 训练指标

训练 Loss 为记录点/日志区间值；验证 Loss 单独列出。不同动作归一化、mask、噪声采样及数据划分下的 Loss 不宜直接比较。
首尾统计为最多100条日志记录；日志间隔可能大于1步，步号间隔不等于丢失训练更新。

| 字段 | 值 |
| --- | --- |
| records | 4 |
| first_step | 1 |
| last_step | 4 |
| duplicate_steps | 0 |
| nonincreasing_transitions | 0 |
| interior_missing_steps | 0 |
| incomplete_tail_records | 0 |
| finite_loss | 0 |
| loss_first100 | 未记录 |
| loss_last100 | 未记录 |
| loss_min | 未记录 |

### 每1000 optimizer steps 的训练记录均值

| 步数范围 | 记录数 | Loss | LR首值 | LR末值 | 梯度范数 | 底盘 | 躯干 | 左臂 | 右臂 | 左夹爪 | 右夹爪 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1–1000 | 4 | 0.296125 | 0 | 1e-06 | 2.3503545 | 0.17676536 | 0.47340509 | 0.19299129 | 0.14851033 | 1.950093 | 0.04598142 |

### 按实际 epoch 字段汇总

| epoch | 记录数 | Loss |
| --- | ---: | ---: |
| unrecorded | 4 | 未记录 |

### 验证指标（独立于训练记录）

| step | FM Loss | 样本数 |
| ---: | ---: | ---: |
| 未记录 | 未记录 | 未记录 |

### 指标覆盖

| 字段 | 有限记录 | 缺失/非有限 | 均值 | 最小 | 最大 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fm | 4 | 0 | 0.296125 | 0.2101 | 0.3683 |
| grad_norm | 4 | 0 | 2.3503545 | 1.8907443 | 2.7595453 |
| lr | 4 | 0 | 5e-07 | 0 | 1e-06 |
| state | 4 | 0 | 0 | 0 | 0 |

## 5. Checkpoint

| step | 路径 |
| ---: | --- |
| 4 | /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2_1__test02__exp01__run003__preflight001/run/checkpoints/v2_1__test02__exp01__run003__preflight001__step000004 |
| 2 | /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2_1__test02__exp01__run003__preflight001/run/checkpoints/v2_1__test02__exp01__run003__preflight001__step000002 |

## 6. 闭环评估（仅明确绑定本 run 的终态记录）

尚未归档匹配的终态评估。训练结束不等于评估完成。

## 7. 证据与局限

- [归档清单](events/31e20d304097534b9bfb-79677fd3d4d3/MANIFEST.json)
- [工作台原始快照](events/31e20d304097534b9bfb-79677fd3d4d3/workbench.json)
- 原始指标按清单中的 SHA-256 分片保存；日志只保存已声明范围。未记录的指标不作推断。
- 归一化审计、实际 processor 配置、源码、终态评估及逐回合数据以归档证据为准。
- [recorded/logs/controller.log](events/31e20d304097534b9bfb-79677fd3d4d3/recorded/logs/controller.log)
