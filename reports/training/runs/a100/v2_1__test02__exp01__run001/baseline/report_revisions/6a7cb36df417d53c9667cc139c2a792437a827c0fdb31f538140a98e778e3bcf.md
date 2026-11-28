# 训练分析报告

由工作台归档程序生成；不含 AI 推断。缺失字段为未记录，不以零替代。

## 1. 实验身份与权重

| 字段 | 值 |
| --- | --- |
| run | v2_1__test02__exp01__run001 |
| device | a100 |
| arm | baseline |
| status | completed |
| step | 3000 |
| 目标步数 | 3000 |
| 归档事件 | 51d79243d755c7c37fb3-e2583a8073bf |
| 归档时间 | 2026-11-27T07:38:12.072639+00:00 |
| 训练备注 | User authorized V2_1 Test02/Exp01 continued baseline formal3000; exact real-preflight source; no evaluation dispatch |
| 源码摘要 | b5d33c67be1f532d1b4bb93ea71ec1a2901f9fe06675c91f5711c531e45c7655 |
| 初始权重/续训 | {"checkpoint": "/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2.1__test01__exp14__run001/run/checkpoints/v2.1__test01__exp14__run001__step006000", "label": "V2_1 原始起点", "mode": "fresh"} |

## 2. 实际训练配置

配置来源：["/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2_1__test02__exp01__run001/config/web_launch.json"]

| 参数 | 值 |
| --- | --- |
| accept_preflight | /nvme3/data_wyh/outputs/memory_vla/v2_1/test02/exp01/evidence/v2_1__test02__exp01__run001__preflight001-preflight.json |
| arm | baseline |
| asynchronous_validation | False |
| category | formal_training |
| checkpoint | /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2.1__test01__exp14__run001/run/checkpoints/v2.1__test01__exp14__run001__step006000 |
| checkpoint_interval | 1000 |
| created_at | 2026-11-27T12:06:08.170784 |
| dataset | /nvme3/data_wyh/datasets/b1k2025_groot_task40_v1_200 |
| enqueue | True |
| existing_pids | [3449291, 3449292, 3449293, 3449294] |
| fusion_layers | 0 |
| global_decisions | 64 |
| global_windows_per_step | 2 |
| gpus | [0, 1, 2, 3] |
| labels | /nvme3/data_wyh/datasets/v2_1_task40_labels/task40_labels.parquet |
| learning_rate | 1e-05 |
| method | baseline |
| microbatch | 2 |
| naming_schema | v2_1-test-experiment-run-step |
| notes | User authorized V2_1 Test02/Exp01 continued baseline formal3000; exact real-preflight source; no evaluation dispatch |
| preflight_only | False |
| prior | topology |
| run_id | v2_1__test02__exp01__run001 |
| run_name | v2_1__test02__exp01__run001 |
| seed | 42 |
| share_existing | False |
| source | /nvme3/projects/memory_vla/deployments/v2_1__test02__exp01__run001 |
| stage | continued |
| state_weight | 0.01 |
| steps | 3000 |
| task | 40 |
| train_episodes | 180 |
| train_number | 1 |
| training_id | v2_1__test02__exp01__run001 |
| validation_enabled | True |
| validation_episodes | 20 |
| validation_gpu | 未记录 |
| version | v2_1 |
| weak_weight | 0.5 |

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
| records | 3000 |
| first_step | 1 |
| last_step | 3000 |
| duplicate_steps | 0 |
| nonincreasing_transitions | 0 |
| interior_missing_steps | 0 |
| incomplete_tail_records | 0 |
| finite_loss | 3000 |
| loss_first100 | 0.014303987 |
| loss_last100 | 0.011696154 |
| loss_min | 0.0010176524 |

### 每1000 optimizer steps 的训练记录均值

| 步数范围 | 记录数 | Loss | LR首值 | LR末值 | 梯度范数 | 底盘 | 躯干 | 左臂 | 右臂 | 左夹爪 | 右夹爪 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1–1000 | 1000 | 0.013272549 | 1e-05 | 1e-05 | 0.11945685 | 未记录 | 未记录 | 未记录 | 未记录 | 未记录 | 未记录 |
| 1001–2000 | 1000 | 0.01249346 | 1e-05 | 1e-05 | 0.11250732 | 未记录 | 未记录 | 未记录 | 未记录 | 未记录 | 未记录 |
| 2001–3000 | 1000 | 0.012376976 | 1e-05 | 1e-05 | 0.11050408 | 未记录 | 未记录 | 未记录 | 未记录 | 未记录 | 未记录 |

### 按实际 epoch 字段汇总

| epoch | 记录数 | Loss |
| --- | ---: | ---: |
| 0 | 1184 | 0.013218667 |
| 1 | 1183 | 0.012450446 |
| 2 | 633 | 0.012264149 |

### 验证指标（独立于训练记录）

| step | FM Loss | 样本数 |
| ---: | ---: | ---: |
| 未记录 | 未记录 | 未记录 |

### 指标覆盖

| 字段 | 有限记录 | 缺失/非有限 | 均值 | 最小 | 最大 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fm | 3000 | 0 | 0.012714328 | 0.0010176524 | 0.033105513 |
| grad_norm | 3000 | 0 | 0.11415608 | 0.051884763 | 0.94722146 |
| lr | 3000 | 0 | 1e-05 | 1e-05 | 1e-05 |
| peak_allocated_gib | 3000 | 0 | 30.063856 | 30.060892 | 30.079653 |
| seconds | 3000 | 0 | 3.6368866 | 1.3736973 | 6.2966604 |
| state | 3000 | 0 | 0 | 0 | 0 |

## 5. Checkpoint

| step | 路径 |
| ---: | --- |
| 3000 | /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2_1__test02__exp01__run001/run/checkpoints/v2_1__test02__exp01__run001__step003000 |
| 2000 | /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2_1__test02__exp01__run001/run/checkpoints/v2_1__test02__exp01__run001__step002000 |
| 1000 | /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2_1__test02__exp01__run001/run/checkpoints/v2_1__test02__exp01__run001__step001000 |
| 10 | /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2_1__test02__exp01__run001/run/checkpoints/v2_1__test02__exp01__run001__step000010 |
| 0 | /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2_1__test02__exp01__run001/run/checkpoints/v2_1__test02__exp01__run001__step000000 |

## 6. 闭环评估（仅明确绑定本 run 的终态记录）

尚未归档匹配的终态评估。训练结束不等于评估完成。

## 7. 证据与局限

- [归档清单](events/51d79243d755c7c37fb3-e2583a8073bf/MANIFEST.json)
- [工作台原始快照](events/51d79243d755c7c37fb3-e2583a8073bf/workbench.json)
- 原始指标按清单中的 SHA-256 分片保存；日志只保存已声明范围。未记录的指标不作推断。
- 归一化审计、实际 processor 配置、源码、终态评估及逐回合数据以归档证据为准。
- [recorded/logs/controller.log](events/51d79243d755c7c37fb3-e2583a8073bf/recorded/logs/controller.log)
