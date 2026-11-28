# 训练分析报告

由工作台归档程序生成；不含 AI 推断。缺失字段为未记录，不以零替代。

## 1. 实验身份与权重

| 字段 | 值 |
| --- | --- |
| run | v2.1__test01__exp14__run001 |
| device | a100 |
| arm | baseline |
| status | completed |
| step | 6000 |
| 目标步数 | 6000 |
| 归档事件 | ca1b08b235aff1cf5f75-680092cc16ac |
| 归档时间 | 2026-11-28T03:11:29.575472+00:00 |
| 训练备注 | Confirmed V2.1 Stage1: generic GR00T, 180/20, 6000 updates, original expert; deterministic CUDA and strict four-rank replay passed. |
| 源码摘要 | d6123c2ed1f441fcbdaf7526b8eaf47855ff43ab60a85e02a4d8902778ea9439 |
| 初始权重/续训 | {"checkpoint": "/nvme3/huggingface/train_model/GR00T/GR00T-N1.6-3B", "label": "V2_1 原始起点", "mode": "fresh", "optimizer_restored": false, "parent": null} |

## 2. 实际训练配置

配置来源：["/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2.1__test01__exp14__run001/config/web_launch.json"]

| 参数 | 值 |
| --- | --- |
| accept_preflight | /nvme3/data_wyh/outputs/memory_vla/v2_1/experiment14/evidence/preflight-passed-003.json |
| arm | baseline |
| asynchronous_validation | False |
| category | formal_training |
| checkpoint | /nvme3/huggingface/train_model/GR00T/GR00T-N1.6-3B |
| checkpoint_interval | 1000 |
| created_at | 2026-11-27T00:48:27.609452 |
| dataset | /nvme3/data_wyh/datasets/b1k2025_groot_task40_v1_200 |
| enqueue | True |
| existing_pids | [] |
| global_decisions | 64 |
| global_windows_per_step | 2 |
| gpus | [0, 1, 2, 3] |
| labels | /nvme3/data_wyh/datasets/v2_1_task40_labels/task40_labels.parquet |
| learning_rate | 1e-05 |
| method | baseline |
| microbatch | 2 |
| naming_schema | v2.1-test-experiment-run-step |
| notes | Confirmed V2.1 Stage1: generic GR00T, 180/20, 6000 updates, original expert; deterministic CUDA and strict four-rank replay passed. |
| preflight_only | False |
| run_id | v2.1__test01__exp14__run001 |
| run_name | v2.1__test01__exp14__run001 |
| seed | 42 |
| share_existing | False |
| source | /nvme3/projects/memory_vla/deployments/v2.1__test01__exp14__run001 |
| stage | stage1 |
| state_weight | 0.01 |
| steps | 6000 |
| task | 40 |
| train_episodes | 180 |
| train_number | 1 |
| training_id | v2.1__test01__exp14__run001 |
| validation_enabled | True |
| validation_episodes | 20 |
| validation_gpu | 未记录 |
| version | v2.1 |
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
| records | 6000 |
| first_step | 1 |
| last_step | 6000 |
| duplicate_steps | 0 |
| nonincreasing_transitions | 0 |
| interior_missing_steps | 0 |
| incomplete_tail_records | 0 |
| finite_loss | 6000 |
| loss_first100 | 0.080086173 |
| loss_last100 | 0.014469525 |
| loss_min | 0.0011126471 |

### 每1000 optimizer steps 的训练记录均值

| 步数范围 | 记录数 | Loss | LR首值 | LR末值 | 梯度范数 | 底盘 | 躯干 | 左臂 | 右臂 | 左夹爪 | 右夹爪 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1–1000 | 1000 | 0.026866097 | 1e-05 | 1e-05 | 0.20623985 | 未记录 | 未记录 | 未记录 | 未记录 | 未记录 | 未记录 |
| 1001–2000 | 1000 | 0.016031595 | 1e-05 | 1e-05 | 0.14567718 | 未记录 | 未记录 | 未记录 | 未记录 | 未记录 | 未记录 |
| 2001–3000 | 1000 | 0.015192305 | 1e-05 | 1e-05 | 0.12977416 | 未记录 | 未记录 | 未记录 | 未记录 | 未记录 | 未记录 |
| 3001–4000 | 1000 | 0.014257625 | 1e-05 | 1e-05 | 0.12503667 | 未记录 | 未记录 | 未记录 | 未记录 | 未记录 | 未记录 |
| 4001–5000 | 1000 | 0.014211261 | 1e-05 | 1e-05 | 0.12454529 | 未记录 | 未记录 | 未记录 | 未记录 | 未记录 | 未记录 |
| 5001–6000 | 1000 | 0.013770034 | 1e-05 | 1e-05 | 0.12637121 | 未记录 | 未记录 | 未记录 | 未记录 | 未记录 | 未记录 |

### 按实际 epoch 字段汇总

| epoch | 记录数 | Loss |
| --- | ---: | ---: |
| 0 | 1184 | 0.025385419 |
| 1 | 1183 | 0.01565528 |
| 2 | 1186 | 0.01479783 |
| 3 | 1187 | 0.014117461 |
| 4 | 1191 | 0.013735616 |
| 5 | 69 | 0.015733506 |

### 验证指标（独立于训练记录）

| step | FM Loss | 样本数 |
| ---: | ---: | ---: |
| 0 | 0.33028296 | 未记录 |
| 3000 | 0.014659625 | 未记录 |
| 6000 | 0.014369918 | 未记录 |

### 指标覆盖

| 字段 | 有限记录 | 缺失/非有限 | 均值 | 最小 | 最大 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fm | 6000 | 0 | 0.016721486 | 0.0011126471 | 0.35997383 |
| grad_norm | 6000 | 0 | 0.14294073 | 0.050562318 | 3.2791696 |
| lr | 6000 | 0 | 1e-05 | 1e-05 | 1e-05 |
| peak_allocated_gib | 6000 | 0 | 30.063813 | 30.060892 | 30.076723 |
| seconds | 6000 | 0 | 3.5211406 | 0.75060764 | 7.4361713 |
| state | 6000 | 0 | 0 | 0 | 0 |

## 5. Checkpoint

| step | 路径 |
| ---: | --- |
| 6000 | /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2.1__test01__exp14__run001/run/checkpoints/v2.1__test01__exp14__run001__step006000 |
| 4000 | /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2.1__test01__exp14__run001/run/checkpoints/v2.1__test01__exp14__run001__step004000 |
| 2000 | /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2.1__test01__exp14__run001/run/checkpoints/v2.1__test01__exp14__run001__step002000 |

## 6. 闭环评估（仅明确绑定本 run 的终态记录）

### evaluation/V_2_1/baseline/task40_eval01/results.json

| 字段 | 值 |
| --- | --- |
| checkpoint | /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2.1__test01__exp14__run001/run/checkpoints/v2.1__test01__exp14__run001__step006000 |
| 状态 | cancelled |
| 完成/计划 | [10, 50] |
| 成功/有效 | [0, 10] |
| Q Score | 0 |
| Task Progress | 45 |
| 实际评估配置 | {"mode": "single2", "preset": "official-eval", "model_name": "groot", "device": "local", "naming_schema": "groot-runs-v2", "evaluation_id": "task40_eval01", "policy_identity": {"version": "V_2_1", "method": "baseline", "training_id": "v2.1__test01__exp14__run001", "train_number": 1, "checkpoint": "/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2.1__test01__exp14__run001/run/checkpoints/v2.1__test01__exp14__run001__step006000", "checkpoint_step": "v2.1__test01__exp14__run001__step006000", "source_dir": "/nvme3/projects/memory_vla/deployments/inference_cfeabb58a50b5aaec34679524f8e441dbea508d775b7104b65916cd3aa67a21a", "source_digest": "cfeabb58a50b5aaec34679524f8e441dbea508d775b7104b65916cd3aa67a21a", "training_seed": 42, "policy_seed": "42", "history_mode": "normal", "task_id": 40, "pid": 4127800, "start_ticks": 1409671215, "host": "172.28.62.17", "port": 5558, "naming_schema": "v2.1-test-experiment-run-step", "task40_diagnostics": false, "task40_trace": false, "gpu_index": "2", "binding": "observed_process"}, "method": "baseline", "version": "V_2_1", "group": "baseline", "protocol": {"id": "b1k-groot-v1", "sha256": "620ecd00b89129120d3a8c9ffdaa4c7c44aa560991a9da3190e29b0bee940508"}, "run_tag": null, "task_id": 40, "n_envs_per_gpu": 2, "instances": {"1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}, "base_seed": 0, "seed_source": "0", "seed_mode": "sequence", "episodes_per_gpu": 50, "repetitions": 5, "episodes_per_round": 10, "episode_layout": {"1": [{"instances": [0, 1, 2, 3, 4], "quota": 25, "offset": 0}, {"instances": [5, 6, 7, 8, 9], "quota": 25, "offset": 0}]}, "synchronized_start": false, "policy_server": "172.28.62.17:5558", "action_chunk": 8, "record_video": true, "video_steps_per_render": 24, "video_fps": 20, "video_speedup": 16.0, "viewer_enabled": true, "task40_stage_trace": true, "runner_max_episode_steps": 999999999, "diagnostics": false, "wait_for_gpu": false, "gpu_wait_interval": 30.0, "appdata": {"1": "/nvme3/tmp/memory_vla/groot_runtime_gpu1/appdata"}, "invocation": ["evaluation/scripts/run_groot_eval.sh", "--gpu-index", "1", "--group", "baseline", "--mode", "single2", "--model-name", "groot", "--policy-host", "172.28.62.17", "--policy-port", "5558", "--preset", "official-eval", "--repetitions", "5", "--seed", "0", "--seed-mode", "sequence", "--task", "40", "--version", "V_2_1", "--video-steps-per-render", "24", "--record-video", "--policy-identity", "{\"version\": \"V_2_1\", \"method\": \"baseline\", \"training_id\": \"v2.1__test01__exp14__run001\", \"train_number\": 1, \"checkpoint\": \"/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2.1__test01__exp14__run001/run/checkpoints/v2.1__test01__exp14__run001__step006000\", \"checkpoint_step\": \"v2.1__test01__exp14__run001__step006000\", \"source_dir\": \"/nvme3/projects/memory_vla/deployments/inference_cfeabb58a50b5aaec34679524f8e441dbea508d775b7104b65916cd3aa67a21a\", \"source_digest\": \"cfeabb58a50b5aaec34679524f8e441dbea508d775b7104b65916cd3aa67a21a\", \"training_seed\": 42, \"policy_seed\": \"42\", \"history_mode\": \"normal\", \"task_id\": 40, \"pid\": 4127800, \"start_ticks\": 1409671215, \"host\": \"172.28.62.17\", \"port\": 5558, \"naming_schema\": \"v2.1-test-experiment-run-step\", \"task40_diagnostics\": false, \"task40_trace\": false, \"gpu_index\": \"2\", \"binding\": \"observed_process\"}"]} |

| 回合 | instance | seed | 成功 | Q Score | Task Progress | 终止 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 0 | False | 0 | 12.5 | {"termination_conditions": {"timeout": {"done": true, "success": false}, "predicate": {"done": false, "success": false}}, "success": false, "goal_status": {"satisfied": [], "unsatisfied": [0, 1]}} |
| 2 | 5 | 1 | False | 0 | 50 | {"termination_conditions": {"timeout": {"done": true, "success": false}, "predicate": {"done": false, "success": false}}, "success": false, "goal_status": {"satisfied": [], "unsatisfied": [0, 1]}} |
| 3 | 1 | 1 | False | 0 | 50 | {"termination_conditions": {"timeout": {"done": true, "success": false}, "predicate": {"done": false, "success": false}}, "success": false, "goal_status": {"satisfied": [], "unsatisfied": [0, 1]}} |
| 4 | 6 | 2 | False | 0 | 50 | {"termination_conditions": {"timeout": {"done": true, "success": false}, "predicate": {"done": false, "success": false}}, "success": false, "goal_status": {"satisfied": [], "unsatisfied": [0, 1]}} |
| 5 | 7 | 3 | False | 0 | 50 | {"termination_conditions": {"timeout": {"done": true, "success": false}, "predicate": {"done": false, "success": false}}, "success": false, "goal_status": {"satisfied": [], "unsatisfied": [0, 1]}} |
| 6 | 2 | 2 | False | 0 | 50 | {"termination_conditions": {"timeout": {"done": true, "success": false}, "predicate": {"done": false, "success": false}}, "success": false, "goal_status": {"satisfied": [], "unsatisfied": [0, 1]}} |
| 7 | 8 | 4 | False | 0 | 50 | {"termination_conditions": {"timeout": {"done": true, "success": false}, "predicate": {"done": false, "success": false}}, "success": false, "goal_status": {"satisfied": [], "unsatisfied": [0, 1]}} |
| 8 | 3 | 3 | False | 0 | 37.5 | {"termination_conditions": {"timeout": {"done": true, "success": false}, "predicate": {"done": false, "success": false}}, "success": false, "goal_status": {"satisfied": [], "unsatisfied": [0, 1]}} |
| 9 | 9 | 5 | False | 0 | 50 | {"termination_conditions": {"timeout": {"done": true, "success": false}, "predicate": {"done": false, "success": false}}, "success": false, "goal_status": {"satisfied": [], "unsatisfied": [0, 1]}} |
| 10 | 4 | 4 | False | 0 | 50 | {"termination_conditions": {"timeout": {"done": true, "success": false}, "predicate": {"done": false, "success": false}}, "success": false, "goal_status": {"satisfied": [], "unsatisfied": [0, 1]}} |


## 7. 证据与局限

- [归档清单](events/ca1b08b235aff1cf5f75-680092cc16ac/MANIFEST.json)
- [工作台原始快照](events/ca1b08b235aff1cf5f75-680092cc16ac/workbench.json)
- 原始指标按清单中的 SHA-256 分片保存；日志只保存已声明范围。未记录的指标不作推断。
- 归一化审计、实际 processor 配置、源码、终态评估及逐回合数据以归档证据为准。
- [recorded/logs/controller.log](events/ca1b08b235aff1cf5f75-680092cc16ac/recorded/logs/controller.log)
- [recorded/logs/policy_v2.1__test01__exp14__run001_v2.1__test01__exp14__run001__step006000_port5558.log](events/ca1b08b235aff1cf5f75-680092cc16ac/recorded/logs/policy_v2.1__test01__exp14__run001_v2.1__test01__exp14__run001__step006000_port5558.log)
- [recorded/logs/policy_v2.1__test01__exp14__run001_v2.1__test01__exp14__run001__step006000_port5564.log](events/ca1b08b235aff1cf5f75-680092cc16ac/recorded/logs/policy_v2.1__test01__exp14__run001_v2.1__test01__exp14__run001__step006000_port5564.log)
- [recorded/logs/policy_v2.1__test01__exp14__run001_v2.1__test01__exp14__run001__step006000_port5565.log](events/ca1b08b235aff1cf5f75-680092cc16ac/recorded/logs/policy_v2.1__test01__exp14__run001_v2.1__test01__exp14__run001__step006000_port5565.log)
- [recorded/logs/policy_v2.1__test01__exp14__run001_v2.1__test01__exp14__run001__step006000_port5566.log](events/ca1b08b235aff1cf5f75-680092cc16ac/recorded/logs/policy_v2.1__test01__exp14__run001_v2.1__test01__exp14__run001__step006000_port5566.log)
- [recorded/logs/policy_v2.1__test01__exp14__run001_v2.1__test01__exp14__run001__step006000_port5594.log](events/ca1b08b235aff1cf5f75-680092cc16ac/recorded/logs/policy_v2.1__test01__exp14__run001_v2.1__test01__exp14__run001__step006000_port5594.log)
- [recorded/run/validation-000000.json](events/ca1b08b235aff1cf5f75-680092cc16ac/recorded/run/validation-000000.json)
- [recorded/run/validation-003000.json](events/ca1b08b235aff1cf5f75-680092cc16ac/recorded/run/validation-003000.json)
- [recorded/run/validation-006000.json](events/ca1b08b235aff1cf5f75-680092cc16ac/recorded/run/validation-006000.json)
