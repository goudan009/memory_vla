# 训练分析报告

由工作台归档程序生成；不含 AI 推断。缺失字段为未记录，不以零替代。

## 1. 实验身份与权重

| 字段 | 值 |
| --- | --- |
| run | v2_1__test02__exp01__run003 |
| device | a100 |
| arm | baseline |
| status | paused |
| step | 5000 |
| 目标步数 | 6000 |
| 归档事件 | 120c379b1fa8ccea5279-0f7d425b57c9 |
| 归档时间 | 2026-11-28T03:11:27.089317+00:00 |
| 训练备注 | User-authorized plan-v005 native_random_v2 Stage1 formal6000 after real four-A100 strict pause/resume gate; then exact step6000 A100 inference and dual4090 dual2 Task40 10 valid episodes; no Stage2 |
| 源码摘要 | 53a2c0ab6c747c0264988ca0b6c53e12c433cbc7354d6c542bfb95158d818615 |
| 初始权重/续训 | {"checkpoint": "/nvme3/huggingface/train_model/GR00T/GR00T-N1.6-3B", "label": "V2_1 原始起点", "mode": "fresh", "optimizer_restored": false, "parent": null} |

## 2. 实际训练配置

配置来源：["/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2_1__test02__exp01__run003/config/web_launch.json"]

| 参数 | 值 |
| --- | --- |
| accept_preflight | /nvme3/data_wyh/outputs/memory_vla/v2_1/test02/exp01/evidence/v2_1__test02__exp01__run003__preflight001-preflight.json |
| accumulation | 8 |
| arm | baseline |
| asynchronous_validation | False |
| category | formal_training |
| checkpoint | /nvme3/huggingface/train_model/GR00T/GR00T-N1.6-3B |
| checkpoint_interval | 1000 |
| checkpoint_root | /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2_1__test02__exp01__run003/run/checkpoints |
| created_at | 2026-11-27T19:39:39.564648 |
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
| notes | User-authorized plan-v005 native_random_v2 Stage1 formal6000 after real four-A100 strict pause/resume gate; then exact step6000 A100 inference and dual4090 dual2 Task40 10 valid episodes; no Stage2 |
| official_root | /nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1 |
| output | /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2_1__test02__exp01__run003 |
| preflight_only | False |
| preflight_steps | 未记录 |
| recipe | native_random_v2 |
| run_id | v2_1__test02__exp01__run003 |
| run_name | v2_1__test02__exp01__run003 |
| scheduler | cosine |
| seed | 42 |
| share_existing | True |
| source | /nvme3/projects/memory_vla/deployments/v2_1__test02__exp01__run003 |
| split | /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2.1__test01__exp14__run001/run/data_split.json |
| stage | stage1 |
| steps | 6000 |
| task | 40 |
| test_id | test02 |
| train_episodes | 180 |
| train_number | 3 |
| training_id | v2_1__test02__exp01__run003 |
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
| accumulation | 8 |
| batch | 2 |
| optimizer | OptimizerNames.ADAMW_TORCH_FUSED |
| scheduler | SchedulerType.COSINE |
| total | 3286608832 |
| trainable | 1418624000 |
| prepared.mode | guarded |
| prepared.train_frames | 582460 |
| prepared.val_frames | 65095 |
| prepared.train_valid_anchors | 576880 |
| prepared.accepted_cells | {"torso": 3, "left_arm": 20, "right_arm": 15} |
| prepared.hashes | {"data_split.json": "739014878990d18a30fc3b30780f6ffaa45de7729b82ea0d157dae0fa6789d6b", "normalization_audit.json": "0eb871c2fc8d7eee61d6ec756654e0059da17aec2f98d8040daad401439a5089", "generic_base/config.json": "106841a8bb58b4c1a9115908a471fa5051492572e10a1d881ef711f5065c048f", "train/meta/stats.json": "c1688f1036ecc118e2b2e2a6ccb1cc251cef155d424578fc563ae339ff6407cb", "train/meta/relative_stats.json": "f9ff3415b68418ea7e255894f7b1ff9351a7950afefd6e3181b2d1d4f8a03f60", "train/meta/episodes.jsonl": "34844eeb8c6c232c296bbf4d6ff43609974bbcacc474157323fd7419ad4db57c", "train/meta/info.json": "821372de6e6977f37920d6a33b98305a1a64092a163d607bbd452657e892a968", "train/meta/modality.json": "af3a354159496192103bcec4d7cd0dd5cfeeba3445af82b5a09c9cbfe37bff19", "train/meta/tasks.jsonl": "fb6717bbb4fa503ac5d2bbf220d13727e29d6d320e5cd0d4a4e9b2200483943b", "validation/meta/stats.json": "c1688f1036ecc118e2b2e2a6ccb1cc251cef155d424578fc563ae339ff6407cb", "validation/meta/relative_stats.json": "f9ff3415b68418ea7e255894f7b1ff9351a7950afefd6e3181b2d1d4f8a03f60", "validation/meta/episodes.jsonl": "2adf6032e88bd0275573fddb49d3b08f93549635af1a9c3cf8ea7fafe5ba136a", "validation/meta/info.json": "d3139d99dbcaa760b125359b64ba6e0448c54f512fb371d61a2f8876c123498a", "validation/meta/modality.json": "af3a354159496192103bcec4d7cd0dd5cfeeba3445af82b5a09c9cbfe37bff19", "validation/meta/tasks.jsonl": "fb6717bbb4fa503ac5d2bbf220d13727e29d6d320e5cd0d4a4e9b2200483943b"} |
| split | {"task": 40, "seed": 42, "train": [400010, 400020, 400030, 400040, 400050, 400060, 400070, 400090, 400100, 400110, 400120, 400130, 400140, 400150, 400160, 400170, 400180, 400190, 400200, 400220, 400230, 400240, 400260, 400280, 400290, 400300, 400310, 400320, 400330, 400340, 400350, 400360, 400370, 400400, 400410, 400420, 400430, 400440, 400450, 400460, 400490, 400500, 400510, 400530, 400540, 400550, 400560, 400570, 400600, 400610, 400620, 400640, 400660, 400670, 400680, 400690, 400700, 400710, 400720, 400730, 400740, 400750, 400760, 400780, 400790, 400800, 400810, 400820, 400830, 400840, 400860, 400870, 400880, 400890, 400900, 400910, 400920, 400930, 400940, 400950, 400960, 400970, 400990, 401000, 401010, 401020, 401030, 401040, 401050, 401060, 401080, 401090, 401100, 401110, 401120, 401130, 401150, 401160, 401200, 401220, 401230, 401240, 401250, 401260, 401280, 401290, 401300, 401310, 401320, 401340, 401350, 401360, 401370, 401380, 401390, 401400, 401410, 401420, 401430, 401450, 401460, 401470, 401480, 401490, 401500, 401510, 401530, 401540, 401560, 401580, 401650, 401860, 401880, 401910, 402010, 402190, 402230, 402470, 402500, 402510, 402520, 402530, 402550, 402560, 402570, 402580, 402590, 402600, 402610, 402620, 402630, 402640, 402650, 402660, 402670, 402710, 402720, 402740, 402750, 402760, 402770, 402790, 402800, 402810, 402820, 402830, 402840, 402850, 402860, 402870, 402880, 402900, 402920, 402930, 402940, 402950, 402960, 402970, 402990, 403000], "validation": [400080, 400210, 400250, 400270, 400380, 400390, 400480, 400520, 400580, 400850, 401140, 401270, 401520, 401710, 401950, 402340, 402680, 402780, 402910, 402980], "train_episodes_per_task": 180, "validation_episodes_per_task": 20, "validation_enabled": true} |
| 初始权重分片 SHA-256 | {"model-00001-of-00002.safetensors": "9710d31d331b79cfa229fd23605ba9ad47e207cb0f4c7722fe0bacdc666c8326", "model-00002-of-00002.safetensors": "eb79bbd7893068897cda18c86ed91064e92e20e356dea97bc398bf9c8bf2fa35"} |
| 统计量来源 | {"data_split.json": "739014878990d18a30fc3b30780f6ffaa45de7729b82ea0d157dae0fa6789d6b", "normalization_audit.json": "0eb871c2fc8d7eee61d6ec756654e0059da17aec2f98d8040daad401439a5089", "generic_base/config.json": "106841a8bb58b4c1a9115908a471fa5051492572e10a1d881ef711f5065c048f", "train/meta/stats.json": "c1688f1036ecc118e2b2e2a6ccb1cc251cef155d424578fc563ae339ff6407cb", "train/meta/relative_stats.json": "f9ff3415b68418ea7e255894f7b1ff9351a7950afefd6e3181b2d1d4f8a03f60", "train/meta/episodes.jsonl": "34844eeb8c6c232c296bbf4d6ff43609974bbcacc474157323fd7419ad4db57c", "train/meta/info.json": "821372de6e6977f37920d6a33b98305a1a64092a163d607bbd452657e892a968", "train/meta/modality.json": "af3a354159496192103bcec4d7cd0dd5cfeeba3445af82b5a09c9cbfe37bff19", "train/meta/tasks.jsonl": "fb6717bbb4fa503ac5d2bbf220d13727e29d6d320e5cd0d4a4e9b2200483943b", "validation/meta/stats.json": "c1688f1036ecc118e2b2e2a6ccb1cc251cef155d424578fc563ae339ff6407cb", "validation/meta/relative_stats.json": "f9ff3415b68418ea7e255894f7b1ff9351a7950afefd6e3181b2d1d4f8a03f60", "validation/meta/episodes.jsonl": "2adf6032e88bd0275573fddb49d3b08f93549635af1a9c3cf8ea7fafe5ba136a", "validation/meta/info.json": "d3139d99dbcaa760b125359b64ba6e0448c54f512fb371d61a2f8876c123498a", "validation/meta/modality.json": "af3a354159496192103bcec4d7cd0dd5cfeeba3445af82b5a09c9cbfe37bff19", "validation/meta/tasks.jsonl": "fb6717bbb4fa503ac5d2bbf220d13727e29d6d320e5cd0d4a4e9b2200483943b"} |

## 4. 训练指标

训练 Loss 为记录点/日志区间值；验证 Loss 单独列出。不同动作归一化、mask、噪声采样及数据划分下的 Loss 不宜直接比较。
首尾统计为最多100条日志记录；日志间隔可能大于1步，步号间隔不等于丢失训练更新。

| 字段 | 值 |
| --- | --- |
| records | 500 |
| first_step | 10 |
| last_step | 5000 |
| duplicate_steps | 0 |
| nonincreasing_transitions | 0 |
| interior_missing_steps | 4491 |
| incomplete_tail_records | 0 |
| finite_loss | 500 |
| loss_first100 | 0.037762 |
| loss_last100 | 0.016921 |
| loss_min | 0.0141 |

### 每1000 optimizer steps 的训练记录均值

| 步数范围 | 记录数 | Loss | LR首值 | LR末值 | 梯度范数 | 底盘 | 躯干 | 左臂 | 右臂 | 左夹爪 | 右夹爪 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1–1000 | 100 | 0.037762 | 3e-06 | 9.6335067e-05 | 0.20444553 | 0.083374884 | 0.033862087 | 0.027187323 | 0.028123721 | 0.054113436 | 0.04171468 |
| 1001–2000 | 100 | 0.023126 | 9.6230801e-05 | 7.9633963e-05 | 0.13159541 | 0.0720063 | 0.012319046 | 0.014492718 | 0.017856055 | 0.0049571746 | 0.035188367 |
| 2001–3000 | 100 | 0.020768 | 7.9411553e-05 | 5.415643e-05 | 0.11722566 | 0.067029526 | 0.010719256 | 0.012537314 | 0.015537666 | 0.0035117441 | 0.033761324 |
| 3001–4000 | 100 | 0.018634 | 5.3881744e-05 | 2.7447924e-05 | 0.10298157 | 0.062093354 | 0.0083081467 | 0.011267081 | 0.014264577 | 0.0023415043 | 0.028012428 |
| 4001–5000 | 100 | 0.016921 | 2.7202313e-05 | 7.4184673e-06 | 0.088733009 | 0.059563652 | 0.0065289738 | 0.0095456315 | 0.012610195 | 0.0014651424 | 0.027949723 |

### 按实际 epoch 字段汇总

| epoch | 记录数 | Loss |
| --- | ---: | ---: |
| unrecorded | 500 | 0.0234422 |

### 验证指标（独立于训练记录）

| step | FM Loss | 样本数 |
| ---: | ---: | ---: |
| 0 | 0.30832182 | 100 |
| 1000 | 0.028893975 | 100 |
| 2000 | 0.027362544 | 100 |
| 3000 | 0.027565452 | 100 |
| 4000 | 0.023294779 | 100 |
| 5000 | 0.021558075 | 100 |

### 指标覆盖

| 字段 | 有限记录 | 缺失/非有限 | 均值 | 最小 | 最大 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fm | 500 | 0 | 0.0234422 | 0.0141 | 0.2854 |
| grad_norm | 500 | 0 | 0.12899624 | 0.047957107 | 1.9101104 |
| lr | 500 | 0 | 5.9507257e-05 | 3e-06 | 9.9999385e-05 |
| part_loss.base | 500 | 0 | 0.068813543 | 0.047738814 | 0.18976038 |
| part_loss.left_arm | 500 | 0 | 0.015006013 | 0.006646777 | 0.1838051 |
| part_loss.left_gripper | 500 | 0 | 0.0132778 | 0.00029728393 | 1.9039082 |
| part_loss.right_arm | 500 | 0 | 0.017678443 | 0.0089684398 | 0.14224702 |
| part_loss.right_gripper | 500 | 0 | 0.033325304 | 0.011681784 | 0.074015673 |
| part_loss.torso | 500 | 0 | 0.014347502 | 0.0034836093 | 0.4383884 |
| state | 500 | 0 | 0 | 0 | 0 |

## 5. Checkpoint

| step | 路径 |
| ---: | --- |
| 5000 | /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2_1__test02__exp01__run003/run/checkpoints/v2_1__test02__exp01__run003__step005000 |
| 4000 | /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2_1__test02__exp01__run003/run/checkpoints/v2_1__test02__exp01__run003__step004000 |
| 3000 | /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2_1__test02__exp01__run003/run/checkpoints/v2_1__test02__exp01__run003__step003000 |
| 2000 | /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2_1__test02__exp01__run003/run/checkpoints/v2_1__test02__exp01__run003__step002000 |
| 1000 | /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2_1__test02__exp01__run003/run/checkpoints/v2_1__test02__exp01__run003__step001000 |

## 6. 闭环评估（仅明确绑定本 run 的终态记录）

### evaluation/V_2_1/baseline/task40_eval02/results.json

| 字段 | 值 |
| --- | --- |
| checkpoint | /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2_1__test02__exp01__run003/run/checkpoints/v2_1__test02__exp01__run003__step005000 |
| 状态 | complete |
| 完成/计划 | [10, 10] |
| 成功/有效 | [0, 10] |
| Q Score | 0 |
| Task Progress | 22.5 |
| 实际评估配置 | {"mode": "dual2", "preset": "official-eval", "model_name": "groot", "device": "local", "naming_schema": "groot-runs-v2", "evaluation_id": "task40_eval02", "policy_identity": {"version": "V_2_1", "method": "baseline", "training_id": "v2_1__test02__exp01__run003", "train_number": 3, "checkpoint": "/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2_1__test02__exp01__run003/run/checkpoints/v2_1__test02__exp01__run003__step005000", "checkpoint_step": "v2_1__test02__exp01__run003__step005000", "source_dir": "/nvme3/projects/memory_vla/deployments/v2_1__test02__exp01__run003", "source_digest": "53a2c0ab6c747c0264988ca0b6c53e12c433cbc7354d6c542bfb95158d818615", "training_seed": 42, "policy_seed": "42", "history_mode": "normal", "task_id": 40, "pid": 2987933, "start_ticks": 1416585703, "host": "172.28.62.17", "port": 5557, "naming_schema": "groot-runs-v3", "task40_diagnostics": false, "task40_trace": false, "gpu_index": "3", "binding": "observed_process"}, "method": "baseline", "version": "V_2_1", "group": "baseline", "protocol": {"id": "b1k-groot-v1", "sha256": "620ecd00b89129120d3a8c9ffdaa4c7c44aa560991a9da3190e29b0bee940508"}, "run_tag": null, "task_id": 40, "n_envs_per_gpu": 2, "instances": {"0": [0, 1, 2, 3, 4], "1": [5, 6, 7, 8, 9]}, "base_seed": 0, "seed_source": "0", "seed_mode": "sequence", "episodes_per_gpu": 5, "repetitions": 1, "episodes_per_round": 5, "episode_layout": {}, "synchronized_start": false, "policy_server": "172.28.62.17:5557", "action_chunk": 8, "record_video": true, "video_steps_per_render": 24, "video_fps": 20, "video_speedup": 16.0, "viewer_enabled": true, "task40_stage_trace": true, "runner_max_episode_steps": 999999999, "diagnostics": false, "wait_for_gpu": false, "gpu_wait_interval": 30.0, "appdata": {"0": "/nvme3/tmp/memory_vla/groot_runtime_gpu0/appdata", "1": "/nvme3/tmp/memory_vla/groot_runtime_gpu1/appdata"}, "invocation": ["evaluation/scripts/run_groot_eval.sh", "--group", "baseline", "--mode", "dual2", "--model-name", "groot", "--policy-host", "172.28.62.17", "--policy-port", "5557", "--preset", "official-eval", "--seed", "0", "--seed-mode", "sequence", "--task", "40", "--version", "V_2_1", "--video-steps-per-render", "24", "--record-video", "--policy-identity", "{\"version\": \"V_2_1\", \"method\": \"baseline\", \"training_id\": \"v2_1__test02__exp01__run003\", \"train_number\": 3, \"checkpoint\": \"/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2_1__test02__exp01__run003/run/checkpoints/v2_1__test02__exp01__run003__step005000\", \"checkpoint_step\": \"v2_1__test02__exp01__run003__step005000\", \"source_dir\": \"/nvme3/projects/memory_vla/deployments/v2_1__test02__exp01__run003\", \"source_digest\": \"53a2c0ab6c747c0264988ca0b6c53e12c433cbc7354d6c542bfb95158d818615\", \"training_seed\": 42, \"policy_seed\": \"42\", \"history_mode\": \"normal\", \"task_id\": 40, \"pid\": 2987933, \"start_ticks\": 1416585703, \"host\": \"172.28.62.17\", \"port\": 5557, \"naming_schema\": \"groot-runs-v3\", \"task40_diagnostics\": false, \"task40_trace\": false, \"gpu_index\": \"3\", \"binding\": \"observed_process\"}"]} |

| 回合 | instance | seed | 成功 | Q Score | Task Progress | 终止 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 5 | 0 | False | 0 | 25 | {"termination_conditions": {"timeout": {"done": true, "success": false}, "predicate": {"done": false, "success": false}}, "success": false, "goal_status": {"satisfied": [], "unsatisfied": [0, 1]}} |
| 2 | 7 | 1 | False | 0 | 25 | {"termination_conditions": {"timeout": {"done": true, "success": false}, "predicate": {"done": false, "success": false}}, "success": false, "goal_status": {"satisfied": [], "unsatisfied": [0, 1]}} |
| 3 | 0 | 0 | False | 0 | 12.5 | {"termination_conditions": {"timeout": {"done": true, "success": false}, "predicate": {"done": false, "success": false}}, "success": false, "goal_status": {"satisfied": [], "unsatisfied": [0, 1]}} |
| 4 | 2 | 1 | False | 0 | 12.5 | {"termination_conditions": {"timeout": {"done": true, "success": false}, "predicate": {"done": false, "success": false}}, "success": false, "goal_status": {"satisfied": [], "unsatisfied": [0, 1]}} |
| 5 | 6 | 1 | False | 0 | 25 | {"termination_conditions": {"timeout": {"done": true, "success": false}, "predicate": {"done": false, "success": false}}, "success": false, "goal_status": {"satisfied": [], "unsatisfied": [0, 1]}} |
| 6 | 8 | 2 | False | 0 | 25 | {"termination_conditions": {"timeout": {"done": true, "success": false}, "predicate": {"done": false, "success": false}}, "success": false, "goal_status": {"satisfied": [], "unsatisfied": [0, 1]}} |
| 7 | 3 | 2 | False | 0 | 25 | {"termination_conditions": {"timeout": {"done": true, "success": false}, "predicate": {"done": false, "success": false}}, "success": false, "goal_status": {"satisfied": [], "unsatisfied": [0, 1]}} |
| 8 | 1 | 1 | False | 0 | 25 | {"termination_conditions": {"timeout": {"done": true, "success": false}, "predicate": {"done": false, "success": false}}, "success": false, "goal_status": {"satisfied": [], "unsatisfied": [0, 1]}} |
| 9 | 9 | 3 | False | 0 | 25 | {"termination_conditions": {"timeout": {"done": true, "success": false}, "predicate": {"done": false, "success": false}}, "success": false, "goal_status": {"satisfied": [], "unsatisfied": [0, 1]}} |
| 10 | 4 | 3 | False | 0 | 25 | {"termination_conditions": {"timeout": {"done": true, "success": false}, "predicate": {"done": false, "success": false}}, "success": false, "goal_status": {"satisfied": [], "unsatisfied": [0, 1]}} |


## 7. 证据与局限

- [归档清单](events/120c379b1fa8ccea5279-0f7d425b57c9/MANIFEST.json)
- [工作台原始快照](events/120c379b1fa8ccea5279-0f7d425b57c9/workbench.json)
- 原始指标按清单中的 SHA-256 分片保存；日志只保存已声明范围。未记录的指标不作推断。
- 归一化审计、实际 processor 配置、源码、终态评估及逐回合数据以归档证据为准。
- [recorded/logs/controller.log](events/120c379b1fa8ccea5279-0f7d425b57c9/recorded/logs/controller.log)
- [recorded/logs/policy_v2_1__test02__exp01__run003_v2_1__test02__exp01__run003__step005000_port5557.log](events/120c379b1fa8ccea5279-0f7d425b57c9/recorded/logs/policy_v2_1__test02__exp01__run003_v2_1__test02__exp01__run003__step005000_port5557.log)
- [recorded/run/native_contract.json](events/120c379b1fa8ccea5279-0f7d425b57c9/recorded/run/native_contract.json)
- [recorded/run/native_runtime.json](events/120c379b1fa8ccea5279-0f7d425b57c9/recorded/run/native_runtime.json)
- [recorded/run/prepared/normalization_audit.json](events/120c379b1fa8ccea5279-0f7d425b57c9/recorded/run/prepared/normalization_audit.json)
- [recorded/run/prepared/train/meta/relative_stats.json](events/120c379b1fa8ccea5279-0f7d425b57c9/recorded/run/prepared/train/meta/relative_stats.json)
- [recorded/run/validation-000000.json](events/120c379b1fa8ccea5279-0f7d425b57c9/recorded/run/validation-000000.json)
- [recorded/run/validation-001000.json](events/120c379b1fa8ccea5279-0f7d425b57c9/recorded/run/validation-001000.json)
- [recorded/run/validation-002000.json](events/120c379b1fa8ccea5279-0f7d425b57c9/recorded/run/validation-002000.json)
- [recorded/run/validation-003000.json](events/120c379b1fa8ccea5279-0f7d425b57c9/recorded/run/validation-003000.json)
- [recorded/run/validation-004000.json](events/120c379b1fa8ccea5279-0f7d425b57c9/recorded/run/validation-004000.json)
- [recorded/run/validation-005000.json](events/120c379b1fa8ccea5279-0f7d425b57c9/recorded/run/validation-005000.json)
