# Task40 记忆干预预实验

## 实验配置

- 起点：复用 `checkpoints/sim_checkpoint.pt`，Task40 instance 4、seed 22004、决策 9 / 仿真步 72。
- 权重：`Output/task40_memory_overnight/replay/original_step4000`；未训练或修改权重。
- A 正常记忆：从前 9 次真实决策回放，恢复 perceptual 与 cognitive 双记忆库。
- B 空记忆：双库内容清空，保留相同决策时钟和写入接口。
- C 错误记忆：使用 B1K Task40 数据集中另一条真实轨迹的特征，维持双库结构、容量和 9 个时间槽。
- 每次策略调用执行 8 个动作步；成功开门、开门前出现抓取/放入/启动等后阶段状态时停止，最多 60 次决策。
- 三组分别恢复仿真、控制器、仿真 RNG 与策略 RNG。相同决策编号、相同输入时钟；记忆回放验证文件均显示重建记忆与原在线记忆精确相等。

## 结果

| 组别 | 决策 / 执行步数 | 停止原因 | 首次错误启动 | 门打开 | 最近指尖到门 AABB | 门体接触 |
|---|---:|---|---|---|---:|---|
| A 正常记忆 | 49 / 392 | 开门前 `ToggledOn=True` | 决策 57，步 464 | 否 | 右 0.0232 m | 无 |
| B 空记忆 | 18 / 144 | 开门前 `ToggledOn=True` | 决策 26，步 216 | 否 | 右 0.0324 m | 无 |
| C 错误记忆 | 16 / 127 | 开门前 `ToggledOn=True` | 决策 24，步 199 | 否 | 右 0.0271 m | 无 |

三组起始真值相同：门关闭、未抓取、爆米花未放入、微波炉未启动。三组都没有开门；接触记录中右手与微波炉发生接触的部件是 `base_link`，没有门体接触。因此三组均在错误阶段启动，不是开门成功或抓取/放入。

## 记忆是否进入动作专家

已在 `action_head.get_action` 调用边界记录 `backbone_output.backbone_features`、mask 和图像 mask；这是真正传给动作专家的特征输入，不只是记忆检索日志。首个决策三路 RGB 与全部策略状态张量在三组之间逐元素相同（最大差为 0）；mask 也相同，但特征发生变化：

| 比较 | 专家特征变化比例 / MAE / 最大差 | 首动作块差异示例 |
|---|---|---|
| A 对 B | 85.32% / 0.22585 / 3.0 | base MAE 0.00840、最大差 0.04696；right arm MAE 0.00955、最大差 0.03788 |
| A 对 C | 3.90% / 0.000152 / 1.0 | base MAE 0.000283；right arm MAE 0.001031 |
| B 对 C | 85.32% / 0.225996 / 3.0 | base MAE 0.00845；right arm MAE 0.00962 |

这证明记忆干预改变了动作专家输入，也改变了首个提议动作。轨迹最大位置差：A-B base 0.0944 m、右末端 0.0592 m；A-C base 0.0782 m、右末端 0.0669 m；B-C base 0.1509 m、右末端 0.0927 m。三组错误启动时刻不同（相差 31 至 33 次决策），但最终阶段类别相同，均为开门前误启动。该结果支持“记忆影响动作和错误发生时机”，不支持“正常记忆纠正了阶段决策”。

## 限制与结论

恢复检查对比显示机器人/物体位姿、关节位置与控制器状态一致；但首次刷新观测时，原始 `state.joint_qeffort` 最大差异为 B 对 A 1431.60、C 对 A 658.05。为满足固定首帧策略输入，B/C 的该字段采用 A 首帧值后再送入模型。差异及覆盖前数值分别保存在各组 `first_input_delta.json`。所以策略接收的首帧张量相同，但仿真刚恢复时的原始力矩观测并不一致；后续观测均来自各自运行中的仿真。这是重要混杂因素，当前结果属于预实验，不应当作正式因果消融。

此外，正常记忆仅包含门前导航阶段的 9 条历史，没有开门失败后的经验。本实验不能证明记忆能解决开门后的阶段跳步。建议正式消融前先查明 `joint_qeffort` 恢复差异来源，并确保三组从一致的物理观测出发；同时增加同条件重复，以估计仿真和动作采样方差。

**结论：** 本次实际验证到记忆特征确实进入动作专家，并使动作/错误启动时间出现差异；但三组都错误启动、都未开门，且恢复后的原始力矩观测不一致。可作为记忆干预预实验结果，不足以宣称记忆能改善 Task40 阶段决策，也尚不建议直接进入正式消融。

## 文件索引

- `comparison.mp4`：A/B/C 同步并排视频。
- `first_frame_comparison.png`：首帧并排图。
- `videos/A_normal.mp4`、`videos/B_empty.mp4`、`videos/C_wrong.mp4`：独立视频。
- `metrics/comparison.json`：首帧张量、专家特征、动作及轨迹差异。
- `metrics/{A_normal,B_empty,C_wrong}/`：逐决策观测、动作、记忆检索与任务真值；`first_input_delta.json` 记录覆盖前本体状态差值。
- `metrics/history_*.json`、`memory_compare_*.json`、`restore_compare_*.json`、`trial_*.json`：记忆、恢复和试验摘要。
- `logs/policy_rerun3.log`、`logs/sim_rerun3.log`：本次成功运行日志。此前不完整运行与启动失败日志保存在 `logs/attempt_*`、`metrics/attempt_*` 和 `videos/attempt_02_failed/`，未覆盖或删除。

## 复现命令

```bash
env PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /nvme3/projects/GR00T/official-baseline-env/bin/python -u evaluation/scripts/task40_replay_policy.py \
  --checkpoint "$PWD/Output/task40_memory_overnight/replay/original_step4000" \
  --source-run "$PWD/Output/test_01/experiment_03/checkpoints" \
  --donor-dataset "$PWD/assets/datasets/b1k2025_groot_task40_v1_200" \
  --output "$PWD/Output/test_01/experiment_03/metrics" --port 5575

env -u OMNIGIBSON_DATA_PATH OMNIGIBSON_GPU_ID=0 bash scripts/b1k_python.sh \
  evaluation/scripts/task40_replay_probe.py \
  --output "$PWD/Output/test_01/experiment_03" \
  --source-run "$PWD/Output/test_01/experiment_03/checkpoints" \
  --memory-intervention --cycles 60 --port 5575 \
  --behavior /nvme3/projects/GR00T/BEHAVIOR-1K-task-progress \
  --groot /nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1

python3 evaluation/scripts/task40_same_memory_v2_audit.py \
  Output/test_01/experiment_03 --memory-intervention
```
