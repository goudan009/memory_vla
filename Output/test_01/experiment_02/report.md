# Task40 推理与仿真重复性验证

- 日期：2026-11-19
- 起点：`experiment_01/checkpoints`，Task40 微波炉门前检查点，决策 9 / 仿真步 72
- 模型：`Output/task40_memory_overnight/replay/original_step4000`；未修改权重

## 结论

| 测试 | 结果 | 结论 |
|---|---|---|
| 固定图像、固定记忆，推理 5 次并恢复相同 RNG | 六类动作全部逐元素相同，最大绝对差为 0；每轮记忆重建均与保存银行完全一致 | 同一输入且随机数状态相同时，当前推理路径可复现 |
| 固定图像、固定记忆，连续推进 RNG 推理 5 次 | 六类动作均不再完全相同；base 最大差 0.203，右臂最大差 0.118 | 扩散动作头确实有采样随机性，RNG 不同会改变动作 |
| 固定动作、同检查点恢复后执行 3 次 | 每次执行 150 个物理步；机器人基座、关节和末端轨迹逐项相同。全状态最大差仅为微波炉关节数值噪声 `2.64e-22` | 在同一个已初始化仿真进程内，检查点恢复及固定动作执行稳定 |

## 判读

原 V2 三轮每次都通过 `task40_rebuild` 恢复保存的策略 RNG。因此，**扩散采样随机性本身不是三轮分叉的首要解释**：同一 RGB、同一记忆、同一 RNG 的复推五次完全相同。若不重置 RNG，模型输出确实会随机变化，这种采样差异在未恢复 RNG 的评估中会成为直接变量。

固定动作三轮完全重合，说明在本次同一模拟器进程、同一资产加载状态下，物理执行没有显示可测轨迹漂移。结合 V2 报告的跨轮首帧 RGB MAE 约 1.09–2.54/255，现有证据更支持：旧三轮差异来自**策略收到的图像/观测不完全相同，输入差异经策略转为不同动作**，而不是同输入时策略不稳定或同进程物理步进漂移。尚未把三轮各自保存的不同首帧逐一喂给同一策略做同 RNG 配对，因此不能量化 RGB 差异对旧分叉的因果贡献。

仿真复跑日志出现检查点对象与当前 USD 文件哈希不匹配的警告（例如 `popcorn_bag_73`）。本测试在同一当前资产加载会话内得到确定性结果，但**没有验证三个全新仿真进程之间**的确定性；上述资源版本差异意味着旧跨进程结果仍不能仅凭本测试排除进程级差异。

本测试只验证重复性，不比较正常、空或错误记忆的效果，也不评估模型任务能力。

## 实验设置与证据

### 推理重复

- 固定输入来自 `checkpoints/anchor_live_observation.npz`，SHA-256：`83fc519e13bbb85aae215110b2f6b11d373d48cc74abe13aed1c27ea2a5e5b2e`。
- 每次将策略记忆清空，再按同一前缀历史重建；每轮都与 `memory_online.pt` 做张量、时间索引、容量和计数的精确比较。
- `restore` 模式每次恢复锚点策略 RNG；5 份动作每个字段逐元素一致。
- `advance` 模式保持相同输入和重建记忆，但连续使用 RNG；5 次动作全部不同。

| 动作字段 | RNG 恢复：最大差 | RNG 连续推进：相对首轮最大差 | 连续推进：相对首轮平均绝对差 |
|---|---:|---:|---:|
| `action.base` | 0 | 0.202585 | 0.037453 |
| `action.torso` | 0 | 0.003630 | 0.000386 |
| `action.left_arm` | 0 | 0.064737 | 0.007399 |
| `action.left_gripper` | 0 | 0.007813 | 0.001831 |
| `action.right_arm` | 0 | 0.118188 | 0.028125 |
| `action.right_gripper` | 0 | 0.015625 | 0.003296 |

### 固定动作仿真

- 三轮都从相同的序列化仿真检查点恢复，且重置 Python、NumPy、Torch 和 CUDA RNG。
- 重放 `repeat_01` 决策 9–27 保存的相同动作块，共 150 个已执行物理步；策略不参与此测试。
- 每物理步记录 42 维状态向量：基座位姿、机器人关节、双臂末端位置和微波炉关节位置。
- 相对第一轮，第二、三轮基座位置、机器人关节、末端位置的最大绝对差均为 0；状态向量整体差异仅为微波炉关节的 `2.6352e-22`。

## 复现

模型服务启动：

```bash
env PYTHONPATH="$PWD/src:/nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1" \
  CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /nvme3/projects/GR00T/official-baseline-env/bin/python -u \
  evaluation/scripts/task40_replay_policy.py \
  --checkpoint Output/task40_memory_overnight/replay/original_step4000 \
  --source-run Output/test_01/experiment_02/checkpoints \
  --output Output/test_01/experiment_02/metrics --port 5573 --seed 22004
```

在服务运行时分别调用端点：

```python
from gr00t.policy.server_client import PolicyClient
c = PolicyClient(host="127.0.0.1", port=5573)
c.call_endpoint("task40_fixed_input_repeat", {"repeats": 5, "rng_mode": "restore"})
c.call_endpoint("task40_fixed_input_repeat", {"repeats": 5, "rng_mode": "advance"})
c.kill_server()
```

固定动作仿真：

```bash
env -u OMNIGIBSON_DATA_PATH OMNIGIBSON_GPU_ID=0 bash scripts/b1k_python.sh \
  evaluation/scripts/task40_fixed_action_repeat.py \
  --source Output/test_01/experiment_02/checkpoints \
  --behavior /nvme3/projects/GR00T/BEHAVIOR-1K-task-progress \
  --groot /nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1 \
  --repeats 3 --decisions 19
```

## 文件索引

- `metrics/fixed_input_comparison_restore.json`、`metrics/fixed_input_comparison_advance.json`：策略重复推理差异汇总。
- `metrics/fixed_input_{restore,advance}_*.npz`：各次推理动作。
- `metrics/fixed_action_comparison.json`：固定动作仿真状态差异汇总。
- `metrics/fixed_actions_repeat_*.npy`：逐物理步状态轨迹。
- `logs/fixed_input_client_sampling.log`、`logs/policy_server_sampling.log`：推理测试日志。
- `logs/fixed_action_sim.log`：仿真恢复与三轮执行日志，包含资产哈希警告。
- `scripts/task40_replay_policy.py`、`scripts/task40_fixed_action_repeat.py`：本次独立保存的诊断脚本副本。
