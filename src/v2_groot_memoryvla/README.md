# V2：本体条件化记忆与检索

本目录独立组合现有 V1，无需修改 V1 服务、训练脚本或权重。已同步 2026-11-11 最新 V1 的 direct 条件、可变尾窗、有效监督权重和训练图像增强。网络说明见 [model.md](model.md)，实现与验证范围见 [架构调查](v2_architecture_audit.md) 和 [实现报告](v2_implementation_report.md)。

## 实际网络

```text
RGB + 任务语言 → 冻结 Eagle → 原 V1 视觉/认知记忆
                               ↓
                   direct：[B,244,2048]（新初始化）
                   residual：[B,S,2048]（旧权重）
                               ↓
URDF + 当前 q/qvel → BodyEncoder → 当前 Body 残差（a/b/c）
                               ↓
当前 VL + Body* → Query → 检索过去具身历史 → 检索残差（b/c）
                         ↑
      历史 Body + 视觉/语言 + 此前已执行命令（同帧对齐）
                               ↓
              当前条件中的认知 Token → 原 vlln → 原 Action Expert
                               ↓
               [B,50,128] → 原 processor → [B,32,23]
```

`Body*`：V2b 的 Query Body 输入为零；V2c 接收当前 Body。两者模块、参数量与历史写入规则一致。direct 时 Body/Memory 残差加在新条件唯一认知 Token，residual 时加在原 user EOS；Query 的视觉/语言来源仍为当前原始 Eagle 输出。历史先读后写，只读过去。

新增 bank 的 `memory_capacity` 默认 64。原 V1 两个 bank 的 `memory_length` 独立配置：从无记忆 GR00T 初始化时默认 direct/64；从已训练 V1 初始化时继承 checkpoint（例如旧权重 residual/16）。配置中 `memory_conditioning=null`、`memory_length=null` 表示自动选择；显式设置与已训练权重冲突会报错。旧 V2 checkpoint 保存的容量 16 保持不变。

## 环境与快速检查

在仓库根目录执行；复用现有环境，不重新安装依赖。

```bash
export PYTHONPATH=src
export V2_PY=src/v1_groot_memoryvla/.venv/bin/python

OMP_NUM_THREADS=2 "$V2_PY" -m pytest src/v2_groot_memoryvla/tests -q
```

五种配置在 `configs/{baseline,v1,v2a,v2b,v2c}.json`。Baseline 应使用原始 GR00T checkpoint；其余可从训练好的 V1 初始化。V1 权重本机只读副本为 `checkpoints/v2_init_v1_task21_step003000`。

默认 pilot 使用最多 4 个连续决策、12 步固定噪声拟合、2 步随机噪声短训；只训练新增层，冻结复用的 V1。固定拟合和动作干预使用 eval 特征；短训默认每步重新增强图像并编码冻结 Eagle。`--no-image-augmentation` 可关闭增强作单独诊断。输出目录必须不存在。

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 "$V2_PY" -m v2_groot_memoryvla.pilot \
  --checkpoint checkpoints/v2_init_v1_task21_step003000 \
  --baseline-checkpoint checkpoints/groot_n1d6_behavior1k \
  --dataset assets/datasets/b1k2025_groot_task21_diagnostic \
  --output outputs/v2_groot_memoryvla/my_five_arm_pilot
```

单独配置；把配置路径替换为另外四种即可选择其他实验：

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 "$V2_PY" -m v2_groot_memoryvla.pilot \
  --checkpoint checkpoints/v2_init_v1_task21_step003000 \
  --config src/v2_groot_memoryvla/configs/v2c.json \
  --output outputs/v2_groot_memoryvla/my_v2c_pilot

# Baseline 使用原始权重
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 "$V2_PY" -m v2_groot_memoryvla.pilot \
  --checkpoint checkpoints/groot_n1d6_behavior1k \
  --config src/v2_groot_memoryvla/configs/baseline.json \
  --output outputs/v2_groot_memoryvla/my_baseline_forward
```

`--raw-dataset assets/datasets/b1k2025` 可读取配对原始 256 维状态中的真实 qvel，会校验 Episode、时间、q 和 action。缺少原始数据时使用完整 30Hz 轨迹的向后差分，首帧速度 mask=0。`--group N` 切换示范窗口。已有可信 Eagle 缓存可用 `--sequence-cache .../fixed_sequence.pt`，用于固定拟合和干预；开启增强的短训仍读取原数据并重编码，要求选中窗口与缓存 Episode/帧匹配。

小规模联合训练可增加 `--train-v1 --lr 1e-5`，此时原 V1 memory 与 action expert 一并训练，保存完整 portable checkpoint。从原始 GR00T 开始新 direct 实验时也建议联合训练，所有对照须使用同一起点；pilot 默认冻结随机新记忆仅适合接线检查。冻结已完整加载的 V1 时可保存约十余 MB 的 extension checkpoint，依赖 manifest 中的原始 V1 路径与校验和；新初始化过 V1 记忆时必须完整保存，避免重载时随机重建该部分。

## 已完成的实际验证复现

此前 residual/16 V2b/c（FP32 attention reduction）结果目录：`outputs/v2_groot_memoryvla/pilot_attention_fp32_20261111/`。V2a、Baseline、V1 的实际验证在 `outputs/v2_groot_memoryvla/pilot_final_20261111/`。这些结果早于本次同步，不能用于宣称新 direct/64/增强配置已完成真实权重训练。

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 "$V2_PY" -m v2_groot_memoryvla.diagnose \
  --checkpoint outputs/v2_groot_memoryvla/pilot_attention_fp32_20261111/v2c_checkpoint \
  --sequence outputs/v2_groot_memoryvla/pilot_attention_fp32_20261111/fixed_sequence.pt \
  --expected outputs/v2_groot_memoryvla/pilot_attention_fp32_20261111/v2c_actions.npz \
  --output outputs/v2_groot_memoryvla/my_fresh_diagnosis.json
```

该命令检查新进程恢复、FK、Query/Attention/动作影响、融合耗时，并离线读取两次已有真实观测抓包跑完整 RGB 推理。不会连接服务器、执行机器人动作或启动仿真。

## 数据、训练与在线接口

训练可复用 `V2HistoryDataset` 和 `V2HistoryCollator`；它们继承最新 V1 的真实短尾窗、未来标签 mask 和 `group_lengths`，按同一帧增加 `body`。`V2Model.forward(inputs)` 与 `get_action(inputs)` 共用 `fuse`；缓存训练使用 `forward_cached`。pilot 按有效动作元素加权；普通 PyTorch optimizer 可训练所有新参数，Eagle 保持冻结/eval。跨窗 Episode 连续训练需要使用 `train_sequence` 和正确的 start/end；现有 V1 正式 DDP worker 尚未自动接管 V2，pilot 是有边界的小规模检查。

```python
from v2_groot_memoryvla.model import V2Model
from v2_groot_memoryvla.config import V2Config

model = V2Model.from_v1(
    "checkpoints/v2_init_v1_task21_step003000",
    V2Config.load("src/v2_groot_memoryvla/configs/v2c.json"),
    "assets/simulation/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf",
).cuda()
# 冻结 Eagle；model.base.memory / action_head 默认仍可训练。
# loss = model(batch["inputs"])["loss"]
# loss.backward()
```

独立推理接口：

```python
from v2_groot_memoryvla.policy import V2Policy, V2SimPolicyWrapper
policy = V2SimPolicyWrapper(V2Policy.load("path/to/v2_checkpoint"))
actions, info = policy.get_action(flat_b1k_observation, options={
    "session_id": "v2-test", "env_ids": [0], "episode_ids": ["episode0"],
    "observation_timestamps": [8 / 30],
    "executed_actions": [last_command_actually_sent_to_env],  # [23]
    "executed_action_timestamps": [7 / 30],
})
```

首帧允许没有此前命令；之后缺失反馈会报错。反馈应为**实际送进环境的最终 23 维命令**，不是 processor 归一化动作、整段预测或未执行缓存。时间戳用图像与状态采集时刻。原 adapter 的底盘变换只做一次，仍需保留。`V2SimPolicyWrapper` 从当前抓包额外传递实际 qvel，防止原 V1 modality 选择把它丢掉。没有速度时保留 mask，线上不会用未来帧计算速度。

原 V1 evaluator 尚未自动发送这些新增反馈字段；本次没有改动它，也没有启动新服务。接短程仿真前，需要在 V2 专用调用方传入时间戳和最后已执行命令。两次抓包离线验证已覆盖该接口。
