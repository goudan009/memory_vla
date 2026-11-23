# Task40 记忆检索与阶段选择诊断

## 当前结果

本次完成真实开门历史审计、固定输入的记忆前向干预，以及 10 次有效的空记忆闭环试验。十次试验共享同一检查点、策略输入、仿真 RNG 和模型权重，只改变策略 RNG。另有一次不同恢复路径的试跑首帧 RGB 不同，明确排除，不计入十次。

| 排查项 | 结论 | 证据与问题 |
|---|---|---|
| 真实开门历史 | 可行，已核验物理真值 | `metrics/opened_history_audit.json` 从原 Task40 stage JSONL 的 8,100 行中发现 6 次 `Open: False -> True`；例如 instance 9 / seed 16009 在 action step 464 到 472 间变为打开。门关节位置随之为正值。 |
| 开门历史图像 | 部分可行 | 每集视频 768x280、269 帧、20 fps；运行配置记录每 24 仿真步渲染一帧，覆盖约 6,456 步，与 6,476 步回合长度吻合。录像是三相机横向拼接画面，不是逐策略决策的原始 RGB 文件。 |
| 阶段历史前向对照 | 已执行，使用稀疏重建历史 | `metrics/opened_memory/comparison.json` 和 `slot_comparison.json`。正常、空、开门历史感知库、开门历史认知库、双库五种设置共享相同首帧输入与 RNG；另有同记忆重复对照。开门历史使用视频帧 20–28，间隔为每帧 24 仿真步。 |
| 空记忆阶段选择分布 | 已完成，10 次有效运行 | `metrics/empty_memory_sweep_summary.json`：5/10 物理开门，4/10 门关闭时 `ToggledOn=True`，1/10 两状态均未发生。10 次首帧输入逐字段相同，最大差 0；恢复控制器状态相等，位姿/关节最大误差 `1.2e-7`。 |
| 正式记忆消融准备度 | 部分具备 | 固定输入前向和空记忆随机基线已具备；但面板没有独立几何接近/接触真值，开门历史又是稀疏视频近似重建，仍需补强标注后再做正式因果结论。 |

## 已核验的开门历史

数据源为 `outputs/overnight_memory_task40_20261114/evaluation/a6000/confirm_s16000_base_remote/fixed_b1k_seeded/task40_eval01/`。物理真值来自 `stages/task40_gpu0_env0.jsonl`，录像路径和 episode 身份来自 `episodes.jsonl`。发现的跃迁包含 instance/seed：1/16001、4/16004、5/16005、6/16006、9/16009（该回合两次）。

最早的候选是 instance 9 / seed 16009，`Open` 在 step 464 到 472 之间从 False 变为 True；回合视频为 `videos/gpu0/episode_010_instance9_seed16009_gpu0_env0_s0.mp4`。候选帧索引可按 `action_steps / 24` 估算，但采样间隔是三个 8 步策略块，不能还原原始在线每次推理的 RGB 序列，也不能宣称由这些图像重建出的记忆等于在线记忆。

## 前向结果

当前 RGB、本体状态、任务指令及策略 RNG 固定。A 正常记忆精确重建自原 door-front 前缀；B 清空双库；C/D/E 将物理真值已打开之后的 9 帧分别编码并替换感知库、认知库、双库。重建开门记忆使用锚点本体状态作为每帧状态输入，因为旧录像没有逐帧 proprioception；因此这是阶段视觉历史敏感性诊断，不是对原始完整历史的精确回放。

| 对照 | 槽位/专家输入差异 | 首 8 步预测动作 MAE / 最大差 | 解释 |
|---|---:|---:|---|
| A 对 A 重复 | 专家输入最大差 0 | 0 / 0 | 同输入、同记忆、同 RNG 完全一致。 |
| A 对空记忆 B | 专家特征 MAE 0.22585 | 0.00278 / 0.05469 | 记忆是否存在会明显改变融合特征。 |
| A 对开门历史感知 C | 感知槽位 MAE 0.08786、余弦均值 0.99611；专家特征 MAE 0.000639 | 0.000827 / 0.015625 | 感知库槽位改变，但注意力/融合后差异大幅缩小。投影后感知残差差 MAE 为 0.000741。 |
| A 对开门历史认知 D | 认知槽位 MAE 0.34341、余弦均值 0.99071；专家特征 MAE 0.00000133 | 0.000847 / 0.015625 | 认知槽位确实不同，但传到动作专家输入的差异近乎消失。 |
| A 对开门历史双库 E | 专家特征 MAE 0.000640 | 0.000861 / 0.015625 | 变化主要由感知分支贡献；认知分支未带来可观的专家输入增量。 |

这些是预测动作张量差异，不是把手/按钮的阶段类别判断。幅度小于空记忆对照，且不能据此宣称阶段选择发生改变。固定 RNG 的同记忆重复完全一致，排除了本次前向差值由普通采样抖动造成。十种子闭环基线现已完成，但它只测空记忆下随机种子的结果分布，并不证明记忆优劣。

## 空记忆闭环结果

每轮从同一个 Task40 door-front 检查点（instance 4、seed 22004、决策 9 / step 72）恢复，先精确重建并校验原在线记忆，再清空 perceptual/cognitive 两库。十个有效策略种子为 73002–73010 与 73012；首帧 RGB 和状态张量逐元素相同。恢复位姿、物体位置、关节最大误差 `1.2e-7`，控制器状态全相等。门真值从 `Open=False` 变为 `True` 5 次；`ToggledOn=False→True` 且门仍关闭 4 次；1 次在 60 决策内两者都未发生。

门关闭时启动是物理结果，不自动等于启动意图。四次提前 Toggle 的试验中，指尖最接近门 AABB 为 2.1–4.9 cm，并出现微波炉 `base_link` 接触，没有门体接触；因此归类为“提前启动/阶段结果异常，意图不确定”，不能仅凭 `ToggledOn` 宣称机器人主动去按按钮。日志未单独记录面板 AABB、按钮接触和末端目标方向，故不能可靠地分类为明确“尝试启动”。`Open=True` 可确认开门完成，但它本身也不是独立意图标签。

种子 73002 提前 Toggle（第 13 决策）；73003 60 决策无事件；73004 开门（第 19 决策）；73005 提前 Toggle（第 16）；73006 开门（第 21）；73007 提前 Toggle（第 16）；73008 开门（第 33）；73009 提前 Toggle（第 20）；73010 开门（第 17）；73012 开门（第 24）。逐决策动作、末端/指尖轨迹、接触和物理真值保存在 `runs/batch_02` 至 `batch_04`、`batch_06` 的 `metrics/` 中。

种子 73011 的一次试跑走了不同恢复路径，三路首帧 RGB 与锚点图像最大像素差 164，已排除并保留为 `videos/excluded_seed_73011_rgb_mismatch.mp4`，不计入十次统计。

## 阻碍与最小后续步骤

1. 若要声称重放真实在线记忆，需补录原始逐决策 RGB、历史 proprioception 及在线记忆快照；当前稀疏视频仅支持近似阶段视觉干预。
2. 增加控制面板和按钮的独立几何距离/接触记录，结合末端方向与夹爪命令，才能区分明确阶段意图和误触结果。

## 复现

开门历史前向：

```bash
env PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /nvme3/projects/GR00T/official-baseline-env/bin/python -u \
  Output/test_01/experiment_05/scripts/diagnose_opened_memory.py \
  --video outputs/overnight_memory_task40_20261114/evaluation/a6000/confirm_s16000_base_remote/fixed_b1k_seeded/task40_eval01/videos/gpu0/episode_010_instance9_seed16009_gpu0_env0_s0.mp4 \
  --frame-start 20
```

历史真值审计：

```bash
python3 Output/test_01/experiment_05/scripts/audit_opened_task40_history.py \
  --run outputs/overnight_memory_task40_20261114/evaluation/a6000/confirm_s16000_base_remote/fixed_b1k_seeded/task40_eval01 \
  --output Output/test_01/experiment_05/metrics/opened_history_audit.json
```

`metrics/opened_history_audit.json` 是审计命令的结构化输出；`metrics/opened_memory/` 保存各分支张量与对比指标；`logs/opened_memory_forward.log` 保存前向运行日志。历史重建使用的原始录像位于上述 `outputs/.../episode_010...mp4`。后续补齐的十次有效空记忆闭环录像另见下文。

空记忆闭环复现使用 `scripts/task40_empty_memory_seed_server.py` 提供策略 RNG 分支，仿真端沿用 `evaluation/scripts/task40_replay_probe.py` 的检查点恢复与真值记录。汇总使用：

```bash
python3 Output/test_01/experiment_05/scripts/summarize_empty_memory_sweep.py
```

`comparison.mp4` 对比提前 Toggle、60 决策无事件、物理开门三种结果；`videos/trial_01...trial_10` 是十个有效独立视频，另保留一个被排除的 RGB 错位样本。
