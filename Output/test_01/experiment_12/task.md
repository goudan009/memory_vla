# Experiment 12｜MemoryVLA 源码对齐与 Task40 验证

## 2026-11-24 家具重置故障修复

用户授权尝试修复后复测。核对工作台评估 `b5e45aa5f040` 中桌椅跨回合遗留；以原实现为对照，在真实 Task40 中主动倾倒桌椅，记录 reset 边界位姿与动作前截图。修复后要求连续三次倾倒后恢复，并复用原 run005 step006000 权重在实例0、1、4各做一次短闭环调试，不重训、不替换原正式评估。执行主线见 [plan-v003](workflow/plan-v003.json)。训练、推理、评估、工作台资源调度、checkpoint、持续巡检、归档与复现均遵循 `robot-vla-experiment-ops` Skill。下方为历史任务。

> 摘要：以作者 OpenVLA 实现为基准修正 V1 双记忆与动作条件，先完成真实前后向验收，再对官方 B1K GR00T 底座进行 Task40 四卡同步训练和 Exp11 同协议评估。

采用 2026-11-23 12:00 CST 修订版。完整原任务书 `Experiment12_V1_MemoryVLA忠实复现与Task40验证_Agent任务书.md` 保留在本地；版本化流程见 [计划](workflow/plan-v002.json)。

## 对照与验收

- 锁定作者 `openvla-codebase`，核对原始写回、门控与动作注意力，保留 GR00T Flow Matching 和 B1K 接口。
- A100 四卡 FSDP FULL_SHARD，microbatch 4、累积 8、有效全局 128，官方 Task40 200 条示教，6000 次更新，保存 2000/4000/6000 完整状态。
- 用户取消旧队列后，正式训练和后续恢复均通过工作台在 A100 0–3 共卡直启。任务书要求的严格续训验收安排在首个正式存档点。
- 全参数微调，BF16 前后向、FP32 梯度归约、非重入 LLM activation checkpointing、AdamW 恒定学习率 2e-5、零权重衰减、零 warmup。
- 官方无记忆、Exp11 旧 V1 6000 与新权重按同起点闭环、固定输入 A/B/C、S1/S2/S3 Probe、Gate 和录像对照。
- 推理优先 RTX 6000 Ada，仿真优先双 4090 共四环境，全部从工作台登记。
- Experiment 13 的启动条件为 `EXPERIMENT_12_ACCEPTED=true`。
