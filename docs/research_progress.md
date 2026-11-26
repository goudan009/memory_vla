# MemoryVLA 实验进展

仓库研究如何把 MemoryVLA 的感知和认知记忆接入 GR00T N1.6，并用 BEHAVIOR-1K（B1K-2025）闭环验证历史信息是否影响操作。当前工作聚焦 Task 40 的同协议对照。

## 2026-11-23 18:08（北京时间）

| 实验 | 状态 | 已完成与验收条件 |
| --- | --- | --- |
| 11 · 旧残差记忆诊断 | 有对照结果 | 30 个共同起点中，官方 B1K 权重有 12/30 在未提前启动的情况下开门，旧 V1 的 6000 步权重为 15/30。另一个完整任务面板中，两者均为 0/30 成功。 |
| 12 · 作者 MemoryVLA 方法适配 | 正式训练 443/6000 步 | 工作台作业 `v1__test01__exp12__run005`。四卡预检实测两次优化器更新，有效 Batch 128，固定输入与随机状态下重载动作逐位一致。正式验收等待 2000、4000、6000 步检查点和三组各 30 回合的配对闭环。进展记录中新权重闭环为 0/90。 |
| 13 · B1K 全 50 任务 | 计划 | 依赖实验 12 验收，随后锁定四卡训练配置和断点续训协议。 |

实验 12 使用 200 条 Task 40 示教，对齐双路记忆与动作头的接入，作者源码钉在 `d732ea9072bc063399ccc817aed74ab172eb50be`。评估锁定了 30 个与历史对照一致的起点，机制分析另有 14 个固定输入锚点和 102 条 Probe 轨迹。

## 继续阅读

- [实验文档索引](../Output/README.md)汇总 Test 01 与 Test 02 的任务书和已封存报告。
- 研究计划和验收条件见 [实验 12 工作流](../Output/test_01/experiment_12/workflow/plan-v002.json)和[实验 13 工作流](../Output/test_01/experiment_13/workflow/plan-v001.json)。
- 方法和训练代码见[记忆融合](../src/v1_groot_memoryvla/model/memory/fusion.py)、[上游对齐预检](../src/v1_groot_memoryvla/experiment/openvla_preflight.py)及[分布式训练](../src/v1_groot_memoryvla/experiment/task21_ddp.py)。
- 评估入口见 [Task 40 仿真](../evaluation/scripts/task40_paired_sim.py)、[前向干预](../evaluation/scripts/task40_forward_worker.py)和[正式评估协议](../evaluation/evaluation.md)。
- 历史 V2 和 V3 实现可在提交 `5b55528` 查看。

进展页是带时间戳的快照。仓库提供代码、文档和版本化计划，训练数据、权重、原始日志及录像保存在实验存储。实时进展见[工作台执行手册](workbench/agent.md)。
