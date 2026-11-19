# 标准化评测协议：b1k-groot-v1

本协议在启动实验前固定。正式结果必须同时报告 Q Score、完整任务成功率和 Task Progress，不根据实验表现临时更换主指标、选择有利实例或只报告最好的一轮。修改评分或运行协议时新增协议版本并重新配对评估，不覆盖已有结果。

## 目录和入口

```text
evaluation/
├── evaluation.md              本协议
├── README.md                  操作说明
├── scripts/                   评估、续跑、查看与记录工具
└── V_1/                       后续 V_2、V_3…；也支持自定义名称
    ├── baseline/
    │   └── groot_task22_1/     每次评估的原始结果
    └── experment_1/           同版后续 experment_2、experment_3…
        └── grootmeory_task22_1/
```

默认 `--version V_1 --group baseline`；实验显式指定 `--version V_1 --group experment_N`，版本和分组也支持自定义单层名称。旧 `--version baseline`、`--version experiment_N` 兼容映射到 V_1 下对应分组。`--model-name` 只是目录标签，不会替换 Policy Server 上的模型。每个版本的每个分组内，同模型同任务的运行编号独立递增，不复用旧输出目录。不同任务、重复实验、失败、续跑与 fast-debug 的原始结果都保留。通过网页、results.json 和 episodes.jsonl 查看，不再生成或追加 results.md；已有 Markdown 保留作历史记录。本次目录调整不改变下面的正式评估规则，历史结果及其协议记录保持原样。

## 固定的正式协议

| 项目 | 规则 |
| --- | --- |
| 任务集合 | 入口支持 B1K-2025 全部50项任务（编号0～49，需备齐对应官方测试实例）；同一对照预先固定相同集合，逐任务全部报告；默认仍为Task21，不自动遍历全集 |
| 测试实例 | 每任务实例0～9，各1个完整有效 Episode；记录真实 instance_id，不用 env_idx 代替 |
| 随机性 | 默认 base seed=0、seed-mode=sequence；每回合实际 seed 落盘；Baseline/实验使用相同实例、槽位分配与实际 seed |
| 默认运行方式 | single1、GPU1；single2/dual1/dual2 需显式选择；配对对照使用同一种分配方式，不能假设换环境数量后 seed 映射相同 |
| 动作执行 | action chunk=8，保持原动作定义、控制频率和 PhysX 参数 |
| 输入 | 保持 GR00T 原三路策略相机和观测预处理，不因快评/并行改变输入 |
| 终止 | 原生 B1K 成功判定与 timeout；official-eval 不允许设置短 runner 上限 |
| 录制/显示 | 默认保存16倍速视频：30 Hz动作、每24动作步采一帧、20 FPS播放；可显式关闭，viewer 保持现有行为；配对运行一致，实际配置落盘；历史/续跑沿用已保存配置 |
| Policy | 固定 checkpoint、代码版本、训练数据/训练步数；在运行前写入版本说明，并保留服务地址与 run-tag |
| Memory | 有状态模型必须在每个 Env/Episode 边界 reset，不能跨环境或回合共享记忆；使用前需验证实现，版本目录本身不提供 reset |

默认 sequence 的 seed=base_seed+env_idx+槽位内回合序号；fixed 为所有回合同一 seed；random 从记录的 base seed 派生。`--seed random` 仅用于探索，不作为默认正式配对协议。改变种子策略时必须让 Baseline 同步重跑，不能把未配对成绩直接比较。

## 预先固定的指标

1. **主要指标：平均 Q Score（0～1）**。直接使用当前锁定版本的原生 TaskMetric：完整成功为1；否则统计初始未满足、结束时满足的目标条件，占该组全部目标条件的比例，多组可替代目标取最大值。不从进度或 success 推算 Q。
2. **共同报告：完整任务成功率**。有效正式回合中原生 success 为真的比例。`status=complete`/`ok=true` 只表示评估流程完成。
3. **补充指标：平均 Task Progress（0～100%）**。当前 GR00T/task-progress 分支的细分检查项“曾经达成”的累计比例，不能称作 Q Score 或完整成功率。
4. **完整性**。报告计划/有效/无效回合数、Q覆盖数、真实实例/种子及终止原因。有效 Episode 的 Q 缺失则显示未记录，不补0。`q_score_episodes` 必须等于 `completed_episodes` 才是完整 Q 成绩。
5. **效率**。性能比较另报动作步/s、墙钟时间、初始化时间和资源峰值（实测有记录时）；诊断短窗效率不与完整 Episode/h 混用。

按任务先平均有效完整回合，跨任务使用预先固定任务集合的等权平均。保留全部重复运行，并明确报告均值与重复数量；没有足够重复时不声称统计显著。任务未完整成功是有效的失败样本，不能因为 Q 低或 success=false 删除。仿真崩溃等无效回合保留并单独报告，补测需沿用相同实例/种子。

## fast-debug 与续跑

fast-debug 只用于动作检查、性能和调试，可设 `--debug-max-steps`，默认1200动作步。结果明确标为调试，正式成功率与平均 Q 为 null；即使碰巧成功也不能混入 official-eval。

续跑从已落盘 Episode 边界恢复，不恢复未完成回合的物理状态。当前只支持 single1/single2 的 official-eval、sequence/fixed，且每个槽位完成数量一致。保存原协议和 seed 映射，向原 JSONL 追加，不删除失败记录。新启动运行保存本文件 SHA256，续跑沿用启动时的协议记录；历史未记录协议的运行标为 legacy-unrecorded，不追认合规。

## 数据与审计

每回合结束、自动 reset 前立即 fsync 写入 episodes.jsonl，包含 task、GPU/env、instance_id、seed、success、q_score、task_progress、valid、sim_steps、duration_s、terminated/truncated、native_termination。results.json 是可更新快照；gpuN.log 用于排错；网页从这些文件读取汇总和回合数据。

kill -9/断电可能使最终状态未更新，但已 fsync 的 Episode 保留，不能伪造缺失指标。正在运行的旧目录不迁移写入路径，网页同时读取旧布局，结束后执行 organize_results.py 完成归档。2026-11-09 起取消 Markdown 汇总，不影响评分、实例、种子或终止规则；已启动的旧 Launcher 仍可能按已加载的逻辑在结束时追加一行。
