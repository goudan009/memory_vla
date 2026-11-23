# 在线评估、记忆机制实验、归档与报告

## 评估前冻结身份

记录：模型checkpoint规范ID（v1__test01__exp12__run001__step002000）、路径/hash/训练步数、Memory模型版本、推理进程配置、策略会话ID、仿真版本、Task ID、scene instance、seed、动作chunk执行步数、max decision/physical steps、物理成功判据、控制器参数。所有配对对照必须用同一套可核查协议。

## 标准闭环记录

每次决策保存模型实际见到的三相机图像、机器人状态（底盘/双臂/升降/头/夹爪如适用）、语言指令、模型输入、产生的动作chunk、实际执行动作、随机数状态、时间戳；各仿真物理步保存关键真值（如关节、接触、物体状态）与事件；保存三相机与总览视频。对失败作业显式标注`INVALID`与原因，而不是加入成绩分母。

## 历史消融：Normal / Empty / Wrong

- 固定当前RGB、proprioception、指令、感知记忆（当只干预认知记忆）、采样RNG、模型checkpoint；清楚指明被替换的是哪个bank。
- 重新按**当前checkpoint自己的网络**回放构建memory；不能把旧权重缓存tensor当作新权重输入。
- 同时记录memory bank、检索/Top-k或实际注意力权重、Gate原始激活、有效历史贡献、门控后value、融合/缩放后条件、DiT状态、动作token和候选动作。若原版没有Top-k硬筛选，不得为记录Top-k而偷偷改动模型方法。
- 按真实Tensor含义解释Gate方向；`1-sigmoid(...)`的历史系数不可直接叫原始`sigmoid gate`。训练参数变化和前向阶段激活是两种不同数据。
- Probe是否能读出状态、Gate是否变化、动作是否有数值差、闭环是否改善分别作结论，防止把其中一个当作另一个。

## 行为判据

- 阶段选择与物理事件拆开：没有开门却触发按钮，可能是意图跳步，也可能是误触。若没有末端轨迹/视频证据，应标记“触发已确认、意图不明”。
- 抓取/门/放置/启动任务按物体状态与接触判据、视频人工核查共同判定。保留首次错误、执行精度失败、阶段切换、超时等类别，允许未判定。
- 成功率必须注明分子/分母：`n_success/n_valid`，并给置信区间。独立实验单元是回合/实例/seed，不是数千个相邻推理帧。
- 对比官方零额外训练权重只证明性能水平；要归因Memory净增益，另需无Memory同数据/相同预算的GR00T继续微调对照。

## 报告最低结构

1. `status`：`DONE` / `PARTIAL` / `RUNNING` / `BLOCKED`，并给实际job ID及验收证据。
2. 数据/模型/训练配置：任务数量与示教条数、checkpoint血缘、参数更新范围、全局batch公式、学习率/优化器/训练总步数。
3. 训练结果：曲线、梯度、权重变更、checkpoint 校验。需要严格续训的任务另附暂停/恢复记录。
4. 闭环评估：逐模型/任务/阶段成功数、失败归因、置信区间、视频索引；单任务子任务与完整任务不能混算。
5. 机制诊断：读出、门控、DiT、动作的实际数值及样本量；因果解释限于受控干预证据。
6. 资源与任务调度：逐卡时序、A100≥80%及4090/6000 Ada≥70%的显存目标时长和原因、推理QPS、有效回合/小时、OOM与仿真失败、Agent自动续派/巡检记录。使用率未达标如实写，不影响科学结论表达但要单独验收。
7. 残留问题：失败、阻塞、不能作出的结论、用户需要批准的变更、可复现实验入口。

## 建议产物

`report.md`, `MANIFEST.json`, `train_config.yaml`, `data_manifest.json`, `train_monitor.csv`, `checkpoint_index.csv`, `workbench_job.json`, `eval_trials.csv`, `video_index.csv`, `mechanism_metrics.csv`, `resource_summary.csv`, `watchdog_state.json`, `watchdog_log.jsonl`。若不适用，写N/A并解释；若任务书还要求门控分布、Probe、设备图、特定事件曲线，则额外归档，不把其强塞到所有实验里。
