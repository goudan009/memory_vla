# Exp13 训练工程加速

用户授权以更多显存换速度。保留同一`v1__test01__exp13__run001`、4×A100、Full FT、50任务10000条、Batch256、seed42、LR2e-5/AdamW/恒定scheduler、原记忆与随机有序group16采样、20000步目标和2000步存档周期。计划见v006/v007，原纯净初始化和科学配置保存在原manifest/ddp.json中。

## 工程对照

| 作业 | 策略 | 每rank microbatch×累积 | 第二步耗时 | 最大allocated | 验收 |
| --- | --- | --- | --- | --- | --- |
| 原正式作业 | FULL_SHARD，每microbatch同步，LLM重算 | 4×16 | 正式25–34步均值41.642秒 | 20.63GiB | 原链路已验收 |
| 临时run998 | SHARD_GRAD_OP，前15次no_sync，LLM重算 | 4×16 | 17.104秒 | 28.13GiB | 四rank全模块更新、完整状态恢复、重载动作逐位一致 |
| 临时run997 | 同上，关闭LLM重算 | 4×16 | 15.731秒 | 30.02GiB | 同上 |
| 临时run996 | 同上，增大microbatch | 8×8 | 13.978秒 | 37.45GiB | 同上 |

这些第二步数字只用于工程筛选，不代替连续训练吞吐。初始步包括模型/视频缓存预热，不计稳态。原mask归一化loss及每step的256个决策保持；改变归约顺序和microbatch分块后不保证逐位相同的随机训练轨迹，方法、采样分布和optimizer预算不变。

## 正式谱系恢复

工作台在34步保存`v1__test01__exp13__run001__step000034`，随后`resume-001`从完整模型、Adam分片、scheduler、四rank RNG与采样位置34恢复。四rank的Adam moment形状匹配，35步真实更新完成；此验收取代原计划的2000步首次暂停恢复。源码digest `f127c947e1efca76d304f687ce3e3e9b547a1d87d18ed0e9cd986345fd50c97c`，对应训练代码`489cacb`及工作台`2806335`。

连续36–46步均值20.634秒，暴露视频预取瓶颈，CPU等待逐渐增加，四rank同步也跟随最慢loader等待。机器224逻辑CPU、约412GiB可用RAM。正式46步再次通过工作台完整保存，随后工程代码`4c64800`将retain策略每rank CPU worker由2增加到4。0/2/4 worker取得相同组顺序、帧、时码、标签；相同每组seed和FIFO顺序保留。第二次恢复的源码digest `68b6d82b8ed2608046b86ab39c58442d22f122f5e46fc4e62c2dafc05324fc85`，路径`deployments/v1__test01__exp13__run001__resume002`。47步真实更新完成后，第二次严格恢复验收通过。

当前科学配置仍保留原文件；工作台根据`web_resume_latest.json`及`runtime-step-XXXXXX.json`显示实际microbatch8/累积8、SHARD_GRAD_OP、关闭重算及loader worker数。后续普通工作台续跑沿用最近工程源码和选项。累计optimizer步数不归零。

## 归档与持续巡检

三组临时训练已先归档，再通过工作台delete移除并复核；临时输出（含旧run999）可从A100的`/nvme2/memory_vla_exp13_engineering_archive/`恢复。它们未用于正式权重初始化。

为保留额外34步原始断点并容纳正式10份约38GiB存档，34步断点移至`/nvme2/memory_vla_exp13_retained_checkpoints/`，原路径保留符号链接；COMPLETE marker SHA256为`b4cc47bfdba213605bfc629b10ce152552306ee0463191ebd0438d943382b3af`。46步和后续常规存档保留原训练盘。没有删除正式断点。

两分钟巡检使用原工作台API、GPU状态与checkpoint完成标记，持久systemd用户timer已启用且Linger=yes。旧自动2000步暂停/恢复动作已移除，巡检尊重用户的夜间暂停和手动续跑，不提交队列。证据在`evidence/optimization_20261125/`，恢复证明在`pause_resume_proof.json`。

## 连续吞吐验收

第二次恢复后取48–57步共10个连续optimizer步：均值13.162秒，中位13.539秒，范围9.798–13.807秒。与原25–34步41.642秒相比快3.164倍；最大rank预取等待均值0.845秒。四rank在10步内均完成每rank64/全局256决策，loss完全一致，loss/gradient norm全部有限，无OOM或训练异常。

实际每卡训练进程显存约49.5–50.5GiB，GPU0连同既有PID1067495总约59.7GiB；详细进程归属见`resource_summary.csv`，不是把其他任务显存算给本实验。训练allocated峰值约37.4GiB，PyTorch reserved峰值各rank约48.8–49.8GiB。采用本轮已验收吞吐最快的8×8配置，保持原全局Batch；显存许可用于提升实际吞吐。

按这10步估算，20000步活跃训练约73.1小时（3.05天），每2000步约7.31小时；若每天只训练8小时，约9–10晚。暂停时间及存档另计，视频解码与共卡负载变化会影响后续均值。没有把13.978秒的单步短测或9.798秒的最快一步当作最终吞吐。

工作台104项、训练恢复与采样并发17项测试，共121项通过。完整验收数字见`evidence/optimization_20261125/acceptance.json`。正式作业已在同一工作台继续运行，20000步目标持续推进中。
