# 训练执行、完整checkpoint和续训验收

## 训练启动前核查

- 检查数据：官方数据任务ID、每任务示教条数、总条数、是否允许使用自主rollout、采样频率/任务比例、训练标签与动作归一化版本；生成 `data_manifest.json`。
- 检查模型：底座权重hash、Memory上游commit与本项目commit、冻结与可训练参数清单、optimizer参数组、动作专家输入/输出token、维度和掩码、历史memory隔离与清零时序。
- 检查科学条件：任务书锁定的Batch、LR、scheduler、WD、loss、优化器epsilon、FSDP策略、梯度累积、模型精度、训练步数、随机种子；只从源码和真实配置读取，未指定内容不编造“官方默认”。
- 对于“忠实复现”的任务，要求上游方法对照表；仅工程适配的地方写原因、张量前向、数值差异和梯度路径。科研改动归独立版本，不混在工程迁移之中。

## Batch和训练规模

公式：`effective_batch = world_size × micro_batch_per_rank × gradient_accumulation_steps`。

例子（仅当任务书为4×A100、128时适用）：

- 4 rank × microbatch 4 × accumulate 8 = 128
- 显存不足时：4 × 2 × 16 = 128；或4 × 1 × 32 = 128

必须审计训练库中 `global_batch_size` 字段的真实定义。有些代码把它作为**累积前跨卡batch**，此时设置16并累积8，最终128。保存实际world size、每rankbatch、积累次数、有效样本/step、样本被mask比例和采样器状态，避免标签有效数/动作chunk混淆。

## Full FT 与显存

- 方法任务要求Full Fine-tuning时，先实测 FSDP FULL_SHARD、激活检查点、BF16、合法的CPU offload、低microbatch。不得因为显存压力悄悄冻结Eagle/DiT、改LoRA或只训练某层。
- 若配置无法放入4卡80GB或吞吐不可接受，停止并输出内存剖析（optimizer state、activation、参数/梯度、网络通信），标为`BLOCKED_FULL_FT_RESOURCE`，由用户决定调整科学配置。

## checkpoint 命名及训练运行身份

每个checkpoint采用`v2_1__test02__exp01__run001__step003000`式五段结构；详细规则见[checkpoint-naming.md](checkpoint-naming.md)。小版本与实验编号分开，独立训练与恢复尝试分开。保存稳定训练ID、resume_event_id、attempt_id；累计optimizer step不中断归零。步数、保存周期和数据划分来自当前任务书。

## 必须保存的训练状态

严格续训及以下验收仅适用于计划累计超过 2,000 个 optimizer steps 的训练。小规模验证训练按任务书保留评估权重。

每个可严格续训checkpoint应保存：

- 模型参数与分片信息/配置；optimizer state以及其参数组映射；scheduler state；global optimizer step；AMP/scaler状态（如果适用）。
- 各rank torch/CUDA/numpy/Python RNG；数据采样器、epoch、样本游标及loader重启策略；梯度累积、有效全局Batch和数据manifest。
- Git commit、基础权重身份、模型实现版本、依赖/环境摘要、保存时间、校验hash、checkpoint完成标志。

若要求支持**任意时刻暂停**，在optimizer更新边界安全完成当前积累并存档；不能在积累一半时将一个不含局部梯度的checkpoint误当完整续训点。

## 暂停/恢复验收

1. 从原工作台发起任务指定配置的训练，并记下 optimizer step、学习率、optimizer 状态摘要和训练采样位置。
2. 点击暂停；观察工作台作业状态变`PAUSED`，GPU进程按服务策略释放，checkpoint标志完整。
3. 从工作台点击继续。其他任务运行时也核查是否可以安全共存。有容量时直接恢复，容量不足时保留在可自动续派队列。按原设备规模和有效 Batch 恢复同一训练谱系、优化器、scheduler、RNG、采样器和累计步数。
4. 继续实际执行若干optimizer step，记录前后step、优化器状态和样本位置；保存`pause_resume_proof.md`与截图/日志。
5. 重载权重前向比较只验证权重恢复，不等于严格续训；分开标注。

## 训练状态诊断

每个正式checkpoint至少记录总Loss和按底盘/左右臂/躯干/夹爪拆分的Loss，grad_norm、实际参数更新量、LR、训练时间、samples/s和OOM次数。标注冻结层不存在梯度是预期还是缺陷。Loss不等价于闭环成功率；如果曲线下降但动作退化，要结合官方初始化checkpoint同样本误差与真实rollout判断。
