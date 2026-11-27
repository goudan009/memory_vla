# 16000步时实际生效的配方

本页从导出的实际工作台参数、checkpoint配置、processor及恢复记录整理。原始证据是[effective_parameters.json](training/effective_parameters.json)、[最新续跑请求](training/recorded/config/web_resume_latest.json)、[runtime-step-011000.json](training/recorded/memory_v1/config/runtime-step-011000.json)、[checkpoint config](training/checkpoint/config.json)和[processor config](training/checkpoint/processor_config.json)。

## 训练身份与配置

| 项目 | 实际记录 |
| --- | --- |
| 身份 | V1 / test01 / exp13 / run001，累计16000 optimizer步 |
| 状态与预算 | 工作台completed、保存后停止点16000；原批准总预算和最近续跑目标仍为20000，尚未完成该总预算 |
| 初始化 | 官方通用GR00T-N1.6-3B；不是BEHAVIOR1k专门微调权重，也不继承旧Residual Memory |
| 数据 | B1K任务0–49，各200示教，合计10000；无留出验证集 |
| 参数更新 | Full FT；全部列出的freeze开关为false，LoRA未启用（配置仍保留未激活的LoRA默认字段） |
| 模型参数量 | 初始训练配置记录3400431328可训参数 |
| 有效Batch | 4个A100 rank × microbatch8 × 累积8 = 256决策；每步16个episode组，每组16个决策 |
| 优化器 | AdamW；LR=2e-5；betas=(0.9,0.999)见实际源码；eps=1e-8；weight_decay=0 |
| 调度与裁剪 | constant；warmup=0；max_grad_norm=1.0 |
| 工程策略 | fsdp_accumulation=retain；SHARD_GRAD_OP；gradient_checkpointing=false；每rank 4个loader worker |
| 精度 | BF16混合精度；具体参数/梯度cast见实际训练实现，不能仅从checkpoint的model_dtype推断所有运行态dtype |
| 随机种子 | 42；续训保留四rank优化器/RNG/数据游标 |
| 保存 | 最近续跑checkpoint_interval=1000；历史最初为2000；停止点与最终步额外保存 |
| 抽样epoch | 每epoch 10000个episode组/625 optimizer步；16000步=25个完整抽样epoch+375步，不是逐帧完整遍历视频 |

## 记忆、采样和动作目标

- 协议`memoryvla_openvla_b1k_group16_raw_v2`；`memory_conditioning=openvla`是本项目路径名称，主干是GR00T。
- 同一episode在stride8决策候选点中无放回随机取16点，按时间排序；时间编码为原始帧索引除以8。组开始/结束清空记忆，微批次间保留组内历史，历史特征detach。
- 感知/认知双库各16槽，检索各2层，perceptual_dim=256，认知通道2048；`memory_update_fused=false`，写入融合前当前特征；`memory_cognitive_qk_norm=none`。
- 作者默认原始写入与论文“融合后写回”的文字差异，以及GR00T适配差异，详见[历史核对](historical/v1_paper_fidelity_review_20261124.md)，不能声称所有论文超参逐项一致。
- 训练目标是GR00T flow-matching velocity MSE；每次采样高斯动作噪声和时间`t`，target=`actions-noise`。Beta(1.5,1.0)，noise_s=0.999，1000时间桶。
- 原生action_horizon=50、max_action_dim=128；B1K processor实际监督32个动作时间点、23个有效动作维度，其余和末尾越界被mask。归一化统计已导出，基座与16000步checkpoint的statistics.json逐字节相同。
- 图像增强以processor_kwargs为准：crop_fraction=0.95、shortest edge=256，颜色扰动brightness/contrast/saturation/hue=0.3/0.4/0.5/0.08。不要误用模型config中另一组0.1默认值。
- 在线推理flow步数4，每次执行8个动作后再观测；具体闭环协议见导出的评估配置。

## 训练记录的口径

第1个epoch均值0.01541065；最近完整epoch（日志epoch=24，step15001–15625）均值0.01059733。最近100步均值0.01026337。这里仅描述训练目标，不据此判断记忆净收益或闭环成功率。

梯度日志里参数级null/缺失、FSDP局部参数片段，以及全局模块范数是不同口径。源码记录`memory_gradients`与`component_grad_norms`的路径不同；审查某模块是否真正训练时，先核对requires_grad、optimizer组、分片和日志采样时刻，不能把单个rank的null等同于模块停止学习。此包没有把稀疏监测插值补齐。

## 应以哪份代码为准

训练与该次推理声明的源码是`deployments/v1__test01__exp13__run001__resume002`，摘要`68b6d82b8ed2608046b86ab39c58442d22f122f5e46fc4e62c2dafc05324fc85`。导出的[source](source)是该快照中的模型/数据/训练/策略模块，逐文件核对了已登记的SHA256。完整源码摘要清单在[source/web_source_manifest.json](source/web_source_manifest.json)，未导出的外部库仍可根据清单核查，但不能假定本包包含全部运行依赖。

历史`ddp.json`和原始`web_launch.json`保留旧microbatch4、梯度重算和2000步保存等初始值；最近无验证续跑实际执行的是`web_resume_latest.json.worker_argv`，不是其中的外层历史`argv`。实际配置优先取worker命令、runtime记录和checkpoint，历史配方用于解释变化。最新源码不等同于本次训练源码，不自动把main上的后续修改归因到这次训练。
