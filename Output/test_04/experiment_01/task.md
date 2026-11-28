# B1K-2026 / V0_0 / Task40 训练链路工程验收

> 用户于 2026-11-28 确认：V0_0属于V0；仅训练任务40；授权查重分配实验编号。沿用已提交审阅的4步工程方案，其余配置不变，Task0不进入本次训练或闭环。正式训练预算为0。

## 身份

benchmark=b1k2026；系列V0；具体版本V0_0（API V_0_0）；test04 / exp01 / run001；run_id=v0_0__test04__exp01__run001。
查重：当前主仓库仅有test01/02，其他历史V3 checkout占用test03；本次使用未占用的test04，独立记录2026。训练与评估产物分别在training/b1k2026及evaluation/b1k2026/V_0_0中。

## 固定配方

- 通用底座 /nvme3/models/GR00T-N1.7-3B，摘要6f381d7c9916112261c5d910ed1a54668c816fa3fb83b45ceef714f9b2b8546d；全部模型权重从底座继承，不新增/重置网络，不用radio专用权重，不加入技能监督。
- 固定上游ace36d935b376fbf25cd56371e23877b95407c40，单GPU官方无记忆R1Pro，BF16、AdamW、原生cosine、clip1.0。
- 仅B1K-2026 Task40 make_microwave_popcorn，训练episodes8000..8179（180），验证8180..8199（20）；物理过滤聚合Parquet，保留原视频span，重新计算各split统计，训练只读train统计。验证集仅隔离和检查，本次不承诺离线验证指标。
- task_description完整任务语言；状态61→25，物理动作23，有效horizon16，padded40；relative动作沿用官方R1Pro。
- tune_llm=false，tune_visual=false，tune_projector=true，tune_diffusion_model=true；LR1e-4，WD1e-5，warmup0.05，dropout0.2，seed42；无LoRA。
- world_size1 × microbatch1 × accumulation1 = effective_batch1；workers0，gradient_checkpointing=false；shard_size4，episode_sampling_rate0.1，num_shards_per_epoch100。
- 累计4 optimizer steps，save_steps2；step2保存并安全暂停、退出释放资源，同run恢复到step4；保存模型、optimizer、scheduler、RNG、cursor和hash。
- 4090本机GPU1，UUID GPU-5c0dd939-b099-8f50-60a4-48f1bab85bb0；提交/恢复前查当前容量，至少30000MiB空闲，不抢占其他作业。

## 闭环与终止

最终step4 checkpoint只跑Task40 public-test index0一回合，从官方初态开始，官方post2/IsaacSim5.1、WebSocket、temporal ensemble、RTCoff、单环境单服务；固定任务语言、控制器及原生超时/成功/Q-score。保留原生JSON、视频、日志、源码/权重身份。工程执行完成与任务成功分别报告；4步不用于学习效果结论。

NaN、无梯度/参数更新、缺少训练状态或OOM时停止问题作业并留证；同配置工程修复记录revision/attempt，保留run身份，不改科学配方。

## 验收

真实有限loss、预期可训练参数进入optimizer、有非零梯度和权重更新；step2完整存档且进程释放；工作台同run恢复后optimizer/cursor和累计step前进至4；确切step4权重重载并输出Task40原生JSON/可播放MP4。已有实验、权重、结果、其他Benchmark及正在运行作业保留。不开发V2.2，不新建前端。

执行计划：workflow/plan-v001.json。实际请求：configs/training.json、configs/dataset-preview.json。进展：progress.md。所有新训练/评估均填写V0_0简短备注。
