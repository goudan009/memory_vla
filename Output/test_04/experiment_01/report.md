# B1K-2026 / V0_0 / Task40 工程链路验收

状态：DONE（本次批准的工程范围）。数据、四步真实训练、完整断点暂停与同 run 恢复、最终step4重载及Task40原生闭环均完成。评估正常退出，success=0/1、Q-score=0，任务效果未通过。正式训练预算及执行量均为0。

## 身份与执行入口

- Benchmark：`b1k2026`；系列 `V0`；具体版本 `V0_0`（API `V_0_0`）。
- Test04 / Exp01 / Run001；规范 run/job ID：`v0_0__test04__exp01__run001`。查重包含当前 test01/02 和历史其他 checkout 的 test03，没有复用2025实验。
- 工作台：`http://127.0.0.1:8765/?benchmark=b1k2026`；所有作业通过现有 preview/start/resume 接口提交。
- 计划：[workflow/plan-v001.json](workflow/plan-v001.json)；任务书：[task.md](task.md)；请求：[训练](configs/training.json)、[数据](configs/dataset-preview.json)、[评估](configs/evaluation.json)。迁移计划 `docs/v1_2/workflow/plan-v031.json`。
- Agent：Codex root session / 4090-WYH / 2026-11-28。工程代码：`f075360`、`57ebcb3`、`6fe2660`；能力验收读取器 `be9f5ed`。

## 数据与模型配方

仅 Task40 `make_microwave_popcorn`。训练 episodes8000..8179，共180条、583195行；验证8180..8199，共20条、64360行。真实 Parquet 物理过滤、训练/验证场景分离、各 split 独立官方统计，训练只使用 train 统计。验证数据完成隔离与可读性检查，本次没有离线验证指标或技能标签监督。

独立数据：`/nvme3/datasets/b1k2026-v0_0-task40-test04-exp01-run001`，资源 `resource:3f17d7b73cf4c0d41e00`；manifest SHA256 `24d71f4d7ff2a8a5832e95598789a0dbc7f378aacecf6fe5e03c29c46d5217fc`。原始 Task0/40 下载数据保持不变。

通用底座 `/nvme3/models/GR00T-N1.7-3B`，摘要 `6f381d7c9916112261c5d910ed1a54668c816fa3fb83b45ceef714f9b2b8546d`。全部权重继承该底座，不使用 radio 专用微调权重；无记忆网络、无 LoRA、无新增网络。冻结 LLM/visual，训练 projector/diffusion。参数总量3144016000，可训练1620515968，optimizer state537项。

固定官方源码 `/nvme3/projects/GR00T/Isaac-GR00T-n1d7-ace36d9`，revision `ace36d935b376fbf25cd56371e23877b95407c40`；使用其 `.venv/bin/python`，Python3.10.12、torch2.7.1+cu128、transformers4.57.3、accelerate1.13.0。固定原生R1Pro状态61→25、物理动作23、有效动作窗口16、padded horizon40，使用完整任务语言。

AdamW LR1e-4、WD1e-5、warmup0.05、cosine、BF16/TF32、clip1、dropout0.2、seed42；`world_size1 × microbatch1 × accumulation1 = effective_batch1`。workers0、gradient_checkpointing=false；shard_size4、episode_sampling_rate0.1、num_shards_per_epoch100。累计4 optimizer steps，step2暂停，恢复至step4；recipe SHA256 `126ae184563e0515d07979754eab93daa3a7785a25499b564a9badc06bf69295` 始终未改。

## 真实训练与完整恢复

| 累计 optimizer step | Loss | 全局 grad norm | 本步实际 LR | 样本游标 | 采样参数更新 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.1937 | 0.696968 | 0 | 1 | 0，官方warmup首步LR为0 |
| 2 | 1.1071 | 0.612904 | 1e-4 | 2 | 3.69e-5..8.60e-5 |
| 3 | 1.0556 | 0.75044465 | 7.5e-5 | 3 | 2.52e-5..5.64e-5 |
| 4 | 1.2040 | 0.83723134 | 2.5e-5 | 4 | 8.04e-6..1.53e-5 |

`metrics.jsonl.lr` 为本步结束后 scheduler 的下一步 LR；表中列本步实际应用值。检查点保存完整模型、processor/统计、optimizer、scheduler、RNG、trainer state和样本游标。step2暂停后实际进程退出、GPU1回到既有6188MiB；同run重新载入537项optimizer状态，最大optimizer step2、scheduler last_epoch2/LR7.5e-5、逐项状态hash匹配，游标2→4且继续真实更新。最终保存完成于19:03:37，工作台 `complete / step4 / worker_alive=false`。详细证据：[pause_resume_proof.md](pause_resume_proof.md)。

保留两次工程失败及其日志：首次PID2999500在更新前触发可选DeepSpeed的CUDA_HOME探测，修复 `57ebcb3` 后由工作台同run重试；首次恢复PID3005755在新增更新前因OmegaConf bool字段收到路径字符串失败，修复 `6fe2660` 后通过同一恢复接口绑定确切checkpoint路径。原错误及源码变更hash分别在 `retry_events.jsonl`、`resume_attempts.jsonl` 中，科学配方未变。

| 检查点 | 权重身份SHA256 | 资源登记 |
| --- | --- | --- |
| `v0_0__test04__exp01__run001__step000002` | `d60948bf1b70c3ef51c6ea60afd5da352175c3481f4b485b047c141190e66683` | `resource:96379d6752e73f808ae2` |
| `v0_0__test04__exp01__run001__step000004` | `20e6ea6d2a322f6ca6b3c1de028d21a2b5063e47c0d309b87f5aa4928d53e286` | `resource:10fdae2493f9c0212bf7` |

两者均在 `training/b1k2026/v0_0__test04__exp01__run001/checkpoints/`，canonical链接指向该run的 `native/checkpoint-2` / `native/checkpoint-4`。全状态hash保存在各检查点 `b1k2026_training.json` 及 `checkpoint_index.json`。工作台保存简短V0_0训练、检查点及评估备注。

## 最终权重原生闭环

实际评估job `076b8962918c`，19:08:43启动；确切step4三分片权重已重载。Task40 public-test index0对应官方instance301、seed0；原生timeout4856，单环境单回合，WebSocket、temporal ensemble、RTCoff。独立策略端口8040，GPU1。仿真固定BEHAVIOR-1K post2 `a8247a8cc1633fe1ca0cc66aa07243d46c64f155` / IsaacSim5.1 / `/nvme3/envs/b1k2026-sim/bin/python`；未使用2025运行协议。

输出目录：`evaluation/b1k2026/V_0_0/self_trained/v0_0__test04__exp01__run001__step000004__task40__instance00__attempt01`。

评估于19:46:15正常完成，runner/evaluator返回码0；工作台job为complete，真实有效回合1，success=0/1，终态Q-score=0。单回合成功率的精确95%二项区间为[0,0.975]，不能外推模型总体性能。原生配置timeout4856，原生JSON实际报告4857步（simulator_time161.9s）；如实保留上游计数，不重写为4856。步数与录像大小不作为任务进度。

视频H.264、672×448、30fps、4857帧、161.9s，共25437478字节，完整解码退出码0。JSON、视频、job及runner通过工作台artifact GET均HTTP200，字节与本地完全一致。JSON SHA256 `675a761e640809e1635c0a657dd5dba3fb3bb5171c2b5a24c591d5598b705302`；MP4 SHA256 `deb77f069b70d5e2ce48208a0655c1fb0951d5a44245dafd687acf4ebc2febdf`。

[工作台原生录像](http://127.0.0.1:8765/api/launch/artifact?run=v0_0__test04__exp01__run001__step000004__task40__instance00__attempt01&path=videos%2Fmake_microwave_popcorn_301_0.mp4)；[视频抽样接触表](evidence/task40-step4-contact-sheet.jpg)，从0至160秒每20秒取一帧，按行阅读。

| 视频时间 | 直接可见内容 | 判断边界 |
| --- | --- | --- |
| 0s | 厨房桌椅及腕部视图 | 起始画面，不推断阶段意图 |
| 20–80s | 桌面、椅子和房间视角显著变化 | 未确认抓袋、放入或启动微波炉事件 |
| 100–160s | 桌椅、地面及柜体近景继续变化 | 具体失败阶段未判定 |
| 原生终态 | success=false，Q-score=0 | 确认任务未完成 |

baseline没有在线阶段预测头，本次未新增逐物理步真值/接触轨迹，未记录的模型意图或阶段不作推测；动作持续执行不代表任务推进。

## 资源、回归与证据

训练GPU1 UUID `GPU-5c0dd939-b099-8f50-60a4-48f1bab85bb0`，实测49140MiB；峰值38527117312bytes（35.88GiB）。总设备占用含既有进程超过70%，未通过额外任务或改batch凑利用率。短测中非首步耗时0.622/0.555s，约1.61/1.80sample/s；仅四步，不能作稳态性能结论。A100不在该已批准单卡预算中。

19:20评估GPU快照：策略约6692MiB、仿真约13238MiB，设备总占用26153MiB（另含原有6038MiB进程），利用率74%。评估显存占用约53%，低于70%目标，原因是已批准单环境/单回合预算；未添加任务或改科学配置凑占用。未观察到OOM或连接中断；评估期间保留所有其他进程。没有持续采样，不能给显存目标时长或推理QPS统计。从工作台启动至runner终态约37分32秒，含冷启动的一回合有效吞吐约1.60回合/小时，不能视为稳态基准。

终态核对：runner3027554、策略3027958、evaluator3028162均已退出；端口8040释放，GPU1回到6188MiB，既有进程保留。无需额外停止服务或删除任何检查点/结果。

后端能力/训练链路/路由/身份回归40 passed、7 skipped；原生训练恢复13 tests OK。通过fixture或静态检查的能力与真实工程验收分开登记。正式训练、多环境、评估暂停恢复、实时相机及任务性能不提前开放。

自动训练报告保留在 `training/b1k2026/v0_0__test04__exp01__run001/training_report.md`。工作台导出的 [训练证据ZIP](evidence/v0_0__test04__exp01__run001-evidence.zip) SHA256 `c8896f832c699d636eb55a911c6111672bc23814a8dd0555febd5e6a2426becc`，含13份实际配置、日志、指标、状态、恢复和检查点索引。最终闭环验收记录在 `docs/v1_2/evidence/training-acceptance-v031.json`；能力读取器实测匹配当前训练记录、恢复状态、源码、权重身份及原生回合，仅开放Task40四步工程训练、严格恢复及自训练推理/评估。

本次仅验收V0_0 Task40四步工程链路，不能据此判断训练收敛、模型泛化、任务成功率或其他任务兼容性。现有历史实验、权重及结果保留，没有开发V2.2或修改其他Benchmark运行链路。

## 主工作台验收

工程证据与报告已在main提交 `e0756ea`。重载前读取303条评估、1条本机训练、16条A100训练、104条A100推理、36条Benchmark记录，未发现活动工作台作业；Ada离线，未对其作业做停止操作。仅重启现有 `memory-vla-web-8765.service` HTTP服务，重载后PID3049461。

实测 `/api/training/options?device=local` 的 `engineering_training_verified=true`，绑定本run，`formal_training_available=false`。原生launch capabilities中Task40工程训练、严格训练恢复、自训练推理/评估为available；评估恢复、多环境、实时相机保持false，task_performance_accepted=false。全部三条历史/本次原生job仍complete，六份原生JSON/MP4经HTTP读取与本地完全一致，回合标注持久化且保留原生episode key。详细验收：`docs/v1_2/evidence/b1k2026-training-http-v031.json`。刷新没有新增训练或评估作业。
