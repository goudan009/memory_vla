# V1.2 训练链路工程验收方案（待确认）

状态：实现与 CPU 接口验证已完成，尚未提交 GPU 训练。此文件是可审阅的建议，不是已获批准的实验任务书或正式训练配置。计划见 [v029](workflow/plan-v029.json)。

2026-11-28 命名更新：用户已确认 `benchmark=b1k2026`、baseline 系列 `V0`、当前具体版本 `V0_0`（接口规范值 `V_0_0`，run 前缀 `v0_0`）。V0 的后续小版本可用于代码或训练参数对照，但不自动登记新版本。本文件以下建议配方尚待确认；命名确认不改动历史作业，也不代表启动正式训练。

目标是验证通用 GR00T N1.7 → 2026 示教 → 参数更新 → 完整 checkpoint → 暂停释放进程 → 同 run 恢复 → 自训练权重推理 → 原生仿真 JSON / 视频。4步只能验收工程链路，不能检验任务学习效果。正式训练预算仍为0，不开发 V2.2，不修改其他 Benchmark 或前端。

## 建议的唯一实跑配方

| 项目 | 待确认值 |
| --- | --- |
| 方法 | 已登记的 `V0_0`（`V_0_0`），属于2026的 `V0` baseline 系列；原生无记忆，无新增网络或技能监督 |
| 身份 | 用户指定 test / experiment / run，或明确授权 Agent 查重后分配未使用编号；当前为空，不猜编号 |
| 初始权重 | `/nvme3/models/GR00T-N1.7-3B`；实体摘要 `6f381d7c9916112261c5d910ed1a54668c816fa3fb83b45ceef714f9b2b8546d` |
| 上游 | `/nvme3/projects/GR00T/Isaac-GR00T-n1d7-ace36d9`，revision `ace36d935b376fbf25cd56371e23877b95407c40` |
| 数据 | Task0 / Task40 各200 episodes；原始下载保持不变 |
| 划分 | 各任务按 episode ID 排序，前180训练、后20验证；train `0..179, 8000..8179`，validation `180..199, 8180..8199` |
| 划分核对 | 只读检查通过：360训练 / 40验证；每任务200个不同实例，无 task+instance 交叉。尚未实际物化该划分 |
| 语言 | `task_description`，使用 `tasks.jsonl` 完整指令，并将训练与评估语言绑定 |
| 归一化 | 每个划分物理过滤聚合 Parquet，单独运行固定官方统计；训练只用 train 统计，不复制100任务原统计 |
| 冻结 | `tune_llm=false, tune_visual=false, tune_projector=true, tune_diffusion_model=true`；这是建议采用的官方 FinetuneConfig 默认范围，不等于已批准范围 |
| 动作 | 官方 R1Pro relative action；有效 horizon16、模型 padded horizon40；物理动作23、原始状态61→25 |
| 优化 | AdamW (`adamw_torch`)，LR `1e-4`，weight decay `1e-5`，warmup ratio `0.05`，原生 cosine scheduler / BF16 / clip1.0 |
| 其他原生设置 | state dropout `0.2`，seed42；旋转/颜色配置沿用官方 FinetuneConfig 的 `None` 语义，不另改增强 |
| Batch | 单GPU × microbatch1 × accumulation1，有效 Batch1；loader workers0；gradient checkpointing关闭 |
| 数据装载 | shard_size4、episode_sampling_rate0.1、num_shards_per_epoch100；保留官方分片采样，分片处理同步并隔离 RNG |
| 更新预算 | 累计4 optimizer steps；step2安全保存暂停，确认进程释放；同 run 恢复到step4；保存周期2，不自动删除旧 checkpoint |
| 设备 | 本机4090上有足够容量的一张卡；当前候选GPU1，提交前重新查 UUID 与进程，至少30,000 MiB空闲。不抢占其他作业 |
| 闭环 | 最终step4权重：Task0 public-test index18、Task40 index0，各1回合，分别独立服务/产物；共2回合 |
| 仿真 | 已有 post2 / IsaacSim5.1 环境，WebSocket、temporal ensemble、RTC off；官方原生超时与成功/Q-score判据 |
| 终止 | 4步完成且两回合有有效原生产物；NaN、无参数更新、状态缺失或 OOM 时记录失败并停止该工程作业，不改冻结范围/Batch冒充通过 |
| 简短备注 | 训练「V0_0 / N1.7：2026原生4步工程验收」；评估「V0_0 / N1.7：2026自训练工程权重，RTC off」 |

## 现有工作台接口

所有请求均发往8765。数据目录先经统一 resource API 分配，再调用 `POST /api/training/dataset`，显式传 `device=local`、source、destination、split（实际 episode ID 列表）、language_mode、`preview=true`；核对后相同配置改为false。返回真实CPU worker的 status / log 路径，使用现有资源浏览接口读取。

新训练使用 `POST /api/training/preview` → `/api/training/start`，显式带 `device=local`、已确认version/run_id、`initiator=agent`、notes、`engineering_only=true`、base_checkpoint/base_sha256、data_manifest、全部 finetune 字段、gpu_index、seed、gradient_checkpointing、minimum_free_mib、pause_after_steps，以及该实验真实最新版 `agent_plan:{version,node}`。现有公共接口先校验 Agent 计划绑定；本文件的 v029 不能代替尚未创建的实验目录计划。

状态与证据沿用 `GET /api/training/jobs|detail|evidence?device=local`。暂停经既有 `POST /api/training/pause`；恢复经 `/api/training/resume`，先 `preview=true`，再false，保留原run、配方、代码摘要、完整训练状态与累计step。

自训练评估仍走 `POST /api/launch/preview` → `/start`，使用 `benchmark=b1k2026`、`version=V_0_0`、`protocol=b1k2026_websocket`、`model_role=self_trained`、`engineering_only=true`、checkpoint、checkpoint_sha256、task、instance、rollouts1、gpu_index、policy_host/port、record_video和notes。必须是本工作台发布的完整2026 checkpoint，任务属于训练清单，语言与训练一致。官方默认仍固定radio权重和Task0。

新训练位于 `training/b1k2026/v0_0__testNN__expNN__runNNN/`。新评估产物位于 `evaluation/b1k2026/V_0_0/self_trained/<run>__stepNNNNNN__taskN__instanceNN__attemptNN/`，沿用原生 `json/`、`videos/`、job/runner与日志。jobs提供 benchmark、版本/系列、policy identity、native_episode_keys / native_metrics及原产物URL；JSON、MP4与标注仍走公共已有接口。历史 `evaluation/V_1_2` 只按原身份读取，不移动或重写。

## 已验证与未验证

已在固定N1.7 Python运行8项CPU测试：原生配置校验、真实官方 shard schedule 中间/边界/跨epoch游标恢复、全局 RNG 隔离、官方绝对/相对统计、聚合文件物理拆分、任务语言、未登记文件泄漏拦截、进程状态与快速退出竞态。使用格式真实的微型数据夹具，不将夹具称为真实模型训练。

另有24项后端测试通过（系统环境跳过5项原生环境测试），27项身份与资源保护测试通过；自训练 Task40 preview、任务/语言/哈希拒绝、原生产物HTTP读取与服务重载恢复均在夹具中覆盖。旧前端测试 `test_benchmark_switch_shows_native_video_and_preserves_playback` 的 DOM stub 缺少 `getAttribute`，且仍断言已被公共前端维护替换的旧元素；本轮未修改前端，也未把该历史测试计入通过数。

尚未执行：真实数据物化、N1.7 GPU前后向/更新、真实保存重载、GPU暂停恢复、自训练仿真。所有相关可用性仍false。完成这些实跑后才更新验收状态；任务成功率与运行完成分开报告。

当前必须由用户确认的是上述工程配方及训练身份的指定/分配授权。依据 [实验 Skill](../../.agents/skills/robot-vla-experiment-ops/robot-vla-experiment-ops/SKILL.md) 的「创建或实质改变实验任务书前……等用户确认」「不自行改变……数据划分或有效 Batch」以及「身份缺失时询问，不用默认版本/编号」。已有“继续完成链路”授权足以继续实现与检查，未提供这些具体研究选择和编号。
