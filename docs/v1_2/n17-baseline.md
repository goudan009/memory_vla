# V1.2：GR00T N1.7 + BEHAVIOR-1K 2026 迁移计划

> 2026-11-27 用户确认：V1.2 和后续 V2.2 全面使用 B1K-2026；取消新底座的 B1K-2025 适配路线。先下载资源、跑通官方无记忆基线，后续再确认正式训练方案。旧运行的保护限制已由下方 2026-11-28 授权更新，历史权重及结果继续保留。

> 2026-11-28 授权更新：先测试 B1K-2026；官方闭环产出有效原生 JSON、视频、success/Q-score 且正常退出后，核对依赖并停用已确认的旧 B1K-2025/V2.1 运行服务。历史数据、权重、源码和结果保留。仍不启动正式训练、不开发 V2.2。最新执行计划见 [plan-v025.json](workflow/plan-v025.json)，实时证据见 [progress.md](progress.md)。

此前已完成的2025 Task40实体检查和96项CPU回归结果保留在[本机预检记录](local-preflight-2026-11-27.md)，仅代表当时的检查结果，不代表2026已通过验收。旧2025候选代码（data_contract.py、training_contract.py中的 native_train_candidate、policy.py中的 b1k2025_policy）仅留存历史检查依据，不是本计划的启动入口。无需继续适配、转换或下载2025数据。尚未创建V1.2正式训练作业。

## 已取消的工作（2026-11-28 再确认）

V1.2 的 B1K-2025 数据转换、模态适配、旧 ZMQ 通信兼容、训练入口、评估入口及相关验收全部取消，不再分配 Agent、算力或开发时间。旧候选实现与预检报告只作历史留档；不得将其作为2026接入的依赖、备用入口或待完成事项。旧 V2.1/2025 运行按最新切换门槛停用，历史资源与2026仍复用的共享库保留。

## 1. 先下载资源（本次授权范围）

先读最新[统一资源管理](../workbench/resource_management.md)，通过现有工作台 resource 接口查询已登记配方、实际资源、磁盘容量和下载状态；仅使用已登记配方的 preview/start/register 流程，缺少配方（包括非HF来源）时报告阻塞，不绕过工作台下载。复用完整的已下载文件，下载使用用户已同意的 /nvme3 下本机规则指定目录，不猜绝对子路径，不重写 N1.6 环境。长下载保存实际进度、来源 revision、文件列表、大小及校验摘要；失败时记录原因，支持断点续传，不重复下载整个仓库。下载阶段无须伪造训练/评估 run ID。

| 优先级 | 资源 | 来源 |
|---|---|---|
| P0 | N1.7 通用基座 | [nvidia/GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B) |
| P0 | Cosmos-Reason2-2B | [nvidia/Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B)；需账号接受 gated 许可 |
| P0 | 官方 radio 微调 checkpoint | [官方基线手册提供的下载](https://drive.google.com/file/d/1OXNm3SPLvWOSJR1e8In6xHMHxOYDp789/view?usp=sharing) |
| P0 | N1.7 challenge fork | [wensi-ai/Isaac-GR00T](https://github.com/wensi-ai/Isaac-GR00T)，固定 ace36d935b376fbf25cd56371e23877b95407c40，独立 Python 3.10 / uv lock |
| P0 | B1K-2026 仿真源码和必要资产 | [v3.9.3-post2](https://github.com/StanfordVL/BEHAVIOR-1K/releases/tag/v3.9.3-post2)，固定 a8247a8cc1633fe1ca0cc66aa07243d46c64f155；独立环境、官方公共测试场景/机器人资产与实例 |
| P1 | 2026 Task0 turning_on_radio | [2026-challenge-demos](https://huggingface.co/datasets/behavior-1k/2026-challenge-demos)，200条示教，后续可用于训练链路验证；官方权重的闭环无需等待该任务示教下载完成 |
| P1 | 2026 Task40 make_microwave_popcorn | 同一数据集，200条示教，后续科研基线候选 |

**仿真版本选择**：官方文档仍写 v3.9.3，但本机预检确认该 tag 不存在。选择可取得的官方 post2 release 而不是滚动 main；包含灯光读取修复和自定义本体起点支持，默认 R1Pro 配置保持不变。此为工程迁移选择，仍须实测与权重及资产匹配，不能写成已验收的原 v3.9.3。

### 数据下载范围

只下载 Task0/40 的 LeRobot v3.0 数据、三路 RGB 视频、对应 episode/task 元数据、语言及技能标注。暂不下载全部100任务、全部深度视频、1.44TB原始HDF5回放库；仿真所需资产和测试实例不能因为仅下载两任务示教而遗漏。官方3.27TB数据总量并非本次下载预算。

- 官方任务元数据确认 task_index 0 为 turning_on_radio，40 为 make_microwave_popcorn。固定 Hugging Face dataset revision 后，以真实任务元数据、文件列表及 episode/video 引用核对映射。
- 当前公开目录使用 data/chunk-000、data/chunk-040 等，annotations/task-0000、annotations/task-0040 等；meta/episodes 与 videos 具体文件范围按实际 pinned revision 的索引选择，保留 info、tasks、stats 及必要元数据。不要只用旧2025路径或机械套用手册中过时的 include pattern；不要把两任务的合并缓存直接当成单任务训练集。
- 先用 HfApi.list_repo_files / repository tree 确认文件清单并估算空间，按 task 选择下载；保留原数据只读，不改原始 metadata 或统计。若真实原生 loader 要求额外文件，按缺失项补齐。
- 检查示教数、episode ID、语言、状态和动作、时间戳、视频索引及技能标注对齐。技能标注仅下载保留，V1.2不加入任务状态监督，不改网络。
- HF许可/网络或官方下载访问失败时标记阻塞，不用第三方权重冒充官方；向用户明确所缺权限。模型和仿真可并行准备，不影响V2.1。

## 2. 官方 Baseline 原生闭环

沿用官方 challenge fork 的 serve_b1k.py 与 BEHAVIOR 原生 evaluator：WebSocket、temporal_ensemble、RTC关闭；原始61维 proprio、选取25维状态、23维物理动作、三相机、16帧有效监督。实际 padded horizon/维度核对 checkpoint 和 processor，不照搬 N1.6。官方 radio checkpoint 只用于 turning_on_radio，不能作为 Task40 的初始化权重。

已有 contracts.py 提供上游 revision、模型类型、processor、动作表示及权重 SHA256 检查和只读 serve/eval spec。verified_radio_sha256 必须来自核实下载的实体摘要而非目录名称。原生 wrapper 有共享动作缓冲区，初始一环境一服务、每回合 reset；核对官方默认 text_prompt 与训练语言，记录真实行为，不悄改协议。

工作台接入仍复用现有 InferenceManager、LaunchManager 和训练控制器，不另设调度器；增加显式 V_1_2 + b1k2026_websocket 路由、/healthz、checkpoint/source身份、原生 JSON/视频、success、Q-score、回合及运行备注。不得将2026评估伪装成2025任务或沿用旧 ZMQ runner。先确认 GPU/资源与真实实验身份并按 Skill 保存 task/progress/plan；preview → start → 查询实际 job。先完成一个官方公共测试实例的短闭环接口验收，记录真实结果，不以单回合推断统计性能。正式回合预算另行确认。

## 3. 2026 原生无记忆训练（后续确认，当前不启动）

V1.2正式训练只从核实的通用 N1.7-3B 开始，使用2026的 LeRobot v3、R1Pro 官方 modality（examples/b1k/r1pro.py/.json）与原生 trainer。先核对真实数据结构，使用 deploy_modality.py 的检查逻辑在独立训练视图配置 modality；不得直接用旧2025训练候选。原生 v3 的聚合 parquet / 视频可能包含多个 episode，必须按真实 episode 行区间及视频索引隔离训练和验证，重新计算训练集统计；不能套用每个 parquet 对应一个 episode 的2025检查。split、horizon、步数、冻结策略、有效 Batch、评估实例及终止条件在用户批准正式任务前锁定，不直接照搬2025或官方八卡预算。原生 global_batch_size 为累积前跨卡 batch；4×2×8 应传8、有效64，仅为算例。

先完成2026 loader、processor、动作编码/解码、真实 batch 前后向、保存重载和工作台接入验收；正式累计超过2000步必须严格同run暂停/恢复，不能依赖上游普通 resume 的 ignore_data_skip=True 冒充完整恢复。通过闭环验证基础动作与任务成功后，才开始讨论 V2.2 的记忆模块。训练/评估备注按最新 Skill 简短填写，例如「V1.2 / N1.7 / B1K2026：原生无记忆基线」。

## 4. 当前执行边界

V_1_2 原生路由已在独立工作台 `8767` 开放 preview/start，`8766` 属于其他项目。源码、官方 radio 权重、Cosmos、本机 Isaac Sim 5.1 和 2026 资产已进入实跑验收；路由就绪不代表闭环已通过。模型版本登记中的正式训练能力仍未开放，当前不创建训练编号或启动正式训练，也不开发 V2.2。旧服务依照上面的最新切换门槛处理。后续正式实验使用已核实的 testNN/expNN/runNNN 与现有规范输出目录，按 AGENTS.md 和 robot-vla-experiment-ops Skill 执行。无需继续完成已取消的2025适配检查。

参考：[官方基线手册](https://behavior.stanford.edu/challenge/baselines)、[2026数据说明](https://behavior.stanford.edu/challenge/dataset)、[2026官方任务描述提交](https://huggingface.co/datasets/behavior-1k/2026-challenge-demos/commit/2add61313bac4f1a42363d00ad03bd45949941a8)。
