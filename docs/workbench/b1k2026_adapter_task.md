# 工作台统一接口与 B1K-2026 本地适配任务

## 分工
公共工作台维护者负责前端、数据契约、通用作业/结果管理、持久化标注、模拟数据和回归测试。当前本地 Agent **仅负责 B1K-2026 / GR00T N1.7 的实际运行链路**。不开展 LIBERO、MIKASA-Robo、SimplerEnv 的本地新接入，不另建 HTML/CSS 或启动独立工作台。B1K-2025 历史数据和现有运行作业保持不变。

## 公共结果契约
- 每个作业记录 benchmark、唯一 job id、模型/权重身份、任务、协议、运行状态、真实回合结果与原始产物索引。状态 complete 与任务 success 分别记录。
- B1K-2026 当前沿用 GET /api/launch/jobs?benchmark=b1k2026；native_metrics 与 native_episode_keys 一一对应。episode key 为原生 JSON 相对路径，不用列表序号。Q-score 为原生终态目标条件评分；不将步数当任务进度。
- LIBERO 沿用 /api/benchmarks/jobs、detail、video、image；回合键 task:任务:trial:回合:initial:初态；debug 不计正式成功率。数据准备、训练、评估分别记录。
- 标注 GET /api/workbench/annotations?benchmark=...&run=...；POST {benchmark,run,episode,value:{notes,stage,cause,reason}}。B1K-2026的run为工作台job id（不是产物目录名），episode为原生JSON相对路径。服务器核对真实回合键。标注独立存储 outputs/workbench_annotations.json，不重写原生回合。
- 暂未接入能力不得显示可启动；无指标用空值，不伪造。原有 2025 接口保留。

## B1K-2026 本地 Agent 任务书（唯一当前本地任务）
1. 保持当前官方 radio baseline、RTC 配置、权重哈希、原生协议及归档可复现；先核对现有两条回合。不得删除既有结果。
2. 通用任务目录：读取实际官方任务清单、训练/验证数据、实例集合，确认每个任务与权重的匹配关系。不得直接解除 radio 专用权重限制。提供明确的已验证任务/权重白名单及未支持原因。
3. 训练链路：保留Task0 / Task40下载及只读校验证据；按用户最新授权，本次仅Task40训练，V0系列/V0_0，test04/exp01/run001，180/20划分、冻结LLM/visual、有效Batch1、四步工程验收，暂停2→同run恢复4→最终权重原生闭环一回合。核实真实batch前后向、动作维度、完整保存重载和optimizer/scheduler/RNG/游标恢复。具体配方见[任务书](../../Output/test_04/experiment_01/task.md)。正式训练仍须另行确认预算与方案，本次不启动。
4. 推理与评估：核实 action/state schema、归一化、chunk/horizon、语言指令、实例与种子；验证训练权重闭环运行。扩展多任务、并发、资源预约、暂停/恢复仅在实际协议支持且通过验收后开放。
5. 原生回合：每回合独立可识别的 JSON、视频、日志、配置、权重/源码哈希；保持原生成功判据及 Q-score。记录真实异常原因；实时相机需要采集端实现，不能把录制视频伪装成实时。
6. 验收提交：提交可复现命令、实际路径、接口样例、日志、最少一条可播放回合、GPU/环境版本和测试结果；注明通过/未通过的能力。仅修改运行链路及对应适配器，不并行修改公共工作台前端。未经确认不启动大规模训练、不干扰当前其他作业。

## 当前边界
官方radio限定权重与现有两条原生回合保留。本次Task40四步训练、step2→4完整恢复及最终step4自训练闭环已按[计划v031](../v1_2/workflow/plan-v031.json)实跑验收：job076b8962918c正常退出，4857步，success0/1、Q-score0，JSON与可播放视频已核对。能力读取器匹配[真实证据](../v1_2/evidence/training-acceptance-v031.json)后仅开放Task40工程能力；文件或身份不匹配即关闭。正式训练、原生评估暂停恢复、多环境和实时相机保持关闭。其他Benchmark的现有入口和运行链路保留。工程链路通过不等于任务性能通过，也不意味着其他任务已验收。
