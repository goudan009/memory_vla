# V2.1 运行进度

> 当前动作：正式Stage1持续训练，工作台与systemd持久巡检已部署。
> 完成量：真实预检通过；完整暂停/续跑通过，A100 01:00:21已达86/6000步；PR #2已合并main；三次预检大产物已清理。
> Agent Session：Codex /root，2026-11-27
> 机器：4090工作台发起，A100 172.28.62.17执行
> 代码版本：main合并9e750f1；正式作业固定独立源码快照，模型摘要c32ae17197824c733288afbc67bea42cfe0de931440a3633f09cb2e06b9868d8
> 配置：src/v2_1_groot_memoryvla/configs/task40.json

| 阶段 | 状态 | 记录 |
| --- | --- | --- |
| 锁定V2.1执行约定 | 已完成 | 通用GR00T，180/20，原研究参数不变，预检后自动正式启动 |
| 四卡与工作台适配 | 已完成 | cbe3c7a；全局2窗口、梯度SUM、各rank运行态/RNG、原子checkpoint及工作台生命周期 |
| 分布式工程回归 | 已完成 | CPU四进程更新/idle tail对照通过；完整依赖环境40项通过；Agent新旧工作台113项通过 |
| A100真实运行预检 | 已通过 | 003开启CUDA确定性算法后，完整HF重载及下一更新模型/optimizer/runtime逐位一致，四rank权重一致、冻结backbone未变；evidence/preflight-passed-003.json |
| 正式训练与持续巡检 | 进行中 | v2.1__test01__exp14__run001；初始20条验证完成；第1步暂停释放GPU后同作业恢复，01:00:21已达86步；memory-vla-v21-patrol.timer每120秒查询同一工作台 |
| 合并与交付 | 已完成工程交付 | PR #2合并9e750f1；ecd7a5a GitHub CI通过；8765已加载main，HTTP详情实际显示四卡step86，其他本机计算进程PID保持；A100自动源码同步恢复且无错误 |

预检001失败日志已归档于evidence/preflight-failed-001.json，并复制远端独立evidence；工作台已永久删除其专属输出与源码快照（evidence/cleanup-preflight-001.json）。002详细分层证据为evidence/resume-diagnostics-002.json。未通过真实恢复门禁前，不启动正式训练。最新CPU V2.1回归31通过，新旧工作台113通过。

根因与修复：输入、视觉特征、损失、完整重载权重均一致，而CUDA默认非确定性反向归约导致grad_norm约1e-6差异；启用torch.use_deterministic_algorithms及CUBLAS_WORKSPACE_CONFIG后003逐位恢复通过。架构、LR、预算、标签与损失不变。最新V2.1+工作台40通过，确定性开关回归1通过。

003成功证据已独立复制到A100 `/nvme3/data_wyh/outputs/memory_vla/v2_1/experiment14/evidence/`，正式作业只引用该小证据文件。三次预检的专属输出与源码快照均经工作台purge删除，删除审计见evidence/cleanup-preflight-00{1,2,3}.json；不删除通用GR00T、数据、标签或正式作业。

暂停续跑证据见 `pause_resume_proof.md` 和 `evidence/resume-{preview,start}.json`。续训使用原独立源码快照，四rank、LR1e-5和累计6000步预算保持；最近更新3.66秒/步、峰值分配30.06GiB/卡。全局固定2窗口，未通过增大batch或空占显存追求占用率。Stage1没有记忆/状态分支，当前state loss为0是预期；状态监督在后续记忆阶段启用，未删除监督配置。

持久监管：远端既有controller每秒记录真实子进程与累计步数，本机systemd用户timer每120秒执行 `scripts/v21_training_patrol.py`，追加 `evidence/patrol/watchdog_log.jsonl`、更新 `watchdog_state.json`，检查完整断点、GPU、磁盘与非有限数值，并尊重后续用户暂停。该巡检不自动创建其他研究分支；Stage1完成后的记忆阶段真实预检和3000步正式训练尚未执行，后续须继续按协议通过门禁。当前工程验收不代替最终闭环效果验收。
