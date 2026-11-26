# V2.1 运行进度

> 当前动作：Stage1持续训练；按用户睡前授权部署后续训练和官方闭环的持久定时编排。
> 完成量：2026-11-27 01:24巡检517/6000步，最近100步3.509秒/步；自动编排和服务适配163项回归通过；定时器已实际运行成功，退出登录后继续运行。
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

## plan-v002：睡前授权后的自动衔接

上段是早先只观察阶段的历史记录。用户明确要求无人值守后续训练与评估后，已提交plan-v002，将同名timer切换为`scripts/v21_overnight.py`。systemd用户服务Result=success/ExecMainStatus=0，timer enabled+active，Linger=yes；不依赖本次对话存活。持久状态/日志改为`evidence/automation/watchdog_{state.json,log.jsonl}`。可通过工作台暂停实际GPU作业，或`systemctl --user stop memory-vla-v21-patrol.timer`停止后续编排。

| 阶段 | 状态 | 记录 |
| --- | --- | --- |
| 持久工作台调度部署 | 已完成 | 每120秒；依赖、预览、提交前意图、重启去重、用户暂停、非有限值停止、证据先归档再purge均落盘；163项回归通过 |
| Stage1完整权重就绪 | 进行中 | run001，等待累计6000且COMPLETE/training_state；当前剩余更新约5.34小时，保存验证额外计时，估计早7–8点 |
| 记忆阶段真实预检 | 待开始 | run002__preflight001；从Stage1step6000，真实校准/梯度/冻结/完整重载/下一更新一致性，通过后归档小证据并工作台永久清理专属大权重与源码 |
| 拓扑记忆正式训练 | 待开始 | run002固定3000；同工作台暂停、释放进程、原checkpoint恢复并确认下一更新 |
| 同预算无记忆继续训练 | 待开始 | run003从相同Stage1独立预检后3000步，保留同参数及180/20；不继承memory权重 |
| 三模型官方闭环评估 | 待开始 | Stage1/memory/continued末尾权重各10回合，实例0–9、seed0 sequence、single1 GPU1、action chunk8、原生结束、24step/20fps视频；实际有效回合与Q/progress/native success完整才验收 |
| 结果与资源归档 | 待开始 | 独立保存指标、视频路径、原始失败；评估结束仅工作台停止本任务创建且无依赖的推理服务 |

已修复真实HTTP权重列表的v2.1/V_2_1身份解析，40份权重目录可正常查询，其中当前V2.1完整step0/1身份正确。无记忆服务保留FP32动作专家、BF16视觉主干，memory使用原V2.1服务。评估使用单独冻结且校验2445文件hash的serving快照（`evidence/automation/serving-source.json`），避免Stage1旧训练快照缺少新baseline服务入口。预检/训练继续使用其各自不可变快照，研究架构、LR、预算、状态监督不变。

验证命令：`PYTHONPATH=src src/v1_groot_memoryvla/.venv/bin/python -m pytest -q tests/test_groot_training_dashboard.py tests/test_v21_overnight.py tests/test_v21_eval_chain.py tests/test_v21_inference.py tests/test_groot_inference_launch.py src/v2_1_groot_memoryvla/tests/test_loading.py` →163 passed。最初用另一个Python跑旧队列取消回归缺pidfd，换回工作台所用完整环境并补PYTHONPATH后全过，未为测试修改生产取消语义。后续GPU预检/推理/30回合尚未发生，不能把入口和定时部署视为已完成研究结果。
