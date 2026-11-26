# 工作台完整暂停与续训验收

运行：`v2.1__test01__exp14__run001`；A100 172.28.62.17；2026-11-27。

1. 工作台于00:48:27启动正式Stage1，初始20条验证后执行第1次optimizer更新。
2. 经同一TrainingManager.stop提交stop_after_step=1，00:54:26终态stopped、累计step1。四个GPU计算进程均退出，nvidia-smi计算进程列表为空。
3. 断点 `run/checkpoints/v2.1__test01__exp14__run001__step000001/` 包含COMPLETE.json、完整HF权重/processor及training_state.pt；完成标志world_size=4，模型源码摘要 `c32ae17197824c733288afbc67bea42cfe0de931440a3633f09cb2e06b9868d8`。
4. 同工作台resume preview核对step1、原源码、GPU0–3和6000步预算；00:54:50 resume启动，同一run，不新开训练谱系。证据 `evidence/resume-preview.json`、`evidence/resume-start.json`。
5. 00:55:48真实累计step8，窗口[2138,767]，world_size=4，lr=1e-5，fm=0.1808018207，grad_norm=1.484509230，更新耗时3.657秒。step2–8均实际完成，不是仅加载断点。

训练入口在resume路径校验identity/source/world_size，恢复optimizer、各rank RNG及memory runtime，推进原采样流至累计step；正式作业保持不可变源码快照。模型/optimizer/runtime下一次更新逐位一致另由预检003实际验证（`evidence/preflight-passed-003.json`）；本文件记录工作台生命周期验收，不声称正式step8与另一个独立作业做过逐位比较。
