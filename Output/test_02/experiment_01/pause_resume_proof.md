# 无记忆对照暂停与严格续训验收

运行身份：`a100:v2_1__test02__exp01__run001`，固定3000 optimizer steps，4×A100，180训练/20验证，seed42、lr1e-5、microbatch2、两条顺序窗口。暂停和恢复未改变运行ID、源码快照或研究配置。

1. 当前不可变源码在独立真实四卡预检中完成模型/优化器/各rank运行态重载，恢复后下一次更新的参数、优化器和运行态均逐值一致。证据为 `evidence/training_watch/preflight-archive.json` 中 `resume_equal=true` 及三项 equality=true。
2. 经工作台stop接口在optimizer第10步保存完整断点 `v2_1__test02__exp01__run001__step000010`，COMPLETE记录step10/world_size4/包源码摘要。停止于A100时间2026-11-27 12:19:58。
3. 停止后实际检查该作业live_pids=[]；四卡显存降至35、32、32、36MiB，确认进程退出并释放资源。
4. 同一工作台resume preview核对resume_step=10，随后提交resume。`resume-event-001.json` 时间为A100 12:20:28，源码仍为 `deployments/v2_1__test02__exp01__run001`，设备仍为GPU0/1/2/3，预算仍3000。
5. A100 12:21:57核对正式进度已到第26步；恢复后的真实第11步及后续更新存在，lr仍1e-5、world_size仍4，loss和grad_norm有限。恢复后最近步耗时中位数约3.67秒。

生产暂停与恢复记录见 `evidence/training_watch/pause-resume-evidence.json`、`resume-preview.json`、`resume-proof-result.json`、`handoff-detail.json`、`state.json`。预检中逐值一致验证与生产工作台暂停/恢复为两项分别完成的验收，不将仅重载前向称为严格续训。

源码身份分开记录：包摘要 `0f06fef7bd50c5d4751c25f8898b0a6b7b92c6aae20a61238325ce0c993b6d10`；工作台整个部署清单摘要 `b5d33c67be1f532d1b4bb93ea71ec1a2901f9fe06675c91f5711c531e45c7655`。4090时钟约慢A100 57秒，以上时间采用A100记录。
