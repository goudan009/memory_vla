# Exp13工作台正式训练启动

用户2026-11-25确认既有配方后，通过4090工作台8765的`preview → 资源核对 → start`启动，无排队、无其他作业抢占。正式作业`v1__test01__exp13__run001`，PID3482804。模型V1，实验test01/exp13；不是旧Residual权重或预检权重继续训练。

## 已实测验收

临时run999完成同配方四卡2步：34.004亿可训参数、全局Batch256、每rank microbatch4、累积16；骨干、动作头及记忆新增模块均有梯度和抽查参数更新。4rank优化器/RNG/采样位置恢复成功，重载动作逐位一致。完整证据在[evidence](evidence/launch_20261125/preflight_acceptance.json)及`preflight/`，随后通过工作台delete移除临时作业，复核列表为空。临时数据保留于远端训练根`.trash/v1__test01__exp13__run999`，可恢复；未作为正式初始化。

正式作业已核对4rank均完成第3步，每rank64个决策、全局256，Loss 0.03240843，未见OOM/NaN。模型初始化、10000条训练示教、0验证、Full FT、LR2e-5、原始特征写入和随机有序group16均与确认配置一致。首两步69.22/41.78秒仅表示初始吞吐，不据此承诺全程耗时。

## 产物与容量

- 远端源码：`/nvme3/projects/memory_vla/deployments/v1__test01__exp13__run001`。
- 远端输出：`/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v1__test01__exp13__run001`。
- 运行日志：输出根`logs/memory_v1_torchrun.log`；逐步及按部位Loss：`memory_v1/metrics/metrics.jsonl`，每rank记录在`memory_v1/ranks/`。
- 完整存档：`memory_v1/checkpoints/v1__test01__exp13__run001__step002000/`等，含模型、optimizer/scheduler、RNG与采样游标，完成后发布`COMPLETE.json`。
- 实测单份38.00GiB，10份380.05GiB；预检回收数据保留后当前空闲409.54GiB。巡检持续记录磁盘与已完成checkpoint。
- 正式源码digest、基础权重身份、GPU UUID、配置与数据指纹见[MANIFEST](MANIFEST.json)；真实工作台请求/响应与配置快照在`evidence/launch_20261125/`。

## 后台监管和暂停续训

通过用户systemd timer `memory-vla-exp13-watchdog.timer`每120秒检查原工作台作业、真实步数/Loss、GPU、磁盘和完成checkpoint，状态/日志分别为`watchdog_state.json`、`watchdog_log.jsonl`。不依赖当前CLI会话持续存活。控制训练仍只调用原工作台API；未创建训练队列。

已通过工作台为首份2000步正式存档安排暂停验收。因此面板当前阶段目标可能显示2000，实际原始optimizer预算仍是20000；结束写盘并释放进程后，巡检调用同一工作台resume preview，再直接resume原run到20000。它只恢复自己安排且未被用户修改的停止控制；失败作业不自动重启。真实续训后保存`pause_resume_proof.json`。当前该正式暂停恢复尚未发生，不能称已经验收。

`checkpoint_index.csv`由巡检维护；其SHA256明确对应`COMPLETE.json`完成标记，不冒充全量权重逐文件hash。训练完成或失败后归档终态并停止timer。
