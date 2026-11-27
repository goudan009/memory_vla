# Agent 路由规则

涉及本项目训练、推理、仿真评估、工作台、checkpoint 或实验验收时，先读取并遵循仓库根 `AGENTS.md` 指定的 `robot-vla-experiment-ops` Skill；新建或接续实验同时遵守 `docs/workbench/agent.md`。

启动前必须对齐已登记方法版本、`test_id`、`experiment_id`、`run_id`、预算和工作台 preview。实验编号不能充当模型版本。未确认/未登记的小版本、缺失或冲突身份、重复 run ID 都必须停止修正，不能猜测或绕过工作台校验。

Skill 包的通用安装方式见 [README_INSTALL.md](README_INSTALL.md)。
