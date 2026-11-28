# MemoryVLA 仓库协作规则

本机环境与存放约定见 `~/.agent-rules.md`。

## Git 分支规则

默认直接在 `main` 上修改和提交。未经用户明确要求，不得新建本地或远端分支，也不得通过新建 worktree 自动创建分支。实验计划中的“并行分支”仅指流程依赖，不代表 Git 分支；只有用户明确指定时才使用其他 Git 分支。

## 工作台资源和启动规则

涉及本仓库训练、推理、闭环评估、GPU调度、checkpoint、实验巡检或验收时，**必须先读取并执行** [.agents/skills/robot-vla-experiment-ops/robot-vla-experiment-ops/SKILL.md](.agents/skills/robot-vla-experiment-ops/robot-vla-experiment-ops/SKILL.md)，再按其中的操作路由读取细则。新实验与接续实验都遵循此要求。

涉及训练、模型推理、仿真评估或相应的排队、续跑与进度查询时，先读 [工作台执行手册](docs/workbench/agent.md)。“按工作台规则”“用工作台资源”“用满算力”都适用这套流程。只需查看状态或做方案时，使用只读查询。

通过 4090 上的 8765 工作台查询实时 GPU、作业和设备能力。优先复用符合配置的作业、权重和推理服务。根据用户已要求的任务量，把相互独立的工作分给有足够容量的设备。优先使用工作台已有接口和队列，提交前调用相同功能的 `preview`，提交后从工作台核对作业 ID、状态和输出位置。遇到入口或资源问题时，定位该问题，保留现有作业和实验配置。

## Agent 实验记录

每次承担新的实验任务时，Agent 先在当前实验的 `workflow/plan-vNNN.json` 保存计划，再执行对应步骤。计划必须有一条明确主线，为每个节点写前置依赖、可并行分支和验收条件。接续任务先读最新版并核对证据。启动前展示锁定的方法版本、`test_id` / `experiment_id` / `run_id`、预算和工作台预览；提交后核对真实job ID与输出。方法小版本须经用户确认并在 `train/run_identity.py` 登记；实验编号绝不作模型版本。身份缺失、冲突或重复时停止修正，不猜编号、不换旧入口绕过校验。调整主线、依赖或验收时，新增版本并说明原因，将计划文件默认提交到 `main`；仅在用户明确指定时提交到其他 Git 分支。阶段进展写入 `progress.md`，沿用计划节点对应的阶段名。格式见[实验记录模板](docs/workbench/agent_experiments.md)。

处理 `Output/test_NN/experiment_NN/` 下的实验时，在该目录维护 `task.md` 和 `progress.md`。接到安排后记录验证目标、对照和验收标准，阶段变化时更新当前动作、状态和实际完成量。使用 [实验记录模板](docs/workbench/agent_experiments.md)，关联 Agent Session、机器、代码版本、配置和已有工作台作业 ID。

## 资源管理（强制）

资源下载、登记、创建目录、回收、恢复前先读 [统一资源管理](docs/workbench/resource_management.md)。优先工作台 GUI/API/CLI 的 resource 入口；先查已有资源和磁盘，再选择登记配方、预检、启动、查询实际状态、校验并登记。缺少配方/环境或权限即报告阻塞，不临时换源、安装环境、curl/wget/git clone 或 rm 绕过工作台。已有下载不得擅自重启；训练 checkpoint 删除沿用受保护接口。

Agent 新训练通过工作台 preview/start 时显式传 `initiator=agent`、`notes`、规范 `run_id` 和 `agent_plan`（`version` 为当前最新真实计划版本，`node` 为实际节点 ID）。工作台验证计划与前置实验；不得通过去掉 initiator、改 run_id 或直接调用训练脚本绕过。提交异常先核对 jobs 和 `outputs/workbench_agent_bindings/` 回执；旧记录和严格恢复保持不变。
