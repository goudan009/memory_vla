# robot-vla-experiment-ops · 安装说明

一个可供 Codex CLI 和 Claude Code 使用的通用科研实验 Agent Skill。文件夹中 `SKILL.md` 为入口，`references/` 为需要时读取的执行细则，`templates/` 为以后写实验任务书的最短模板。

## 安装到当前项目（推荐）

把整个 `robot-vla-experiment-ops/` 文件夹复制到项目仓库：

- Codex：`<repo>/.agents/skills/robot-vla-experiment-ops/SKILL.md`
- Claude Code：`<repo>/.claude/skills/robot-vla-experiment-ops/SKILL.md`

也可以安装到个人用户目录，供多个仓库调用：

- Codex：`~/.codex/skills/robot-vla-experiment-ops/SKILL.md`
- Claude Code：`~/.claude/skills/robot-vla-experiment-ops/SKILL.md`

两边需要使用时可复制同一份文件夹（或在支持符号链接的系统建立共享目录链接）。Skill安装后重启/刷新相关Agent会话，确保被识别。

**建议再做一步**：把 `AGENT_RULE_SNIPPET.md` 的内容追加到仓库根目录的 `AGENTS.md`（Codex）或 `CLAUDE.md`（Claude Code），这样后续Agent接到训练评估任务就会先读Skill，不必每次手动重复提醒。

## 已固定的训练身份与资源规则

- Checkpoint采用`v1__test01__exp12__run001__step002000`，`V1`只代表模型版本，Experiment 12/13不能自动生成`V12/V13`。同一`run001`安全暂停恢复后仍沿用run编号，另记resume事件。
- A100单卡显存有真实工作负载时目标≥80%（80GB卡≥64GB），其他GPU默认≥70%；显存充足时可共存任务并继续训练/恢复，不必等全部机器或GPU完全空闲。
- 查看[命名细则](references/checkpoint-naming.md)和[资源并行细则](references/hardware-workbench.md)。

## 以后任务书怎么写

只需要一行：“本实验训练/推理/评估和资源调度遵循 `robot-vla-experiment-ops` Skill”，加上实验独有的数据、模型、对比、训练超参、目标和验收指标。可直接复制 `templates/experiment-task-template.md`。

## 检查Agent有没有真的应用

让它在启动前回答：

> 请先读取 `robot-vla-experiment-ops` Skill，核对本次任务书与Skill配置的优先级，输出实验配置锁定摘要、GPU资源盘点、工作台preview job ID、预期checkpoint与评估矩阵。确认无冲突后再执行。

**注意：**Skill文件只能规范行为，安装本身不会自动连接用户远端服务器，也不会主动运行训练。Agent必须真的拥有仓库和工作台权限，才能执行。
