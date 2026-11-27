# 安装

本目录是 Skill 包本身，包含 `SKILL.md`、`references/` 和 `templates/`。复制整个目录（不要再包一层同名目录）到目标 Agent 的 Skills 根目录：

- Codex 项目级：`<repo>/.agents/skills/robot-vla-experiment-ops/`
- Claude Code 项目级：`<repo>/.claude/skills/robot-vla-experiment-ops/`
- Codex 用户级：`~/.codex/skills/robot-vla-experiment-ops/`

本仓库当前因历史布局保存在 `.agents/skills/robot-vla-experiment-ops/robot-vla-experiment-ops/`；仓库根 `AGENTS.md` 指向这一实际入口。新安装使用上面的标准单层路径。

复制后检查目标 `SKILL.md` 和它引用的 `references/`、`templates/` 都在同一目录中。刷新或新开 Agent 会话后，确认能发现 Skill。仓库内的 `AGENTS.md` 负责强制路由；本 Skill 不会自动连接工作台或启动实验。
