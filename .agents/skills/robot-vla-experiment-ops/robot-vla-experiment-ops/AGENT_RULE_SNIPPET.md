# 放入仓库根目录 AGENTS.md（Codex）或 CLAUDE.md（Claude Code）的最小引用规则

当任务涉及本仓库的机器人训练、推理、评估、GPU资源、工作台、checkpoint暂停续训、实验巡检或实验报告时，**必须先读取并执行** `robot-vla-experiment-ops` Skill（项目内 `.agents/skills/robot-vla-experiment-ops/SKILL.md` 或 `.claude/skills/robot-vla-experiment-ops/SKILL.md`）。

实验任务书只定义该实验特有的模型方法、数据、训练配置、比较方案和验收指标；通用执行与资源规范由 Skill 负责。用户当前明确指令和该实验专有配置优先于 Skill 中的默认值。启动任何正式实验前，先展示**配置锁定摘要及现有工作台作业预览**；必须通过现有工作台启动、监控、暂停与恢复，不得产生工作台不可见的长期作业。

若缺少必要权限、工作台能力或原始数据，不得假称已执行；按 `BLOCKED` 记录并说明可恢复步骤。

**强制Checkpoint身份：**采用`v1__test01__exp12__run001__step002000`式五段ID；实验编号不等于版本号，禁止自动生成`V12/V13`等伪版本。允许安全资源共存下的持续训练/严格续训，不以GPU全部空闲为条件；A100有有效负载时显存目标≥80%，不可为凑数虚占显存。
