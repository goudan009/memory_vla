# V1.2 资源下载进度

计划：[plan-v001.json](workflow/plan-v001.json)。本阶段为基础设施准备，无训练/评估实验编号，无 GPU 作业。源码、权重和数据独立存储；不修改 V2.1 环境及仿真软链接。

## inventory

2026-11-27：main 同步至 `f18607e`（下载续传修复已推送）。已按最新实验 Skill 查询工作台资源、训练和推理作业；V2.1 正在运行，全部保留。/nvme3 初始可用约575GiB。官方 Hugging Face 经已有代理可访问。

已锁定 GR00T 模型 373d34e2723d928870157d07dc216a19a624b5bb（6,927,377,348 bytes），示教数据 4f50b44796641a4d526a19d9aeadc8aa51e2f2c2，资产仓库 9f0d57d465726976ed98138d3f8b8ca3e2186775。数据清单仅包含 Task0/40、RGB、技能标注和所需元数据，不下载深度或全100任务。

## pinned_downloads

部分完成（2026-11-27 22:08）。主资源状态文件 `/nvme3/outputs/memory_vla/v12-downloads/progress.json`：466 项中 461 项已逐项校验，`verified_bytes=5,282,703,865`；5 项因 Hugging Face 连接超时失败，`.part` 保留并可继续断点（N1.7 两个 safetensors、资产 31GB 归档、两段 RGB 视频）。资源单文件重试上限已提高到 10 次，当前轮次已结束；重试前不能把主权重或资产包称为完整。官方 radio 状态 `radio-progress.json` 已为 `verified_archive`：6,917,651,266 bytes，SHA256 `74eab585f4cf0f92326c9d4d74732343f6d789b05e945d3e71afe46bac69cb46`，13 个 ZIP 成员，`zipfile.testzip()` 通过。

Cosmos-Reason2-2B 固定 9ce19a195e423419c349abfc86fd07178b230561；实际配置文件请求返回 GatedRepoError / HTTP401。本机未发现已登录 token，需要具有该模型访问权的 Hugging Face 账号；其他下载继续。

官方源码已 checkout：Isaac-GR00T `ace36d935b376fbf25cd56371e23877b95407c40`，BEHAVIOR-1K `a8247a8cc1633fe1ca0cc66aa07243d46c64f155`（tag `v3.9.3-post2`）。BEHAVIOR 资产归档选择 `behavior-1k-assets-3.9.0.zip`、`omnigibson-robot-assets-3.8.2.zip`、`2026-challenge-task-instances.zip`，均不覆盖旧目录。

源码实际目录：`/nvme3/projects/GR00T/Isaac-GR00T-n1d7-ace36d9`、`/nvme3/projects/GR00T/BEHAVIOR-1K-2026-a8247a8`；两目录均为干净 checkout，revision 与上表一致。精确 `v3.9.3` tag 不存在，post2 的灯光读取修复、可选机器人起点支持和 CI 更新已在 [V1.2 迁移计划](../n17-baseline.md) 记录，不能冒充原 tag。

官方 radio 已解压至 `/nvme3/checkpoints/memory_vla/official-n17-radio/turning_on_radio_GR00T-checkpoint-150000`。`inspect_upstream()`、`inspect_checkpoint(role="official_2026_radio")` 均通过：native `Gr00tN1D7Processor`、40 padded horizon、16 有效动作步、23 物理动作维、61 原始状态维、25 选取状态维、`b1k2026_websocket`；工作台绑定用的解压 checkpoint identity digest 为 `cd97a22ff74bd91838350e18fe94b48e1902d407f1256f5d48eb20befe277dc3`（不同于 ZIP SHA256）。

## integrity_and_data_contract

数据选择已核对：Task0 episode_index 0--199、Task40 8000--8199，各 200 条；任务映射来自 pinned `meta/tasks.parquet`。抽查 `annotation_path` 指向对应任务，技能边界保留原始 JSON。`meta/info.json` 确认为 LeRobot v3.0，三路 RGB 选择正确；深度列仍存在于官方 episode 元数据但未下载深度视频。

技能标注兼容性检查：Task0 200/200、Task40 200/200 的 JSON 已落盘；`meta/episodes` 的 `length` 是 LeRobot 帧数，而 `meta_data.task_duration` / `skill_annotation[*].frame_duration` 使用另一套端点口径，抽查显示 Task0 25/200、Task40 12/200 的最后边界不相等。原始技能标注不改、不删除、不用于 V1.2 任务状态监督；loader 接入前必须按 episode 行区间和视频索引重新核对时间对齐。

## github_handoff

待提交本次清单、下载记录和检查结果。不包含并行进行的 V2.1 工作台修改。当前训练提交 0、评估提交 0、闭环回合 0；无正式训练、无评估成绩。后续新训练备注固定为「V1.2 / N1.7 / B1K2026：原生无记忆基线」，新评估备注须使用真实 2026 字段并写明版本及对照。
