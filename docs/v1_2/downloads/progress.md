# V1.2 资源下载进度

计划：[plan-v001.json](workflow/plan-v001.json)。本阶段为基础设施准备，无训练/评估实验编号，无 GPU 作业。源码、权重和数据独立存储；不修改 V2.1 环境及仿真软链接。

## inventory

2026-11-27：main 同步至 f86a76d。已按最新实验 Skill 查询工作台资源、训练和推理作业；V2.1 正在运行，全部保留。/nvme3 初始可用约575GiB。官方 Hugging Face 经已有代理可访问。

已锁定 GR00T 模型 373d34e2723d928870157d07dc216a19a624b5bb（6,927,377,348 bytes），示教数据 4f50b44796641a4d526a19d9aeadc8aa51e2f2c2，资产仓库 9f0d57d465726976ed98138d3f8b8ca3e2186775。数据清单仅包含 Task0/40、RGB、技能标注和所需元数据，不下载深度或全100任务。

## pinned_downloads

进行中（2026-11-27 21:35）。断点进程 PID 2199716，状态文件 `/nvme3/outputs/memory_vla/v12-downloads/progress.json`，radio PID 2199717，状态文件 `radio-progress.json`。代理可访问 Hugging Face 和 Google Drive，下载会保留 `.part` 并在进程重启后继续。已完成的小文件逐项 SHA256/blob 校验；大文件下载完成后再校验。

Cosmos-Reason2-2B 固定 9ce19a195e423419c349abfc86fd07178b230561；实际配置文件请求返回 GatedRepoError / HTTP401。本机未发现已登录 token，需要具有该模型访问权的 Hugging Face 账号；其他下载继续。

官方源码已 checkout：Isaac-GR00T `ace36d935b376fbf25cd56371e23877b95407c40`，BEHAVIOR-1K `a8247a8cc1633fe1ca0cc66aa07243d46c64f155`（tag `v3.9.3-post2`）。BEHAVIOR 资产归档选择 `behavior-1k-assets-3.9.0.zip`、`omnigibson-robot-assets-3.8.2.zip`、`2026-challenge-task-instances.zip`，均不覆盖旧目录。

## integrity_and_data_contract

数据选择已核对：Task0 episode_index 0--199、Task40 8000--8199，各 200 条；任务映射来自 pinned `meta/tasks.parquet`。抽查 `annotation_path` 指向对应任务，技能边界保留原始 JSON。`meta/info.json` 确认为 LeRobot v3.0，三路 RGB 选择正确；深度列仍存在于官方 episode 元数据但未下载深度视频。注意示例标注 frame duration 与 episode length 可能有不同端点口径，须在 loader 验收时进一步对齐。

## github_handoff

待提交本次清单、下载记录和检查结果。不包含并行进行的 V2.1 工作台修改。
