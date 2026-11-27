# V1.2 资源下载进度

计划：[plan-v001.json](workflow/plan-v001.json)。本阶段为基础设施准备，无训练/评估实验编号，无 GPU 作业。源码、权重和数据独立存储；不修改 V2.1 环境及仿真软链接。

## inventory

2026-11-27：main 同步至 f86a76d。已按最新实验 Skill 查询工作台资源、训练和推理作业；V2.1 正在运行，全部保留。/nvme3 初始可用约575GiB。官方 Hugging Face 经已有代理可访问。

已锁定 GR00T 模型 373d34e2723d928870157d07dc216a19a624b5bb（6,927,377,348 bytes），示教数据 4f50b44796641a4d526a19d9aeadc8aa51e2f2c2，资产仓库 9f0d57d465726976ed98138d3f8b8ca3e2186775。数据清单仅包含 Task0/40、RGB、技能标注和所需元数据，不下载深度或全100任务。

## pinned_downloads

准备中。Cosmos-Reason2-2B 固定 9ce19a195e423419c349abfc86fd07178b230561；实际配置文件请求返回 GatedRepoError / HTTP401。本机未发现已登录 token，需要具有该模型访问权的 Hugging Face 账号；其他下载继续。

## integrity_and_data_contract

待实体下载完成。不将资源下载等同于环境或闭环验收。

## github_handoff

待提交本次清单、下载记录和检查结果。不包含并行进行的 V2.1 工作台修改。
