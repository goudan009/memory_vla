# V1.2 N1.7 / B1K-2026 实机验收

当前状态：`PARTIAL`。环境、官方权重加载、原生 WebSocket 回合、任务语言接入和 8765 工作台验收完成；两种语言配置的 radio 任务均未成功。旧 V2.1/Exp13 自动派发与看护流程已停用，后续开发和评估使用 2026 主线。不能宣布所有任务、模型效果和训练均已验收。正式训练提交为 **0**。

## 固定资源与身份

| 项目 | 实际配置 |
| --- | --- |
| 方法 / 协议 | `V_1_2` / `b1k2026_websocket` |
| GR00T | `ace36d935b376fbf25cd56371e23877b95407c40`；独立 Python 3.10 |
| BEHAVIOR-1K | `a8247a8cc1633fe1ca0cc66aa07243d46c64f155` / `v3.9.3-post2`；Python 3.11 |
| Isaac Sim / 资产 | `/opt/isaac-sim-5.1` / `/nvme3/datasets/b1k2026-assets/data` |
| 官方权重 | `/nvme3/checkpoints/memory_vla/official-n17-radio/turning_on_radio_GR00T-checkpoint-150000` |
| 权重与 processor 摘要 | `cd97a22ff74bd91838350e18fe94b48e1902d407f1256f5d48eb20befe277dc3` |
| Cosmos | `/nvme3/models/Cosmos-Reason2-2B`，revision `9ce19a195e423419c349abfc86fd07178b230561`，实际离线加载成功 |
| Cosmos safetensors | 4,877,470,304 字节；SHA256 `fa5a6e6ef4fce40216b185cc48a3b24d31637ac3e2ba69c107ed1f389c1e6ede` |
| 闭环 | `turning_on_radio`，public index 18 / native instance 319，一环境一回合，temporal ensemble，RTC off，原生超时 3224 |
| 设备 | 4090 GPU0，UUID `GPU-49fbbb98-a702-2adb-f148-a55b95f77252`，实测 49140 MiB |
| 主工作台 | `http://127.0.0.1:8765`；8767 临时测试入口已收尾；8766 属于其他项目 |

没有新建训练 `test/experiment/run` 身份。两个上游 checkout 均保留固定版本；语言修正在本仓库的启动 adapter 中完成。

## 真实回合

| 配置 | Job | 状态 | Steps | Success | Q-score |
| --- | --- | --- | --- | --- | --- |
| 原样官方服务，默认语言 | `97f45ef78924` | COMPLETE，returncode 0 | 3225 | false | 0.0 |
| Task0 任务原文，尝试 2 | `3fc189e832d8` | COMPLETE，returncode 0 | 3225 | false | 0.0 |

原样服务结果：[原生 JSON](../../evaluation/V_1_2/official/public_test_instance_18/json/turning_on_radio_319_0.json)，[完整录像](../../evaluation/V_1_2/official/public_test_instance_18/videos/turning_on_radio_319_0.mp4)。录像经 ffprobe 解码计数：672×448 / 30 fps / 3225 帧 / 107.5 秒，5,710,785 字节。JSON SHA256 为 `980f8baf685b206a4a748a93d7e281d691bea29506b709b342eee26c5d806d80`；录像 SHA256 为 `dbeb8cadfb5b61279087bc0fd2cd9c609ab1f41820d7b05fbb487aa30c10c424`。原生 JSON 未改写。

这是一个有效完成的**官方默认输入配置**回合，成功数 0/1；Wilson 95% 区间约 0%–79.3%，不用于估计整套 benchmark 性能。此前没有完成有效回合的启动失败不计入分母。复测改变语言输入，不能将两个配置混为同一成功率；没有固定双方推理 RNG，也不作因果配对结论。

任务原文复测：[原生 JSON](../../evaluation/V_1_2/official/public_test_instance_18_attempt_02/json/turning_on_radio_319_0.json)，[完整录像](../../evaluation/V_1_2/official/public_test_instance_18_attempt_02/videos/turning_on_radio_319_0.mp4)。视频同为 672×448 / 30 fps / 3225 帧 / 107.5 秒，5,838,507 字节，已完成解码计数。JSON SHA256 `e2a9498583a375c35ce84271c3cdca72049f7821fcc4f15be1c8e6317a68e3aa`；MP4 SHA256 `df9335fd392f214c13c92680738ca0efcb4a47f971b1401f2171e50745be796a`。该输入配置同样是 0/1（Wilson 95% 约 0%–79.3%）。抽帧显示视角和机械臂运动，但不足以确定完整失败原因；原生任务判据未通过。提示词修正没有在该实例上取得任务成功。

两条原生 JSON 的指标与来源摘要发布在[小型证据快照](evidence/native-acceptance-2026-11-28.json)；大录像保留在实验存储，不进入 Git。

## 语言核查及修正

固定源码 `serve_b1k.py` 没有给 `B1KPolicyWrapper` 传 `text_prompt`，因此实际语言为 `pick up the object and place it on the table`。checkpoint 的训练配置指向 `turning_on_radio_v3.0`；R1Pro 将 `annotation.human.task_description` 映射到 `task_index`，原生 loader 再由 task metadata 取语言。下载的 Task0 原文是 `Turn on the radio receiver that's on the table in the living room.`。默认服务的语言与该任务原文不一致；这不证明它是唯一失败原因。

该 checkpoint 未保存当时的 tasks 元数据。本次修正依据已下载的官方 Task0 原文，不宣称逐字重建了历史训练提示。

`9b3527b` 添加显式 `language_mode=task_description`：检查 Task0 元数据、记录文本和 SHA256，并通过启动 adapter 将原文交给官方 wrapper。`official_default` 模式仍可复现原样服务。Task metadata SHA256 为 `8d8222310c3611316c6e5fe822a8dd3c6e552b32bcfddd10870c0672bfb8427c`。真实 N1.7 Python 中调用上游 `process_input`，已确认构造出的模型语言输入为该 radio 原文，batch=1，三相机和六组状态均存在。

复测从 8765 经 preview/start 提交于 `2026-11-28 14:27:32`，GPU0 / port 8033；输出 [尝试 2](../../evaluation/V_1_2/official/public_test_instance_18_attempt_02/job.json)，备注 `V1.2 / N1.7 / B1K2026: radio task prompt, RTC off`。同一实例采用独立尝试目录，原结果保留。

## 工作台与数据

8765 通过既有 `memory-vla-web-8765.service` 重载后，`options.model_versions.V_1_2.launch_ready=true`。已实际核对两条完成作业的备注、steps/success/Q-score；JSON 与 MP4 的 artifact GET 均返回 HTTP 200，MP4 Content-Length 与实体字节数相同。原生界面支持任务指令、尝试编号、简短备注、预览与原生产物链接。收尾时发现 URL 校验未允许 `_attempt_02`，由真实 HTTP 回归复现 404 后修复；重载后实际 JSON/视频 GET 均通过。相关接口/原生 JS 回归共 75 项通过。真实 Chromium 可执行文件缺失，未声称浏览器实测通过；历史 UI 的既有 10 项失败已单独记录于 progress。

原临时 8767 进程仍加载旧 Python 后端，与已经更新的共享 HTML 不一致。确认其无活动作业、原始完成结果已由 8765 接管后，仅终止该测试进程 PID `2656156` 并核对端口释放；未停止 CloudCLI、8766 或当前复测。此后使用 8765。

2026 Task0 / Task40 元数据各有 200 个不同 episode，各有 200 个技能 JSON；RGB 文件分别 10 / 14，聚合 data parquet 分别 2 / 3。数据为 LeRobot v3.0，state 61 / action 23。合并下载缓存含完整任务元数据，不可直接当成已经隔离好的训练集。旧 2025 Task40 的 schema 检查留在历史报告，不作为 2026 训练兼容性证明，也不再转换适配。

训练仍缺独立任务与划分视图、训练集统计、实际 loader/batch 前后向、保存重载与严格恢复验收；`V_1_2` 的训练 `launch_ready=false` 保持不变。技能边界端点口径尚未全部对齐，不能宣称标签全量通过。没有正式训练，也没有 V2.2 开发。

## 迁移门槛与资源

本轮运行兼容性门槛通过：正确任务语言的真实回合正常退出，生成完整原生 JSON/视频、可读指标，且完成服务释放和主工作台访问。任务效果仍失败；运行兼容性通过不等于所有 2026 任务、模型表现或训练就绪。

实时核对无活动本地评估、无 A100 活动推理或训练后，于 `2026-11-28 14:51:14` 停用 `memory-vla-v2-1-training.timer`、`memory-vla-exp13-watchdog.timer`，再停止对应 supervisor service。两 timer 为 inactive/disabled，service MainPID=0；systemd 保留本次主动 SIGTERM 15 的 `Result=signal` / failed 状态，该状态指看护脚本的停止，不是新增训练失败。没有取消或恢复那条旧 paused 训练。原 unit 文件保留，必要时可重新启用，不删除历史作业、数据、权重、源码或结果。

`memory-vla-training-archive.timer` 与 8765 保持 active；无关的 8766 仍监听。Ada 上归属或实时状态不能确认的服务未停用，不声称所有旧机器上的进程均已下线。

2026 evaluator 已实测通过 Warp 1.12、WebSocket 16.1、CFFI 1.17.1、Kit 后 Numba 0.66 / llvmlite 0.48 的 JIT 导入边界。它仍复用 `/nvme3/envs/b1k2025-policy/lib/python3.11/site-packages` 和 SERF 环境中的已安装通用依赖，目录名不代表使用 2025 仿真；不能随旧服务一起删除这些目录。两次测试的 policy port 8032/8033 均已释放，复测 policy/evaluator PID 2809041/2809619 已退出。

复测 14:33 样本：模型 6686 MiB、仿真 13095 MiB，共约 19.3 GiB；不是峰值统计。未达到本机单卡 70% 的运营显存目标，因为本轮限定一环境正确性验收，不用额外回合填充显存。native evaluator 总耗时约 20 分 49 秒（包含场景加载），3225 个控制步约 2.58 步/秒，不代表纯推理 QPS。结束后 GPU0 总占用回落至 1809 MiB，其他项目 GPU 进程保留。

执行依据：[task](task.md)、[plan-v027](workflow/plan-v027.json)、[progress](progress.md)、[官方基线文档](https://behavior.stanford.edu/challenge/baselines)。
