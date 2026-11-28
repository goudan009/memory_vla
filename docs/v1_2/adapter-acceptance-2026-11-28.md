# B1K-2026 本地运行适配验收（v028）

状态：**PARTIAL**。官方 radio 的推理→仿真→原生产物链路已实跑；Task0/40 数据结构和官方 loader 已实测。训练、自训练权重、严格续训及自训练闭环未验收，不能称为全部适配完成。本轮新增训练提交、optimizer 更新和评估回合均为 0。

任务范围来自 [本地适配任务书](../workbench/b1k2026_adapter_task.md)，执行计划为 [plan-v028](workflow/plan-v028.json)。后端代码提交 `e5af129`、`2943448`。没有改公共前端、创建页面或独立服务，也没有改其他 Benchmark 的链路。原有实验、权重、回合及作业全部保留。

## 验收范围

| 能力 | 真实证据与结论 |
| --- | --- |
| Task0 / Task40 数据读取 | 400 episodes，各任务 200；共 1,077,483 rows；5 个 v3 聚合 Parquet、24 个 RGB 文件、1200 个 episode-camera 时间范围、400 份技能标注检查通过。 |
| 原生 loader | 固定 N1.7 `LeRobotEpisodeLoader`，OpenCV 后端；读取 episode 0、199、8000、8199 的首/中/末帧，三相机共解码 36 帧。仅验证读取，不代表 processor、训练 batch 或梯度通过。 |
| 官方 radio 推理及评估 | 复核既有两条真实回合；每条正常结束、3225 步、任务失败、终态 Q-score 0。两种语言配置分别为 0/1，不合并为一个性能样本。 |
| 公共接口与原生产物 | 既有 8765 服务；原生 JSON 相对路径作为回合键，与 native_metrics 一一对应；JSON/MP4 可通过 HTTP 读取，SHA256 与原验收记录相同。标注 GET 可读，不新写/覆盖用户备注。 |
| 任务/权重目录 | 直接读取官方 100 任务目录。白名单仅为已验收 radio checkpoint 与 `turning_on_radio`；隐藏任务缺失的 instruction 保持 null。Task40 等其他任务有明确未支持原因。 |
| 训练及自训练权重 | 不可用。没有真实 B1K-2026 自训练 checkpoint，也未完成模型前后向、参数更新、保存重载和严格恢复。 |
| 评估续跑、多环境、实时相机 | 未实跑验收，明确不可用；回合录像不当作实时相机。 |

## 数据与训练前置缺口

原始下载 `/nvme3/datasets/b1k2026-task0-task40` 保持不变。状态向量 61 维、按官方 R1Pro 选择 25 维，物理动作 23 维。实际下载仅 Task0/40，但 `info.json` / `stats.json` 仍覆盖官方 100 任务、20,000 episodes、210,916,774 rows，不能用作当前训练划分的统计。

原下载缺少 `meta/modality.json`。独立验证视图 `/nvme3/datasets/b1k2026-native-loader-v028` 只增加与固定官方 `examples/b1k/r1pro.json` 相同的 modality，其他文件链接到原始下载。视图已通过资源 API 登记，ID 为 `resource:7c73ce13456f6aeaeeca`，名称明确标注“CPU loader 验证视图（非训练划分）”。该视图没有完成训练/验证隔离，不应直接拿来启动训练。

原生 loader 从 `tasks.parquet` 读到 `turning_on_radio` / `make_microwave_popcorn` 短任务名，而 `tasks.jsonl` 是完整自然语言指令。本次记录这项实际差异，没有擅自改写训练语言。数据中的任务实例取自 1–300，每个任务 200 个实例；它们不等于 public-test 的索引 0–19。固定 evaluator 的 public-test 索引映射到实例 301–320，已跑索引 18 对应实际实例 319。

技能标注以原始帧坐标描述 `valid_duration`，不是从 0 起的轨迹下标。例如 `[576,3105]` 的有效时长是 2529，技能结束帧 3105 本身合法。审计已修正最初误读，并加入非零起点回归。400 份标注内部范围合法，但其中 369 份的时长与发布轨迹长度不同；逐帧对齐未通过，未裁剪、重采样或加入技能监督。

[数据证据](evidence/b1k2026-data-v028.json) 含元数据哈希、真实实例集合、抽帧位置、相机尺寸、各组状态/动作宽度及全部 23 动作维的 min/max/mean/std。这些统计覆盖所有已下载行，仅用于审计，不是训练归一化统计。

后续训练仍按迁移计划从通用 `/nvme3/models/GR00T-N1.7-3B` 开始，不能用 radio 专用权重初始化 Task40。需先锁定规范训练身份、任务范围、split、语言、冻结策略、horizon、有效 Batch 和工程自检预算；实现按 episode 行范围与视频索引隔离的训练/验证集并重算 train-only statistics；再通过工作台 preview/start 验证真实 batch 前后向、参数更新、保存重载、完整 optimizer/scheduler/RNG/采样位置恢复及自训练闭环。上游普通 resume 的 `ignore_data_skip=True` 不是严格恢复验收。

## 固定环境与既有回合

- GR00T：`/nvme3/projects/GR00T/Isaac-GR00T-n1d7-ace36d9`，revision `ace36d935b376fbf25cd56371e23877b95407c40`；其 `.venv/bin/python` 用于本轮 CPU 审计，已有 torch 2.7.1+cu128 / pyarrow 23.0.1 / pandas 2.2.3 / transformers 4.57.3 / av 16.1.0。
- BEHAVIOR：`/nvme3/projects/GR00T/BEHAVIOR-1K-2026-a8247a8`，revision `a8247a8cc1633fe1ca0cc66aa07243d46c64f155`，`v3.9.3-post2`；Python `/nvme3/envs/b1k2026-sim/bin/python`，Isaac Sim 5.1 / Kit 107.3.1。没有安装新环境或修改上游受版本控制的源码。
- 官方权重：`/nvme3/checkpoints/memory_vla/official-n17-radio/turning_on_radio_GR00T-checkpoint-150000`，摘要 `cd97a22ff74bd91838350e18fe94b48e1902d407f1256f5d48eb20befe277dc3`。模型 horizon 40、有效 action steps 16、物理动作 23、temporal ensemble、RTC off、单环境。
- 原样默认语言 job `97f45ef78924`：`evaluation/V_1_2/official/public_test_instance_18`；任务原文 job `3fc189e832d8`：`evaluation/V_1_2/official/public_test_instance_18_attempt_02`。原始备注、语言、配置、日志、JSON 和 MP4 保留。完整历史证据见 [官方验收报告](acceptance-2026-11-28.md)。

## 复现与接口

以下命令只读审计已有资源，不启动模型或作业：

```sh
/nvme3/projects/GR00T/Isaac-GR00T-n1d7-ace36d9/.venv/bin/python \
  scripts/audit_v12_b1k2026.py /nvme3/datasets/b1k2026-task0-task40 \
  --native-view /nvme3/datasets/b1k2026-native-loader-v028 \
  --upstream /nvme3/projects/GR00T/Isaac-GR00T-n1d7-ace36d9
```

已有工作台地址 `http://127.0.0.1:8765/?benchmark=b1k2026`。接口样例：

```text
GET /api/launch/options?device=local
GET /api/launch/jobs?benchmark=b1k2026
GET /api/workbench/annotations?benchmark=b1k2026&run=3fc189e832d8
GET /api/launch/artifact?run=public_test_instance_18_attempt_02&path=json%2Fturning_on_radio_319_0.json
GET /api/launch/artifact?run=public_test_instance_18_attempt_02&path=videos%2Fturning_on_radio_319_0.mp4
```

`model_versions.V_1_2` 提供 `task_catalog`、`checkpoint_whitelist`、`verified_job_ids` 和带 `available/reason` 的能力状态。目录存在不足以标为已验收：还须匹配原生完成回合、结果摘要、源码 revision 和环境/权重路径；启动 preview 仍执行原有完整权重摘要检查。每条原生作业显式输出 `benchmark=b1k2026`；回合键为 `json/turning_on_radio_319_0.json`，由各自 job ID 限定，不使用列表序号。Q-score 只取原生终态指标，不用步数推断任务进度。

本轮定向回归 21 项通过，现有 N1.7 环境下真实 Parquet/MP4 fixture 与标注测试 4 项通过。没有新建或下载浏览器，也未声称前端页面实测。8765 后端激活与真实 HTTP 记录见 [接口证据](evidence/b1k2026-adapter-http-v028.json)。

服务更新前核对本机评估 302 条、Benchmark 36 条、A100 训练 16 条、A100 推理 104 条，均无活动/排队作业；保留一条暂停训练记录。仅重启既有 `memory-vla-web-8765.service`，其 `KillMode=process` 不停止子作业。A6000 离线状态按未知保留，不操作；8766 未触碰。CPU 审计不占 GPU，不为占用率生造训练或评估。检查时两张 4090 均为 49140 MiB；已有 GPU0/GPU1 使用约 30510/6261 MiB 的其他进程保留。
