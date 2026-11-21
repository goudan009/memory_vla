# 训练、推理与评估的工作台执行手册

在 4090 仓库所在主机使用 `http://127.0.0.1:8765`。设备的训练、推理与仿真能力见 [设备说明](../train/DEVICES.md)。实际在线状态、支持任务、GPU 和路径以当前 API 响应为准。

新部署时，将 [设备模板](../configs/workbench_devices.example.json) 和 [外部源码模板](../configs/workbench_sources.example.json) 分别复制为 `configs/workbench_devices.json` 和 `configs/workbench_sources.json`，填写实际 SSH 地址和绝对路径。这两个本机配置受 Git 忽略规则保护。

Task 40 的采集和检查点恢复还需要实验存储中的仿真检查点、历史决策前缀和策略权重。仿真入口随源码提供。采集默认从 `Output/test_01/experiment_01/checkpoints` 读取锚点，从 `Output/task40_memory_overnight/replay/original_step4000` 读取权重元数据，并使用 `Output/test_01/experiment_09` 作为实验目录。

## 查看资源并分配任务

1. 读取 `GET /api/services`。当前顶层 `local` 对应 4090，`remote` 对应 A100，`a6000` 对应 RTX 6000 Ada。结合 `ok`、`checked_at`、GPU 显存、利用率和进程判断状态。离线、监控失败或数据过期时按未知处理。
2. 查询相关设备的 `options` 和 `jobs`。核对目标任务的数据、权重、源码和仿真是否就绪，同时检查已提交、排队和加载中的作业。将 GPU 编号和设备一起记录，核对已占用及即将使用的显存。
3. 优先复用相同任务和配置的已有结果、作业和就绪服务。推理服务按设备、权重、源码、历史模式和服务状态核对，评估端点还应从仿真设备可达。训练的权重等保存完成后再用于推理，推理就绪后再提交依赖它的评估。
4. 优先让无依赖的训练、推理或评估在合适的空闲设备上并行。在支持的设备上，按同类配置的显存峰值、模型加载峰值和实际吞吐选择多卡、单卡双环境或共卡。给波动和后续阶段留出显存。容量未知时先使用已验证的并发；确需提高时，从小规模验证起。保留任务范围、训练预算、验证划分和评估回合、实例、种子与协议。现有作业优先保留，避免争抢 GPU 或抢占端口。
5. 有余量时提交已明确要求、相互独立且能放下的下一项。当前无合适设备时使用已有排队功能，例如训练的 `enqueue` 和评估的 `wait_for_gpu`。在资源释放、作业完成或故障后重新查资源，再调度后续任务。

## 调用工作台接口

已安装 `memory-workbench` 时，可直接使用 `status`、`train|infer|eval options/jobs` 和 `train|infer|eval preview/start -f <配置.json>`。CLI 的 `start` 会先预检，再通过同一工作台 API 提交。安装和用法见 [工作台 CLI](../workbench-cli/README.md)。

| 任务 | 接口前缀 | 设备和参数来源 |
| --- | --- | --- |
| 训练 | `/api/training` | `options?device=a100` 或 `a6000`。核对任务数据、权重、`gpus`、验证配置和训练预算。请求字段见 `train/dashboard.js` 的 `parameters()` |
| 推理 | `/api/inference` | `options?device=a100` 或 `a6000`。核对目标设备上的 `checkpoint`、`source_dir`、`python_path`、`gpu_index`、端口和历史模式。字段见 `collectInferenceRequest()` |
| GR00T/B1K 评估 | `/api/launch` | `options?device=local` 或 `a6000`。核对 `ready`、任务、`policy_host` / `policy_port`、`mode`、协议、回合数、实例和种子。字段见 `collectLaunchRequest()` |
| Benchmark 面板已有任务 | `/api/benchmarks` | 先从 `options` 核对当前支持的任务和设备。请求字段见 `evaluation/scripts/benchmark_workbench.js` |

两个 `collect` 函数位于 `evaluation/scripts/result_viewer.html`。设备选择型接口的 GET 请求通过查询参数指定 `device`，对应的启动 POST 将 `device` 放在 JSON 内。评估的 `GET /api/launch/jobs` 汇总本机和远程记录。

以训练请求为例，`REQUEST_JSON` 是按当前任务填写的 JSON 文件，包含 `device`。推理和评估使用表中的接口前缀。

```sh
WORKBENCH=http://127.0.0.1:8765
curl --noproxy '*' -fsS "$WORKBENCH/api/services"
curl --noproxy '*' -fsS "$WORKBENCH/api/training/options?device=a100"
curl --noproxy '*' -sS --fail-with-body -H 'Content-Type: application/json' \
  --data-binary @"$REQUEST_JSON" "$WORKBENCH/api/training/preview"
```

检查 `preview` 返回的设备、参数和计划，核对资源后，用同一份 JSON 调用同前缀的 `/start` 提交实际任务。远程评估预览可能同步源码或诊断依赖，只查状态时使用 GET。实际启动和记录由现有工作台负责。接口报错时查看返回的 `error` 和设备状态。提交超时或结果不明确时先查 `jobs`，确认是否已创建再决定是否重试。

从对应的 `/jobs` 查询状态和进度，训练可从 `/api/training/detail` 查询运行详情，评估可从 `/api/results` 查看回合进度与结果。记录实际运行 ID、设备和 GPU、状态及日志或输出路径。需要续跑时使用已有记录和工作台提供的续跑接口。汇报提交、排队、实跑和完成的实际状态，并注明仍等待的资源或依赖。
