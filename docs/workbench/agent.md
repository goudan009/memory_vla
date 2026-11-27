# 训练、推理与评估的工作台执行手册

在 4090 仓库所在主机使用 `http://127.0.0.1:8765`。设备的训练、推理与仿真能力见 [设备说明](../../train/DEVICES.md)。实际在线状态、支持任务、GPU 和路径以当前 API 响应为准。

Agent先完整读取 [robot-vla-experiment-ops](../../.agents/skills/robot-vla-experiment-ops/robot-vla-experiment-ops/SKILL.md)，其规定执行、设备用途和验收；本手册只说明工作台接口。具体配方及授权以当前实验任务为准。当前推理接口支持 A100 和 RTX 6000 Ada，4090 的 `local` 入口目前用于仿真评估。

新部署时，将 [设备模板](../../configs/workbench_devices.example.json) 和 [外部源码模板](../../configs/workbench_sources.example.json) 分别复制为 `configs/workbench_devices.json` 和 `configs/workbench_sources.json`，填写实际 SSH 地址和绝对路径。这两个本机配置受 Git 忽略规则保护。

Task 40 的采集和检查点恢复还需要实验存储中的仿真检查点、历史决策前缀和策略权重。仿真入口随源码提供。采集默认从 `Output/test_01/experiment_01/checkpoints` 读取锚点，从 `Output/task40_memory_overnight/replay/original_step4000` 读取权重元数据，并使用 `Output/test_01/experiment_09` 作为实验目录。

## 查看资源并分配任务

1. 读取 `GET /api/services`。当前顶层 `local` 对应 4090，`remote` 对应 A100，`a6000` 对应 RTX 6000 Ada。结合 `ok`、`checked_at`、GPU 显存、利用率和进程判断状态。离线、监控失败或数据过期时按未知处理。
2. 查询相关设备的 `options` 和 `jobs`。核对目标任务的数据、权重、源码和仿真是否就绪，同时检查已提交、排队和加载中的作业。将 GPU 编号和设备一起记录，核对已占用及即将使用的显存。
3. 优先复用相同任务和配置的已有结果、作业和就绪服务。推理服务按设备、权重、源码、历史模式和服务状态核对，评估端点还应从仿真设备可达。训练的权重等保存完成后再用于推理，推理就绪后再提交依赖它的评估。
4. 优先让无依赖的训练、推理或评估在合适的空闲设备上并行。在支持的设备上，按同类配置的显存峰值、模型加载峰值和实际吞吐选择多卡、单卡双环境或共卡。给波动和后续阶段留出显存。容量未知时先使用已验证的并发；确需提高时，从小规模验证起。保留任务范围、训练预算、验证划分和评估回合、实例、种子与协议。现有作业优先保留，避免争抢 GPU 或抢占端口。
5. 有余量时提交已明确要求、相互独立且能放下的下一项。当前无合适设备时使用已有排队功能，例如训练的 `enqueue` 和评估的 `wait_for_gpu`。在资源释放、作业完成或故障后重新查资源，再调度后续任务。
6. 每批评估或诊断结束后，核对本次启动的推理服务。按设备和端口确认没有运行中、排队中或已安排续跑的作业依赖后，及时通过工作台停止该服务，并复查端口和显存是否释放。仍有依赖、用户要求保留或归属不明的服务不停止；不要仅凭 GPU 利用率判定服务空闲。

旧Benchmark入口`v2_train`/`v2_evaluate`的仅started标签、固定1000步及离线评分/机制诊断均为历史实验细节，不是当前执行步骤，也不授权重启旧配方。V2.1划分和预算见当前task；正式训练按已批准用途和预算分类，不设通用最少步数。

## 调用工作台接口

训练身份分为方法版本、测试号、实验号和独立训练号。规范 run ID 为 `v2_1__test02__exp01__run001`，checkpoint 再追加累计 optimizer step，如 `__step003000`；对应目录为 `Output/test_02/experiment_01/`。`V_1` / `V_2_1` 是展示名，`v1` / `v2_1` 是 ID 前缀。源码模块名 `v2_1_groot_memoryvla` 和 CLI 的 `--source-version v2_1` 只是代码入口，不是模型版本登记。

工作台 `options.model_versions` 是唯一可启动版本目录；新训练必须显式传已登记的 `version` 与 `run_id`。preview/start 校验版本登记、版本与 ID 一致、目录唯一；身份缺失或校验失败即停止，不得猜编号或换旧入口绕过。通用训练表单分别填写模型版本、测试编号、实验编号和独立训练次数；V2.1 仍走专用入口。方法小版本（如未来的 `V_2_2` 或 `V_2_1_1`）只有在用户确认其方法含义并登记后才可用，实验编号、seed、重训次数、恢复作业和工程修复都不能自动增加版本。历史已核实的 `V_12_memory_task40_train01/02` 仅显示映射到 V1/Exp12，原作业 ID、配置和目录不改。

自动配对队列和独立分支续训同样必须携带 `test_id` / `experiment_id`；旧记录缺失这两个字段时不得推测并创建新 run。所有入口均以 `train/run_identity.py` 校验身份。

普通闭环评估支持 `repetitions`（正整数，默认1）和 `episodes_per_gpu`（每轮每卡回合数，默认单卡10、双卡各5），CLI 对应 `--repetitions`、`--episodes-per-gpu`。默认设置下5轮共50回合，仍归入一个评估目录。实例ID仍为0–9；多回合按环境的原生实例分区循环，保存每槽配额、偏移和逐回合seed。sequence/random模式为同一实例后续回合产生新seed，fixed模式的重复不能作为独立随机样本。单卡中断续跑保留原实例池与各槽偏移；旧版单轮续跑规则保持兼容。双卡续跑及剩余回合少于原环境数的续跑暂不支持。

早期 V_2_1 Stage1/baseline 的训练源码快照只包含记忆模型推理入口。对这类 checkpoint，工作台优先推荐已同步的兼容推理源码；启动器也检查该兼容性，必要时使用 `deployments/web_live` 并冻结为独立推理快照，实际路径和哈希保存在服务记录中。原训练源码和 checkpoint 保留；缺少兼容源码时，预检明确提示先同步源码。

实验台账的「数据权重」页支持逐层打开实际目录、当前目录文件名筛选、分页和 UTF-8 文本预览（最多 64 KiB），二进制权重显示文件信息，可继续查看同目录配置。磁盘卡片实时显示所在设备的总容量、已用与剩余空间；设备不可达时保留带有「上次扫描」标记的数据。目录总占用按需点击「统计目录大小」，浏览时不递归扫描整个数据盘。

「添加已有目录」登记本机或已配置远端上已有的数据 / 权重文件夹，不执行上传或复制；登记保存在 `outputs/registry_resources.json`，重新扫描后仍保留。在目录内可新建文件夹，删除文件或文件夹时移入同级 `.workbench-trash`，页面提供恢复刚删除项目的入口。回收站仍占用磁盘，存储根目录不能删除，恢复时不覆盖已有同名数据。

资源接口为 `GET /api/registry/storage`（`action=hosts|disks|browse|size`，`host=4090|A100|A6000`，浏览时传绝对 `path`，可附 `query`、`offset`、`limit`），以及同路径的 POST（`action=register|mkdir|trash|restore`）。远端使用既有 `configs/workbench_devices.json` 的 SSH 配置，浏览和预览限制在已登记资源与数据存储目录中。

已安装 `memory-workbench` 时，可直接使用 `status`、`train|infer|eval options/jobs` 和 `train|infer|eval preview/start -f <配置.json>`。CLI 的 `start` 会先预检，再通过同一工作台 API 提交。安装和用法见 [工作台 CLI](../../workbench-cli/README.md)。

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

Exp13训练支持工程选项 `fsdp_accumulation=retain`：以更多显存保留参数和梯度，累计到最后一个microbatch再同步，保留有效Batch和优化器更新频率。默认`shard_each`保留旧行为。已有Exp13迁移时，通过`/api/training/resume`的`preview:true`先核对完整断点；显式传入`fsdp_accumulation`、可选的`gradient_checkpointing`及`microbatch`（1/2/4/8/16），并传入已同步`engineering_source_digest`，再提交同一请求（去掉preview）。工作台建立独立源码快照、记录`engineering_revision.json`，不改原科学配置或原快照；后续普通续跑沿用最近一次工程源码及选项。关闭激活重算仅以显存换计算，不改变参数冻结配置。Adam分片须与原四rank布局一致，worker在载入时检查moment形状。当前retain实现每rank使用4个CPU loader worker，组顺序和每组seed独立于worker数；实际工程配置在详情中显示，ETA采用最近工程恢复之后的步耗时。

Exp13关闭验证的训练可在「续跑训练」中编辑「检查点保存间隔（步）」，默认1000步，支持1000的正整数倍；实际总步数仍遵循原续跑预算限制。对应`/api/training/resume`字段为`checkpoint_interval`，页面先发`preview:true`再提交。间隔从该次续跑起生效，按累计optimizer步数保存，后续续跑保留最近选值；历史断点和原配置记录保留。保存内容仍包含完整模型、优化器与各rank训练状态；更频繁保存会增加磁盘容量需求和保存耗时。

临时训练自检结束并记录验证证据后，在训练页选中对应终态作业，使用现有「删除」操作清理记录。确认作业的输出目录专用于该次自检且没有依赖后，勾选回收输出目录。删除后复查训练列表。

V2 本机短测接在现有 BenchmarkManager：`kind=v2_training_smoke`，`gpu` 为本机编号，`steps` 为 1–100（默认3）。它使用已同步的 Exp11 V1 6000步权重、两条官方示教各三帧经当前主干新编码的特征/动作和明确标注的人工状态测试值，只验收训练链路。HTTP 沿用 `/api/benchmarks/preview`、`start`、`jobs`、`detail`、`stop`。终态详情归档为当前实验的 `evidence/4090-<job_id>.json` 后，通过 `/api/benchmarks/delete` 删除专属临时记录和产物。已有工作台进程尚未载入新入口时，可直接调用同一 `BenchmarkManager.prepare/start/detail/delete`；作业仍使用相同 worker、GPU 检查、记录目录和 HTTP 查询，不需重启服务或另建调度器。
