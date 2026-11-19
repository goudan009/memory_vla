# 评估与结果查看

双套指标（2026-11-13）：网页汇总区同时显示 **B1K 官方 Q-score、B1K 完整任务成功率、GR00T 累计任务进度、GR00T 回合成功率**；逐回合成绩表与录像页也分别展示。B1K 2025 榜单按 Q-score 排名，另列完整任务成功率，二者不是同一指标，见[官方榜单定义](https://behavior.stanford.edu/challenge/archive/2025/leaderboard.html)和[官方评估协议](https://behavior.stanford.edu/challenge/archive/2025/evaluation.html)。

| 展示指标 | 实际来源与含义 |
| --- | --- |
| B1K Q-score | 原生 OmniGibson `TaskMetric.final_q_score`；完整成功为1，否则统计各候选目标方案中初始未满足、终局满足的谓词比例并取最大值。显示百分比，同时保留0–1原值；不从GR00T进度估计。 |
| B1K 完整任务成功率 | 原生终局 `info["done"]["success"]` 的平均值，历史回合从已存的 `native_termination.success` 读取；新回合另存 `b1k_success`。 |
| GR00T 累计任务进度 | 细分检查项曾经达成即累计，后来丢失也不扣回；保持原 `mean_task_progress`。 |
| GR00T 回合成功率 | 原GR00T记录器在回合中对success取OR，保持原 `success_rate`。正常成功即终止的环境里通常与B1K终局成功率相同，不能为了区别两套指标人为更改判定。 |

实例：`V_0/baseline/groot_task21_1` 的GR00T平均累计进度34.8%，B1K平均Q为3.3%（0.0333），两种完整成功率均0%。34.8%不表示34.8%的回合完整成功。原生代码核对位置：本机实际环境的 `/nvme3/shared/b1k2025/BEHAVIOR-1K/OmniGibson/omnigibson/metrics/task_metric.py`、`learning/eval.py`，以及项目 `evaluation/sim/BEHAVIOR/behavior_env.py`。

新 `results.json.b1k_metrics` 包含平均Q及百分比、`full_success_rate`、Q／终局成功覆盖数量、缺失数量和统计范围；旧字段保持不变。历史记录通过网页只读补充，不改写原始JSONL；缺少原生终局记录时不从Q=1、累计进度100%或旧success推算，只显示“未记录”，并报告分母。fast-debug不汇总正式B1K分数，无效回合不混入有效回合均值。此处为本次有效已完成Episode的本地均值；官方提交另有50任务×10实例和缺失实例计0的规则，见[官方提交说明](https://behavior.stanford.edu/challenge/archive/2025/submission.html)，未完成本地运行不标为官方总榜成绩。已运行的仿真保持原进程，新采集字段从下次启动生效；网页仍可从旧进程保存的原生终局标记读取两套结果。

视觉诊断服务（2026-11-13）：在“推理评估”左侧的“视觉诊断服务”卡片，可手动启动／关闭 A6000 上的 Qwen3-VL-8B-Instruct、刷新状态和查看日志；固定使用 GPU 0、`100.100.2.30:8901`。服务由该机 `systemd --user` 的 `VLMService.service` 管理，网页关闭不会自动停止它。在“评估结果 → 录像与相机”的“诊断模型”中选择 Qwen 或“不启用视觉诊断”，再点“诊断这一集／整轮诊断”；这里也有启停按钮。选择模型本身不会自动启动服务或执行诊断，人工评分可独立使用。

录像页的回合下拉框下方直接显示该 Episode 的任务进度、Q 分数、结果和动作步数，切换回合同步切换指标；不使用整轮均值代替本回合指标。路径、seed、GPU/env 等身份信息仍可在日志与配置、回合记录悬停中查看。VLM 视觉进度和人工视觉进度分别展示，不替代官方任务进度。

本次“离线”是网页后端继承的 HTTP 代理指向已失效的 `127.0.0.1:12000`，Python urllib 不识别 `NO_PROXY` 中的 `100.64.0.0/10` CIDR，导致内网请求误走代理。VLM 健康检查和诊断请求现直接连接内网。后端入口 `evaluation/scripts/vlm_service.py`，接口 `GET /api/vlm/service`、`GET /api/vlm/service/logs`、`POST /api/vlm/service/start`、`POST /api/vlm/service/stop`；只控制现有诊断 unit，不调用 GR00T 的启停接口。

新训练使用 `V_2_memory_task22_train02`，新评估使用 `evaluation/V_2/memory/task22_eval03/`。方法统一为 `official`（未微调）、`baseline`、`memory`、`ablation_说明`；历史名称仍兼容。网页会把实际推理进程的所属训练、checkpoint 和源码哈希写入评估记录。完整规则、续跑边界和主要后端脚本见 [命名与版本管理](NAMING.md)。下文保留历史布局和命令兼容说明。

先读 [evaluation.md](evaluation.md)：正式比较固定报告 Q、成功率、任务进度及完整性，不临时挑指标。

工作台统一由 4090 控制，训练、推理与评估先选择设备，再选择 GPU。A6000 支持单卡开发训练、本机推理及已准备任务的快速仿真评估，设备离线时保留已有结果。4090 修改源码后单向同步到 A100 / A6000，运行使用独立快照；设备路径、同步命令和实测范围见 [工作台设备](../train/DEVICES.md)。

人工评价（2026-11-13）：在“评估结果”选择运行，点击任一回合编号，在“录像与相机”页的 VLM 诊断下方填写“人工评分与备注”。支持执行评分（0–100）、视觉进度（0–100%）、行为类别、失败／卡住阶段、是否接触目标、是否卡住、是否认同 VLM，以及文字依据；未判断的字段可留空，至少填写一项。点击“保存评价”可创建或修改，点击“删除评价”只删除本回合的人工评价。回合成绩表增加“人工评分 / 备注”列，方便逐集查看。没有 VLM 结果、VLM 离线或已结束回合没有录像时，也能独立填写。

人工评价以运行目录和 Episode 编号隔离，包含实例、seed、GPU/env 身份，保存在各评估目录 `human_reviews/*.json`，与 `results.json`、`episodes.jsonl`、`vlm_diagnosis/` 分开；0分、“否”和未判断分别保存。旧版未关联 Episode 的录像按完整录像相对路径隔离，避免不同GPU同名文件混淆。A6000 的评估镜像也把人工评价保存在4090的对应运行目录，刷新或远端数据同步不会清空。切换回合和5秒自动刷新保留当前浏览器标签内的未保存草稿；重新加载网页前请先保存。不会用人工分覆盖官方 Q、成功率、任务进度或 VLM 判断。

后端接口为 `GET/POST/DELETE /api/human-review`，用 `run`、`episode_index`（旧录像使用 `path`）定位，POST 的 `review` 保存结构化字段。实现集中在 `evaluation/scripts/human_review.py` 和现有 `view_results.py` / `result_viewer.html`。84项网页、HTTP、录像、结果隔离回归通过，真实 Chromium 另检查草稿保留、逐回合切换、保存、刷新重载、删除和手机布局；浏览器验收拦截人工写入，没有给真实评估数据添加测试评分。截图和部署检查位于 `outputs/shared/diagnostics/human_review_20261113/`。

所有命令在项目根目录执行。工具集中在 `evaluation/scripts/`，默认结果写入 `evaluation/V_1/baseline/`；用 `--version` 选择版本、`--group` 选择版本内的分组。旧命令路径保留兼容入口，不维护第二份实现。

```text
evaluation/
├── V_1/
│   ├── baseline/task22_eval01/
│   ├── memory/task22_eval01/
│   └── ablation_no_perception/
└── V_2/
    ├── baseline/
    └── experment_1/
```

推荐版本名 `V_1`、`V_2`，分组名 `baseline`、`experment_1`、`experment_2`（按此拼写）。两项也接受自定义单层目录名，如 `--version memory_v2 --group ablation_1`；不接受路径分隔符、空白或 `..`。各版本、各分组的运行编号独立递增。`--run-tag` 只记录备注，不影响目录名。旧参数 `--version baseline` 和 `--version experiment_N` 分别兼容映射到 `V_1/baseline`、`V_1/experment_N`；`--group experiment_N` 也接受并统一为 `experment_N`。这些参数只管理结果，不切换模型或改变评估规则。

仿真资产统一从 `assets/simulation/` 读取，训练演示数据放在 `assets/datasets/`，结构及外部链接目标见 [根 README](../README.md#资产与数据路径)。三个仿真资产目录已恢复。2026-11-09 曾实测 Task21/GPU0/single1/seed0、视频开启、256 动作步（32 次策略循环）通过，退出码 0，当时目录为 `baseline/groot_task21_3/`，此为 fast-debug，不计正式成功率。初始化 366.33s、reset 26.60s、动作窗口 35.82s；底层录像器未保留此截断回合的 MP4。

2026-11-10 新实测：Task0/instance0/seed0/GPU1/single1，连接原版多任务模型端口5556，256动作步正常退出；结果见 `V_1/debug/groot_task0_2/`。初始化180.31s、reset 8.61s，32次 Policy/Env 循环合计约31.36s（8.16动作步/s）；录像85帧完整解码，三路实时预览已实测更新。正式回合数为0，正式 Q/成功率为 null。这验证评估与录像链路，不代表任务已成功，也不代表其他49个任务均经过实际仿真验收。

**当前 Task21 推荐 single1。** 当天 single2 在 GPU0 已有约 8GB 占用时，实测出现 Vulkan `ERROR_OUT_OF_DEVICE_MEMORY`，随后 worker 段错误，主控报 BrokenPipe；GPU1 测试期间有其他 GPU 作业，亦未通过。源码中 AsyncVectorEnv 会先在主进程构建探测环境，其 `close()` 为 no-op，主进程仍保留仿真显存，再启动两个 worker。下面 single2/dual2 命令仅说明参数用法，当前资源条件下不保证可运行；不能把小于 10GiB 的启动检查当作双环境容量保证。

## 公共评估与模型版本的边界

本目录供 Baseline、V1 和后续 V2 共用：协议、任务/实例/seed、仿真适配、评分、录像、续跑和结果查看统一维护。各版的模型、训练、依赖、服务与在线历史状态放在 `src/<版本>/`；原版模型入口放在 `baselines/groot_n1d6/`。整体目录见 [根 README](../README.md)。

公共参数的存放约定见 [configs/README.md](configs/README.md)，模型/权重/服务/结果对应关系见 [targets/README.md](targets/README.md)。这两个目录目前提供职责与记录约定，现有启动器仍从 CLI 读取参数，不自动加载 target 或新配置文件。`--version`、`--group` 和 `--model-name` 仅决定结果归属/命名，切换模型需要连接确实加载该版本 checkpoint 的服务。

从固定官方 N1.6 源码集中来的 `sim/`、`rollout_policy.py`、`open_loop_eval.py` 和 `real_robot/` 是各版本可共用的适配代码，来源提交 `9b37aa1ce69c73c6d165233fa88128283bba4508`，许可见 [LICENSE.NVIDIA-GR00T](LICENSE.NVIDIA-GR00T)。仿真内部导入与 Gym entry point 使用 `evaluation.sim.*`，共享客户端/数据类型仍依赖固定 N1.6 的 `gr00t.*`；它们不导入 v1 或 V2 模型实现。

这些通用入口从根目录以 `python -m evaluation.rollout_policy --help`、`python -m evaluation.open_loop_eval --help` 调用，要求环境能导入固定 N1.6 `gr00t` 及相应依赖。不同版本通过远端 Policy Server 评估；上游保留的直接加载 `model_path` 分支仍只加载原版模型。Open-loop 是上游单步离线诊断，尚未接通记忆的顺序历史/reset，不能替代记忆模型的闭环验证。真机和其他仿真平台为上游参考，依赖与完整运行未在本次整理中验收。

**当前正式脚本仍调用原先外部固定 GR00T runner**（`GROOT_REPO`），不因代码迁移自动切换到这里的通用 rollout。公共迁移代码尚未进行仿真对齐验收；已有评分、运行路径、结果和协议文件均保持原状。`sim/` 内上游平台 setup 脚本依赖原仓库外部子模块布局，不是当前项目的通用安装器。

## 启动

```bash
# 默认：baseline，GPU1，单环境，Task21，10个完整Episode，保存16倍速视频
./evaluation/scripts/run_groot_eval.sh

# baseline：GPU1，Task22，双环境各5回合
./evaluation/scripts/run_groot_eval.sh --version V_1 --group baseline \
  --mode single2 --gpu-index 1 --task 22 --seed 0

# 实验第1版：相同任务、模式、实例分配与种子；先确保服务器加载实验模型
./evaluation/scripts/run_groot_eval.sh --version V_1 --group experment_1 \
  --model-name grootmeory --mode single2 --gpu-index 1 \
  --task 22 --seed 0 --run-tag memory_v1

# 快评：仅调试，最多1200底层动作步；不报告正式Q/成功率
./evaluation/scripts/run_groot_eval.sh --version V_1 --group experment_1 \
  --preset fast-debug --task 22 --debug-max-steps 1200

# 检查全部50个任务的编号、名称与本机实例是否齐全，不启动仿真
./evaluation/scripts/run_groot_eval.sh --list-tasks

# 短窗检查：Task0，单环境，最多256动作步；录像更密，便于观察动作
# policy-port 必须对应你要评估的模型；跨任务检查不代表该模型能完成新任务
./evaluation/scripts/run_groot_eval.sh --version V_1 --group debug \
  --mode single1 --gpu-index 1 --task 0 --seed 0 \
  --preset fast-debug --debug-max-steps 256 \
  --policy-port 5560 --video-steps-per-render 3

# 只检查参数，不启动仿真、不创建结果
./evaluation/scripts/run_groot_eval.sh --version V_2 --group experment_1 --dry-run

# GPU0 有其他作业时先排队，空闲后再启动单卡双环境
./evaluation/scripts/run_groot_eval.sh --mode single2 --gpu-index 0 \
  --task 21 --wait-for-gpu
```

模型名只是文件夹标签，模型由 Policy Server 决定。默认连接 `172.28.62.17:5556`，可用 `--policy-host`、`--policy-port` 修改。每张卡的进程只评估一个任务类型；`--task` 支持固定 GR00T 适配器注册的全部 B1K-2025 任务（0～49），不自动连续评估全部任务。默认仍为 Task21。

`--list-tasks` 显示本机官方测试实例是否齐全。全部50个任务的实例0～9现已备齐；新补的46项来自官方 `behavior-1k/2025-challenge-hidden-instances`，固定 revision `3f99a48ba0b3bc72d4441220c3d98cb0ed8417da`，460个 JSON 均核对了官方 Git blob SHA1。文件在 `/data/datasets/b1k2025/2025-challenge-hidden-instances/`，通过 `assets/simulation/2025-challenge-hidden-instances/` 及固定 GR00T 的 `test_instances/<task>` 链接读取；原有 Task0/21/22/26 的实例文件未覆盖。后续若文件缺失，启动前会明确报缺失，不进入 Isaac 初始化。官方实例准备入口是 `$GROOT_REPO/gr00t/eval/sim/BEHAVIOR/prepare_test_instances.py`；不能用训练实例或公开数据中的其他实例编号代替。任务全集是50项 B1K-2025 挑战任务，不是 BEHAVIOR-1K 的所有1000项活动。

首次检查建议单环境、256动作步、独立 `--group debug`。快速模式只缩短 runner 窗口，仍需完整加载场景；初始化可能需要数分钟。`--video-steps-per-render 3` 只加密录像采样，便于看动作；默认24为16倍速。确认动作和结果链路后再用1200步性能窗口；正式成绩必须移除步数参数并切换 `--preset official-eval`，使用预先约定的任务、实例与种子。

仿真加载期间请保留对应 Policy Server。评估的 EpisodePolicy 会将客户端的 `timeout_ms` 真正应用于 ZMQ 发送/接收（当前默认15秒）；推理服务断连或超时会报错退出并保留已有结果，不再无限等待。这里不自动重发动作请求，也不更换模型或实例。

允许与其他作业共卡：每卡 1 环境要求空闲显存严格大于 10240 MiB（10 GiB），每卡 2 环境要求严格大于 24576 MiB（24 GiB），不再限制已用显存必须低于10 GiB。双卡逐卡应用相同规则。未满足时默认退出；加 `--wait-for-gpu` 后每30秒检查（`--gpu-wait-interval` 可调整），日志显示 `free` 和 `required_free`。续跑默认等待。检查不预留显存，不保证后续场景加载峰值，且不会停止其他占卡进程。

“评估结果”选中运行后，详情顶部可“暂停评估”“续跑评估”“删除评估”；“推理评估”的启动记录中也有操作按钮。暂停通过该运行自己的调度器停止仿真，保留已完成回合和录像，未完成回合续跑时重新开始。等待显存的队列也可暂停。运行或清理期间禁止删除；停止后删除将整个目录移到 `evaluation/.trash/`（接口返回恢复路径），不永久清空文件。旧等待进程仍保留旧准入规则，暂停后续跑才采用新规则。

| 模式 | GPU | 正式默认回合 |
| --- | --- | --- |
| single1（默认） | 指定一张，默认1 | 10 |
| single2 | 指定一张，默认1 | 两环境各5，共10 |
| dual1 | 0和1 | 每卡一个环境，各5，共10 |
| dual2 | 0和1 | 每卡两环境，2+3回合，共10；需先快评验证资源 |

两卡分别评估不同任务时，在两个终端分别运行 `--mode single2 --gpu-index 0 --task 21` 与 `--mode single2 --gpu-index 1 --task 22`。两卡共同评估一个任务用 `--mode dual1` 或 `--mode dual2`，不再加 gpu-index。

默认 `--seed 0 --seed-mode sequence`；`fixed` 表示每回合同一seed；`random` 从已记录base seed生成可复现的回合种子；`--seed random` 会随机选择base seed。Baseline与实验必须使用相同的逐实例实际seed和环境分配，不能只看base seed一致。

默认开启录像：每24个底层动作步保存一帧，20 FPS播放，相对30 Hz仿真时间为16倍速；不改变策略相机输入、动作与任务判定。关闭录像加 `--no-record-video`；32倍速用 `--video-steps-per-render 48`，64倍速用 `--video-steps-per-render 96`。底层 `run_groot_priority_eval.sh --profile-cycles` 性能诊断仍默认关闭录像。续跑沿用已保存的录像配置，可显式用上述参数覆盖；已经运行的进程不会因修改默认值而改变。

## 网页推理服务

页面最上方统一选择 benchmark，下方始终使用「启动训练 → 推理评估 → 评估结果」三个页签，切换 benchmark 后对应的任务、可用操作和运行记录一起切换。

| Benchmark | 启动训练 | 推理评估 | 评估结果 |
| --- | --- | --- | --- |
| BEHAVIOR-1K | 现有训练表单；折叠的「训练前检查」可运行 V1 网络回归或真实动作短训练 | 现有模型服务与 B1K 闭环评估 | 「策略评估 / 开发快测」切换，分别查看闭环成绩和开发检查 |
| LIBERO | 五套件按任务准备官方示教；GR00T Baseline / V1 Memory；动作头全量或 LoRA；新建与断点续训 | 选择 LIBERO 权重自动加载策略；短回合调试、完整时长评估；保留零动作环境检查 | 训练 Loss、逐回合结果、成功率、双相机录像和截图、日志与配置 |
| MIKASA-Robo / SimplerEnv | 待接入 | 待接入 | 独立记录区域，目前无可启动接口 |

LIBERO 已接入本机单卡完整流程。训练页选择套件与任务，点击「准备所选任务数据」，完成后返回训练页启动训练。示教按任务下载到 `assets/datasets/libero/` 并校验固定版本的 SHA-256；Spatial Task0 的 50 条示教已准备。新训练从通用 GR00T N1.6 初始化，使用 LIBERO 自己的动作／状态归一化。两种微调方式均冻结 Eagle；Memory 同时训练记忆模块。训练预算在启动按钮前显示：快速迭代按每任务新增 300 步、最多 3000 步计算，是调试预算；论文步数仅作预算参考，不代表相同模型或完整复现协议。

训练结果可点「用此权重评估」跳转第二步，自动选择该权重与训练任务；也能选择其他评估任务。默认 32 步、1 回合用于调试，不统计完整时长成功率。「完整时长」预设采用每任务 50 回合及套件时长上限，可手动减少回合数；报告显示实际回合数，比较模型时须保持一致。每回合采用对应官方初态、重新清空记忆，结果页可切换回合播放双相机录像。模型与仿真由同一运行管理，结束后释放模型。仅需要检查安装时可改选「零动作 · 环境检查」，每次最多 5 个任务。GPU 指本机 GPU，启动前重新检查可用显存。协议、命令和验证范围见 [LIBERO 接入说明](../src/v1_groot_memoryvla/benchmarks/libero/README.md)。

开发快测使用当前 V1 开发代码与固定 Task0/34/40 缓存。「网络回归」先执行测试，再比较认知检索；「真实动作短训练」默认从同一 GR00T-N1.6-3B 初始化顺序运行 Baseline、关闭历史、Memory 三组，各 100 步。它不评估训练表单中选择的 checkpoint，不产生可用于正式推理的新模型。其结果显示动作 MAE、检索指标、梯度和短训练 loss；检查通过不等同于任务效果提升。诊断范围与 CLI 入口见 [快速迭代说明](../src/v1_groot_memoryvla/audit/QUICK_ITERATION.md)。

网页 Benchmark 记录保存在 `outputs/benchmarks/web/<运行编号>/`，包括计划、状态、日志、报告与媒体；LIBERO 训练还保存模型、归一化配置和优化器状态。每次运行独立保存，页面每 4 秒刷新；关闭浏览器或重启网页服务后仍可继续查看独立运行的进程。一次只运行一项网页 Benchmark 作业，可在详情中「停止本次运行」。LIBERO 训练会完成当前优化步并保存 checkpoint；此操作只针对所选作业。

在“推理评估 → 1. 启动推理”中先选 `V_几`，再在 Baseline / Memory 分组内选择权重；页面同步填写该次训练记录的源码路径，官方原始模型单列。也可以不启动新服务，直接在评估设置中选择已有服务。

“推理服务 · 管理与日志”按版本、Baseline / Memory 分组，包含网页和命令行启动的服务。点击“关闭服务”并确认后，会向对应 A100 进程发送退出信号；退出前显示“关闭中”，确认进程退出后自动从列表和可选服务中移除。页面每5秒检查，服务日志保留。关闭正在被评估使用的服务会中断相应评估。

## 看结果

```bash
# 浏览器查看全部版本，局域网地址 http://192.168.2.102:8765/
./evaluation/scripts/view_results.sh --host 0.0.0.0 --port 8765 --no-browser

# 终端直接汇总全部版本
./evaluation/scripts/view_results.sh --once

# 只看一版
./evaluation/scripts/view_results.sh --root evaluation/V_1 --once

# 只看某版的一个分组
./evaluation/scripts/view_results.sh --root evaluation/V_1/experment_1 --once
```

8765 是统一工作台入口：`/` 查看训练与评估，`/registry` 查看实验台账，`/ai` 打开研究控制台预览，`/ai/architecture` 查看解释版。也就是说，直接访问 `http://<主机>:8765/ai` 即可，不需要另开 8772；8772 仅用于独立预览服务。界面浏览器检查：`PLAYWRIGHT_BROWSERS_PATH=outputs/browser-tools/browsers node tests/result_viewer_browser.cjs http://127.0.0.1:8765`。

页面每5秒刷新数据。顶部折叠显示4090/A100服务状态，展开可查具体模型、运行和端口。左侧按 V_版本折叠分组，版本内区分 baseline 与实验组；版本按数字倒序排列，自动刷新保留展开状态，搜索时自动展开匹配版本。可按版本、任务、运行名或备注搜索，也可按状态筛选。右侧保留有效回合、平均Q、成功率和平均任务进度，并分为“回合成绩 / 录像与相机 / 日志与配置”三个页签。“回合成绩”显示全部已记录回合，不截取最后8集；长表格内部滚动，表头固定，悬停回合行可查看 GPU、env 和 seed。点击回合编号直接切到对应录像与相机。自动刷新保留所选页签、回合、表格滚动位置和播放进度；离开画面页签会暂停录像并关闭该页面的实时订阅。电脑端录像与相机并排，窄屏自动上下排列。仅更新页面 HTML 时，强制刷新浏览器即可，无需重启服务；修改服务端 Python 后需重启查看服务。关闭查看服务不会停止评估。

回合联动：点击“回合成绩”中的“第 N 集”，或使用“录像与相机”页签中的“查看回合”下拉框，录像和相机回放同时切换到该集。下拉框包含所有已记录回合；没有录像的回合会明确提示，不会显示其他集的画面。打开“显示三路相机”后，从同一个 MP4 同步展示头部、左腕、右腕；播放、暂停和拖动均跟随下方录像，采样频率沿用原录像，不是另外存储的高频观测。旧录像中已烧录的大字无法通过网页去除；新启动评估的录像将小字放在额外的24像素底栏，保留完整相机画面，策略输入不变。

实时查看：选中正在运行的评估，在“查看回合”中选择“当前回合 · 实时相机”，再打开“开启预览”。默认关闭，无需额外启动参数或 WebRTC 端口，仍使用本页的8765端口。每个 GPU/env 独立显示头部、左腕、右腕画面及实例、seed、环境内回合编号、动作步数和采样时间。最多每秒更新一次；若一次策略/环境循环超过1秒，更新频率随之降低。这是最新观测快照，不是30 FPS直播。超过5秒未更新时会提示可能正在 reset 或等待推理，不把旧画面显示为实时画面。切换到历史回合后取消该页面的实时订阅，避免当前集与历史集混用。

预览复用 `MultiStepWrapper` 已返回的三路策略相机最新帧，不额外调用 `render/get_obs`，不修改策略观测、动作、任务判定、原生 Horizon 或随机种子。开启时会有 JPEG 编码与少量磁盘/网络开销。关闭开关、切换评估记录或离开页面后停止该页面的预览订阅；没有其他页面查看时，采集端下一次检查（约1秒，受循环耗时影响）停止编码。页面异常退出的订阅最多15秒过期；多个设备独立订阅，一个设备关闭不会影响其他设备。`.runtime/preview/` 只覆盖保留每个环境的最新一份快照与订阅，不积累录像，运行结束随临时目录清理。

采集逻辑从本次改动之后新启动或续跑的评估生效；已启动的旧进程不会自动加载新代码，页面会提示“尚未收到预览采集器信号”。无需为启用功能中断已有评估。未在运行的评估不能打开实时预览，仍可查看下方已保存的回合录像。

顶部另有两张服务卡片，同样每5秒刷新：4090显示实际 rollout 主进程数、环境数、GPU及连接的推理地址；双环境按“1个评估进程、2个环境”计数，不重复计算启动器和 worker。评估进程是客户端，没有自己的评估监听端口；8765为网页端口。A100显示 GR00T Baseline/Memory 推理进程数、GPU、模型名及实际监听端口，并区分进程已启动与端口已监听。通过启动网页服务的同一用户现有免密 SSH 只读检查，默认主机 `172.28.62.17`，可用 `--policy-monitor-host` 指定；无法连接时显示“无法检查”，不当作0个服务。检查不发送推理或reset请求，监听端口存在不等于模型推理健康检查通过。

本地卡片同时显示 Task 编号及本次评估的版本/分组/运行名；A100 模型名包含 checkpoint 上级目录，例如 `memory_v1/checkpoints/step-010000`，避免同一步数的不同模型混淆。悬停运行路径或模型标题可查看完整路径。

直接在网页查看各版结果；也可读取每次评估目录内的 `results.json` 和 `episodes.jsonl`。不再生成或追加 `results.md`，已有文件保留作历史记录。

```text
V_1/baseline/
└── groot_task22_1/
    ├── results.json       配置、状态、汇总
    ├── episodes.jsonl     每回合即时落盘、含Q/实例/实际seed
    ├── gpu0.log           初始化/运行/报错
    └── videos/gpu0/       MP4录像，续跑另存gpu0_resumeN
```

实验放在 `V_数字/experment_数字/grootmeory_task任务号_次数/`。模型标签仍可用 groot；版本与分组由两级父目录区分，网页详情显示“版本 / 分组”。每个分组独立递增运行编号。仅 `--diagnostics` 额外保留诊断目录，平时运行中的隐藏 `.runtime/` 收尾后清理。

录像放在本次运行的 `videos/gpuN/` 中，每个 MP4 对应一个回合，不另建每回合目录。例如 `evaluation/V_1/baseline/groot_task21_1/videos/gpu1/`；续跑另存 `gpu1_resumeN/`。评估运行中在本次运行的 `.runtime/gpuN/` 下录制，正常结束、报错或中止后统一移入公开的 `videos/gpuN/`，不再依赖运行结束日志查找 `/tmp` 视频。每集结束封存视频后，网页会在随后一次5秒刷新中直接显示该录像，无需等待整次评估结束；正在写入、尚未封存的 MP4 不展示，不提供直播。网页只读打开已封存文件，不移动或修改录像。播放地址在最终归档前后保持不变，刷新不会重置选中视频和播放位置。`results.json` 的 `videos` 仍在整次运行结束时汇总；网页运行中另行发现已封存文件。

新启动的评估会将完成回合命名为 `episode_001_instance0_seed0_gpu0_env0_s0.mp4`：`001` 是本次运行 `episodes.jsonl` 中的回合编号，后面依次为实例、seed、GPU、环境槽和成功标记（`s0` 未成功、`s1` 成功）。编号在写成绩的同一把锁下分配，双环境和续跑不会各自从第1集重新命名。每条回合记录的 `video_file` 保存准确的公开路径，例如 `videos/gpu0/episode_001_instance0_seed0_gpu0_env0_s0.mp4`，网页据此显示“第 1 集 · 实例 0 · seed 0 · GPU 0 / env 0”。未完成回合仍保留已录制的视频，但不会分配正式完成编号。旧 UUID 视频没有可靠的回合映射时显示“未关联回合”，不会按排序或文件时间猜测。已在运行的进程继续使用原逻辑，新命名从下次启动或续跑生效；无需中断当前评估。

新启动的评估会保留所有已录制的视频，包括成功、失败、原生 timeout、调试截断和未完成回合；不再依据成功状态或 runner 上限删除。`reset()` 保留上一回合，`close()` 封存最后一回合。使用 Ctrl+C 或向启动入口发送 SIGTERM 时，runner 完成当前循环后关闭环境及视频编码器；Python 异常也会执行关闭。未完成回合的录像保留，但不生成正式完成成绩。请等待启动入口退出后再检查视频，不要用 `kill -9`；SIGKILL、断电或原生进程崩溃仍可能留下无法解码的 MP4。没有采样到任何视频帧的短回合不会生成 MP4；评估结果仍以 `episodes.jsonl` 为准。录像保留不修改策略观测、动作、任务判定、原生 Horizon 或随机种子。

环境明确返回 `valid=False` 的结束回合时，评估记录该次无效尝试后立即停止，不再 reset 或自动更换种子重试。单卡双环境中任一环境触发也会停止本次运行并关闭全部子环境。已有有效成绩与已录制视频保留，状态为 `failed`；`results.json` 的 `failure_reason` 记录错误类别、GPU/env、实例、seed 和动作步数，网页详情直接显示原因。`unrecoverable_physx_error` 表示 BEHAVIOR 已置 `_physx_crashed=True`，`invalid_episode` 表示其他明确无效回合；具体原始错误仍查 GPU 日志。`success=False, valid=True` 的正常任务失败会继续评估，不受此机制影响。

2026-11-09 修复同时应用于项目内共享适配器和正式脚本实际调用的外部固定 GR00T 录像器。修复前已启动的评估仍使用内存中的旧逻辑，不会自动更新；此前被删除的录像无法恢复。本次没有重启或中断正在运行的评估。

Task40 新启动的录像另存 `stages/video_gpu{gpu}_env{env}.jsonl`，每个实际编码帧记录 `recording_id`、回合实例/seed、`frame_index`、`action_steps`、`sim_time_s` 和 `video_fps`。这层记录复用原编码调用，不增加相机渲染、不改变动作或官方指标。当前stride24的首帧实测对应动作步23，之后每24步一帧；reset后帧号归零并更换recording_id。阶段时间轴按最近的实录帧跳转，物理采样恰好落在该帧时才标精确；帧间动作仍不可见。VL证据保存实录帧的动作步与仿真时间。旧录像没有索引时保持近似换算；不会补造精确时间或修改已在运行的进程。重复或损坏的索引也回退并保留warning。

| 字段 | 如何理解 |
| --- | --- |
| status / ok | 流程是否完整结束，不代表任务成功 |
| completed_episodes / requested_episodes | 有效完成数 / 计划数 |
| invalid_episodes | 仿真异常等无效回合，不能静默删除 |
| mean_q_score | 有记录的有效正式回合平均Q，0～1；缺失或调试为null |
| q_score_episodes / missing_q_score_episodes | 有Q / 缺Q的有效回合数，覆盖全才是完整Q成绩 |
| success_rate | 有效正式回合中的完整任务成功比例，0～1 |
| mean_task_progress | 过程中曾达到的细分检查项累计比例，0～100%，不等于Q |
| protocol | 协议ID和文件SHA256；旧运行标记未记录 |

正式报告确认 `official-eval`、`complete`、`ok=true`、完成数等于计划数、Q覆盖完整、实例/seed配对；固定任务集合全部报告。

## 网页启动评估

打开工作台后，进入“推理评估”，在同一页选择模型服务并配置评估；已有结果在“评估结果”中查看。训练和结果记录按版本、任务分组，可展开或收起。所有页面顶部直接显示 A100 和本机 4090 各卡的已用/总显存与 GPU 利用率，约每 5 秒刷新；连接或读取失败时显示未知，不按空闲处理。

训练页目前直接提交训练，没有通用的网页训练队列。GPU 已有进程时，未显式勾选共卡会拒绝提交；评估页的“GPU 忙时排队”仅适用于仿真评估。任务数据的“A100 已就绪”表示已有登记的训练格式数据与划分；“待转换 / 上传 A100”表示本机原始示教和三路视频齐全，可点击准备数据后再训练。

页面沿用“评估结果”的双栏布局：左侧“1. 启动推理”，右侧“2. 启动评估”。已有模型服务时，直接跳过第一部分，在“启动评估”中选择服务。窄屏自动改为上下排列。

需要新模型时，展开“启动推理”，选择 A100 权重、GPU 与独立端口；“推理高级设置”可核对源码和 Python。检查配置会在 A100 检查路径和端口，启动通过现有免密 SSH 调用同一套推理脚本。新服务自动选入下方评估，加载期间可配置任务，就绪后才开放评估启动按钮。端口监听状态不代表已验证模型推理质量。

“启动评估”中填写模型服务、任务、运行方式、种子及结果位置。版本与分组只决定结果保存位置。默认单卡单环境、正式评估、10 集、16 倍速录像；单卡双环境默认每环境 5 集。只有选择 fast-debug 才会出现步数上限。“更多设置”中可调整录像、复制已有配置、指定实例、种子模式、回合数和 GPU 排队。

“检查配置”只校验参数并显示命令、实例与输出位置，不启动仿真；“启动评估”在后台运行，关闭网页不会中断。启动状态与日志收在各自区域内，有记录时才显示入口，每 5 秒刷新。已有运行在“评估结果”中查看。切换“续跑已有评估”可从已保存回合边界恢复。

## 进度、停止、续跑

```bash
tail -f evaluation/V_1/baseline/groot_task22_1/gpu0.log
tail -f evaluation/V_1/baseline/groot_task22_1/episodes.jsonl

# 失败或中断的单卡正式评估，从Episode边界恢复
./evaluation/scripts/run_groot_eval.sh \
  --resume-from evaluation/V_1/baseline/groot_task22_1 --dry-run
# 核对剩余实例/seed后去掉 --dry-run
```

续跑沿用保存的版本/分组/协议/实例/seed，写回同一目录，不生成 Markdown 台账。支持single1/single2、sequence/fixed，每个槽位已完成数量须一致；未完成回合重新reset，不恢复物理快照。暂停同卡其他任务后再运行，或等待队列显存检查。

续跑默认等待 GPU 空闲，包括启动前被显存检查拦住、还没产生任何 Episode 的运行；不用额外加 `--wait-for-gpu`。等待的检查间隔也可用 `--gpu-wait-interval` 设置。

评估终端按Ctrl+C中止评估；查看服务终端按Ctrl+C只关闭网页服务。每回合Q在终止时记录，未结束不会显示实时Q。旧回合缺失Q仍显示“未记录”，不会推算补值。

## 旧结果迁移

```bash
python3 evaluation/scripts/organize_results.py
```

旧 `baseline/` 整体移到 `V_1/baseline/`，旧 `experiment_N/` 移到 `V_1/experment_N/`；顶层旧运行目录移到 `V_1/baseline/`。原始 JSON、JSONL、日志、录像与历史 Markdown 文件原样保留，不改写或合并成绩，也不覆盖同名运行。默认目标版本为 `V_1`，可加 `--version V_2` 指定其他版本。状态为 running 的旧目录保留原位置，网页仍可读取，待停止后再迁移；新迁移不创建兼容软链接。新布局的续跑使用新路径，并从路径继承版本/分组。
