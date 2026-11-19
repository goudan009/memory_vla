> 当前目录与正式协议以 [evaluation/evaluation.md](../evaluation/evaluation.md) 和[使用说明](../evaluation/README.md)为准。默认结果在 baseline，实验须加 `--version experiment_N`；网页读取 JSON 结果，不再生成或追加 results.md。

# GR00T 优先任务评估

单卡1/2环境、双卡各1/2环境可使用新入口`evaluation/scripts/run_groot_eval.sh`，不加参数默认GPU1单环境正式评估Task21的10个实例；`--mode single2`为每环境5回合，快评必须显式加`--preset fast-debug`。支持`--seed-mode sequence/fixed/random`与`--seed random`，解析后的base seed和逐回合seed均落盘。完整命令和资源说明见[模式启动命令](GROOT_EVAL_MODES.md)。双卡4环境只提供入口，尚未实测验收。

新统一入口将结果保存为`evaluation/baseline/groot_task21_1/`，同模型同任务自动递增编号；Memory可用`--model-name grootmeory`。默认只保留`results.json`、`episodes.jsonl`和每卡一份`gpuN.log`，配置与性能合并进`results.json`；`--diagnostics`才额外保留详细诊断。每回合立即落盘。旧结果保持原位置，下面旧priority诊断入口的详细文件布局仅用于诊断；日常评估以[模式启动命令](GROOT_EVAL_MODES.md)为准。

Task21两卡各一个原生完整Episode已完成；实际38373动作步、逐回合种子/终止证据、全程吞吐和最终推荐配置见[完整验收报告](../evaluation/groot_official_task21_20261107.md)。

最新单卡n_envs=1/2匹配短窗：7.320→12.586动作步/s（1.719倍），但双环境初始化359→1131s、显存峰值43.55GiB；日常快速开发仍推荐n_envs=1。实测证据与模式定义见[单卡环境数报告](../evaluation/groot_single_gpu_nenv_20261107.md)。

`--profile-cycles 150` 统计前 150 个外层循环，窗口结束立即打印 `DIAGNOSTIC_PROFILE` 并保存 `diagnostic_profile.json`；它不会修改原生 Horizon，后续 Episode 继续正常运行。增加 `--diagnostic-only` 可在该文件完整写入后停止自己的仿真进程，状态记录为 `diagnostic_complete`，未结束的 Episode 不参与成功率或 Episode/h。配合 `--appdata-path /绝对路径/已有appdata` 复用已有缓存，默认关闭视频。日志 `GROOT_TRACE` 标记 Isaac Sim、OmniGibson、gym.make、场景导入、reset 和首次 policy/env.step 边界；同目录 `system_metrics.csv` 每 5 秒记录资源。不自动使用 strace，以免扰动性能。实测见 [单卡诊断报告](../evaluation/groot_single_gpu_diagnostic_20261107.md)。

分层诊断在命令前设置 `GROOT_EVAL_DETAIL_PROFILE=1`，并搭配 `--profile-cycles 150`。`diagnostic_profile.json` 的 `detail` 包含各调用边界的均值、P95、次数和累计时间，以及 chunk 内第 1–8 步的分别统计。`inclusive` 包含子调用，`self` 只扣除已埋点的直接子调用；父子 inclusive 不可相加。采用主机墙钟计时，不额外同步 CUDA，不跳过任何观测或渲染。场景导入五阶段计时独立记录在 `GROOT_TRACE`，不计入 rollout 窗口。

150轮分层实测与优化优先级见 [仿真分层诊断报告](../evaluation/groot_sim_detail_diagnostic_20261107.md)。

深入 `Simulator.step` 时再设置 `GROOT_EVAL_SIM_PROFILE=1`，统计 Kit app.update、现有 PhysX 前/后回调及两者间原生区间、接触回调、非物理状态/visual/system/transition 更新与源码中显式 fetch/fabric 同步调用。该开关默认关闭；设为0可保留上一层计时作对照。`physics.native_interval` 是现有 OG pre 回调退出至 post 回调进入的主机区间，包含引擎及其内部等待，不等价于纯 GPU 核执行时间。Kit 原生 app.update 的 self 残余也包含渲染、传感器、其他 Kit 更新与未单独暴露的同步，不能全部称作渲染。

可独立设置 `GROOT_EVAL_CAMERA_COUNTERS=1`，在计时窗口前后读取已附着 annotator 的 `Node.get_compute_count()`（包含 viewer 相机），结果在 `detail.camera_graph_counters`。不新建 annotator、不执行图节点、不复制图像或同步CUDA。图节点执行次数与RGB读取次数、物理子步数分别统计；这些次数也不等价于GPU内部采样数。

150轮深层开关对照、PhysX子步/相机图计数、非物理更新分解和候选优化风险见 [Simulator深层诊断报告](../evaluation/groot_simulator_deep_diagnostic_20261107.md)。后续 viewer、空 visual 和双 GPU 的独立 A/B 结果见 [有边界 A/B 报告](../evaluation/groot_ab_20261107.md)。`--audit-semantics` 额外保存真实任务实例、原生终止配置、相机/产品归属、初始物理状态和逐请求内容摘要；审计在 Policy/Env 计时之外，报告另列包含审计的墙钟吞吐。该开关只用于诊断。

`--disable-viewer` 在初始 reset 完成后暂停 viewer 的两个 render-product，保留相机 prim、annotator 和原始创建顺序，三个策略相机继续完整更新。默认保留原生 viewer；不传该参数即可回滚。它和 `--audit-semantics` 均只支持每进程 `--n-envs 1`。当前单卡短窗实测 viewer 优化约1.01倍；空 visual 过滤没有端到端收益，已从执行路径和CLI中撤回。双卡对照不叠加 viewer 开关。

双 GPU 诊断入口会检查实例不重叠、AppData 不共享、服务路由正常，并在两端 reset 后同时开始窗口：

```bash
python3 scripts/run_groot_dual_eval.py --run-tag paired_diagnostic \
  --instances-gpu0 0 --instances-gpu1 1 --profile-cycles 150
```

默认每卡一个进程、一个环境、一个诊断窗口。`evaluation/groot_dual_<run-tag>/` 保存计划、两个队列日志、路由探针和联合统计；任务目录另按 GPU 分开。只有传入 `--full-episodes` 才运行到原生 Episode 结束。正式运行可加 `--episodes-per-gpu 5 --instances-gpu0 0 2 4 6 8 --instances-gpu1 1 3 5 7 9`；Baseline 与 Memory 必须复用相同分配和 seed。两个独立进程共享服务并不代表模型并行推理。

Task21的本轮150轮实测为13.827动作步/s，较单卡匹配窗口约1.78倍（相对最快单卡约1.76倍）；完整Episode/h尚未测得。推荐完整Episode验收先保持每卡1环境、视频关闭和原生viewer，使用`--appdata-gpu0`/`--appdata-gpu1`复用各卡已有独立缓存。固定环境seed不会隔离服务端采样随机流；当前路由验证也不等于验证有状态Memory服务的会话隔离。详细证据、命令和回滚见A/B报告。

每次执行 `scripts/run_groot_priority_eval.sh` 管理一个仿真进程，所选任务在这个队列中串行执行。两块 GPU 分别启动一个队列，仍连接同一个 Policy Server：

```bash
# 终端 1：GPU 0，任务 21
./scripts/run_groot_priority_eval.sh --gpu-index 0 --tasks 21 --episodes 10 --run-tag paired

# 终端 2：GPU 1，任务 22、26
./scripts/run_groot_priority_eval.sh --gpu-index 1 --tasks 22 26 --episodes 10 --run-tag paired
```

若已经在 `scripts/` 目录，使用 `./run_groot_priority_eval.sh`。`--gpu`/`--gpu-index` 支持 0、1。两卡运行时输出和 AppData 会按 GPU 隔离；同一任务若并行执行，必须显式分配不重叠的实例 ID，例如 GPU0 用 `--instance-ids 0 2 4 6 8 --episodes 5`，GPU1 用 `--instance-ids 1 3 5 7 9 --episodes 5`。同一组实例应在不同方案之间复用做配对比较，而不是同时重复跑。

预设：`--preset fast-debug` 默认仅跑任务 21 的实例 0、1 个调试窗口，runner 上限为1200动作步（chunk=8时150轮），可用 `--debug-max-steps 80` 缩到10轮。它不修改底层原生 timeout、相机、动作或任务检查，只让外层 runner 提前结束；窗口记录带 `evaluation_mode=fast-debug`，`success_rate` 和 `valid_episodes_per_hour` 为 `null`，不能作为正式成功率。原 MultiStepWrapper 的 runner 上限返回 `terminated=True`，因此必须同时查看模式和 `native_termination`，不能仅凭 terminated 将调试截断视为原生任务结束。

`--preset official-eval` 默认跑任务 21、22、26，每个任务10个完整 Episode，实例0–9；runner上限仍为999999999，保留原生timeout和任务判定，并拒绝 `--debug-max-steps`。可用显式 `--tasks`、`--episodes` 覆盖预设默认值。`--n-envs` 支持1、2、4；多环境按 `env_idx` 分配实例列表的连续分区，实际实例编号以 `runtime_env*.json` 为准。

```bash
# 快速开发：150轮，保留三策略相机与默认viewer，不录视频
./scripts/run_groot_priority_eval.sh --preset fast-debug --tasks 21 \
  --gpu-index 1 --n-envs 1 --instance-id 0 --seed 0 --action-steps 8 \
  --debug-max-steps 1200 --monitor-resources --run-tag debug_task21

# 正式评测协议示例（不会因调试上限提前停止）
./scripts/run_groot_priority_eval.sh --preset official-eval --tasks 21 \
  --gpu-index 1 --n-envs 1 --episodes 1 --instance-id 0 --seed 0 \
  --action-steps 8 --run-tag official_task21
```

默认服务地址为 `172.28.62.17:5556`，可用 `--policy-host`、`--policy-port` 修改。以后切换双服务，只需让两个队列指定不同端口。

先查看计划、不启动评估：

```bash
./scripts/run_groot_priority_eval.sh --dry-run --gpu 0 --tasks 21
```

每个运行标识自动附加 `_gpu0` 或 `_gpu1`。队列日志和任务输出按 run-tag 隔离；OmniGibson AppData/shader cache 按 GPU 持久分开，以便后续运行复用编译缓存。`CUDA_VISIBLE_DEVICES` 保持不设置，使用 OmniGibson 的 `active_gpu`、`physics_gpu` 选择 GPU。停止一个队列只会终止其自己的仿真进程组。日常评估默认录像，每24个动作步采一帧，20 FPS播放为16倍仿真时间；用 `--no-record-video` 关闭，用 `--video-steps-per-render` 修改采样间隔。策略需要的三路 RGB 相机观测保持开启；`--profile-cycles` 性能诊断默认关闭录像。

每个回合结束、环境自动重置之前，会立即写入任务输出目录：

| 文件 | 内容 |
| --- | --- |
| `episodes.jsonl` | 每回合追加并同步保存：真实 instance ID、成功标志、结束时任务进度、有效性、步数、时间 |
| `manifest.csv` | 每回合更新表格；任务结束后补视频路径 |
| `results.json` | 每回合更新有效回合数、累计成功率、逐回合进度、平均进度和运行状态 |
| `profiling.json` | 环境初始化、reset、policy、vector env step、单环境 action-chunk step、视频编码、Episode 总时长的均值/P95及有效 Episode/h |
| `run.log` | 仿真日志及 `EPISODE_COMPLETE` 行，包含累计成功率 |

`episodes.jsonl` 与 `manifest.csv` 每回合写入 task、env_idx、instance_id、seed、success、task_progress、valid、实际动作步数、耗时及 terminated/truncated；JSONL 另存原生 timeout/predicate 判定，且每回合 `fsync`，中途退出可保留已结束的回合。`seed` 是实际重置种子，沿用既有 `base_seed + env_idx + 该槽位回合序号`（序号从0开始）；不同模型必须复用相同实例列表、顺序及进程分配。Episode/h 使用有效 Episode 数除以从仿真子进程启动到 rollout 完成的墙钟时间。

`--instance-id 0` 是 `--instance-ids 0` 的别名。`runtime_env0.json` 在 reset 后记录实际实例、种子和原生终止配置，并每100个外层循环更新动作步数和现有任务进度，便于完整 Episode 期间检查进展。双卡完整模式同样在释放启动屏障前核验真实实例编号。

原 MultiStepWrapper 只在 `terminated` 时中断 chunk，会忽略原生 `truncated` 并额外执行最多7个动作。已修复为收到原生 truncation 后立即停止；没有改动 B1K Timeout 的阈值或内部步数定义，也不把动作数强制裁成配置中的 max_steps。该边界修改不影响150轮未终止窗口。

客户端为每个请求携带独立 `session_id`、`env_ids`、`episode_ids`（通过既有 options 参数，不加入模型观测）。回合结束只重置已结束槽位；Gym NEXT_STEP 模式下先获取该槽位 reset 观测，避免下一回合的 Memory 读入上一回合末帧。现有 GR00T 忽略 options 且 reset 无状态，模型输入和推理保持原定义。

Memory 服务应在进程内创建 `EpisodeMemorySessions(memory_module)`，将 `policy.reset(options)` 转发为 `sessions.reset(options)`，并对每行观测用 `sessions.episode_id(options['session_id'], env_id)` 作为所有 Memory read/write 的 episode_id。这个键同时包含会话、环境与回合；重置会清除旧回合的 cognitive/perceptual 两个 bank，不影响其他请求。该控制器在 JIT 外调用。当前仓库只有 π0.5 Memory 集成快照，没有可运行的 GR00T Memory Policy Server；组件隔离测试通过不等于已部署的 Memory 服务验收。A100 Baseline 服务的共享随机流也尚未按会话隔离，相同环境 seed 不保证并发动作轨迹逐位重现。

成功率分母是已完成的有效回合数，未结束的回合不计入；PhysX 等导致的无效回合仍保留记录，但不参与成功率和平均进度。尚未完成任何有效回合时成功率为 `null`。任务进度直接取 BEHAVIOR 在结束时提供的 `task_progress`（0–100），该指标累计各子目标是否曾经达成。缺失时记录 `null`。

若后续回合崩溃或收到 Ctrl+C，之前的回合记录会保留，`results.json` 状态更新为 `failed` 或 `interrupted`。视频若启用，会在任务完成后收集。

本地 GR00T N1.6 的 Policy Server 使用 ZeroMQ REP，可接收多个独立 REQ 客户端，但逐条串行推理，不做双请求并行计算或自动批处理。当前策略的 `reset()` 不修改状态，适合先共用一个服务。是否能提高整体评估速度需要用两个真实仿真测量；协议层连通不能代表实际模型吞吐。
