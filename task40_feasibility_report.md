# Task40 历史回放与仿真状态可行性报告

范围：GR00T N1.6 / MemoryVLA v1 / B1K-2025 `make_microwave_popcorn`。检查 `evaluation/` 中 Task40 产物，以日志最齐全的 `evaluation/V_1/memory/a6000_task40_eval10`（下文简称 eval10）做读取与对齐验证。仅运行单实例短探针；不运行完整 Episode、不训练、不改权重、不覆盖历史结果。

| 排查项目 | 结论 | 证据与问题 |
|---|---|---|
| 旧轨迹重建记忆 | **不可行（原样重建）** | 实测解码三路拼接视频并对齐帧索引；它不是逐策略调用观测。物理 trace 也不是完整策略状态。可制作稀疏视觉回顾，但未将近似输入写入 MemoryVLA，不能称为原记忆回放。 |
| 仿真状态恢复 | **部分可行** | 实例 4 的门前诊断快照恢复后，GR00T N1.6 BEHAVIOR1k 根据实时观测生成动作，仿真成功执行首步。旧日志没有中途快照，历史原状态和 MemoryVLA 会话均未恢复。 |
| 任务状态真值读取 | **部分可行** | 实际记录读取到开门、关门、抓取及启动的正例；`Inside` 接口匹配对象，旧样本只有负例。仅能确定采样区间；不能用中间状态代替 BDDL 终局成功。 |

## 1. 旧数据能否重建记忆？

eval10 的 `episodes.jsonl` 有 10 回合、任务名、实例、seed、长度及原生终止/目标结果。两个 env 各有以下数据：

| 文件 | 实际记录 | 能否重放 |
|---|---|---|
| `stages/actions_task40_gpu0_env*.jsonl` | 每 env 4,050 条；包含 `action_start/end`、`sim_time_s` 和六组策略输出。首条覆盖 `[0,8)`，但每组有 32 个预测位置（如 `action.base` 为 `32×3`）；不能把 32 个预测动作都当成已执行。 | 可取实际执行窗口内的动作作为开环重放输入；时间戳是在窗口执行后采集，不是策略观测时间。 |
| `stages/task40_gpu0_env*.jsonl` | 每 env 405 条；通常每 80 步，含位姿、关节、末端、袋子位置和真值。 | 可用作稀疏锚点；缺少完整逐请求 proprio（例如速度等），也不是模拟器快照。 |
| `stages/video_gpu0_env*.jsonl` + `videos/gpu0/*.mp4` | 每 env 1,345 个帧索引；每回合第 23 步开始，每 24 步一帧。播放器 20 FPS 不等于仿真时间。 | 可以重看三路稀疏视频。逐帧身份和时间关联后，与原策略动作起点精确重合 **0/2,690**。 |
| 任务指令 | 录像底栏显示 `Make microwave popcorn.`；环境从任务名构造相同指令，键为 `annotation.human.coarse_action`。 | 固定指令可重建；旧数据不包含实际逐请求 observation 包。 |

**实测读取：** `ffprobe` 检出实例 0 视频为 `768×280`、269 帧，恰等于其 frame trace 行数。实际用 `ffmpeg` 解码首 3 帧，逐帧对应仿真步 `23,47,71`（`1.1083,1.9083,2.7083 s`）。`behavior_env.py::preprocess_obs` 与 `video_recording_wrapper.py::step` 说明上方为按左腕、右腕、头部顺序拼接的 `256×256` RGB，另有 24 像素底栏；编码后像素并非无损原图。全 `evaluation/` Task40 路径未找到 `.npy/.npz/.pt/.h5/.parquet/.usd` 观测或仿真快照。`Output/task40_memory_overnight/replay/policy_cache/` 来源是独立的人类演示数据（见其 `configs/rebuild_policy_cache.py`），不是旧模型评估轨迹。

**为什么不能原样重建：** `src/v1_groot_memoryvla/policy/gr00t_policy.py::_get_action` 从 video/state/language 构建输入，并由 `MemoryState` 管理会话、episode 和顺序；`model/memory/adapter.py` 使用观测对应的视觉与语言上下文，按 `episode_keys/timesteps` 读写；`model/memory/bank.py::_write` 保存融合特征并在超容量时合并。缺帧、错一个控制步或丢失会话边界都不能靠动作序列补齐。v1 的记忆写入本身使用 backbone 特征，不要求历史动作作输入；完整策略调用及仿真重放仍各自需要对应状态与动作。

**最小修改：** 在 RPC 策略调用前记录原样 `observation`（三路命名 RGB、state、语言）与 `session/env/episode/decision_id`、输入仿真步和时间；调用后记录预测及实际执行动作范围。挑一个短窗验证同 checkpoint / processor 从同序输入重建 bank，并比对 bank 槽内容、时间和写入计数。也可同时保存 bank、`last_written`、merge 计数与策略时钟，作为恢复基准。无需为此重跑整集。

## 2. 微波炉门前仿真恢复

运行版本：Isaac Sim 4.5、项目实际 TaskProgress OmniGibson fork `4a6b45d02196c815bd0bfd5f6ff77e693b373821`，以及机器上固定 N1.6 评估源码。已核对 fork 的 `omnigibson/simulator.py::load_state` 和 `objects/controllable_object.py::_dump_state`：`og.sim.dump_state(serialized=True)` 打包已加载场景状态；同场景在 playing 时可 `og.sim.load_state(..., serialized=True)`，机器人对象还序列化控制器状态（基础状态包含目标有效位和目标数组）。`src/references_code/serf/src/serf_b1k/learning/eval.py` 另有初始状态 NPZ/RNG 参考实现，不是当前 GR00T 的中途续跑功能。

**探针方法：** 用官方隐藏 TRO 初始化同一 Task40 实例；把机器人 base 和关节放到历史 trace 的闭门、未抓取锚点，保留新实例的其余场景状态。执行一条旧动作令控制器有实际目标，保存 NPZ；用下一条旧动作改变状态；加载 NPZ，读取并比较，再执行同一旧动作。接入本机隔离 GR00T 服务时，在恢复后的第一个重放步之后取新 RGB/proprio/指令、推理并执行一个新动作。该探针检验恢复后评估路径是否可运行，不用历史录像充当实时观测。

**已验证的初次状态回环（实例 0，step 1120，厨房内但非门前）：** `TASK40_RESET_OK`、`TASK40_ROUNDTRIP_OK`，序列化 7,483 个数；机器人坐标最大差 `8.94e-8 m`，微波炉和袋子坐标差 `0`，机器人与炉门关节差 `0`，控制器 dump 一致；恢复后执行旧动作，返回三个 `256×256×3` 相机观测。仿真 `current_time` 从保存时 `0.3750 s` 到加载后 `0.4083 s`，并未倒退。因此不能单靠 sim dump 续接历史步数、任务累计状态和记忆。实例 0 的 GR00T 请求因诊断脚本把语言送成 ndarray 而被真实服务拒绝，脚本已改为标准 `[str]`。

**门前锚点：** 实例 4，seed 22004，历史 trace step 480，`Open=false`；最近末端到微波炉物体原点约 0.261 m。`episode_009_instance4_seed22004_gpu0_env0_s0.mp4` 的帧 19（step 479）头部视角可见右臂伸到微波炉旁，下一物理样本 step 560 为 `Open=true`。用该姿态新建锚点，不等于复原历史场景全部状态。

**门前恢复实测结果：** NPZ 含 7,387 个数。以下差值取加载前保存值与加载后立即读数的分量最大绝对差。

| 对象 | 位置差（m） | 四元数分量差 | 关节差 |
|---|---:|---:|---:|
| `robot_r1` | `5.96e-8` | `5.96e-8` | `0` |
| `microwave_hjjxmi_0` | `0` | `1.19e-7` | `0` |
| `popcorn_bag_73` | `0` | `2.98e-8` | 不适用 |

控制器 dump 完全一致。恢复后的旧动作执行正常，得到三路 `256×256×3` RGB。私有服务收到 1 个推理请求，返回六组有限值动作（预测视野 32 步），正常执行首步，仿真时间从 `0.44167 s` 走到 `0.47500 s`。服务使用 `checkpoints/groot_n1d6_behavior1k`，`memory=false`。因此已验证实时采集→正常 GR00T 推理→执行的单步闭环，MemoryVLA 记忆续接和后续阶段成功仍未验证。

**仍需处理：** 同版本、同场景、同一进程恢复不证明跨进程确定性；此次仿真启动也报告了部分资产 USD hash 不匹配。`load_state` 源码提示，`Inside` 等依赖物理后端更新的关系在加载后立刻读取可能失真，需至少一次 sim step 再比较。真正分支评估时，给关键帧一起保存场景/资源版本和实例、wrapper 步计数/历史、task progress/metrics、RNG 及 policy/MemoryVLA 会话；恢复后消费同输入并比对后续短轨迹。旧评估不能用姿态 JSON 代替该检查点。

## 3. 能否自动判断关键阶段真的完成？

实际历史对象：`microwave_hjjxmi_0`、`popcorn_bag_73`、`robot_r1`（assisted grasp）；Task40 BDDL 绑定分别为 `microwave.n.02_1`、`popcorn__bag.n.01_1`、`agent.n.01_1`。下表来自实际已采样结果；步数区间为 `(上次样本, 本次样本]`，不是事件精确时间。同一实例内比较，不能把 episode reset 算成转移。

| 状态 | 实际接口与值样例 | 判定和时间 |
|---|---|---|
| 微波炉门打开 | `microwave.states[Open].get_value()`；810 行中 143 行 `true`，实例 4 在 480→560 为 `false→true`。 | `Open=true` 即越过关节开门阈值，不保证完全敞开；可记录关节位置辅助判断可达性。 |
| 爆米花抓取 | `robot.is_grasping(arm=..., candidate_obj=bag)` 返回 `TRUE/FALSE/UNKNOWN`；实际实例 0 在 1440→1520 由未抓取到 `TRUE`；122 个正样本。 | 任一臂明确 `TRUE`，不能将 `UNKNOWN` 转 bool 或仅用私有 `_ag_obj_in_hand` 非空，因为 assisted 释放期间缓存可能滞后。 |
| 爆米花放入 | `bag.states[Inside].get_value(microwave)`；810 个样本都是 `false`。 | `true` 表示袋子 AABB 中心进入微波炉的容积；若要判断完成投放，应再检查已松手且状态稳定，不能以中心进入等价于整袋无碰撞。现有样本未证实正例；不能排除采样之间短暂发生。 |
| 微波炉门关闭 | 同 `Open`；实例 9 在 880→960 为 `true→false`。 | 关门事件需先开过，再转为 `false`；初始 `false` 不是完成关门，`false` 按关节阈值而非角度恰好为零。 |
| 微波炉启动 | `microwave.states[ToggledOn].get_value()`；实例 1 在 1200→1280 为 `false→true`，总计 179 个正样本（都在 env0）。 | 上升沿是启动，不证明已加热；可组合 `!Open && Inside && ToggledOn` 检查执行顺序，但不用它代替官方终局目标。 |

`problem0.bddl` 的原生终局是 `real(cooked__popcorn.n.01_1)` 且 `contains(popcorn__bag.n.01_1, cooked__popcorn.n.01_1)`。`cooked__popcorn` 是系统实体；袋子上探测 `Cooked` 并不检查这两个目标。建议通过 `env.task.object_scope` 绑定对象并直接记录原生 `done.goal_status`/目标谓词；用现有 `read_state` 在每个控制步的物理更新后记录状态及边沿（现有步长约 `1/30 s`，80 步采样约 `2.67 s`）。放入阶段补一个可控正例验证，再把它纳入自动阶段判定。子任务曾达成与最终仍成立分别记录，不用累计 TaskProgress 代替当前状态。

## 复现命令与日志

```bash
run=evaluation/V_1/memory/a6000_task40_eval10
out=$(mktemp -d /nvme3/tmp/task40-feasibility.XXXXXX)
python3 evaluation/scripts/task40_feasibility_audit.py "$run" > "$out/history.json"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q -p no:cacheprovider tests/test_task40_trace.py
```

离线审计依赖 `ffmpeg` / `ffprobe`；针对现有 trace 的测试结果为 `3 passed`（关闭本机不兼容的全局 ROS pytest 插件）。仿真应使用带 `TaskProgressWrapper` 的 fork 和含 Task40 `test_instances/` 的固定 GR00T 源码；默认 OmniGibson 不含该 wrapper，仓库内 `evaluation/sim/BEHAVIOR` 没有所需 TRO 目录。

用独立终端启动本机专用推理服务，确保端口空闲后再运行探针；也可省略 `--policy-port` 只检查仿真回环：

```bash
groot=/nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1
behavior=/nvme3/projects/GR00T/BEHAVIOR-1K-task-progress
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 \
  /nvme3/projects/GR00T/official-baseline-env/bin/python scripts/groot_b1k_live_probe.py serve \
  --source "$groot" --checkpoint "$PWD/checkpoints/groot_n1d6_behavior1k" \
  --host 127.0.0.1 --output "$out/policy" > "$out/policy.log" 2>&1 &
policy_pid=$!
# 日志出现 PROBE_SERVER_READY 后继续。
env -u OMNIGIBSON_DATA_PATH OMNIGIBSON_GPU_ID=0 bash scripts/b1k_python.sh \
  evaluation/scripts/task40_state_probe.py "$run" --behavior "$behavior" --groot "$groot" \
  --output "$out/probe" --instance 4 --policy-port 5568 > "$out/sim.log" 2>&1
kill "$policy_pid"
```

原始证据文件：

| 内容 | 路径 |
|---|---|
| 离线对齐、解码及逐事件区间 | `/nvme3/tmp/task40-feasibility.lJ9s12/history.json` |
| 门前历史录像截图 | `/nvme3/tmp/task40-feasibility.lJ9s12/instance4-anchor.png` |
| 门前仿真退出码 0，含 `TASK40_RESET_OK` / `TASK40_ROUNDTRIP_OK` / `TASK40_POLICY_STEP_OK` | `/nvme3/tmp/task40-feasibility.dBGZZW/sim.log` |
| 保存数据、恢复前/扰动后/恢复后/续动后的读数、策略动作结果 | `/nvme3/tmp/task40-feasibility.dBGZZW/probe/` 下的 `state.npz`、`roundtrip.json`、`policy.json` |
| 私有 GR00T 推理请求日志（末行 `REQUEST 1`） | `/nvme3/tmp/task40-feasibility.9BpXd7/policy-isolated.log` |
