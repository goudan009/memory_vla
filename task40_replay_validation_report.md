# Task40 历史记录与短时闭环验证

范围：Task40（`make_microwave_popcorn`）单环境、单实例，一条实时策略前缀和同一锚点上的两段短时闭环。未训练、未修改权重、未覆盖历史结果，未运行正式消融。

**工程验收通过。** 新轨迹的无损记录重建了与在线完全一致的双记忆库；在同一仿真进程恢复门前锚点后，MemoryVLA 连续完成 **10 次决策、80 个控制步**。原始和恢复分支均未开门；仿真恢复后首帧与原分支不完全相同，因此本实验不证明确定性续跑或已具备正式消融条件。

## 实测结果

| 排查项 | 结论 | 运行证据 |
|---|---|---|
| 完整历史记录 | 通过 | 实例 4，seed 22004；以历史 step 320 的机器人姿态初始化后，实时运行 9 次决策抵达门前。本次共 29 份实际请求及执行动作，逐决策无损读取通过；前缀 72 步、原分支 80 步、恢复分支 80 步。样本 `prefix/0000` 含三路 `uint8 [1,1,256,256,3]`、37 个 `float32` 本体字段、指令 `Make microwave popcorn.` 及 `float32 [8,23]` 控制数组。 |
| 记忆重建 | 通过 | `memory_compare.json`：两库各 9 条，时间索引依次为 0–8；张量和 dtype 完全相同，最大绝对误差 0；会话时钟、写入游标、容量与合并元数据一致。每库容量 64；本次截至决策 18 的合并次数为 0，未覆盖容量合并路径。 |
| 检查点恢复与闭环 | 通过（同进程） | 锚点控制步 72、下一决策 9。基座到微波炉原点的 XY 距离 1.348 m，右末端到原点 0.561 m。恢复后决策 9–18、控制步 73–152，未出现 RPC、仿真步进或记忆时钟错误；两库保持同一会话，最终各 19 条、`last_written=18`、`next_decision=19`。 |
| 任务真值读取 | 持续读取通过；未出现正例 | 跨三段 232 条逐控制步真值。两段续跑都出现右臂向门/面板区域伸手，门关节未变化且 `Open=false`；未抓取、未放入、未启动，原生目标谓词均为假。 |

### 恢复对比与非确定性

| 对象 | 位置最大绝对差（m） | 四元数分量最大差（考虑符号等价） | 关节最大绝对差 |
|---|---:|---:|---:|
| `robot_r1` | 0 | 0 | 0 |
| `microwave_hjjxmi_0` | 0 | `1.19e-7` | 0 |
| `popcorn_bag_73` | 0 | `2.98e-8` | 不适用 |

控制器 dump 完全一致；锚点任务真值保持一致。环境步计数回到 72，OmniGibson 全局时钟未回拨（保存时 `2.741667 s`，恢复时 `5.408334 s`）。后续每条分支仍连续递增，不能用全局时间戳直接对齐两个分支，应使用本地步数和决策号。

实测第一条恢复请求的 RGB 不与原分支逐值相同，左腕／右腕／头部 RGB 分别有 `192595 / 195031 / 189693` 个通道分量不同（各 `196608` 个分量；不代表变化幅度）。记录的 `state.joint_qeffort` 最大差 `1648.32`；首个八步机器人控制块最大分量差 `0.10784`。因此只能确认所测位姿、关节、控制器和任务状态恢复，以及策略能续接；不能声称渲染历史、求解器派生观测或后续动作也完全复原。仿真日志另有 USD 资产哈希告警，未阻止本次运行。

### 开门、抓取、放入、关门和启动

| 阶段 | 实际接口及结果 | 自动判定 |
|---|---|---|
| 开门 | `microwave_hjjxmi_0.states[Open].get_value()` 持续 `false` | `false→true` 为打开。 |
| 抓取 | `robot_r1.is_grasping(arm=…, candidate_obj=popcorn_bag_73)` 两臂始终 `FALSE` | 任一臂明确 `TRUE`；不把 `UNKNOWN` 当作抓取。 |
| 放入 | `popcorn_bag_73.states[Inside].get_value(microwave)` 持续 `false` | `true` 表示进入；判断投放完成还需检查已释放和持续保持。本次没有验证正例。 |
| 关门 | `Open` 未出现先开后闭 | `true→false` 为关门；初始关闭不计。 |
| 启动 | `microwave.states[ToggledOn].get_value()` 持续 `false` | `false→true` 为启动，不等同于已烹饪。 |

Task40 原生目标另行逐步读取：已存在熟爆米花系统且袋中含熟爆米花；两项始终为假。若发生状态变化，可定位到相邻控制步间隔（约 33 ms）。关键帧显示两段都在微波炉门/面板附近伸右臂，但不证明夹住门把手；两段右末端到微波炉原点最小距离分别为 `0.275 m`、`0.337 m`，门关节均保持约 `-4.69e-22 rad`。按“尚未打开便出现抓取、放入或启动真值”的可观测规则，**未发现实际后续阶段跳步**；不能仅凭负样本和画面判断策略是否内部误切阶段或有未成功的后续尝试。

## 消融准备度

已具备同进程、同锚点的短时分支工程能力，**还不应启动正式三组消融**。最小后续工作：让三种记忆条件均走相同的 `load_state→刷新观测` 路径（不要把未经加载的原始分支当正常记忆对照），先做一次同记忆重复恢复基线，比较首请求和短时控制漂移；随后仅替换银行内容，固定会话和决策时钟并校验组间隔离。若使用子任务完成率作为指标，另补一个可控的 `Inside=true` 且已释放的正例。超过容量的回放、跨进程恢复和自然起点的全程前缀均不在此次验收范围，按正式实验是否需要再验证。

## 验证方法

- 以隐藏测试实例 4 初始化。历史评估仅贡献机器人基座位姿和关节位置；不使用其动作、视频或记忆作为输入。新前缀每个决策都用实时三路 RGB、本体状态和指令调用同一 v1 MemoryVLA 权重。
- 独立探针复用现有 `BEHAVIORGr00tEnv`、动作预处理和任务真值记录模块。每个决策目录的 `observation.npz` 无损保存发送给策略的数值数组（三路原始**策略输入** RGB，即评估器输出的 `256×256` 视图，并非更高分辨率渲染缓冲）；`decision.json` 保存完整指令、会话、Episode、决策号、控制步和仿真时间范围及记忆游标；`executed.npz` 保存执行的八个解码动作及对应的机器人 23 维控制量。视频只供人工复核，不参与回放。
- 锚点要求门闭、未抓取袋子、机器人基座距微波炉原点的 XY 距离不超过 1.6 m，至少一侧末端到物体原点不超过 0.60 m，并已产生至少三次新决策。锚点保存 OmniGibson 序列化状态、控制器状态（随机器人状态）、评估器进度和指标、任务成功标记、两端 RNG 和策略会话/时钟，以及在线两套记忆。原分支继续执行后，清空服务端记忆并恢复初始 RNG，按决策编号从磁盘重放真实请求；对比两库张量、时间戳、写入游标、条目结构、容量和合并计数。重建库本身用于恢复分支，不以直接加载原银行冒充重建。
- 恢复后重新获取实时相机和本体状态，不使用原分支的录像、动作或策略输出。每次连续执行 8 个控制步，并核对服务端两库写入时钟。每个控制步后从实际 Task40 对象读取真值，不用模型推断标签或用累计进度代替当前状态。

## 运行命令

使用已安装的 B1K/Isaac Sim 和 GR00T 环境。OmniGibson TaskProgress fork 为 `4a6b45d02196c815bd0bfd5f6ff77e693b373821`；评估器为本机带隐藏 TRO 的 `Isaac-GR00T-n1d6-9b37aa1`。两端共享同一新产物目录，服务端只监听回环地址。

```bash
run=$(mktemp -d "$PWD/Output/task40_replay_validation.XXXXXX")
PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /nvme3/projects/GR00T/official-baseline-env/bin/python -u evaluation/scripts/task40_replay_policy.py \
  --checkpoint "$PWD/Output/task40_memory_overnight/replay/original_step4000" --output "$run" \
  > "$run/policy.log" 2>&1 &
# 等 policy.log 出现 TASK40_POLICY_READY 后再运行仿真。默认预留端口 5570。
env -u OMNIGIBSON_DATA_PATH OMNIGIBSON_GPU_ID=0 bash scripts/b1k_python.sh \
  evaluation/scripts/task40_replay_probe.py --output "$run" \
  --behavior /nvme3/projects/GR00T/BEHAVIOR-1K-task-progress \
  --groot /nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1 \
  --approach-trace evaluation/V_1/memory/a6000_task40_eval10/stages/task40_gpu0_env0.jsonl \
  > "$run/sim.log" 2>&1
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q -p no:cacheprovider tests/test_task40_trace.py
python3 evaluation/scripts/task40_replay_audit.py "$run" > "$run/history_truth_audit.json"
```

## 产物索引

目录：`Output/task40_replay_validation.GXcT5J/`。测试 `tests/test_task40_trace.py`：4 passed；离线审计逐一读取并核验 29 个请求的图像、动作、会话、步数、时间与写入游标。

- `policy.log`：`TASK40_MEMORY_MATCH 9`；`sim.log`：`TASK40_CHECKPOINT 9` 和 `TASK40_REPLAY_DONE 10 decisions`。
- `memory_online.pt`、`memory_rebuilt.pt`、`policy_rng.pt`、`memory_compare.json`：在线/重建记忆、策略 RNG 与逐项比较。
- `sim_checkpoint.pt`、`anchor.json`、`restore_compare.json`、`summary.json`：检查点、状态差值和闭环计数。
- `history_truth_audit.json` 及各段 `truth.jsonl`：逐步真值、阶段顺序、首请求差异。`prefix/short.mp4`（72 帧）、`original/short.mp4` 和 `restored/short.mp4`（各 80 帧）；均为 30 FPS 三机位，原始与恢复分支另存 `keyframes.png`。
