# 任务一：原版 GR00T N1.6 / B1K 手臂操作诊断

日期：2026-11-13。实验对象为 **B1K-2025 50 任务微调权重**，不是通用 GR00T 权重，也不是此前三个其他任务训练 1000 步的 checkpoint。只研究 Task21/22/26；没有加载、修改或训练 Memory / Body 模块。

## 1. 结论及其边界

**本次没有发现双臂动作映射、反归一化或 loss mask 的明确错误。实际仿真能执行手动命令和真实示教动作；原模型也能通过短训改善操作动作的拟合。问题已收窄到操作预测的精度与时序、训练参数的稳定性，但尚不能宣称找到了全部历史闭环失败的单一根因。**

直接证据：

- 180 个操作决策点、完整 23 维动作的编码—解码最大误差为 `1.981e-7`。32 步 × 23 维标签全部参与 loss，没有把双臂 mask 掉。
- 9 条轨迹的原始与现有转换文件逐行比较：动作、状态、时间戳最大差均为 **0**，帧索引和语言索引一致。
- 实际仿真完成 **580 个动作步**：手动运动与夹爪开合 200 步、姿态过渡 60 步、真实示教回放 320 步。两臂控制器目标与输入命令逐值一致；示教回放的实际关节跟踪 MAE 为左臂 `0.000082 rad`、右臂 `0.000221 rad`。
- 180 点上，原模型两臂前 8 步 MAE 在三个任务里都未优于“保持当前关节位置”的参考预测。夹爪切换窗口比总体准确率暴露出更多错误。**不能把较小的总体 Loss 解释为已经学会抓取。**
- 同一抓取片段、相同初始化与样本顺序下，学习率 `1e-5` 训练 300 步后，左/右臂 32 步动作 MAE 分别降低 **45.6% / 73.7%**；`1e-4` 的 300 步结果反而变差。低学习率继续到 1000 步，手臂误差也回升。说明模型可学习该片段，但训练效果并不随步数单调改善。
- 数据并非全是导航：三个任务的操作候选窗口约占 **51.6% / 39.9% / 53.4%**。但单臂静止窗口很多，夹爪约 **98%～99%** 的 32 步窗口不切换，确有操作关键事件稀疏的问题。

另有两类独立问题：Task26 的 4 个原始 Parquet 损坏、现有转换目录元数据不完整；仿真日志出现任务物体 USD 与场景快照预期 hash 不一致。这些需要处理或核实，但本次没有证明它们就是抓取失败的原因。

## 2. 实验身份与范围

| 项目 | 本次实际值 |
| --- | --- |
| 权重 | `/nvme3/projects/memory_vla/checkpoints/groot_n1d6_behavior1k` |
| 官方模型源码 | `/nvme3/projects/GR00T/official-baseline-9b37aa1`，固定官方 `9b37aa1ce69c73c6d165233fa88128283bba4508` |
| 实际仿真适配源码 | `/nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1`，沿用现有评估适配与终止处理 |
| OmniGibson | `/nvme3/projects/GR00T/BEHAVIOR-1K-task-progress/OmniGibson`；Isaac Sim 4.5.0；当前资产 VERSION `3.7.2rc1` |
| 模型加载 | missing / unexpected / mismatched keys / error messages 均为空 |
| 状态、动作 | 82 个有效状态；23 个有效动作；标签 32 步；模型 padding 上限 128 维、50 步 |
| 相机与推理 | 原 head / left_wrist / right_wrist 三相机；官方 Processor；BF16；原 4 步 flow 推理 |
| 仿真控制实验 | Task21 / test instance0；Python 数值随机状态固定为 0；单环境；原相机、控制器、物理与任务规则 |

证据位于 [本次产物目录](../../../outputs/diagnostics/task1_groot_b1k/)。权重两分片完整性核对复用 [此前 SHA256 记录](../../../outputs/diagnostics/groot_b1k_20261110/checkpoint_integrity.json)；本次实际重新加载权重，没有以训练后权重冒充原始模型。统计文件 SHA256 为 `cefa9f4c612a350a4d406a501c6640d0a2371004d63d7b7e3c9703aac41a455b`。

本次只新增独立诊断脚本和产物，没有更改生产训练器、评估入口、模型结构、动作定义、任务判定或 Horizon。580 步控制实验不是正式 Episode，也没有计算成功率。`BEHAVIORGr00tEnv.reset()` 的实际实现不向下传递 `seed` 参数，因此不能以调用 `reset(seed=0)` 声称下层环境得到该参数；本次在构造环境前固定 NumPy / Torch 为 0，并使用固定的 test instance0。

## 3. 动作与控制链路

### 3.1 23 维顺序与关节映射

| 原始动作切片，左闭右开 | 模型字段 | 数据与最终控制器语义 |
| --- | --- | --- |
| `[0:3]` | base | 底盘速度控制输入；控制器从 `[-1,1]` 映射到最大 `[0.75,0.75,1.0]` 的平移/旋转速度 |
| `[3:7]` | torso | 4 个躯干关节的绝对位置目标；训练时转为相对当前 trunk_qpos |
| `[7:14]` | left_arm | left_arm_joint1…7 的绝对位置目标，单位 rad；训练时减当前左臂 qpos |
| `[14:15]` | left_gripper | 左夹爪 smooth 标量，`+1` 打开、`-1` 闭合；不是一个手指的 qpos |
| `[15:22]` | right_arm | right_arm_joint1…7 的绝对位置目标，单位 rad；训练时减当前右臂 qpos |
| `[22:23]` | right_gripper | 右夹爪 smooth 标量，定义同左 |

**关节底层存储是左右交错的，动作向量则按整条手臂分组。** 实际机器人左臂索引为 `[10,12,14,16,18,20,22]`，右臂为 `[11,13,15,17,19,21,23]`；控制器通过关节名称映射，不能直接取连续的 7 个机器人 DOF。实际 controller_order 是 `base → trunk → arm_left → gripper_left → arm_right → gripper_right`，与 23 维动作组一致。

源码证据：

- [官方 B1K schema 与相对动作定义](/nvme3/projects/GR00T/official-baseline-9b37aa1/gr00t/configs/data/embodiment_configs.py:211)。
- [实际 23 维拼接 preprocess_action](/nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1/gr00t/eval/sim/BEHAVIOR/behavior_env.py:281)。
- [实际控制器参数](/nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1/gr00t/eval/sim/BEHAVIOR/og_teleop_cfg.py:196)：手臂和躯干 `use_delta_commands=False`，没有额外把输入当增量；双臂没有再做一次输入/输出幅值缩放。
- [R1Pro 七关节名称](/nvme3/projects/GR00T/BEHAVIOR-1K-task-progress/OmniGibson/omnigibson/robots/r1pro.py:42)、[按名称产生控制索引](/nvme3/projects/GR00T/BEHAVIOR-1K-task-progress/OmniGibson/omnigibson/robots/manipulation_robot.py:839)。

### 3.2 训练预处理与动作反归一化

实际过程是：

```text
原始 action[t:t+32] + 当前 state[t]
→ 躯干/双臂逐步目标都减同一个当前关节状态
→ checkpoint 自带 min/max 统计归一化
→ 官方模型预测归一化动作
→ 同一统计反归一化
→ 躯干/双臂逐步目标都加回同一个当前关节状态
→ 绝对关节目标送入仿真位置控制器
```

不是把每个预测点累加到上一个预测点；也不是模型输出的相对值直接交给绝对位置控制器。依据是 [apply_action](/nvme3/projects/GR00T/official-baseline-9b37aa1/gr00t/data/state_action/state_action_processor.py:300) 和 [unapply_action](/nvme3/projects/GR00T/official-baseline-9b37aa1/gr00t/data/state_action/state_action_processor.py:388)。

本次使用 checkpoint 自带 Processor 和 statistics，没有重新估计统计量。196 个样本包含 180 个操作点和 16 个短训点，编码再解码的最大绝对误差 `1.981e-7`；23 个有效维度各有 32 个有效 mask。加载无缺失键。见 [open_loop.json](../../../outputs/diagnostics/task1_groot_b1k/open_loop.json)。

原始示教状态 256 维，当前仿真状态 258 维，后者多两个 grasp 状态，不能混用切片。本次读取正确的 raw256 切片；此前插入 grasp 哨兵后的逐字段验证复用 [interface_checks.json](../../../outputs/diagnostics/groot_b1k_20261110/interface_checks.json)。本次再检查 9 条完整轨迹的 raw256→packed82：全部逐值一致，见 [conversion_audit.json](../../../outputs/diagnostics/task1_groot_b1k/conversion_audit.json)。

### 3.3 Chunk 时间对齐和执行

- [官方 extract_step_data](/nvme3/projects/GR00T/official-baseline-9b37aa1/gr00t/data/dataset/sharded_single_step_dataset.py:13) 用当前时刻 `t` 的状态与图像，标签是 `t…t+31`，不是额外偏移到 `t+1`。
- [实际 MultiStepWrapper.step](/nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1/gr00t/eval/sim/wrapper/multistep_wrapper.py:258) 按 `value[step,:]` 顺序执行前 8 个目标，随后使用最新状态/图像再次请求模型；没有把第一个动作重复执行 8 次。原生 terminated / truncated 会结束当前 chunk。
- 实测动作周期 `1/30 s`，物理积分周期 `1/120 s`，每动作步对应 4 个物理子步。8 动作步是约 `0.267 s` 的仿真时间，不是 8 个物理子步。
- 本次控制实验直接走相同的 BEHAVIOR 单步与底层控制器；chunk 层的真实网络输入/执行对应关系复用此前 [400 步在线诊断](../../../outputs/diagnostics/groot_b1k_20261110/live_sim_400/analysis.json)，没有把离线数组比较当作在线执行验证。

### 3.4 手动与真实示教的实际执行验证

先分别把左/右臂第一关节平滑增加 `0.12 rad` 并返回，再进行两夹爪开/合。随后用 60 步逐渐过渡到示教片段起点姿态，执行 episode210120 的 `[788:1108]` 共 320 条原始 23 维动作。

| 检查项 | 实测结果 |
| --- | --- |
| 手动左臂运动段 | 左臂第一关节段内变化 `0.11527 rad`，左臂所有关节平均目标跟踪误差 `2.08e-5 rad` |
| 手动右臂运动段 | 右臂第一关节段内变化 `0.11527 rad`，右臂平均误差 `2.55e-5 rad` |
| 夹爪闭合 | 左右手指最终 qpos 约 `0…1.03e-6 m` |
| 夹爪打开 | 左右手指最终 qpos 约 `0.05000 m` |
| 所有 580 步：输入到双臂 controller goal 的最大差 | **0** |
| 示教回放：原始动作到实际发送动作的最大差 | **0**，包含底盘在内全部 23 维 |
| 示教左臂 MAE / P95 / 最大跟踪误差 | `0.000082 / 0.000347 / 0.000630 rad` |
| 示教右臂 MAE / P95 / 最大跟踪误差 | `0.000221 / 0.000903 / 0.007952 rad` |
| 示教右臂各关节段内变化范围 | `0.525 / 0.513 / 0.182 / 0.255 / 0.746 / 0.399 / 1.317 rad` |
| 完成情况 | 580 步完整执行，未 terminated / truncated；动作窗口耗时 `73.84 s`，不含初始化 |
| 原生任务配置 | `max_steps=38372`，没有修改 |

证据：[控制报告](../../../outputs/diagnostics/task1_groot_b1k/control/report.json)、[核对与 P95](../../../outputs/diagnostics/task1_groot_b1k/control/summary.json)、[逐步数据](../../../outputs/diagnostics/task1_groot_b1k/control/steps.jsonl)、[控制曲线](../../../outputs/diagnostics/task1_groot_b1k/control/tracking.png)、[三相机视频](../../../outputs/diagnostics/task1_groot_b1k/control/manual_and_demo.mp4)。视频为 H.264、768×256、193 帧/10fps，已用 ffprobe 全帧解码核对。

这些结果排除了本次窗口中“双臂不接收命令”“左右臂目标映射反了”“夹爪输出无效”一类直接控制故障。**回放没有恢复原示教片段的物体状态，不能据此宣称抓取成功或接触行为完全等价。** 此诊断窗口的 valid 字段尚未由 Episode 结束流程生成，不计入正式评估。

## 4. 三任务数据与动作分布

### 4.1 覆盖量与数据问题

| 任务 | 原始文件数 / 可读数 | 可读动作帧 | stride8 有效统计点 | 已有完整三相机转换轨迹 |
| --- | --- | --- | --- | --- |
| 21 collecting_childrens_toys | 200 / 200 | 3,837,265 | 478,947 | 12 |
| 22 putting_shoes_on_rack | 200 / 200 | 1,538,490 | 191,604 | 200 |
| 26 assembling_gift_baskets | 200 / 196 | 5,086,143 | 635,070 | 153，目录训练元数据未完成 |

共 **596 条可读示教、10,461,898 帧、1,305,621 个统计点**。统计遍历全部可读原始轨迹，不只统计用于预测的 9 条。动作与状态全部 finite；采样保留后续 32 步完整的窗口。

Task26 无法读取的文件为 `episode_00261770.parquet`、`00261780`、`00261790`、`00261820`，报 Parquet footer/magic 错误。本次明确排除并保存错误路径，没有静默补零或删除文件。见 [完整性记录](../../../outputs/diagnostics/task1_groot_b1k/parquet_integrity.json) 和 [详细统计](../../../outputs/diagnostics/task1_groot_b1k/data_statistics.json)。Task26 的 153 条转换文件不等于一个可直接启动官方 DataLoader 的完整数据集。

### 4.2 导航和操作候选片段

原始数据没有直接提供本诊断可用的逐帧“导航/抓取/放置”人工阶段标签，因此下面是可复现的动作代理分类：底盘最大绝对命令 `>0.05` 视为导航活动；任一手臂 `|a[t+31]-a[t]|` 的最大关节变化 `>0.03 rad`，或临近夹爪切换，视为操作候选。夹爪闭合/打开事件附近取 `[-16,+32]` 帧窗口，分别作为抓取/释放候选；**不等同于已经抓住或放置成功**。

| 任务 | 仅导航 | 仅操作候选 | 导航与操作混合 | 其余窗口 | 近夹爪闭合 | 近夹爪打开 |
| --- | --- | --- | --- | --- | --- | --- |
| 21 | 22.26% | 37.28% | 14.31% | 26.14% | 1.87% | 1.88% |
| 22 | 42.83% | 26.37% | 13.53% | 17.26% | 2.58% | 2.58% |
| 26 | 29.47% | 31.66% | 21.72% | 17.15% | 3.78% | 3.08% |

前四列互斥且合计 100%；最后两列与前四列重叠。操作数据并未整体缺失，Task22 的导航占比更突出，但“有操作片段”仍不代表单臂或开合关键事件样本足够。

### 4.3 各部件接近静止的比例

定义：32 步窗口首末目标之差，各关节最大值低于阈值。躯干/双臂阈值 `0.01` 关节原生单位；底盘/夹爪 `0.05` 控制标量。这个首末差代理会漏掉中间往返运动，不能当作严格逐帧静止标签。

| 任务 | 底盘 | 躯干 | 左臂 | 右臂 | 左夹爪 | 右夹爪 |
| --- | --- | --- | --- | --- | --- | --- |
| 21 | 61.11% | 52.27% | 58.01% | 51.30% | 98.93% | 98.48% |
| 22 | 49.68% | 63.79% | 62.91% | 69.65% | 98.38% | 98.36% |
| 26 | 58.25% | 42.48% | 50.43% | 53.11% | 97.60% | 97.60% |

### 4.4 动作幅值和归一化后的尺度

完整每关节 mean/std/min/max、相对当前状态的绝对幅值 P50/P90/P99、32 步首末变化分位数均已写入 [data_statistics.json](../../../outputs/diagnostics/task1_groot_b1k/data_statistics.json)，不把 23 个不同单位的量直接加总。

下表为各组第一目标相对当前状态、经过权重原统计量归一化后的各维标准差范围；底盘/夹爪没有减关节状态。**仅描述 chunk 第 0 个 offset，不代表全部 32 个 offset 的损失占比。**

| 任务 | 底盘 | 躯干 | 左臂 | 右臂 | 左/右夹爪 |
| --- | --- | --- | --- | --- | --- |
| 21 | 0.0757～0.3473 | 0.00135～0.00255 | 0.00414～0.01111 | 0.00342～0.00855 | 0.8978 / 0.8967 |
| 22 | 0.1215～0.4145 | 0.00091～0.00257 | 0.00244～0.00517 | 0.00251～0.00582 | 0.9764 / 1.0000 |
| 26 | 0.1243～0.5069 | 0.00132～0.00307 | 0.00334～0.00747 | 0.00301～0.00729 | 0.9972 / 0.9985 |

三个任务的手臂第一目标相对当前关节位置的 P90，跨关节约为 `0.0010～0.0119 rad`。这解释了为何接近“不动”的预测也可能得到较小动作误差，而抓取所需的精确小位移仍然容易出错。总体标准差大、局部 32 步不切换，两者可同时存在：夹爪的 ±1 状态在整条轨迹变化，但关键切换很少。

上述第 0 个 offset 的统计没有发现越过 checkpoint min/max 范围的值。对所测 196 个完整 chunk，往返误差检验也未发现明显裁剪损失。不能据此保证未测全部未来 offset 均不存在极端值。

![数据分布与近静止比例](../../../outputs/diagnostics/task1_groot_b1k/data_distribution.png)

## 5. 原版模型开环预测：180 个操作决策点

每任务取已有三相机数据的 3 条轨迹，每条 20 点，共 60 点/任务。采样优先包含操作、夹爪切换和较大关节变化，并用固定随机种子抽取部分操作点；这是操作诊断样本，**不是随机总体成功率估计**。轨迹和精确帧号见 [selection.json](../../../outputs/diagnostics/task1_groot_b1k/selection.json)。

使用当前帧三相机、当前 82 状态与原示教指令，官方 Processor / 原 B1K 权重预测。每点使用保存的固定 flow 噪声种子 `91000+i`；没有多噪声重复实验。以下按实际执行的前 8 步算 MAE，躯干是关节原生单位，双臂为 rad，底盘/夹爪为控制输入标量，不能横向混成一个物理误差。

| 任务 | 底盘 MAE | 躯干 MAE | 左臂 MAE | 右臂 MAE | 左夹爪 MAE | 右夹爪 MAE |
| --- | --- | --- | --- | --- | --- | --- |
| 21 | 0.007213 | 0.002548 | 0.007413 | 0.008802 | 0.309182 | 0.040556 |
| 22 | 0.013067 | 0.001417 | 0.008958 | 0.006625 | 0.186341 | 0.068186 |
| 26 | 0.008538 | 0.002467 | 0.006218 | 0.006551 | 0.273007 | 0.056481 |

对照保持当前关节位置的参考预测：

| 任务 | 左臂原模型 / 保持当前位置 | 右臂原模型 / 保持当前位置 |
| --- | --- | --- |
| 21 | 0.007413 / 0.007094 rad | 0.008802 / 0.007144 rad |
| 22 | 0.008958 / 0.008088 rad | 0.006625 / 0.004620 rad |
| 26 | 0.006218 / 0.005221 rad | 0.006551 / 0.005311 rad |

这不证明原模型完全不会操作：MAE 对随机生成预测、细小动作和多种可行轨迹均有局限，样本也有选择偏差。但它明确说明：**即便输入仍在示教轨迹上，所测操作点的手臂精确预测也未表现出可靠优势**，因此目前不能把全部问题归为闭环运行很久后的累计偏差。

夹爪总体符号准确率较高，例如右夹爪 96.7%～98.1%，但其中大量是保持开或保持闭。只看前 8 步真实发生符号切换的点：

| 任务 | 左夹爪：切换样本数 / 符号准确率 | 右夹爪：切换样本数 / 符号准确率 |
| --- | --- | --- |
| 21 | 12 / 47.9% | 3 / 62.5% |
| 22 | 10 / 42.5% | 6 / 66.7% |
| 26 | 17 / 58.1% | 2 / 18.8% |

样本数尤其右夹爪很少，只能作为切换时机问题的线索；不是抓取成功率。见 [conditional_metrics.json](../../../outputs/diagnostics/task1_groot_b1k/conditional_metrics.json)。

完整 32 步误差、逐点误差及原始预测/标签保存在 [open_loop.json](../../../outputs/diagnostics/task1_groot_b1k/open_loop.json)、[open_loop_arrays.npz](../../../outputs/diagnostics/task1_groot_b1k/open_loop_arrays.npz)。推理主体 196 点耗时 `34.11 s`，不含加载和准备。

动作曲线：[Task21](../../../outputs/diagnostics/task1_groot_b1k/task21_initial.png)、[Task22](../../../outputs/diagnostics/task1_groot_b1k/task22_initial.png)、[Task26](../../../outputs/diagnostics/task1_groot_b1k/task26_initial.png)。每组展示方差最大的一个关节，曲线横轴是抽样 chunk 的拼接，不能当作连续闭环轨迹。

## 6. 单轨迹操作片段短训

### 6.1 片段、训练实现和 mask 核对

选 Task21 episode210120 第 788～1107 帧，320 步 / 10.67 秒仿真时间。右夹爪在第 916 帧闭合；右末端相对机器人根的抬升约 `0.305 m`。已查看 [示教头部相机帧](../../../outputs/diagnostics/task1_groot_b1k/overfit_clip_head.jpg)，可见向地面粉色 teddy bear 伸手、闭合、抬起；相对末端位移不是物体世界坐标提升量。

从该片段取 16 个当前观测点，各带 32 步原始动作标签。只优化原版模型，AdamW、batch2、weight decay `1e-5`、grad clip1；初始模型/训练随机种子42；评估每点固定噪声 `120000+i`。为了检验记忆该片段，固定预处理后的图像、不做随机增强，不使用正式 warmup/cosine 调度。这是可控的片段拟合实验，不冒充完整官方训练配方。

默认可训练参数实际是 **1,619,968,000**：backbone `201,344,000`、action head `1,418,624,000`。配置 `tune_llm=False` 并不意味着整个 backbone 全冻结，因为官方仍解冻最后 4 层；依据 [Eagle 训练参数设置](/nvme3/projects/GR00T/official-baseline-9b37aa1/gr00t/model/modules/eagle_backbone.py:70)。本次保留此默认行为，没有擅自改解冻层数。

Loss 是原版 masked flow-matching velocity MSE：

```text
sum((pred_velocity - (action - noise))² × action_mask)
---------------------------------------------------
                   sum(action_mask)
```

依据 [官方 loss 实现](/nvme3/projects/GR00T/official-baseline-9b37aa1/gr00t/model/gr00t_n1d6/gr00t_n1d6.py:246) 与 [Processor mask](/nvme3/projects/GR00T/official-baseline-9b37aa1/gr00t/model/gr00t_n1d6/processing_gr00t_n1d6.py:340)。两臂共 14/23 个有效动作维度，并非按维数被底盘压倒；没有单独的手臂 mask 或人为低权重。首步 action head 有 473 个梯度张量、backbone 有 44 个，梯度非零。flow loss 不是反归一化关节误差，更不是抓取成功率。

### 6.2 单变量学习率对照

两次实验在 300 步之前只改变学习率，初始化、样本、顺序、评估噪声和评估时间点均相同。低学习率实验再延长到 1000 步，单独观察步数效应。以下均为该片段 **32 步**解码动作 MAE：

| 条件 | 底盘 | 躯干 | 左臂 rad | 右臂 rad | 左夹爪 | 右夹爪 |
| --- | --- | --- | --- | --- | --- | --- |
| 初始 B1K 权重 | 0.002570 | 0.002970 | 0.008002 | 0.040625 | 0.000040 | 0.031781 |
| LR1e-4，300 步 | 0.007998 | 0.005727 | 0.027602 | 0.053845 | 0.000588 | 0.027033 |
| LR1e-5，50 步 | 0.000956 | 0.001314 | 0.005015 | 0.019398 | 0.000897 | 0.011425 |
| LR1e-5，150 步 | 0.001402 | 0.001714 | 0.005190 | 0.016475 | 0.004308 | 0.004956 |
| LR1e-5，300 步 | 0.001189 | 0.000858 | 0.004352 | 0.010695 | 0.000675 | 0.003351 |
| LR1e-5，1000 步 | 0.005178 | 0.003166 | 0.006977 | 0.022496 | 0.000340 | 0.001727 |

300 步时低学习率相对起点：左臂降低 45.6%、右臂降低 73.7%、右夹爪降低 89.5%；左夹爪原本接近零误差，不能说所有部件都改善。1000 步右臂相对起点仍好 44.6%，但明显弱于 300 步结果。

由此可以确认模型与当前标签/梯度链路**有能力学习这段手臂操作**，不支持“手臂无法训练”或“手臂 loss 根本没生效”。尚未达到整段近零误差，也未进行该短训权重的闭环抓取验收，不能宣布操作问题已修复。这里的最佳测量点 300 步也不能直接推广为 600 条示教的最佳训练步数。

实验用时：LR1e-4/300 步 `159.51 s`；LR1e-5/1000 步 `425.46 s`，包含末尾 checkpoint 保存；两次 PyTorch 峰值 allocated 显存均约 `33.38 GiB`，不是整卡所有进程占用峰值。

证据：[高学习率记录](../../../outputs/diagnostics/task1_groot_b1k/overfit.json)、[低学习率记录](../../../outputs/diagnostics/task1_groot_b1k/overfit_lr1e5/overfit.json)、[逐更新 loss 与梯度](../../../outputs/diagnostics/task1_groot_b1k/overfit_lr1e5/overfit_updates.jsonl)。两次保存的是各自最后一步 checkpoint；300 步低学习率保留预测数组，没有另存权重。

![短训动作误差](../../../outputs/diagnostics/task1_groot_b1k/overfit_lr1e5/overfit_errors.png)

预测与真实动作曲线：[片段各部件](../../../outputs/diagnostics/task1_groot_b1k/overfit_action_curves.png)、[夹爪闭合附近的完整 chunk](../../../outputs/diagnostics/task1_groot_b1k/grasp_boundary_chunk.png)。该闭合示例中原始模型已经能预测开合边界，不能用它单独解释其他 180 点的夹爪错误；两者覆盖的是不同样本。

## 7. 对四类原因的判断与最小建议

| 原因类别 | 本次证据支持到哪一步 | 判断 |
| --- | --- | --- |
| 控制接口/动作转换错误 | 正确分组与交错关节映射；转换往返误差极小；手动/示教实际运动；双臂目标逐值一致 | 所测链路不支持此类直接错误；未覆盖所有接触状态 |
| 数据分布不合理 | 操作窗口约40%～53%，单臂近静止约50%～70%，夹爪关键切换稀疏；另有损坏/不完整数据 | 支持关键操作信号稀疏，不能简化为“全是导航”或断言它是唯一根因 |
| 训练不足/损失实现问题 | mask 完整、梯度存在、低 LR 可改善手臂；高 LR 变差，低 LR 延长也回升 | 不支持手臂 loss 失效；支持优化参数/训练稳定性需要逐项核验，不能直接归因“训练步数不足” |
| 开环正常、仅闭环累计误差 | 操作点上的原模型误差已不优于保持位置；没有执行同状态恢复的模型闭环对照 | 不能认定为“开环完全正常”；闭环偏移可能进一步放大，但本次没有量化其贡献 |

**最优先的模型侧线索是细小手臂动作和夹爪切换预测不够可靠；最直接可干预的训练线索是优化参数敏感。** 这是证据排序，不是已确认的单一根因。尤其不能根据本次 LR1e-4 结果断言历史训练一定用了错误学习率；此前某些 Task21 训练记录本来就是 LR1e-5。

最小建议按先后执行，下一次仍只改一个变量：

1. **数据问题先做定点修复**：恢复 4 个损坏文件；完成并验证 Task26 转换元数据。现有文件不删、不补零。它们能造成加载失败/漏数据，但不能解释 Task21/22 的全部操作表现。
2. **保留已验证的动作/统计/控制定义**：不尝试交换关节、放大手臂输出、把绝对控制改 delta，也不改物理/碰撞/成功规则。当前没有证据支持这些改动。
3. **先用小预算校准训练稳定性**：沿用原 B1K 起点，固定同一操作片段与相同预测噪声，采用已显示改善的 LR1e-5 作为对照；同时保留反归一化手臂轨迹、夹爪切换与若干 checkpoint 的短时真实操作结果。不要因总 Loss 下降就继续延长训练。本次固定 LR/无 warmup 配方仅用于定位，不替代正式完整训练配置。
4. **若训练稳定仍不足，再单独验证操作窗口采样**：提高伸手、夹爪切换和抬起窗口的出现比例，先不改动作权重或 loss 结构。静止/导航样本仍需保留，避免修好抓取后破坏导航。这是待验证建议，本次没有实施采样改动，也没有训练新网络。
5. **闭环归因仍缺一项同状态证据**：在可恢复的同一操作起始状态下，对照真实示教动作和原模型短窗口，分别看末端偏差、夹爪切换与是否建立接触；否则不能区分几何/接触条件变化与策略偏离的贡献。本次没有把姿态过渡后的演示回放当作这个实验。

## 8. 仿真环境中的已知未决项

- 首次控制诊断因当前 shell 继承的旧项目 `OMNIGIBSON_DATA_PATH` 失败，尚未执行控制。修正此次启动命令为本项目现有资产路径后，580 步成功完成；失败日志保存在 `control_failed_assetpath.log`，没有改资产软链接或生产脚本。
- 正确路径运行仍有资产 hash 与场景快照预期值不同的警告，按对象名去重为 **601 个**，包含 teddy bear、board game、dice 等。触发点是 [USDObject.check_hash](/nvme3/projects/GR00T/BEHAVIOR-1K-task-progress/OmniGibson/omnigibson/objects/usd_object.py:171)；[加载入口](/nvme3/projects/GR00T/BEHAVIOR-1K-task-progress/OmniGibson/omnigibson/objects/usd_object.py:103) 在加密资产情形检查的是密文文件。它证明所检查文件的字节不同，**不直接证明解密后几何、碰撞形状或视觉外观不同**。完整对象名单保存在 `control/summary.json`，尚未核实为内容版本差异还是重新加密差异，不能据此认定抓取失败的原因。
- 当前出现的 `observation not within observation space` 为 Gym 警告，未导致本次控制实验中断；本次未逐字段定位这个警告，不能把它当作抓取失败根因，也不能宣称所有字段空间声明完全正确。
- 场景导入本次在启动约 8 分钟后完成，控制从启动约 9 分 40 秒后开始。加载阶段仍有 CPU/GPU 活动与缓存写入，不是最后一条日志停止就代表卡死。本任务没有进一步进行启动性能优化。

## 9. 可复现实验入口与产物

在 `/nvme3/projects/memory_vla` 执行。以下只针对本任务的独立诊断输出，不会创建正式评估结果：

```bash
# 全部可读示教统计；已有产物可直接查看，无需重复跑
/nvme3/projects/GR00T/official-baseline-env/bin/python \
  scripts/diagnose_groot_task1.py stats

# 原始/现有转换轨迹逐行核对
/nvme3/projects/GR00T/official-baseline-env/bin/python \
  scripts/diagnose_groot_task1.py audit

# 原 B1K 权重，180 个操作点 + 16 个短训片段观测点
CUDA_VISIBLE_DEVICES=0 /nvme3/projects/GR00T/official-baseline-env/bin/python \
  scripts/diagnose_groot_task1.py infer

# 片段拟合两档学习率；运行会重写相应诊断产物，比较前先另存旧产物
CUDA_VISIBLE_DEVICES=0 /nvme3/projects/GR00T/official-baseline-env/bin/python \
  scripts/diagnose_groot_task1.py overfit --steps 300 --lr 1e-4
CUDA_VISIBLE_DEVICES=0 /nvme3/projects/GR00T/official-baseline-env/bin/python \
  scripts/diagnose_groot_task1.py overfit --steps 1000 --lr 1e-5 --subdir overfit_lr1e5

# 真实控制测试：固定 test instance0，使用现有资产；不计入正式成功率
OMNIGIBSON_DATA_PATH=/nvme3/projects/memory_vla/assets/simulation \
OMNIGIBSON_GPU_ID=1 \
OMNIGIBSON_APPDATA_PATH=/nvme3/tmp/memory_vla/task1_control_gpu1/appdata \
GROOT_EVAL_INSTANCE_IDS=0 GROOT_EVAL_SEED=0 GROOT_EVAL_SEED_MODE=fixed \
bash scripts/b1k_python.sh -u scripts/diagnose_groot_task1_control.py

# 从结果数组重新生成曲线和条件指标
/nvme3/projects/GR00T/official-baseline-env/bin/python \
  scripts/summarize_groot_task1.py
```

主要实现：[诊断脚本](../../../scripts/diagnose_groot_task1.py)、[实际控制脚本](../../../scripts/diagnose_groot_task1_control.py)、[汇总和绘图脚本](../../../scripts/summarize_groot_task1.py)。数值可追溯到 JSON/NPZ 和逐步日志；本次诊断训练与仿真进程均已正常结束，未停止其他用户进程。
