# GR00T-N1.6 / B1K 推理专项诊断

日期：2026-11-10。证据目录：[本次诊断产物](outputs/diagnostics/groot_b1k_20261110/)。

## 1. 当前结论与证据边界

**尚未定位到能解释历史 GR00T Baseline 闭环乱动的单一根因。不能据此认定模型能力不足。**

已经取得的直接证据：

1. 从真实示范 episode 210020 抽取 8 个时刻（0、96、192、800、808、1448、1456、8192），覆盖初始姿态、移动和左右夹爪闭合前后。在同一官方权重、同一输入、同一噪声种子下，发布 commit `4e62473`、官方 `9b37aa1` 的本地评估源码、当前部署的 v1 源码，输入处理及模型原始动作、解码动作、Wrapper 输出、适配器输出逐值一致。比较的是完整保存的数组，不只是 shape。
2. 在上述样本上，动作归一化往返最大误差约 `3.8e-8`；Wrapper 和环境动作适配函数引入的差值均为 0。未发现动作反归一化两次、漏加当前关节位置、左右臂互换、把关节绝对目标当增量等问题。
3. 有一个已确认的来源管理错误：`baselines/groot_n1d6` 声称调用固定官方源码，实际默认导入 `owenliang/Isaac-GR00T` 的 `a96255c`。但当前正式评估走另一个源码目录；这项错误尚不能解释历史闭环失败。
4. 当前 5560/5561 加载的是 Task 21 训练后的 `step-010000` Baseline/M1，不能当作未经训练的官方原始模型。本次离线对照与短时诊断单独使用原始官方权重。
5. **一次 400 步真实仿真已经完成**：50 次请求在发布版与当前部署源码重放，raw / decoded 动作全部逐值一致；网络输入输出、动作拼接、前 8 步执行、最新状态取用均无数值错位。双臂实际位置对控制器目标的最大差为左臂 `0.001969 rad`、右臂 `0.002192 rad`。该窗口没有复现异常转臂，末尾双腕已看到地面 teddy bear；没有覆盖最终抓取与放置，不能据此解释历史完整任务失败。

示范上的输出合理，不能证明真实相机、在线状态、控制器和物理执行正确。下面分别标记静态证据、示范输入验证与真实在线证据；不会用 Task Progress 接近官方数字替代这些验证。

## 2. P0：实际版本、权重、配置

### 2.1 源码身份

| 位置 / 用途 | 实际身份 | 状态与意义 |
| --- | --- | --- |
| 当前项目 `/nvme3/projects/memory_vla` | `main`，`6919a4f7bf9eb618eb11cfe694c03fad917a23fa` | 大量既有修改与未跟踪文件，完整保留 |
| `src/references_code/Isaac-GR00T` | `main`，`a96255c00fca8c84fb8b27607e00f68da3713ef1`，2026-02-01，提交信息 `up` | 干净浅克隆；origin 是 `https://github.com/owenliang/Isaac-GR00T.git`，不是 NVIDIA 仓库 |
| 正式仿真和原始模型评估目录 `/nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1` | 无 `.git` 的源码归档 | 目录名不是 commit 证明；本次独立获取官方 commit 后，核对了主要模型、Eagle、Processor、状态动作处理文件。主要推理实现与官方 `9b37aa1` 对应；仿真 / rollout 另有本地修改 |
| 当前实际服务源码 | `/nvme3/projects/memory_vla/deployments/20261109_2100_task21_ddp_async_s42_resume_r4/src/v1_groot_memoryvla` | 服务进程 cwd、PYTHONPATH 与参数已记录；不是当前工作区文件直接提供服务 |
| A100 上另一个 `/nvme3/projects/GR00T/Isaac-GR00T` | `main`，`7b304f95b95790490b79bd263ba57015c3c12dfd` | 干净的更新版源码；当前服务没有从这里加载 N1.6 |
| OmniGibson 所在 BEHAVIOR 仓库 | `feat/task-progress`，`4a6b45d02196c815bd0bfd5f6ff77e693b373821` | 5 个 dirty 文件：env_base、task_progress_wrapper、vision_sensor、simulator、usd_utils。已读差异，主要为计时 / profiling，未发现其中新增动作重映射或额外物理推进 |
| 仿真运行时 | `/nvme3/envs/b1k2025-sim`，Isaac Sim 4.5.0；OG 源码 `__version__=3.7.1` | 本次实际启动日志确认 Isaac 4.5；资产 VERSION 文本为 `3.7.2rc1`，不能把源码版本、资产版本混为一谈 |

本次对照全部使用独立目录，没有对现有仓库执行 reset / checkout：

- `outputs/diagnostics/groot_b1k_20261110/release_4e62473`：`4e62473d5226c55784697944a5c9606a51927bfc`。
- `outputs/diagnostics/groot_b1k_20261110/official_9b37aa1`：`9b37aa1ce69c73c6d165233fa88128283bba4508`。
- 本次 `git ls-remote` 核实官方 **`n1d6` 分支指向 `9b37aa1`**。另一个名为 **`n1.6-release` 的引用是标签**，指向 `ead52833afbbf4243f8cd5e7664f48a94de03b19`；不能把它自动等同于用户指定的最初发布 commit `4e62473`。

关键文件的原文 SHA256 和去除注释 / docstring、统一包名后的 AST 指纹保存在 [source_comparison.json](outputs/diagnostics/groot_b1k_20261110/source_comparison.json)。AST 相同只是静态证据，最终还进行了实际模型推理比较。

### 2.2 实际正在使用的 checkpoint

证据：[runtime_inventory.json](outputs/diagnostics/groot_b1k_20261110/runtime_inventory.json)，采样时间 `2026-11-10 16:17:04 +0800`，主机 gpuserver。

| 服务 | 实际权重 | 关键参数 |
| --- | --- | --- |
| `172.28.62.17:5560`，PID 2038509 | `.../20261109_2100_task21_ddp_async_s42/baseline/checkpoints/step-010000` | `memory_enabled=false`，`BEHAVIOR_R1_PRO`，开启 sim wrapper，seed 42，CUDA_VISIBLE_DEVICES=3 |
| `172.28.62.17:5561`，PID 2038009 | `.../20261109_2100_task21_ddp_async_s42/memory_v1/checkpoints/step-010000` | `memory_enabled=true`，其余对应，CUDA_VISIBLE_DEVICES=2 |
| 本次独立原始权重对照 | `/nvme3/huggingface/train_model/GR00T/GR00T-N1.6-BEHAVIOR1k` | 官方原始模型，不是上述两个训练产物 |

`...` 的完整公共前缀是 `/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training`。没有更改、重启或向两个已有服务发送诊断请求。任务开始时已有两组正式仿真在运行；等它们自然退出、GPU 空闲后，才使用独立服务执行本次短时诊断。

原始模型本地 HF 下载元数据 revision：`300db814db8ab5dd010026d5631f280048d06b91`。实际读取两个完整 safetensors 分片计算 SHA256，与本地 HF LFS 下载元数据相符：

```text
model-00001-of-00002.safetensors
fe30dd24dccf6d1929fc63822927212c9cb88da4d86cc97a6b7b33404e6d16b5
model-00002-of-00002.safetensors
be85e06694f547f9f2d842f14d54df83acaac1285acfc5629c86fdbb8789eca2
```

这是文件完整性证据，不表示重新下载并核实了 HF 当前最新 revision。详见 [checkpoint_integrity.json](outputs/diagnostics/groot_b1k_20261110/checkpoint_integrity.json)。

上述“当前服务”是 **16:17 的进程快照**。交付前再次只读核对时，5560/5561 已不监听，两个旧 PID 也已退出；本次诊断没有向它们发送 kill 或 reset，不能从这次核对推断退出原因。本次专用 5568 服务在 400 步后由自己的客户端关闭，GPU 已释放。

### 2.3 模型、Processor 与训练 / 推理一致性

| 项目 | 实测 / 配置值 |
| --- | --- |
| 模型架构 | `Gr00tN1d6`，Eagle + AlternateVLDiT，`use_vlln=true` |
| 模型输出上限 | action horizon 50、动作维度 128；实际 B1K schema 取前 32 个时间点、23 个有效动作维度 |
| 状态 | 82 个有效量，padding 到 128；单个当前时刻 `[0]` |
| 模型推理 | BF16，4 个 flow matching 推理时间步 |
| Embodiment | 配置、服务参数均为 `behavior_r1_pro`，内部 ID 均为 **24** |
| 视频顺序 | head → left_wrist → right_wrist；三路当前帧 |
| 图像 | shortest edge 256、eval 中心 crop fraction 0.95、最终 Eagle pixel_values 为 252×252 |
| 相对动作 | torso、left_arm、right_arm 是相对当前原始关节状态的 NON_EEF；base 和两夹爪是 ABSOLUTE NON_EEF |
| 动作反处理 | min/max 反归一化，再对三组关节加当前关节状态；不是相邻预测步逐步累计 |
| statistics | 原始、训练后 Baseline 和 M1 的 statistics.json SHA256 全部为 `cefa9f4c612a350a4d406a501c6640d0a2371004d63d7b7e3c9703aac41a455b` |
| Processor 配置 | 原始 `e95db539fd9d9f28504c5c9aa87abde38d4d5edcfe19d0e4d5778dd97f4baa71`；训练后两组相同，为 `bb8d1613eb71889481c446fadf72c9cdfb3ad7b852aa67b2c7d6920151ea1692`，增加了 B1K 语言映射 |

原始 `config.json` SHA256：`cd265fe9f1c037188a09d317155d3a4c28d1db1a5d8f0bb129934b5e170fdd27`。

训练恢复记录 [resume-step-001000.json](outputs/diagnostics/groot_b1k_20261110/resume-step-001000.json) 明确给出训练源码 `...resume_r4/src/v1_groot_memoryvla/experiment/task21_ddp.py`，与当前服务源码树一致。[ddp.json](outputs/diagnostics/groot_b1k_20261110/ddp.json) 记录原始 checkpoint、冻结 Eagle、1e-5 学习率、10000 步等。未发现“训练用 N1.6、当前服务却导入 N1.7”的证据。训练确实改变了 action head 参数；不能把训练后的 Baseline 输出归因于原始模型。

官方模型卡明确要求相同 embodiment 与 sim wrapper：[NVIDIA 模型卡](https://huggingface.co/nvidia/GR00T-N1.6-BEHAVIOR1k)。官方 50 任务评估说明也指定 `feat/task-progress` 和 `n_action_steps=8`：[NVIDIA BEHAVIOR 评估说明](https://github.com/NVIDIA/Isaac-GR00T/blob/9b37aa1ce69c73c6d165233fa88128283bba4508/examples/BEHAVIOR/README.md)。

## 3. 与官方实现有关的关键差异

下表行号指当前文件 / 本次独立官方 worktree，不能用文件夹命名冒充版本证据。

| 文件、函数、行号 | 实际差异及影响路径 | 结论 |
| --- | --- | --- |
| `baselines/groot_n1d6/README.md:3`，`serve.py:10` | 文档宣称官方固定版本，入口实际无校验地把 `src/references_code/Isaac-GR00T` 放入 sys.path；其 HEAD 实为 fork `a96255c` | **确认来源错误**；尚无其造成当前正式评估乱动的证据 |
| 官方 `gr00t/model/gr00t_n1d6/gr00t_n1d6.py:464`，fork 同名 `prepare_input` | fork 直接包成 BatchFeature，官方调用 backbone / action head prepare_input；需要沿调用判断，不能仅凭 diff 断言不兼容 | 当前正式服务并非直接用这个 fork；不作为已确认根因 |
| 官方 `gr00t/data/state_action/state_action_processor.py:388`，`unapply_action` | 最初发布到后续官方版有 EEF 相对动作转换调整；本权重三组关节均为 NON_EEF | **本次 B1K 样本未触发差异**；8 帧实际输出一致 |
| 官方 `gr00t/model/modules/dit.py:23` 起 | 后续加入针对 Spark / sm121 的 SDPA 路径选择 | 本次 A100 推理未发现该分支造成差异；不能据此解释症状 |
| 官方 `gr00t/model/gr00t_n1d6/image_augmentations.py:304` 起及 Processor | 后续加入可选 letterbox；该权重未开启此选项，当前 v1 为原路径 | 输入处理和输出实测一致 |
| `src/v1_groot_memoryvla/model/gr00t_n1d6/processing_gr00t_n1d6.py:218`，`canonical_language` | 新增训练示范长指令与仿真短任务名的映射，保存在训练后 Processor 中 | **训练后已处理**；不是当前已确认语言漏传 |
| `src/v1_groot_memoryvla/policy/gr00t_policy.py:308`，`_get_action` | 增加可选 Memory context；memory=false 时仍按 processor → model → decode 的公共流程 | 同原始权重对照时逐值相同；不代表 M1 Memory 更新本身已通过本次全部检查 |
| 实际归档 `gr00t/eval/sim/wrapper/multistep_wrapper.py:275`，`step` | 增加 native truncated 时停止当前 chunk | 最初发布代码有“truncated 后多执行 chunk 剩余步”的问题；当前已修，不解释从开局即乱动 |
| 实际归档 `gr00t/eval/rollout_policy.py:437,537` 起 | profiling、按结束槽位 reset、避开 NEXT_STEP autoreset 丢弃一次动作 | 静态路径每轮使用 next_obs；仍须真实轨迹核实 |
| 实际 OG 五个 dirty 文件 | 加入 profiling 包装、计时和计数；逐段保留原动作 / physics / render 调用 | 已读差异，未发现其中直接引入动作语义变化；不等于整个仿真已排除问题 |

## 4. Issue #574 在当前项目是否存在

[Issue #574](https://github.com/NVIDIA/Isaac-GR00T/issues/574) 是 **G1 PnPAppleToPlate** 的问题报告：用户描述异常转臂、手部不激活、0/11；2026-03-18 有用户报告回退 `4e62473` 后默认模型和仿真恢复。评论中提到归一化统计、公开权重等排查方向，但没有给出适用于 B1K 的已定位根因。

| Issue 线索 | 当前项目核查 |
| --- | --- |
| main 更新后代码 / 权重不匹配 | 原始 N1.6 推理没有导入另一个 N1.7 main；8 个真实示范输入上，发布版与实际评估版输出完全一致。**不支持直接套用同一根因** |
| 忘记 sim wrapper / 用错 embodiment | 实际两个服务参数及本次独立服务均正确；ID 24。**已排除本次服务配置错误** |
| 不正确的统计 / 反归一化 | 三组 checkpoint 的 statistics 文件相同；实际解码路径、往返与适配比较通过。**已排除所测输入上的明显错误** |
| G1 的手部控制问题 | B1K 为 R1Pro smooth scalar gripper，不是 G1 手部控制接口。不能移植结论 |
| 回退发布 commit 是否必然修好 B1K | **证据不足**。本次发布版在固定示范上没有产生不同动作，不建议直接替换当前运行环境赌结果 |

## 5. P1：观测链路

实际代码路径：OG `get_obs` → `behavior_env.preprocess_obs` → `MultiStepWrapper` → `Gr00tSimPolicyWrapper` → `Gr00tPolicy._to_vla_step_data` → Processor → 模型。

### 5.1 相机、颜色和图像处理

`/nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1/gr00t/eval/sim/BEHAVIOR/behavior_env.py:100,250,315`：

| 相机 | 仿真 sensor 名 | Policy 字段 |
| --- | --- | --- |
| head | `robot_r1:zed_link:Camera:0` | `video.observation.images.rgb.head_256_256` |
| left wrist | `robot_r1:left_realsense_link:Camera:0` | `video.observation.images.rgb.left_wrist_256_256` |
| right wrist | `robot_r1:right_realsense_link:Camera:0` | `video.observation.images.rgb.right_wrist_256_256` |

相机 wrapper 实际设置三路为 256×256；head horizontal aperture 设为 40，源码注释说明与采集设置一致。preprocess 仅取 RGBA 前三通道，没有 RGB/BGR 交换。Processor 按 checkpoint 的 modality_keys 排序，不能把 Python 字典插入顺序误认为视觉 token 顺序。独立相机哨兵通过 actual preprocess 后 head/left/right 分别保留自己的标记。

真实示范第 0 帧读取结果如下，完整字段、shape、范围和 SHA256 已打印并保存在 `release_official/report.json`：

| 输入 | shape / dtype | 数值范围 |
| --- | --- | --- |
| head RGB | `[1,1,256,256,3]` / uint8 | 0–254 |
| left wrist RGB | 同上 | 0–124 |
| right wrist RGB | 同上 | 0–128 |
| Processor head / left / right | 各 `[1,3,252,252]` / float32 | `[-1,0.945098]` / `[-1,-0.058824]` / `[-1,-0.003922]` |
| Processor state | `[1,1,128]` / float32 | `[-0.890799,1]`，包括 padding |
| Processor embodiment_id | `[1]` / int64 | 24 |

PNG 保存在每个离线输出目录。这里是**真实示范图像**，不是伪造的在线观测；不能凭这些图证明在线相机没有过期或渲染错误。

### 5.2 状态字段、坐标系和 256 / 258 差异

所有 Policy state 字段的 batch / 时间前缀为 `[1,1]`，dtype float32，全部 finite。第 0 帧范围：

| 字段（均有 `state.` 前缀） | 维数 | min / max |
| --- | --- | --- |
| robot_pos | 3 | 0.004967 / 22.271994 |
| robot_ori_cos | 3 | 0.999358 / 0.999999 |
| robot_ori_sin | 3 | -0.001515 / 0.035829 |
| robot_2d_ori | 1 | 0.035837 |
| robot_2d_ori_cos | 1 | 0.999358 |
| robot_2d_ori_sin | 1 | 0.035829 |
| robot_lin_vel | 3 | -0.018744 / 0.000334 |
| robot_ang_vel | 3 | -0.010095 / -0.000008 |
| arm_left_qpos | 7 | -0.000038 / 0.000164 |
| arm_left_qpos_sin | 7 | -0.000038 / 0.000164 |
| arm_left_qpos_cos | 7 | 1 / 1 |
| eef_left_pos | 3 | 0.121313 / 0.437996 |
| eef_left_quat | 4 | -0.022446 / 0.999748 |
| gripper_left_qpos | 2 | 0.049746 / 0.049747 |
| arm_right_qpos | 7 | -0.000057 / 0.000141 |
| arm_right_qpos_sin | 7 | -0.000057 / 0.000141 |
| arm_right_qpos_cos | 7 | 1 / 1 |
| eef_right_pos | 3 | -0.252004 / 0.437996 |
| eef_right_quat | 4 | -0.022442 / 0.999748 |
| gripper_right_qpos | 2 | 0.049747 / 0.049748 |
| trunk_qpos | 4 | -1.449776 / 1.024939 |

OG 源码语义证据：`robots/robot_base.py:330,339` 按 `_proprio_obs` 给定顺序拼接；robot_pos / Euler sincos / 世界速度来自根刚体；`robots/manipulation_robot.py:642` 的 eef 使用相对机器人根的 pose，四元数保持原样，不改为 Euler 或度数。旋转关节位置为 rad，位置为 m；夹爪状态是两根平移手指的关节位置，**不是输出动作里的一个 ±1 标量**。

录制 parquet 为 256 维，没有 grasp_left / grasp_right。仿真 `behavior_env.py:548` 明确请求包含这两个字段的 `_proprio_obs` 顺序，形成 258 维。所以 `convert_b1k.py:21` 的录制数据切片与 `behavior_env.py:47` 的在线切片不同是必要的。

本次读取整个真实原始 episode `[36383,256]`，验证冗余 sin/cos 与四元数；在所选 8 帧插入两个 grasp 哨兵，执行 actual `preprocess_obs`，与 converter 的 21 个字段逐项比较，全部差值 0。证据：[interface_checks.json](outputs/diagnostics/groot_b1k_20261110/interface_checks.json)。**这验证切片数学与 schema，不冒充真实仿真状态采样。**

### 5.3 语言和第一帧

仿真 Task 21 输入为 `Collecting childrens toys.`。示范的长指令描述两个 dice、两个 teddy bear、两个 board game 放进 bookcase；本次实际打印了全文。原始官方 Processor 没有新增映射；训练后 Processor 保存了长指令 → 仿真短指令的 canonical mapping。没有发现语言字段漏传。

同一第 0 帧仅切换长 / 短指令，动作组最大变化约：base 0.00595、torso 0.02618、left_arm 0.01624、right_arm 0.02454。说明模型对语言存在响应；不能凭这一组差异认定官方短指令有错。

`env_base.py:717` 的 reset 在取观测前有 sim.step 和 3 次 render，静态代码没有直接返回 reset 前缓存帧。但**首帧新鲜度需要在线帧 / 相机绑定证据**，不能仅靠这段源码排除。下节给出本次在线诊断结果。

## 6. P2：动作语义与控制接口

### 6.1 23 维语义

| 索引 | 分组 | Policy 解码后语义 | 控制器 |
| --- | --- | --- | --- |
| 0:3 | base | 本体局部坐标下的 x/y/yaw 速度控制命令 | HolonomicBaseJointController，输入 [-1,1]，x/y 缩放至 ±0.75 m/s，yaw 至 ±1 rad/s，再转换到 canonical frame |
| 3:7 | torso | 4 个躯干关节绝对位置目标 | JointController position，`use_delta_commands=false`，输入输出 limits=None |
| 7:14 | left_arm | left_arm_joint1…7 的绝对位置目标（rad） | JointController，同上 |
| 14:15 | left_gripper | 一个 smooth 开合控制量 [-1,+1] | MultiFingerGripperController，将范围映射到实际手指关节限位并广播至两根手指；正向为打开 |
| 15:22 | right_arm | right_arm_joint1…7 的绝对位置目标（rad） | JointController，同上 |
| 22:23 | right_gripper | 同左手 | 同上 |

**“base 是 ABSOLUTE”是 Processor 的动作表示属性，不是让底盘执行世界坐标位置。它仍由 velocity controller 执行。** 同理，模型内部三组 relative joint action 在 decode 后已经转为绝对位置；环境侧不能再加一次当前 qpos。

证据位置：

- 实际 `behavior_env.py:118,281`：ACTION_MAP 和 preprocess_action；只拼接，不归一化 / 反归一化。
- 实际 `og_teleop_cfg.py:195`：控制器配置；`og_teleop_utils.py:138` 设置 action_normalize=false。
- OG `robots/r1.py:166,220`：base → trunk → arm_left → gripper_left → arm_right → gripper_right；arm_names=[left,right]。
- OG `robots/r1pro.py:42`：左右臂 joint1…7；finger_joint1…2。
- OG `controllers/controller_base.py:225`：输入裁剪和物理范围缩放；`joint_controller.py:164`：绝对目标；`holonomic_base_joint_controller.py:138`：局部速度旋转；`multi_finger_gripper_controller.py:166` 起：smooth 标量处理。

实际 BaseController 的纯数学离线探针结果：`[1,-1,0.5] → [0.75,-0.75,0.5]`；arms 的 `[0.3,-0.7]` 保持原值。没有把这类控制器正常缩放误判成“重复反归一化”。夹爪实际限位必须在运行时测量，未用猜测的限位替代。

### 6.2 模型 → 解码 → Wrapper → 适配器的数值证据

第 0 帧模型原始动作 shape `[1,50,128]`，前 23 维第一个时间点（归一化空间）：

```text
base          [-0.019531, -0.001953, -0.000977]
torso         [-0.053711, 0.007812, 0.310547, 0.021484]
left_arm      [-0.025391, -0.404297, -0.197266, 0.082031, 0.066406, -0.080078, 0.212891]
left_gripper  [1.0]
right_arm     [0.091797, 0.285156, 0.109375, 0.082031, 0.021484, 0.152344, -0.306641]
right_gripper [0.996094]
```

Processor decode 后第一个动作：

```text
base          [-0.000327, -0.001367, -0.000293]
torso         [1.015333, -1.453072, -0.475709, -0.000086]
left_arm      [0.008562, 0.004575, 0.000555, -0.002646, -0.000442, 0.007804, -0.011487]
left_gripper  [1.0]
right_arm     [-0.002405, -0.004663, -0.005925, 0.005156, -0.001108, 0.006691, 0.006530]
right_gripper [0.996094]
```

Wrapper 后数值完全一致，仅字段加 `action.` 前缀；actual preprocess_action 后按上表拼接为 23 维，仍完全一致。三个阶段完整数组已保存在各目录 `boundaries.npz`。`env_first8` 是**调用 actual 环境适配函数的离线输出**；只有在线 step 文件中的 `env_action` 才是实际仿真 step 里传入环境的动作，二者不混用。

动作块索引探针使用真正 MultiStepWrapper：给定序列 0…31，实际执行 `[0,1,2,3,4,5,6,7]`，下一观测哨兵为 8；reset 会重置 deque、reward、done、info。原发布版 native truncated 首步后仍执行 8 步；当前源码只执行首步，说明该已知边界 bug 当前已经处理。该探针不等同于在线时序全部正确。

### 6.3 非 MAE 的示范动作检查

没有重算 MAE，也没有把本次当作成功率评估。

| 帧 / 阶段 | 观察到的输出行为 |
| --- | --- |
| 0 / 初始 | torso 接近示范的 `[1.025,-1.45,-0.47,0]`，两臂小幅动作，两夹爪打开 |
| 192 / 移动 | base 有 yaw 命令，未出现把底盘速度当坐标；两臂第一动作块最大相邻跳变约 0.013–0.015 rad |
| 800 / 右夹爪闭合前 | 示范未来索引 3 开始闭合；模型未来索引 14 才转为负值。**模型闭合时机偏晚**，三个代码版本一致 |
| 808 / 右夹爪闭合后 | 右夹爪预测约 -1，左夹爪约 +1；左右分工未颠倒 |
| 1448 / 左夹爪闭合前 | 示范未来索引 1 开始闭合，模型未来索引 10 开始闭合；前三个版本一致 |
| 1456 / 左夹爪闭合后 | 两夹爪均约 -1；右臂姿态对应示范所在范围 |
| 8192 / 后续片段 | 右夹爪约 +1、左夹爪约 -1；没有所有输入都固定输出同一开合动作 |

以上显示模型能产生开合与姿态相关输出，也显示切换时机存在偏差；不能由单个示范、单组种子推断闭环抓取成功或失败的主因。前 8 步关节动作连续，不代表 32 步内的夹爪 ±2 开合跳变有错；二值开合本来允许切换。

## 7. P3：一次短时真实仿真诊断

**已完成 400 个真实环境动作步、50 次 Policy 请求**，并已关闭专用诊断服务 / 仿真。见 [completion.json](outputs/diagnostics/groot_b1k_20261110/live_sim_400/completion.json)、[逐边界分析](outputs/diagnostics/groot_b1k_20261110/live_sim_400/analysis.json) 和 [在线请求跨版本重放](outputs/diagnostics/groot_b1k_20261110/live_replay_comparison.json)。

诊断约束：Task 21、instance 0、环境 seed 42、一个环境、n_action_steps=8、最多 50 次 policy 请求；独立原始模型服务端口 5568，A100 空闲 GPU1；仿真使用空闲 GPU1。每个请求固定种子 `10000 + request_index` 并记录，便于离线精确重放。未训练、未运行完整 episode、未执行 10 次评估。

脚本：[groot_b1k_live_probe.py](scripts/groot_b1k_live_probe.py)。只在新建诊断进程中包装函数，不修改现有 OG / GR00T 源文件，不与 5560/5561 交换请求。保存内容：

- server：每个请求的原始输入、模型 raw action、decode、Wrapper 输出和种子。
- sim：每次请求输入 / 输出、每个动作步环境收到的 23 维命令、前后关节位置、原始 proprio、各控制器 `_goal`。
- head / left wrist / right wrist 同步视频（从实际返回 Policy 的相机字段采样）；相机 RenderProduct / viewport 所有权和时间步配置。

启动经过一次 **0 动作步** 的失败：诊断直接启动时继承 shell 里其他项目的 `OMNIGIBSON_DATA_PATH`，场景目录不存在；正式评估入口会覆盖它。错误与回溯保存在 `live_sim/failure.json`。本次明确指定 `assets/simulation` 后继续原定短时测试。该诊断启动错误不计为模型 / 既有 Baseline 的根因。

加载时出现若干 scene/object expected USD hash 与当前文件 hash 不同的警告。源码 `usd_object.py:107,172` 对加密资产检查的是 **encrypted 文件** MD5，而不是解密后几何、碰撞、关节参数的语义指纹。所以这里只能确认文件 hash 不同，不能直接断言碰撞模型不兼容；尤其不能靠 VERSION 文本就排除或确诊资产问题。

### 7.1 真实输入和相机

首个真实请求：三路相机 shape 均为 `[1,1,256,256,3]`、uint8；head 范围 2–209、left 0–123、right 0–143；语言确实为 `Collecting childrens toys.`。实际原始 proprio 为 258 维；Policy 请求包含额外的 joint/qvel/grasp 等字段，Wrapper 按 checkpoint 只选需要的状态项。全部字段名称、shape、范围和 finite 检查已输出到 `analysis.json/first_real_input_fields`。

对第一个真实请求的 Processor 重放结果：state `[1,1,128]`，范围 `[-0.890625,1]`；input_ids `[1,285]`；三路 pixel_values 各 `[1,3,252,252]`，范围分别为 `[-0.984375,0.640625]`、`[-1,-0.098145]`、`[-1,0.058838]`。这是送入模型的 BF16 值转 float32 保存后的范围，与前面示范 Processor 的 float32 检查阶段有精度区别。

真实 RenderProduct 绑定：left=`Replicator_01`、right=`Replicator_02`、head=`Replicator_03`，各自绑定对应机器人 camera prim；分辨率均为 256×256，head aperture 为 40。每一路在 50 次请求中均有 **50 个不同图像 hash**，未见视角互换、常驻全黑或冻结帧。已人工查看请求 0、24、49 的图像；末尾双腕看到同一地面 teddy bear 的不同视角。

这排除了本窗口的持续冻结 / 明显绑错相机；**没有记录渲染 frame ID / exposure timestamp，不能宣称已经证明首帧绝对零延迟**。

Gym 的“obs 不在 observation_space”警告也作了核查：官方 `behavior_env.py:376` 声明 `gym.spaces.Text(max_length=512)`，默认只接受字母数字，而真实任务指令包含空格和句号。实测 `contains('Collecting childrens toys.')=False`。这是 observation_space 声明的问题；本次日志只是警告，实际服务器收到的语言字符串正确，没有因此被截断或替换。

### 7.2 真实输出与最终控制

整个 400 步窗口的边界比较：

| 比较 | 最大绝对差 |
| --- | --- |
| 仿真客户端输入 → 服务器收到的输入 | 0 |
| 服务器 Wrapper 输出 → 客户端收到的输出 | 0 |
| Processor decode 转为 float32 → Wrapper | 0 |
| Wrapper 对应 request 的索引 0–7 → 每步实际 action 字典 | 0 |
| action 字典按语义拼接 → 环境收到的 23 维 env_action | 0 |
| 上个动作块结束后的 raw proprio → 下一次请求所有 state 字段 | 0 |
| 双臂 / torso 动作 → 对应控制器 target | 0 |
| 上一步 joint_after → 下一步 joint_before | 0 |
| 底盘命令缩放后的平面速度模长 → controller target 模长 | `7.30e-7` |
| 底盘 yaw 命令 → controller yaw target | 0 |
| 夹爪 `0.025*(clip(command,-1,1)+1)` → 实际 controller target | `3.73e-9` |

运行时控制器实际给出的夹爪关节限位为 **[0,0.05] m**；左右臂 dof_idx 分别为 `[10,12,14,16,18,20,22]` 和 `[11,13,15,17,19,21,23]`，并非直接从全关节数组连续取 7 维。实际 controller order 与 ACTION_MAP 一致。physics_dt=`1/120 s`，action / rendering dt=`1/30 s`，与 30 Hz 示范一致。

第一步实际边界举例（所有 23 维完整值在 `analysis.json/first_action_boundaries`）：

| 分组 | 模型原始值 | decode 后 / Wrapper 后 / 环境收到 | Controller 目标 |
| --- | --- | --- | --- |
| base | `[-0.019531,-0.003906,-0.141602]` | `[-0.000327,-0.002734,-0.042480]` | `[0.0000967,-0.0020631,-0.042480]`，包含速度缩放与坐标旋转 |
| torso | `[-0.0390625,0.0102539,0.3066406,0.0253906]` | `[1.024539,-1.450618,-0.479041,0.000111]` | 与解码值一致 |
| left_gripper | `[1]` | `[1]` | `[0.05] m` |
| right_gripper | `[0.996094]` | `[0.996094]` | `[0.0499023] m` |

关节对目标的跟踪情况（每个环境步结束后采样，属于控制跟踪误差，不是模型预测 MAE）：

| 分组 | 最大绝对误差 rad | 95% 分位 rad |
| --- | --- | --- |
| torso | 0.010032 | 0.000383 |
| left_arm | 0.001969 | 0.000507 |
| right_arm | 0.002192 | 0.000468 |

全部执行动作 finite，无 PhysX crash。最大单步关节变化出现在 **step 230 / request 28 / chunk index 6**，为 `0.056521 rad`（约 3.24°）；该处不是复现到的异常旋转。动作自身的相邻步最大跳变：left_arm `0.045296 rad`、right_arm `0.056781 rad`、torso `0.016358 rad`。没有发现模型发出平稳目标而机器人突然反向大幅转臂的证据。

这 400 步的两夹爪命令都为正（left 最小 0.980469、right 最小 0.976562），**模型尚未发出闭合指令**。所以本窗口的张开状态由模型输出决定，不能解释为“给了闭合命令但夹爪控制器失效”。同时，窗口没有覆盖成功抓取 / 放置，不把尚未抓住当作失败率样本；没有进行额外示范动作回放。

### 7.3 保存的在线请求跨版本重放

使用保存的 **50 个实际在线请求** 和对应 `10000+i` 噪声种子，在最初发布 `4e62473` 与实际部署的 v1 源码中分别加载同一官方权重重放。对照对象为本次由官方 `9b37aa1` 归档源码运行的实际服务：

- 50/50 请求 raw action 最大差 **0**。
- 50/50 请求各组解码 / Wrapper 动作最大差 **0**。
- 发布版与 v1 的模型实际输入 state、图像 tensor、token、embodiment ID 比较，最大差 **0**。

这比只在示范输入上比较更直接：**这条真实在线轨迹也不支持“回退发布版就会改变动作、修好本次现象”的解释。** 结论限定于这组权重、硬件、输入和配置，不外推到其他 G1 模型或之后的抓取阶段。

视频：[head / left / right 三视角，400 帧、30 FPS、13.33 秒仿真时间](outputs/diagnostics/groot_b1k_20261110/live_sim_400/head_left_right.mp4)。有冷启动 shader / 场景编译等待，实际只执行上述 400 步，没有延长任务或重复评估。

## 8. 问题状态汇总

| 检查项 | 结论 | 直接证据 / 限制 |
| --- | --- | --- |
| 声称固定官方源码、实际默认导入 fork | **确认异常** | serve.py 默认路径 + git origin / HEAD；不代表当前正式评估由该入口启动 |
| 将当前 5560 Baseline 解释为官方原始权重 | **确认身份不一致** | 进程参数为 baseline/step-010000；是否曾被误标需结合具体结果目录 |
| 原始权重文件损坏 / 下载不完整 | **已排除所检分片问题** | 完整 SHA256 与 HF 本地 LFS metadata 对应，模型加载成功 |
| 用错 embodiment | **已排除本次已核服务** | tag 与 ID 24 |
| sim wrapper 没启用 | **已排除本次已核服务** | 进程参数 + 离线实际调用 |
| Issue #574 相同版本回归 | **不支持；不能全局排除** | 8 个真实输入三版本各边界相同；不等于所有输入 / GPU / 环境条件均相同 |
| 示例 256 维与在线 258 维的必然切片错位 | **已排除这种推断** | 两 grasp 字段解释 + actual 适配器重构逐项比较 |
| RGB/BGR 交换、dict 插入顺序导致相机互换 | **静态及哨兵检查已排除** | actual preprocess、checkpoint modality 顺序；在线绑定仍看真实证据 |
| 语言缺失 / 训练后长短指令未映射 | **已排除配置层面问题** | Processor 保存 mapping；语言敏感性探针 |
| 明显重复反归一化 / absolute-relative 错误 | **已排除所测输入** | decode 源码、往返、三版本实际输出和 adapter 比较 |
| 左右臂 / 关节顺序 / 夹爪标量映射错位 | **静态与示范检查已排除** | ACTION_MAP、R1Pro joint names、controller order |
| 将 base 控制器正常缩放当成 double denorm | **已排除此误判** | actual BaseController `[1,-1,.5]→[.75,-.75,.5]` |
| n_action_steps 没执行前 8 步 | **离线执行索引检查已排除** | 实际 wrapper 执行 0…7，输出最新观测 |
| native truncated 后多走步 | **原发布代码确认存在；当前已修** | 相同 mock 环境：发布版 8 步、当前 1 步 |
| 在线视口绑定错误 / 持续冻结帧 | **已排除本次 400 步窗口** | 真实 RenderProduct 绑定正确；三相机各 50 个不同请求图像 |
| 第一帧严格新鲜度 | **证据不足** | 未记录 render frame ID / exposure timestamp，不能证明绝对零延迟 |
| 下游动作缓存 / 丢步 / 旧状态重新输入 | **已排除本次 400 步窗口** | 所有动作和最新状态边界逐值匹配，索引 0–7 正确 |
| 手臂 Controller 不跟随正确目标 | **已排除本次 400 步窗口的明显异常** | 两臂实际目标跟踪最大差约 0.002 rad，无突发反转 |
| 夹爪收到闭合命令却不闭合 / 接触及 assisted grasp | **证据不足** | 窗口内模型没有发负的闭合命令；未覆盖接触抓取阶段 |
| 模型闭合命令晚于示范 | **确认样本上的输出差异** | 帧 800 和 1448；不是已经定位闭环根因 |
| language observation_space 不接受实际指令 | **确认声明 Bug，不是已定位的乱动原因** | 默认 Text charset 拒绝空格 / 句号；实际请求字符串仍完整送到服务器 |
| 原始 Baseline 正常而 M1 异常 | **本次证据不足** | 本次着重原始模型；没有额外启动 M1 仿真 |

## 9. 最小修复方案与未应用补丁

目前没有足够证据提供“修好乱动”的生产补丁。不会把回退版本、改变夹爪符号、重复加 qpos、修改归一化或重训作为未经验证的修复。

针对**已经确认的 Baseline 入口来源错误**，提供 [baseline_source_guard.patch](outputs/diagnostics/groot_b1k_20261110/baseline_source_guard.patch)，尚未应用到当前源码。它只允许入口使用指定的干净官方 commit，并支持显式 `GROOT_REFERENCE_ROOT` 指向独立 worktree；当前 fork 会被清楚地拒绝。该补丁修复来源声明与执行不一致，**不声称提高闭环成功率**。

验证已执行：`git apply --check` 通过；在内存应用补丁并用桩替代真正服务启动，验证当前 fork 被拒绝、独立官方 `9b37aa1` worktree 被接受。详见 [guard_validation.json](outputs/diagnostics/groot_b1k_20261110/guard_validation.json)。没有真正应用补丁、切换服务或改模型。

另提供 [observation_text_space.patch](outputs/diagnostics/groot_b1k_20261110/observation_text_space.patch)，对独立 GR00T 仿真源码将 language Text charset 设为可打印字符，修正实际自然语言被 observation_space 判为非法的问题。已验证补丁可应用、修正后的 Text space 接受本次真实指令；**未应用到正在使用的外部环境，不声称改变动作结果**。

## 10. 下一步与是否保留 Baseline

### 10.1 根因候选的证据排序

**没有证据支持现在锁定一个工程根因。** 本次真实窗口已经证明：所观察到的早期运动是模型本身给出的命令，适配器和 Controller 忠实传递、执行了它；不存在“模型动作合理但被这条链路扭成大幅乱转”的证据。最初发布版在相同在线输入上也会给出完全一样的命令。

还需要区分两类尚未解决的问题：

1. **更晚阶段的策略决策 / 观测分布 / 闭合时机。** 示范存在提前量不足的直接样本，但本次在线窗口没有覆盖抓取失败的首次发生，不能升级为历史失败根因。后续优先级高于盲目回退版本。
2. **后来才触发的接触、夹爪执行或 online observation 问题。** 本窗口支持关节位置执行正确，不能排除抓取接触时的异常，也没有帧级时间戳证明相机绝对零延迟。

来源错误与 checkpoint 身份混淆是已确认的实验管理问题，应该纠正；它们与历史原始 Baseline 失败之间仍缺直接因果证据。资产 encrypted hash 警告也尚不足以证明物理资产语义不兼容。

### 10.2 下一步最小代价验证

**先从已有失败视频标出第一次出现“反向大转臂 / 已到物体却始终不闭合 / 已闭合仍不抓住”的具体时间步。** 本次 step 0–399 没有复现该症状，继续在同一开局重复 400 步价值有限。

随后在下一次已有评估到达该时刻时，只保存 **一次 Policy 请求、该动作块、原始 proprio、controller goal、前后关节状态和对应图像**，复用本次探针离线重放，不需要重训或重新跑 10 条完整 episode：

- 如果命令从 Wrapper 到环境 / goal 已变形，直接修那一处映射。
- 如果负夹爪命令已变为接近 0 m 的目标、手指仍不动，检查实际接触 / controller / assisted grasp。
- 如果链路一致、Controller 跟随，而模型在目标前始终不给闭合命令，才把焦点放到策略输入分布、语言、时间提前量或权重行为；仍不直接概括为模型没有能力。

50 次真实请求的版本对照**已经完成**，无需再为了相同证据启动仿真。也不建议把 n_action_steps、夹爪符号或归一化改成其他值再盲试。

建议继续保留 GR00T，标记为**通过开局 400 步链路检查、待抓取阶段验收的官方多任务 Baseline**，与训练后 Baseline 分开。真实输入的跨版本一致性、Controller 准确跟随和示范中的有效开合输出，不支持现在直接淘汰模型；但也不足以宣布完整工程链路全对或 M1 结果可信。可以继续小规模诊断实验，扩大训练和重复评估应等抓取失败时刻被解释后再进行。

官方公开表的 Task 21 / 22 / 26 是 **Task Progress**，不是成功率或 Q-score；它不能证明当前推理正确。官方完整来源见 [BEHAVIOR 说明](https://github.com/NVIDIA/Isaac-GR00T/blob/9b37aa1ce69c73c6d165233fa88128283bba4508/examples/BEHAVIOR/README.md)。

## 11. 可复用产物与复现

- [离线边界探针](scripts/groot_b1k_inference_probe.py)：`versions`、`inventory`、`offline`、`interfaces`、`compare`、`replay`。均无 optimizer / 训练调用。
- [offline_comparison.json](outputs/diagnostics/groot_b1k_20261110/offline_comparison.json)：8 帧跨版本逐字段差异与输出范围。
- `release_official`、`archive_official`、`v1_official`：0 / 96 / 192。
- `release_motion`、`archive_motion`、`v1_motion`：800 / 808 / 1448 / 1456 / 8192。
- 每个目录的 `report.json`、`boundaries.npz`、各视角 PNG；HF dynamic-module cache 独立放在诊断目录，没有覆盖已有服务 cache。
- A100 原始产物根：`/nvme3/data_wyh/outputs/memory_vla/groot_b1k_diagnosis_20261110`。
- `live_sim_400`：50 组真实请求输入 / 输出、400 个 `step*.npz`、视频、runtime / completion / analysis JSON。
- `live_server_400`：50 组模型 raw / decoded / wrapped 边界和请求种子。
- `live_replay_release` / `live_replay_v1`：对上述在线请求的发布版 / 当前部署版重放，含实际模型输入 tensor。

示例复现命令（A100，新建独立诊断进程，不连接现有服务）：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 \
  /nvme3/projects/memory_vla/envs/n1d6-p0/bin/python \
  /nvme3/data_wyh/outputs/memory_vla/groot_b1k_inference_probe_20261110.py offline \
  --source /nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1 \
  --checkpoint /nvme3/huggingface/train_model/GR00T/GR00T-N1.6-BEHAVIOR1k \
  --dataset /nvme3/data_wyh/datasets/b1k2025_groot_task21_v1_200 \
  --episode 210020 --frames 0 96 192 800 808 1448 1456 8192 \
  --env-adapter /nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1/gr00t/eval/sim/BEHAVIOR/behavior_env.py \
  --output /nvme3/data_wyh/outputs/memory_vla/groot_b1k_diagnosis_reproduction
```

切换到本次独立发布版源码目录即可复现发布 commit 对照；使用 `--package v1_groot_memoryvla --source ...resume_r4/src` 可复现当前部署源码对照。不要为复现而重置现有目录。
