# V2 架构与数据接口调查

日期：2026-11-11，已按当日最新 V1 再次核对。V2 组合导入当前 V1；本次只重写 V1 model.md，不改 V1 可执行代码、不接管训练或评估进程。当前网络详见 [model.md](model.md)。

## 1. 原模型与 V1 的真实接入点

以下路径均相对 `src/v1_groot_memoryvla/`。

| 实现位置 | 实际行为 / 形状 |
|---|---|
| `model/modules/eagle_backbone.py:EagleBackbone.forward` | 三视角及任务文字经过 Eagle；上下文输出 `[B,S,2048]`。V1 额外取投影后、LLM 前视觉特征 `[B,3,81,2048]`，不是新增视觉网络。 |
| `model/gr00t_n1d6/gr00t_n1d6.py:_backbone_with_memory` | Eagle → V1 Memory → 原 ActionHead；训练与推理共享该位置。 |
| 同文件 `Gr00tN1d6ActionHead.forward/get_action_with_features` | VLM 特征先经过非原地 `vlln`；原状态 `[B,1,128]` 通过 embodiment MLP 编码为 1536 维；动作和时间步也各自编码。原 32 层 AlternateVLDiT 交替处理视觉、文字及动作。输出 `[B,50,128]`；B1K 有效动作 23 维。推理为 4 次 flow Euler 更新。 |
| `model/memory/adapter.py:MemoryAdapter.forward` | BottleneckSE 得到 `[B,243,256]`，最后有效 user EOS 为 `[B,1,2048]`。direct 输出投影感知+融合 EOS `[B,244,2048]` 及重建 masks；旧 residual 才保留 H 并加残差。 |
| `model/memory/bank.py:MemoryBank` | 两个独立 bank；新启动容量默认64，可配置。2层检索、Key时间编码、门控；先读过去再 detach 写融合特征，超容量合并最相似相邻记录。不跨历史反传。 |
| `experiment/task21_ddp.py:TrainableStep` | memory/action_head 同处 DDP forward；连续Episode每窗最多32决策，stride8。支持尾窗、固定槽/rank所有权、有效动作元素加权。增强默认开启，每次窗口重编码冻结Eagle；不能长期复用增强缓存。 |

原 action_head 与 V1 Memory 参数都可以被训练；历史 bank 不是 optimizer 参数。V1 state 编码、action 解码、归一化与控制器链路均保留。当前 Body 分支接在同一个 pre-vlln 位置，不替换状态或动作 Token。

## 2. state / action / Controller 映射

显式名称、OG 索引、数据字段、控制顺序见 `configs/r1pro_mapping.json`。实际仿真源码为 `../GR00T/BEHAVIOR-1K-task-progress/OmniGibson/omnigibson/robots/r1pro.py`，其父类为 `r1.py` 与 `holonomic_base_robot.py`。

| 控制组 | 环境动作索引 | OG q 索引 | 意义 |
|---|---|---|---|
| base | 0:3 | 虚拟底盘 0:6，不能当车轮 q | 机体系平面速度与偏航命令，经 wrapper 缩放/旋转；不是绝对关节角 |
| trunk | 3:7 | 6:10 | torso_joint1..4，绝对位置 rad |
| arm_left | 7:14 | 10,12,14,16,18,20,22 | left_arm_joint1..7，绝对位置 rad |
| gripper_left | 14 | 24,25 | 单命令控制双指，[-1,+1]；观测双指位置单位 m |
| arm_right | 15:22 | 11,13,15,17,19,21,23 | right_arm_joint1..7，绝对位置 rad |
| gripper_right | 22 | 26,27 | 同左夹爪 |

GR00T 学习空间使用 checkpoint processor 的相对动作/分位归一化；`get_action` 先返回模型空间动作，随后必须通过原 processor 解码，再经过原 `sim-policy-wrapper` 执行。V2 的历史命令输入使用**解码且实际下发的 23 维命令**，不是归一化预测值，也不是整个未执行的动作块。

原始 B1K 示范 `observation.state` 为 256 维，包含完整 q、qvel、底盘、两臂、夹爪、EEF 等；V1 转换为 82 维，删去 qvel，保留 base velocity、q、sin/cos、EEF。原始 qvel 在 `[84:112]`，与 28 维 OG 关节顺序一致。V2 优先核验配对原始数据读取 qvel；只有转换数据时，在原始 30Hz 采样上做向后差分，首帧速度 mask=0。禁止用下一帧差分。

## 3. 时间语义与可用字段

数据含 `timestamp`、`episode_index`、逐帧 `action[23]`，30Hz。已检查原始/转换后 task0 episode10 的行数、时间、动作与 q 对应关系。

生成代码证据：OmniGibson `envs/data_wrapper.py:flush_partial_traj` 对 observation 写入槽位 `[count-data_len+1:count+1]`，动作写入 `[count-data_len:count]`；回放导出脚本保留行索引。因此 `obs[t]` 对应上一次 `action[t-1]` 的执行后状态，`action[t]` 是标签，不能进入 `Body[t]`。首帧没有上一次动作。在线调用之间有 8 次环境执行：需要传入**最后真正执行**的动作及其时间，不能直接取上次预测块的末项。

新增 Body 输入只使用测量关节、已有本体速度、URDF FK、历史已下发命令。没有额外使用 task_info、接触/抓持真值、物体真值、世界位置。机器人 orientation 只用于把已测世界速度旋转到 base_link；各时刻 FK 都在该时刻自身 base_link 中，不把旧局部坐标当当前局部坐标。当前不提供世界轨迹对齐或里程计重建。

## 4. URDF / USD 差异

结构文件：`assets/simulation/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf`。35 joints、36 links；USD 添加了 6 个虚拟底盘自由度。URDF 的 6 个 steering/wheel 节点与这些虚拟自由度没有对应关系，V2 保留节点但测量 mask=0。不能把两组按数组顺序相连。

USD 还添加两个 EEF 固定标记：gripper_link → eef_link，偏移 `(0,0,-0.06)m`，旋转 wxyz `(0,0,1,0)`。补充固定变换有明确 USD 来源，写入 mapping；总计 38 个 Body 节点（含根）。网络按图长度运行，没有固定 38 的权重。FK 与示范中仿真记录的相对 EEF pose 对比，仅将记录真值用于测试。多机器人 padding 与缺测 mask 含义分开。

连杆静态属性包含质量、惯性、首个 visual 几何尺寸或 mesh scale，不解析 mesh 顶点。mimic/非 fixed/revolute/continuous/prismatic 类型当前明确不支持。

## 5. V2 修改范围

- `kinematics.py` / `data.py`：解析结构、显式映射、因果数据读取、base_link FK。
- `body.py`：静态 27 维属性与动态 16 维属性，当前 base velocity 6 维；拓扑限制的注意力输出 `[B,N,128]`，池化结果用于当前残差与检索。之前的已执行动作只进入 V2b/c 历史写入，不进入 V2a 或当前 Query；按 URDF 属性加权汇总为 27 维，加底盘命令及 availability 共 31 维，再由历史 MLP 编码。
- `memory.py` / `adapter.py`：保留 V1 两个 bank；新增具身 bank 默认容量64，与原V1容量独立。历史输入 detach，读取时重编码。V2b/c 历史写入相同；direct 残差定位新条件唯一认知Token，residual 定位原EOS；短组使用 group_lengths，empty不读写全部历史。
- `model.py`：组合完整 V1，训练/推理共用fuse；原版checkpoint默认新建direct/64，训练过的V1严格继承conditioning/length。已有V1不允许丢失已训练参数或隐式切换结构；完整保存覆盖新初始化记忆。
- `pilot.py`：尾部loss/backward按有效action_mask元素加权；固定eval特征只用于拟合/干预，随机短训默认每步重新增强和编码。正式V1 DDP worker尚未自动用于V2，不能声称已经验证V2多卡训练。

V2a 没有新增历史；V2b 的新增检索仅依当前视觉/任务 EOS，V2c 才加入当前 Body。两者参数量完全匹配。V1 本身的历史检索保持原实现，新增具身融合 bank 是独立 residual，其容量需要与原 V1 两 bank 分别报告。
