# V2 实现与验证报告

日期：2026-11-11，包含最新 V1 同步。V2a/b/c 均已实现；此前完整权重实验使用旧 residual/16 V1。本次增加 direct 接口兼容及训练协议同步，以下先给本次变更，再保留此前实验记录。尚未进行任务成功率评估。

## 本次：同步最新 V1

依据当前代码重写 [V1 model.md](../v1_groot_memoryvla/model.md)，增加 [V2 model.md](model.md)，更新 README、架构调查和五种配置。没有修改 V1 可执行代码、正在运行的服务或既有权重，没有启动新的训练/仿真任务。

| 文件 | 已同步内容与直接证据 |
| --- | --- |
| `adapter.py` / `model.py:fuse` | 修复 direct 时仍用原 H EOS 下标注入的问题。回归先复现 `index 244 is out of bounds ... size 244`，修复后按条件 mask 定位唯一认知 Token；旧 residual 仍用原 EOS。 |
| `adapter.py` | 旧固定长度取模边界不能处理 `[1,3]` 的短组，回归复现 `Group crosses Episode`。现按累计 group_lengths reset；empty 同时禁用新具身 bank 读写，保留当前 Body 路径。 |
| `model.py:resolve_memory_config` | 无记忆原始 GR00T 新建默认 direct/64；已训练 V1 继承已保存接口/容量。缺字段旧权重为 residual；显式冲突报错。 |
| `model.py:forward` | 修复Baseline/V1共享带Body字段的batch时，在空extension上取参数引发的StopIteration；无Body配置跳过该字段的设备迁移。两种配置均完成复现与动作前向回归。 |
| `config.py` / 五种 JSON | 新具身 bank 默认64；原 V1 conditioning/length 默认为自动继承。旧 V2 保存配置的16容量仍可恢复，不改变已保存网络。 |
| `pilot.py:loss_sequence` | 从观测数平均改为有效 action_mask 元素加权；不等尾部监督的最小回归由错误2.5恢复到期望1.9047619，反向梯度也一致。 |
| `pilot.py:augmented_training_sequence` | 固定拟合/动作干预仍用eval特征；普通短训默认每步重新增强、重编码冻结Eagle并恢复eval。测试确认逐步调用/种子变化与异常后的模式恢复。 |
| `model.py:save_extension` / `pilot.py` | 新初始化的 V1 记忆即使冻结也必须完整保存，避免只保存 extension 后重载随机重建该部分。训练过且完整恢复的冻结 V1 仍支持轻量保存。 |
| `diagnose.py` | 支持完整和轻量 checkpoint；计时按真实窗口长度，避免短尾窗被固定四帧截断或越界。 |

本次最终联合验证 **109 passed**，耗时53.12秒：包括V1全部本地测试、V2的33项测试及根目录网页启动测试。两种conditioning均覆盖五配置前向/反向、Query与动作敏感性、V1动作一致、短组/空历史、有效监督加权。原始GR00T、缺字段旧residual、direct三类HF初始化与完整保存/重载均测试，已有记忆参数逐位保留，恢复后固定噪声动作逐位一致。HF测试使用真实记忆/动作头的小宽度配置和离线Eagle替身；不是完整3B模型训练。已有V1两进程Gloo等回归也包含在上述范围。pilot命令行帮助检查通过，新增增强开关可用。

本次没有重跑完整尺寸 V2 的 direct/64 训练，也没有 closed-loop 评估；下面的loss、耗时与显存是**同步前 residual/16、固定eval图像缓存**实验，不能当作新配置的结果。V2目前的训练入口仍为有界单设备pilot；正式V1 DDP worker尚未接入V2，因此这里只继承其网络/数据规则，未宣称V2多卡训练已验证。后续同起点/同数据对照应显式使用新的direct配置、增强设置和容量。

验证命令（仓库根目录）：

```bash
PYTHONPATH=src OMP_NUM_THREADS=2 NO_ALBUMENTATIONS_UPDATE=1 \
  src/v1_groot_memoryvla/.venv/bin/python -m pytest \
  src/v1_groot_memoryvla/tests src/v2_groot_memoryvla/tests \
  tests/test_groot_web_launch.py -q
```

## 此前：residual/16 完整权重实验记录

## 1. 修改范围与架构选择

此前网络、数据、训练诊断及推理扩展在 `src/v2_groot_memoryvla/`，另向根 `pyproject.toml` 增加 V2 包与资源声明。没有修改 V1 可执行代码、原权重或服务，也没有启动长程训练/评估。当前数据调查见 [v2_architecture_audit.md](v2_architecture_audit.md)，命令见 [README.md](README.md)。

复用 `src/v1_groot_memoryvla/model/gr00t_n1d6/gr00t_n1d6.py` 的完整 Eagle、V1 Memory 和 ActionHead。根据实际接口采用组合式 `V2Model`，没有重写 DiT：

| 文件 / 位置 | 实现 |
|---|---|
| `kinematics.py:RobotGraph` | URDF 图、属性、拓扑掩码、可微 FK，根坐标系 base_link |
| `configs/r1pro_mapping.json` | URDF 名称 ↔ OG q ↔ 82 维 state ↔ 23 维动作；额外 EEF 固定连接来源 |
| `data.py:BodyMapper` | 当前状态、过去动作、缺测 mask、速度读取/向后差分；原始/转换数据交叉校验 |
| `dataset.py:V2HistoryDataset/V2HistoryCollator` | 复用 V1 视觉、文字、动作标签，按同一 frame_indices 添加 Body |
| `body.py:BodyEncoder.forward` | 2 层拓扑注意力；动态节点数、跨机器人 padding，显式区分 unknown 与 padding |
| `memory.py:EmbodiedMemory.read_write` | 严格过去读取、detach 写入、容量约束、相邻相似合并；读取时训练历史编码器 |
| `adapter.py:V2Adapter.forward` | 当前 Body 残差；V2b/c 历史融合与 Query；post-Eagle / pre-vlln 的 EOS 残差 |
| `model.py:V2Model.fuse/forward_cached/forward` | 缓存训练和 RGB 推理共用融合路径，原 action_head 产生动作和 flow loss |
| `policy.py:V2Policy/V2SimPolicyWrapper` | 原 processor、语言/相机 schema、动作解码与 wrapper；补充 qvel、执行反馈与采集时间 |
| `pilot.py` / `diagnose.py` / `tests/test_v2.py` | 实际训练、梯度/动作影响、独立进程恢复、FK/精度/因果性回归 |

与初始任务书的具体适配：保留 V1 两个 bank，增加一个轻量具身融合 bank；不替换原 V1 视觉检索。Body 按节点编码后池化，而不是分成固定数量的左右臂专家。历史动作通过 URDF 属性加权汇总，再进入可学习历史投影；当前 Body/Query 不直接接收此前动作，以维持 V2a 的当前构型对照。语言使用现有 VLM 的最后有效 user EOS，其本身已有视觉语言上下文，没有另起语言网络。

## 2. 张量与因果约定

| 张量 | 此前完整权重实验尺寸 |
|---|---|
| Eagle 上下文 / 三视角特征 | `[4,285,2048]` / `[4,3,81,2048]` |
| 原状态 / 原动作训练目标 | `[4,1,128]` / `[4,50,128]`，仅 B1K 有效维度参与 loss |
| Body q、qvel、上次命令及各自 mask | `[B,N]`，R1Pro 当前 N=38，其他图可改变 N |
| 静态 / 动态节点属性 | `[N,27]` / `[B,N,16]` |
| base 记录 / Body 使用部分 | `[B,10]`：局部速度6 + 此前底盘命令3 + 有效标记1；当前 Body 仅用前6 |
| Body Tokens / pooled | `[B,N,128]` / `[B,128]` |
| 历史命令摘要 | `[B,31]`：27维 URDF 属性加权命令 + 底盘3 + 有效标记1 |
| 新历史输入 / Key、Value、Query | 每条 4255 维 detached 特征 / 各256维 |
| 原动作专家输出 / 解码后有效动作 | `[1,50,128]` / 六组 `[1,32,{3,4,7,1,7,1}]` |

`obs[t]` 只使用 `action[t-1]`，首帧无此前动作；`action[t:t+31]` 只作标签。速度缺失使用 `q[t]-q[t-1]`，不用下一帧。历史姿态都是各时刻自身 base_link 中的 FK，世界位置、物体状态、接触/抓持真值不进入新增分支。训练 bank 的历史输入 detach，读取时重新通过 `history_encoder`，历史投影层可训练，且不跨历史计算图反传。

## 3. 实际权重与测试数据

- 原始 Baseline：`checkpoints/groot_n1d6_behavior1k`。
- V1 初始化：远端只读 `V_1_mem_task21/memory_v1/checkpoints/step-003000` 的本机独立副本 `checkpoints/v2_init_v1_task21_step003000`。所有已训练键严格恢复，missing/unexpected/mismatched 均为空。复制没有修改源 checkpoint。
- 原始与 V1 冻结 Eagle 参数 hash 相同：`c32cfca95dfc98f49a4166a205f03368f5c6b39b84da0d76142a7b4eb92544ec`。
- 神经网络实际测试：本机 GPU1，原 32 层 Action Expert、4 次推理更新，BF16 autocast；原模块参数保持原加载精度。数据为现有 Task21 episode212070，帧 `[0,8,16,24]`。
- 运动学交叉测试：Task0 episode10 配对原始 256 维状态和 V1 82 维状态；9 帧、左右 EEF 共18个位姿。原始状态只用于 q/qvel 核对及独立测试，不额外给网络仿真特权输入。

## 4. 测试结果

此前 `tests/test_v2.py` **15 passed**，包含五配置真实类缩小宽度的 forward/backward、23有效动作维度、映射、raw qvel、FK、padding、多关节类型、过去命令对齐、未来干预不影响过去、bank容量/reset隔离、V1动作逐位等价、V2a不接过去命令、V2b Query不接当前Body、Query独立动作影响、严格state_dict恢复、BF16 attention精度、collator对齐。最新回归见本报告开头。

完整尺寸权重另行跑通五种配置。V2 三分支各做12次固定噪声小样本拟合，再做2次训练模式随机噪声更新；冻结复用 V1，仅优化新增层。没有将这组数据作为任务成功率实验。

| 配置 | 新增参数（相对 V1） | 固定噪声初始 loss → 12步后 | GPU峰值分配 |
|---|---:|---:|---:|
| Baseline / V1 | 0 / 0 | 完整尺寸 forward 通过 | — |
| V2a | 567,041 | 0.01420364 → 0.01382465 | 9.37 GiB |
| V2b | 3,281,538 | 0.01420436 → 0.01382479 | 9.42 GiB |
| V2c | 3,281,538 | 0.01420436 → 0.01384564 | 9.42 GiB |

V2b/c 参数结构和容量完全匹配，不需要额外容量补偿。固定噪声测试用于检查可学习性；两次随机训练 loss 不应与固定噪声前后值比较。所有 loss/梯度有限，无 NaN。首批 V2c `body_encoder.dynamic.0.weight` 梯度范数约 `5.05e-7`，`history_encoder.0.weight` 约 `3.70e-7`，`query.fuse.0.weight` 约 `4.58e-11`，`history.q.weight` 约 `3.89e-11`，`memory_projection.weight` 约 `8.97e-8`。V2b 的 `query.body.weight` 按设计梯度为零，其余活跃路径有梯度。

独立进程 FK 最大位置误差：左 `8.996e-7 m`，右 `1.080e-6 m`；旋转矩阵最大元素误差分别 `4.92e-7`、`8.27e-7`。已知 EEF 固定偏移和双臂交错 q 顺序得到独立仿真记录验证。

### 动作影响：实际完整专家，固定动作噪声

以下为训练后 V2c，对首8步、有效23维的**动作变化 RMS**，不是对示范的 MAE，也不是成功率。动作仍在模型归一化空间。

| 干预 | Query 变化 | Attention变化 | 动作变化 RMS |
|---|---|---|---:|
| 仅当前左臂 Body q 改变0.3rad，原 state 输入保持固定 | 有 | 有 | 0.00098699 |
| 只改过去 Body q | 当前 Query不变 | 有 | 0.00103459 |
| 只改过去已执行命令 | 当前 Query不变 | 有 | 0.00099542 |
| 只清空新增具身 bank，保留 V1 bank | 当前 Query不变 | 空历史 | 0.00099644 |
| **仅屏蔽 Query 的 Body 输入，保持当前Body残差与历史完全相同** | 有 | 有 | **0.00094146** |

最后一项隔离 Query 检索路径，排除了“只是当前 Body 残差改变动作”的解释。V2b 对当前 Body 干预的 Query、Attention 均不变，当前 Body 残差仍改变动作；符合消融定义。V2a 当前 Body 干预也改变动作。

保存恢复：三分支训练后 extension 严格恢复均预测逐位一致；V2c 额外独立新进程验证完整 V1+extension 加载，预测逐位一致。轻量 checkpoint manifest 校验两个原 V1 shard、config、索引、URDF和mapping。原 V1 参数训练前后逐参数 hash 未变。完整 portable 保存路径已提供；此次实测的是冻结 V1 的轻量 checkpoint 路径。

真实观测端到端：用已有 `live_server_400/request000.npz`、`request001.npz`，以及 `live_sim_400/step0007.npz` 中实际执行命令，跑通 RGB→Processor→Eagle→V1/V2→ActionHead→decode，两次都输出有限的六组32步动作。没有环境执行。本次不把网络输出合理直接等同于控制器闭环正确。

### 精度问题及修复

本轮开发的新增检索初版在 BF16 中累加点积，并将 softmax 概率转回 BF16。实际当前构型改变了 Query，但 Attention 记录完全相同；用相同投影后的 Q/K 在 FP32 重算，最大概率变化 `2.04e-5`，直接证实被舍入。

`memory.py` 现将 Q/K 点积累加、softmax、加权求和保留FP32。V1、Eagle和ActionHead没有变化。修复后重新做 V2b/c 的12+2步验证：当前构型干预的 Attention 最大变化 `4.09e-5`，Query-only 消融最大变化 `0.002419`，都影响实际动作。最终使用 `pilot_attention_fp32_20261111` 结果，不能混用早期 BF16 版本结论。

## 5. 计算开销

单环境、四个顺序决策、排除首次预热、5轮中位数的融合耗时：V1 `4.37ms`，V2a `40.76ms`，V2b `42.48ms`，V2c `47.44ms`。这是包含 V1 Memory 的融合段，**不包含 Eagle 和 Action Expert**；共享机器上小样本计时有波动，不能把 b/c 的差值归因于网络复杂度。当前逐节点 FK 有较多小GPU操作，首版尚未做向量化。完整视觉与动作路径总延迟应在短程集成时另测。

## 6. 证据文件

仓库根目录下：

- `outputs/v2_groot_memoryvla/pilot_final_20261111/{results.json,sample.json}`：Baseline/V1/V2a实测，V2a checkpoint及动作数组。该目录早期 V2b/c 结果不作为最终检索结论。
- `outputs/v2_groot_memoryvla/pilot_attention_fp32_20261111/results.json`：最终 V2b/c 学习曲线、参数量、动作影响。
- 同目录 `{v2b,v2c}_gradients.json`、`{v2b,v2c}_actions.npz`、`{v2b,v2c}_checkpoint/`：实际梯度、动作、可重载新增参数及原权重指纹。
- 同目录 `fresh_diagnosis.json`：新进程恢复、FK误差、Query独立干预、耗时及真实抓包推理的完整结果。
- 更早的 `pilot_20261111/` 为开发中间结果，未完成最终消融边界验证，不用于验收。

## 7. 尚存边界与下一步

1. 本次证明分支可训练且影响动作，尚未证明提高成功率、鲁棒性或样本效率。四帧短序列的变化有限，Query/qk 梯度明显小于 Body/Value 路径；仅看 loss 不能认定学到了有用检索。
2. 当前历史每帧压缩为一个融合向量；历史动作是结构属性加权摘要，没有保留完整逐关节历史 Token。它满足首版低成本验证，但可能损失细粒度关节动作关联。应先做 V2b/c 对照，再决定是否保留更细粒度历史。
3. URDF visual 目前读取尺寸或 mesh scale、质量和惯性，没有加载 mesh 顶点；mimic 等其他关节类型需显式扩展。URDF车轮在USD中不可按虚拟底盘 q 对应，当前缺测并 mask。
4. 转换数据只有向后差分速度，在线通常有真实 qvel。建议短程训练和评估都使用已有原始 qvel；缺测时 mask 明确，但差分噪声及不同采样间隔仍需关注。没有世界轨迹对齐、SLAM或恢复规则。
5. 在线 evaluator 仍需在 V2 专用入口发送观测采集时间、最终实际执行动作及其时间。V1服务没有被改动；不要直接用旧客户端忽略新增字段。旧抓包的两次推理已经验证该数据契约。
6. 当前轻量 checkpoint 不包含 optimizer、bank运行态或中途 Episode 时钟，不支持逐位继续中途训练；新训练从 Episode 边界初始化。联合更新 V1 时使用完整 `save()`，不要使用依赖只读 V1 的轻量保存。
7. 尚未跑 Task0/21/22 短程仿真、控制器目标闭环或成功率比较。建议下一步先用同一 V1 初始权重，在有实际手臂运动的少量训练窗口上等数据、等预算比较 V2b/c，继续记录 Query/Attention/动作敏感性；确认梯度与计算开销可接受后，再各做一次短程闭环验证。无需现在启动大规模任务。
