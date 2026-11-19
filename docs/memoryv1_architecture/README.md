# 当前 MemoryV1 网络架构与 MemoryVLA 差异

核对日期：2026-11-17。依据当前本地源码、Task40 原版 step4000 checkpoint 配置，以及仓库内 MemoryVLA 参考源码与论文 LaTeX。本文描述网络及数据流，不评价任务成功率。

## 1. 本文画的是哪一个 V1

当前 V1 可以概括为：**在 GR00T N1.6 的完整视觉语言条件上，通过双路记忆检索加入历史 Value 残差，再交给原有 flow-matching 动作专家。**

主图对应这一份可核对的实际配置：

[`Output/task40_memory_overnight/replay/original_step4000/config.json`](../../Output/task40_memory_overnight/replay/original_step4000/config.json#L33)。它是 Task40 原版 step4000 的本地归档，不表示已经检查了某个远端在线服务。

| 配置项 | 主图对应值 |
| --- | --- |
| 起点 | GR00T-N1.6-BEHAVIOR1k，见原实验记录 |
| `memory_conditioning` | `residual`，保留完整 GR00T 上下文 |
| `memory_residual_readout` | `value`，读出末层注意力的历史 Value 消息 |
| `memory_length` | 每一路各 64 个时间槽 |
| `memory_perceptual_dim` | 256 |
| 认知维度 | 2048，来自 Eagle 隐藏维度 |
| `memory_retrieval_layers` | 每一路 2 层，单头交叉注意力 |
| `memory_cognitive_qk_norm` | `layernorm`，Q/K 投影前无参数 LayerNorm |
| `memory_residual_identity_init` | `true`，新建时残差输出为零 |
| `memory_cognitive_residual_projection` | `false`，认知输出用可学习标量 |
| `memory_cognitive_residual_scope` | `eos`，认知增量仅注入 user EOS |
| `memory_update_fused` | `true`，写回门控融合结果 |
| 训练范围 | 冻结 Eagle、视觉投影和整个原动作专家，仅训练 Memory |
| 历史训练窗口 | 最多 32 个决策；与 64 槽容量分开定义 |

**V1 并不是一个固定不变的接入图。** 当前训练参数入口默认是 `residual / delta / capacity=64 / cognitive-QK-LN`；本图的 `value` 是实际 checkpoint 显式选中的分支。`direct`、旧 `residual/delta`、矩阵认知投影、全有效文本注入、动作专家联合训练都仍有对应代码或实验，但不并入这张主图。源码配置类还保留了旧权重兼容默认值，不能用它替代具体 checkpoint。

图与源文件：

- **演示用 16:9 横版：[当前 V1 SVG](architecture_16x9.svg) · [PNG](architecture_16x9.png)；[两者对照 SVG](comparison_16x9.svg) · [PNG](comparison_16x9.png)。**
- [设计疑问与改动依据](design_rationale.md)：特征位置、EOS、残差接入、Value 读出及容量选择。
- [当前 V1 结构图 SVG](architecture.svg) · [PNG](architecture.png) · [可编辑 DOT](architecture.dot)
- [MemoryVLA / V1 对照图 SVG](comparison.svg) · [PNG](comparison.png) · [可编辑 DOT](comparison.dot)

## 2. 当前观测如何进入双路记忆

### 2.1 原 GR00T 视觉语言主干

输入为当前时刻的 head / left-wrist / right-wrist 三路 RGB，以及当前语言指令；本体状态另走动作头。历史图像不作为视频序列重新送入 Eagle。

```text
三路当前 RGB
  → Eagle 的 SigLIP2 视觉编码器
  → 空间重排 / 下采样 + 原 mlp1 视觉投影
  → V_t [B, 3, 81, 2048]，当前 B1K 256×256 输入时每路 9×9

V_t + 指令 / prompt tokens
  → Eagle 保留的 16 层语言模型（Qwen3 系列）
  → 完整上下文 H_t [B, S, 2048]
```

`S` 随指令和 padding 改变，不应在图上写死为 283。这里的 16 层是 GR00T 截取后的实际语言主干，Eagle 内部原始配置的 28 层不是本配置最终执行深度。

源码：[EagleBackbone](../../src/v1_groot_memoryvla/model/modules/eagle_backbone.py#L55)、[视觉特征导出](../../src/v1_groot_memoryvla/model/modules/eagle_backbone.py#L117)、[extract_feature](../../src/v1_groot_memoryvla/model/modules/nvidia/Eagle-Block2A-2B-v2/modeling_eagle3_vl.py#L355)。

### 2.2 感知路：LLM 前的视觉投影特征

`V_t` 已经过 Eagle 原视觉投影，但还没经过 LLM。每个相机分别使用同一套 `BottleneckSE(2048,512,256)`：

```text
每相机 [81,2048]
  → 恢复 9×9 空间布局
  → 1×1 Conv: 2048→512 + ReLU
  → SE 通道门控: GAP → 512→32→512 → sigmoid
  → 1×1 Conv: 512→256
  → [81,256]

三相机按原顺序拼接 → P_t [B,243,256]
```

压缩的是通道数，不减少该分支的空间 token 数。不能把 243 个 token 当成一个正方形网格输入 SE，也不能把它画成 LLM 后的图像 token 特征。

### 2.3 认知路：LLM 后的 user EOS

从完整 `H_t` 中抽取最后一个有效 user `<|im_end|>`：

`C_t = H_t[user_EOS] ∈ R^[B,1,2048]`。

当前单 user 模板下，以 token id `151645` 和有效 attention mask 定位。它不是固定索引、system EOS、padding 或最后换行，也不是额外加入的可学习 Body Query。

源码：[MemoryAdapter.forward](../../src/v1_groot_memoryvla/model/memory/adapter.py#L112)。

## 3. 双记忆内部：读出给动作，与融合写回分开

两路模块结构相同、参数互不共享；当前没有感知库到认知库的直接跨路注意力。

| 分支 | 当前 Query | 单个历史槽 | 64 槽展开后的 K/V |
| --- | --- | --- | --- |
| 感知 | `[1,243,256]` | `[243,256]` | `[1,15552,256]` |
| 认知 | `[1,1,2048]` | `[1,2048]` | `[1,64,2048]` |

表内按一个环境表示；代码逐样本按 episode key 读写。64 表示压缩后的时间槽上限，不是精确保留最近 64 张原始帧。

### 3.1 两层检索与时间编码

设任一路当前特征为 `x`，历史内容拼接为 `M`，历史时间编码为 `TE`：

```text
q₀ = x
K_input = M + TE(t_history)
V_input = M

第 l 层：
  Q_l = Wq_l N(q_{l-1}) + bq_l
  K_l = Wk_l N(K_input) + bk_l
  A_l = softmax(Q_l K_lᵀ / sqrt(d))
  u_l = A_l (M Wv_lᵀ + bv_l)
  z_l = LayerNorm(q_{l-1} + u_l)
  q_l = LayerNorm(z_l + FFN_l(z_l))
```

`N` 在认知路为无可学习参数的 LayerNorm；感知路为恒等映射。两层都读取同一份历史 K/V，第二层 Query 来自第一层输出。时间编码只加到 K 的输入，不加到历史 V，也不是动作 flow time。

### 3.2 门控融合：用于写回

按当前源码约定，`g` 是“当前特征”的权重：

```text
g = sigmoid(Linear([x, q₂]))
F = g ⊙ x + (1-g) ⊙ q₂
```

正常历史模式下，写入库的是 `F.detach().clone()`。论文公式的门控符号方向与源码命名不同，图中用“当前权重 g / 历史权重 1-g”可避免歧义。

### 3.3 `value`：用于动作残差

当前主图的动作读出，不是 `F`，也不是 `F-x`。取同一次第二层注意力中的**无 V 投影偏置**消息：

```text
m = A₂ (M Wv_2ᵀ)
r = (1-g) ⊙ m
```

其中 `A₂` 由完整两层检索路径产生，门控也依赖当前特征与检索结果。`value` 仍是当前观测条件下的历史读出，并非不依赖 Query；它移除了外层 `F-x` 中显式的负当前项。

常规 Transformer 融合路径仍补回 `bv_2`、做残差 / LayerNorm / FFN，从而保持 `F` 的原有融合写回。没有额外增加第三次检索。

这使两种特殊情况清楚成立：

- 没有历史：`F=x`，`r=0`，正常模式下仍写入当前特征。
- 当次读取的所有历史 V 内容均为零：`m=0`，动作残差为零，即使保留历史槽位和时间编码。写回的 `F` 此后未必仍为零，不能把这一点扩大为“清零一次后未来永久不变”。

`history_mode=empty` 是另一种干预：完全旁路历史读写。主图画正常模式。

源码：[CrossTransformerBlock](../../src/v1_groot_memoryvla/model/memory/bank.py#L48)、[GateFusion](../../src/v1_groot_memoryvla/model/memory/bank.py#L101)、[read_fuse_write](../../src/v1_groot_memoryvla/model/memory/bank.py#L130)。

## 4. 历史消息怎样接回原 GR00T

记感知和认知读出为 `r_p`、`r_c`，原上下文为 `H_t`。主图对应 scalar / EOS 版本：

```text
图像 token 位置： H'_image = H_image + W_p r_p
user EOS 位置：  H'_EOS   = H_EOS   + α_c r_c
其余 token：    H'_other = H_other

W_p: 无 bias Linear(256→2048)，新建时全零
α_c: 可学习标量，新建时为 0
```

整个 `H'_t` 仍是 `[B,S,2048]`；image mask、attention mask 与原 GR00T 一致。所有历史槽不会拼接成新的 LLM 上下文，也不重新送回 LLM。空历史时，原条件保持不变；新建 Memory 时，即使有历史，零初始化也使初始条件保持不变。

变体单列：

- `cognitive_residual_projection=true`：把标量换成零初始化、无 bias `Linear(2048→2048)`。
- `cognitive_residual_scope=text`：将同一认知增量广播到全部有效非图像 token。
- `residual_readout=delta`：将 `r` 换成 `F-x`，其余接入位置由各自配置决定。
- `conditioning=direct`：使用 `[Linear(P_fused), C_fused]` 作为动作条件；当前三相机下是 `[B,244,2048]`，不保留原完整 `H_t`。这是先前阶段的独立版本。

源码：[残差注入与 direct 分支](../../src/v1_groot_memoryvla/model/memory/adapter.py#L137)、[零初始化](../../src/v1_groot_memoryvla/model/memory/adapter.py#L35)。

## 5. 原动作专家：保留 GR00T 的 flow matching

```text
H'_t → 原 vlln → 原视觉 / 非图像条件路由

当前 state → 原 state_encoder → state token [B,1,1536]
噪声 / 中间动作 + flow time + embodiment id
  → 原 action_encoder → action tokens [B,50,1536]

[state token; action tokens]
  → 原 32 层 AlternateVLDiT
  → 原 action_decoder
  → 速度场 [B,50,128]
  → 推理时 4 次 Euler 更新
  → 模型动作 [B,50,128]
  → 原 processor 解码当前 B1K 动作 [B,32,23]
```

1536 是 Transformer 内部 token 宽度；1024 是该 checkpoint 的输出 / 解码器隐藏宽度。32 层内部交替进行条件交叉注意力与状态动作自注意力；条件交叉注意力再按原 mask 选择非图像或图像位置。认知 EOS 因而通过原非图像条件分支影响动作，不会作为新 token 拼到状态动作序列中。

训练采用 `a_t=(1-t)ε+t·a`，目标速度 `a-ε`，以有效动作 mask 计算 MSE。没有为主图增加单独的记忆辅助损失。

Memory 每次新观测只读写一次；同一观测下四次 Euler 更新复用条件，不能画成四次写库。

源码：[flow-matching 动作头](../../src/v1_groot_memoryvla/model/gr00t_n1d6/gr00t_n1d6.py#L151)、[Euler 推理](../../src/v1_groot_memoryvla/model/gr00t_n1d6/gr00t_n1d6.py#L292)、[条件路由](../../src/v1_groot_memoryvla/model/modules/dit.py#L324)、[每观测一次读写](../../src/v1_groot_memoryvla/model/gr00t_n1d6/gr00t_n1d6.py#L535)、[实际模型配置](../../Output/task40_memory_overnight/replay/original_step4000/config.json)、[实际 processor](../../Output/task40_memory_overnight/replay/original_step4000/processor_config.json)。

## 6. 记忆生命周期

正常模式依次执行：**读取严格早于当前决策的历史 → 两层检索 → 门控 → 输出历史消息并写入融合结果 → 超容量时合并**。

- 感知、认知各自挑选时间相邻且平均余弦相似度最高的一对槽位，逐元素取平均合并；两路选中的合并位置可以不同。
- 合并保留较早槽的代表时间戳，不记录该槽完整的原始时间范围；不是简单 FIFO，也不是每次删除最旧槽。
- 历史保存 `detach/clone` 的特征，不跨历史写入链反传；当前观测的检索、门控与残差投影仍可训练。MemoryVLA 原参考代码也 detach 历史。
- 训练窗口至多 32 个决策，同 Episode 窗口间延续 detached 历史，Episode 结束重置。容量 64 不等于反传窗口 64。
- 在线按 `(session, env, episode)` 隔离状态；一个环境结束不会清掉其他环境历史。时间按决策推进，不按每个 Euler 子步推进。

源码：[合并](../../src/v1_groot_memoryvla/model/memory/bank.py#L178)、[在线时钟与隔离](../../src/v1_groot_memoryvla/policy/memory_state.py#L4)、[跨窗处理](../../src/v1_groot_memoryvla/model/memory/adapter.py#L104)。

## 7. 与原 MemoryVLA 的改动差异

对照的是仓库内固定参考版，而不是未经检查的最新上游。

| 比较项 | MemoryVLA 论文 / 本地参考代码 | 当前 V1 主图 | 归类 |
| --- | --- | --- | --- |
| 基础 VLA | Prismatic：DINOv2 + SigLIP + LLaMA-7B | GR00T N1.6：Eagle / SigLIP2 + Qwen3 系列截取主干 | 底座迁移 |
| 观测视角 | 方法主图单路第三人称 RGB | B1K 三路当前 RGB | 输入适配 |
| 感知取特征位置 | 双视觉编码器输出拼接，视觉语言投影之前 | Eagle `mlp1` 之后、LLM 之前 | 特征取出位置改变；两者均来自视觉编码器 |
| 感知 token | 常用 `256×256`（数量×宽度） | `243×256`，3×81 | 保持 256 通道，调整空间数量 |
| SE 输入通道 | 参考双视觉配置 2176→512→256 | 2048→512→256，每相机共享应用 | 维度与多视角适配 |
| 认知 token | 1 个 EOS，4096 通道 | 1 个有效 user EOS，2048 通道 | 底座 / 模板适配 |
| 记忆基本结构 | 感知 + 认知双库 | 相同双库思想 | 保留 |
| 历史检索 | 两层单头交叉注意力，K 加时间、V 不加 | 保留，并给认知 Q/K 输入增加无参数 LN | 数值适配 |
| 门控 | 源码 `g·current+(1-g)·retrieved` | 保留 | 保留 |
| 给动作头的记忆内容 | 融合 `P_fused`、`C_fused` 直接作为条件 | 门控后的末层无偏置历史 Value 消息作为残差 | 关键方法改变 |
| 动作条件接口 | 认知条件与时间结合后拼入动作序列；感知条件做 cross-attention | 原完整 `H` 上按位置加残差；复用 GR00T 图像 / 非图像 cross-attention | 关键接口改变 |
| 新增输出初始化 | 没有本项目这种 `H+零初始化输出` 适配器 | `W_p=0`、`α_c=0`，初始保留原 GR00T 条件 | 新增适配设计 |
| 写入何种内容 | 论文写融合结果；参考代码 `update_fused=False` 默认写当前特征，可设 True | 固定融合结果写回 `update_fused=True` | 遵循论文，与参考代码默认不同 |
| 历史容量 | 默认每路 16 槽 | 本配置每路 64 槽 | 超参改变 |
| 满库维护 | 相邻最相似槽平均合并 | 保留，双路各自决定 | 保留 |
| 历史梯度 | 写入 detach / clone | 同样 detach / clone | 保留，非 V1 独有差异 |
| 动作专家 | 默认 DiT-L，24 层、1024 宽；认知 self-attention + 感知 cross-attention | 原 GR00T AlternateVLDiT，32 层、内部 1536 宽，原状态分支保留 | 动作底座迁移 |
| 生成目标 | 参考代码噪声预测 MSE，通常 10 步 DDIM | GR00T 速度预测 MSE，4 步 Euler flow matching | 生成范式改变 |
| 动作接口 | 典型 16×7 末端动作 | 模型 50×128 → B1K processor 32×23 | 本体 / 动作协议适配 |
| 训练范围 | 参考默认允许 VLM 全参微调，具体依配置 | 这份 checkpoint 冻结原 VLM 与动作专家，仅训练 Memory | 实验选择，非所有 V1 必然如此 |
| 训练历史 | 参考有 group / stream；论文 LIBERO 为同 Episode 随机采样 16 帧 | 当前正式流程同 Episode 连续最多 32 决策窗口，跨窗保留历史 | 数据时序协议适配 |
| 在线隔离 | 参考 predict_action 使用单 Episode / 单样本路径 | 支持 session / env / episode 独立生命周期 | 工程适配 |

应描述为：**“MemoryVLA 风格双路记忆在 GR00T 上的 residual/value 迁移实现”**。双库、检索、门控与合并来自 MemoryVLA；完整上下文残差接入、value 读出、零初始化接口及认知 Q/K LN 是本项目的适配改动。不能仅凭方法来源称为原网络逐算子复现。

参考位置：[MemoryVLA 记忆模块与配置](../../src/references_code/memoryvla/vla/memory_vla.py#L71)、[特征来源与动作条件](../../src/references_code/memoryvla/vla/memory_vla.py#L513)、[原 DiT 的条件使用](../../src/references_code/memoryvla/action_model/models.py#L279)、[原噪声损失](../../src/references_code/memoryvla/action_model/action_model.py#L57)、[论文方法](../../related_work/MemoryVLA/latex/chapters/3_method.tex)、[论文训练协议](../../related_work/MemoryVLA/latex/chapters/6_appendix.tex#L119)。

## 8. 可直接交给绘图 AI 的说明

> 绘制一张 1920×1080、16:9 横版论文风格的网络结构图，标题为 “MemoryV1: Residual Value Memory for GR00T N1.6”。从左向右依次排列原视觉语言主干、双路记忆检索、残差接入、原动作专家。主图严格采用 residual/value、scalar/EOS、每路64槽的配置。蓝色为冻结的原GR00T模块，青色为可训练Memory模块，橙色突出新增残差接口；虚线表示跨决策的持久记忆读写。
>
> 上方画当前三路RGB图像和语言指令。图像经过Eagle的SigLIP2、空间下采样和mlp1，输出V_t [B,3,81,2048]。V_t一支与文本一起进入保留16层的语言主干，得到完整H_t [B,S,2048]；另一支在每相机9×9网格上使用共享BottleneckSE(2048,512,256)，再拼成P_t [B,243,256]。从H_t最后有效user EOS取C_t [B,1,2048]。
>
> 中部并列画感知和认知两路MemoryBank，参数独立，各64时间槽。当前P_t/C_t作Query，历史融合特征作Value，历史特征加时间编码作Key。各有两层单头交叉注意力，仅认知路在Q/K投影前做无参数LayerNorm。检索后用g=sigmoid(Linear([current,retrieved]))门控。
>
> 每一路一定要画两种不同输出：第一种F=g·current+(1-g)·retrieved，经过detach/clone写回各自记忆库；满库时合并时间相邻且最相似的一对，取平均。第二种为动作残差r=(1-g)·A_2·(M Wv_2^T)，来自第二层同一次注意力，Value投影不含bias。不要把融合写回F与历史读出r画成同一个节点。
>
> H_t保留一条粗实线到相加节点。感知r_p经零初始化无bias Linear(256,2048)，只加在H的图像位置；认知r_c乘零初始化可学习标量α_c，只加在user EOS位置。其余token与mask保留，形成H'_t [B,S,2048]。初始化时残差为0；空历史时残差也为0。不要画成244 token直接替换，不要把历史token送回LLM。
>
> 下方画原vlln与冻结的32层GR00T AlternateVLDiT动作专家。state经原state_encoder；噪声动作和flow time经原action_encoder；状态和动作token进入DiT，H'_t作为原图像/非图像交叉注意力条件。动作头输出速度场，4次Euler更新得到[B,50,128]，processor解码为当前B1K的[B,32,23]。标注Memory每个观测只更新一次，4次Euler不重复写库。不要画DDIM、7维动作、Body Query、额外辅助loss或LoRA。
>
> 在右下角用小框列出与原MemoryVLA的主要变化：Prismatic→Eagle；感知特征来自视觉投影后；认知4096→2048；16→64槽；认知Q/K LN；融合直接条件→历史Value残差；原动作专家迁移为GR00T flow matching。写回融合特征遵循论文，参考源码默认update_fused=False应在图注单独说明。

## 9. 本次核对范围与重绘

源码主干与归档 `source_original4000` 的记忆模块差异已检查：当前相关新增内容为诊断采集，不改变上述主线的网络参数与读出 / 写回设计。本次未重新运行 GPU 模型或评估，不把静态核对当作性能验证。

现有 `src/v1_groot_memoryvla/media/model/architecture.dot` 仍画 `direct` 的 244-token 接入。它与 README 中“新运行 direct”的历史段落不能代表本文选择的 residual/value checkpoint；本次另存图文，不覆盖旧实验材料。

在仓库根目录重绘：

```sh
dot -Tsvg docs/memoryv1_architecture/architecture.dot -o docs/memoryv1_architecture/architecture.svg
dot -Tpng -Gdpi=140 docs/memoryv1_architecture/architecture.dot -o docs/memoryv1_architecture/architecture.png
dot -Tsvg docs/memoryv1_architecture/comparison.dot -o docs/memoryv1_architecture/comparison.svg
dot -Tpng -Gdpi=140 docs/memoryv1_architecture/comparison.dot -o docs/memoryv1_architecture/comparison.png
```
