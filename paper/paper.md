# Action-Subspace-Conditioned Memory Retrieval for Mobile Bimanual Manipulation

## Abstract

Mobile bimanual manipulation is inherently partially observable: objects leave the field of view during navigation, task progress cannot always be inferred from the current frame, and the information required for locomotion is not identical to that required for fine manipulation. Existing memory-augmented vision-language-action (VLA) methods address this problem through temporal history, recurrent or bank-based memory, or persistent spatial representations. However, most methods construct a shared memory representation or perform retrieval independently of the action subgroup that will consume it. EchoVLA already introduces base–arm per-part diffusion control, but its scene and episodic memories are retrieved according to scene/task context and then fused to guide the downstream control policies; the retrieval query itself is not conditioned on the action subgroup. We therefore study a narrower question: **should different action subspaces retrieve different historical evidence from the same memory?** We formulate action-subspace-conditioned retrieval under a partially observable decision process. A shared lightweight memory is queried using the current multimodal state together with an action-group descriptor derived from the robot embodiment, producing a global memory context and group-specific memory contexts. These contexts condition a shared π0.5 flow-matching action expert through lightweight modulation while preserving cross-group interaction. The proposed formulation does not assume that action groups are independent and does not require reconstructing a complete 3D world model. Its central empirical hypothesis is that, under a fixed memory budget, action-conditioned retrieval is more decision-efficient than a single uniform retrieval result when the task contains strongly heterogeneous mobile-manipulation actions.

**Keywords**: vision-language-action, memory retrieval, mobile bimanual manipulation, partial observability, action subspace, flow matching

---

## 摘要

移动双臂操作具有天然的部分可观测性：机器人移动后目标会离开视野，当前图像不一定能够反映任务已经执行到哪一步，而底盘移动与精细操作所需要的历史信息也并不相同。现有记忆增强视觉-语言-动作（VLA）方法主要通过历史帧、循环或记忆库、以及持久空间表示缓解这一问题，但多数方法仍构造一个面向整个策略的共享记忆表示，或以当前场景与任务上下文进行统一检索，而不显式考虑“接下来由哪个动作子空间使用这段记忆”。EchoVLA 已经采用 base–arm 分部件扩散控制，但其场景记忆与情节记忆仍先依据当前观测、任务历史与指令进行检索，再经粗/细粒度注意力融合后服务下游控制；检索查询本身并不以动作子空间为条件。基于此，本文研究一个更具体的问题：**同一共享记忆中，不同动作子空间是否应该检索不同的历史证据？** 本文在部分可观测决策框架下，将记忆检索表述为动作条件的充分历史压缩问题，并提出动作子空间条件记忆检索：共享一个轻量记忆库，以当前多模态状态和由机器人本体结构得到的动作组描述共同生成查询，同时得到全局记忆上下文与动作组特定的记忆上下文，再通过轻量调制作用于共享的 π0.5 流匹配动作专家。该方法不假设各动作组相互独立，也不要求恢复完整三维世界状态。本文核心待验证假设是：在相同记忆预算下，当任务包含明显异构的移动与操作动作时，动作条件检索能够比统一检索提供更高的决策信息效率。

---

## 1. Introduction

移动双臂机器人在长时任务中需要持续在导航、位姿调整、抓取、双臂协作和接触操作之间切换。机器人一旦移动，曾经可见的目标可能离开当前视野；当任务包含多个连续子步骤时，仅凭当前图像也未必能够判断已经完成了哪些操作。因此，移动操作并不是严格的单帧马尔可夫决策问题，而是典型的部分可观测控制问题。

现有 VLA 记忆方法大致从三条路线缓解这一问题。第一类直接扩大时间上下文，例如 MEM [2] 在 π0.5 上引入短时视觉与本体历史；第二类维护可检索的隐式记忆，例如 MemoryVLA [3] 使用感知—认知记忆库对历史进行压缩与检索；第三类维护更显式的空间状态，例如 SERF [4]、SOMA [8] 和 EchoVLA [9] 使用空间或场景级持久表示支持视野外操作。RoboMME / MME-VLA-Suite [5] 进一步表明，记忆的表示方式与记忆注入动作策略的方式都会显著影响长时任务表现。

这些工作已经充分说明“VLA 是否需要记忆”不再是本文的问题。本文关注的是记忆被**如何读取**。在移动双臂系统中，联合动作可由多个具有不同控制语义的动作子空间构成。例如，底盘运动更依赖目标的大尺度位置和跨视角历史，而精细抓取更依赖近期物体姿态、抓取状态与局部交互历史。若所有动作维度只接收同一个压缩后的检索结果，则该表示需要同时服务多个不同的动作需求；在固定 token 或记忆容量约束下，这可能造成无关历史占用有限的表示与注意力预算。

EchoVLA [9] 已经说明移动操作中的动作生成可以采用 base–arm 分部件控制，因此“将动作空间分解”本身不是本文的新颖性来源。本文进一步研究的是另一层问题：**是否应在动作生成之前，就根据动作子空间决定从共享历史中读取什么。** 换言之，EchoVLA 关注的是检索后的记忆如何服务不同控制部件，而本文关注的是检索操作本身是否应该由动作子空间条件化。

本文研究的核心问题为：

> **在共享记忆和固定记忆预算下，统一记忆检索是否足以支持具有异构动作子空间的移动双臂控制？**

围绕该问题，本文给出三点工作：

1. **动作条件的充分历史形式化。** 将长时历史压缩为面向具体动作子空间的充分统计量，并用条件信息瓶颈刻画“保留动作相关历史、压缩动作无关历史”的目标。
2. **动作子空间条件检索。** 所有动作组共享同一记忆库，但使用包含当前状态、动作组身份与本体结构信息的不同查询；同时保留一个全局记忆上下文，以避免把底盘与双臂人为割裂。
3. **面向 π0.5 的统一动作专家接入。** 保留共享的 Flow-Matching Action Expert，通过轻量 memory modulation 让不同动作组读取对应的检索结果，并将 prefix context 与独立 memory expert 作为对照注入方式。

---

## 2. Related Work

### 2.1 Temporal and Episodic Memory in VLA

MEM [2] 通过短时视觉历史和本体历史增强 π0.5，使当前动作能够利用过去若干观测。MemoryVLA [3] 不直接保留全部历史，而是维护感知记忆与认知记忆，以当前状态为查询对记忆库进行交叉注意力检索，并通过门控融合与记忆合并控制长期容量。RoboMME / MME-VLA-Suite [5] 在统一 π0.5 骨干上系统比较了感知、循环和符号记忆，以及 context、modulation 和 memory expert 等不同注入机制。

这些方法构成本文最直接的技术基础。本文不重新定义“长期记忆”本身，而把变量集中到检索条件：在相同共享记忆与相同预算下，比较统一查询与动作子空间条件查询。

### 2.2 Spatial Memory for Mobile Manipulation

SERF [4] 通过持续累积环境与机器人特征形成时空表示，以机器人状态查询历史空间信息。SOMA [8] 构建空间—语义记忆，并依据当前任务上下文检索视野外目标相关信息。EchoVLA [9] 为移动操作维护互补的场景记忆与情节记忆：两类记忆在训练和推理过程中分别存储、更新和检索，其检索结果再通过粗粒度与细粒度注意力融合，用于引导移动底盘与机械臂的扩散控制。

EchoVLA 与本文最接近，但两者的分解位置不同。EchoVLA 的差异化主要体现在**控制端**：融合后的记忆上下文服务 base–arm 分部件扩散策略；本文的差异化发生在**检索端**：动作子空间首先产生不同的查询，从同一共享记忆中读取不同历史证据，再由共享动作专家联合生成动作。因此，本文研究的是 *per-action-subspace retrieval*，而不是再次提出 *per-part control*。

### 2.3 Heterogeneous Action Spaces and Shared Action Experts

通用 VLA 需要处理不同机器人本体与动作定义。π0 / π0.5 [10] 使用跨本体训练与 Flow-Matching Action Expert 支持单臂、双臂及移动操作等不同机器人数据。官方 π0.5 方法以统一动作专家和异构数据共训练为核心，并不需要把本文的创新建立在“π0.5 已经具有独立 base/arm denoiser”这一前提上。

本文把动作向量按运动学链与控制语义划分为若干动作子空间，但不假设各子空间统计独立。该划分仅用于生成不同的记忆查询与调制条件，最终仍通过共享动作专家建模底盘、躯干与双臂之间的耦合。

### 2.4 Partial Observability and Action-Sufficient Memory

在部分可观测马尔可夫决策过程（POMDP）中，最优策略原则上依赖于对历史的充分表示 [1]。对于高维机器人控制，恢复完整信念状态或完整世界模型通常并非唯一选择；只要一个历史压缩表示保留了预测当前动作所需的信息，就可以作为策略的充分条件。

本文据此把记忆设计从“尽可能恢复完整过去”改写为“针对当前动作需求保留足够的过去”。这一观点不否定三维地图或完整历史的价值，而是提出一个更弱、也更容易检验的假设：在固定表示预算下，不同动作子空间对应的最优历史压缩可以不同。

---

## 3. Problem Formulation

### 3.1 Partially Observable Decision Process

设环境隐藏状态为 $s_t$，当前观测为 $o_t$，语言任务指令为 $\ell$，动作历史为 $a_{<t}$。定义截至时刻 $t$ 的历史为

$$
\mathcal{H}_t = (o_{1:t}, a_{1:t-1}, \ell).
$$

在部分可观测条件下，最优动作一般依赖于 $\mathcal{H}_t$，而不仅是当前观测 $o_t$。

### 3.2 Heterogeneous Action Subspaces

将联合动作写为

$$
a_t = \big[a_t^1, a_t^2, \ldots, a_t^G\big], \qquad a_t^g \in \mathcal{A}_g,
$$

其中 $G$ 为当前机器人本体中定义的动作组数量，$\mathcal{A}_g$ 为第 $g$ 个动作子空间。对移动双臂平台，可根据具体机器人将其划分为底盘、躯干/升降机构、左臂—夹爪链和右臂—夹爪链等动作组。该划分来自机器人运动结构与控制语义，而不是固定要求所有机器人必须具有相同的四个专家。

本文不作如下独立性假设：

$$
p(a_t \mid \mathcal{H}_t) = \prod_{g=1}^{G} p(a_t^g \mid \mathcal{H}_t).
$$

双臂协作和移动操作通常具有明显耦合，因此动作子空间只用于记忆查询条件化，而不用于把联合策略完全拆成相互独立的控制器。

### 3.3 Action-Conditional Sufficient History

设共享记忆由历史压缩得到：

$$
\mathcal{M}_t = \Psi(\mathcal{H}_t) = \{m_1, m_2, \ldots, m_K\},
$$

其中 $m_k$ 为历史记忆条目，$K$ 为当前记忆容量。

统一检索使用同一个查询 $q_t$：

$$
r_t = \mathrm{Retrieve}(q_t, \mathcal{M}_t),
$$

并让所有动作维度共享 $r_t$。本文改为对第 $g$ 个动作子空间生成条件查询

$$
q_t^g = f_\theta(h_t, e_g, k_g),
$$

其中 $h_t$ 为当前多模态状态表示，$e_g$ 为动作组的可学习嵌入，$k_g$ 为该动作组的本体/运动结构描述。对应检索结果为

$$
r_t^g = \mathrm{Retrieve}(q_t^g, \mathcal{M}_t).
$$

理想情况下，$r_t^g$ 应当近似满足动作充分性：

$$
p(a_t^g \mid \mathcal{H}_t)
\approx
p(a_t^g \mid o_t, \ell, r_t^g).
$$

该式并不要求 $r_t^g$ 恢复完整世界状态，只要求它保留决定 $a_t^g$ 所需要的历史信息。

### 3.4 Conditional Information Bottleneck View

在固定记忆预算下，动作子空间 $g$ 的检索目标可写成条件信息瓶颈：

$$
\mathcal{L}_{\mathrm{IB}}^{g}
=
I(r_t^g; \mathcal{H}_t)
-
\beta I(r_t^g; a_t^g \mid o_t, \ell),
$$

其中 $I(\cdot;\cdot)$ 表示互信息，$\beta>0$ 控制压缩与动作相关性的权衡。第一项鼓励 $r_t^g$ 不携带过多历史细节，第二项鼓励其保留与当前动作子空间相关的信息。

该目标并不能先验证明动作条件检索一定优于统一检索。它给出的只是一个可检验推论：如果不同 $a_t^g$ 对历史变量的依赖不同，则在有限表示容量下，不同的最优压缩 $r_t^g$ 没有必要完全相同。

### 3.5 Global Context and Group-Specific Retrieval

为了保留动作组之间的耦合，本文同时维护一个不依赖动作组的全局检索结果

$$
r_t^{\mathrm{global}} = \mathrm{Retrieve}(q_t^{\mathrm{global}}, \mathcal{M}_t),
$$

并将第 $g$ 个动作组实际使用的记忆上下文定义为

$$
z_t^g = \mathrm{Fuse}\big(r_t^{\mathrm{global}}, r_t^g\big).
$$

因此本文并不是为底盘、左臂和右臂分别建立独立记忆系统，而是采用“**共享记忆 + 全局上下文 + 动作组条件读取**”的结构。

### 3.6 Testable Hypotheses

本文验证以下两个核心假设：

**H1 — Action-dependent retrieval.** 在包含移动与精细操作的任务中，不同动作子空间学习到的检索分布应存在稳定差异；若所有动作组最终学习到近似相同的检索权重，则动作条件化缺乏必要性。

**H2 — Fixed-budget benefit.** 在相同记忆库、相同 token 预算与相同动作专家容量下，动作子空间条件检索相对于统一检索的增益应随着任务中动作类型切换和跨子空间协同需求增加而增大。

---

## 4. Method

### 4.1 Overview

本文以 π0.5 为基础策略。当前观测首先由原有视觉—语言模块编码为多模态状态 $h_t$；历史信息被写入所有动作组共享的轻量记忆；随后由一个全局查询和 $G$ 个动作子空间查询读取该记忆；检索结果作为条件作用于共享的 Flow-Matching Action Expert。

```text
Current observation o_t + instruction ℓ
                 │
                 ▼
          π0.5 multimodal encoder
                 │
          ┌──────┴────────┐
          │               │
          ▼               ▼
    current state h_t   shared memory M_t
          │               ▲
          │               │ write / update
          │               │
          ├── global query ─────────→ r_global
          │
          ├── action-group query q_1 → r_1
          ├── action-group query q_2 → r_2
          └── ...                     ...
                 │
                 ▼
        global + group memory context z_g
                 │
                 ▼
      shared π0.5 Flow-Matching Action Expert
                 │
                 ▼
             joint action a_t
```

### 4.2 Shared Lightweight Memory

记忆构建本身不是本文的主要创新。默认实现采用与 MemoryVLA [3] 相近的轻量隐式记忆思路，将历史表示压缩为共享的感知与语义/认知条目：

$$
\mathcal{M}_t = \{\mathcal{M}_t^{\mathrm{per}},\mathcal{M}_t^{\mathrm{cog}}\}.
$$

感知记忆保存压缩后的视觉表征，认知记忆保存当前多模态状态的低维语义摘要；两类记忆均由所有动作组共享。记忆容量固定，当容量达到上限时采用相似条目合并或替换策略，避免记忆规模随 episode 长度线性增长。

该设计的目的不是重新复现 MemoryVLA 的原始动作专家，而是复用其“压缩—检索—合并”的记忆机制，并把检索条件从统一当前状态扩展为动作子空间条件。

### 4.3 Action-Subspace-Conditioned Query

对动作子空间 $g$，定义动作组描述

$$
c_g = [e_g; k_g],
$$

其中 $e_g$ 为可学习动作组 embedding，$k_g$ 描述该组对应的运动学链、动作维数和控制语义。查询由当前多模态状态与动作组描述共同生成：

$$
q_t^{g,x} = W_q^x [h_t; c_g], \qquad x \in \{\mathrm{per},\mathrm{cog}\}.
$$

对每一类共享记忆执行交叉注意力检索：

$$
r_t^{g,x}
=
\mathrm{softmax}\left(\frac{q_t^{g,x}(K_t^x)^\top}{\sqrt d}\right)V_t^x,
$$

其中 $K_t^x$ 和 $V_t^x$ 分别由对应记忆库线性投影得到。随后融合感知、认知与全局记忆：

$$
z_t^g = \mathrm{Fuse}\big(r_t^{\mathrm{global}}, r_t^{g,\mathrm{per}}, r_t^{g,\mathrm{cog}}\big).
$$

所有动作组仍读取同一份历史，只是查询不同，因此不会为每个部件复制完整记忆库。

### 4.4 Memory Modulation of the Shared π0.5 Action Expert

本文保留 π0.5 的共享 Flow-Matching Action Expert 与流匹配训练目标，不使用 MemoryVLA 原始的独立 DiT Action Expert。动作专家产生的共享隐藏状态记为 $u_{t,\tau}$，其中 $\tau$ 表示 action chunk 中的动作时间位置。对动作组 $g$，使用轻量调制器将 $z_t^g$ 注入共享隐藏状态：

$$
\hat u_{t,\tau}^{g}
=
\mathrm{Mod}_g(u_{t,\tau}, z_t^g).
$$

$\mathrm{Mod}_g$ 可由 memory cross-attention、FiLM/AdaRMS 式缩放偏置或轻量 adapter 实现。随后由动作组读出头预测对应维度的 Flow-Matching velocity：

$$
v_{t,\tau}^{g}=W_{\mathrm{out}}^{g}\hat u_{t,\tau}^{g},
$$

联合输出为

$$
v_{t,\tau}=\mathrm{Concat}_{g=1}^{G}\left(v_{t,\tau}^{g}\right).
$$

所有 $\hat u_{t,\tau}^{g}$ 均由同一个动作专家主干产生，因此动作组之间仍共享当前视觉—语言状态和动作动力学表示；动作子空间条件化只发生在记忆读取与轻量调制接口处。

### 4.5 Training Objective

主训练目标保持 π0.5 的 Flow Matching 形式。设 $a^0$ 为真实动作 chunk，$a^1$ 为噪声端点，$a^\tau$ 为中间状态，目标速度为 $u^\tau$，则

$$
\mathcal{L}_{\mathrm{FM}}
=
\mathbb{E}\left[
\left\|
v_\theta(a^\tau,\tau,o_t,\ell,\mathcal{M}_t)-u^\tau
\right\|_2^2
\right].
$$

动作子空间检索模块、记忆调制器与动作组读出头均通过该动作损失端到端学习。信息瓶颈式目标用于解释设计原则，是否作为显式辅助损失加入训练由消融实验决定。

---

## 5. Experiments

### 5.1 Research Questions

实验围绕四个问题展开：

- **RQ1：** 在移动双臂长时任务中，π0.5 是否确实受益于显式历史记忆？
- **RQ2：** 在相同共享记忆和相同 token 预算下，动作子空间条件检索是否优于统一检索？
- **RQ3：** 全局记忆上下文是否有助于保留底盘与双臂之间的耦合？
- **RQ4：** 检索策略与 memory-to-action 注入方式是否相互独立，即 context、modulation 与 memory expert 是否呈现一致趋势？

### 5.2 Benchmark and Task Stratification

主要评测采用 BEHAVIOR-1K 2026 Challenge [6,11] 的移动操作设定，并在相同机器人、本体观测和 evaluator 下比较不同记忆方法。任务按照**动作子空间参与程度**而不是主观难度进行分层：

| 层级 | 动作特征 | 目的 |
|---|---|---|
| Low | 主要为局部机械臂操作，底盘参与较少 | 检查条件检索是否在低异构任务中退化 |
| Medium | 底盘移动与单/双臂操作连续切换 | 检验移动与操作的历史需求差异 |
| High | 长距离移动、多阶段操作、双臂协作 | 检验强异构和长时历史条件下的收益 |

### 5.3 Baselines

| 方法 | 记忆方式 | 检索是否依赖动作组 | 作用 |
|---|---|---:|---|
| π0.5 | 无显式历史记忆 | — | 基础下限 |
| π0.5 + MEM-short | 固定短时视觉/本体历史 | 否 | 强短时记忆基线 |
| π0.5 + Uniform Memory Bank | 与本文相同共享 bank | 否 | 核心统一检索对照 |
| π0.5 + Uniform Memory + Modulator | 与本文相同 bank 与调制接口 | 否 | 排除收益来自注入方式 |
| **Ours** | 共享 bank + global/group retrieval | **是** | 动作子空间条件检索 |

其中 Uniform Memory Bank 与本文共享相同的记忆写入、容量、token 预算和动作专家结构，仅移除 $g$ 条件，从而直接检验动作子空间条件化本身。

### 5.4 Metrics

主任务指标采用 benchmark 官方任务完成指标。除此之外，记录以下机制指标：

1. **Retrieval divergence**：不同动作组在同一时刻的记忆注意力分布差异，用于检验 H1。
2. **Counterfactual memory sensitivity**：遮蔽某类历史条目后，不同动作组输出变化的幅度。
3. **Memory efficiency**：在不同 bank 容量与检索 token 预算下的性能—开销曲线。
4. **Compute overhead**：显存、检索延迟与策略推理时间。

### 5.5 Ablations

| 消融 | 回答的问题 |
|---|---|
| Uniform query vs group-conditioned query | 动作条件检索是否必要 |
| Group-only vs global + group retrieval | 是否需要共享上下文维持动作耦合 |
| 去掉 $k_g$，仅保留 group embedding | 运动结构先验是否有额外价值 |
| 固定动作组 vs 不同粒度动作组 | 检索收益是否依赖人为划分粒度 |
| Prefix context vs modulation vs memory expert | 注入方式是否改变检索结论 |
| 不同 memory budget | 统一检索是否在有限预算下更早饱和 |

---

## 6. Discussion

本文不把“动作空间可以分解”作为创新。EchoVLA 已经采用 base–arm 分部件控制，通用机器人策略也长期面对异构动作维度。本文只讨论一个更窄的问题：**动作结构是否应该反向影响历史记忆的读取方式。**

该假设并不必然成立。共享 Transformer 可能已经能够从统一 memory representation 中自动忽略与当前动作无关的信息；若统一检索与动作条件检索在固定预算下没有稳定差异，则说明显式动作子空间条件化并非必要。相反，若增益主要出现在移动—操作频繁切换与跨动作组协同任务中，则可以支持“不同动作需求对应不同历史压缩”的解释。

本文也不主张三维地图不必要。对于需要持续全局空间推理、导航可达性或高精度空间一致性的任务，显式空间表示可能更合适。本文研究的是另一种设定：当策略主要需要从过去恢复少量与当前动作有关的证据时，能否通过共享轻量记忆和条件检索避免构建完整世界模型。

---

## 7. Conclusion

本文研究移动双臂 VLA 中的动作子空间条件记忆检索。与“先统一检索、再由不同控制部件使用同一记忆上下文”的方案不同，本文让动作子空间参与记忆查询本身：共享历史记忆保持统一，全局上下文维持跨部件耦合，不同动作组仅对其所需的历史证据进行条件化读取。该设计可以被视为对 POMDP 中历史充分表示的一种动作条件近似，其有效性最终取决于一个可直接验证的问题：**在固定记忆预算下，不同动作子空间是否确实需要不同的历史压缩。**

---

## References

1. Kaelbling, L. P., Littman, M. L., Cassandra, A. R. *Planning and Acting in Partially Observable Stochastic Domains.* Artificial Intelligence, 101(1–2), 1998.
2. *MEM: Multi-Scale Embodied Memory for Vision-Language-Action Models.* arXiv:2603.03596, 2026.
3. *MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation.* arXiv:2508.19236, 2025/2026.
4. *SERF: Spatiotemporal Environment and Robot Feature Map for Long-Horizon Mobile Manipulation.* arXiv:2606.12956, 2026.
5. *RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies.* arXiv:2603.04639, 2026.
6. BEHAVIOR-1K 2026 Challenge, Stanford Vision and Learning Lab.
7. π0.5 BEHAVIOR baseline implementation, `wensi-ai/openpi`, behavior branch.
8. *SOMA: Spatial Memory for Out-of-Vision Manipulation in Vision-Language-Action.* arXiv:2605.22283, 2026.
9. Lin, M., Liang, X., Lin, B., et al. *EchoVLA: Robotic Vision-Language-Action Model with Synergistic Declarative Memory for Mobile Manipulation.* arXiv:2511.18112, 2025; v3, 2026.
10. Black, K., Brown, N., Darpinian, J., et al. *π0.5: a Vision-Language-Action Model with Open-World Generalization.* Physical Intelligence, 2025.
11. *BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation.* arXiv:2403.09227.

---

# Supplementary Material

## S1. Draft Status and Scope Notes

- 本文当前为方法与实验设计稿，不包含实测结果；正文中的 H1/H2、RQ1–RQ4 均为待验证假设，而非已有结论。
- 本文的创新边界限定为 **action-subspace-conditioned memory retrieval**。不把“VLA 需要记忆”“移动操作需要长期历史”“动作空间可以分解”“base/arm 使用不同控制头”作为创新。
- 记忆构建默认借鉴 MemoryVLA/PCMB 的轻量记忆思路，论文变量尽量集中在“统一查询 vs 动作子空间条件查询”。

## S2. EchoVLA and π0.5 Interpretation Notes

### S2.1 EchoVLA

EchoVLA [9] 的论文原文要点应按以下方式理解：

1. EchoVLA 维护 **scene memory** 与 **episodic memory** 两类互补记忆。
2. 两类记忆在训练与推理过程中分别存储、更新和检索；检索依赖当前观测、任务历史和语言指令。
3. 检索后的表示通过 coarse- and fine-grained attention 融合，并用于引导 mobile/base–arm diffusion control。
4. 因此 EchoVLA 已经属于 **per-part control**，但其论文并未把“动作子空间 ID/运动学子树”作为记忆检索 query 的条件，也没有为 base、left arm、right arm 分别从同一共享 bank 生成独立 action-conditioned retrieval result。

本文与 EchoVLA 的边界应统一写成：

```text
EchoVLA:
scene / episodic memory
        ↓ retrieval by observation / task context
   fused memory context
        ↓
   per-part base–arm control

Ours:
shared memory
        ↓
action-subspace-conditioned queries
        ↓
global + group-specific memory contexts
        ↓
shared flow-matching action expert
```

不要再把“per-part action generation”本身写成本论文创新。

### S2.2 π0.5

原稿曾写“π0.5 introduces per-part decomposition for mobile base and manipulator”。该说法容易与 EchoVLA 的独立 base/arm diffusion policy 混淆，因此正文已删除。

按 Physical Intelligence 官方 π0.5 论文与公开 openpi 架构理解，π0.5 的核心是跨本体/异构数据共训练、VLM prefix 与 Flow-Matching Action Expert，而不应在本文中把“EchoVLA 式独立 base/arm denoiser”当作 π0.5 官方结构前提。

因此本文只依赖以下事实：π0.5 可以输出移动操作机器人的联合动作，并具有共享 Flow-Matching Action Expert；动作子空间划分是本文用于记忆检索与轻量调制的接口，而不是对 π0.5 官方动作头结构的复述。

## S3. Implementation References

### S3.1 MEM on π0.5

优先检查以下开源实现：

- Hugging Face LeRobot: `src/lerobot/policies/pi05/memory.py`
- Hugging Face LeRobot: `src/lerobot/policies/pi05/modeling_pi05.py`
- `heliang-pu/vla-mem`: 同时提供 LeRobot/PyTorch 与 openpi/JAX 风格实现

这些实现可用于参考“增加历史记忆但保留 π0.5 Flow Matching Action Expert”的数据管线与模型接口。

### S3.2 MemoryVLA

官方代码：`github.com/shihao1895/MemoryVLA`。

建议复用：

- perceptual / cognitive memory bank；
- retrieval；
- gate fusion；
- memory consolidation / merge；
- episode-level memory reset。

不建议直接复用其原始 DiT Action Expert。本文将 MemoryVLA 的记忆机制与 π0.5 的 Flow-Matching Action Expert 解耦。

### S3.3 RoboMME / MME-VLA-Suite

代码：`github.com/RoboMME/robomme_policy_learning`。

重点参考三类 memory-to-action 接口：

- memory as context；
- memory as modulator；
- memory as expert。

本文主方案优先采用 **modulation**，context 与 memory expert 作为消融。

## S4. Engineering Notes for π0.5 Integration

1. **Memory-off parity**：关闭 memory 后，模型输入输出路径应尽可能与基础 π0.5 保持一致，作为第一项代码验收。
2. **Shared expert**：不为 base/left/right 复制完整 Flow-Matching Transformer；动作组差异只通过 query、modulator 和轻量 readout 产生。
3. **Group descriptor**：第一版可只使用可学习 group embedding；若有效，再加入来自 TF tree / URDF / action specification 的运动结构描述 $k_g$。
4. **Action grouping**：不固定死“底盘/头/左臂/右臂”。每个 embodiment 根据其 action specification 定义 $G$ 个 action groups。
5. **Cross-group coupling**：必须保留 global retrieval 或共享 expert trunk，避免把左右臂和底盘错误地视为条件独立。
6. **History batching**：数据加载必须 episode-aware，历史帧、历史 state 和 memory reset 不能跨 episode。
7. **KV cache**：π0.5 推理时 prefix KV cache 的具体生命周期需要按所用 openpi/LeRobot 分支再次核对；不要把缓存本身当作跨控制步长期记忆。

## S5. Experimental To-Do List

- 从 BEHAVIOR-1K 2026 官方任务中确定 Low / Medium / High 三档任务，不根据任务名称臆造 ID。
- 统一 baseline 的训练步数、batch、冻结策略、输入相机、动作维数和 memory token budget。
- 首先跑：`π0.5 → π0.5 + MEM-short → π0.5 + Uniform Bank → Ours`。
- 若 Uniform Bank 已经明显优于 MEM-short，再继续比较 action-conditioned retrieval。
- 若动作条件检索没有稳定提升，不继续扩展固定四角色专家；优先检查动作组定义是否与模型实际的动作依赖结构一致。
- 注入方式消融必须与检索方式分开：不要同时改变 memory construction、retrieval 和 action expert，否则无法归因。

## S6. Additional Method Comparison

| 方法 | 历史表示 | 读取/检索条件 | 动作端 | 与本文的主要区别 |
|---|---|---|---|---|
| MEM | 短时视觉 + proprio history | 时间窗口，无显式动作组 query | π0.5 Flow Matching | 固定短时历史，不做长期 action-conditioned retrieval |
| MemoryVLA | perceptual + cognitive bank | 当前状态统一查询 | 专用 DiT action expert | 有 bank retrieval，但 query 不按动作子空间区分 |
| SOMA | 空间—语义记忆 | 指令/当前上下文 | VLA policy | 重点是 out-of-vision 空间检索，不按动作组查询 |
| EchoVLA | scene + episodic memory | 当前观测/任务历史/指令 | per-part base–arm diffusion | 分部件发生在控制端，不是 action-subspace-conditioned retrieval |
| SERF | 时空环境/机器人特征地图 | 位姿与当前状态 | VLA policy | 重点是显式时空地图 |
| Ours | 共享轻量长期 memory | **current state + action-subspace descriptor** | shared π0.5 expert + lightweight modulation | 检索本身由动作子空间条件化 |

## S7. Terminology

- **Action subspace / action group（动作子空间 / 动作组）**：根据机器人运动学链和控制语义划分的一组动作维度，不等同于简单的连续维度切片。
- **Action-subspace-conditioned retrieval（动作子空间条件检索）**：查询由当前状态与动作组描述共同生成，使不同动作组可从同一共享记忆中读取不同历史证据。
- **Global retrieval（全局检索）**：不依赖动作组的共享历史上下文，用于维持底盘、躯干与双臂之间的耦合。
- **Action-sufficient memory（动作充分记忆）**：不要求恢复完整世界，只要求保留预测对应动作所需的历史信息。
- **Memory modulation（记忆调制）**：使用检索到的 memory context 调节共享 Action Expert 的隐藏状态，而不是替换原 Flow-Matching 动作生成机制。

## S8. Source and Repository Notes

- MEM code reference: `github.com/heliang-pu/vla-mem`；LeRobot 中另有 π0.5 memory implementation。
- MemoryVLA code: `github.com/shihao1895/MemoryVLA`。
- RoboMME / MME-VLA-Suite code: `github.com/RoboMME/robomme_policy_learning`。
- SERF code: `github.com/ExistentialRobotics/SERF-mapping` 及相关 SERF-VLA repository。
- BEHAVIOR-1K challenge: `behavior.stanford.edu/challenge`。
- π0.5 BEHAVIOR baseline: `github.com/wensi-ai/openpi`, branch `behavior`, config `pi05_b1k`（使用前再次核对当前分支与配置名称）。
- 原稿记录称相关 PDF 已存于项目 `papers/` 目录；该本地目录状态需要在实际项目环境中再次确认。
