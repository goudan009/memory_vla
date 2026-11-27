# V1基线开跑前：MemoryVLA论文、作者源码与当前实现审计

> 结论：当前V1的`openvla`路径复现了作者公开默认记忆算子，并作了GR00T条件接入适配；不能声称与论文全文及训练配方完全一致。此次仅完成审计，未启动新训练，未修改生产模型、训练入口或旧实验任务书。
> 代码起点：`3397963`；审计计划：`workflow/plan-v002.json`（提交`088ab3e`）。2026-11-24，4090工作台发起，A100输入与资源只读核查。

## 本次用户要求与旧任务区别

用户新要求：V1加记忆的GR00T，从官方纯净通用基座开始，在50任务10000条示教上训练，参数对照MemoryVLA论文；第一阶段20000 optimizer steps，每2000步完整存档，之后可以持续延长预算严格续训；直接四卡同步运行，不排队。

旧Exp13任务书使用B1K-50已微调基座、Batch128并依赖旧Exp12总验收。它不等于本次要求。本次不采用Exp12 Task40权重或旧Residual权重，也不把旧报告的未更新验收标记当成本次代码审计结论。待用户确认论文/作者实现分歧和必要GR00T适配后，另行更新任务书及执行计划。

## 证据来源

- [MemoryVLA论文](https://shihao1895.github.io/MemoryVLA/static/files/MemoryVLA_paper.pdf)，本地`related_work/MemoryVLA/MemoryVLA_2508.19236.pdf`与LaTeX方法及附录：方法§3、附录C.1/C.3/C.4、超参数表。
- 作者`openvla-codebase`锁定提交[d732ea9072bc063399ccc817aed74ab172eb50be](https://github.com/shihao1895/MemoryVLA/tree/d732ea9072bc063399ccc817aed74ab172eb50be)，不是MemoryVLA+分支。已从作者仓库重新读取并逐字节确认本地以下五份参考文件一致：`vla/memory_vla.py`、`conf/vla.py`、`train.py`、`vla/datasets/datasets.py`、`script/train/libero/train_libero_spatial.sh`。
- 当前生产路径：`model/memory/bank.py`、`adapter.py`；`model/gr00t_n1d6/gr00t_n1d6.py`、`model/modules/dit.py`；`data/dataset/history_dataset.py`、`episode_windows.py`；`experiment/task21_ddp.py`、`task21_formal.py`和`train/dashboard_remote.py`。
- [官方通用基座配置](https://huggingface.co/nvidia/GR00T-N1.6-3B/blob/main/config.json)与[官方B1K微调模型](https://huggingface.co/nvidia/GR00T-N1.6-BEHAVIOR1k)。工作台`GET /api/training/options?device=a100`及A100 SSH只读核查。

## 方法一致性

| 内容 | 论文/作者源码 | 当前V1实际行为 | 判断 |
| --- | --- | --- | --- |
| 双路输入 | 视觉网格经SE压缩；多模态EOS认知token | Eagle投影前网格按三相机分别SE压缩，用户EOS认知token | 方法对应，编码器/相机/维数为GR00T适配 |
| 检索 | 两层cross-attention；时间编码仅加K；V为记忆原值 | `openvla`相同，无额外Top-K；训练要求QK norm为none | 对应 |
| Gate | 两输入逐通道sigmoid融合 | 对应作者代码`g*current+(1-g)*retrieved` | 论文将g写在retrieved上，作者代码写在current上；属于门的参数化约定，不能用g大小跨定义比较 |
| 写入 | 论文§3/Fig3写融合后的token；作者`update_fused=False`默认写当前原始token | `adapter.py`对openvla硬编码`update_fused=False`，初始化也设false | 符合锁定作者默认，**不符合论文融合后写入描述**。不能仅改config中`memory_update_fused`字段就宣称已切换，因为adapter未读取该开关 |
| 容量与合并 | 常规任务容量16；最相似相邻槽平均，保留前槽时间戳，历史detach | 相同；感知256通道、两层检索、容量16 | 对应；其他旧分支仍可用64槽/Residual，不能靠V1名称判断实际模型 |
| 动作条件 | 感知供视觉cross-attention；认知投影后与时间嵌入结合并前置到动作序列 | 感知投影供原生视觉cross-attention；认知前置进入原生自注意力，保留原生语言cross-attention和本体状态 | 信息流对应，但采用GR00T预训练动作专家，**不是原论文DiT逐层同构** |
| 动作目标 | OpenVLA/Prismatic、DDIM扩散、16步动作、训练重复4噪声、推理10步/CFG1.5 | GR00T Eagle/Flow Matching、原生horizon50、训练每决策单次噪声、默认推理4步 | GR00T平台适配差异，不能把这部分写成“全部参数与论文一样” |

旧Residual/value/QK-LayerNorm路线和route_a/route_b研究分支仍保留以兼容历史权重；本次基线应明确选择哪条新训练路径，不继承这些训练参数。

## 训练配方差异

| 项目 | 论文或锁定作者源码 | 当前已验证的openvla入口 | 本次拟定 |
| --- | --- | --- | --- |
| 全局有效Batch | 256；作者8卡×32 | 固定4卡×micro4×累积8=128 | 若按用户“跟论文一样”，用4卡×micro4×累积16=256；16个完整16帧组，每rank4组 |
| LR/optimizer | 2e-5；作者AdamW、WD0、constant、0 warmup，默认betas(0.9,0.999)/eps1e-8；clip1.0 | 上述设置一致，Full FT、BF16、FP32归约、FULL_SHARD | 保持；不LoRA、不冻结视觉/LLM/动作专家 |
| 历史采样 | LIBERO：同一回合随机取16帧，再按时间排序，组间清零 | 固定stride8连续16决策，窗口起点顺序固定、组索引打乱；排除不完整尾组 | 与论文LIBERO分组采样对齐；B1K时间单位/候选决策帧与在线步长的映射必须显式记录 |
| 图像增强 | 作者random crop面积0.9到224、brightness0.2、contrast/saturation0.8–1.2、hue0.05 | 沿用GR00T处理器crop0.95、shortest_edge256、jitter各0.1 | 若严格对齐增强数值须修改处理器并验证Eagle网格；保留三相机与GR00T必要预处理差异，不能假称目前一致 |
| 训练集 | 本次用户明确50任务10000条 | 既有逐任务准备记录180训练/20验证 | 所有50×200示教参与训练，不默默变9000/1000；无需额外状态标签，只用原示教观测、指令和动作 |
| 预算 | 论文不同任务不同预算；用户指定20000 | 工作台openvla只接受2或6000 | 首轮20000，累计5,120,000个决策样本；这不是5,120,000条示教，也不能直接除以10000声称数据遍历轮次 |
| 存档/续长 | 用户每2000步完整存档，后续续跑 | 每2000存档已有；但原协议到预算不允许再续长 | 首轮10个存档，支持20000之后增加预算，保持run ID、optimizer、scheduler、各rank RNG与采样位置连续 |

当前实现**尚未满足新任务**的工程位置：

1. `train/dashboard_remote.py:316`限制`tasks=[40]`、B1K专用checkpoint、8组/128样本、steps2或6000；run ID也只接受exp12。
2. `experiment/task21_ddp.py:382`再次限制4rank与8组；多处累积日志硬编码32/microbatch，不能只放开网页表单就当Batch256有效。
3. `experiment/task21_ddp.py:70`的`resume_manifest`禁止原openvla协议增加总预算；`dashboard_remote.py`的resume_info亦把到预算视为不可恢复。
4. `train/run_identity.py`的规范命名解析只覆盖exp12，需要让exp13新run及其checkpoint可查、可暂停、可续长。

修改应新增本次清晰协议/配置，不覆盖Exp12冻结源码和旧权重行为。新方案先做真实batch四卡前后向、模块更新及保存重载预检，再经现有工作台preview与资源核对启动。新run要实测工作台暂停→释放GPU→同run严格恢复，不能用“权重加载成功”代替续训验收。

## 基座与数据核对

纯净起点位于A100：`/nvme3/huggingface/train_model/GR00T/GR00T-N1.6-3B`，下载来源记录`nvidia/GR00T-N1.6-3B`，revision `d0814e7ecb19202e7c8468b46098b0b7ef3a6d61`。两个分片实际SHA256与来源记录一致：

- `model-00001-of-00002.safetensors`：`9710d31d331b79cfa229fd23605ba9ad47e207cb0f4c7722fe0bacdc666c8326`
- `model-00002-of-00002.safetensors`：`eb79bbd7893068897cda18c86ed91064e92e20e356dea97bc398bf9c8bf2fa35`

该基座与B1K专用权重均为AlternateVLDiT、horizon50，纯净基座从结构配置上兼容当前接入方式；**本次尚未执行纯净基座真实GPU前向验收**。继承其Eagle、投影层、动作专家等通用参数，新建双记忆、SE压缩及必要条件投影；不继承Exp12/旧V1记忆。这里“纯净”指官方通用权重身份，不推断官方预训练数据完全不含B1K。

原始数据核对：50任务，每任务200份Parquet，共10000条；metadata同为10000条、119094660帧。对全部10000文件只读检查文件头/尾，发现Task26四份footer损坏：

- `episode_00261770.parquet`（9437184 bytes）
- `episode_00261780.parquet`（36700160 bytes）
- `episode_00261790.parquet`（29360128 bytes）
- `episode_00261820.parquet`（3145728 bytes）

原始路径为`assets/datasets/b1k2025/data/task-0026/`，链接到共享SERF-VLA数据；本次未覆盖共享文件。工作台实查49/50准备就绪、Task26失败，与四份损坏相符。不能把文件数10000当作10000份可读取训练数据，更不能丢弃Task26后称全50任务。需从官方来源恢复四份完整数据到本实验可控副本并完成原转换流程。本次没有全量解码视频或逐行审查所有Parquet，其余数据完整性仍由准备流程和真实batch验收约束。

A100实查为4×81920 MiB，UUID见当前工作台；GPU0已有9617 MiB作业、GPU3已有7633 MiB推理服务，GPU1/2各32 MiB。不能停止已有任务；按新配置峰值核对能共存后四卡直接运行，不等全机空闲。远端磁盘当前可用约471 GiB；10份完整Full FT存档需按真实预检大小核算，不自动删除旧存档。

## 已执行验证及结论边界

- 9项现有相关回归通过：`src/v1_groot_memoryvla/tests/test_memory_v1.py`全部8项，以及`tests/test_exp10_fusion_model.py::test_openvla_independent_conditions_train_through_native_attention`。覆盖因果读取、分组/在线一致、梯度、合并、会话隔离、重载和两路条件到动作头的梯度。
- 使用锁定作者实际类、相同权重、CPU FP32、20次写入与溢出合并对照：`update_fused=False`最大输出绝对误差`1.1921e-7`；`True`最大`4.7684e-7`；两组均16槽、保留时间戳相同。当前openvla adapter实际两库为False。这证明共享算子对应，不证明采样、训练配方或整网与论文相同。
- 本次未执行新训练、未做新仿真效果评估，也没有把内部算子检查当作记忆收益证据。

建议用户确认的执行方案：以论文为准对齐融合后写入、同回合随机有序16帧、Batch256与可迁移优化参数；保留GR00T编码器/三相机/原生动作头与Flow Matching必要适配，明确列出horizon50和噪声训练/推理差异，不称逐参数复刻OpenVLA。先修Task26四份数据与工作台/续长限制，预检通过后四卡直接训练20000步，每2000步完整存档；本轮不额外启动无记忆模型或未指定回合的仿真。

若目标是精确复刻锁定作者默认实现而非论文文字，则应保留原始特征写入，报告这一分歧；不能静默把两种选择混为同一基线。

## 后续审阅纠正与用户选择（保留前文审计历史）

作者发布的LIBERO Spatial和Mikasa checkpoint配置均使用`update_fused=false`。据此修正前文偏向融合写入的建议，推荐作者实际发布配置的原始特征写入。用户随后明确要求按推荐修改代码：原始特征写入、同回合随机有序group16；准备完成后列明全部信息，等待确认启动。

方法与入口修改、125项CPU回归、真实preview的Task26阻塞及完整配方见[启动前配置审阅](training_review_20261124.md)。图像增强来源也需纠正：纯净基座实际加载的processor颜色扰动为0.3/0.4/0.5/0.08，而不是模型config中的各0.1。此次仍未执行GPU训练或启动排队任务。
