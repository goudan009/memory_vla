# v1：GR00T N1.6 + MemoryVLA

认知检索饱和修复、真实特征／动作快测、LIBERO 环境和同初始化对照结果见 [快速迭代与验收](audit/QUICK_ITERATION.md)（2026-11-12）。从原始 GR00T 新建训练默认启用认知 Q/K 归一化；旧 Memory checkpoint 保持原配置。

## 训练模式与组件冻结（2026-11-12）

训练网页和唯一入口 `train/run_groot_train.sh` 均已支持以下模式。刷新网页，在“微调方式”中选择，在“组件冻结与 LoRA 参数”中展开细项；每个组件的“跟随模式／冻结／训练”会传入实际训练、独立验证和断点恢复。既有运行仍使用各自固定源码和参数。

| `--fine-tuning-mode` | 默认训练范围 |
| --- | --- |
| `action_full`（默认） | 冻结整个 Eagle；全参数更新动作专家，Memory 组额外更新记忆模块。兼容之前的训练方式。 |
| `full` | 全模型微调：LLM、视觉编码器、视觉投影、动作专家及启用的记忆模块。 |
| `custom` | 默认同 `action_full`，用组件开关指定训练范围。 |
| `lora` | 默认仅为 LLM 和动作 DiT 的 Linear 层加入 LoRA；它们的原始权重冻结，状态／动作投影、LayerNorm 和启用的记忆模块默认全参数训练，视觉编码器和视觉投影默认冻结。 |

组件开关优先于模式默认值；`--freeze-xxx` 冻结，`--no-freeze-xxx` 解冻。LoRA 模式下解冻目标组件表示训练该组件的低秩参数；解冻非目标组件表示全参数训练该组件。Baseline 没有记忆参数。

```bash
# 在原有完整训练命令后选择一组参数；使用 --dry-run 可以先检查命令。

# 整个模型全参数微调
--fine-tuning-mode full --microbatch 1 --gradient-checkpointing

# LoRA：LLM + 动作 DiT，冻结状态/动作投影和 LayerNorm
--fine-tuning-mode lora --lora-targets llm,action_dit \
  --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05 \
  --freeze-action-projector --freeze-action-vlln \
  --microbatch 1 --gradient-checkpointing

# 自定义：只解冻 LLM 最后 4 层和视觉投影；原动作专家保持训练
--fine-tuning-mode custom --no-freeze-llm --llm-last-n-layers 4 \
  --no-freeze-vision-projector --freeze-vision --microbatch 1
```

常用新增参数：

```text
--freeze-llm / --no-freeze-llm                              LLM
--freeze-vision / --no-freeze-vision                        视觉编码器
--freeze-vision-projector / --no-freeze-vision-projector    视觉投影 mlp1
--freeze-action-dit / --no-freeze-action-dit                动作 DiT
--freeze-action-projector / --no-freeze-action-projector    状态编码、动作编码/解码及位置编码
--freeze-action-vlln / --no-freeze-action-vlln              动作输入 LayerNorm
--freeze-memory / --no-freeze-memory                        整个记忆模块
--lora-targets llm,action_dit                               可选 llm,vision,vision_projector,action_dit
--lora-rank 16 --lora-alpha 32 --lora-dropout 0              默认 LoRA 参数
--llm-last-n-layers 0                                       0 为全部；N>0 只训练末 N 层，Embedding/末层Norm冻结
--gradient-checkpointing                                   对可训练的 LLM/视觉编码器进行激活重算
--weight-decay 0.01                                        AdamW weight decay
--max-grad-norm 1.0                                        梯度裁剪阈值
```

LoRA 目标组件内的所有 Linear 层都会适配（不含 LLM 输出 lm_head），不是只适配 Q/V；状态/动作投影使用带本体索引的自定义权重，不套用 Linear LoRA，继续支持全参数训练或冻结。`llm-last-n-layers` 作用于 Eagle 实际保留的层（当前16层）。学习率、优化步、全局窗口数、microbatch、随机种子、GPU 和验证配置仍沿用原参数。

可训练 Eagle 现在在 DDP 的微批次计算图内前向及反传；只有完全冻结时才缓存无梯度特征。可训练参数保留 FP32，前向使用 BF16 autocast，冻结的 Eagle 参数使用 BF16。加载训练权重时先恢复训练范围，再转换冻结参数，避免 FP32 主权重在续训及验证前被舍入。32 决策窗口、同 Episode 跨窗记忆及历史 detach 语义不变。DDP 仍是每卡完整模型，不合并显存；全模型微调应从 microbatch=1 检查实际显存，当前未在占用中的 A100 上实测全模型峰值。

每次训练的 `<run>/<arm>/config/finetuning.json` 记录实际各组件总参数/可训练参数和 LoRA 模块路径，`config/ddp.json` 保留全部训练选项。checkpoint 保存完整模型及 `config.json` 中的 LoRA 参数和模块路径，验证及当前版本推理入口可自动重建适配器；不是仅保存低秩增量。断点恢复要求冻结、LoRA、weight decay、裁剪设置与原运行一致。

验收：155 项训练/网页/恢复回归通过，包含两进程 DDP 的骨干全参数及 LoRA 梯度同步、真实动作头与记忆模块的 HF 保存/重载、FP32 加载精度，以及网页参数提交。真实 GR00T 通用权重 + Task0 Episode10 帧0，在本机空闲 GPU0 做一次独立 LoRA 单批优化：四个可选目标组件均有非零梯度，LLM adapter 确实更新，loss=0.00726964，峰值11.30 GiB。该检查使用 rank4、LLM末1层、冻结动作投影/LayerNorm及梯度检查点，不能用来推算默认 LoRA 或全模型显存，也不代表训练效果。报告在 `outputs/shared/training_dashboard/verification/finetuning/real_b1k_lora_smoke.json`；A100 已通过全量模式脚本 dry-run，网页三种新模式的真实预检通过，未提交正式训练、改权重或启停原训练/评估服务。

## 评估速度调查（2026-11-11 16:54）

本次检查针对旧 Task40 step3000 checkpoint，使用其固定部署 `deployments/V_1_task40_mem`（残差回注、容量16），未使用下文开发版 direct/64。只读检查闭环结果，并在起初空闲的 A100 GPU1 新建独立离线进程，输入真实 Task40 Episode400010 的帧0/8、双样本、三路256×256图像；未调用在线服务的动作或 Reset，未改网络、权重、服务及训练。

- 满历史并持续合并时，分段计时的记忆 Adapter 平均10.10ms/双样本请求，Eagle35.89ms、动作头55.61ms；计时包含显式CUDA同步。感知特征 `[2,3,81,2048]`、主序列 `[2,283,2048]`，感知检索每样本243个Query、满库3888个Key；认知为1个Query和16个Key。两个连续60决策试验均保持每环境容量16，最终各44次合并，没有历史无限增长。逐样本读写、逐对相似度和 `int(argmax)` 的CPU/GPU同步有额外开销，本轮未优化。
- 首轮测试卡无其他训练时，独立Baseline均值117.24ms、Memory两个连续轮次136.26/138.49ms；但同一Memory模型旁路记忆也为139.32/140.48ms，端到端波动掩盖差值。不能拿这些整段均值声称记忆使推理变慢17%，更不能将历史RPC的2.5倍全部归因于记忆。
- 历史Task34各10回合、同实例/种子、各47810动作步：对照组RPC173ms、记忆组432ms；仿真vector step也从826ms升至1220ms，和模型前向分开计时。约2176秒总耗时差中，累计RPC多774秒、vector step多1177秒，其余含初始化/重置。对照policy_identity为unlinked，两组非同时测量，不能据此完成严格checkpoint因果归因。
- 当前Task40按实例/种子匹配前6个已结束回合，两组均38856动作步，记忆平均1821.70秒（30.36分钟）、对照1856.12秒（30.94分钟），未见已完成部分变慢；这12条记录全部timeout。Task40单回合6476步、Task34为4781步，不能跨任务直接比较总用时。Task40记忆评估仍运行，`policy_get_action.count=0`表示主进程尚未结束写入计时，不代表零延迟。

调查期间出现新共卡占用：A100于16:48:49启动 `V_1_baseline_task0_34_40_train02` 四卡训练，rank0 PID2569006与5560记忆服务PID3007839共用GPU0；16:54快照GPU0利用率94%、约43.5GB显存。后半段paced/model-paired/threads1测试与训练重叠，均受干扰，不能用于宣称纯模型延迟或CPU线程优化收益。默认112个Torch CPU线程改为1的测试未消除波动，线程数不是已证实的主因。所有诊断进程已退出，未停止其他进程。后续公平比较应让训练避开推理卡，固定任务、实例、种子、录像及仿真设置。

脚本、原始JSON及逐Episode快照归档于 `outputs/shared/diagnostics/memory_latency_20261111/`；A100同路径保留独立测试报告。`investigation.json`标记受共卡影响的后半段试验。仅更新README与handoff，未新增Markdown。

## 本轮方法与训练修复（2026-11-11）

新运行默认使用 `direct` 双记忆动作条件、容量 64 和训练图像增强；保留 GR00T backbone、原动作头和 flow matching。已修复 Episode 连续窗口调度、尾窗丢失、越界动作标签、不同长度窗口的 DDP loss 权重，以及空/错误历史验证的接口一致性。网页与正式启动入口均可配置容量和增强。完整说明及重训参数见 [RETRAIN.md](RETRAIN.md)。本轮仅修改源码，未启动训练或重启服务；旧运行及其固定快照保持原有协议。以下历史记录中的残差接入、固定容量 16 和每窗必为 32 决策不再描述新运行。

## 训练页：多任务、排队、可选验证与预训练起点（2026-11-11）

刷新工作台8765端口，进入“训练 → 新建训练”：

1. **任务多选**：搜索并勾选多个任务，如0、34、40；已选任务可单独移除，无需按住Ctrl。它们共同训练一个模型，按任务均衡循环；不是分别启动三个单任务模型。所选任务须已转换并在A100就绪，未就绪项可用数据准备按钮提交。
2. **排队启动**：勾选后，后台等待所选训练GPU及启用的验证GPU空闲，同一GPU资源按提交顺序使用；不同GPU可独立启动。关闭浏览器仍继续排队。高级参数中的显式共卡许可仍有效；默认不共卡，不停止已有服务。排队任务可在详情取消，取消后从列表消失。
3. **初始权重**：新增官方`GR00T-N1.6-3B`通用预训练起点，A100路径`/nvme3/huggingface/train_model/GR00T/GR00T-N1.6-3B`；原`GR00T-N1.6-BEHAVIOR1k`明确标为“已经B1K微调”。基础版不是本项目训练结果；[官方模型说明](https://huggingface.co/nvidia/GR00T-N1.6-3B)没有保证其预训练完全未使用B1K，因此不标为“从未见过B1K”。两份官方权重都保持原样。
4. **是否验证**：默认开启。取消勾选后隐藏、禁用验证GPU和比例输入，200条/任务全部用于训练，不启动验证进程；仍每1000优化步及结束时保存模型、优化器和记忆状态。开启时选择验证GPU和每任务验证比例1–50%，其余作为训练集；按完整Episode隔离，不能按帧混分。默认10%即180/20，默认seed42保留既有划分，改变比例或seed后确定性重划并记录到本次运行配置。

新建训练的“实验设置”可填写可选备注（最多2000字），记录实验目的、改动和对比对象；保存于本次运行的 `config/web_launch.json`，在左侧列表和详情顶部显示，并支持按备注搜索。选择任意历史训练，进入“参数与日志”查看当次学习率、全局窗口Batch、最多决策数、每GPU Microbatch、GPU、seed、数据划分、验证和记忆配置；优先读取实际 worker 配置，缺失项标为“未记录”，不使用新建表单的默认值。原始配置和权重起点仍可展开查看。续跑步数与确认按钮、指定停止步数与按钮统一对齐，桌面并排，窄屏上下排列。

训练详情显示累计优化循环耗时、最近50个有效计时步的平均耗时、最近一步耗时，以及剩余优化步数乘最近均值的预计剩余时间。计时来自主rank逐步日志的`seconds`（各rank取最大值），不乘GPU数量；不包含初始化、保存权重、排队及独立验证。缺失历史计时显示“—”，不会按文件修改时间推算；已完成训练的剩余时间为0。停止点会更新剩余优化步数。

网页默认开启“自动推荐参数”，根据任务实际轨迹长度、验证划分、训练GPU数量及显存、GR00T初始权重和方法自动填表，并显示推荐依据。Task0/34/40、10%验证、通用GR00T起点、两张80GiB训练卡在默认 action_full 下推荐18000步、lr=1e-5、全局4窗口（最多128决策/步）、Microbatch4、direct/64、开启增强；三张卡推荐24000步及全局3窗口。完整验证每3000步，保存每1000步，推荐关闭自动早停。新加入的 full/custom/lora 默认推荐 microbatch=1，full 从通用权重起步的推荐学习率为2e-6；组件冻结和LoRA参数见本页首节。手动修改推荐字段会切换为手动模式，刷新不会覆盖；点击“恢复推荐”重新填入。Baseline和Memory对比须固定起点、划分、种子和全局Batch。推荐是首轮起点，不会修改已有运行，也不会自动提交训练。旧推荐规则背景见 [RETRAIN.md](RETRAIN.md#网页自动推荐参数)，新增微调模式以本页为准。

例：选择0/34/40、验证20%，得到480训练/120验证Episode；关闭验证得到600/0。各任务划分、Episode→任务映射及种子保存在运行的`config/data_split.json`；联合数据在该运行`dataset/`下合并少量元数据，以软链接复用原始已转换视频和parquet。命名仍为`V_1_baseline_task0_34_40_trainNN`或`V_1_memory_task0_34_40_trainNN`。32决策窗口、跨窗口detach记忆、Eagle冻结、原动作专家和记忆结构不变。不验证时没有验证曲线或“最佳验证”权重，已保存的各步checkpoint仍可送去推理评估。

网页继续调用唯一入口`train/run_groot_train.sh`；新增`train/web_training_queue.py`仅负责后台排队，不执行模型优化。命令行新增`--no-validation`用于正常保存而不验证，与不保存模型的短检查`--train-only`不同。已有运行使用固定源码快照，不受网页/开发代码同步影响。

基础权重版本为`d0814e7ecb19202e7c8468b46098b0b7ef3a6d61`，下载来源及各文件SHA记录在权重目录`download_provenance.json`，本机副本为`checkpoints/groot_n1d6_base/`。A100核查1106项参数名称/形状兼容、B1K模态配置和归一化统计可用；未为此启动GPU训练。78项网页/推理/数据/评估界面测试及22项训练采样、划分、DDP与恢复测试通过；A100两种验证设置的真实数据合并和统一入口dry-run通过。下载日志位于`outputs/shared/training_dashboard/base_checkpoint/download.log`，配置与数据合并检查位于A100项目的`outputs/shared/training_dashboard/verification/multitask_web/`。本轮仅提供新启动能力，未自动重新提交已取消的联合训练，也未停现有训练或评估。

## 网页取消排队与停止训练（2026-11-11 10:35）

打开工作台8765端口的训练页，在左侧选择运行，进入 **权重与任务控制**：

- 等待前序训练完成：点击“取消本组排队”。
- 等待GPU释放：点击“取消 GPU 等待及本组排队”，撤销任务，不会跳过GPU占用检查强行启动。
- 已开始训练：填写未来的1000整数倍停止步数，点击“设置停止点”；保存后停止，已排队的验证继续。

当前Task0/34/40联合Baseline与Memory由同一队列成对启动，所以从任意一组点击取消，都会撤销页面明确列出的两组。仅在两组都尚未启动时允许取消；如果点击时训练已经启动，会拒绝取消并提示刷新。取消后两组立即从页面列表和数量统计中移除，并返回新建训练页；后台保留配置、日志、源码快照及“已取消”状态，不会再自动启动。不会停止Task40正在运行的训练、推理服务或评估，也不会取消独立的数据准备队列。取消后的联合训练需要重新安排，单任务可从“新建训练”重新提交。

本次只是增加控制入口，**没有撤销当前任何真实任务**。42项训练网页、数据登记及推理网页测试通过，包含临时真实进程取消、启动竞态拒绝与其他进程存活检查。8765后端更新为PID3349026，原评估PID3267977的启动时间及记录保持不变；其他账号的8766面板未操作。

## 数据就绪状态和后台补齐（2026-11-11 10:17）

网页列出50个任务，之前只有Task0/21/34/40已转换并上传A100，其余46项的“待转换 / 上传”指的是训练格式数据，原始示教本机已经齐全。已逐文件核查这四个任务各200个parquet、600个256×256三路视频，以及180/20划分、30Hz和82/23维状态动作；补齐早期Task0/21缺失的统一登记文件，保留原有数据和划分。网页的准备状态现在以A100登记为准，不再把这两个可训练任务显示为未准备。

已启动本机持久数据队列PID3286542，`train/prepare_all_datasets.py`按任务号顺序补齐剩余46项，当前Task1转换中；不是所有任务已经传完。每次仅准备一个任务，4线程进行转换，CPU低优先级；不使用训练GPU。转换复用既有`prepare_web_dataset.py`，官方归一化、seed42及180/20协议不变。检查本机完整文件、上传，再检查A100的200个parquet和600个视频全部非空后发布`preparation/ready.json`。失败任务记录错误并继续其余任务，最终汇总未完成项。网页每30秒自动更新数据选项，区分排队、转换、上传和A100就绪；已在总队列中的任务不再重复提交。只关闭浏览器或网页后端不会停止这个独立进程；本机需要保持开机及网络。

本机数据队列状态、全量清单、日志统一在 `outputs/shared/training_dashboard/datasets/all_tasks/{status.json,inventory.json,prepare_all.log,process.json}`，各任务日志为 `datasets/taskN/prepare.log`。转换目标仍为本机`assets/datasets/b1k2025_groot_taskN_v1_200/`，远端`/nvme3/data_wyh/datasets/b1k2025_groot_taskN_v1_200/`。这些数据准备与已存在的三任务联合训练数据完全独立；联合数据600份parquet、1800个视频已经齐全，不等待其余46个任务。

联合排队由A100上的`train/joint_task_queue.py`固定快照进程PID740220执行，每10秒读取原四组队列及各运行状态；必须四组各3000步、模型落盘、最终验证完成且包含step3000，才进入GPU检查并启动两个统一训练入口。状态写入`training/queues/V_1_joint_task0_34_40_train01/status.json`，不依赖网页定时器或浏览器。10:17时Task34两组已完成，Task40约Baseline2127/3000、Memory1712/3000，联合队列仍正常等待四组完成。

昨晚提交后用户新增了推理服务等GPU进程；队列只允许提交时明确登记的原进程，因此前四组完成后仍可能进入`waiting_for_gpus`。网页现在能显示联合队列的逐组依赖进度、等待GPU的PID和错误，不把排队误称为训练中；不会自动停止新增服务或其他账号任务。若服务持续运行，应另行选择兼容共卡安排；本次未修改原训练快照、GPU分配或进程许可。35项数据就绪/队列显示/训练及推理网页测试通过，网页后端已重载为PID3292212，原本机评估进程3267977仍保持相同启动时间及运行记录。

## 已安排：Task34 → Task40 → 三任务联合训练（2026-11-11 01:08）

Task40 的 Baseline、Memory 已在原队列中，各3000步；它们会在Task34两组的训练及最终验证完成后自动启动。随后再启动 Task0＋34＋40 **联合训练的两组模型**，不是再单独跑六组。新增队列已在A100后台运行，PID740220，当前为 `waiting_for_four_runs`；关闭浏览器、SSH或本机不会取消队列。前四组任一失败时，新队列记录错误并暂停接续，不会跳过失败或停止其他服务。

| 新运行 | 训练GPU | 独立验证GPU | 预算 |
| --- | --- | --- | --- |
| `V_1_baseline_task0_34_40_train01` | 0、1 | 0 | 18000优化步 |
| `V_1_memory_task0_34_40_train01` | 2、3 | 3 | 18000优化步 |

遵循最新命名 `V_版本_方法_task任务号_train次数`；三任务用 `task0_34_40`。两组都从 `/nvme3/huggingface/train_model/GR00T/GR00T-N1.6-BEHAVIOR1k` 重新初始化，不续训Task0/21/34/40的项目权重。Eagle冻结，原动作专家全参数更新，Memory组另更新既有记忆模块；没有LoRA或网络结构改动。两组固定seed42、AdamW学习率1e-5、weight_decay=.01、梯度裁剪1、global_windows=4即128决策/更新、microbatch4。每1000步保存，每3000步完整独立验证；验证与训练为独立并行进程。关闭基于验证平台期的自动早停，保持两组相同18000步预算；用户设置的显式停止文件仍有效。

数据合计600条，沿用每任务既定180训练/20独立验证，共540/60；Task0不重拆、其余任务不重抽样。A100组合目录 `/nvme3/data_wyh/datasets/b1k2025_groot_tasks0_34_40_v1_600` 通过软链接复用已转换的parquet/视频，保持官方归一化、动作标签及语言协议。训练按任务1:1:1循环，每任务保留自己的Episode顺序和采样游标；32决策窗口间延续detach历史，每个Episode固定rank所有权，不跨Episode共享记忆。各任务独立结束一遍后才重排自己的Episode；三任务长度不同，联合运行没有单一完整Epoch，优先看逐任务50/100更新均值。

Loss约0.0025为观察目标，不能保证18000步达到；不以某个低Loss单步宣称达标。队列汇总每任务最近100次更新均值，并保留各Episode验证Loss、正常/空/同任务且时间匹配错误历史MAE、记忆梯度、显存和吞吐。训练曲线为各run的 `plots/training_curves.png` 和 `plots/per_task_curves.png`；队列的 `summary.json` 每30秒更新。最低验证Loss仅用于保留候选checkpoint，不代表最佳机器人策略。

**时间预期：01:08时Task34约Baseline640/3000、Memory540/3000，后面还有Task40两组。按当前速度，联合训练明早很可能尚未开始，不能承诺明早训练完成或Loss达标。** 明早可验收队列是否正常推进、Task40是否已接续，以及已有训练指标。真实A100双卡6步检查已覆盖40→34→0两个循环，全部记忆梯度为正，rank参数完全一致；62项相关自动测试通过，模型目录与原四组固定源码完全一致。

A100完整输出根目录为 `/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/`，其下两个新运行目录分别存配置、日志、逐步指标和权重。队列统一放 `queues/V_1_joint_task0_34_40_train01/`，没有新增Markdown。查看方式：

```bash
# 在A100执行：先看队列；status比等待阶段的日志更详细。
QUEUE=/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/queues/V_1_joint_task0_34_40_train01
cat "$QUEUE/status.json"
cat "$QUEUE/summary.json"
tail -F "$QUEUE/queue.log"

# 新训练实际启动后，分别查看每步Loss。
ROOT=/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training
tail -F "$ROOT/V_1_baseline_task0_34_40_train01/logs/baseline_torchrun.log"
tail -F "$ROOT/V_1_memory_task0_34_40_train01/logs/memory_v1_torchrun.log"
```

以下历史段落中的短名、单任务限制和“联合尚未实施”是旧阶段记录，以本节为准。网页创建表单仍选择单个任务；本次联合运行由队列调用同一个 `train/run_groot_train.sh`，联合记录可在训练列表查看。

本版在 GR00T N1.6 原动作专家前接入 MemoryVLA 双路记忆、两层检索、时间编码、Gate Fusion、融合写回和相邻 ToMe 合并。Task21的M1和无记忆对照均已完成10000步；Task0的两组均已完成3000步及独立验证。下文记录的是 Task34/40 各3000步的历史训练安排。每组均从官方 `GR00T-N1.6-BEHAVIOR1k` 权重开始，未继承本项目其他任务的微调权重。该官方起点已做过 B1K 任务适配；这些历史运行未使用 LoRA。当前训练入口已支持可选 LoRA，仍无 Body Query 或辅助损失。

2026-11-10 本轮更新：Task0 记忆组已按要求保存并停止在3000步，独立验证完成。Task34/40各200条示教已转换并传到A100，每任务固定180训练/20独立验证、seed42，划分无交集。四组使用短名称 `V_1_bas_task34`、`V_1_mem_task34`、`V_1_bas_task40`、`V_1_mem_task40`，同任务两组并行，Task34全部训练/验证完成后自动进入Task40；不是将两个任务混合训练。

这四组统一使用GPU1/2双卡DDP，同任务的两组共卡并行；每组global_windows_per_step=4，即128决策/优化步，microbatch=4，lr=1e-5，seed42，每1000步保存。GPU0运行独立异步验证，启动训练时保留GPU3已有推理服务。Eagle冻结，动作专家全参数更新，记忆组额外更新原记忆模块。与之前Task0的global96不同，比较新任务Baseline/M1时使用本轮相同global128预算。两组均已产生有限训练Loss，记忆组已有非零记忆梯度；初始数十步不能判断训练效果。

A100统一输出根目录为 `/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/`，上述四个短名分别为运行目录。持久队列状态与日志在该根目录的 `queues/V_1_task34_40/{status.json,queue.log}`；训练日志和逐步指标在各run的 `logs/` 与 `baseline/metrics/metrics.jsonl` 或 `memory_v1/metrics/metrics.jsonl`。每次运行使用固定源码快照，网页自动同步后续代码不会改变本轮运行。

2026-11-11目录更新：本机和A100的Task0/21/34历史物理目录均已改成 `V_1_mem_task任务号` / `V_1_bas_task任务号`，Task21原配对目录拆为两组，模型与优化器文件直接移动、没有重写。Task40启动后由迁移程序整理为同一规则。A100暂留被进程/队列引用的旧路径软链接（不是第二份权重），`train/migrate_run_names.py`后台检查，引用结束后自动更新路径元数据并清除兼容链接。记录在A100 `training/queues/name_migration/`；旧源码部署目录保持原样，用于复现。网页隐藏兼容链接，只列每组一个新目录。

数据位置已核对：A100 `/nvme3/data_wyh/datasets/b1k2025_groot_task{0,21,34,40}_v1_200/` 已有四任务可训练数据；本机 `assets/datasets/b1k2025/` 已有全部50任务，每任务200条parquet和三路各200条视频，总原始数据约500GiB。其他任务在网页点“准备任务数据”，从本机已有文件转换并同步到A100 `/nvme3/data_wyh/datasets/`，不需要再下载，也不必预先复制整套原始数据。

模型结构与改动详解见 [model.md](model.md)，历史交接见 [handoff.md](handoff.md)。原始指标和检查结果归根 `outputs/`。

## 网页训练与 4090 → A100 自动同步（2026-11-10）

启动原来的评估网页即可使用：在仓库根执行 `./evaluation/view_results.sh --no-browser`，打开 <http://localhost:8765/>，切换顶部“启动训练”。后端已经运行时直接刷新网页，无需重复启动。

1. 填写代码版本（默认 `V_1`，用于运行标识），选择Task0–49、Baseline或MemoryVLA v1，以及A100初始化checkpoint。任务下拉列出全部50项并标记“可训练/需准备数据”；当前Task0、21、34、40已可训练。其他任务选择后可点“准备任务数据”，后台转换本机200条原始示教并传到A100，固定180训练/20独立验证；完成后点刷新。缺少原始示教或三路视频时页面显示数量并禁止准备，不会把未准备的数据提交训练。当前同步源码仍为本V1工程，填写版本标识不会切换网络实现。
2. 勾选训练 GPU，可用单卡或多卡；再选择异步验证 GPU、步数、Microbatch、全局窗口数、学习率和种子。全局窗口数必须能被训练卡数整除；每窗口32决策，全局 Batch = 窗口数 × 32。3窗口为96决策，4窗口为128决策；切卡后注意页面给出的全局 Batch。
3. 点“检查参数”仅预检；点“同步代码并启动训练”才创建新运行。所选卡已有进程时默认拒绝；显式勾选共卡才允许保留原进程并一起运行。异步验证用独立进程，与训练并行，也可与训练共卡。
4. 训练记录每10秒更新实际优化步、单步 Loss、显存、吞吐和梯度范数。曲线可切换50步均值或完整 Epoch 均值；独立验证的 checkpoint 步数单独展示。可查看训练/验证日志，从已完成保存的 checkpoint 选择“带到推理页”，再选择 GPU、端口启动推理。可设置未来1000整数倍保存后停止；不会直接杀训练进程。

网页新建训练复用入口 `train/run_groot_train.sh`，后端与页面实现在 `train/dashboard.py`、`dashboard_remote.py`、`dashboard.html`、`dashboard.js`，HTTP 路由由 `evaluation/scripts/view_results.py` 提供。默认冻结 Eagle、更新原动作专家，记忆组额外训练既有记忆模块；现在也支持全模型、自定义冻结和 LoRA，详见本页首节。从所选权重新建训练会重建优化器，**不是恢复旧训练的 AdamW 和采样游标**。

网页断点续训：选择已结束或异常退出的单组网页训练，在详情顶部点击“续跑训练”，填写**训练总步数（包含已完成步数）**，再点击“确认续跑”。自动使用最新完整模型、AdamW、各 rank 游标和记忆状态，保留原源码快照、GPU数量、Batch、种子、数据划分和验证开关；沿用同一训练目录和曲线。旧停止标记及未保存到断点的失败尾段指标会归档到 `config/web_resumes/resume-NNN/previous/`，尾段从完整断点重算。默认不共卡；原卡已有服务时可显式勾选允许共卡。`train/web_training_resume.py` 为旧版关闭验证的启动器提供调度兼容，仍调用原快照的 DDP 训练实现，不修改模型或训练算法。配对运行及缺少原网页启动命令的历史运行继续使用原 CLI 的 `--resume-step`；网页会说明不可续跑的原因。

自动同步是单向源码镜像：本机改动后每4秒检查，同步到 A100 的 `/nvme3/projects/memory_vla/deployments/web_live/`。范围为 `src/v1_groot_memoryvla/`、`train/` 和根 `pyproject.toml` / `uv.lock`（存在时）；不包含数据、模型、outputs、虚拟环境和评估目录。同步范围内删除文件也会反映到专用镜像目录，不触碰其他部署目录。依赖配置同步不等于安装依赖，A100 继续使用 `/nvme3/projects/memory_vla/envs/n1d6-p0/bin/python`，新增依赖需另外安装。

每次启动训练会把镜像复制成独立源码快照。后续命名与本地评估的 `V_0 / V_1` 版本约定一致，统一为 **`V_版本_mem或bas_task任务号`**，例如 `V_1_mem_task0`、`V_1_bas_task0`、`V_2_mem_task21`；相同名称再次运行使用 `_r2`、`_r3`，避免覆盖。时间、随机种子、任务英文名、SHA和完整参数写入配置，不再堆到目录名中。历史运行的物理目录与网页名称均已迁移；正在被进程引用的旧路径暂留兼容软链接，引用结束后自动清理。

```text
A100 /nvme3/projects/memory_vla/deployments/
  web_live/                                      # 持续更新的开发镜像
  V_1_bas_task0/                                  # 单次训练固定源码
  V_1_bas_task0_r2/                               # 同名第二次训练
A100 /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/
  V_1_bas_task0/
    config/                                      # 参数、数据划分、源码 SHA256 清单、进程记录
    logs/                                        # controller、训练、独立验证
    baseline/                                    # 或 memory_v1；指标、权重、验证队列
4090 outputs/shared/training_dashboard/
  sync_state.json                                # 自动同步开关、最新 SHA、时间与错误
```

关闭浏览器不影响同步；关闭网页后端则停止同步，下次启动读取持久化开关。源码同步失败会在页面显示，点击启动时也必须同步成功；运行中的训练始终读取本次快照，后续本机修改用于下一次训练。无需手动传源码，也不会让正在训练的代码随编辑变化。已有训练/推理服务的源码、权重与进程保持原状。本轮已更新网页后端，保留正在运行的本机评估及A100推理服务。

`.sh`训练入口没有Task21任务白名单；内部文件名中的task21是历史命名，实际任务由 `--dataset` 和 `--split` 中的数据与语言标签决定。它支持所有0–49任务的单任务训练，仍要求每任务200条、固定180/20划分；不代表当前已支持多任务混合训练。命令行准备其他任务可执行下面两条；第二条只预检，不启动训练，确认后另选全新输出目录并去掉 `--dry-run`：

```bash
# 在4090仓库根执行：后台网页按钮使用同一准备程序。
src/v1_groot_memoryvla/.venv/bin/python train/prepare_web_dataset.py \
  --root /nvme3/projects/memory_vla --host 172.28.62.17 --task 34

# 在A100执行：task34可替换为其他已准备任务；输出目录必须是新的。
bash /nvme3/projects/memory_vla/train/run_groot_train.sh \
  --source-dir /nvme3/projects/memory_vla/deployments/web_live \
  --dataset /nvme3/data_wyh/datasets/b1k2025_groot_task34_v1_200 \
  --split /nvme3/data_wyh/datasets/b1k2025_groot_task34_v1_200/preparation/data_split.json \
  --checkpoint /nvme3/huggingface/train_model/GR00T/GR00T-N1.6-BEHAVIOR1k \
  --output /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/V_1_bas_task34_r2 \
  --arm baseline --gpus 1,2 --async-validation-gpu 0 \
  --steps 3000 --global-windows-per-step 4 --microbatch 4 \
  --learning-rate 1e-5 --seed 42 --dry-run
```

## 后续三任务训练方案（尚未实施）

建议先完成当前Task0的同协议闭环比较，确认官方/微调接口与评估流程，再开展三任务联合训练；不要求先把Task0 Loss压到接近0。Task0用作短流程对照，另两个任务应结合历史依赖证据和评估耗时选择，不能仅选最短任务就声称记忆收益。三个任务可以共用一个GR00T+MemoryVLA v1模型，任务由既有语言指令区分，不增加Body Query、网络或辅助损失。多任务是否改善当前微调退化尚未验证，不能承诺。

两种实验回答的问题不同：三个任务分别训练，每任务各有Baseline/M1，共6组，更容易归因且可分别选预算；三任务联合训练，只训练一组混合Baseline和一组混合M1，共2组，更适合得到同时执行三任务的同一模型，但需检查任务间干扰。对当前快速迭代目标，先闭环跑通单任务，再联合三任务，是建议顺序；若最终主张记忆跨任务收益，仍需逐任务报告、必要时补单任务对照。

联合方案保持每任务180训练/20独立验证（若三任务各200条，即540/60），逐任务固定Episode划分，两组使用同划分/官方起点/冻结设置/seed/全局批量和优化预算。按任务的32决策窗口约1:1:1采样，任务内严格顺时序、同Episode跨窗延续detach记忆；不同任务/数据集/Episode的唯一键和Reset必须隔离，不把任务数据逐段接成一条历史。官方归一化、语言协议与动作标签继续一致。每任务分别报告Loss、MAE、正常/空/同任务且时间匹配错误历史、闭环结果及逐Episode结果，再算任务等权宏平均，不能只看混合总Loss。

预算必须按数据曝光解释：global96保持、三任务等权时，联合3000步平均每任务仅获得单任务3000步的约三分之一决策预算；想匹配每任务曝光，联合总步数约9000。每遍总窗口数则取决于各任务实际长度，不能仅凭Episode数量估计；联合不自动使单任务学习更快。三卡是DDP共享同一模型，并非训练三个彼此独立的策略。

当前代码差距仅记录、未修改：`data/prepare_task21_formal.py`一次转换一个任务、固定200条并拆180/20；`experiment/launch_task21_ddp.py`及`task21_formal.initialize`写死180/20；`data/dataset/episode_windows.py`正式封装单路径且按裸Episode整数分组；`EpisodePlan`按全部窗口平衡rank而非按任务平衡；`task21_formal.validate`错误历史从全部训练Episode选donor，联合时必须限制同任务，并补逐任务指标与任务等权汇总。底层HistoryDataset已有多路径及dataset命名空间能力，但正式入口仍需贯通。未启动多任务训练、未改变当前训练/服务。

## 当前 A100 训练

21:38更新：Baseline step3000的独立验证已结束，训练/验证进程均完成，仍保持用户要求的3000步暂停。验证Loss在1000/2000/3000步分别为.00341435/.00349244/.00346592，固定动作面板MAE分别为.00944524/.00892072/.00903884；3000比2000验证Loss降0.76%、MAE升1.32%，没有稳定持续改善。快速迭代建议先同协议闭环比较已有2000/3000权重，再判断是否延长至5000；不宣称已收敛，也不以最低验证Loss认定最佳策略。此轮没有恢复Baseline或改变仍运行的M1预算。下文“3000验证仍在运行”为21:30历史快照。

**最新状态（2026-11-10 21:30）：Task0 Baseline已准确停在3000优化步，三rank均恰好记录1–3000步并退出，没有3001步。** 通过该run的`metrics/stop_training.json`控制正常保存后退出，没有杀进程；step3000模型、AdamW和三rank游标均已保存，之后可续训。原配置/历史逐步日志的5000仍为原始上限，以stop文件、`baseline/status.json`的实际完成3000及`metrics/pause_step-003000.json`为准。原5556服务未动，M1继续5000预算（当时974步），GPU2/3原Baseline显存已释放；Baseline独立validator仍在GPU1计算step3000，完成后自动退出，训练结束不等于验证已经结束。以下时间和并行卡数表是此前快照。

Baseline单步3000 Loss=0.0016990430；更可比的完整数据轮次平均Loss：1408–1876为.0029779238、1877–2345为.0028833324、2346–2814为.0028180673，末轮再降2.26%，比第一轮.0035335104累计降20.25%。2815–3000只是第7轮前186步，均值.0021206743，不与完整轮次直接比较。step3000独立验证尚在运行，不能先给出验证Loss或策略改善结论。[训练曲线已刷新至3000步](../../outputs/v1_groot_memoryvla/training/V_1_bas_task0/plots/training_loss.png)，本机已同步；每步JSONL、完整轮次和保存/退出核查记录均在本run。

用于评估的权重完整路径（A100）：

```text
/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/V_1_bas_task0/baseline/checkpoints/step-003000
```

20:45估时快照：Baseline2368步、近100步均4.02秒，距3000约42分钟纯训练，含保存约45–55分钟；M1为436步、约5.05秒/步，距2000约2小时12分钟、距3000约3小时36分钟纯训练。以上为当前共卡速度外推，不含完整独立验证，速度会随另一组结束与验证阶段改变。Baseline第5个完整数据轮次（1877–2345）平均Loss .00288333，比上一轮仍降3.18%；2000可作为快速迭代的预算停止点，不能等同模型已收敛。此次只回答时间与预算，没有停止训练或改变两组5000步上限。

卡数与Loss检查（2026-11-10 20:39）：两组均3个DDP rank，Baseline物理GPU1/2/3、2286/5000，M1物理GPU0/2/3、371/5000；共4张卡，2/3共卡。公共入口按`--gpus`列表设置进程数，单/双/三/四卡均已有实测；`--global-windows-per-step`必须能被卡数整除，当前3窗口可用1/3卡，从头用2/4卡应选可整除窗口数（例如2/4），全局batch随窗口数改变。断点续训要求原卡数、窗口数等配置相同，不能对本轮直接换卡数。脚本是单机DDP。Baseline最近完整数据轮次训练Loss降幅从7.10%、6.80%缩至2.67%，收益减小但未证实收敛；第5轮尚未完整，不直接比较其均值。独立验证step1000→2000的Loss为.00341435→.00349244（升2.29%），固定动作面板MAE为.00944524→.00892072（降5.55%）。指标不完全一致，不能把最低验证Loss称为最佳机器人策略；快速迭代可先比较已有1000/2000权重的同协议闭环。此次未改变两组5000预算或停止训练。

Loss查看：当前训练没有接入W&B，每步指标写JSONL。已用既有绘图工具刷新[Baseline训练曲线](../../outputs/v1_groot_memoryvla/training/V_1_bas_task0/plots/training_loss.png)（2042步快照）和[M1训练曲线](../../outputs/v1_groot_memoryvla/training/V_1_mem_task0/plots/training_loss.png)（176步快照）。上图每50步取均值，下图每500步取均值，灰虚线标记新一遍数据开始；最后一个分组可能不足50/500步，不宜把尾点骤降当作整体改善。两组需对齐优化步比较，M1尚不足一遍。PNG是快照，JSONL才是实时逐步记录；training_curves.png随验证完成更新，training_loss.png可用下述命令随时重新生成，均不需要暂停训练。

```bash
# 在A100执行；M1_RUN可替换为Baseline的run路径。
M1_RUN=/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/V_1_mem_task0
env PYTHONPATH=/nvme3/projects/memory_vla/deployments/20261110_task0_baseline_resume_5000/src \
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /nvme3/projects/memory_vla/envs/n1d6-p0/bin/python \
  -m v1_groot_memoryvla.audit.plot_task21_loss --run "$M1_RUN"
# 图片输出到该run的plots/training_loss.png；本机副本需重新同步。
```

快速迭代预算参考（2026-11-10快照）：建议先以每组2000优化步、约4.3遍训练数据作为一轮预算，1000步先检查闭环，2000步有继续改善再考虑3000–5000。依据为Baseline前四个完整epoch训练Loss约0.003534→0.003283→0.003060→0.002978，首末下降15.7%，最后一遍下降2.7%；不是已确认2000步收敛，也不要求Loss接近0。核查时Baseline1967步、M1仅107步，尚不能确认记忆组学习速度或收益；仍需同预算与独立Episode结果比较。M1当前约5.8秒/步，2000步约3.2小时纯训练，另有保存/验证且共卡速度会变化。此建议未改当前两组5000步上限或训练进程。

2026-11-10 20:10新增记忆组：`V_1_mem_task0`，与下述Task0 Baseline并行，目标同为5000优化步。复用既有MemoryVLA v1结构，冻结Eagle，更新原动作专家及记忆模块；感知256维、容量16、两层Cross-Attention、时间编码、Gate Fusion、融合写回及相邻ToMe合并保持不变。32决策窗口及同Episode跨窗detach记忆延续不变，Episode之间独立。两组同180/20划分、seed42、动作专家官方初始SHA、Episode采样顺序、全局96决策、micro4、AdamW lr1e-5/wd.01/clip1；M1不是加载Task21微调权重继续训练。

| GPU | 当前用途 |
| --- | --- |
| 0 | M1 rank0、M1独立异步验证、原5556官方推理服务 |
| 1 | Baseline rank0及其独立异步验证 |
| 2 | Baseline rank1与M1 rank1共卡 |
| 3 | Baseline rank2与M1 rank2共卡 |

启动前实际共卡试跑4步，M1峰值allocated35.54GiB、双路记忆梯度有效，三rank更新后参数SHA逐值相同。20:09正式M1到24/5000、Baseline到1852/5000，M1独立验证同时完成4/20条Episode，全部已有Loss/梯度有限；近期M1约5.44秒/步、Baseline约3.85秒/步。GPU2/3正式总占用约72.3GiB/80GiB，GPU0含服务与验证约63.1GiB；占用会随阶段变化。瞬时GPU利用率低不代表没有算力瓶颈；共卡争用会让原Baseline单组变慢，不能称为增加GPU数量后的加速。原Baseline进程、优化器、当前游标及原5556服务均未重启。每1000步保存并交给独立验证器，训练不等待验证；初始step0验证也在独立进程中进行。

```bash
# 在A100查看记忆组；本命令只读日志，不启动另一份训练。
M1_RUN=/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/V_1_mem_task0
tail -n 3 -F "$M1_RUN/logs/memory_v1_torchrun.log"
# 每步完整记忆梯度、显存及吞吐；验证日志中的step是checkpoint编号。
tail -n 1 -F "$M1_RUN/memory_v1/metrics/metrics.jsonl"
tail -n 3 -F "$M1_RUN/logs/memory_v1_validation.log"
cat "$M1_RUN/status.json"
```

M1模型会保存到`$M1_RUN/memory_v1/checkpoints/step-001000`至`step-005000`；配置在`config/run_config.json`、`data_split.json`、`ddp_launch.json`，共卡试跑在`diagnostics/shared_gpu_preflight/`、结论在`config/preflight_result.json`。公共训练入口仍为`train/run_groot_train.sh`，准确调用记录为本run的`config/start_remote.py`，已启动不要重复执行。源码复用`deployments/20261110_task0_baseline_resume_5000`，没有新网络改动。输出独立于Baseline与Task21，本机仅镜像轻量记录，权重保留在A100；本轮不自动启动闭环。下文19时的单独Baseline速度是新增M1之前的历史快照。

2026-11-10运行：`V_1_bas_task0`，Task0 `turning_on_radio`，200条示教固定180训练/20独立验证，seed42。首段1000步训练和验证已完成，按用户要求从step1000完整续训到总计5000步（再训练4000步），保留1000步权重。正式续训保持A100 GPU1/2/3三卡DDP与原优化器、Episode游标；GPU0上的其他训练已自行退出，官方5556服务保留。仅训练原动作专家全部参数，Eagle冻结/eval，无记忆、无LoRA；AdamW固定lr1e-5、wd0.01、clip1，microbatch4，每优化步3个32决策窗口，共96次决策，决策间隔8环境步。每1000步保存和异步验证，验证器共享GPU1。训练与原Task21产物各有独立目录，不继承Task21模型或优化器。

```bash
# 在A100查看，每步都有step/total_steps、Loss、梯度、显存和吞吐。
RUN_DIR=/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/V_1_bas_task0
tail -n 3 -F "$RUN_DIR/baseline/metrics/metrics.jsonl"
# 验证用的checkpoint步数与训练优化步数分别记录。
tail -n 3 -F "$RUN_DIR/logs/baseline_validation.log"
cat "$RUN_DIR/status.json"
```

本轮数据：`/nvme3/data_wyh/datasets/b1k2025_groot_task0_v1_200`；续训源码：`/nvme3/projects/memory_vla/deployments/20261110_task0_baseline_resume_5000`；起点：`/nvme3/huggingface/train_model/GR00T/GR00T-N1.6-BEHAVIOR1k`。已保存`$RUN_DIR/baseline/checkpoints/step-001000`，后续2000/3000/4000/5000同样保存在checkpoints下；`best.json`仅为验证Loss候选，不代表最佳机器人策略。训练完成后自动更新`plots/training_curves.png`及`plots/training_loss.png`；训练日志继续追加到`logs/baseline_torchrun.log`，验证到`logs/baseline_validation.log`，续训调度到`logs/controller_extend_5000.log`，不重置原日志或step。配置、固定划分、转换核查、源码SHA及准确启动命令均在`config/`；原1000步启动记录为`start_remote.py`，延长记录为`extend_5000.py`和`extension_5000.json`，已启动的运行勿重复执行。没有W&B。比较是否退化仍须官方/训练后权重使用相同Task0实例、seed和闭环设置；本轮不自动启动闭环评估。

原1000步独立验证Loss：官方step0为0.0042052847，step1000为0.0034143548；它们只反映固定20条验证Episode上的拟合，不是机器人成功率。为响应用户的多卡速度比较，本轮正式续训前在`diagnostics/scaling_1_2_4cards/`归档真实模型单/双/四A100短测：统一每步4个32决策窗口（128决策）、同seed/前8步数据、microbatch4、冻结Eagle、不验证/不保存；去掉2步预热，统计后6步。短测不写入正式训练日志或权重，GPU0官方服务保留；正式续训批量仍为96决策，不能直接以两种批量的步数换算训练预算。结果和准确命令写入该目录的`summary.json`。

| A100数量 | 平均秒/优化步 | 决策/秒 | 相对单卡加速 | 单卡分配显存峰值 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 8.0592 | 15.88 | 1.00× | 33.10GiB |
| 2 | 4.5641 | 28.04 | 1.77× | 33.13GiB |
| 4 | 2.6128 | 48.99 | 3.08× | 33.13GiB |

四卡是双卡的1.75倍、单卡的3.08倍，四卡并行效率约77.1%。计时包括读视频、Processor、冻结Eagle前向、动作头反传、梯度同步和优化器更新，不含初始化、模型保存与验证；每组仅6个计时步，因此是本机短测，不保证长跑同样加速。四卡短测期间GPU0官方5556服务保持运行；每组全部rank训练参数SHA逐值相同，证据为world-N/baseline/config/real_ddp_check.json，短测没有额外模型权重。

19:32续训已实际越过1000步；核查三rank均到1019/5000，AdamW从1000步恢复，Episode游标为epoch2/position62，日志1001起连续追加，Loss/梯度有限且梯度非零；官方权重文件size/mtime均未变。当前正式三卡每步约2.66秒，剩余约3小时训练（另有保存/验证开销）。这是从1000步续训后总计5000，不是重新训练5000步。

19:48–19:49 Task0 Loss只读核查（1383步快照）：单步范围0.0005450–0.0148184，所有Loss/梯度有限，无零梯度，裁剪前梯度最大0.35828、无超过1的步。每遍469优化步、1406窗口；第470/939步进入新一遍，窗口按Episode内时间顺序轮询，每步还重采样Gaussian动作噪声与Beta时间点，单步或连续100步均值不要求单调。第一遍100步分块均值约0.00257→0.00428，第二遍再次从约0.00257升到0.00427，起伏与采样阶段一致，不应只截两段判断变差。完整两遍按实际决策数加权的训练Loss为0.0035335104→0.0032827811，下降7.10%（两遍均1406窗口/44992决策，模型沿途更新且噪声不同，非固定checkpoint同噪声重评）。独立固定20Episode验证0.0042052847→0.0034143548降18.81%，截至核查只有step0/1000两点，不能断言已收敛或后续持续改善。Loss是动作有效维度上的归一化flow-matching速度MSE，源码model/gr00t_n1d6/gr00t_n1d6.py:203–208、249–251；不是任务成功率、动作MAE或原始物理单位误差。没有“低于0.001/0.003就训练完成”的统一阈值；看完整数据遍历/平滑趋势、固定独立验证及相同协议闭环表现。当前未见NaN/梯度爆炸或断梯度证据，但不据此保证机器人能力；本次未改训练、学习率或模型，5000步任务继续运行。

本轮实现将原200条Task21转换器扩展为`--task 0`（默认21不变），单组训练支持`--async-validation-gpu`，绘图兼容只有baseline的数据；31项测试通过。当前本地公共评估脚本已支持0–49全部任务，包括Task0，无需更改现有评估服务。以下Task21的服务/评估状态均为当时的历史核查。

2026-11-10短任务筛选：逐任务汇总本地 `assets/datasets/b1k2025/meta/{tasks,episodes}.jsonl` 的全部50任务×200示教，无未匹配项。按平均环境步数排序，最短为Task0 `turning_on_radio`（打开桌上的收音机），平均2149.64、中位2109.5、范围970–4353步；其后Task34 `hanging_pictures` 平均2390.17，Task40 `make_microwave_popcorn` 平均3237.775。对比Task21平均19186.325，Task0约短8.93倍。当前2025本地GR00T仿真适配器使用int(2×平均示教步数)作为原生timeout，因此Task0阈值4299、Task34为4780、Task40为6475，实际Episode记录可能多一步；这些不是墙钟秒数。已交叉核对环境实际读取的 `/nvme3/shared/b1k2025/datasets/2025-challenge-task-instances/metadata/episodes.jsonl`，统计一致，available_tasks.yaml含turning_on_radio。

此前短任务检索时公共入口仅支持21/22/26，现已支持0–49，包括Task0。Task22平均7692.45步、原生阈值15384；Task26平均26060.055。若用Task0定位推理/评估问题，应先让原生官方服务与v1服务加载同一份未微调官方权重、同实例/种子/单环境/action8/原生终止协议，避免用仅Task21微调权重在Task0的表现判断接口正确性；Task0短任务不能代替Task21的记忆收益验收。官方2026文档的默认超时已是1.5倍，不能套用到本地2025的2倍实现。

2026-11-10闭环退化核查（评估仍在运行）：用户报告原版任务进度约34.8%、训练后约17.9%。已核对 `evaluation/V_0/baseline/groot_task21_1/` 原版10/10回合，平均进度34.7619%、成功率0%；`evaluation/V_1/baseline/groot_task21_1/`、`evaluation/V_1/experment_1/groot_task21_1/` 核查时各4/10回合，进度均17.8571%、成功率均0%，均达到38373步原生timeout，无invalid回合。5560实际为对照step010000，5561为M1 step010000；服务分别A100 GPU3/2，使用resume_r4源码及sim wrapper，未混用早期2000/3000候选权重。两个服务seed42；原官方旧服务启动RNG seed未留同等记录。

两次评估不是完全匹配：旧single1，新single2；新env1的instance5/6使用seed1/2，旧对应seed5/6，protocol摘要也不同（摘要来自evaluation.md文本，不能单凭hash判断执行语义改变）。共同instance0/seed0旧42.8571%、新两组均23.8095%；共同instance1/seed1旧38.0952%、新对照9.5238%、M1 23.8095%，提示退化值得调查，不能仅归因回合未跑完，但目前不是完整配对评估。归一化statistics.json与官方SHA完全相同，BEHAVIOR本体ID保持24；processor仅增加已保存语言映射与显式letter_box_transform=False（原默认也是False），未发现这些配置的明显错配。动作头forward/get_action数值逻辑与参考源码一致；已知旧double-vlln bug位于离线MAE复用特征路径，不是此次闭环单次get_action路径，不能当成闭环退化根因。

当前不能从这些汇总值确定退化根因。两组共有的约14.19亿动作专家全参数微调、单任务小批量监督与闭环分布差异是待验证方向，过拟合/能力遗忘只是候选解释，非已确认结论；记忆不是两组退化的唯一可能原因。训练Loss下降不保证闭环提升，val最低也不代表最佳策略。下一项有区分力的检查是官方原权重走相同v1服务及相同single2/实例/种子协议，再与训练权重逐Episode比较，必要时补早期checkpoint；这次未启动该检查、改训练/服务/模型或停止正在进行的评估。

训练曲线：[training_curves.png](../../outputs/v1_groot_memoryvla/training/V_1_mem_task21/plots/training_curves.png)。左上训练Loss按每50优化步平均；原图右上normal动作MAE含已确认的旧指标bug，仅作历史溯源，修复后结果见下表。训练末尾9501–10000步平均Loss：对照0.0028560320，M1 0.0028585138，两条训练曲线非常接近。训练采样同Episode顺时序且跨窗口延续记忆，不同优化步对应不同片段，优先查看同预算、同数据进度下的训练Loss趋势。

选模解释（用户2026-11-10明确要求）：对照step2000与M1 step3000只代表本轮各自最低独立验证Loss的候选点，不称为“最佳机器人策略”。两项最低验证Loss相差约0.53%，不支持机器人实际任务能力的高低结论。训练Loss用于判断拟合和收敛，实际策略能力、正确历史的作用仍需同协议B1K闭环评估。保留原selected_checkpoints.json/best.json作为历史验证指标记录，不自动将它们替换成训练Loss最低点；当前服务入口由用户显式选择checkpoint。

训练Loss核查（2026-11-10，只读日志与源码）：首尾单步对照0.00364504→0.00143439（降60.65%）、M1 0.00378130→0.00144483（降61.79%），但样本和扩散噪声不同，不能用作整体收敛幅度。首尾各1000步均值对照0.00329397→0.00283149（降14.04%）、M1 0.00330821→0.00283509（降14.30%）；首尾各500步仅降2.10%/2.47%，说明窗口选择影响较大。

更可比的检查：从两rank日志按epoch/Episode内窗口序号重建，确认每遍前3240优化步恰好覆盖相同180Episode的6480个窗口、无遗漏或重复；第一遍与第二遍这些窗口的在线训练均值，对照0.0031020411→0.0030191927（降2.67%），M1 0.0031160598→0.0030205104（降3.07%）。这仍是模型持续更新、噪声不同的在线训练记录，不是固定checkpoint/固定噪声重评。整体有学习，但下降较小，不能据此宣称充分收敛或策略改善。

曲线回升的主要证据：采样器按窗口序号轮询180条Episode，时序不打乱；每遍6630优化步，第6631步回到Episode前部。两组逐步Loss相关系数0.99806，7001–8000步均值同步回升到约0.00331，与第一遍早期片段的高Loss相呼应，不能直接解释为优化退化。GR00T监督为随机噪声/时间点的flow-matching速度MSE，并非闭环动作MAE；每步重采样，进一步产生波动。两组10000步Loss/梯度范数均有限，记忆两库及回注层每步都有非零梯度；梯度裁剪阈值1，记录为裁剪前范数，对照/M1分别13/7步超过阈值，未见持续爆炸。AdamW固定lr=1e-5、无衰减，Eagle冻结，10000步约1.51遍数据。现有证据没有显示发散或反传中断，但不足以认定训练充分；若继续诊断，应在相同固定训练窗口/噪声下比较官方起点与保存checkpoint，区分真实拟合改善和采样波动。此次未改训练配置、网络、权重或启动新任务。

2026-11-10 LayerNorm 修复与 MAE 重算完成：动作头 `process_backbone_output` 改为返回新字典，保留调用者的 pre-vlln 特征，修复 normal 在验证 Loss 后被再次归一化的问题。没有重训、加载优化器或改写 checkpoint。沿用已选定 M1 step3000/对照 step2000、原20条独立验证 Episode、中段32决策面板、从Episode起点预热的正常历史，以及原错误历史、噪声和标签协议。

| 模型 / 历史 | 前8步 MAE | 完整32步 MAE |
| --- | --- | --- |
| M1 normal | 0.0070459466 | 0.0089131389 |
| M1 empty | 0.0071129214 | 0.0089410975 |
| M1 wrong | 0.0070490461 | 0.0089148999 |
| 无记忆对照 step2000 | 0.0072682604 | 0.0091835783 |

M1 normal 原MAE 0.0283838754，修复后0.0089131389。20条Episode的empty/wrong与旧预测全部逐值相等；显式额外归一化可逐值复现全部旧normal，确认异常来自指标路径。实际推理审计确认三个条件调用同一个 `get_action`，每次1次vlln、4次DiT，microbatch2、BF16、RNG状态、state、embodiment和mask一致；三个分片各自以相同特征控制验证动作逐值相等。normal比empty平均仅低0.313%，比wrong低0.0198%，逐Episode分别12/20和10/20更低，仍不足以证明正确历史收益。原验证Loss及选模不变；旧metrics/曲线保留作溯源，其中normal MAE含bug，不能继续作有效比较。本次仅重算上述已选定权重，其余9个M1 checkpoint的动作面板尚未重算。

新汇总、逐Episode指标、动作NPZ、旧结果复现与推理审计在 [mae_layernorm_fix_20261110/summary.json](../../outputs/v1_groot_memoryvla/training/V_1_mem_task21/mae_layernorm_fix_20261110/summary.json)，远端位于同一run下同名子目录。重算源码独立部署在 `/nvme3/projects/memory_vla/deployments/20261110_mae_layernorm_fix/`，入口 `audit/recompute_task21_mae.py`；28项测试通过。旧部署快照未修改，本次未操作原5555/5556服务，仅使用GPU1–3；服务按另一操作停止的记录见下文。

2026-11-10 10:03–10:04核查：GPU1/2/3此前占用来自本轮离线动作MAE重算，现三个分片均完成，四卡compute进程均为空、利用率0%，仅32–35MiB驱动占用；对照20Episode、M1三片7+7+6Episode日志均completed。此次只读查询，未启停进程。旧服务已停，新权重闭环尚未启动。

2026-11-10 10:00后：按用户最新要求，旧官方5555/5556模型服务已停止，GPU0释放约13.84GiB；权重、日志保留。4090上已无旧仿真评估，新的训练权重服务及闭环评估尚未启动。下方保留旧服务的表述为此前训练期间记录。

2026-11-10 09:27–09:28最新核查：闭环队列在对照阶段收到SIGTERM后中断，状态failed；当前两组均未在评估，专用5562（对照）/5561（M1）均未监听，M1尚未启动。该脚本原安排为先对照、后M1顺序评估。原5555/5556服务仍在，训练及独立验证均已完成。此次只查询状态，未重启评估。

2026-11-10 09:12核查：**两组均完成10000优化步，1000至10000的全部10轮独立验证完成**，09:04生成按验证Loss选出的候选记录。M1最低验证Loss位于step3000（0.0027759974），对照位于step2000（0.0027613138）；不据此判断机器人任务能力。曲线在本轮plots/training_curves.png。该时刻闭环队列进入对照候选权重的评估阶段，随后中断，尚无本轮闭环完成结论。以下22:32状态为恢复过程记录。

运行标识：`V_1_mem_task21` / `V_1_bas_task21`，主机 `172.28.62.17`。按最新清理要求，所有旧训练、试训/验收产生的权重、日志、缓存及对应回收副本已永久删除；本地与A100只保留本轮训练。保留原180/20划分，新运行不继承旧优化器或旧记忆。官方权重、数据、GPU0 的原5555/5556服务保持原状。

2026-11-09 22:32核查：1000步停顿已修复，两组从完整模型、优化器、各rank游标与记忆恢复。M1训练已到1102、对照1089，同时两个独立验证器正在计算checkpoint1000，对照已完成首个验证Episode；确认训练与验证持续并行。根因为验证队列父目录未创建导致写文件异常，随后集体销毁等待掩盖了主日志中的报错；现已补目录创建并立即打印异常。没有改网络结构，也没有丢弃前1000步进度。

| 项目 | 当前设置 |
| --- | --- |
| 设备 | M1：GPU2、3；无记忆对照：GPU0、1；各两个NCCL rank，两组同时训练 |
| 独立验证 | M1验证用GPU1，对照验证用GPU0，与对照训练共享；读取已保存且不再变化的检查点，不阻塞训练 |
| 数据 | 同200条Task21示教；固定180训练、20独立Episode验证；沿用官方归一化统计 |
| 冻结与优化 | Eagle冻结/eval；原动作专家全参数更新，M1另更新原记忆模块；AdamW lr=1e-5、wd=.01、clip1、seed42 |
| 时序 | 每次决策对应8环境步；每条32决策窗口内部及跨窗口严格顺时序；同Episode延续detach记忆，Episode之间独立 |
| 全局批量 | 两组每次优化均处理两条32决策窗口，共64次决策；microbatch4，每组每卡一窗 |
| 预算 | 两组各最多10000优化步；每1000步保存，并向异步验证队列提交该检查点 |
| 验证范围 | 20条验证Episode的全部完整窗口，Episode等权Loss；正常/空/时间匹配错误历史的固定中段动作面板 |
| 本轮历史选模与早停规则 | 最低独立验证Loss只作候选指针，不代表最佳策略；历史早停条件为两组均连续4轮未改善0.1%，共同后续1000步边界停止，至少2000步，上限10000 |

异步验证存在滞后，因此早停依据的检查点可能早于当前优化步；共同停止点会留出两组到达同一预算的时间。每1000步仍需短暂停顿保存一致检查点，随后训练立即继续，独立验证器读取该固定版本；训练不等待逐Episode验证结果。真实M1双卡保存约30.87秒后继续反传已通过，见 `diagnostics/preflight/checkpoint_fixed_memory/`。训练结束后等待验证队列完成再选最终checkpoint与执行闭环。GPU0/1共卡计算有开销；22:28实际GPU0/1总占用约62.5/48.9GiB，GPU0原服务保持运行。

按追加要求，对照于21:15切换为GPU0/1双卡：只重启对照，切换状态中旧单卡对照125步记录已永久删除；M1进程、优化器、数据游标和跨窗记忆持续运行，没有重跑。切换证据在 `config/transition_2plus2.json`，两组仍使用同一个180/20划分及相同全局批量。

训练集共13259个完整窗口，两槽分配6630/6629窗，每遍6630次优化。新全局批量是旧单卡运行的两倍，10000步约1.5遍数据，不能把新旧曲线当作相同数据预算比较；本轮M1/对照之间的划分、采样、标签、随机种子、全局批量及优化预算一致。末尾空槽不重复取样，按实际窗口数归一化。

验证Loss覆盖全部完整窗口；动作MAE每Episode取固定中段32决策，分别报告前8步/完整32步，不称为全轨迹MAE。错误历史来自另一训练Episode相同相对进度，以32决策预热，并匹配正常库合并后的时间槽。动作噪声seed不当作独立Episode；尚不宣称这些面板已经证明历史必需。

真实双A100/NCCL检查已通过4次完整模型优化，两rank全部可训练参数SHA256逐值一致。固定64决策/优化步，预热后单卡约4.90秒，双卡约2.66秒，短测约1.84倍；正式长跑以实际日志为准。另补真实M1保存后继续训练与队列目录回归，当前测试25 passed。`--resume-step`要求同GPU数量、虚拟槽、数据、seed和优化设置，恢复HF模型、AdamW、各rank游标与detach记忆；当前两组已实测从1000步恢复。CPU恢复测试中下一窗口Loss和全部参数更新与不中断运行逐值相同。

## 路径与进度

只看训练Loss：[training_loss.png](../../outputs/v1_groot_memoryvla/training/V_1_mem_task21/plots/training_loss.png)。上图每50步平均，下图每500步平均，虚线标出第6631步开始第二遍数据；不含旧MAE指标，也不据此选最佳机器人策略。A100路径为本轮输出根目录下plots/training_loss.png。重画入口：audit/plot_task21_loss.py --run <本轮目录>，只读JSONL，不运行模型。

本轮没有启用W&B：实际DDP训练/启动/验证循环未调用wandb，也没有W&B run目录或事件文件；并非已开启但未显示。每步记录保存在两组metrics/metrics.jsonl，绘图直接读取这些日志。当前没有W&B仪表板或自动在线同步；本次未补传外部平台。

以下均为 A100 绝对路径：

| 内容 | 路径 |
| --- | --- |
| M1输出根目录 | `/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/V_1_mem_task21/` |
| 对照输出根目录 | `/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/V_1_bas_task21/` |
| M1训练日志 | `/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/V_1_mem_task21/logs/memory_v1_torchrun.log` |
| 对照训练日志 | `/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/V_1_bas_task21/logs/baseline_torchrun.log` |
| 验证日志 | 各组根目录下 `logs/memory_v1_validation.log` 或 `logs/baseline_validation.log` |
| M1 / 对照权重 | 各自根目录下 `memory_v1/checkpoints/` 或 `baseline/checkpoints/` |
| 当前训练/验证/控制器代码 | `/nvme3/projects/memory_vla/deployments/20261109_2100_task21_ddp_async_s42_resume_r4/` |
| 1000步前来源及闭环服务代码 | `/nvme3/projects/memory_vla/deployments/20261109_2100_task21_ddp_async_s42_2plus2_r2/`，历史M1来源为同部署目录下不带后缀的run目录 |
| 200条数据 | `/nvme3/data_wyh/datasets/b1k2025_groot_task21_v1_200/` |
| 官方原始权重 | `/nvme3/huggingface/train_model/GR00T/GR00T-N1.6-BEHAVIOR1k/` |
| Python环境 | `/nvme3/projects/memory_vla/envs/n1d6-p0/bin/python` |

在A100执行下列命令可直接看每一步；`step`是优化步，`total_steps=10000`是上限，`progress_percent`是百分比。结构化metrics不混入警告，适合查看训练进度；torchrun日志仍保留完整控制台信息。Ctrl+C只退出查看：

```sh
RUN_DIR=/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/V_1_mem_task21
BAS_DIR=/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/V_1_bas_task21
tail -n 3 -F "$RUN_DIR/memory_v1/metrics/metrics.jsonl" "$BAS_DIR/baseline/metrics/metrics.jsonl"
# 每组分别查看训练与验证状态
cat "$RUN_DIR/status.json"
# 查看验证
tail -n 3 -F "$RUN_DIR/logs/memory_v1_validation.log" "$BAS_DIR/logs/baseline_validation.log"
```

验证日志的`checkpoint_step=1000`表示正在评估第1000步保存的固定权重；训练日志的`step`会继续增长。验证每10个窗口报告一次Episode内进度，完整Episode结束再报告该Episode的Loss。

本地镜像：[运行目录](../../outputs/v1_groot_memoryvla/training/V_1_mem_task21/)。从本地项目根目录把RUN_DIR改为 `outputs/v1_groot_memoryvla/training/V_1_mem_task21` 即可使用同样命令，约30秒同步一次；完整模型/optimizer不回传。曲线是 `plots/training_curves.png`，配对验证完成时更新；每步Loss、全部记忆梯度、显存、分段耗时与吞吐在各组 `metrics/metrics.jsonl`，每rank记录在 `ranks/rank-XXX/`。

```text
<run>/
├── config/                  固定划分、启动参数、远端位置与显存检查
├── logs/                    控制器、两组训练和独立验证日志
├── metrics/                 共同早停决策
├── plots/                   配对曲线
├── baseline/                对照独立产物
├── memory_v1/               M1独立产物
│   ├── config/ddp.json      初始化哈希、冻结/批量/采样配置
│   ├── logs/                每rank及验证详细日志
│   ├── ranks/               每rank训练指标
│   ├── metrics/             训练/验证Loss和逐Episode误差
│   ├── validation_queue/    已完成保存的检查点通知
│   ├── diagnostics/actions/ 逐Episode动作数组
│   └── checkpoints/
│       ├── step-001000/ ... 完整模型与processor，每1000步保存
│       ├── best.json        最低验证Loss候选指针，不代表最佳机器人策略
│       ├── latest.json      最新数据游标与检查点路径
│       ├── latest_optimizer.pt
│       └── latest_rank-XXX.pt 各rank的detach记忆/游标
├── evaluation/              训练完成后的B1K闭环队列
└── status.json
```

启动器 `experiment/launch_task21_ddp.py` 使用 `torchrun`；训练在 `task21_ddp.py`，独立验证和异步早停在 `task21_async.py`，沿用 `task21_formal.py` 的验证实现。两卡通过DDP平均原动作专家/记忆梯度，冻结Eagle不参与归约，不拼接显存。每条Episode固定归属一个虚拟槽，再映射到rank；窗口与历史不跨rank搬移。

重现启动时须使用新的输出目录；当前运行已启动，不要重复执行。入口参数为 `--gpus 2,3 --paired --baseline-gpus 0,1 --async-validation-gpu 1 --baseline-validation-gpu 0 --global-windows-per-step 2 --steps 10000 --microbatch 4 --seed 42`，另提供 `--dataset --checkpoint --split --output` 绝对路径。启动器会拒绝已有计算进程的训练卡；本次通过 `--allow-service-pids 4055792 905468` 明确允许GPU0现有两个服务共卡，不结束服务。实际命令在 `config/ddp_launch.json`；本次接管脚本为 `config/start_remote_2plus2.py`，其中 `--adopt-running-memory` 仅适用于此次保留M1的切换，不能用于新run。

公共 `evaluation/scripts/evaluate_task21_trained.py` 的原队列读取两组最低验证Loss候选点，用独立5562/5561端口执行Task21实例0–9闭环评估，single1、seed0/sequence、每次执行8步、原生timeout，产物归 `evaluation/experiment_1/`，以run-tag追溯本轮。该队列已中断，当前由用户显式选择checkpoint启动服务；候选点不代表最佳机器人策略，尚无本轮闭环能力结论。

## 本轮正式训练方法与入口

用户训练统一从 [train/run_groot_train.sh](../../train/run_groot_train.sh) 启动；单独M1、单独无记忆对照、两组同时训练及单/多卡均使用这个入口，通过参数选择。脚本只负责定位源码、Python环境并传递参数，不复制训练逻辑。本轮使用自定义DDP循环：experiment/launch_task21_ddp.py调度torchrun，experiment/task21_ddp.py训练，task21_async.py独立验证；这些是内部实现，不需逐个启动。通用train.py/Trainer不用于本轮Task21流程。

本机和A100的公共脚本完整路径均为 `/nvme3/projects/memory_vla/train/run_groot_train.sh`。本机默认当前workspace源码与v1的.venv；A100默认 `/nvme3/projects/memory_vla/deployments/20261110_task0_baseline_resume_5000/` 源码与 `/nvme3/projects/memory_vla/envs/n1d6-p0/bin/python`，包含LayerNorm指标修复、单组异步验证及延长预算续训。可用 `--source-dir`、`--python-path` 显式覆盖。已完成Task21使用的resume_r4及Task0首段源码快照保持原样。

```bash
# 登录A100，在Bash中执行。以下公共路径先设置一次。
cd /nvme3/projects/memory_vla
train_common=(
  --dataset /nvme3/data_wyh/datasets/b1k2025_groot_task21_v1_200 \
  --checkpoint /nvme3/huggingface/train_model/GR00T/GR00T-N1.6-BEHAVIOR1k \
  --split /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/V_1_mem_task21/config/data_split.json
)
train_output_root=/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training

# M1 + baseline：M1用GPU2/3，对照用GPU0/1；两组各双卡DDP。
# 验证与训练并行，两个验证器分别共享GPU1/0；每1000步保存并提交验证。
./train/run_groot_train.sh "${train_common[@]}" \
  --output "$train_output_root/NEW_PAIRED_RUN" \
  --gpus 2,3 --paired --baseline-gpus 0,1 \
  --async-validation-gpu 1 --baseline-validation-gpu 0 \
  --steps 10000 --global-windows-per-step 2 --microbatch 4 \
  --learning-rate 1e-5 --seed 42 --dry-run

# baseline：仅无记忆对照，GPU0/1双卡，验证并行并共享GPU1。
./train/run_groot_train.sh "${train_common[@]}" \
  --output "$train_output_root/NEW_BASELINE_RUN" \
  --arm baseline --gpus 0,1 \
  --async-validation-gpu 1 \
  --steps 10000 --global-windows-per-step 2 --microbatch 4 \
  --learning-rate 1e-5 --seed 42 --dry-run

# M1：仅记忆模型，GPU2/3双卡，验证并行并共享GPU2。
./train/run_groot_train.sh "${train_common[@]}" \
  --output "$train_output_root/NEW_MEMORY_RUN" \
  --arm memory_v1 --gpus 2,3 \
  --async-validation-gpu 2 \
  --steps 10000 --global-windows-per-step 2 --microbatch 4 \
  --learning-rate 1e-5 --seed 42 --dry-run
```

以上命令带 `--dry-run`，仅预览，不启动训练；实际运行时更换为自己的全新输出目录，再去掉该参数。三个示例按需选择，不必都执行。单卡只需把 `--gpus 0,1` 改为 `--gpus 0`（M1同理），保留global-windows-per-step=2可保持每步总决策数相同。

```text
# 参数说明：按需添加或修改，路径建议使用绝对路径。
--dataset 数据集目录              # 必填，已转换的同一任务200条示教，支持Task0–49
--checkpoint 初始权重目录         # 必填，两组同时训练时共用起点；不等同完整断点恢复
--split 划分JSON路径              # 必填，固定180训练/20独立验证Episode，无重叠
--output 新运行目录               # 必填，日志/配置/曲线/权重均归此处；新训练用新目录
--arm memory_v1                   # memory_v1 / baseline，默认memory_v1；切换训练模型
--gpus 2,3                        # 当前组使用的物理GPU编号；单卡示例：--gpus 2
--paired                          # 同时训练M1和对照；arm须为memory_v1
--baseline-gpus 0,1               # paired时必填，对照GPU不能与M1 GPU重叠
--async-validation-gpu 1          # 单组/paired均可启用独立异步验证，可与训练共享GPU
--baseline-validation-gpu 0       # 对照验证GPU；不填则两组验证共用上一个参数指定的卡
--steps 10000                     # 每组优化步上限，默认10000；正式模式须为1000的正整数倍
--global-windows-per-step 2        # 每组每优化步的总窗口数，默认2；须能被各组GPU数整除
--microbatch 4                    # 每rank一次动作头计算的决策数，默认4；不是优化步总批量
--learning-rate 1e-5              # AdamW固定学习率，默认1e-5，无学习率调度
--seed 42                         # 训练随机种子，默认42；公平对照保持相同
--resume-step 1000                # 恢复原output最新完整断点，单组/paired均可，要求异步验证
--train-only                      # 单组工程短测，允许少于1000步；不验证也不保存checkpoint
--source-dir 源码根目录           # 可选，含src/v1_groot_memoryvla的目录
--python-path Python完整路径      # 可选，覆盖自动选择的Python环境
--dry-run                         # 只打印启动计划，不加载模型、不创建输出、不启动训练
--help                            # 查看帮助；不带参数运行也显示帮助
```

每条窗口固定32次决策，global-windows-per-step=2即每组每优化步64次决策；microbatch只影响拆分计算。保存间隔1000固定在当前启动器中；异步验证每1000步，单组串行验证每500步。目前无用户级 `--checkpoint-interval`、`--eval-interval` 或W&B开关。Eagle冻结、AdamW weight_decay=0.01、clip_grad_norm=1及既有记忆结构保持当前实现。`--resume-step` 必须匹配对应组latest.json及完整优化器/各rank状态，数据、起点、seed、卡数、批量和学习率与原配置一致；steps允许延长总优化步上限，但不能小于原上限且必须超过断点。续训前须等旧控制器和验证器退出。原始ddp.json保留，续训计划另存config/ddp_resume_step-001000.json；从某个模型权重重新开始训练使用checkpoint，不是完整续训。

脚本前台运行，Ctrl+C中止。输出目录下 `logs/memory_v1_torchrun.log`、`logs/baseline_torchrun.log` 看训练step/Loss，`logs/*_validation.log` 看异步验证，`memory_v1/metrics/metrics.jsonl`、`baseline/metrics/metrics.jsonl` 保留逐步指标，`plots/` 存曲线。

本轮已完成；以下参数说明用于理解/复现，新训练必须换新输出目录。

| 参数 | 本轮实际值 |
| --- | --- |
| 起点 | 官方GR00T-N1.6-BEHAVIOR1k，两组相同动作专家初始权重 |
| 数据 | Task21 200示教；config/data_split.json固定180训练、20独立验证，seed42 |
| 冻结/训练 | Eagle全部冻结；原动作专家全参数更新，非LoRA；M1另更新记忆模块 |
| 可训练参数 | 对照1,418,624,000；M1 1,527,984,416 |
| 精度 | 冻结Eagle BF16；可训练参数FP32，前向/反向BF16 autocast |
| 卡数 | M1用2/3双卡，对照用0/1双卡；两组各自DDP，同时训练 |
| 全局批量 | 每组每次优化2条32决策窗口=64决策；每rank一窗，microbatch4，累积8块后更新一次 |
| 时序 | stride8环境步，动作监督horizon32，23维有效动作；同Episode跨窗延续detach记忆，严禁跨Episode共享 |
| 优化器 | AdamW，lr=1e-5固定无衰减，weight_decay=0.01，梯度范数裁剪1.0 |
| 预算 | 各10000优化步；13259窗口/遍、6630优化步/遍，约1.51遍数据 |
| 保存/验证 | 每1000步保存；独立进程完整验证20Episode，训练继续；M1验证GPU1、对照验证GPU0 |
| 记忆 | 感知256D/认知2048D，容量16，2层检索，时间编码、Gate、update_fused=True、ToMe；32决策梯度窗口 |
| 日志 | 每步JSONL及控制台日志；W&B未启用 |

常用启动器参数为--dataset、--checkpoint、--split、--output、--gpus、--paired、--baseline-gpus、--steps、--global-windows-per-step、--microbatch、--learning-rate、--seed、--async-validation-gpu、--baseline-validation-gpu。保存间隔1000是当前启动器内固定值。实际恢复启动参数（含两组完整torchrun argv）保存在本轮config/ddp_resume_step-001000.json；config/resume_remote.py记录当时r4源码、环境和控制器启动过程，是历史恢复脚本，不要对已完成run重复执行。最初config/ddp_launch.json中M1旧eval_interval=500，实际恢复后两组均为1000，以恢复记录为准。

## A100 Policy Server 启动命令


统一入口为 evaluation/scripts/run_groot_server.sh，本机和A100均已放好，原scripts目录入口已移除。只需明确指定三个参数：--checkpoint、--gpu-index、--port。没有模型模式或自动选模；对照/M1按checkpoint中的memory_enabled配置加载。默认后台运行并打印日志查看命令，只启动模型服务，4090仿真仍使用原评估入口。

```bash
# 登录A100后执行；可自由更换checkpoint、GPU、端口
cd /nvme3/projects/memory_vla
bash evaluation/scripts/run_groot_server.sh \
  --checkpoint /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/V_1_mem_task21/memory_v1/checkpoints/step-003000 \
  --gpu-index 1 \
  --port 5561

# --dry-run仅打印命令，--foreground前台运行
bash evaluation/scripts/run_groot_server.sh --help
```

A100 Task21完整权重分别在 `/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/V_1_mem_task21/memory_v1/checkpoints/` 与同层 `V_1_bas_task21/baseline/checkpoints/`。每组均有10个完整权重目录，从step-001000到step-010000、间隔1000，每个均有config.json及两份safetensors分片。独立验证候选为对照step-002000、M1 step-003000；可以手选任意其他步数，不会自动替换。selected_checkpoints.json仅记录选模路径，没有另复制best权重。4090输出目录仅镜像轻量日志/曲线/指标，不包含这些完整模型。

默认模型源码为本轮A100 resume_r4部署、Python为envs/n1d6-p0，--source-dir和--python-path可覆盖。训练checkpoint的服务日志归所属run/logs，文件名包含组名、checkpoint步数与端口，例如policy_memory_v1_step-003000_port5561.log；--log-dir可指定其他目录。其他权重默认日志归/nvme3/data_wyh/outputs/memory_vla/policy_servers/logs/。端口占用直接退出，不替换现有服务。旧官方5555/5556服务现已按用户要求停止。启动器此前完成参数检查和A100三种手选checkpoint dry-run（含非最佳step10000）、必填参数/已占端口拒绝检查，尚未用新入口启动模型或评估。

## 项目架构

```text
memory_vla/
├── pyproject.toml                         统一打包源码
├── assets/datasets/
│   ├── b1k2025/                           原始示教，只读
│   └── b1k2025_groot_v1_smoke/            已转换的完整 Episode 210020
├── checkpoints/groot_n1d6_behavior1k/      固定 revision 的官方初始权重
├── outputs/v1_groot_memoryvla/             当前 v1 训练、检查及 checkpoint 产物
├── baselines/groot_n1d6/                   固定原版基线入口
├── train/run_groot_train.sh               统一用户训练入口，参数选择M1/对照及GPU
├── evaluation/                            Baseline / V1 / V2 公共评估
├── scripts/groot_policy_lifecycle.py       公共评估端携带 session/env/episode
└── src/v1_groot_memoryvla/
    ├── README.md                          用户说明、架构和运行方法
    ├── handoff.md                         进度交接，每个窗口结束前更新
    ├── LICENSE
    ├── train.py / serve.py                本版训练与策略服务入口
    ├── configs/
    │   ├── model/gr00t_n1d6.py            可保存的记忆配置；模型默认关闭记忆
    │   └── finetune_config.py             v1 训练 CLI；默认启用记忆
    ├── data/
    │   ├── convert_b1k.py                 256 维原始状态 → 82 维具名状态
    │   ├── prepare_task21_formal.py        200条并行转换、固定180/20划分、官方统计
    │   └── dataset/                       history_dataset + episode_windows 连续Episode窗口
    ├── model/
    │   ├── gr00t_n1d6/                    模型、processor、pipeline
    │   ├── modules/                       Eagle / 原动作头组件
    │   └── memory/
    │       ├── bank.py                    SE、双层检索、时间编码、Gate、相邻合并
    │       └── adapter.py                 双库和增量残差回注
    ├── experiment/
    │   ├── first_training.py              单 Episode 32 决策分块训练、曲线和固定验证
    │   ├── task21_formal.py               正式单组训练、完整验证、错误历史对照
    │   ├── launch_task21_formal.py        双组调度、共同早停、最佳权重指针
    │   ├── task21_ddp.py                  独立多卡worker、Episode固定分配、全局梯度归一化
    │   ├── launch_task21_ddp.py           DDP单组/配对启动、明确服务共卡与M1续跑接管
    │   ├── task21_async.py               独立验证队列、共同早停与保持M1的切换
    │   └── ...                            原 Trainer、组采样、保存
    ├── policy/
    │   ├── gr00t_policy.py                模型推理和动作解码
    │   └── memory_state.py                session/env/episode 与决策时钟
    ├── env/                               本版依赖、安装和上游留档
    ├── .venv/                             本机独立模型环境，忽略提交
    ├── tests/test_memory_v1.py             机制与隔离测试
    ├── audit/
    │   ├── train_official_batch.py         真实数据 + 官方模型训练验证
    │   ├── validate_memory_v1.py           真实模型六项集成验收
    │   ├── analyze_first_training.py       历史时间匹配干预、合并后梯度与结构对照
    │   └── artifacts/                     接入前输入记录
    └── utils/                             视频解码等内部工具
```

模型、记忆、历史 Dataset、训练入口、依赖和 Reset 随版本维护。仿真、任务/实例/seed、评分、录像和结果协议放在根 `evaluation/`，供各版本共用；不复制到本包。训练文件和依赖清单使用真实文件，不需要软链接；大资源和外部环境可以使用软链接。当前 v1 环境实体为本目录 `.venv/`。

## 输出目录管理

[index.json](../../outputs/index.json) 是产物入口。按用户最新要求，仅保留当前训练；此前40步训练、200步诊断、旧1930正式run、试训/验收权重及其日志、曲线、缓存和回收副本已永久删除，无法从回收站恢复。本轮恢复继续使用原run目录，源码固定于resume_r4；排查生成的约62GiB临时权重已删除，证据只保留小日志和JSON。

```text
outputs/v1_groot_memoryvla/
└── training/
    ├── V_1_bas_task0/
    ├── V_1_mem_task0/
    ├── V_1_bas_task21/
    ├── V_1_mem_task21/
    ├── V_1_bas_task34/
    ├── V_1_mem_task34/
    ├── V_1_bas_task40/           # 排队，启动后创建
    └── V_1_mem_task40/           # 每组内部保留config/logs/plots及自身arm目录
```

本地与A100采用同一run归档。当前运行的详细布局与绝对路径见上文；本轮启动检查已移到 `diagnostics/preflight/`，不再散放独立validation目录。原数据集、官方权重、环境和共用评估目录保留；当前四卡训练与原Baseline服务持续运行。后续新增训练使用独立run目录。

## 固定机制与数据流

固定 `perceptual_dim=256, mem_length=16, retrieval_layers=2, history_stride=8, fusion_type=gate, consolidate_type=tome, update_fused=True`。`memory_group_size=16` 保留为旧配置兼容字段，新的 `memory_train_sequence_length=32` 控制训练序列，容量仍为 16；二者不再绑定。时间编码只加入历史 K，V 保留历史特征。每次先读取过去、两层检索、门控融合，再 detach/clone 写入当前融合结果。此次没有增加或改变网络参数的名称、形状。

```text
当前三路相机 + 语言 → 原 processor / Eagle
   ├─ mlp1 视觉特征 [B,3,81,2048]
   │      → 每相机共享 BottleneckSE → P [B,243,256]
   │      → 感知历史检索 / Gate → P_fused
   │      → Linear256→2048(P_fused − P) → 加到原 LLM 图像位置
   └─ 最后有效 user EOS 隐藏状态 C [B,1,2048]
          → 认知历史检索 / Gate → C_fused
          → C_fused − C → 加到原 user EOS 位置
                         ↓
保留原 [B,S,2048]、image_mask、attention_mask
                         ↓
当前本体状态 → 原 vlln / AlternateVLDiT / flow matching 动作头
```

**不直接用 LLM 前的视觉特征覆盖 LLM 后的图像 Token。** 残差投影没有 bias；空历史显式返回当前特征，增量严格为零。未增加本体 Query、辅助损失、额外检索网络或历史序列拼接。

训练 `forward()` 和推理 `get_action()` 共用 `_backbone_with_memory()`，记忆只更新一次，不随四步扩散重复更新。动作损失向 SE、双路检索、时间编码、Gate 和残差投影反传，历史写入切断跨时间梯度。

## 历史、Reset 与保存

新的训练 CLI 默认每个 Dataset item 是同 Episode 的完整 32 决策序列，例如帧 `0,8,...,248`，下一组从 256 开始。按完整序列 shuffle / DDP 分发，序列内按序处理，序列边界清空；观测仅取当前帧，未来 32 帧动作只作为标签。30 Hz 示教下，决策时钟为 `frame_index // 8`。checkpoint 的模型 horizon=50、动作/状态 padding 维度=128；B1K 实际标签为 32×23，状态为 82 维。旧 16 项模式仍可显式选择，但不能用于学习合并后的历史分布。

首轮小训练使用 `first_training.py`：缓存冻结且处于 eval 的真实 Eagle 输出，按每次 2 决策处理完整 32 决策序列。每个微批次立即 backward，保留 detach 的融合历史；第 32 次结束后清库并做一次 optimizer step。参数在序列中途不更新，边界重置与在线 episode 语义一致。此模式不使用随机图像增强，缓存吞吐不能当作完整在线或端到端训练速度。

语言统一为公共 B1K 仿真的任务简称：`task_name.replace('_',' ').capitalize()+'.'`。Episode 210020 的完整示教指令与在线 `Collecting childrens toys.` 都在 Processor 中映射为 `collecting childrens toys`，真实 `input_ids` / attention mask 完全一致，序列长度为 285。映射写进 `processor_config.json`，Policy 重载后仍然生效；原始示教完整文本不改，Baseline 服务不改。

在线由 `options={session_id,env_ids,episode_ids}` 隔离双库，`MemoryState` 每次决策递增一次时钟。单环境调用可以不传 options；多客户端共用服务时使用公共 `EpisodePolicy` 包装器。`reset(options)` 只清理指定环境；新 Episode ID 也会清理该环境的旧 Episode。无 options 的 reset 清空当前策略全部记忆。

第 17 次先读已有 16 条，写入后将相邻项中平均 Token 余弦相似度最高的一对平均合并，保留较早时间戳。两库独立选对。第 18 次开始读取已合并的 bank；32 次序列内每库合并 16 次，合并后读取 15 次，训练和在线沿用同一机制。

`save_pretrained()` 保存原模型、`memory.*` 和全部记忆配置；processor 与官方归一化统计一起保存。运行时 bank / session / 时钟不写入 checkpoint。加载官方基座仅允许新增 `memory.*` 缺失；加载完整 v1 checkpoint 要求无缺失或多余键。

## 运行

从项目根目录执行，本机使用 `src/v1_groot_memoryvla/.venv/bin/python`。以下命令的 `python` 指目标模型环境的 Python；也可以换成上述完整路径。先设置环境变量并选择当前空闲 GPU：

```sh
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src
export NO_ALBUMENTATIONS_UPDATE=1
```

```sh
# 原始数据单独转换，三路视频与 parquet 逐帧对齐
python -m v1_groot_memoryvla.data.convert_b1k \
  --source assets/datasets/b1k2025 \
  --destination assets/datasets/b1k2025_groot_v1_smoke --episodes 210020

# 先验证官方基座真实 loss/backward/AdamW 更新
python -m v1_groot_memoryvla.audit.train_official_batch \
  --checkpoint checkpoints/groot_n1d6_behavior1k \
  --dataset assets/datasets/b1k2025_groot_v1_smoke

# 六项真实模型验收，同时保存并重载完整 v1 checkpoint
python -m v1_groot_memoryvla.audit.validate_memory_v1 \
  --checkpoint checkpoints/groot_n1d6_behavior1k \
  --dataset assets/datasets/b1k2025_groot_v1_smoke

# 首轮小训练：独立新目录，冻结 Eagle 缓存 + 32 决策 + 微批次 2
python -m v1_groot_memoryvla.experiment.first_training \
  --checkpoint checkpoints/groot_n1d6_behavior1k \
  --dataset assets/datasets/b1k2025_groot_v1_smoke \
  --output outputs/v1_groot_memoryvla/training/YYYYMMDD_HHMM_episode210020_s42 \
  --steps 40 --sequence-length 32 --microbatch 2 --learning-rate 1e-5 --seed 42

python -m v1_groot_memoryvla.audit.analyze_first_training \
  --run outputs/v1_groot_memoryvla/training/YYYYMMDD_HHMM_episode210020_s42

# 原 Trainer：global-batch-size 按决策帧计，必须整除 GPU数×训练序列长度
# 完整 32 项一次 forward 的显存高于上述缓存分块模式。
python -m v1_groot_memoryvla.train \
  --base-model-path checkpoints/groot_n1d6_behavior1k \
  --dataset-path assets/datasets/b1k2025_groot_v1_smoke \
  --embodiment-tag BEHAVIOR_R1_PRO --global-batch-size 32 \
  --memory-train-sequence-length 32 \
  --dataloader-num-workers 0 --max-steps 1 --save-steps 1 \
  --learning-rate 1e-5 --warmup-ratio 0 \
  --output-dir outputs/v1_groot_memoryvla/training/YYYYMMDD_HHMM_trainer_s42

# 启动一个独立 v1 服务端；共用根 evaluation 的评估客户端
python -m v1_groot_memoryvla.serve \
  --model-path /path/to/selected/checkpoint \
  --embodiment-tag BEHAVIOR_R1_PRO --device cuda \
  --use-sim-policy-wrapper --port 5566

python -m unittest discover -s src/v1_groot_memoryvla/tests -p test_memory_v1.py -v
```

默认冻结整个 VLM（包括 `tune_top_llm_layers=0`），训练原动作头和记忆分支。PyAV 索引读取一次 seek 后解码整组，避免官方 ffmpeg 逐帧从视频开头解码导致的长等待。多 GPU / ZeRO 尚未实测；单 GPU 不需要安装 DeepSpeed。

实验参数使用 CLI，实际配置以产物中的 JSON 保存。每次记录初始 checkpoint、数据划分、seed、冻结设置、目标帧数、训练步数、优化器和产物路径；对照实验保持同样的目标帧与训练预算。原 Trainer 输出周期性 checkpoint；首轮小训练只保存 `checkpoints/final/`，包含完整模型、processor 与统计，不含恢复训练所需的 optimizer 状态。

## 环境与源码来源

当前独立 `.venv` 为 Python 3.10、Torch 2.7.1+cu128、Transformers 4.51.3、Diffusers 0.35.1、FlashAttention 2.7.4.post1。训练使用 FP32 参数与 BF16 autocast，冻结的 Eagle 为 BF16；Policy 推理将整模转为 BF16。时间采样保持 CPU FP32。官方单批次训练峰值约 30 GiB，v1 验收约 27.5 GiB。

`env/requirements.txt` 是单 GPU 依赖清单；`requirements-distributed.txt` 是可选 DeepSpeed/ZeRO 依赖，需要 CUDA Toolkit / nvcc；`requirements-deploy.txt` 用于可选导出。`env/upstream.pyproject.toml` 与 `env/upstream.uv.lock` 是上游留档，包含原 workspace 成员，不能直接在本项目 `uv sync --locked`。

新环境安装示例（Linux x86_64、Python 3.10，从项目根目录执行）：

```sh
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-deps \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
python -m pip install -r src/v1_groot_memoryvla/env/requirements.txt
python -m pip install -e . --no-deps
```

其他 Python/平台使用匹配 wheel。模型环境与根公共仿真环境独立，通过策略协议连接。

源码来自 `NVIDIA/Isaac-GR00T` 的 `n1d6`，固定提交 `9b37aa1ce69c73c6d165233fa88128283bba4508`；原版在 `../references_code/Isaac-GR00T/`，保持干净。官方 BEHAVIOR1k checkpoint 固定 revision `300db814db8ab5dd010026d5631f280048d06b91`。本版沿用原参数命名并增加 `memory.*`，保留上游版权和 LICENSE；MemoryVLA 方法参考在 `../references_code/memoryvla/`，具体校验信息见交接文件。

## 结果与下一步

当前M1与无记忆对照均已完成10000步双卡训练及全部独立验证，最低验证Loss分别为0.0027759974和0.0027613138，约0.53%的差异不用于判断任务能力；末尾500步训练Loss分别为0.0028585138和0.0028560320。尚未确认可重复的正确历史收益。本轮闭环评估队列在对照阶段收到SIGTERM后中断，当前未运行，M1尚未启动；历史小规模产物已按要求删除，不再提供旧产物链接。

本轮完整验证按20个独立Episode报告Loss与正常/空/时间匹配错误历史动作误差，保留逐Episode明细；使用选定checkpoint完成B1K闭环后再判断收益。训练状态和路径以本轮status.json及config为准；详细进度见 [handoff.md](handoff.md)。
