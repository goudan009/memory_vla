# V2_1 Stage1 step6000 动作异常排查

状态：DONE（诊断范围完成；长期修复效果未执行）。2026-11-27 17:16。只调查现有6000步权重；新增optimizer更新为0；未修改网络架构或启动正式重训。

## 已确认的原因

1. **底盘指令主要在Action Chunk重新采样时跳变。** 原失败/中断闭环8600个chunk中，边界绝对差分均值0.026494，块内0.005567，比例4.759。最大横向指令跳变0.584295（+0.323694→−0.260601）。这是解码后、控制器执行前的请求，不是反归一化往返错误。底盘指令为无量纲量，经仿真底盘控制器分别乘[0.75,0.75,1.0]得到m/s、m/s、rad/s；该横向跳变对应0.438222 m/s的速度目标变化。
2. **双臂异常关节目标在进入仿真控制器前已经存在。** 左肘最高请求0.507397rad、右肘0.485772rad，资产上限均0.349100rad，分别超出0.158297rad（9.07°）、0.136672rad（7.83°）。同一旧轨迹中，左/右臂执行请求的50941/481320、18917/481320个关节元素越过资产限位；这不是实际物理位置越限比例。最极端右肘请求在step2719，step2720实测为0.349062rad，控制器/物理关节停在限位附近。
3. **相对动作解码本身正确，但不能保证绝对关节目标合法。** 每个未来offset均用当前原始观测q(t)作为锚点，q_target(t+h)=q(t)+delta(t,h)；没有把chunk内部delta做累计相加。闭环中，执行后的新q(t)又作为下一块锚点，所以反复出现的小偏置仍能随物理反馈逐块积累为漂移。处理器裁剪的是归一化delta，不是最终q_target。因此位于上限附近时，合法的归一化正delta仍会产生越限请求，并使策略持续要求向外运动。
4. **部分躯干跳变来自物理状态变化后被新相对锚点继承。** N4/H8的instance3，step176的最后目标1.010747rad，执行后实测第一躯干关节1.175064rad；下一块step184首目标1.175061rad。该次目标跳变主要跟随实测锚点，不能说是网络突然预测了0.164rad相对动作。每chunk前后真值能定位先后关系，尚不能独立区分碰撞、底盘倾斜与具体控制器瞬态。

以上证明存在模型请求不连续、解码后限位缺口及物理反馈传播三个环节；没有把全部异常归因于一个超参数。没有逐物理步高频控制器内部goal记录，也没有覆盖手肘的完整外部相机，不能将所有视觉外翻逐一归因。

## 权重、源码和运行身份

- 原checkpoint：`/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2.1__test01__exp14__run001/run/checkpoints/v2.1__test01__exp14__run001__step006000`。
- 两个真实权重分片已流式读取计算SHA256：`model-00001-of-00002.safetensors`（4,996,585,112bytes）为`9a153cd089c88feaff369f12ef72dbf9f29f6a806986be5cb9a556f00a61da0c`；`model-00002-of-00002.safetensors`（4,414,037,400bytes）为`d225018d9ce2f0d5dd72f3e41a2038f92927f79a51ccd74b586b148a607aa2be`。不是仅用目录名或配置hash代替权重hash。
- 原通用初始化：`/nvme3/huggingface/train_model/GR00T/GR00T-N1.6-3B`。官方任务权重配置参考：`/nvme3/huggingface/train_model/GR00T/GR00T-N1.6-BEHAVIOR1k`。两者训练历史不同，官方任务权重不能直接用来隔离自定义训练循环的因果影响。
- 原训练源码：`/nvme3/projects/memory_vla/deployments/v2.1__test01__exp14__run001/src/v2_1_groot_memoryvla`。按原算法重算digest为`c32ae17197824c733288afbc67bea42cfe0de931440a3633f09cb2e06b9868d8`，与6000步COMPLETE.json一致。整个部署树manifest的digest口径不同，不能据此认定源码被修改。
- 本次记录归入`Output/test_02/experiment_01/baseline_diagnostic/`。历史run/checkpoint原名保留，不移动、不重命名为新run。
- 诊断推理冻结源码：`deployments/inference_13b51a8721ce3cdca2c4d57c616f0923586baf155e9e4e39cb91cf2b031d1395`。
- 原训练、旧5558推理和本次诊断推理的核心GR00T模型、DiT、Eagle、processor、state/action处理器及model config源文件SHA256分别完全相同；loading.py差异仅V2.1→V2_1文字。原训练快照不含baseline_policy，现用兼容baseline入口保持原动作专家FP32载入。详见`training_evidence.json:training_serving_source_comparison`。
- 官方源码实际路径：`/nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1/gr00t`，该目录没有独立.git，使用`training_evidence.json`中源文件SHA256定位，不冒称目录名为已核验的upstream commit。

## 训练流程与参数

| 项目 | 官方GR00T默认微调实现 | 实际Stage1 | 判断 |
|---|---|---|---|
| 数据 | 分片、随机帧，默认episode_sampling_rate=0.1 | Task40固定180/20；按episode洗牌，两个连续窗口槽位，每窗口最多32决策，观测stride8且固定phase0 | 有效batch数相近，样本相关性和覆盖不同 |
| 全局batch | finetune默认64，accumulation1 | 满组64决策；4rank×microbatch2×每rank8个microbatch；尾窗口会缩小，实际均值61.4465 | 不是batch8，也不是4×64 |
| 观测覆盖 | 随机时间步 | 每epoch固定72886/582460个训练观测锚点，约1/8；动作监督仍含未来0..31的每帧 | 可能降低覆盖；没有证明它单独导致抖动 |
| 学习率 | 默认1e-4，cosine，warmup_ratio0.05 | 恒定1e-5，无warmup/scheduler | 与默认不同；不能直接判为错误或直接调大 |
| AdamW | 默认weight_decay1e-5 | betas(.9,.999),eps1e-8,weight_decay0.01 | decay不同；没有独立实验支持将其列为主因 |
| 冻结 | finetune默认tune_llm/visual=False；projector/diffusion=True | 整个Eagle冻结BF16；原动作专家、projector、VLLN可训练FP32，计算用BF16 autocast | 原backbone没有参与训练 |
| 动作损失 | `(pred_velocity-(action-noise))² * action_mask`，除有效元素和 | 同一分子，在整个双窗口有效元素和上归一化，各rank分担microbatch，最后SUM梯度一次 | 未发现重复平均或漏mask |
| 梯度 | 官方trainer负责累积/同步 | 每个全局update先zero_grad；逐microbatch backward；SUM一次；clip_norm1；step一次 | 源码检查无明显累积倍数错误 |
| 状态监督 | 原动作基线无任务状态头 | Stage1 `model=None`，state loss固定0；状态名仍在元数据中 | 是基线阶段行为，不是状态损失失效 |

6000步共见368679个决策观测，完成5个完整epoch，在第6个epoch执行69次update（日志epoch从0开始）。6000条梯度norm均有限，中位数0.124516，最大3.279170，为裁剪前值。验证flow loss从step0的0.330283到step3000的0.014660、step6000的0.014370；后半程验证收益很小，但该loss不能代替闭环成功率，也不能证明继续加步数一定有效。

原源码依据：`src/v2_1_groot_memoryvla/train.py`中的`initialize/run_update/optimizer_for/stream`、`data/windows.py`中的`schedule/fit_statistics`；固定采样相位在`src/v1_groot_memoryvla/data/dataset/history_dataset.py:frame_indices`；官方`gr00t/configs/finetune_config.py`、`configs/data/data_config.py`、`model/gr00t_n1d6/gr00t_n1d6.py`、`experiment/trainer.py`。真实文件hash和配置见`training_evidence.json`。

## 数据与转换的数值核验

使用checkpoint保存的真实StateActionProcessor和LeRobotEpisodeLoader，对全部200条示教各取5个32步窗口，共1000个窗口。包括180条训练及20条验证。最大物理动作往返误差4.10e−8，所抽窗口均没有目标被minmax裁剪，batch/unbatch解码相同。未将抽样检查表述成全部示教帧认证。

minmax规则为z=2(x−min)/(max−min)−1；decode先裁剪z到[−1,1]再执行逆变换。相对臂/躯干的min/max按未来offset分别存储；逆变换最大线性增益：base0.35、torso0.22525、left_arm1.55122、right_arm1.61542。没有发现度/弧度混用、重复缩放或把delta作为累加速度执行的证据。左夹爪示教动作恒为+1，常量归一化为0后能恢复+1，并非误删信号。

全量180条训练示教的horizon0物理delta另做了分布检查：左肘绝对delta中位数0.000780rad、P99为0.031218rad、最大1.226627rad；右肘分别0.001398/0.084404/1.081292rad。保存minmax的左肘h0跨度为[−1.226627,+0.798524]，对应单位归一化误差约1.01258rad的物理增益，较常见微动作很大。大值来源已记录到episode/frame（左极值episode401060/frame2095，右401370/frame445），尚未将这些真实示教瞬态判为错误标签。该尺度会让小归一化误差变成可见的关节偏差，是量化到的误差放大因素；不是反归一化公式错误，也没有擅自将既定minmax改为percentile。自训统计量确由180条训练集拟合，与自身权重配套；官方权重统计范围不同，不能拿来替换。

还将真实推理记录`request_000000.npz`与同一已执行chunk严格匹配：GPU0两个环境的左右臂及躯干原始输入qpos与chunk前仿真qpos最大误差均为0，见`events/events.json:online_input_alignment`。该证据排除了这两个真实样本的关节切片错位，不代表所有模态所有帧已全量认证。

仿真双臂/躯干`JointController`为绝对位置(rad)、`use_delta_commands=False`、输入输出limits为None且机器人`action_normalize=False`；只在processor中相对→绝对一次。配置来源`evaluation/sim/BEHAVIOR/og_teleop_cfg.py:R1_CONTROLLER_CONFIG`和`og_teleop_utils.py:generate_robot_config`。258维实时状态与256维示教源格式的差别包含额外抓取字段，需按现有converter切片，未据维度不同虚构错位bug。

## 推理设置与六组对照

官方任务权重、通用初始化和Stage1保存配置均为Euler积分4、模型horizon50、padding维128、有效前32×23，当前观测1帧，三相机输入；use_relative_action=True、use_percentiles=False、clip_outliers=True。实际服务精度有差别：官方Gr00tPolicy将整个模型转BF16；V2_1 BaselinePolicy保留动作专家FP32参数并使用BF16 autocast，与本次训练精度契约一致。不能称所有实际设置完全相同；尚未单独量化精度差异的行为影响。

诊断服务A100端口5564/5565/5566分别固定积分4/8/16；原5557/5558服务保留。闭环统一dual2（两张4090各2环境）、Task40实例0/1/2/3、seed0 fixed、每环境256物理动作、每2步视频1帧。H8/H16分别约每0.267/0.533仿真秒重采样。各组闭环不使用严格相同噪声流；同输入/同噪声的独立内部检查用于分离积分影响。

六组全部完成：24个有界短回合、6144物理动作、576chunks；每组4环境×256动作。完整汇总见`matrix_summary.json/csv`，24段视频见`video_index.csv`。这些是有界故障复现，不能把256步超时当成正式任务成功率。

| 积分N / 执行H | 作业ID | 底盘全部相邻步差分均值 | 底盘边界差分均值 | 左臂全部相邻步差分(rad) | 右臂全部相邻步差分(rad) |
|---|---|---:|---:|---:|---:|
| 4 / 8 | cd6285f573aa | 0.014009 | 0.058622 | 0.002449 | 0.003284 |
| 4 / 16 | bcede46e7bb4 | 0.012157 | 0.091122 | 0.002599 | 0.003592 |
| 8 / 8 | cbcb177c8c47 | 0.014400 | 0.057893 | 0.002466 | 0.003389 |
| 8 / 16 | 508ed90b37fa | 0.012798 | 0.086375 | 0.003121 | 0.004249 |
| 16 / 8 | 572d58f1ae87 | 0.014559 | 0.062517 | 0.001849 | 0.002586 |
| 16 / 16 | 866636e0425e | 0.011985 | 0.087727 | 0.001772 | 0.002566 |

全部相邻步均值同时包含块内和块边界，按真实执行步数计算；底盘为无量纲速度指令。H16每256步只有15个边界，H8有31个，不能只比较边界频次或只看boundary/interior比例。N16/H8相比N4/H8左/右臂总体差分小24.5%/21.2%，底盘并未改善；N16/H16比N4/H8左/右臂小27.7%/21.8%，但底盘单边界跳变更大。N8无稳定收益，H16双臂也没有跨积分设置的稳定收益。结论仅适用于这四实例、单seed、早期256步，噪声流未严格配对，无统计显著性或长期成功率结论。

同输入检查已完成三批来自真实闭环的RGB/状态，固定Torch seed42，分别Euler4/8/16，原服务积分设置及RNG在RPC后恢复。三批块内平均差分：base约0.006465/0.006983/0.006702；左臂0.003067/0.003719/0.001774rad；右臂0.004168/0.004735/0.002708rad。N16在这三个输入上臂内更平滑，N8不单调优于N4；这不是完整闭环效果结论。原始normalized/decoded输出见`paired_integrators/`。

进一步固定同一个真实RGB/状态输入，仅将初始噪声seed换为42/43/44（积分仍4），底盘同一预测chunk的最大差异为0.164464–0.265631，左右臂分别0.100676–0.136034rad、0.110193–0.267963rad。说明该权重在不改变观测时也会随重新采样产生明显不同的动作，是切块抖动的一个直接来源；不把三个seed视为成功率统计。见`paired_integrators/noise_sensitivity.json`。

H8真实客户端policy_get_action均值（每请求两个环境、含RPC和按需记录）随积分4/8/16约为0.232–0.241 / 0.321–0.344 / 0.449–0.513秒；16步约为4步的两倍。每组场景初始化约432–482秒，故初始化主导短复现墙钟时间。原performance记录已收录`matrix_summary.json:per_gpu_performance`，没有把客户端耗时宣称为纯模型内核耗时。

## 控制执行与图像证据

旧轨迹每80物理步记录一次实测qpos，能对齐860个chunk末快照：左右臂绝对跟踪误差均值0.001082/0.001259rad，P95为0.007022/0.005952rad。多数目标被较好跟踪；极端越限请求会被限制，稀疏采样不能排除瞬态、接触或所有执行误差。新短复现每chunk都有before/after真值。

instance0/seed0的左肘从step80的实测0.037031rad逐步升至step560的0.326942rad，随后step640实测0.349102rad、最后请求0.357144rad，显示模型请求与实测一起向限位漂移。该回合首个左肘越限请求位于step579。六组每环境只跑到256步，因此能比较早期抖动，不能证明任何组消除了后续手肘漂移。

`existing_trace_analysis.json`给出每关节请求范围、越限次数、首次事件、块内/边界/重叠区差分；`events/events.json`给出极端请求的实例、seed、物理动作步、最近实测及视频帧偏移。极端左肘视频对应step4175（距请求step4183为−8帧动作步），右肘step2711（距step2719为−8）；视频`frame_index`来自原记录，未将视频帧与仿真帧误当一一对应。三相机画面不能直接完整展示手肘外翻，物理限位归因以关节数值为依据。

## 是否需要重训、后续应改什么

**当前没有证据要求丢弃6000步权重立即重训。** 归一化/相对动作codec、输入关节切片抽样、损失及梯度累积均未发现可直接解释全部异常的低级错误。首先应保留权重，处理输出连续性与绝对物理约束，再用同协议闭环判断剩余的策略质量问题。

建议修改的位置和性质（本轮仅诊断，以下未应用到策略默认行为）：

- `StateActionProcessor.unapply_action`之后、发送仿真之前：明确最终绝对q_target的真实关节限位，并记录触发量。当前仿真实测已表现出限位约束；若控制器已经执行等价裁剪，再加同样裁剪主要改善接口约束和可观测性，不能据此承诺手肘姿态或成功率改善。模型持续预测向外的delta仍属于策略问题。
- 执行端chunk衔接：若采用跨块融合/限加速度，臂/躯干先转换为同一绝对参考再融合，底盘按速度指令处理；不要直接融合不同观测锚点的relative delta。需另做闭环验收，不能把平滑当作任务成功修复。
- 积分16是缓解早期双臂块内变化的候选，增加约两倍客户端推理耗时；没有修复底盘或证明消除长期手肘漂移。执行16不建议直接作为通用修复。为保留原结果可比性，本轮不改默认4/8；若后续验证缓解措施，优先在现有6000步权重上验证N16/H8及绝对参考下的块衔接，不混入重训。
- 如果上述执行约束后仍无任务推进，才讨论保持通用GR00T、180/20和现有架构的重新微调：优先核对Stage1固定phase0覆盖及连续样本相关性，并将LR/scheduler/weight_decay与官方默认差异作为需受控比较的候选。不能从当前loss或一次失败直接决定升LR、改Batch、加训练步数。本轮不修改这些研究参数。

## 可复现入口及验证

```sh
python3 scripts/collect_baseline_training_evidence.py --hash-weights --output Output/test_02/experiment_01/baseline_diagnostic/training_evidence.json
# 真实保存processor的roundtrip需在含数据和checkpoint的CPU环境运行：
python scripts/baseline_action_audit.py --checkpoint <确切6000步目录> --dataset <Task40数据目录> --output processor_audit.json
# GPU服务和短闭环经原工作台Manager的prepare/start；不要在已有同名服务仍运行时重复services：
python3 scripts/run_baseline_fault_diagnostic.py services
python3 scripts/run_baseline_fault_diagnostic.py simulation 4 8
python scripts/analyze_baseline_action_trace.py evaluation/V_2_1/baseline/task40_eval01 --output existing_trace_analysis.json
```

诊断RPC的RNG/积分恢复（含异常）、可选参数转发及已有推理/启动回归已通过78项，另新增H16双卡命令及非法窗口检查通过。扩展测试曾遇到与本改动无关的registry fixture（cached mock不接受force），未改用户正在修改的UI/registry文件，不称全仓测试通过。仿真初始化存在资产hash不一致告警，保留在日志中；机器人关节映射、codec核验及所报告请求数值不依赖用它推断bag_inside。

定向复跑：工作台启动/推理70项通过（1项无关registry暂排除），诊断RPC及V2_1推理9项通过。诊断代码提交`adc1510`，核心默认推理仍为checkpoint内Euler4、执行8帧。

资源：真实设备为四A100各81920MiB、两4090各49140MiB。A100诊断服务单份约9.5–10GB，仿真加载后每4090约23–28GB（包含其他项目进程）；没有达到A10080%、4090/70%目标，不为诊断复制模型或改变科学batch来占显存。两4090各两个环境保留，其他项目进程不动。初始化占每组大部分墙钟时间，A100在场景载入时低利用率。`resource_snapshot.json`记录快照，不是连续显存时序，不能声称达到目标时长或完整QPS测量。

本次可选参数尚未被已启动的HTTP工作台旧模块加载，因此直接调用同一工作台`InferenceManager/LaunchManager.prepare/start`，沿用原worker、GPU队列、身份解析和输出；实际job/PID、preview和终态在本目录归档。HTTP `/api/services`能核对真实仿真进程和端口，但其`/api/launch/jobs`不自动纳入这些临时Manager进程的内存记录；不能宣称六组均可从旧HTTP jobs列表查询。没有为刷新模块重启用户工作台。

六组终态、配置、动作及视频索引已归档。确认依赖结束后经工作台InferenceManager停止本次专用5564/5565/5566服务，停止记录为inference4/8/16_stop.json；原5557/5558服务保留。诊断RGB/action原始记录总量约0.2GB，短回合视频/真值数十MB，没有新建大型训练预检权重，无需删除正式checkpoint。保留这些故障证据供复查。

最终判断：已定位异常进入控制器前的模型请求与块衔接来源，未发现要求立即重训的转换/梯度低级错误。权重确实存在长期策略漂移；限位约束和增加积分不能保证修复该策略。是否需要为任务质量重训仍需后续长期闭环区分，本轮证据不支持直接启动或承诺免于重训。尚未独立验证精度差异、接触瞬态、单个训练超参的因果贡献；以上为明确剩余问题。
