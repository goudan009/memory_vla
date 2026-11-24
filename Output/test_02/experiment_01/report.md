# V2 / Test 02 / Experiment 01：真实标签单轮验证

status: PARTIAL（计划内执行与归档完成；状态学习未通过，严格配对闭环因恢复不一致未验收）。核对时间：2026-11-24 05:06 +08。

唯一run001完成1000步训练及末尾完整重载，20条留出示教、100个机制窗口全部评分。started预测全部为负，F1=0，未学会正类；检索、门控、DiT条件和动作有数值响应，尚不能证明动作更正确。24/24恢复尝试全部结束，均因原始RGB无法复现判无效，有效配对0/12。工作台评估终态complete仅表示协议执行结束，不表示方法有效。没有追加训练、改阈值、换起点或放宽恢复门禁。

## 身份、标签和固定配置

按[计划v007](workflow/plan-v007.json)及 `robot-vla-experiment-ops` Skill执行。用户明确授权从原定Ada改为本机GPU0的4090，UUID `GPU-49fbbb98-a702-2adb-f148-a55b95f77252`，实测49140MiB。GPU1既有仿真及四卡A100既有训练保持运行。

初始化为V1 / test01 / exp11 / 6000步：`checkpoints/v1__test01__exp11__step006000`；旧作业名V_11不表示模型V11。V2独立实现状态head、事件优先保留、逐层Query状态投影及原gate logit状态项，保留原容量、全库注意力、先读后写和残差路径。冻结视觉/语言主干、动作专家、感知库和原缩放，只更新认知记忆分支。完整参数/冻结清单见[实际配置](train_runs/run001/train_config.json)。

官方200条成功示教共647555帧。原始HDF5中20条转换记录不在demo_0，现按完整动作逐值相等唯一匹配录制组并核对T动作/T+1状态，未截断或补标签。200条全部认证，只启用 `microwave_hjjxmi_0 / non_kin.ToggledOn.value` 对应started；door_open、bag_inside禁用。state[t]对齐obs[t]，切换边界±8帧unknown并mask。[认证清单](evidence/certified_started_labels.json)及三条图像审计：[400010](evidence/label_alignment_400010.png)、[400020](evidence/started_alignment_400020.png)、[403000](evidence/started_alignment_403000.png)。400010全帧回放不一致数0。

| 固定划分 | episodes | 正帧 | 负帧 | unknown |
| --- | ---: | ---: | ---: | ---: |
| 训练 | 180 | 19098 | 560780 | 3060 |
| V2更新留出 | 20 | 2314 | 61963 | 340 |

初始化V1已见全部200条，因此这里只是本轮V2更新留出，不是端到端未见数据。失败轨迹仅用于评估，不进入训练。

单轮固定1000 optimizer steps；AdamW、lr1e-5、WD0.01、constant、seed42、clip1；BF16 autocast，FP32认知参数。名义batch为1卡×microbatch2×最多16累积=32个连续观测决策，episode尾部按实际数量计。原动作流匹配loss加masked BCE，λ一次定标 `5.4601637888766955e-05`；无搜索、无第二轮训练。输出horizon50，参考32步，实际执行前8步。

## 训练与工程验收

工作台训练作业 `20261124-005439-a051a4af` complete。实际执行commit `98311640040eab06e3107dee6e495bfc74478139`；runner SHA256 `307489cf8811879124c7a04015cfa0f37d26b192de4d8024452090f20184977e`。汇总工具版本另记，不冒充训练版本。

| 项目 | 实测结果 |
| --- | --- |
| 步数及实际数据 | 1000/1000；30956决策，75条训练episode，未完整遍历180条 |
| 状态监督决策 | 有效30801，其中正例983；连续决策有关联 |
| 耗时 | 优化步均值4.267秒；含前检/保存/重载4440.88秒 |
| 动作loss | 前100步均值0.004178，后100步0.003535 |
| 状态BCE | 前100步0.756446，后100步0.232176；不代表正类学会 |
| 梯度/更新 | 前检head、两层状态Query、状态gate、原Query/融合均有限非零且更新 |
| 冻结/异常 | 冻结参数hash不变；未观察到NaN/OOM |
| 完整HF重载 | 前检与末尾的action_loss_unreduced、state_logits、state_feature、retrieved_memory、memory_gate最大差均0 |
| 显存 | peak allocated 12.685GiB；未达默认70%目标，未空占显存 |

[训练监测](train_runs/run001/train_monitor.csv)、[汇总](train_runs/run001/training_summary.json)、[曲线PNG](train_runs/run001/training_curves.png)、[曲线PDF](train_runs/run001/training_curves.pdf)、[权重索引](train_runs/run001/checkpoint_index.csv)。根目录train_monitor/resource_summary为历史人工3步自检，不是本轮训练。

末尾权重 `outputs/benchmarks/web/20261124-005439-a051a4af/v2__test02__exp01__run001__step001000/`，COMPLETE两分片SHA256：

- shard1：`2f8fc0243231ea3ee042043dbd37c6edb7627ccb2a8b9ba0453b70d041ef7073`
- shard2：`abcb6e2b302c9120c8453373c2bd77f260e6e665d682ccf6409c38782ee23929`

历史198项CPU回归及13个subtest通过，录制组对齐修复另通过30项相关回归。集中评估Path字符串、BF16诊断副本问题分别经9项、6项相关测试验证，当前实跑已越过原失败点。失败作业 `20261124-030822-02da589c`、`20261124-031012-0c7259b6` 保留证据，均未追加训练。

## 集中评估：状态、记忆和动作

评估作业 `20261124-031205-a32b5587`，产物在 `outputs/benchmarks/web/20261124-031205-a32b5587/`。[协议v002](evidence/evaluation_protocol-v002.json)在看到效果前锁定，SHA256 `fa821b41e3dae7cc4290eb3cefa2443bc560ad38ddc8359373e0dac70bc3729d`。20条留出episode各5窗口；固定RGB/本体/语言/感知库/RNG，干预认知历史；关闭整体引导时按原始前缀重建关闭引导的认知库。

stride8状态评分：TP0、TN7758、FP0、FN287，P/R/F1均0。accuracy=96.43%，与恒0完全相同；恒1基线F1=0.06889。20条episode的F1也均0。真实正类预测概率范围0.125–0.158203，中位数0.146484；真实负类范围0.094238–0.170898，中位数0.127930，分布重叠。本预算下正类未学会，不能拿accuracy或BCE下降证明有效；未按留出结果调阈值。

预测事件0、保留事件0，未取得关键事件保留的正面证据。正常历史系数 `1-g` 均值0.542336。状态整体关闭使注意力slot-mass L1均值1.21889、Top1改变99/100；错历史对应0.11525、99/100。注意力为FP32 Q/K softmax重算，非SDPA kernel逐位权重；整体开关还改变历史维护，不能单独归因于Query投影。

| 相对正常历史的干预 | DiT条件MAE | 归一化动作MAE |
| --- | ---: | ---: |
| 空认知历史 | 0.00141322 | 0.000609976 |
| 错认知历史 | 0.000102583 | 0.000600433 |
| 状态整体关闭 | 0.0000756195 | 0.000597095 |

状态整体关闭的gate MAE=0.000313955，认知读出MAE=0.0358942，原缩放后认知value MAE=0.000383853。原路径有响应，未证明改善。

| 前8步部位MAE | 空历史 | 错历史 | 状态关闭 |
| --- | ---: | ---: | ---: |
| 底盘 | 0.000555592 | 0.000526205 | 0.000555150 |
| 躯干/升降 | 0.000643708 | 0.000643663 | 0.000656632 |
| 左臂 | 0.000550236 | 0.000556869 | 0.000543948 |
| 左夹爪 | 0.00113216 | 0.00111825 | 0.00105968 |
| 右臂 | 0.000572852 | 0.000567586 | 0.000569364 |
| 右夹爪 | 0.00110728 | 0.00105508 | 0.00117067 |

单位为checkpoint归一化坐标，不是米/弧度；100窗口来自20条有关联的episode，不是100独立闭环。完整数值、参考动作误差见[评估汇总](eval_runs/eval001/evaluation_summary.json)、[逐窗口](eval_runs/eval001/mechanism_metrics.csv)、[episode聚合](eval_runs/eval001/mechanism_per_episode.csv)、[身体部位与参考误差](eval_runs/eval001/action_part_metrics.csv)。[离线诊断图PNG](eval_runs/eval001/evaluation_diagnostics.png)及[PDF](eval_runs/eval001/evaluation_diagnostics.pdf)展示概率、真值轨迹和干预响应分布。

## 固定起点恢复与资源终态

原Exp11面板固定12起点，V1/V2各准备按自己checkpoint从原始前缀重建记忆，计划60决策×8执行步。实际V1 12/12、V2 12/12恢复尝试全部结束，24次均INVALID_RESTORE_RGB，在策略动作执行前退出。有效闭环回合0，有效配对0/12；没有执行满60决策的回合。各阶段成功率、置信区间、误切比较及有效回合/小时为N/A，不能写成任务成功率0%。无闭环视频，视频索引仅表头；恢复原图与物理审计仍保留。

首组控制器一致，物体/关节最大误差约1.47e-4，三路RGB MAE为12.0487/7.06565/21.77184（像素0–255）。加载出现USD资产hash警告，但未证明具体因果贡献。[Exp10历史渲染审计](../../test_01/experiment_10/report.md)已记录固定动作、相同物理轨迹而RGB不同；现有物理快照不含完整渲染状态。未放宽RGB门禁、替换缓存首帧、改起点或改渲染配置。

24次控制器状态全部一致；全体位置/四元数/关节最大误差分别0.00197947/0.0000158073/0.00647354。三路RGB MAE范围分别7.55710–15.36464、6.76020–17.28104、17.86176–22.62745。恢复接近不等于策略输入相同，不能据此估计V2阶段净增益。[逐尝试表](eval_runs/eval001/eval_trials.csv)、[恢复审计](eval_runs/eval001/restore_audit.csv)、[证据哈希索引](eval_runs/eval001/restore_evidence_index.json)、[视频索引](eval_runs/eval001/video_index.csv)。

评估工作台耗时6715.94秒，2795个双秒附近的GPU0采样中峰值22506MiB，70%目标达标采样0。该值含策略、仿真及桌面，不是模型独占显存；不能把GPU1/A100其他作业占用算作本实验贡献。独立逐场景启动与CPU初始化/恢复占主要时间；门禁拒绝后没有策略推理，不报告闭环QPS。未为占用率另起无关任务。训练peak allocated为12.685GiB，口径不同。[评估资源汇总](eval_runs/eval001/resource_summary.csv)、[全部采样](eval_runs/eval001/resource_samples.csv)、[跨阶段巡检](evidence/persistent_execution_resource.csv)。

05:04 +08，工作台两评估阶段均complete，worker已退出自有策略与仿真；5598端口拒绝连接，GPU0回到约946MiB，GPU1原仿真继续。模型与100窗口原始数组留在原工作台目录，路径和COMPLETE哈希见[产物索引](eval_runs/eval001/artifact_index.json)。最后一次重新计算两个权重分片哈希与COMPLETE一致。

## 四问与当前验收

| 问题 | 当前结论 |
| --- | --- |
| 状态学会了吗 | 未通过：started正类Recall/F1=0，等同恒0 |
| 记忆变了吗 | 检索/融合有数值变化；事件0，未证明事件机制有效 |
| 动作受影响且更正确吗 | 有约6e-4归一化响应；正确方向与任务效益未证实 |
| Task40哪些阶段更好 | 无有效配对，不能判断；24次恢复核验全部结束且均无效 |

| 验收项 | 最终状态 |
| --- | --- |
| 实现、回归、原始录制对齐修复 | 完成 |
| started可靠标签及180/20划分 | 完成；另两字段未启用 |
| 唯一1000步训练、冻结和完整重载 | 完成 |
| 20条状态评分与100机制窗口 | 完成；状态学习未通过 |
| 12组恢复配对执行 | 完成24次尝试；有效闭环受阻，0对可验收 |
| 阶段改善、事件保留效益 | 未证实；无有效闭环证据 |
| 归档与本次服务释放 | 完成；权重及原始证据保留 |

本次原定预算内的执行与结果归档结束，研究有效性未通过。要改善正类学习需要另行修改配方/预算并建立新实验，要取得严格配对证据则需解决渲染状态恢复；这两项没有通过重复提交或更改本轮锁定条件冒充解决。[验收记录](acceptance.json)与[MANIFEST](MANIFEST.json)列明实际状态。[历史阶段审计](evidence/report-history-before-final.md)仅保留既往故障与进展。
