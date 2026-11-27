# V2_1 Stage1 原生训练重训接入说明

> 当前状态：**BLOCKED_WORKBENCH_INTEGRATION**。上一提交 724d3c0 的独立CLI、手写输出地址、非规范 checkpoint 名称及非严格续训方案不符合仓库执行规则，已撤销其启动授权。候选科学配置保留，不能直接执行；原有训练/推理/评估记录保持不变。

## 唯一执行依据

先读 [AGENTS.md](../../AGENTS.md)、[robot-vla-experiment-ops Skill](../../.agents/skills/robot-vla-experiment-ops/robot-vla-experiment-ops/SKILL.md) 和 [工作台手册](../workbench/agent.md)；接续实验读取 [当前任务书](../../Output/test_02/experiment_01/task.md)、[plan-v004](../../Output/test_02/experiment_01/workflow/plan-v004.json)、progress 及诊断证据。任何本机环境及存储约定还需本地 Agent 读取 ~/.agent-rules.md。

## 身份与路径

- 已登记方法 V_2_1，Test02 / Exp01；本次独立重训是新 run，不是新方法版本。完整身份格式 `v2_1__test02__exp01__runNNN`；NNN仅为占位，不是已分配编号。查实际工作台 jobs、目标设备训练目录、deployments、回收站及已保留身份，避免重用，显式锁定。
- 预检使用同一新谱系的规范 `__preflightNNN` 后缀；正式作业由工作台 preview/start 分配路径。作业根为工作台实际 TRAINING / run_id；实际训练目录为作业根/run；正式 checkpoint 为作业根/run/checkpoints/<run_id>__stepNNNNNN。示例仅为结构，不指定任何绝对目录。
- 数据、初始化权重、标签、Python、源码快照和 GPU UUID 从本机规则、工作台 options/preview、历史任务证据核实。历史 `v2.1__test01__exp14__run001__step006000` 保留原路径，仅用于历史划分和对照，不用于新初始化。不得复制或改名历史产物。
- 实验文档维护在 `Output/test_02/experiment_01/`；执行证据、数据清单、job ID、配置、日志和实际 checkpoint hash 依既有规则归档。工作台是唯一提交、暂停、续跑、队列和资源入口。

## 已确认训练配方（不变）

官方原生 GR00T N1.6 训练器及随机分片采样；通用 GR00T-N1.6-3B 起点、Task40原180/20、seed42、四A100 DDP、microbatch2×accum8、全局64、6000 optimizer steps、LR1e-4、Cosine、300步warmup、WD1e-5、AdamW fused、clip1、BF16、全部Eagle冻结、动作专家/projector/VLLN训练；预测50、有效32×23、积分4、执行8、保存间隔1000。不是B1K冠军配方。候选相对动作归一化：只用训练集拟合逐offset/关节原始Min-Max和Q01–Q99，范围缩小≥4倍、宽度≥0.02rad、裁剪≤2.1%、>0.1rad裁剪≤0.1%、平均裁剪误差≤0.001rad才接受候选，否则Min-Max；保留原始统计，不删除示教。阈值为已提出的保守工程选择，实际数据需检查。

## 本地 Agent 接入与验收

1. 先通过工作台只读 services/options/jobs 核对身份、已有作业、资源及路径，记录当前实验计划与真实状态。
2. 接入现有 `train/v21_workbench.py` 的专用 V_2_1 recipe（`native_random_v2`）。将候选逻辑放在版本源码快照可同步范围，沿用 workbench controller、queue、日志、status、metrics 和现有 checkpoint 发现/暂停恢复机制，不新增私人调度或直接CLI；保留旧recipe默认行为。当前该recipe明确拒绝预览/启动，不会静默回退。
3. 解决上游 Gr00tTrainer 的 resume 重置种子与忽略数据跳过问题，保证保存/恢复模型、optimizer、scheduler、适用scaler、各rank RNG、采样游标、累计step及hash。完整 checkpoint 使用规范命名和现有 COMPLETE/training_state 合同；原生内部 checkpoint-N 不对外发布。不得用仅重载权重冒充严格续训。
4. 在同一工作台执行真实四卡≤100步工程预检、暂停释放资源、同run恢复后实际继续更新；归档证据并依规则清理临时产物。核对真实训练配置、有效batch、冻结梯度、归一化统计和动作解码。只有通过后才使用同一配置 preview/start 正式6000步。
5. 完成后用既有工作台推理及 Task40 闭环（实例0–9，10有效回合，seed0 sequence，原协议），与旧基线和官方参考对比。内部检查仅服务具体故障，不派发独立离线MAE评估。结果保存实际路径和 ID；不自动重启旧Stage2。

候选文件 `scripts/stage1_native_prepare.py`、`scripts/stage1_native_train.py` 的辅助函数保留供接入；直接执行路径已关闭。此前5项CPU测试仅覆盖候选统计和配置，不能作为真实工作台、A100或严格暂停恢复验收。没有启动训练。
