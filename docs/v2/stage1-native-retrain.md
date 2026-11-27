# V2_1 Stage1 原生训练重训

> 代码已接入既有V_2_1工作台。真实本地CPU回归、A100四卡及暂停恢复尚需验收。不要从shell直接调用训练worker，不要沿用旧顺序配方。

## 唯一执行依据

先读 [AGENTS.md](../../AGENTS.md)、[robot-vla-experiment-ops Skill](../../.agents/skills/robot-vla-experiment-ops/robot-vla-experiment-ops/SKILL.md)、[工作台手册](../workbench/agent.md) 和本机 ~/.agent-rules.md；读取 [task.md](../../Output/test_02/experiment_01/task.md)、[plan-v005](../../Output/test_02/experiment_01/workflow/plan-v005.json) 与本地 progress。当前方法 V_2_1、test02/exp01；核对 workbench services/options/jobs、训练目录、已保留/删除身份及部署后锁定未使用新run，不能从文档猜编号。预检为同一新谱系加 __preflightNNN 后缀，正式run去掉该后缀。历史 v2.1__test01__exp14__run001__step006000 保留原位，只作为划分和效果参考。

## 锁定配置

通用 GR00T-N1.6-3B；Task40原180/20、seed42；4 A100 × microbatch2 × accumulate8 = 64；原生 Gr00tTrainer、官方随机 shard 构造和 mixture schedule；LR1e-4、Cosine、warmup300步（5%）、WD1e-5、AdamW fused、clip1、BF16冻结Eagle/FP32动作专家参数；6000 optimizer steps、每1000保存、预测50×128/有效32×23、积分4/执行8。非B1K冠军实际配方。为严格恢复，使用确定性rank shard occurrence、样本内顺序及增强种子，游标=累计step×16/卡；不使用上游恢复时重置seed的实现。不改变原动作表示、网络结构、记忆模块。

归一化仅训练集拟合相对臂/躯干逐offset、逐关节原始Min-Max与Q01–Q99；范围缩小≥4倍、宽度≥0.02rad、训练裁剪≤2.1%、超过0.1rad的裁剪≤0.1%、平均裁剪误差≤0.001rad才接受，否则原始Min-Max；这些是保守工程阈值，需核对极端样本和实际动作，保留原始统计，不删除示教。验证集不参与拟合。采用保存统计的实际有效min/max，不单纯切换use_percentiles。内部验证loss仅作训练监控，不单独派发MAE、积分或执行窗口消融。

## 工作台执行

按本地实际状态填请求JSON，通过既有 `/api/training/preview` → `/api/training/start`（或已安装的memory-workbench CLI）提交相同配置：

| 必须字段 | 取值 |
|---|---|
| device | a100 |
| version / recipe | V_2_1 / native_random_v2 |
| stage / task | stage1 / 40 |
| run_id | 实时核查后锁定的完整规范预检或正式run ID |
| dataset / checkpoint / split | 工作台实际Task40数据、已登记通用底座、原历史6000步的v2_1_stage1.json或等价原始data_split.json；不得使用旧权重作初始化 |
| official_root | 本地Agent依据baseline_diagnostic证据核实的真实原生N1.6源码根（包含gr00t/experiment/experiment.py），非模型权重目录 |
| gpus | 经实时核对的4个不同A100编号 |
| preflight_only | 首次true，正式false |
| accept_preflight | 正式时必填；独立归档的同谱系预检证据 |
| enqueue | 资源繁忙时按Skill使用现有队列 |

recipe固定训练参数；可显式传 learning_rate=1e-4、steps=6000、microbatch=2、accumulation=8、seed=42。视频后端默认torchcodec，若实际官方环境仅支持decord或ffmpeg可显式指定，同一预检与正式保持一致。workers固定2。不要传旧recipe的labels/global_windows_per_step/状态参数。输出路径由工作台服务器 TRAINING 配置决定；`preview` 返回 output/checkpoint_root。模型权重只发布在 `<实际job根>/run/checkpoints/<run_id>__stepNNNNNN`，内部HF checkpoint-N仅为写入暂存，不作为对外身份。

1. 本地同步main到工作台，运行CPU测试（见下），核对本机实际数据、源码、GPU UUID及目录，不覆盖历史；归一化准备发生在预检作业的run/prepared下。
2. 真实预检使用正式6000步Cosine计划但只运行4次optimizer update，默认第2步自动保存并暂停。核对工作台确实paused、进程释放后，用同一run调用原resume preview/start接口恢复，累计到4。严格重载检查真实模型张量、optimizer、scheduler、每rank RNG、确定性样本游标和实际更新；预检通过后独立归档run/preflight.json，再按Skill清理专属临时输出。
3. 正式请求同recipe/同源码/数据/归一化/底座、preflight_only=false、传独立归档门禁文件，工作台preview后start。正式独立初始化，不沿用预检模型或历史Stage1。0及每1000步做固定20条验证集×5个有效动作块的内部FM loss；归一化尺度不同不直接比较normalized loss。完整HF权重、optimizer/scheduler、各rank RNG、采样游标和hash一起原子发布；暂停续跑使用原run。
4. 使用确切6000步权重及兼容源码，经原工作台按计划评估Task40实例0–9、seed0 sequence、10个有效回合，积分4/执行8，按相同协议对比既有旧Stage1与官方参考。归档视频、回合/阶段成功、抖动和姿态，非有限值或接口不一致先停止定位。不得自动继续旧Stage2。

## CPU回归

在本机既有可用Python环境：
```bash
PYTHONPATH=scripts:train python -m pytest -q tests/test_stage1_native.py tests/test_v21_workbench.py
```
CI workflow也运行新增测试。静态源码接入和合成CPU测试不等于本地真实训练验收；遇到上游或本地依赖不匹配，修复工程接口而不改科学参数；无法通过则记录阻塞。没有从本会话启动训练。
