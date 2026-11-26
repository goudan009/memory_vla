# V2.1 实现与真实环境验收

状态：开发分支 / Draft PR，尚未完成真实 GPU 训练、工作台注册和 B1K 闭环评估。不要把 CPU 单测通过当成真实权重验收。

## 代码入口
- model/embodiment.py：实际 R1Pro 82 维本体、23 个动作通道、拓扑和基座系 FK。
- model/memory.py：双历史库、状态增强、联合检索、A/B/C 及深度对照 D、速度场残差。
- model/velocity.py：保留原始 V1 动作专家 FM 与 Euler 路径，仅在解码后修正速度场。
- data/labels.py、data/windows.py：21 项标签接口，默认 8 项；认证/弱/未知质量掩码，固定 180/20，按时间顺序训练。
- train.py、runtime.py：两阶段、同预算继续训练对照、校准、保存和断点恢复。
- serve.py、policy.py：原始动作反归一化、每环境记忆隔离、gamma 参数。
- diagnose.py：固定验证集历史干预、gamma 扫描和状态指标。

## 本地交接（先验收，不启动正式训练）

在已安装 V1 完整依赖且能读取 Task40 parquet/video 与实际 GR00T 权重的本地环境中：
```bash
git switch dev/v2.1-implementation
export PYTHONPATH="$PWD/src"
python -m compileall -q src/v2_1_groot_memoryvla
OMP_NUM_THREADS=2 python -m pytest -q src/v2_1_groot_memoryvla/tests
python -m v2_1_groot_memoryvla.train --help
python -m v2_1_groot_memoryvla.serve --help
```

环境变量 DATASET 指向已转换 B1K Task40 数据目录，BASE 指向通用无记忆 N1.6-3B checkpoint 目录，LABELS 指向经核验的 label_extension/task40_labels.parquet，OUT 指向新建实验输出根目录。GPU 必须由工作台分配且空闲；不要占用正在工作的设备。先验证 CLI preview 并记录工作台任务、源码提交、路径及 GPU。
```bash
python -m v2_1_groot_memoryvla.train --stage stage1 \
  --dataset "$DATASET" --checkpoint "$BASE" --labels "$LABELS" \
  --output "$OUT/stage1_preflight" --preflight-only
```
上述命令只打印配置。实际运行经工作台四卡 torchrun 并加 --execute；输出目录必须不存在。预检权重只用于工程认证。核对 preflight.json 的 passed 和身份字段，证据独立归档后清理专属权重。用户已确认预检通过后直接正式 Stage1，使用同一配置并指定 --accept-preflight，输出独立目录；默认 6000 步。

Stage1 完成且基线有效后，将 S1 指向其 checkpoints/v2.1__test01__exp14__run001__step006000（必须有 COMPLETE.json）。同样先 preview，再进行：
```bash
python -m v2_1_groot_memoryvla.train --stage memory --prior topology \
  --dataset "$DATASET" --checkpoint "$S1" --labels "$LABELS" \
  --output "$OUT/A_preflight" --preflight-only
```
真实短测将检查初始零残差、gamma=0/empty 同噪声原模型一致、校准、各模块梯度、冻结参数哈希及训练后 sidecar 重载一致性。记录证据且预检通过后按已确认协议进行正式 Stage2。B 使用 --prior fk；C 使用 --prior none；D 另加 --fusion-layers 2。B/C/D 指定 A 的 --calibration-reference。continued 使用 --stage continued，无记忆，同 Stage2 3000 步。

训练采用两个顺序窗口、每窗32决策、stride8、microbatch2，BF16 autocast、FP32 trainable master、AdamW lr1e-5、梯度裁剪1。必要时先以 microbatch1 重做预检。resume 必须是同输出目录、同源码和同配置的已完成 checkpoint；不要把不可信 training_state.pt 载入。中断续训逐步一致性仍需真实 GPU 检验。

## 放行前必做
1. 本地复跑 CPU 回归、核对全部 200 条数据和 label episode/帧对齐、训练集统计、真实 processor 维度，执行 Stage1 预检。
2. Stage1 有效完成后，A/B/C（D 可选）真实短测：零残差/empty/gamma0 一致；冻结权重不变；确认有效标签、梯度、参数变化和峰值显存。
3. 真实 checkpoint 保存、完整进程重载、断点重跑对照；多环境 reset 和 episode key 不串库；真实仿真接口单回合 smoke。
4. 工作台 backend 注册、代码同步与 preview 检查，确保显式选 V2.1 才路由新入口，不破坏 V1；现有 shell 入口支持 --source-version v2.1。
5. 用户于2026-11-27已确认通用GR00T、180/20和原V2.1配方：A100四卡真实预检通过后直接正式训练，执行记录见 Output/test_01/experiment_14。闭环评估沿用已确认协议与工作台。

离线 diagnose.py 保留为故障定位工具；本次不新增离线Probe或错历史实验。效果交付使用仿真闭环。状态真值只用于监督和验证，绝不输入推理模型。gamma=0 不计算原始 residual，相关诊断记 null。

详细配置与定义见 docs/v2/2.1-protocol.md 和 configs/task40.json。
