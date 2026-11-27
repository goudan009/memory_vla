# Stage1 基线重新训练（native-random-v2）

独立实验入口，原 V2.1-6000 和 Stage2 不改、不覆盖。使用本地 NVIDIA Isaac-GR00T N1.6 原生 `experiment.run`、`Gr00tTrainer`、随机分片数据集；不复用之前的自定义顺序训练循环。原生默认微调参数不是 B1K 冠军训练配方。

| 参数 | 本次 |
|---|---|
| 初始化 | 原通用 GR00T-N1.6-3B |
| 数据 | 原 Task40 180/20 固定划分 |
| 采样 | 官方 sharded random，episode_sampling_rate=0.1，shuffle，非固定 phase0 |
| A100 | 4卡 DDP，microbatch 2，累积8，全局64 |
| LR | 1e-4，Cosine，warmup 300（5%） |
| AdamW | 官方 adamw_torch_fused，weight decay 1e-5，clip 1 |
| 训练 | 6000 optimizer steps，BF16，FP32可训练参数 |
| 冻结 | 全部 Eagle（明确 top_llm_layers=0）；训练 action head/projector/VLLN |
| 动作 | 50×128 padding，有效32×23；相对动作定义不变 |
| 保存 | 1000步一次，保留全部6个权重 |
| 推理 | Euler4、执行8帧不变 |

## 归一化决定

官方处理器相对动作会直接读取 relative_action 中的 min/max，设置 use_percentiles=True 不会改变这条路径。本方案仅针对相对动作的躯干/双臂，每个未来offset、每个关节独立比较全训练集 min/max 与 Q01/Q99。正常状态、底盘、夹爪保持 Min-Max；候选区间包含零（若原范围包含零）、宽度至少0.02 rad；只有范围缩小至少4倍、训练裁剪≤2.1%、超过0.1rad裁剪≤0.1%、平均裁剪误差≤0.001rad 才接受。阈值是保守工程选择，不代表已验证能改善效果。原始统计和候选数值、极端样本及验证集裁剪误差均记录；不删除示教，不用验证集决定范围。未达条件保持原始 Min-Max。新 checkpoint 保存实际生效范围，processor use_percentiles=False 不变，以兼容原解码和 Stage2。也可显式使用 --normalization minmax 作纯原生对照；本轮默认 guarded。

## 本地执行

从仓库根目录运行，先使用工作台确认4张A100确实可用并分配，记录新实验 ID；不要覆盖现有目录。以下路径来自原诊断报告，实际路径由本地 Agent 核验。PY 指向**官方 N1.6 已安装完整依赖的 Python**，不是未验证的系统解释器。

```bash
PY=/path/to/official/n1d6/python
OFFICIAL=/nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1
DATA=/nvme3/data_wyh/datasets/b1k2025_groot_task40_v1_200
BASE=/nvme3/huggingface/train_model/GR00T/GR00T-N1.6-3B
OLD=/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2.1__test01__exp14__run001/run/checkpoints/v2.1__test01__exp14__run001__step006000
# OUT 必须为全新目录；选择本地工作台认可的路径。
OUT=/nvme3/data_wyh/outputs/memory_vla/stage1_native_random_v2

"$PY" scripts/stage1_native_prepare.py --dataset "$DATA" \
  --split "$OLD/v2_1_stage1.json" --base "$BASE" \
  --output "$OUT/prepared" --normalization guarded

# 检查 PREPARED.json 和 normalization_audit.json：
# 有多少 offset/channel 采用保守分位区间，实际训练/验证截断量如何。
# 若数据异常需要人工核实，不应盲目启动。
# CUDA_VISIBLE_DEVICES 由本地工作台按已分配的4张A100设置。
"$PY" scripts/stage1_native_train.py --official-root "$OFFICIAL" \
  --prepared "$OUT/prepared" --output "$OUT/preflight" --preflight
# 上述不加 --execute 只打印计划。确认无误后：
"$PY" scripts/stage1_native_train.py --official-root "$OFFICIAL" \
  --prepared "$OUT/prepared" --output "$OUT/preflight" --preflight --execute

# 预检通过后，正式6000步；独立原始通用权重，不继承短测权重。
"$PY" scripts/stage1_native_train.py --official-root "$OFFICIAL" \
  --prepared "$OUT/prepared" --output "$OUT/run" \
  --accept-preflight "$OUT/preflight/preflight.json" --execute
```

原生默认视频后端 torchcodec；若本地官方环境不支持，允许显式 --video-backend decord 或 ffmpeg，必须重做预检，不能用旧 preflight.json。预检3次真实 optimizer update、检查梯度/损失、冻结、有效Batch、实际保存的统计、重新加载和 held-out 10个动作块的真实推理。不会启动仿真。

原生分片训练器没有在线 eval dataset 接口（Factory 明确要求 eval_strategy=no），所以在正式训练结束后，对每1000步保存权重独立做固定验证集20条 × 5个动作块的 FM loss 及前8帧物理MAE，避免验证泄漏。两种归一化下的 normalized FM loss 不可直接等量比较，以物理MAE及闭环为准。最终6000步生成 Stage2 所需的 v2_1_stage1.json、COMPLETE.json，保留原生 HF 格式。若后处理出错，可使用相同参数和输出目录、去掉 --execute、改用 --finalize-only，仅重新做验证和导出，无需重训。正式训练断点恢复遵循原生 Trainer 语义（重置随机采样种子，不承诺逐步复现）；此新入口暂不提供 resume，不覆盖已有目录。

正式权重路径：`$OUT/run/native/checkpoint-6000`。在现有 V2.1 推理服务或 Stage2 使用前，先实际用原 `loading.load_base` 和 `Gr00tN1d6Processor` 严格加载并短测。现有 Stage2 检查6000步、固定split、统计hash和 raw82 边界，不应为了适配而放松。此代码未注册旧工作台的 HTTP/GUI 路由，使用已分配资源上的显式 CLI，现有入口和实验均不动。本次不自动启动训练。
