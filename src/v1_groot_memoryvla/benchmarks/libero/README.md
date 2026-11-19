# LIBERO：GR00T / Memory 三步工作流

网页顶部选择 LIBERO，仍使用「启动训练 → 推理评估 → 评估结果」。本轮提供本机单卡训练、评估；MIKASA-Robo、SimplerEnv 保留待接入状态。

1. 选择套件／任务，准备官方示教。完成后返回训练页，选择 Baseline 或 Memory、动作头全量或 LoRA，检查配置并启动。两种方式都冻结 Eagle，LoRA 默认仅作用于动作 DiT，rank 16、alpha 32；投影层及 Memory（若启用）正常训练。
2. 训练结果点击「用此权重评估」，或在推理评估页选择 checkpoint。策略自动加载，评估结束释放。默认 32 步用于调试；完整时长预设为每任务 50 回合。完整时长下手动减少回合数也会如实记录，不能与 50 回合成绩直接比较。
3. 结果页分开保存数据准备、训练和策略评估。可查看 Loss、逐回合结果、双相机录像、起止截图、运行配置与日志。异常回合单列，短回合调试不生成完整时长成功率。

## 数据与协议

- 官方原始 HDF5：`yifengzhu-hf/LIBERO-datasets`，固定 revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`。默认通过 `hf-mirror.com` 下载，使用固定 revision 的 LFS SHA-256 校验；可用 `LIBERO_HF_ENDPOINT=https://huggingface.co` 切换官方域名。仅下载所选任务，单任务约几百 MB。
- 存放于项目 NVMe 的 `assets/datasets/libero/`。清单记录示教数、长度、SHA-256 和协议。当前 `/data/datasets` 不可写，未使用用户 home 存放大文件。
- 原始导出脚本把最终 reward/done 统一置 1；这不能证明每条示教成功。准备阶段排除显式失败标记，并检查数据形状与有限值，但未对全部轨迹重放筛除失败示教。论文的完整成功示教筛选仍需额外审计。
- GR00T 使用预留的 `libero_panda` embodiment：第三人称与腕部两路 RGB，末端位置／轴角 6 维及夹爪位置 2 维，输出 7 维原生 OSC_POSE 动作。训练／推理相机均旋转 180°；原生夹爪符号为 -1 打开、+1 闭合，无 RLDS 的额外反转。
- 状态与动作统计从所选训练示教生成并随 processor 保存；预测 16 个动作，实际执行前 8 个。动作尾部超出回合的标签被掩码。训练按任务均衡、从单回合有序抽样最多 16 个决策帧，组间清空记忆；Memory 长度 16、direct 接口和认知 Q/K 归一化沿用修复后的 V1。
- 仿真固定 LIBERO commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`。每个任务依次使用官方初态 0–49，每回合清空策略记忆，再执行 10 步稳定动作。上限：Spatial 220、Object 280、Goal 300、Long-10 520、Long-90 400；成功判定取自环境。
- 这是 MemoryVLA 方法向 GR00T 的适配，保留 GR00T 网络。论文 LIBERO 使用单相机，这里采用 GR00T 的双相机及状态输入。五套件可以分别使用，目前未实现论文 Long-10 与 Long-90 联合训练。

## 预算与断点

快速迭代预算为每任务新增 300 步，上限 3000 步；目的是排查接口和做初步对照，不保证收敛。论文步数参考为 Spatial/Object/Goal 20k、Long-10/90 40k；这里的单卡分组训练与论文 batch 不同，不能把相同步数视为等量训练。任务效果通过一致评估协议的 Baseline / Memory 对照判断。

checkpoint 包含模型、processor、`libero.json`、`optimizer.pt`；每个完整保存会占用数 GB 以上磁盘。网页从列表选择断点后恢复原任务、模型、微调模式、分组帧数、seed 和学习率，目标步数表示训练到的总步数。采样与随机数按优化步确定，续训恢复优化器状态。停止训练会完成当前步并保存；评估停止针对当前作业的策略与仿真进程。

## CLI

在项目根目录使用 V1 Python，设置 `PYTHONPATH=src PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 NO_ALBUMENTATIONS_UPDATE=1`。正式使用可设置离线模型加载环境变量。

```bash
# 下载所选任务的原始示教（已有且校验过的任务复用）
src/v1_groot_memoryvla/.venv/bin/python -m v1_groot_memoryvla.benchmarks.libero.prepare \
  --suite libero_spatial --tasks 0 --output outputs/benchmarks/libero_dataset_task0.json

# 短训练；对照 Baseline 改 --arm baseline，其余保持一致
src/v1_groot_memoryvla/.venv/bin/python -m v1_groot_memoryvla.benchmarks.libero.train \
  --checkpoint checkpoints/groot_n1d6_base --suite libero_spatial --tasks 0 \
  --arm memory --steps 300 --group-size 16 --fine-tuning-mode action_full \
  --output outputs/benchmarks/libero_example --report outputs/benchmarks/libero_example/report.json

# 真实策略调试；完整时长用 --max-steps 0 --trials 50
src/v1_groot_memoryvla/.venv/bin/python -m v1_groot_memoryvla.benchmarks.libero.evaluate \
  --checkpoint outputs/benchmarks/libero_example/checkpoints/step-000300 \
  --suite libero_spatial --tasks 0 --trials 1 --max-steps 32 \
  --output outputs/benchmarks/libero_example_eval/evaluation.json
```

网页 checkpoint 自动发现仅覆盖网页作业，以及 `outputs/benchmarks/libero_integration_*/checkpoints/` 下的联调权重。其他 CLI 训练可继续通过 CLI 评估。

## 本轮验证

Spatial Task0 已下载并校验 50 条真实示教。Memory 动作头全量训练 2 步、checkpoint 恢复优化器续训到第 3 步已通过；两回合各 16 步的真实策略调试有效，并保存录像。网页实际启动 Baseline LoRA 训练 1 步，再加载该权重完成 Spatial Task0 的 220 步、28 次策略决策和录像播放，结果为 0/1 成功。229 项相关回归通过，浏览器覆盖三步切换、任务推荐预算、数据准备、训练后选择权重、评估和视频 Range 请求。模型只训练了几步，以上用于验证工作流，不代表模型已学会任务或达到论文指标。
