# V1 快速迭代与检索修复验收

2026-11-12。本轮保留 GR00T backbone 和 direct 条件接口，只修复已复现的认知检索饱和，并提供可重复的网络检查、同初始化对照和仿真环境检查。

## 修复是什么

认知 MemoryBank 的 Q/K 在投影前做无可学习参数的 LayerNorm。K 仍含时间编码，V、残差、门控、写入顺序和 ToMe 不变；感知 MemoryBank 不变。

- 新训练默认 `--memory-cognitive-qk-norm layernorm`，写入模型 config、训练参数和网页记录。
- 旧 checkpoint 缺少该字段时按 `none` 加载，保留历史结果。网页从旧 Memory 权重启动时也沿用旧设置。
- 要训练修复版本，从原始 GR00T 权重新建训练。不要直接把旧 16000 步模型的 config 改成 layernorm 当作修复后训练结果。
- 训练协议更新为 `memoryvla_direct_tail_qknorm_v3`；旧任务续跑使用它自己的不可变源码快照。
- 这项归一化是针对 GR00T 特征尺度的适配，不是 MemoryVLA 官方 post-norm 检索的逐算子照搬。

## 本机一键检查

网页使用同一套三步流程：顶部选 **BEHAVIOR-1K** 后，在「启动训练 → 训练前检查」运行网络回归或动作短训练，在「评估结果 → 开发快测」查看报告；顶部选 **LIBERO** 后，可准备官方示教、训练 Baseline / Memory、加载权重做真实闭环评估，在结果页查看训练 Loss、逐回合结果、录像及日志。零动作环境检查仍作为独立选项保留。网页每次使用独立的 `outputs/benchmarks/web/<运行编号>/`，具体交互见 [评估说明](../../../evaluation/README.md#网页推理服务)，LIBERO 协议与命令见 [LIBERO 接入说明](../benchmarks/libero/README.md)。

在项目根目录执行。GPU 通过 `CUDA_VISIBLE_DEVICES` 选择；不会访问或重启正在运行的策略／训练服务。

```bash
# CPU 回归 + 真实 EOS 检索回归，通常约一分钟
CUDA_VISIBLE_DEVICES=0 bash scripts/run_v1_quick_checks.sh fast

# 真实动作标签，动作头和记忆模块共同训练 100 步，约三分钟 / 26 GiB
CUDA_VISIBLE_DEVICES=0 bash scripts/run_v1_quick_checks.sh action

# LIBERO 固定初态、双相机、零动作仿真冒烟
CUDA_VISIBLE_DEVICES=0 bash scripts/run_v1_quick_checks.sh libero --tasks 0 --steps 20
```

`unit` / `retrieval` / `action` / `libero` 可单独运行；`all` 顺序执行四层。数值检查不通过时非零退出。报告存放于 `outputs/benchmarks/`。

当前本机诊断夹具：

- `outputs/diagnostics/v1_real_replay_features.pt`：4 条真实 B1K Episode（10、340020、400010、400020），每条前 128 个决策的 EOS，共 512 个；另有 40 个时点的完整特征和动作标签。
- `outputs/diagnostics/v1_cognitive_16000_weights.pt`：`V_1_memory_task0_34_40_train02 / step-016000` 的认知模块权重。
- 特征来自该任务冻结的 GR00T-N1.6-3B backbone，前一轮诊断已与原始 backbone 权重逐张量核对。不能拿这个缓存测试改过 backbone、图像预处理或不同模型的效果。
- 原始缓存位于 A100 的 `/nvme3/data_wyh/outputs/memory_vla/diagnostics/v1_memory_16000_real_replay_20261112/`。这是本机实际数据夹具，不包含在 Git 中；迁移机器时须复制或重新采集。

## 已得到的结果和它们的边界

136 项回归通过，覆盖真实动作头前后向、HF 保存重载、训练分块与流式记忆一致、Episode 隔离、因果顺序、历史 detach、空／错误历史、网页参数传递和断点恢复。

同一 16000 步权重、同一 512 个真实 EOS，480 次历史长度至少 8 的查询：

| 第一层认知检索 | 旧实现 | Q/K 归一化 |
| --- | ---: | ---: |
| 最大权重 > 0.999 的比例 | 100% | 0% |
| 最早一帧平均权重 | 0.999999968 | 0.05995 |
| 归一化注意力熵 | 0.000000465 | 0.99486 |

真实动作短训练中，修复版 100 步均有认知 Q 梯度，最小值 `5.31e-6`，中位数 `5.81e-5`。loss 第 1 步 `0.19862`，第 100 步 `0.01460`；各步交替任务，单步 loss 不应当作固定数据学习曲线。耗时 161 秒，显存峰值 25.36 GiB。

三组均从同一 GR00T-N1.6-3B 初始化，冻结 backbone、相同真实样本／噪声／学习率，训练 100 步。训练使用前三条 Episode，诊断使用第四条：

| 训练分支 | 诊断时条件 | 前 8 动作归一化 MAE，越低越好 |
| --- | --- | ---: |
| 原 GR00T，无记忆接口 | 原始条件 | 0.00601 |
| direct 接口，训练时关闭历史 | 空历史 | 0.04995 |
| direct + 修复后的记忆 | 正常历史 | 0.03993 |
| 同一个修复后模型 | 空历史 | 0.03973 |
| 同一个修复后模型 | 错误历史 | 0.03966 |

**检索数值故障已解除，但这个短实验还没有证明记忆提升任务效果。** direct 接口在少量训练后仍有明显适应成本；正常／空／错误历史几乎相同，不能据此称为已经学会利用历史。

这些是 40 个稀疏时点的诊断，不是完整生产窗口，也不是闭环成功率。第四条 Episode 未参与本次 100 步优化，但曾在旧 16000 步任务中训练，不能当作旧模型的独立验证集。

重跑同初始化三组（每组报告单独保存）：

```bash
for bench_arm in baseline direct_empty memory; do
  CUDA_VISIBLE_DEVICES=0 bash scripts/run_v1_quick_checks.sh action \
    --arm "$bench_arm" --steps 100 \
    --output "outputs/benchmarks/action_smoke_${bench_arm}.json"
done
```

## MemoryVLA 采用了哪些 benchmark

本地依据：`related_work/MemoryVLA/latex/chapters/6_appendix.tex` 的训练／评估协议，以及 `tables/tab_mikasa.tex`。

| Benchmark | 论文设置 | 对当前项目的用途和状态 |
| --- | --- | --- |
| LIBERO | Spatial、Object、Goal、Long-10、Long-90；共 130 任务，每任务 50 示教；记忆长度 16 | 已接入官方示教准备、LIBERO 专用训练、GR00T / Memory 策略闭环与网页三步流程。本机单卡；五套件分别训练，Long-10 + Long-90 联合训练尚未提供。 |
| MIKASA-Robo | Shell Game Touch、Intercept Medium、Remember Color 3/5/9；1250 示教联合训练 20k 步，记忆长度 16；每任务 100 回合，每 1k 步验证 | 最适合检验历史是否必要。建议先 Remember Color 3 + Shell Game Touch；本轮完成官方资料核对，尚未安装或进行策略评测。 |
| SimplerEnv | Bridge/WidowX 与 Fractal/Google Robot，使用相应真实机器人数据训练 | 后续迁移和泛化测试；数据与机器人接口准备较重，不作为本轮最快入口，未安装。 |
| 当前 B1K 小面板 | 不是 MemoryVLA 论文的 benchmark | 直接兼容当前机器人，能在不更换数据与动作接口的情况下检查真实策略。已有 fast-debug 入口；本轮只核对其启动计划，未占用运行中的策略服务。 |

官方来源：[MemoryVLA](https://github.com/shihao1895/MemoryVLA)、[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)、[MIKASA](https://github.com/CognitiveAISystems/MIKASA-Robo)、[SimplerEnv](https://github.com/simpler-env/SimplerEnv)。

MIKASA 当前主分支已经扩展为 90 个 VLA 任务，不能直接当作论文当时的五任务设置。原 RL 版本在 `mikasa-robo-rl` 分支／`mikasa-robo-suite==0.0.5`；当前 VLA 版本的 ID 例如 `RememberColor3-VLA-v0`，且官方要求使用 `apply_mikasa_vla_wrappers`。确定复现版本后再转换数据、设置相机、状态字段、7 维末端动作、归一化和 embodiment。不要将 B1K 的双臂移动机器人动作截成 7 维塞进 Franka 环境。

LIBERO 的三个套件都已实跑 Task0、固定初态、双相机和 20 步仿真：Spatial 21.76 秒（含首次编译），Goal 7.17 秒，Long-10 8.74 秒。报告分别是 `libero_smoke.json`、`libero_goal_smoke.json`、`libero_10_smoke.json`。零动作 smoke 只证明 reset、渲染、动作接口和仿真 step 可运行；不会输出虚构的 GR00T 成功率。

## LIBERO 环境重建

源码固定到 `8f1084e3132a39270c3a13ebe37270a43ece2a01`，独立 Python 3.10 环境，不修改训练环境。首次只需仓库内资产，不下载全部示教。

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
git -C third_party/LIBERO checkout --detach 8f1084e3132a39270c3a13ebe37270a43ece2a01
UV_CACHE_DIR=/nvme3/projects/memory_vla/env/.uv-cache uv venv --python python3.10 env/libero-smoke
UV_CACHE_DIR=/nvme3/projects/memory_vla/env/.uv-cache uv pip install \
  --python env/libero-smoke/bin/python -r env/libero-smoke-requirements.txt
```

已存在的目录不需重复创建。脚本只读复用现有 V1 环境的 Torch；仿真依赖优先从独立环境加载，配置输出到 `outputs/benchmarks/libero_config/`。

## 下一轮训练顺序

1. 每次改记忆模块先跑 `fast`；接口／训练改动再跑三组 100 步 smoke。饱和或梯度门槛失败先排错。
2. 从原始 GR00T 新建小规模正式训练，三任务各留 20 条独立 Episode（540 训练／60 验证），先跑 1000 步，检查逐任务动作误差和正常／空／错误历史差异。公平 baseline 使用同一原始初始化、划分和训练预算；B1K 专项官方权重只能作为额外强参考。
3. 再用 B1K Task0/34/40 的固定实例和 seed 检查短闭环动作、进度、reset 和会话隔离；确认后才扩大预算。200 步窗口属于 debug，不能计作完整成功率。

现有 B1K 入口计划检查示例：

```bash
bash evaluation/scripts/run_groot_eval.sh --preset fast-debug --mode single1 \
  --gpu-index 0 --task 0 --instances-gpu0 0 --seed 42 \
  --debug-max-steps 200 --no-record-video --group memory --dry-run
```

真正执行时选择对应新权重的专用策略服务，显式填写 `--policy-host`、`--policy-port`、`--policy-checkpoint`，再去掉 `--dry-run`。`--group memory` 只是结果分组，不会自动切换服务端模型。不要用正在执行其他评估的服务代替新模型。
