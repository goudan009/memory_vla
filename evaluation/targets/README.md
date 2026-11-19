# 评估对象与版本映射

统一规则见 [命名与版本管理](../NAMING.md)。网页启动评估时会核对实际服务进程，并把训练 ID、方法、checkpoint、实际源码路径和清单哈希写入 `results.json.policy_identity`。命令行未声明关联的运行标记 `unlinked`，不能仅凭目录名或端口判断模型。

| 模型对象 | 源码与服务入口 | 建议结果分组 |
| --- | --- | --- |
| 官方未微调 N1.6 | 官方 checkpoint，独立服务进程 | `--version V_1 --group official` |
| 无记忆微调 | 使用 baseline 模型类型的训练权重 | `--version V_1 --group baseline` |
| 完整记忆 | `python -m v1_groot_memoryvla.serve`，memory_v1 实现 | `--version V_1 --group memory` |
| 后续消融 | 对应源码快照及实际消融配置 | `--version V_2 --group ablation_说明` |

V1 已接入记忆。实验版本名不等于 Python 包名，方法标签也不会自动关闭模块；实际模型由源码快照、权重及模型配置决定。历史 `experment_N` 分组继续读取。`--run-tag` 是备注，旧 `--model-name` 只保留为元数据，不再决定新目录名。

每次记录：代码提交及未提交补丁、配置快照、完整 checkpoint 路径或模型仓库 revision、服务地址、模型环境、seed、run tag 与对应结果路径。正式脚本已有配置、协议 SHA256 和逐回合实例/seed 记录；它不能仅凭远端地址核实服务实际加载的代码或权重，模型侧记录需另行保留。

示例：先在模型机器用本版环境启动服务，指定已存在的 checkpoint；再在仿真机器执行公共入口。以下路径、主机名按实际替换：

```bash
# 模型侧，项目根目录
python -m v1_groot_memoryvla.serve \
  --model-path checkpoints/v1_groot_memoryvla/RUN/checkpoint-N \
  --embodiment-tag BEHAVIOR_R1_PRO --use-sim-policy-wrapper --port 5557

# 仿真侧，项目根目录
./evaluation/scripts/run_groot_eval.sh --version V_1 --group memory \
  --policy-host MODEL_HOST --policy-port 5557 \
  --model-name grootmeory --task 22 --seed 0 --run-tag v1_memory
```

对照使用相同任务、实例/实际 seed、动作 chunk、预算及评分规则，见 [固定协议](../evaluation.md)。
