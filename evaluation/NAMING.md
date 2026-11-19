# 实验命名与版本管理

新记录使用 `groot-runs-v2` 命名规则；历史目录、原始成绩和运行中的进程保持原路径。

## 固定名称

- 实验版本：`V_1`、`V_2`……，表示一版方法方案。开发期修复可保留版本；确切代码由每次训练的源码快照及 SHA256 标识，正式比较时固定具体快照。
- `official`：官方未微调的 GR00T N1.6；与微调后的 Baseline 分开。
- `baseline`：无记忆微调对照。
- `memory`：完整记忆方案。
- `ablation_说明`：消融方法，如 `ablation_no_perception`。必须记录实际代码或配置改动；名称本身不会改变模型结构或关闭模块。
- `memory_v1`：现有训练器内部的实现名称，保留兼容；对外完整方法统一叫 `memory`。

新命令不使用 `bas`、`mem`、`experment_N`、`grootmeory` 区分方法。旧参数与旧结果继续读取，不能仅凭旧文件夹名推定实际权重。

## 训练、权重、推理和评估

```text
实验版本：V_2
方法：memory
任务：Task22
训练 ID：V_2_memory_task22_train02
权重：该训练目录内 memory_v1/checkpoints/step-003000
推理显示：V_2_memory_task22_train02 / step-003000 · 端口5560
评估目录：evaluation/V_2/memory/task22_eval03/
```

训练在 A100 的现有 training 存储根下；源码快照在其 deployments 下；评估在 4090 的 evaluation 下，不复制大型权重到评估目录。

每次新建训练分配 `train01`、`train02`……。计数按版本、方法、任务独立，兼容识别已有旧训练编号。改变种子、训练参数、修复代码后重新训练均是新训练；从已有权重启动也是新训练，不能称为优化器断点续训。当前网页不提供恢复优化器状态的训练续跑入口。

每次新评估分配 `task22_eval01`、`task22_eval02`……，按版本、结果分组、任务独立计数；不同训练和不同 checkpoint 共用此评估计数。已有旧评估编号参与计数。已有目录不会覆盖，删除最高编号目录后，仅靠目录发现无法永久保留已删除的编号。

评估中断续跑沿用原目录、评估编号、协议和已落盘回合，新增 resume 尝试编号；不恢复未完成回合的物理状态。已有模型关联的续跑会核对 checkpoint、训练 ID、源码哈希和推理种子，不允许偷偷换权重继续计分。旧运行缺少关联时仍标记缺失。

## 关联记录

训练的 `config/web_launch.json` 保存版本、方法、训练 ID/次数、训练参数、初始权重、方法说明与源码快照；`config/source_manifest.json` 保存代码文件哈希。启动后的源码快照不随网页自动同步修改。

网页评估启动时，后端通过 SSH 读取对应推理进程的 checkpoint、实际 PYTHONPATH 源码目录、源码清单哈希和启动参数，写入 `results.json` 的 `policy_identity`，包含所属训练、checkpoint、训练种子及推理种子。`binding=observed_process` 表示启动时核对了实际进程，不代表对服务整个运行期作持续健康认证。

命令行可通过 `--policy-checkpoint ABSOLUTE_PATH --training-id TRAINING_ID` 记录关联，标记 `user_declared`；未提供且没有网页核对时为 `unlinked`。不根据端口号或 `--model-name` 猜测模型。`--model-name` 只保留为旧元数据，不再影响新目录名。

训练种子、推理种子和环境逐回合 seed 分别记录。Baseline/Memory/消融对比固定训练数据划分、训练预算、任务集合、实例、逐实例评估 seed、环境分配及协议；checkpoint 的选择规则在测试前确定。复用旧 Baseline 时引用原训练和原评估 ID，不复制成绩伪装成新运行。

## 操作示例

```bash
# 先在网页选择已有 Memory 推理服务，再检查/启动评估，关联自动记录。
# 命令行使用相同公共入口；下面关联由使用者声明，不会自动加载该权重。
./evaluation/scripts/run_groot_eval.sh \
  --version V_2 --group memory --task 22 --seed 0 \
  --mode single1 --gpu-index 1 --policy-port 5560 \
  --policy-checkpoint /A100/实际训练目录/memory_v1/checkpoints/step-003000 \
  --training-id V_2_memory_task22_train02
```

网页现有链路：训练记录中选权重 →“用于推理评估”→ 启动/选择服务 → 启动评估 → 查看结果。版本、方法和权重随选择带入；训练结束自动排队推理/仿真的完整后台流水线尚未接入。

训练侧栏按“版本 → 任务 → 方法/训练次数”排列。同一任务的 Baseline、Memory 和消融放在一起，各方法内新训练在前；版本与任务均可折叠，轮询或手动刷新列表时保留折叠状态。简短卡片用于浏览，完整训练 ID 在详情和悬停提示中查看。

评估结果侧栏同样按“版本 → 任务 → 方法 → 评估记录”排列，方法内新评估在前。版本与任务均可折叠，列表刷新后保留状态；搜索时临时展开匹配项，清空搜索后恢复。仅改变展示，不移动结果目录或改写成绩。

## 主要后端脚本

| 脚本 | 职责 |
| --- | --- |
| `evaluation/scripts/view_results.py` | 网页 HTTP/API 总入口；读取结果，分发训练、推理、评估请求 |
| `train/dashboard.py` | 4090 训练管理器：代码同步、源码哈希、通过 SSH 调用 A100 |
| `train/dashboard_remote.py` | A100 训练管理：参数检查、分配训练编号、冻结源码、启动和查询训练 |
| `train/run_identity.py` | 训练与推理共用的命名、方法和训练身份解析；随 SSH 请求发送相同实现 |
| `train/run_groot_train.sh` | 公共训练入口，调用源码快照内 `experiment/launch_task21_ddp.py`；后者负责训练/异步验证子进程 |
| `evaluation/scripts/groot_inference_launch.py`、`groot_inference_remote.py` | 推理权重目录、实际进程身份、启动/关闭与评估关联 |
| `evaluation/scripts/run_groot_server.sh` | 按准确 checkpoint、源码、GPU、端口启动模型推理进程 |
| `evaluation/scripts/groot_web_launch.py` | 网页评估管理：核对服务身份、调用公共评估 CLI、观察进程状态 |
| `evaluation/scripts/run_groot_eval.sh` | 评估总入口：版本/分组/评估编号、single/dual 分配、续跑与收尾 |
| `evaluation/scripts/run_groot_priority_eval.sh` | 每卡队列：资源等待、环境与 rollout 子进程启动 |
| `evaluation/scripts/groot_eval_output.py` | 评估目录分配、逐回合 JSONL 与 results.json 更新 |

`train/dashboard.html`、`train/dashboard.js`、`evaluation/scripts/result_viewer.html` 是前端。根 `scripts/run_groot_eval.sh` 等同名入口用于兼容旧命令，实际公共逻辑在 evaluation/scripts 下。`train/task_queue.py` 和 `evaluation/scripts/evaluate_task21_trained.py` 是已有特定实验的队列工具，不能当成通用网页流水线。
