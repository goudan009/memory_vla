# V1.2 本机预检记录（2026-11-27）

状态：**PARTIAL / 实际环境与闭环 BLOCKED**。本记录是基础设施与数据预检，未分配实验身份，不代替实验 task / workflow / progress。没有创建 V1.2 GPU 作业，没有正式训练，没有闭环结果或视频，不能据此称 N1.7 基线已跑通。

本轮已将最新远端 main 合并到本地 main，保留原有 V2.1 计划提交；预检代码基点为 `fab19b604609728bd95ba5076f678c0d61416c92`。本机时间为 UTC+08:00；以下工作台时间均为主机返回值。

## 已完成：2025 Task40 实体数据检查

只读检查目录：`assets/datasets/b1k2025_groot_task40_v1_200`。

```sh
PYTHONDONTWRITEBYTECODE=1 \
  /nvme3/projects/GR00T/official-baseline-env/bin/python \
  scripts/audit_v12_task40.py assets/datasets/b1k2025_groot_task40_v1_200
```

此 Python 仅用于 CPU 文件检查，未安装依赖、修改 N1.6 环境或加载模型。脚本需要 numpy、pyarrow、av，输出 JSON；失败返回非零。摘要及元数据哈希见 [实体检查证据](task40-audit-2026-11-27.json)。

| 检查 | 实际结果 |
|---|---|
| 示教 / Parquet / 总帧数 | 200 / 200 / 647,555 |
| 原划分 | train 180，validation 20，交集 0，合集等于实体 episode 集合 |
| 全量行检查 | 30 Hz；82 维 state、23 维 action；有限值、连续 frame_index、episode/task ID 与 timestamp 正确 |
| 状态布局检查 | 双臂 qpos 的 sin/cos 冗余字段相符，末端四元数模长为 1 |
| 语言 | task_index 与 coarse_action 均为 40，查表指向 make_microwave_popcorn 且指令非空 |
| 视频头 | 600 个，全部 256×256、30 fps，帧数与对应 episode 行数一致 |
| 实际视频解码 | 3 条 episode × 3 相机 × 首/中/末帧，共 27 帧；RGB 形状与时间戳通过 |

解码 episode / 帧索引：400010 `[0,1939,3878]`；401060 `[0,1346,2691]`；403000 `[0,1586,3172]`。这不是逐帧回放认证，也未对模型动作质量作离线评分。

另外通过只读 SSH 取得 A100 上原始 `preparation/data_split.json`：

`/nvme3/data_wyh/datasets/b1k2025_groot_task40_v1_200/preparation/data_split.json`

其 SHA256 为 `6e6d5bb26085210e187b8617f27f90be8779f08145289840bafb013ab25d8fb7`。本机 info.json 的 train、validation 数组与该文件**逐项相同**，不仅数量相同。未复制、重排、修改源数据或统计。

## 已完成：训练接口源码兼容性检查

挑战 fork 固定在 `ace36d935b376fbf25cd56371e23877b95407c40`，已通过上游 Git tree 核对 contracts.UPSTREAM_FILES 的五个 Git blob ID。这里只证明远端源码匹配，尚无本机独立 checkout 或环境验收。

- 原生 LeRobotEpisodeLoader 支持 v2.1/v3.0，按 tasks.jsonl 的 task_index 查语言；命名 state/action 切片与现有 82/23 packed schema 可以对应。不能把 2025 的 robot_lin_vel 当作 2026 robot-local base_qvel。
- 原生 ShardedSingleStepDataset 根据实际 episode 列表采样，不能仅靠原 info.json 的 split 字段隔离验证集。现有完整 200 条目录**不能直接用于 train-only**。训练前需要独立的 180 条实体视图，并重新计算训练集统计；本轮没有生成或覆盖统计。
- 双臂和 torso 的 relative-action reference 分别对应 arm_left_qpos、arm_right_qpos、trunk_qpos；底盘速度和夹爪命令保留原定义。真实 processor 的相对/绝对动作编码、掩码、padding、解码和模型前后向尚未运行验证。
- 上游 `global_batch_size` 是累积前的跨卡 batch；`4 × microbatch 2 × accumulation 8` 应传 global_batch_size=8，有效 batch=64。该算例不是本任务训练配置。
- 上游 train_b1k.py 强制 use_wandb=True，后续工程验证需明确日志方式，例如 WANDB_MODE=offline；不能把隐式默认当作已批准配置。
- 上游 experiment.py 设置 ignore_data_skip=True。普通 resume 不能直接作为严格恢复采样位置的证据；累计超过 2000 optimizer steps 前仍需按 Skill 验收完整暂停/恢复。

训练接口结论是“文件与源码契约可映射，真实运行兼容性未验收”。没有选择正式 horizon、有效 batch、训练步数或冻结配方。训练起点仍必须是核实来源的通用 N1.7-3B，不能使用官方 radio 或 N1.6 权重。

## 阻塞：官方 2026 环境与工作台闭环

1. **实验身份未分配。** 仓库 task/progress/plan 和工作台记录中未找到本任务的 V1.2 test_id / experiment_id / run_id。Skill 要求“身份缺失时询问，不用默认版本/编号”；本轮没有猜编号，也没有伪造实验目录、计划或 job ID。
2. **手册与可取得的 tag 不一致。** [官方基线手册](https://behavior.stanford.edu/challenge/baselines)仍指定 v3.9.3，但 [精确 tag API](https://api.github.com/repos/StanfordVL/BEHAVIOR-1K/git/ref/tags/v3.9.3)返回 404，git ls-remote 同样没有该 tag。可取得 v3.9.3-post1（`bd049de3119acdcdf2334fe9e1ebe060fa20c108`）及 v3.9.3-post2（`a8247a8cc1633fe1ca0cc66aa07243d46c64f155`）。两个 post tag 的 `OmniGibson/omnigibson/eval/eval.py` 内容相同，SHA256 均为 `93c4dbf4a6bcdf17f2edcafda7132a3b83bebf6d5dba3dda90d7c157862d4ccd`，包含当前候选 spec 所需参数；**这不证明两套仿真源码或资产相同**。需要明确固定哪个可用 revision，不能悄用 main/post2 冒充 v3.9.3。
3. **存储约定缺少可写落点。** 本机规则要求代码/数据/模型放 /data，但 /data 与 /data/datasets 均为 root:root 0755，当前 uid=1001 无写权限；约定的 `/data/模型与数据存放速查手册.md` 不存在。现有工作台实际使用 /nvme3。安装独立环境、源码及下载权重前，需给出可写的合规路径，或明确本任务沿用 /nvme3 的例外。
4. **权重与新环境未就绪。** 本机/A100 现有注册资源未找到可复用 N1.7 环境、通用权重或官方 radio checkpoint。Hugging Face 访问尝试失败，许可证与下载访问状态尚未验证，不能记成许可已通过或被拒绝。没有产生实体 checkpoint / processor SHA256。
5. **工作台接入未验收。** 现有 InferenceManager 尚无 N1.7 WebSocket route；LaunchManager 继续拒绝 V1.2 误走 B1K2025。需要在确认上述版本、真实路径及实验身份后接入现有 manager 的生命周期、健康检查、结果 JSON 和视频。当前没有新增可启动路由、没有调用 start，也没有用私人 launcher 代跑。

现有 `/nvme3/projects/GR00T/BEHAVIOR-1K-task-progress` 为带本地改动的 2025 仿真，不作为官方 2026 环境。2026 官方 radio 权重只用于 turning_on_radio，不能进入 2025 Task40。

## 工作台与 V2.1 保护记录

通过 8765 工作台只读查询 services、options、training/inference/launch jobs；未重启工作台或修改设备配置。

- A100：V2.1 `v2_1__test02__exp01__run003` 仍为 training，20:24 左右观察到 970/6000 步，四卡作业保留。
- 现有 V2.1 推理 PID 4127800 / 5558、V1 推理 PID 3857420 / 5557 均为 ready，未停止。
- 4090 两卡总显存各 49,140 MiB，检查时占用 1,156 / 6,188 MiB；A100 四卡总显存各 81,920 MiB，占用 35,245 / 35,245 / 45,343 / 42,841 MiB。容量只作快照，实际启动必须重新查 GPU UUID、并发和加载峰值。
- RTX 6000 Ada 工作台返回离线/SSH 不可达；不能当作已可用资源。

本任务训练提交 0、推理提交 0、评估提交 0、有效闭环回合 0；成功率无分母，不报告为 0% 或成功。

## 验证与接续条件

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:train \
  /nvme3/projects/GR00T/official-baseline-env/bin/python -m pytest -q \
  tests/test_audit_v12_task40.py tests/test_v12_baseline_contract.py \
  tests/test_run_identity.py tests/test_groot_inference_launch.py \
  tests/test_groot_web_launch.py
```

结果：**96 passed in 9.98s**。新增四项测试使用真实小型 Parquet/MP4，覆盖正常只读检查、时间错位、split 重叠、实体 episode 缺失。其余测试覆盖身份和旧工作台路由，不替代真实 N1.7 环境或闭环。

接续时先确认实验身份、存储落点和仿真 revision，再在对应真实实验目录写入并提交 task.md、progress.md、workflow/plan-vNNN.json。明确的主线是：独立环境和来源核验 → 现有工作台原生 WebSocket 接入 → 官方 radio 公共测试单环境短闭环 → 归档真实 job、权重/源码哈希、JSON、视频 → 完成 2025 原生 loader/processor/transport 验证。正式训练继续禁止，V2.2 不在范围内。

后续所有新训练填写简短 notes（如“V1.2 / N1.7：原生无记忆基线”）；2025 评估填写 run_tag（如 `v1_2-n17-baseline-task40`）。2026 沿用接入后的真实备注字段，建议说明“V1.2 / N1.7：官方 radio，RTC 关闭”，并核对 preview/start 后实际保存及显示。当前无新作业，不回写或覆盖既有用户备注。
