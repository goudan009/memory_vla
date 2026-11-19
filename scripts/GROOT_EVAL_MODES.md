> 当前目录与正式协议以 [evaluation/evaluation.md](../evaluation/evaluation.md) 和[使用说明](../evaluation/README.md)为准。默认结果在 baseline，实验须加 `--version experiment_N`；网页读取 JSON 结果，不再生成或追加 results.md。

# GR00T 评测入口

在项目根目录运行 `evaluation/scripts/run_groot_eval.sh`。不加参数即 GPU1、单环境、Task21 的10个实例各一个完整 Episode，seed0、action chunk=8、视频关闭、原生 viewer。每张卡的队列只评估一个任务类型。

## 正式评估命令

```bash
# 默认：GPU1，单环境依次跑10个实例
./evaluation/scripts/run_groot_eval.sh

# GPU1，两个环境各跑5个实例（2×5，总共10回合）
./evaluation/scripts/run_groot_eval.sh --mode single2 --gpu-index 1 --task 21

# GPU0 单环境；更换任务使用 --task 22 或 --task 26
./evaluation/scripts/run_groot_eval.sh --gpu-index 0 --task 21

# 双卡各单环境，一次命令启动两卡
./evaluation/scripts/run_groot_eval.sh --mode dual1 --task 21

# 双卡各双环境，一次命令启动4个环境
./evaluation/scripts/run_groot_eval.sh --mode dual2 --task 21

# 后续 Memory 结果按用户指定的 grootmeory 拼写保存
./evaluation/scripts/run_groot_eval.sh --model-name grootmeory --mode single2 --task 21
```

`--model-name`只标识结果，不负责加载或切换模型。运行 Memory 前须让对应 Policy Server 加载 Memory 模型，必要时指定 `--policy-host` 和 `--policy-port`。默认服务为 `172.28.62.17:5556`。

| 模式 | GPU | 每卡环境数 | 正式评估默认实例/回合分配 |
| --- | --- | ---: | --- |
| single1（默认） | GPU1，可指定GPU0 | 1 | 实例0–9，共10回合 |
| single2 | GPU1，可指定GPU0 | 2 | env0实例0–4，env1实例5–9，各5回合 |
| dual1 | GPU0、GPU1 | 1 | GPU0实例0–4，GPU1实例5–9，各5回合 |
| dual2 | GPU0、GPU1 | 2 | 每卡5回合，卡内2+3，总10回合 |

“10个任务”实际是同一个任务的10个不同测试实例，`env_idx`不等于实例编号。两个环境完成各自配额后会等待，不会重复评估实例。双卡模式强制检查实例不重叠，每卡独立 AppData；不要在双卡运行时另开一份占用同一GPU的仿真。

需要两张卡分别跑不同任务时，在两个终端执行，例如：

```bash
# 终端1：GPU0跑Task21，2×5
./evaluation/scripts/run_groot_eval.sh --gpu-index 0 --mode single2 --task 21
# 终端2：GPU1跑Task22，2×5
./evaluation/scripts/run_groot_eval.sh --gpu-index 1 --mode single2 --task 22
```

这属于两次评测，各自一个简单命名目录；一次 dual 命令评估同一个任务的不同实例。

## 输出：一次评估一个文件夹

自动按“模型名、任务编号、次数”命名，目录已存在时继续尝试下一个编号，不覆盖。不同任务、不同模型独立编号；快评和正式评估共用编号，模式写入结果。并发启动会原子分配不同目录。旧 `--run-tag`仍接受，但只记录为备注，不再拼进文件名。

```text
evaluation/
├── groot_task21_1/
│   ├── results.json
│   ├── episodes.jsonl
│   └── gpu1.log
├── groot_task21_2/
│   ├── results.json
│   ├── episodes.jsonl
│   ├── gpu0.log
│   └── gpu1.log
└── grootmeory_task21_1/
    ├── results.json
    ├── episodes.jsonl
    └── gpu1.log
```

默认仅保留这三类文件：

| 文件 | 用途 |
| --- | --- |
| `results.json` | **先看这个**：执行状态、累计成功率、平均进度、有效/无效回合数，以及配置、实例、种子和每卡性能汇总 |
| `episodes.jsonl` | 每回合一行，结束立即追加并 fsync：GPU、env_idx、真实 instance_id、seed、success、task_progress、valid、动作步数、耗时、原生终止原因 |
| `gpu0.log` / `gpu1.log` | 每张使用的卡一份完整日志，包含启动、仿真与失败原因 |

正式成绩查看 `results.json` 的 `evaluation_mode=official-eval`、`status=complete`、`completed_episodes`、`success_rate`、`mean_task_progress`。`ok=true`表示流程完成，不表示任务成功。成功率为0–1，进度为0–100；只用有效回合计算，双卡按实际回合汇总，不平均两卡成功率。

配置位于 `configuration`，每卡配置、性能及最后环境身份位于 `per_gpu`。不再单独留下 plan、summary、manifest、profiling、run_meta、pid 或重复日志。运行时的辅助状态临时位于同目录隐藏的 `.runtime/`，所有子进程退出且汇总后自动清理；强制断电或 kill -9 可能留下该目录，已完成回合仍在根目录 JSONL。Ctrl+C 或崩溃后已完成回合保留，状态会标记为 interrupted/failed。

如需保留详细诊断和资源采样，显式加 `--diagnostics`，额外文件集中在 `diagnostics/`。默认不启动资源 CSV 监控。旧结果不自动搬动或删除；直接调用旧 priority 诊断入口仍保留旧布局。

```bash
cat evaluation/baseline/groot_task21_1/results.json
tail -f evaluation/baseline/groot_task21_1/gpu1.log
tail -n 3 evaluation/baseline/groot_task21_1/episodes.jsonl
```

## 种子与 Baseline/Memory 对照

```bash
# 每个实例都用相同种子，适合配对比较
./evaluation/scripts/run_groot_eval.sh --seed 0 --seed-mode fixed
./evaluation/scripts/run_groot_eval.sh --model-name grootmeory --seed 0 --seed-mode fixed

# 可复现的随机种子序列
./evaluation/scripts/run_groot_eval.sh --seed 42 --seed-mode random

# 每次随机选择base seed；解析后的数值与每回合实际seed均保存
./evaluation/scripts/run_groot_eval.sh --seed random --seed-mode random
```

默认 sequence 沿用原逻辑：base + env_idx + 槽位回合序号（从0开始）。fixed 所有回合用 base；random 用 NumPy SeedSequence([base, env_idx, episode_number])派生。重现随机实验时从 `results.json.configuration.base_seed`读取数值，替换 random。Baseline/Memory 应保持相同 seed 模式、实例顺序与环境分配；环境种子不控制共享 A100 服务端的独立采样随机流。

## 快速调试与资源

```bash
# 不启动、不创建目录：核对实际参数与下一个目录编号
./evaluation/scripts/run_groot_eval.sh --mode dual2 --dry-run

# 每环境150轮，1200动作步；步数参数只允许用于 fast-debug
./evaluation/scripts/run_groot_eval.sh --mode single2 --preset fast-debug --debug-max-steps 1200

# 只检查动作可缩短为80步，并按需保留资源监控等诊断文件
./evaluation/scripts/run_groot_eval.sh --preset fast-debug --debug-max-steps 80 --diagnostics
```

快评默认每环境一个窗口；`success_rate=null`、`official_episodes=0`，不能当作正式成功率。正式模式不设短 runner 上限，保留 B1K 原生 timeout、任务成功判定、动作与策略相机。Task21完整回合可达38373动作步。

已有单卡双环境显存峰值约43.55GiB；双卡4环境尚未完成实际验收，仍需自行先用短窗确认资源。缓存默认在 `/nvme3/tmp/memory_vla/groot_runtime_gpu{0,1}/appdata`，与结果目录分离，不随结果清理。可用 `--appdata-gpu0`/`--appdata-gpu1`覆盖。共享 Policy Server 接收多个客户端但串行推理，双卡4环境发送两份 batch=2 请求。

Ctrl+C 只停止本次启动的队列及其仿真进程组。结束后用 `nvidia-smi`检查显存释放；本次文件精简没有改变仿真、模型或 worker 关闭逻辑。
