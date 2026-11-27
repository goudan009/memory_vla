# Exp13 / V1 全50任务：16000步诊断资料

这是2026-11-27从现有工作台和原始运行文件导出的时点快照，用于在线AI检查训练实现、配置和闭环失败。没有启动新实验，也不代表已定位故障。训练运行是 `v1__test01__exp13__run001`，对应权重 `v1__test01__exp13__run001__step016000`。

## 建议阅读顺序

| 要看什么 | 文件 |
| --- | --- |
| 当前实际训练配方、证据优先级与已知限制 | [ACTIVE_RECIPE.md](ACTIVE_RECIPE.md) |
| 工作台实际生效参数及来源 | [training/effective_parameters.json](training/effective_parameters.json) |
| 每个optimizer步的原始loss，无平滑 | [training/loss_per_step.csv](training/loss_per_step.csv) |
| 每步梯度范数、学习率和部位loss，缺失字段留空 | [training/gradients_and_body_loss.csv](training/gradients_and_body_loss.csv) |
| 按625步一轮汇总的loss | [training/epoch_means.json](training/epoch_means.json) |
| 保留全部原始字段的16000条指标，按1000步拆成16个文件 | [training/raw_metrics](training/raw_metrics) |
| 训练stdout、恢复记录和启动参数 | [training/recorded](training/recorded) |
| 对应16000步checkpoint的模型与processor配置、动作归一化统计 | [training/checkpoint](training/checkpoint) |
| 初始化基座配置和归一化统计 | [training/base_checkpoint](training/base_checkpoint) |
| 实际训练源码，区别于当前main | [source/src/v1_groot_memoryvla](source/src/v1_groot_memoryvla) |
| 该checkpoint绑定的闭环结果索引 | [evaluation/index.json](evaluation/index.json) |
| Task40原始汇总和逐回合终止原因 | [evaluation/task40_eval15/results.json](evaluation/task40_eval15/results.json)、[episodes.jsonl](evaluation/task40_eval15/episodes.jsonl) |
| 原始运行器逐回合完整记录、goal predicates | [evaluation/task40_eval15/runtime/gpu0/episodes.jsonl](evaluation/task40_eval15/runtime/gpu0/episodes.jsonl) |
| 在线阶段/场景重置/动作记录 | [evaluation/task40_eval15/stages](evaluation/task40_eval15/stages) |
| 初始完整配方及论文/代码差异核对（历史版本） | [historical/training_review_20261124.md](historical/training_review_20261124.md)、[v1_paper_fidelity_review_20261124.md](historical/v1_paper_fidelity_review_20261124.md) |
| 每个文件的来源、字节数、SHA256与截取规则 | [MANIFEST.json](MANIFEST.json) |

## 如何使用原始指标

`raw_metrics`中的16个JSONL按文件名排序拼接，会逐字节还原A100上的原始`memory_v1/metrics/metrics.jsonl`。16000条训练记录覆盖step 1–16000，无重复、无缺失；原文件SHA256为`8557ae96dd6bb98c7dbdc4e8e178ecdb28a4ac2d3569327110fd89eae8ca2969`。

CSV用于快速阅读，原始JSONL保留`memory_gradients`、模块梯度、部位flow MSE、episode选择、耗时和显存等字段。`component_grad_norms`和`memory_gradients`只有1606个记录点；未采样、JSON null或CSV空白都不能当作梯度为0。总梯度与部位flow MSE有每步记录。此处只有原来记录的梯度统计，没有完整梯度张量，也没有每步所有参数的更新差值。

可在任何有Python3的仓库副本执行：

```sh
python3 scripts/export_exp13_diagnostics.py --verify-only
```

它只核对本地文件哈希及原始指标覆盖，不需要原始数据集、模型、SSH或GPU。MANIFEST记录203份导出载荷文件；本阅读说明与ACTIVE_RECIPE是在导出后整理的导航，不属于原始载荷。

## 评估覆盖与限制

导出时Task40评估`task40_eval15`（工作台job `6f32f5b13ab3`）仍为running，计划10回合，完成4个有效回合，成功0/4。四个已完成回合均为原生timeout截断。该时点不能表述成“10回合全部失败”，也不是50任务完整评估。

原始`results.json`、`episodes.jsonl`和阶段日志在各自读取时截取，运行中写入可能产生少量时点差异；以各文件时间、计数和MANIFEST为准，不拼接推断未完成回合。动作chunk日志只导出每个环境/回合/实例/seed的前8及最后8条完整记录，保留原始行和完整日志截取时的哈希/字节数；其余阶段文本保留完整已写入行。动作样本不足以覆盖整个回合，不能据此断言中间动作或做完整轨迹归因。

视频、RGB图像、完整动作chunk大日志、模型/优化器权重和训练视频/Parquet未纳入Git；对应路径和视频文件名保留在原始记录中。需要视觉行为判断仍须查看工作台录像。本目录未新增离线评估。

## 可直接交给在线AI的提示

> 请先阅读本目录ACTIVE_RECIPE、effective_parameters、epoch_means和evaluation/index，再按需读取step15001–16000原始指标、实际源码以及Task40逐回合记录。请区分可证实的实现错误、训练现象和仍需闭环录像/权重验证的假设，给出对应文件及代码位置。以保存的实际源码与worker_argv为准，不把当前main、历史初始配方或未采样梯度当作当前事实。不要把4/10回合的运行中快照当作完整50任务评估，也不要直接修改已确认的训练方法。
