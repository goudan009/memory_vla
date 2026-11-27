# 训练自动归档

这是工作台维护功能，不改变实验方法、训练配置或 GPU 作业。

## 实施计划

1. 从工作台只读查询训练终态，接入现有 systemd 用户定时机制。
2. 原样保存指标、实际配置、恢复记录、checkpoint 元数据与源码记录；按固定规则生成一份汇总文档。
3. 单独提交并推送自动归档目录，保留仓库其他未提交和暂存的修改。
4. 用历史真实训练和隔离 Git 仓库验证内容、重复扫描及发布；安装并核对定时器。

验收：无需 AI 会话，终态后自动生成证据与汇总；重复扫描不重复提交；推送失败保留本地产物并重试；不启动、停止或修改训练和评估。

### 修复计划 v2：中断评估漏收

用户要求所有自动汇总进入 GitHub。已复现 Stage1 `task40_eval01` 的训练未补录，导致终态评估被已归档训练列表过滤；原生 `interrupted` 也未列入评估终态。

1. 用未归档历史训练及原生中断结果建立回归测试。
2. 评估独立归档并进入同一汇总，保留 run/checkpoint 原始身份；补全中断状态，继续排除运行中的评估。
3. 明确自动报告目录不受 Git 忽略，保留原始运行目录的媒体/大文件忽略策略。
4. 扫描补录真实结果，校验原始文件和重复扫描，提交并核对 GitHub 远端包含新报告及证据。

验收：Stage1的10个已完成回合以 interrupted / 工作台failed 如实展示；未归档训练不再阻止评估入库；无身份的旧评估保留未记录字段，不猜测模型归属。

## 使用入口

唯一汇总入口为 [reports/training/README.md](../../reports/training/README.md)。它由程序确定性生成，不需要 AI、提示词或 API key。不要手改该文档；人工分析继续写在各实验目录。

安装时补录 Exp13 的 16,000 步记录，其他首次发现的历史终态只建立基线。之后 A100 / RTX 6000 Ada 工作台新作业、已有作业的新终态均自动归档；训练恢复后沿用 run，保留每个归档事件。初次离线的设备恢复连接时会补录其可见终态。无需逐个登记未来 run。

触发范围为 `complete / completed / stopped / failed / cancelled`。GPU 作业仍由工作台控制；归档服务只读 GET 和 SSH，不调用训练、评估的 preview/start/stop，也不参与排队。定时器每五分钟扫描，工作台和训练设备需可达；删除作业或回收输出前应先确认已归档。

## 自动记录内容

- 原始逐步指标 JSONL（保留原始字节，按1000行分块；未变化的块复用），包括已有的 Loss、梯度、分体损失、LR、耗时、吞吐与显存字段。
- 工作台终态、实际配方及字段来源、初始化权重、严格恢复记录、数据划分、保存间隔、checkpoint JSON 元数据。
- 运行源码路径与 digest、冻结源码 manifest，以及 V1 / V2.1 的项目实现和训练入口。第三方依赖不做完整环境镜像。
- 训练日志末尾256 KiB，明确记录来源文件大小及截取偏移；完整原始日志仍在训练设备。
- 本机评估目录中已结束且明确绑定本 run / checkpoint 的闭环结果和逐回合记录。工作台判定失败但结果文件还标 running 时同时保留两个状态；运行中的评估暂不计入完成结果。

每个事件的 `MANIFEST.json` 记录原始来源、大小和 SHA-256。汇总只做均值、覆盖率、缺步及重复计数，不自动生成“为什么没学好”的结论。未记录的梯度不当作0；Loss变化不作为闭环成功的证据。工作台当前停止目标与原始训练预算分别显示。

权重、optimizer tensor、图像/视频和数据集不会进入这个 Git 目录。原始指标及必要文本证据随训练量增长；无需在线 AI 访问训练设备即可阅读。

## 运行与发布

服务和定时器在 `scripts/systemd/memory-vla-training-archive.{service,timer}`。在本机4090上安装：

```sh
systemctl --user link /nvme3/projects/memory_vla/scripts/systemd/memory-vla-training-archive.service /nvme3/projects/memory_vla/scripts/systemd/memory-vla-training-archive.timer
systemctl --user daemon-reload
systemctl --user enable --now memory-vla-training-archive.timer
```

服务执行 `python3 scripts/training_archive.py --once --publish`，更新文档及证据后，仅提交 `reports/training/`，再普通推送至 `origin/main`。其他已暂存或未暂存的修改会保留。它不会自动 pull/rebase、强制推送或解决冲突。`main` 是仓库现有主线，普通 push 也会发布主线上已有的本地提交；不要在此仓库保留尚不准备发布的本地提交。

网络失败或远端分支领先时，产物和本地提交保留，下轮重试 push，不重复提交相同文档。单设备离线不会阻断另一设备采集。状态及错误写在 `outputs/shared/training_archive/state.json`（不进入 Git），可用 journal 查看详细输出：

```sh
systemctl --user status memory-vla-training-archive.timer
journalctl --user -u memory-vla-training-archive.service -n 30 --no-pager
```

手动补录尚未归档的历史训练、执行单次扫描、或离线验证证据：

```sh
python3 scripts/training_archive.py --once --devices a100 --backfill v1__test01__exp13__run001
python3 scripts/training_archive.py --once --publish
python3 scripts/training_archive.py --verify-only
```

不加 `--publish` 只生成文件。程序通过文件锁避免手动调用和定时任务同时导出。

## 验收记录

- 单元测试覆盖原始指标统计、V2.1目标组合、稀疏梯度、重复step、不完整尾行、采集范围、恢复事件、失败重试、终态去重及评估绑定。
- 隔离 Git 仓库验证自动提交/推送和重复扫描；原有暂存内容及未暂存内容均保持。
- Exp13真实16,000步完成首次自动采集；离线验证原始指标可逐字节重组、全部导出文件哈希及所采集源码与冻结manifest一致。
