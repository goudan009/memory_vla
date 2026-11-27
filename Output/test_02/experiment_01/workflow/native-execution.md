# plan-v005 执行入口

本次任务沿用 plan-v005；主线、预算和科学配置均未更改。身份由现有工作台核验为 `V_2_1/test02/exp01/run003`。run001和run002的历史记录保留。

已完成35项CPU回归、四A100真实step2暂停退出及同run恢复到4，模型、optimizer、scheduler、各rank RNG与样本游标均验收通过。初次预检PID1473647，恢复PID1490788。JSON证据在 `../evidence/native_random_v2/v2_1__test02__exp01__run003__preflight001-archive.json`；A100独立归档位于 `/nvme3/data_wyh/outputs/memory_vla/v2_1/test02/exp01/evidence/`。工作台已purge专属预检输出及快照；数据与历史实验保留。

正式训练已通过 preview/start 提交并实际四rank更新；PID1501955，run为 `v2_1__test02__exp01__run003`，输出为 `/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2_1__test02__exp01__run003`。请求见 native-formal-request.json。冻结训练源码digest为 `53a2c0ab6c747c0264988ca0b6c53e12c433cbc7354d6c542bfb95158d818615`。监督客户端放在本实验workflow下，不进入训练源码hash。训练源未因后续监管代码而变更。

继续机制使用既有 `memory-vla-v2-1-training.timer`，每120秒调用 native_supervisor.py，已启用user linger。旧的run001及Stage2巡检入口不再由此service调用。所有GPU作业仍通过8765工作台原有HTTP接口；客户端只检查依赖、记录动作、提交相同配置preview/start，不另起GPU队列。

主线为6000 optimizer steps完成及完整checkpoint hash核验 → A100 GPU0/端口5567推理（Euler4） → 工作台分配标准 `V_2_1/baseline` 评估目录，dual2、每GPU5回合、实例0–4/5–9、seed0 sequence、执行8、原生终止 → 10有效回合/视频验收 → 归档指标和95% Wilson区间 → 释放本次推理服务。原有5557/5558服务保留。仿真原生seed按环境内回合累计，并非每实例直接用其ID作seed。

watchdog_state.json/watchdog_log.jsonl、latest-detail.json和latest-resources.json保存在 `../evidence/native_random_v2/`；progress.md同步更新。失败、非有限loss、身份不一致、未知提交结果或用户暂停时保留证据，不改变参数、不启动Stage2。旧Stage1和官方B1K历史结果仅作性能参考，既有环境数、seed或终态不一致时明确记录，不能宣称配对改善。

可复现CPU检查：`python3 -m unittest discover -s Output/test_02/experiment_01/workflow -p test_native_supervisor.py -v`。只读巡检：`systemctl --user status memory-vla-v2-1-training.timer` 和工作台 `/api/training/detail?device=a100&run=v2_1__test02__exp01__run003&arm=baseline`。本轮运行中请勿改动训练配置或重新提交同名作业。
