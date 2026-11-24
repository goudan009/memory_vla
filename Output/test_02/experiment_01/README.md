# Test 02 / Experiment 01：任务状态引导的历史记忆

V2 探索任务，验证预测状态能否引导认知记忆写入、检索和融合，改善 Task40 阶段决策。真实标签已完成200条、647555帧的对齐与认证，仅启用可靠的started字段。用户授权后在本机4090完成唯一run001的1000步验证训练；冻结检查和末尾完整HF重载通过。

集中评估已结束：20条本轮V2更新留出episode、100个机制窗口全部完成；started F1=0，预测事件0。记忆/动作存在数值响应，未证明改善。12组V1/V2共24次恢复尝试全部结束，均因图像不一致无效，有效配对0/12，闭环效果未能验证。工作台complete表示执行结束，科学结果未达标。权重、曲线和证据已归档，本次自有策略/仿真已退出。[最终验收](acceptance.json) · [运行进度](progress.md)。

[任务书](task.md) · [代码与调用说明](../../../src/v2_groot_memoryvla/README.md) · [运行进度](progress.md) · [报告](report.md) · [计划v007](workflow/plan-v007.json) · [训练结果](train_runs/run001/training_summary.json) · [训练曲线](train_runs/run001/training_curves.png) · [离线评估](eval_runs/eval001/evaluation_summary.json) · [认证标签](evidence/certified_started_labels.json) · [198项测试](evidence/full_review_gpu_ready.xml) · [实验索引](../../README.md)
