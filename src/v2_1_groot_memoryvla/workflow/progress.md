# V2.1 开发验收记录

- contracts：完成，基于 main a2a7b23 的 Task40 扩展标签、真实控制映射与 FK 审计。
- core：完成第一版，A/B/C/D、双库与状态增强、联合检索、masked residual、gamma。
- integration：完成代码层接入，原 V1 模型不修改；两阶段训练、服务、离线诊断、显式 shell 入口、断点保存。
- checks：CPU CI 三轮实际成功（10、12、13 项）；最新代码测试 commit de4263c546fb645a5b6c74410614ff29213aead8，run 45841624016，job 139984498016，13 passed in 4.22s。检查涵盖原 FM/Euler 路径的轻量替身对照、真实 FK 样例、标签质量、梯度、因果/压缩/reset、三先验、gamma 与 runtime roundtrip。真实 GR00T 权重/数据/GPU、工作台和仿真验收未运行。
- handoff：README 已提供本地检查命令和放行标准。PR #2 保持 draft；未合并 main，未提交正式训练或改动现有作业。

下一步：本地 Agent 依 README 进行实际环境检查；先 Stage1 预检，再由用户决定正式训练。待 Stage1 有效完成后执行 Stage2 预检。GPU 显存、速度、实际学习效果均未知。
