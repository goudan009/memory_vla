# 具身记忆研究架构 · V2.1

## 当前方案的主要阅读入口

**[V2.1 完整方案网页（HTML）](V2.1/index.html)**

这是用户后续查看与讨论架构的主要入口：按一次真实决策依次展示完整网络、双记忆库、联合检索、速度场修正、训练反传及评估。独立 HTML 内置 SVG、原生 MathML；无外部脚本依赖。GitHub 文件页通常展示源码，下载 HTML 后用浏览器打开即可。未启用或变更 GitHub Pages。

当前设计为 A（局部拓扑 + 全局）优先、B（FK + 全局）备选、C（单层全局）对照，尚未实现或正式训练。V2.1 的新实现拟在 src/v2_1_groot_memoryvla/；旧实验及 checkpoint 身份不变。

## 技术依据与历史记录

| 文件 | 职责 |
|---|---|
| [V2.1 架构原理与统一符号](V2.1_embodied_memory_architecture.md) | 理论与数学依据，保留完整研究推导 |
| [V2.1 实施协议](V2.1_implementation_protocol.md) | 训练、模型参数、评估与开发门禁的依据 |
| [旧版 SVG](V2.1_detailed_network.svg) | 历史绘图，已由新版 HTML 内嵌总图取代 |
| [V2.0 架构](V2.0_embodied_memory_architecture.md) | 历史设计 |
| [早期记忆构建设计](2026-11-25_memory_construction_design.md) | 历史讨论，不覆盖当前方案 |

## 接续讨论和开发约定

1. 新对话、Agent 或汇报接续时，优先读取 V2.1/index.html，再根据需要核对架构原理与实施协议。后续方案讨论以此 HTML 为主要展示载体，不另做与当前公式和架构脱节的网页。
2. 公式沿用 V2.1 统一符号，新增符号必须定义含义、维度与梯度边界；架构、训练或评估变化同步更新 HTML 与对应技术依据。
3. HTML 只展示研究方法、训练、评估与必要的开发门禁，不包含目录整理等管理信息。已归档的实验结果不覆盖、不改名。
4. 模型代码尚未实现；训练前须完成真实模型形状、标签认证、物理控制映射、先验资产、梯度及资源检查。网页不是实测结果，也不是现有可运行 CLI。

## 相关依据

- [V1 源码说明](../../src/v1_groot_memoryvla/README.md) 与 [顺序窗口训练协议](../../src/v1_groot_memoryvla/RETRAIN.md)
- [历史 V2 状态监督实验报告](../../Output/test_02/experiment_01/report.md)
- [公共评估协议](../../evaluation/evaluation.md) 与 [工作台执行手册](../workbench_agent.md)
