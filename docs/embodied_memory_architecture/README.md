# 具身记忆研究架构 · V2.1

## 主要阅读入口

**[打开 V2.1 完整交互式方案网页](V2.1_review_new.html)**

此 HTML 是后续架构、训练和评估讨论的统一展示入口。页面只展示研究相关内容：完整网络、双库与状态增强、联合检索、速度场修正、A/B/C、梯度、训练及评估。内置重新绘制的 SVG、独立公式排版和交互，无外部脚本依赖。GitHub 文件页通常显示源码；下载 HTML 后在浏览器中打开即可。未启用或修改 GitHub Pages。

## 当前技术依据

| 文件 | 职责 |
|---|---|
| [V2.1_review_new.html](V2.1_review_new.html) | 唯一主要用户审阅网页 |
| [V2.1 架构原理](V2.1_embodied_memory_architecture.md) | 完整公式、定义、数学推导与方法设计 |
| [V2.1 实施协议](V2.1_implementation_protocol.md) | 开发、训练、模型参数、评估及门禁 |
| [archive/](archive/) | 历史架构、早期构建稿、旧网页及旧 SVG |

当前 A（局部拓扑 + 全局）优先、B（FK + 全局）备选、C（单层全局）对照；尚未实现或正式训练。拟新增 src/v2_1_groot_memoryvla/，保留原实验和 checkpoint 身份。V2.0及早期稿全文保留在 archive，不覆盖现有结果。

## 后续维护约定

- 后续对话和 Agent 优先读取 V2.1_review_new.html，再核对完整架构与实施协议；所有新的方案展示以此 HTML 的组织、符号和交互为基础。
- 架构、训练或评估变化同步更新 HTML 及相应技术依据；不额外产生重复审阅网页。公式用统一符号，说明张量维度、时间尺度、梯度边界。
- 主 SVG 保持精简；详细公式与解释分区呈现。发布前进行浏览器渲染、文字包围盒重叠和越界检查、配置切换及交互验收。目录管理不放进研究网页。
- 真实模型形状、标签认证、控制映射、先验资产及梯度与资源检查仍是开发和开训门禁；设计稿不是实验结果。

## 相关依据

- [V1 源码说明](../../src/v1_groot_memoryvla/README.md) 与 [顺序窗口训练协议](../../src/v1_groot_memoryvla/RETRAIN.md)
- [历史 V2 状态监督实验报告](../../Output/test_02/experiment_01/report.md)
- [公共评估协议](../../evaluation/evaluation.md) 与 [工作台执行手册](../workbench_agent.md)
