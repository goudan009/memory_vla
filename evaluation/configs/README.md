# 公共评估配置

Baseline、V1、V2 共用任务集合、实例、seed、动作执行节奏、Episode 终止条件、评分和录像设置。模型内部参数、记忆长度与训练参数放在版本自己的 `configs/`。

当前正式入口是 `evaluation/scripts/run_groot_eval.sh`，参数由 CLI 和现有脚本默认值提供，详见 [评估说明](../README.md)。正式规则由 [evaluation.md](../evaluation.md) 固定。此目录暂不另建一份会偏离实际脚本的 YAML 默认值，也未引入新的配置加载器。

后续如将评估 preset 抽出为文件，在这里实现唯一配置来源，并同步接通所有版本的公共入口。已有根 `configs/evaluation/` 属于旧仿真/旧 SERF 入口，仍有兼容脚本引用，保留原路径。
