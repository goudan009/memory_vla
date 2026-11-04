# OpenPI 集成快照

`pi0.py` 与 `pi0_config.py` 是从 `/nvme3/projects/vla_memory/code/openpi/src/openpi/models/` 复制的记忆接入快照；入口模型在 `Pi0.embed_prefix` 接入 `vlamemory.MemoryModulation`。导入使用本项目 `vlamemory` 包，不依赖旧的 `pi05_memory_b1k` 绝对路径。

原 `code/openpi` 工作树仍保留其未提交改动，作为当前训练/推理工作副本；本目录快照是 `memory_vla` 项目下可审阅的集成版本。两处后续修改需要同步，避免分叉。

训练循环中的 episode/timestep 传递及 JIT 外的记忆写入尚未闭环，因此本快照说明模型接入代码，不代表端到端训练完成。
