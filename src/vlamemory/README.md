# VLA Memory 核心模块

本目录包含记忆核心实现（迁自 `vla_memory/pi05_memory_b1k/src/vla_memory`），仅保留代码。

- `memory/`：共享记忆库、均匀记忆、动作子空间条件检索
- `modulation.py`：检索结果转 prefix token 的注入接口
- `data/`：episode sampler、历史帧窗口、memory dataloader
- `config.py`：记忆类型、容量、动作分组、注入模式

未迁入 benchmark/评测代码、第三方完整仓库、环境、权重、数据与缓存。OpenPI 的记忆接入见 `integrations/openpi/`；训练期记忆写入仍需在 JIT 外由训练循环推进，尚未闭环。
