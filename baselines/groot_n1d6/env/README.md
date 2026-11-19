# 原版基线环境

baseline 使用自己的模型环境，依赖依据是固定官方参考目录中的 `src/references_code/Isaac-GR00T/pyproject.toml`、`uv.lock` 和安装说明；不引用 v1 的依赖清单，以免后续 v1 升级改变基线。

从项目根目录，在已准备好上游模型依赖的 baseline 环境中使用入口；入口自行定位官方源码，无需在参考目录内安装或生成包元数据：

```bash
python baselines/groot_n1d6/serve.py --help

# 如需调用整理后的公共通用入口，显式指定固定参考代码
PYTHONPATH=src/references_code/Isaac-GR00T python -m evaluation.rollout_policy --help
```

普通 pip 不读取上游 uv 的 CUDA wheel 来源设置。新环境需按固定上游说明准备 Torch/FlashAttention/CUDA；上游锁文件包含 workspace 依赖，不能假定在当前参考快照中直接 `uv sync --locked` 可用。此处不复制依赖清单，也不自动安装或修改已有环境。

与公共仿真进程分开运行，通过同一 Policy Server 协议通信。当前已有模型环境的 DeepSpeed 导入受缺少 `CUDA_HOME` 限制，新增入口不代表已验收真实训练或权重推理。
