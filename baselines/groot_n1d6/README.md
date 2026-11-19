# GR00T N1.6 原版基线

入口固定使用 `src/references_code/Isaac-GR00T/` 的官方源码，提交 `9b37aa1ce69c73c6d165233fa88128283bba4508`。这里不复制或修改第二份模型。实际依赖及环境准备见 [env/README.md](env/README.md)。

在独立 baseline 模型环境中，从项目根目录执行：

```bash
python baselines/groot_n1d6/train.py --help
python baselines/groot_n1d6/serve.py --help
```

两入口只将固定官方参考目录加入 Python 路径，再调用原版训练/服务模块；不自动下载权重、安装环境或启动仿真。训练输出使用 `checkpoints/baseline/<运行>/` 与 `outputs/baseline/<运行>/` 的约定，并显式传给原生 CLI。

该目录表示模型侧原版入口；`evaluation/baseline/` 是评估结果目录，两者职责不同。评估连接这里启动的服务，与 v1/v2 共用根 `evaluation/scripts/`、评估协议和结果查看工具。现有外部 baseline 服务和正式 runner 不会因新增此入口自动切换。

原版预训练权重可作零样本参照。评估记忆训练的收益时，还需使用相同训练数据、预算、seed 和冻结设置微调一个无记忆对照；不能只与未微调原版比较。
