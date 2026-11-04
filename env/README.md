# 运行环境

4090 用于仿真评估，A100 用于模型训练。

## 4090：公共仿真环境

- 项目入口：`env/b1k2025-sim` → `/nvme3/envs/b1k2025-sim`（Conda 环境，由 WZR 环境复制）。
- 主要组件：Python 3.10.20、PyTorch 2.5.1+cu121、Isaac Sim 4.5.0、OmniGibson 3.7.1。
- OmniGibson 源码：`src/simulation/omnigibson` → `/nvme3/shared/b1k2025/BEHAVIOR-1K/OmniGibson`。
- 仿真资源：`datasets/simulation/b1k2025` → `/nvme3/shared/b1k2025/datasets`。
- 已验证：`turning_on_radio` 任务运行 200 步，三路 RGB 正常；π0.5 策略推理及成功率测评尚未验证。

在项目根目录运行，默认使用 GPU 1：

```sh
sh scripts/b1k_python.sh scripts/check_b1k_sim.py --steps 200 --output outputs/b1k_check
```

运行其他仿真脚本也使用 `sh scripts/b1k_python.sh <脚本路径>`，它会设置资源路径并隔离用户私有 Python 包。

已验证结果：`outputs/b1k2025_smoke_20261104/`。当前使用 Conda 内的 Isaac Sim，`/opt/isaac-sim-4.5` 保留但未使用。

## A100：训练环境

- 环境地址：`/home/1024201092WYH/.venvs/lerobot_pi05`（venv 环境）。
- 主要组件：Python 3.10.12、LeRobot 0.4.3、PyTorch 2.7.1+cu126、Transformers 4.53.2。
- 已有记录：π0.5 完成过 5 步训练，日志位于 `/home/1024201092WYH/env_build_logs/pi05_train_smoke.log`，checkpoint 位于 `/home/1024201092WYH/train_smoke/pi05_smoke/checkpoints/000005`。

在 A100 机器上查看训练参数：

```sh
/home/1024201092WYH/.venvs/lerobot_pi05/bin/lerobot-train --help
```

权重：`/nvme3/huggingface/train_model/pi/pi05_base`；兼容目录：`/nvme3/data_dyd/compat/pi05_base_lerobot052`。正式训练前需核对权重、代码和数据配置是否匹配。

备选 OpenPI 环境：`/nvme3/projects/vla_memory/code/openpi/.venv`，已有导入及配置加载记录，尚未完成真实训练验证。
