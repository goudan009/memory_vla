# 运行环境

4090 负责源码维护、工作台和仿真评估，A100 负责多卡训练 / 推理，RTX 6000 Ada 负责单卡小规模验证。统一源码同步、A6000 独立训练 / 仿真环境及当前验收范围见 [工作台设备](../train/DEVICES.md)。以下为原有环境与历史记录。

这里放各版本共用的仿真环境说明与已有环境入口。v1 模型/训练依赖随版本维护，见 [v1 环境说明](../src/v1_groot_memoryvla/README.md#环境与源码来源)；原版基线见 [baseline 环境说明](../baselines/groot_n1d6/env/README.md)。各模型服务与仿真进程使用独立环境。下列 π0.5 条目为历史环境记录。

## 4090：公共仿真环境

- 项目入口：`env/b1k2025-sim` → `/nvme3/envs/b1k2025-sim`（Conda 环境，由 WZR 环境复制）。
- 主要组件：Python 3.10.20、PyTorch 2.5.1+cu121、Isaac Sim 4.5.0、OmniGibson 3.7.1。
- OmniGibson 源码：`/nvme3/shared/b1k2025/BEHAVIOR-1K/OmniGibson`，由公共环境使用；原 `src/simulation` 软链接目录已移除，外部源码保留。
- 仿真资源统一入口：`assets/simulation/`。默认由 `scripts/b1k_python.sh` 和评测启动器读取；目录结构与当前缺失资产见 [根 README](../README.md#资产与数据路径)。
- 已验证：`turning_on_radio` 任务运行 200 步，三路 RGB 正常；π0.5 策略推理及成功率测评尚未验证。

在项目根目录运行，默认使用 GPU 1：

```sh
sh scripts/b1k_python.sh scripts/check_b1k_sim.py --steps 200 --output outputs/b1k_check
```

运行其他仿真脚本也使用 `sh scripts/b1k_python.sh <脚本路径>`，它会设置资源路径并隔离用户私有 Python 包。

已验证结果：`outputs/b1k2025_smoke_20261104/`。当前使用 Conda 内的 Isaac Sim，`/opt/isaac-sim-4.5` 保留但未使用。

## 4090：π0.5 推理环境

- 项目入口：`env/b1k2025-policy` → `/nvme3/envs/b1k2025-policy-public`（独立 Python 3.11 venv，与仿真环境分开）。
- 策略源码：`/nvme3/shared/b1k2025/policy`；权重使用 `baselines/behavior_50t_checkpoint`。
- 配置和依赖导入已验证；当前 Codex 执行容器看不到 NVIDIA 设备，真实 GPU 权重加载及动作推理尚未验证。

```sh
sh scripts/b1k_policy_python.sh scripts/test_pi05_policy.py
```

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
