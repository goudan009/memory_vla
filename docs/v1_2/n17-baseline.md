# V1.2：GR00T N1.7 原生 Baseline 迁移架构

> 当前阶段：源码骨架、上游接口和防误启动契约已建立；没有在本机启动作业，未完成真实环境、工作台或仿真验收。只做无记忆 V1.2，不开始 V2.2，不改变正在运行的 V2.1。

## 目标及边界

- V_1_2 是已获用户确认的 N1.7 原生无记忆底座，不是历史 V_1 / V_2_1 的重命名。登记在 train/run_identity.py，独立 backend=v1_2；launch_ready=False，工作台 options.model_versions 暂不暴露为可启动训练版本。只有实际专用后端和验收通过后才能开放；不能为绕过校验改用 V_1 或 V_2_1。
- 保留 N1.6 的环境、源码快照、权重、训练/推理/评估记录；不停止当前任务，不覆盖数据或统计。
- 暂不设计或登记 V_2_2 网络。之后以经过闭环验收的 V1.2 为唯一新底座，讨论新增记忆模块。
- 真机最终沿用同一 N1.7 原生 policy API，单独验证真实机器人传感、动作、标定和控制，不把仿真 R1Pro 的 23 维控制器直接发送给真实移动双臂。

## 固定参考来源

| 项目 | 来源 |
|---|---|
| NVIDIA 原生底座 | [nvidia/GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B) |
| VLM | [nvidia/Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B)，需本机账号接受许可并缓存 |
| 首次官方闭环 | [BEHAVIOR 2026 基线手册](https://behavior.stanford.edu/challenge/baselines)，官方提供 turning_on_radio 权重 |
| 官方挑战适配 | [wensi-ai/Isaac-GR00T](https://github.com/wensi-ai/Isaac-GR00T)，代码已核对 commit ace36d935b376fbf25cd56371e23877b95407c40（默认分支快照）；本地核对 Git HEAD 与实际 checkpoint 的匹配 |
| 官方评估环境 | 手册指定 BEHAVIOR-1K v3.9.3；本地核对 tag、依赖、资产、入口及参数。远程会话未直接取得该 tag 的 evaluator 文件，不以 main 的接口冒充已验收 tag |

独立 N1.7 Python 3.10 环境，官方 fork 使用 uv lock；不要复用或修改 N1.6 Python。2026 仿真环境与现有 B1K2025 分离。先核对是否已有本机可用 checkout、缓存和评估资产，再下载缺失资源；不下载整个 3.27TB 数据集。官方权重下载后记录出处、实体路径、所有权重和 processor 文件 SHA256；不得从目录名或模态配置推断权重是官方或某项任务微调。用 contracts.checkpoint_digest 计算实体摘要，official_serve_spec/official_radio_policy 必须传来自已核实下载记录的 verified_radio_sha256，禁止以任意当前目录的摘要冒充来源。摘要只验证文件一致性，不是官方发布的签名。

## 代码分层

| 层 | 新代码/现有代码 | 职责 |
|---|---|---|
| 模型 | 独立官方 N1.7 fork | 原生 Cosmos-Reason2 + Flow Matching DiT；不复制 N1.6 网络、不自行修改loss |
| 版本契约 | src/v1_2_groot_baseline/contracts.py | 固定源 revision、检查模型类型、模态、horizon，生成只读 native serve/eval spec |
| 2025 数据 | src/v1_2_groot_baseline/data_contract.py | 核对现有82/23命名/切片，生成原生 NEW_EMBODIMENT 候选配置 |
| 推理 | src/v1_2_groot_baseline/policy.py | 官方2026 wrapper；未来2025独立权重的原生 flat-observation wrapper |
| 训练 | src/v1_2_groot_baseline/training_contract.py | 基于官方 train_b1k.py 的候选配置，明确 pre-accumulation batch；不可直接执行 |
| 工作台 | train/dashboard.py、train/dashboard_remote.py、train/run_identity.py | 同步源码、身份、资源、计划及启动；V1.2专用后端尚待本机接入验收 |
| 推理/评估控制 | evaluation/scripts/groot_inference_launch.py、run_groot_server.sh、groot_inference_remote.py、groot_web_launch.py | 复用现有 manager/控制器接入新 provider，旧 N1.6 launcher 已拒绝 N1.7 错误路由 |
| 验收 | tests/test_v12_baseline_contract.py | 合成CPU契约测试；真实模型、视频及GPU闭环不能由这些测试替代 |

所有生成的 argv 都是供现有工作台绑定的计划，不是另一个 shell 启动入口。现有 workbench CLI / HTTP 继续是唯一作业入口。现阶段没有新调度器、无自动启动，版本登记不等于已可运行。

## 迁移方式：先验证官方，再适配2025

### A. 官方 2026 参考（第一优先）

原生 serve_b1k.py + 原生 OmniGibson evaluator；WebSocket、默认 temporal_ensemble、RTC关闭。原始 proprio 61维，实际选取25维状态，动作23维；三个相机、有效16帧动作标签。实际模型 padded horizon/维度以 checkpoint config/processor 为准，不把 N1.6 的50×128复制过来。

上游 wrapper 使用共享动作缓存，先一环境一服务，逐回合 reset；禁止多个独立环境共享 temporal buffer。先核对官方 wrapper 的默认 text_prompt（代码为通用提示）与权重训练的任务语言及官方协议；原生复现保留原行为，不擅自覆盖。若存在错误，记录具体证据并由用户确认影响效果的变化。原生 temporal ensembling 不等于 RTC。

只做官方权重的闭环接口验收：建议从一个公共测试实例开始，确认真实模型输出、控制器执行、完整视频及物理成功记录。一次成功不能证明统计性能，后续比较回合数须单独锁定。官方权重只属于 turning_on_radio，不能当作 Task40 权重或训练起点。

本机工作台接入：在现有 InferenceManager 中增加显式 V_1_2 + b1k2026_websocket 路由，使用已核实的实际设备模型Python、上游路径、权重路径和端口，复用同一进程生命周期/日志/身份和 /healthz；在现有 LaunchManager 中增加对应协议，原生 evaluator 的 per-rollout JSON、真实 success、Q-score、视频关联到原结果目录。保留 native 原始文件，不编造2025 task_id、seed或任务进度。preview无副作用，start同一请求；运行、暂停/停止、清理、关联checkpoint按Skill。禁止独立CLI绕过或用旧run_groot_eval.sh误连WebSocket服务。真实接入验收前旧入口明确拒绝。

### B. 2025 Task40 无记忆基础模型（第二优先）

优先复用现有已经转换的 LeRobot v2.1、30Hz、82维状态、23维动作和原180/20划分，N1.7挑战fork支持v2.1/v3.0。无需将82维向量强行转换成2026的61维。按原有命名生成 N1.7 NEW_EMBODIMENT 模态配置，原生 max_state_dim/max_action_dim 与 action_horizon 在通用 checkpoint 中核对。保留底盘速度和夹爪命令，双臂/躯干相对当前状态的动作定义。原始状态中的 robot_lin_vel 不能自动替换2026 robot-local base_qvel。

实际使用前必须核对训练loader读取视频、语言、frame_index及 action qpos，验证真实 observation/action 组顺序、单位、绝对/相对解码和 action_mask，训练时重新计算训练集统计。候选训练接口要求独立 train-only 数据视图、原 split、显式 horizon 及已核实通用权重 verified_generic_sha256；同时校验 episodes.jsonl 与实体 parquet 文件集合，拒绝验证集泄漏。不复用N1.6归一化，也不修改源数据目录。2025语义任务名称和动作协议沿用原代码，不能通过相同数字 ID 推测2026任务。

正式训练起点是通用 N1.7-3B；不继承 N1.6 或官方 radio 微调权重。模型结构、loss、optimizer保持官方原生，VLM冻结/原action expert更新为推荐选项。候选有效 horizon 16（官方挑战配方）或32（沿用2025监督窗口）；训练步数、有效batch、horizon、控制步数、归一化与评估回合在正式任务前再确认，不从2026官方8卡×2048、150000步直接照搬。上游 CLI 的 global_batch_size 是梯度累积前的全卡batch；例如4×2×8时应传8而不是64，实际有效64。

既有 N1.7 原生 Gr00tSimPolicyWrapper 和 PolicyServer 可作为旧2025 flat observation/ZMQ 适配候选，但需要实际两端序列化、reset、get_modality_config、动作块和闭环验证；当前 B1K2025 route仍封闭，直到兼容性验收。正式训练必须接入现有工作台、规范 checkpoint 及暂停恢复；上游忽略数据skip/重置采样行为不能直接算严格续训。超过2000步先按Skill完成真实预检；不能把上一轮 N1.6 的候选配置直接复制成N1.7科学配置。

## 本机交接与验收门槛

1. 阅读 AGENTS.md、robot-vla-experiment-ops Skill、docs/workbench/agent.md 和本机规则；只读 services/options/jobs，核对设备和实际路径。保留运行中的V2.1；优先使用有容量的RTX6000 Ada模型环境。当前仅获授权架构与基线链路搭建。
2. 按真实目录和任务记录锁定V_1_2新实验 testNN/expNN（本会话未分配编号），遵循 Output/test_NN/experiment_NN/task.md、progress.md、workflow/plan-vNNN.json；编号及真实run ID由用户/本地工作台核对，不能以V1.2推导编号或另建输出地址。
3. 拉取并核对固定fork、通用权重、官方 radio 权重、v3.9.3仿真，补齐独立环境和缓存。执行CPU测试 `PYTHONPATH=src:train python -m pytest -q tests/test_v12_baseline_contract.py`。按照上述 A 完成原生推理与评估接入和短闭环，归档真实工作台job、source digest、checkpoint SHA、协议、视频和回合结果。没有成功回合或接口未验证时如实记录，不能称基线已跑通。
4. 在 A 验收后，按 B 检查2025 data/processor/model和旧runner兼容性，再提交正式训练配置供确认。正式训练后用真实Task40闭环评价基础动作质量、任务阶段和成功率；不直接启动后续V2.2。

2026-11-27复核：加强 processor 类、model/processor padding、相对/绝对动作表示、归一化区间、checkpoint digest、原始摄像头及语言字段、train-only文件集合检查。训练和评估按 Skill 填一句简短备注，训练 notes 示例「V1.2 / N1.7：原生无记忆基线」，2025评估 run_tag 示例 v1_2-n17-baseline-task40。真实权重 schema 和上游接口必须本地实测，出现不匹配应查证，不放宽检查或修改科研配置来掩盖错误。

本次远程没有真实 GPU/文件系统访问，也未取得新增CPU CI运行结果。只交付源码骨架与迁移合同，不将未验证的 workbench route 标为 launch_ready。
