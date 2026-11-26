# 调查复现说明

Task40新增弱标签与历史事件的独立复现顺序见 [label_extension/reproduce.md](../label_extension/reproduce.md)。原脚本及根目录标签保留v001认证基线，新标签在`label_extension/`，不由下列v001命令生成。

这些脚本生成本次交付物，不改模型，不训练。完整重提取依赖原始数据与真实资产；仅阅读报告、标签、运动链和源码副本不需要加载Isaac。

## 已使用的环境与外部依赖

在仓库 `/nvme3/projects/memory_vla` 根目录执行。CPU脚本可用 `/nvme3/envs/b1k2025-sim/bin/python`，已核对包含numpy、pandas、pyarrow、h5py、scipy、av、Pillow、torch；USD解析另需pxr，画SVG需Graphviz的dot。不要用无关环境中的同名python。所有CPU命令不启动Isaac应用。

实际路径、revision及哈希见 `../report.md`、`../samples/version_manifest.json`、`../robot_description/asset_provenance.json`。完整再生成还需要：

- 转换后200条Task40 parquet、600路视频及原始256维parquet和原分辨率视频，位于仓库 `assets/datasets/` 的已记录引用。
- `/nvme3/datasets/b1k2025_raw_task40_v2_audit/task-0040/` 的200个原始HDF5。
- 实际R1Pro USD/URDF、OmniGibson源码、`checkpoints/groot_n1d6_base/` 的processor配置和统计。
- 当前仓库Git对象、历史V2备份及 `outputs/diagnostics/v1_real_replay_features.pt`，用于重新核验模型证据。相关源码和一个真实张量样例已随包提供。
- `finalize.py` 查询本机8765工作台，归档两个既有作业；仅在原宿主机可原样执行。它不会启动或停止作业。

脚本默认更新交付目录，应在工作副本中复现，保留原包及其MANIFEST用于对照。不要直接重写已封存证据。

## 仅核验和重新打包

```sh
/nvme3/envs/b1k2025-sim/bin/python v2_1_preflight/scripts/package.py --check-only
/nvme3/envs/b1k2025-sim/bin/python v2_1_preflight/scripts/package.py
```

检查覆盖200条/647555帧、nullable标签与valid一致性、逐episode统计、记录值与NPZ一致、历史边界策略、82/23映射、43/44资产计数、FK误差、参考比较报告、视频元数据和源码摘要。它不会重新回放200条，不重新扫描完整HDF5或做模型forward。打包输出在目录同级，包内附MANIFEST；读取tar成员再次核验每个文件摘要。`--check-only`只读。

## CPU再生成顺序

已有包包含完整解码与schema，无需重新启动仿真。以下在工作副本中执行，顺序有依赖。

```sh
/nvme3/envs/b1k2025-sim/bin/python v2_1_preflight/scripts/offline_decode.py --all --schema-only --official-report v2_1_preflight/samples/workbench/20261126-201130-d9b4bb46/preflight_extract.json
/nvme3/envs/b1k2025-sim/bin/python v2_1_preflight/scripts/data_audit.py --bundled-started --decoded-report v2_1_preflight/samples/cpu_decode_report.json
/nvme3/envs/b1k2025-sim/bin/python v2_1_preflight/scripts/robot_assets.py
/nvme3/envs/b1k2025-sim/bin/python v2_1_preflight/scripts/build_mapping.py
/nvme3/envs/b1k2025-sim/bin/python v2_1_preflight/scripts/inventory.py
/nvme3/envs/b1k2025-sim/bin/python v2_1_preflight/scripts/model_audit.py
/nvme3/envs/b1k2025-sim/bin/python v2_1_preflight/scripts/samples_and_validation.py
/nvme3/envs/b1k2025-sim/bin/python v2_1_preflight/scripts/finalize.py
/nvme3/envs/b1k2025-sim/bin/python v2_1_preflight/scripts/package.py
```

`--schema-only`按已初始化schema重提取，另写 `cpu_redecode_report.json`，不会声称新做了与独立官方参考的比较；原37条比较报告仍保留。全量解码输出是确定性的，在未改变依赖时与包内CPU NPZ逐值相同。`data_audit.py --bundled-started`从包内记录值重建历史±8帧mask，并逐episode核对原认证的切换点及正负未知数量，不依赖已迁移的历史200个NPZ。省略该开关才使用历史认证原文件及其SHA256。

`inventory.py`重新生成基础清单，`samples_and_validation.py`与`finalize.py`补全全量样例、范围和认证范围，不能仅运行基础清单就当成最终交付。`robot_assets.py`会以原资产重新生成轻量USD，并用6条×25帧×双末端做300次数值对照；不启动物理回放。`model_audit.py`只在CPU读取既有缓存，不能用其结果声称做过新模型推理。

## 官方接口验证的历史复现入口

本次已通过工作台运行两次只读提取，分别完成13和37条后主动停止。原配置、日志、状态和报告均在 `../samples/workbench/`。完整200条导出由CPU完成，不能把这两个作业称为200回合完成。

`workbench_extract.py preview --gpu <实际卡号>` 调用既有BenchmarkManager的prepare；`start`调用同管理器start。重新启动前必须按仓库 `docs/workbench/agent.md` 查询实时容量、先preview再核对；该步骤会实际占用GPU，因此不属于上述CPU复现命令。`sim_extract.py`仅通过工作台worker和 `scripts/b1k_python.sh` 启动，不直接运行，不重启工作台服务。

若重新取得独立官方NPZ，把它们放入工作副本 `samples/decoded/`，并使用对应官方报告运行 `offline_decode.py`（不带 `--schema-only`）。脚本先核对每个参考NPZ的原报告SHA256，再逐数组比较；若目标已经是CPU扩展文件则拒绝冒充官方参考。原37条六字段参考文件已被扩展CPU导出覆盖，无法仅凭当前包重新完成这一独立认证；本次报告保留的是原比较的历史证据。
