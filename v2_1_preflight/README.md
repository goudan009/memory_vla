# V2.1 开发前调查交付

最新标签入口：[Task40扩展v002](label_extension/README.md)，包含21状态、认证/弱/未知分级、历史事件、独立掩码及多视角样例。请读取`label_extension/task40_labels.parquet`；下列根目录标签保留原认证基线。

[调查报告](report.md)记录200条Task40示教、647555帧、真实机器人运动链、82维本体/23维动作映射和V1/历史V2源码接口。结论区分已验证、接口定义及不可恢复的信息。

完整交付包与SHA256位于同仓库的 [V2.1 Preflight Release](https://github.com/goudan009/memory_vla/releases/tag/v2.1-preflight-2026-11-26)。当前目录保留可浏览的全部交付文件；压缩包不重复放入Git历史。

| 入口 | 内容 |
| --- | --- |
| [report.md](report.md) | 总报告、证据边界和未解决问题 |
| [task40_labels.parquet](task40_labels.parquet)、[label_statistics.csv](label_statistics.csv) | 逐帧标签、valid掩码与统计 |
| [state_inventory.json](state_inventory.json) | 87项状态接口，含55个对象状态/基础设施类 |
| [robot_description/](robot_description/)、[joint_topology.json](joint_topology.json) | 实际USD派生模型、原URDF及43关节/44连杆拓扑 |
| [state_action_mapping.json](state_action_mapping.json) | 完整82状态/23动作及GR00T列索引 |
| [fk_validation.json](fk_validation.json) | 300次双末端FK对照及误差 |
| [model_interfaces.md](model_interfaces.md) | Eagle/EOS、动作专家、历史V2和顺序窗口接口 |
| [samples/](samples/) | 200条解码记录、原图、源码副本和工作台证据 |
| [scripts/README.md](scripts/README.md) | 依赖、复现顺序、校验与打包命令 |
| [MANIFEST.json](MANIFEST.json) | 包内文件大小和SHA256 |

`door_open`、当前`started`和熟爆米花粒子存在有可认证记录；`bag_grasped`、`bag_inside`全部保留未知，原因见报告。没有修改模型或启动训练。
