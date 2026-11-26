# Task40 标签扩展 v002

最新统一标签为本目录的 [task40_labels.parquet](task40_labels.parquet)，覆盖200条、647,555个原始观测帧、21个布尔状态。[报告](report.md)说明资产恢复、弱标签规则、图像抽查和认证缺口；[统计](label_statistics.csv)逐episode及全量汇总正/负/未知与证据等级。

每个状态都有 nullable 值、`_quality`、`_valid`、`_certified_valid`、`_weak_valid`。`quality=0/1/2`分别表示未知/认证/弱标签。`_valid`包含弱标签，严格监督应按`_certified_valid`筛选；不要把null填成false后丢掉掩码。另提供可单独读取的 [task40_masks.parquet](task40_masks.parquet) 和完整 [label_schema.json](label_schema.json)。

`bag_inside`和`bag_grasped`均未升级为认证真值。前者用找回的官方历史资产重算几何，后者用双末端轨迹、物体运动、夹爪开度及原始三视角抽查补充。明确缺少原始资产身份和逐帧AG抓取约束；无法支持的区段保持未知。

`door_ever_opened`等历史状态在每个episode重置，只聚合当前及过去的有效证据；不会用未来事件填充前面的未知帧。原先已发布的v001标签在上级目录保持不变，本目录为新增版本。

复现接口为 [label_api.py](../scripts/label_api.py)，Task40绑定和阈值在 [task40_config.json](task40_config.json)。其他任务复用三值逻辑、历史聚合、Inside几何和抓取候选接口，替换对象角色与观测适配器；不要假定Task40对象名、关节或坐标系通用于所有任务。

全量交付下载：[Task40 标签扩展 v002 Release](https://github.com/goudan009/memory_vla/releases/tag/v2.1-preflight-task40-labels-v002)。该包包含原调查及本次扩展；旧Release独立保留。
