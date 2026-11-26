# V2.1 开发前数据与机器人调查报告

最新Task40标签扩展见 [label_extension/report.md](label_extension/report.md)：新增几何/抓取弱标签、历史事件及统一掩码。本报告以下保留v001调查结论；根目录标签为原认证基线，最新统一标签位于`label_extension/task40_labels.parquet`。

调查日期：2026-11-26。范围：用户提供的六项任务；仅数据、资产和接口验证，无训练、无模型修改。设计文件为 `docs/v2/2.1.html`、`2.1.md`、`2.1-protocol.md`、`notation.md`，哈希见 [version_manifest.json](samples/version_manifest.json)。

已取得200条、647,555帧示教的可认证记录状态，完整82维本体与23维动作映射、真实USD运动链，以及300次末端FK对照。`door_open`、`started`、熟爆米花粒子存在可导出；`bag_grasped`、`bag_inside` 全部保留未知。后两项没有满足认证条件，不能据此宣称四个目标标签全部就绪。

## 1. 验收与证据边界

| 调查项 | 结果 | 证据与限制 |
| --- | --- | --- |
| 仿真状态全面清单 | 部分完成 | [state_inventory.json](state_inventory.json) 87项，包含55个对象状态类/基础设施类；列出所有发现接口，未对所有对象、传感器、其他任务逐一动态测试 |
| 200条状态标签 | 部分完成 | [task40_labels.parquet](task40_labels.parquet) 完整帧覆盖；2个所需状态已认证，2个未知；额外粒子存在、温度、门角和粒子数 |
| 机器人实际拓扑 | 完成 | [joint_topology.json](joint_topology.json)：43关节、44刚体连杆、28非固定自由度；USD为准，保留URDF差异 |
| 本体与动作映射 | 完成 | [state_action_mapping.json](state_action_mapping.json)：逐维82、逐通道23、物理轴/限制/填充列/归一化与统计 |
| 拓扑先验与FK | 完成（限定验证范围） | 23节点25条候选边；300次双末端对照；中间连杆可计算但没有独立逐连杆真值对照 |
| V1与历史V2接口 | 完成 | 历史V2确实找到；[model_interfaces.md](model_interfaces.md) 区分实测张量与源码推导形状 |

证据分级：全量原始记录验证、原始初始快照验证、初始化接口存在、仅源码定义。JSON逐项记录 `verification`、`verification_scope`、`actual_examples` 和范围；未验证项的值、范围或形状可以为null，不能当作已实测字段。55项中含抽象基类与Mixin，不是55种可直接调用的物理谓词。本次没有穷尽Isaac全部底层API。

### 数据和版本

| 来源 | 实际路径/版本 |
| --- | --- |
| 当前V1源码 | 仓库初始提交 `0d28bc506c3def15b2f76bdbdc0a3ccb1b4567a8` |
| 历史V2 | `/nvme3/projects/memory_vla-v2-v3-backup-oEWaKh/src/v2_groot_memoryvla/`；逐文件匹配 `b710fc4df586216b4b6a1a3776f09c725300cd98` |
| 转换后的Task40 | `assets/datasets/b1k2025_groot_task40_v1_200/`，82状态/23动作/三路256×256 RGB |
| 原始256维与原始RGB | `assets/datasets/b1k2025/`（实际为共享数据引用） |
| 官方物理状态HDF5 | `/nvme3/datasets/b1k2025_raw_task40_v2_audit/task-0040/` |
| 官方原始数据revision | `b3c9a714b8730701c480e77173d8aa4a8f19e1c0` |
| OmniGibson源码 | `/nvme3/projects/GR00T/BEHAVIOR-1K-task-progress/OmniGibson/omnigibson/`，版本及相关文件哈希随包保存 |

200个episode ID不连续。每个HDF5可能包含多个重试，按完整动作矩阵与转换后parquet逐值相等唯一匹配 `demo_N`，不假定 `demo_0`。逐文件哈希、选择组、尝试清单和全量范围见 [dataset_audit.json](samples/dataset_audit.json)。原始256→82切片和23维动作在200条上全部精确相等。

## 2. 仿真状态支持范围

统一用 `env.task.object_scope[BDDL实例].unwrapped` 绑定对象，再调用 `obj.states[State].get_value(...)`。Task40实例绑定：`microwave.n.02_1→microwave_hjjxmi_0`，`popcorn__bag.n.01_1→popcorn_bag_73`。不要将显示名称或数组顺序硬编码为通用身份。

| 类别 | 可获取信息与真实接口 | 官方数据情况/本次验证 |
| --- | --- | --- |
| 本体 | `get_joint_positions/velocities/efforts()`；28物理/虚拟DOF，按关节为rad、m及相应导数/力矩 | raw256保存q、sin、cos、qvel、effort，200条验证；转换82删除了大部分速度/力矩 |
| 底盘/末端/连杆 | `get_position_orientation()`、`get_linear_velocity()`、`get_angular_velocity()`；世界根状态、base_link参考末端 | 世界根与双EEF在原始观测中；其他连杆可FK；实时连杆速度未逐帧提取 |
| 物体 | 根位置[3]、xyzw四元数[4]、线/角速度[3]、可动关节q/qvel | HDF序列化保存；本次全量提取袋/微波炉pose、门q与温度；其他物体为初始快照/接口证据 |
| 相对关系 | Inside、OnTop、Under、NextTo、Touching、Contains、Covered、Filled、AttachedTo、Overlaid、Draped等 | 多数为根据碰撞、AABB、体积和粒子实时推导；未直接保存逐帧布尔值 |
| 接触/抓取 | ContactBodies、ContactParticles、IsGrasping、`robot.is_grasping()`、RigidContactAPI | 接触缓存与辅助抓取约束不可由q和动作可靠恢复；本次没有认证逐帧接触/抓取 |
| 温度/材质/粒子 | Temperature、MaxTemperature、Cooked、Burnt、Heated、Frozen、OnFire、Saturated、ModifiedParticles等 | 序列化的状态与派生谓词须分开；全量有袋/微波炉温度及两种爆米花粒子数 |
| 设备 | Open、ToggledOn、HeatSourceOrSink；ToggledOn含当前开关值及按钮接触累计步数 | 门q及当前开关值直接认证；“开始过”“正在加热”“任务成功”均不能等同当前开关 |
| 任务 | `goal_status`、`ground_goal_state_options`、`env.task.success`、BDDL谓词 | 有通用接口；原始帧未直接保存全部predicate真值，本次未重建成功率 |
| 评估 | `TaskMetric`的`q_score.final`、`info['task_progress']` | 源码核查；不是现成示教帧标签 |
| 视觉 | RGB、depth、depth_linear、normal、seg_semantic、seg_instance、seg_instance_id、flow、bbox_2d_tight/loose、bbox_3d、camera_params、pointcloud | 当前示教保存3路RGB；其他模态仅接口定义，不能把重新渲染结果认定为原始观测 |

其余基础设施类、状态依赖、实现片段、函数签名及对象是否初始化均在清单中。所有源码副本位于 `samples/source_evidence/omnigibson/`。几何状态的坐标定义以实现为准：物体序列化pose是scene参考；当前200条记录scene变换为单位变换，所以所提取物体pose数值与world相同。四元数均为xyzw。

### 任务判定、Q Score与Task Progress

原始 [Task40 BDDL](samples/source_evidence/task40_problem0.bddl) 的目标是 `Real(cooked__popcorn)` AND `Contains(popcorn__bag,cooked__popcorn)`。门、抓取、袋在微波炉、开机是过程状态，不能代替最终目标。

`metrics/task_metric.py:18` 保存初始各grounding谓词，终态成功返回1，否则对每个grounding计算“初始假、终态真”的谓词数/该组总数，取最大值。它需要初始与终态；不是四个人工阶段的完成百分比。

`learning/utils/task_progress_utils.py:826` 对Task40返回8个近似检查：近微波炉、近台面、门打开、袋不再OnTop台面、袋Inside微波炉、门关闭、设备开启、熟爆米花对象存在。`popcorn_bag_picked_up` 实现是“非OnTop台面”，不等于机器人抓住袋；opened/closed是当前关系的互补，不能当作单调历史阶段。`near`是机器人根及末端到对象连杆的平面距离阈值0.5m。

同一源码字典枚举50个任务的progress处理器，`check_progress()`、`object_scope`、通用对象状态和BDDL目标接口可跨任务使用。其他任务本次未启动验证；应依据实例、能力和predicate参数访问，避免绑定Task40名称。

## 3. Task40逐帧标签认证

认证主线是直接读取记录，而非重放后猜测：初始化官方环境取得schema → `og.sim.deserialize`读原始序列化向量 → CPU按源码顺序解析registry → 比较官方输出 → 全量导出。CPU解析不做浮点模式搜索，每帧核对完整消费长度，对未知schema报错；可变粒子数和对象UUID均按真实语法解析。

本次官方反序列化参考完成37条、127,312帧；CPU结果与这些官方输出逐字段逐值相等，并完成200条全量解码。前13条的较慢作业和后37条的优化作业均主动停止，不能把工作台37/200状态写成200个仿真作业完成。详见 [cpu_decode_report.json](samples/cpu_decode_report.json) 与 [workbench归档](samples/workbench/job_summary.json)。`official_decode_report.json`仅是兼容文件名的CPU报告副本；较早的 `cpu_decoder_comparison.json` 是35条中间结果，以37条最终报告为准。

37条官方参考与CPU输出的比较发生在全量CPU导出之前。导出在同一目标路径写入了增加粒子数字段的NPZ，因此包内NPZ的文件哈希与原官方六字段NPZ不同；原官方文件没有单独保留。包内保留原工作台报告中的历史哈希、当时逐数组比较结果与全量CPU文件的新哈希，不把重建文件宣称为独立官方参考。

### 标签语义和统计

| 标签 | 正例 | 负例 | 未知 | 掩码前记录切换 |
| --- | ---: | ---: | ---: | ---: |
| door_open | 398,237 | 249,317 | 1 | 400 |
| started | 21,412 | 622,743 | 3,400 | 200 |
| cooked_popcorn_exists | 8,012 | 639,543 | 0 | 194 |
| bag_grasped | 0 | 0 | 647,555 | 未知 |
| bag_inside | 0 | 0 | 647,555 | 未知 |

[label_statistics.csv](label_statistics.csv) 同时提供200条逐episode统计及ALL_200汇总。`adjacent_valid_switches`只数相邻且都有效的帧；started在边界设未知后该计数为0，不代表无开关切换。未知标签的掩码前切换次数为null，不能用0替代。

`door_open`：官方Open的5%关节区间阈值；当前唯一门关节 `j_link_0`，正方向、`both_sides=false`，阈值0.07853977382183075rad。`q>threshold`为真；距阈值≤1e-5rad保留未知，共1帧。资产门限[0,1.5707954168]rad，实测有约1e-6rad浮点超限。源码 `object_states/open_state.py`，不是动作类型或时间位置推断。

`started`：`ToggledOn._get_value()`返回的当前布尔开关值，不是“曾经启动”。沿用已有已认证记录的±8帧边界未知策略，每次共17帧；底层值保留在 `started_recorded_value`，没有移动切换点。原始记录哈希匹配历史认证，且新CPU解码与其所有200条底层开关值相等。原认证中的180/20仅作为历史provenance保留，本调查不创建训练划分。

`cooked_popcorn_exists`：序列化 `cooked__popcorn` 系统当前粒子数>0，系统名有两个下划线。它不保证粒子在袋内，也不等同完整任务成功或已认证的BDDL Real+Contains。另保存原始popcorn粒子数。

`bag_grasped`：实时有 `robot.is_grasping(arm=...,candidate_obj=bag)` 和IsGrasping。`robots/manipulation_robot.py:1868,1879` 的序列化/反序列化仍TODO AG；初始JSON有辅助抓取信息，但逐帧float序列没有该约束，raw256的proprio也排除了抓取字段。夹爪闭合、动作指令、离开台面都不是抓取真值，因此全未知。

`bag_inside`：`Inside._get_value(other)`用袋AABB中心在外物体AABB内，并检查微波炉容器meta-link体积，不是仅比较两物体中心。原始袋资产预期MD5 `8f40ea1093f6c42f4a837128a11e80ee`，现加载 `c3c3fc108cc259127e8a5781cf04ef9b`；[历史日志摘录](samples/historical_asset_mismatch.json) 与 [回放对照](samples/label_alignment_400010.json) 保留证据。旧回放末端位置最大差0.00281525m，机械臂q最大差0.01071595rad；仅拿现资产重放不能认证原始几何关系。当前没有生成Inside伪标签。

### 时间、原图和执行顺序

控制/渲染30Hz，物理120Hz。HDF5状态T+1、动作T：`state[t]=observation/RGB[t]`是执行 `action[t]` 前的状态；`reward/terminated/truncated[t]`是该动作结果，额外 `state[T]` 不进入无对应RGB的标签表。`frame_index=t`、时间=t/30；parquet时间误差<1e-4s。600路视频metadata与长度/30Hz核对；72张原始分辨率关键帧PTS逐个核对=t/30。[video_alignment_metadata.json](samples/video_alignment_metadata.json) 提供相机、源路径、尺寸、PTS和时间。

官方 `DataCollectionWrapper.enable_dump_filters` 省略未变化/休眠对象，缺席不是false或未知：从各episode原始初始状态初始化，后续保留最近显式记录。标签表保留 `*_explicitly_recorded` 标记。此规则不授权用固定物体的旧速度替代真实动态接触信息。

抽查 [400010面板](samples/certified_transition_panel_400010.png)、[400020面板](samples/certified_transition_panel_400020.png)：门开闭与q一致，按钮红→绿与ToggledOn一致，随后腔体内可见粒子。原图前后取切换±16帧；这支持观测一致性，不是亚帧渲染延迟认证。第三组 [403000](samples/certified_transition_panel_403000.png) 也提供原图以供复核。旧 `label_alignment_400010.png` 含临时Inside标注，其JSON明确 `labels_certified=false`，不可混入当前认证集。

200条门均恰好2次原始切换（开→关），开机均1次，无再次开门或开机后关闭；熟粒子194条发生出现切换。最后一个有RGB的观测帧仍未出现熟粒子的ID为400100、400220、400540、401000、401390、401530；因未计无RGB的terminal额外状态，不能据此判这些示教失败。抓取、放入、重复拿放的执行顺序仍未知。完整事件时间线见 [transition_analysis.json](samples/transition_analysis.json)。

真实样例：episode400010门q在frame959≈0、991=0.3149、3317=0.1888、3349=0.0186rad；3719开关false，3751 true；3795熟粒子0，3827为256。袋温度全量23–59.49484°C、微波炉温度23°C，熟粒子0–269、原始粒子5–280。物体pose七维样例与逐维范围见 [decoded_ranges.json](samples/decoded_ranges.json)，原始256/82/23样例见 [raw_state_action_samples.json](samples/raw_state_action_samples.json)。

## 4. 实际机器人运动链

使用B1K加载的R1Pro实际USD：`assets/simulation/omnigibson-robot-assets/models/r1pro/usd/r1pro.usda`，真实路径及SHA256见 [asset_provenance.json](robot_description/asset_provenance.json)。原文件131,100,140字节，米制、Z-up。包内提供原始3份URDF、从真实USD复制关节属性生成的无mesh `r1pro_kinematics.usda`、[完整拓扑SVG](robot_description/topology.svg)、DOT和Mermaid。轻量USD供运动学使用，不是完整碰撞/渲染资产；原始完整USD按绝对路径与哈希引用。

结构：虚拟x→y→z→rx→ry→rz串联接 `base_link`；base_link接4关节躯干链，`torso_link4`经固定左右肩基座接左右7关节臂，再固定接夹爪，各有2个滑动手指。四个躯干均revolute，前三绕局部Y、第四Z；不是单一垂直滑台。各关节local0/local1的位置和xyzw姿态、父子链接、轴、角度/速度范围、drive最大力均逐项保存。

物理资产43关节中15固定、28非固定。底盘6个虚拟自由度仅x/y/rz受23通道控制，z/rx/ry是未控制的虚拟坐标；本体躯干4+臂14+手指4。运行时还有world→虚拟根的固定rootJoint，不混进资产43计数。USD把3个轮转动与3个转向关节固定，3份URDF仍声明可转动，差异列表已保存；FK和拓扑以实际USD为准。

双手各2个独立prismatic关节，单命令软件等量驱动，不存在本报告假设的URDF mimic。两指局部frame含180°旋转，所以相同局部轴/正q在物理空间朝相反方向运动。所有限位的USD角度以degree存储，JSON转换为rad；未写的速度限制是null，不是0，运行时可能用 `JointPrim.max_velocity` 默认值，应与控制器范围分开。

## 5. 本体82维、动作23维与GR00T

以下均0起始、半开切片；单维含义、样例、200条实测范围和每个动作的统计参数在 [state_action_mapping.json](state_action_mapping.json)。

| packed82 | 原始256 | 内容与单位/参考系 |
| --- | --- | --- |
| 0:3 | 140:143 | 移动base_link世界位置m |
| 3:6 / 6:9 | 143:146 / 146:149 | Euler XYZ的cos/sin，无量纲 |
| 9 / 10 / 11 | 149 / 150 / 151 | yaw rad / cos / sin |
| 12:15 / 15:18 | 152:155 / 155:158 | 世界线速度m/s、角速度rad/s |
| 18:25 / 25:32 / 32:39 | 158:165 / 165:172 / 172:179 | 左臂q rad / sin / cos |
| 39:42 / 42:46 | 186:189 / 189:193 | 左EEF位置m、xyzw四元数，相对base_link |
| 46:48 | 193:195 | 左两指q m |
| 48:55 / 55:62 / 62:69 | 197:204 / 204:211 / 211:218 | 右臂q rad / sin / cos |
| 69:72 / 72:76 | 225:228 / 228:232 | 右EEF位置m、xyzw四元数，相对base_link |
| 76:78 | 232:234 | 右两指q m |
| 78:82 | 236:240 | 躯干4关节q rad |

原始256前140维为28个q、28sin、28cos、28qvel、28effort。q顺序：0..5=x,y,z,rx,ry,rz；6..9躯干；10..23按L1,R1,L2,R2,…L7,R7交错；24,25左指，26,27右指。不能用action索引直接索引q，也不能把旧258维布局用于当前256维记录。

| 动作索引=GR00T有效列 | 对应自由度 | 直接对应82 q索引 | 仿真控制 |
| --- | --- | --- | --- |
| 0,1,2 | 底盘x,y,rz；q索引0,1,5 | 无直接虚拟q；相关世界pose0:12、velocity12:18 | 局部机体系速度，输入±1分别对应±0.75m/s、±0.75m/s、±1rad/s，再旋转到规范虚拟关节系 |
| 3:7 | torso_joint1..4；q6..9 | 78:82 | 绝对关节位置rad |
| 7:14 | left_arm_joint1..7；q10,12,…22 | 18:25 | 绝对关节位置rad |
| 14 | 左两指；q24,25 | 46,47 | 输入a∈[-1,1]，两指q=(a+1)×0.025m |
| 15:22 | right_arm_joint1..7；q11,13,…23 | 48:55 | 绝对关节位置rad |
| 22 | 右两指；q26,27 | 76,77 | 同左手 |

以HDF内实际controller_config为准，双臂是JointController绝对位置、`use_delta_commands=false`；不能套用当前默认yaml的IK配置。原始指令、processor表示和flow速度场是三层不同量。

GR00T对躯干/双臂先算所有未来 `a[t+h]-q[t]`（同一个当前观测作为基准），不是相邻动作差；底盘/夹爪仍绝对输入。之后使用checkpoint `statistics.json` 的min/max，非q01/q99：`clip(2*(x-min)/(max-min)-1,-1,1)`；`isclose(min,max)`时输出0。逆变换先clip、线性还原，相对组再加当前q。相对统计是32×部件维度的逐horizon数组，不能只取第一行。

状态填充 `[B,1,128]`，82:128为0；动作 `[B,50,128]`，通常有效区域 `[:,0:32,0:23]`，episode尾还会进一步mask；时间32:50和列23:128无监督。配置、统计和embodiment ID副本位于 `samples/processor/`；本体ID24。全部动作列的物理连接通过真实USD父链计算，并提供可FK关联连杆。

## 6. 两种物理先验的可用性

方案A提供候选图，未决定网络如何使用。23节点的0/1/2代表同一移动底盘的平面速度组，三者同体相连；均接躯干第1节点3；躯干3→4→5→6，分叉到左臂7和右臂15；左臂7→…→13→夹爪14，右臂15→…→21→夹爪22。中间固定/未控虚拟关节按真实父链收缩，两指收为一命令节点，共25条无向候选边，依据逐边保存。底盘三角边是同刚体分组选择，并非三个物理连杆串接的断言。

方案B：按USD `T_child=T_parent·T_local0·Motion(q)·inv(T_local1)` 计算。使用原始28个SI关节值可计算完整含虚拟根的树；使用82中的躯干4、左右臂14、手指4可计算所有base_link后代。结合82中的世界base_link pose可转到世界。EEF观测参考系是移动base_link，不是虚拟articulation根，源码 `utils/usd_utils.py:941` 的替换逻辑及R1的base_footprint_link_name已核对。

调用见 `scripts/robot_assets.py:fk(joints,q)`；构造 `{joint_name:q_SI}` 后得到各link的4×4矩阵，若使用默认虚拟根则 `inv(T['base_link']) @ T[link]` 得到base_link参考。已有 [fk_samples.json](samples/fk_samples.json) 给出6条示教首尾共12组所有连杆数值。例episode400010 frame0，左末端位置[0.1607606051,0.2519998780,0.6570998608]m、quat=[0.0000014148,0.0224751668,0.0000037068,0.9997474015]，参考base_link。

[fk_validation.json](fk_validation.json) 在6条示教、各25帧、双EEF共300次对照原始录制PhysX连杆pose派生的EEF观测：位置最大1.937292633e-6m、均值6.680870175e-7m；姿态角最大1.309196541e-6rad、均值5.580480627e-7rad。它是与原始仿真观测的独立数值对照，不是把自算FK互相比较。未重新做物理轨迹回放，也没有所有中间link的独立测量误差。FK末端位置/姿态与现有82的14维EEF信息重复；新增可得的是中间连杆，最终特征选择留给V2.1开发。

## 7. 收尾、复现与未解决项

本次运行遵循 `robot-vla-experiment-ops` 的工作台preview/start与证据归档流程；只读反序列化作业ID `20261126-200257-15a1de03`、`20261126-201130-d9b4bb46`，均已停止且原进程退出。最终工作台快照见 `samples/workbench/final_services.json`，已有4×A100的 `v1__test01__exp13__run001` 进程仍在；没有暂停、重派或修改其配置。后续全部处理为CPU。

复现命令、依赖和输出顺序见 [scripts/README.md](scripts/README.md)。包内包含轻量运动学资产、源码副本、200条解码NPZ、原始关键帧及真实缓存特征样例，不复制完整HDF/RGB数据集、完整模型权重和131MB渲染USD。外部资源都有路径/哈希或版本依据；包内 `MANIFEST.json` 覆盖交付文件。

未解决项有明确范围：逐帧抓取约束不可从当前记录恢复；Inside需匹配原始袋碰撞/容器资产并验证几何恢复；Q score/Contains/成功条件没有逐帧认证；非RGB传感器和多数物体状态仅源码/初始化证据；FK中间link缺少独立逐连杆对照。包内相应项保留null或接口级证据，不生成伪值。这些缺口不影响已认证字段、82/23映射和运动链交付，但四个目标状态的完整监督尚未就绪。
