# Task40 资产恢复与状态标签扩展

本次按用户授权补充数据，不修改模型、不训练、不启动仿真或模型推理。计划为 [plan-v002.json](../workflow/plan-v002.json)，已先提交`286f9be`。旧调查及其200条认证记录保持原样；最新统一标签位于本目录。

已完成200条、647,555帧的21状态导出、显式证据等级与三种有效性掩码、因果历史事件、逐episode统计和可复用接口。`bag_inside`与`bag_grasped`的严格认证仍部分未完成，原因分别是未找到与原记录身份一致的物体资产、逐帧AG约束未序列化。本次补充的是明确分级的弱标签，没有将其包装成仿真真值。

## 原始资产恢复

从官方`behavior-1k/zipped-datasets`的历史revision `015e37a26823a110fedb6a0c3d943f573de00ffe`，通过ZIP字节范围取回`popcorn_bag/hdcpqg`及`microwave/hjjxmi`。官方源为 [历史资产包](https://huggingface.co/datasets/behavior-1k/zipped-datasets/blob/015e37a26823a110fedb6a0c3d943f573de00ffe/behavior-1k-assets.zip)，实际使用hf-mirror传输，ZIP成员CRC校验和成员摘要已记录。仅取相关成员，没有下载整个33GB包；未声称验证整包SHA256。[恢复记录](asset_recovery_015e37a2.json)保留URL、revision、大小和摘要。

历史包VERSION为3.7.0rc23，采用加密USDZ；本地资产为3.7.2rc1，采用加密USD。200条示教均记录同一袋子和微波炉模型、同一scale及同一预期哈希；原始场景元数据标注og_dataset 1.2.0rc21。找回的版本不等于已确认的原始版本。

| 对象 | 示教预期加密USD MD5 | 本地加密USD MD5 | 找回的加密USDZ MD5 |
| --- | --- | --- | --- |
| 袋子hdcpqg | 8f40ea1093f6c42f4a837128a11e80ee | c3c3fc108cc259127e8a5781cf04ef9b | 9569364eb30989dbf57079d5aaf47543 |
| 微波炉hjjxmi | d4f14541ccd2e3716e66022ef7a71ae6 | 5c2589635156d13380562f613e84785f | 1c436aa95229f915ece875507858027e |

`USDObject._prepare_to_load/check_hash`比较的是加密文件。包装方式和加密随机性使哈希不匹配不能直接证明几何不同；同样，当前两个版本几何相同也不能证明缺失的原始版本相同。本次对历史/本地两版全部网格变换后顶点、面索引及joint physics属性做规范化比较，两个模型均完全一致，各物理连杆碰撞凸包最大坐标差为0米。详见 [geometry_provenance.json](geometry_provenance.json)。

检查了本地实际资产及其他本地缓存、官方历史包和后续版本历史；尚未找到匹配示教预期身份的原资产。恢复资产和解密文件隔离保存在`/nvme3/datasets/task40_asset_recovery/`，没有替换共享资产。公开目录仅保留来源、摘要和少量派生碰撞/容器凸包；不包含密钥或解密美术资产。

## bag_inside

严格遵循`Inside._get_value`的语义：袋子世界碰撞AABB中心通过微波炉AABB门限，并落在fillable/openfillable容器体积内。不是“袋根坐标接近微波炉”，也不是整个袋子完全被容纳。

袋子固定label连杆也纳入碰撞AABB。对其全部碰撞点合并求凸包，按原始scale及逐帧根pose变换后求世界AABB中心。微波炉容器为Mesh，沿用源码Delaunay体积判定。容器凸体完全位于微波炉base-link碰撞凸包内，最大平面违反量为−0.0098183973米，因此容器内的点必然通过外AABB测试；该证明对共同刚性变换/非均匀缩放成立，不依赖近似门FK。

有符号容器边界距离保存在`inside_geometry_margin_m`，正值为内部，绝对值≤2毫米保留未知。这是边界容差，不是原始资产误差的估计；所有可用Inside候选仍是弱标签。全量弱正182,822、弱负463,533、未知1,200，严格认证0帧。

直接从本地实际源码编译并调用Inside/GeomPrim/RigidPrim方法体，以恢复几何和记录pose提供适配对象，检查200条各初帧、末帧及最接近边界帧，共600次，数值布尔结果600/600一致，其中有效候选424次。[inside_source_validation.json](inside_source_validation.json)保留逐次结果。这验证CPU重构与源码数值语义一致；没有加载原始运行时，不替代资产身份、PhysX或回放图像认证。

## bag_grasped

原始序列化没有AG抓取约束，不能从关闭夹爪或离开台面直接生成真值。此次分别计算左右EEF的世界pose：原始82维moving base_link世界pose × 该base参考系下的EEF pose；袋pose来自已核对解码。四元数均xyzw、距离米、角度弧度。

弱正证据需要过去10帧中至少一指小于0.04米，袋与EEF分别移动超过0.012米、距离小于0.26米、相对位置最大偏差小于0.012米、相对姿态偏差小于0.16弧度。有共同运动证据后，可在相对变换仍稳定且夹爪满足条件时保持弱正：位置偏差门限0.025米、姿态0.30弧度。弱负依据为EEF距袋根超过0.30米，或两指均大于0.047米。其余未知。所有阈值均为弱推断工程规则，未估计全量准确率，未用任务时间百分比或动作类型。

抽查400450时发现初始“两指都小于0.04米”的规则漏检：frame2069两指约[0.04205,0.0000014]米，但袋子/EEF约0.138/0.139米共同位移，相对误差约0.0001米，原图显示搬运。物理手指可因接触停在不同位置，所以最终规则改成至少一指闭合，保留共同运动条件；frame2609两指约0.04975米、袋子静止而EEF移动约0.042米，仍为弱负。这是对实际接触样例的规则修正，没有人工回填整条示教。

`bag_grasped_left/right`及其三值OR均输出，未知区段不补成false。最终任一手抓取弱正125,435、弱负515,088、未知7,032，200条均有弱正区段，各有一次进入/退出弱抓取证据，切换间可能有未知帧。统计以 [label_statistics.csv](label_statistics.csv) 中最终`ALL_200`行为准；详见 [summary.json](summary.json)。`grasp_*_reason`记录共同运动证据、保持、距离负例、开爪负例或证据不足，便于后续调整阈值后重新提取。

## 原图、时间与回退

全部帧沿用已核验的episode/原始demo绑定：观测/RGB/state[t]在action[t]之前，30Hz，额外terminal state[T]无对应RGB而不纳入标签。保留原先started ±8帧离线边界mask；不移动状态切换点。历史值只使用截至当前帧的输入标签，但该既有离线边界mask自身不是在线传感器信号。原始reward/terminated/truncated[t]是动作后结果，不能作为观测t前的当前物理状态特征。

抽查400010、400020、403000的普通搬运，400450的规则漏检案例和400640的Inside边界回退，共15路原始视频；原始分辨率帧的最终数量以验证记录为准。PTS与t/30逐张一致。[validation.json](validation.json)列出源视频、相机、PTS及文件；[samples/](samples/)保存原图和三视角面板。检查了三视角袋子随手抬起/搬运、进入腔体及松手附近的图像。左腕多数遮挡袋子，不能把三相机数量当作三份独立可见证据。

400010例：frame2027抓取未知；2067为弱正且相对位置误差0.000491米；2441继续搬运、Inside边界距离−0.297677米；2640为−0.029806米；2680为+0.024848米；2817仍在手中且Inside为正；2857手已离开袋口，弱抓取负、Inside边界距离+0.061045米。原图支持这些过程关系，但不保证精确到一帧的辅助抓取切换。

Inside候选的“相邻有效观测间最少变化”分布为169条1次、27条3次、4条5次，边界附近进入/退出原样保留；并非全部为重复拿放，微小回退/几何中心跨界也会触发。抓取推断的切换可能跨未知区间，`events`记录最后反值帧和首个新值帧，不能将未知段两端直接当作精确切换时刻。最终完整时间线见 [episode_evidence.json](episode_evidence.json)。

## 历史事件与统一接口

21个状态包含当前门、开机、熟/生粒子、左右/任一抓取、Inside，以及`door_ever_opened`、`bag_ever_inside`、`bag_ever_grasped`、`microwave_ever_started`、`cooked_popcorn_ever_observed`、开门后关闭及其历史、Inside且关门、袋在内部且释放候选、开机且门开/关、Inside且开机。完整语义和每列见 [label_schema.json](label_schema.json)。温度、门角、粒子数和原始显式记录标记同时保留。

历史事件只对episode前缀做OR：此前任一认证真值可认证“曾经为真”；此前存在未知且没有正证据则不能认证“从未为真”；仅弱正证据不能升级认证；每条示教重置。`door_ever_opened`全量认证正490,469/负157,086/未知0。开门后关闭是事件组合，不是把当前closed当作完整历史。

`_valid`表示有可用标签，包括弱标签；`_certified_valid`仅认证，`_weak_valid`仅弱标签，未知值为null且三掩码均false。三值复合状态可能有认证负例，例如开关已认证为off，即使Inside未知，`Inside AND on`仍可认证false；这不代表Inside本身获得认证。严格认证与弱标签的统计分别给出，禁止只看valid覆盖率就宣称真值覆盖率。

`label_api.py`公开`Label`、`negate/conjunction/disjunction`、`ever`、`inside_candidate`和`grasp_candidate`。Task40对象角色、模型、EEF/finger切片和阈值由JSON配置；其他任务须提供自己的对象绑定、真实几何和记录坐标适配器。Inside接口要求调用者验证外AABB门限或证明其冗余。没有将Task40的门、袋名称写入通用算法。

未知/历史传播、逻辑短路、弱标签不升级、全200条主键/时间/掩码/统计、认证原标签不变均已检查。已核对的认证字段保持原值；没有推断完整BDDL Real+Contains任务成功、真实支撑/接触或热源实际运行状态。下一步若取得匹配原始资产或可验证的运行时抓取数据，可替换几何/抓取适配器重新计算，保持相同标签协议。
