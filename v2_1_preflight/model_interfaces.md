# V1与本地历史V2模型接口调查

当前V1源提交 `0d28bc506c3def15b2f76bdbdc0a3ccb1b4567a8`；历史V2备份 `/nvme3/projects/memory_vla-v2-v3-backup-oEWaKh/src/v2_groot_memoryvla/` 与Git提交 `b710fc4df586216b4b6a1a3776f09c725300cd98` 逐文件相等。历史V2未丢失。所有选中源码及SHA256、Git blob比较结果见 [model_interface_evidence.json](samples/model_interface_evidence.json)，副本在 `samples/source_evidence/current_v1/`、`historical_v2/`。以下路径均相对各版本根；不需要读取不存在的推测接口。

## 1. Eagle视觉Token与当前用户EOS

`src/v1_groot_memoryvla/model/modules/eagle_backbone.py:112` 的 `EagleBackbone.forward(vl_input)` 返回 `BatchFeature`：`backbone_features`为LLM最后层序列，另有 `backbone_attention_mask`、`image_mask`。将已有标志 `return_visual_features=True` 打开，可取得并reshape为 `[B,cameras,grid,D]` 的 `visual_features`。

```python
result = self.model(**vl_input, output_hidden_states=True,
                    return_visual_features=self.return_visual_features, **visual_options)
outputs = result["hidden_states"][-1]
image_mask = vl_input["input_ids"] == self.model.config.image_token_index
features["visual_features"] = visual.reshape(b, cameras, grid, visual.shape[-1])
```

`model/modules/nvidia/Eagle-Block2A-2B-v2/modeling_eagle3_vl.py` 的 `extract_feature()`（约226–240）、`forward()`（约298、360–392）在视觉编码后、进入LLM前保留视觉网格。默认是经过 `mlp1` 投影的2048维视觉Token；`return_preprojected_vision=True` 是另一条可用路径，不能把预投影特征和2048维接口混称同一张量。本次真实缓存验证的是2048维投影版本。

当前认知Token提取实际在 `model/memory/adapter.py:120`：最后一个有效 `im_end`（ID151645），不是padding最后一位、系统EOS或最后换行。

```python
eos = (input_ids == self.eos_token_id) & attention
positions = torch.arange(h.shape[1], device=h.device).expand(b, -1)
eos_indices = positions.masked_fill(~eos, -1).max(dim=1).values
current_cog = h[i:i + 1, eos_indices[i]:eos_indices[i] + 1]
```

原有适配器读当前三相机网格，经perceptual_encoder形成 `[1,243,256]`，认知当前值为 `[1,1,2048]`，随后执行各自read/fuse/write。该接口可复用，但不代表已实现V2.1设计。

## 2. 真实张量与源码推导张量

本次只在CPU读取已有 `outputs/diagnostics/v1_real_replay_features.pt` 的40个真实样本，无新模型forward或训练。缓存SHA256=`4e670e5165f9b18dc4f945e6523780c6a97cc3ec23e06e4f9966ef4bf23bf87f`。首样本提取EOS与缓存原EOS逐值相等，EOS索引282；图像位置共243。首样本有效action_mask之和736=32×23。便携数值样例 [real_feature_sample.npz](samples/real_feature_sample.npz) 以float32导出，原始dtype保存在证据JSON。

| 张量 | 形状 | 证据 |
| --- | --- | --- |
| backbone_features | [1,284,2048] BF16 | 真实缓存；序列长度284不是通用常量 |
| attention/image_mask | [1,284] bool | 真实缓存 |
| visual_features | [1,3,81,2048] BF16 | 真实缓存，3相机×9×9网格 |
| 当前用户EOS | [1,2048]；adapter切片后[1,1,2048] | 缓存读取/源码切片 |
| processor state | [1,1,128] BF16 | 真实缓存；原始82补零 |
| processor action、action_mask | [1,50,128] BF16 | 真实缓存；正常有效32×23 |
| embodiment_id | [1] int64，B1K ID24 | 真实缓存/配置 |
| 最终action_pred | [1,50,128] | 已有真实缓存predictions |
| state_encoder输出 | [B,1,1536] | 源码+checkpoint config推导，本次无新中间激活采集 |
| DiT输入state+action | [B,51,1536] | 源码拼接与config推导 |
| DiT输出 | [B,51,1024] | config output_dim与源码推导 |
| action_decoder速度场 | [B,50,128] | 源码切片推导；不是缓存最终action_pred本身 |

缓存metadata保留 `task=0/episode=10`，这不构成Task40 episode身份认证；也不能证明该历史缓存的完整checkpoint/源码revision。它证明实际三相机Eagle与动作接口的运行形状。Task40的82/23、padding、物理字段来自本次全量Task40审计与checkpoint processor配置，两条证据应分开。

## 3. 原动作专家训练与推理速度场

`src/v1_groot_memoryvla/model/gr00t_n1d6/gr00t_n1d6.py:275`：

```python
noisy_trajectory = (1 - t) * noise + t * actions
velocity = actions - noise
action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)
sa_embs = torch.cat((state_features, action_features), dim=1)
# DiT uses full vl_embeds and masks as conditioning
pred = self.action_decoder(model_output, embodiment_id)
pred_actions = pred[:, -actions.shape[1]:]
action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
loss = action_loss.sum() / (action_mask.sum() + 1e-6)
```

训练 `pred_actions` 虽名为actions，实际上是flow matching速度场预测，目标是 `actions-noise`。它位于归一化动作空间，不是rad/s的真实关节速度。公共train outputs当前返回loss、逐元素action_loss、mask、backbone_features、state_features；没有单独公开速度场。若开发需取原始 `u0`，真实接入位置就是decoder后动作段切片，不能从最终action_pred反推。

推理同文件约455：

```python
pred = self.action_decoder(model_output, embodiment_id)
pred_velocity = pred[:, -self.action_horizon:]
# optional existing unconditional guidance branch intervenes here
actions = actions + dt * pred_velocity
```

默认config `num_inference_timesteps=4`、horizon50。推理返回 `action_pred` 为Euler积分后的轨迹；若需要每次flow时间的原专家速度，须在每步decoder切片处读取，并明确是否在可选guidance前后。本次只定位，没有加hook或修改模型。原专家条件包括完整VLM序列、本体和noisy action/time，不是仅EOS。

## 4. Episode顺序窗口和记忆重置

`data/dataset/episode_windows.py:9` 的 `EpisodeWindows` 默认 `group_size=32`、传给HistoryDataset的 `history_stride=8`；`get_group()` 输出episode、window、episode_start/end。`order(seed)`仅打乱episode，episode内部窗口连续消费。

```python
ids = np.array(sorted(self.by_episode))
np.random.default_rng(seed).shuffle(ids)
return [index for e in ids for index in self.by_episode[int(e)]]
```

`experiment/task21_formal.py:34` 的window_context产生 `episode_keys`、`timesteps`、`sequence_start/end`，与adapter chunk_mode配合。`model/memory/adapter.py` 的 `reset(keys=None)`、`cognitive/perceptual` bank reset可复用。序列开始reset当前key，序列末结束清理；跨episode chunk直接报错，继续chunk却没有start也报错。bank写入detach，记忆值与计算图截断须按现有实现理解，不能假定完整跨episode反传。

## 5. 历史V2标签、状态头和loss

历史V2的 `model/state.py` 确有：

```python
PHYSICAL_LABELS = ("door_open", "bag_inside", "started")
class StateHead(nn.Module):
    # norm = LayerNorm(dim); proj = Linear(dim,len(labels))
    def forward(self, current):
        return self.proj(self.norm(current))
```

其输入是检索前的当前用户EOS，不是动作或检索后状态。没有 `bag_grasped` 标签实现。`masked_state_loss(logits,targets,known)`校验相同shape/bool known；先按known取目标再算BCE，因此未知NaN不会污染loss；全未知返回 `logits.float().sum()*0`。

`task40_training.py:38` 的 `validate_label_manifest()`实际只接受已认证started清单；`window_labels(archive,frames)`按真实frame索引取 `labels/known`，而非按窗口位置或任务百分比。训练组合动作loss与加权state loss。历史180/20划分和历史训练选择不移植为本次V2.1决策。

原标签提取在当前仍存在的 `evaluation/scripts/v2_label_audit.py`：可在DataPlaybackWrapper `_process_obs`读Open/Inside/ToggledOn，同时收集proprio；`--deserialize-only`路径使用官方deserialize读记录ToggledOn，按完整action匹配重试组，保留休眠对象前值，±8帧置未知。`evaluation/scripts/v2_certify_started.py`校验200条manifest、哈希、图像与既有回放结果后认证started子集。本包保存这两份真实源码，其提交和blob一致性在证据清单。新提取实现 `scripts/sim_extract.py`、`offline_decode.py` 增补门q、pose、温度、粒子数；标签有效性语义见 `samples/label_schema.json`。

历史 `model/adapter.py` 包含state query、Q/gate、事件阈值滞回以及运行时bank/event的state_dict/load；`from_v1`有复制继承逻辑，但它继承的是既有V1残差适配路径，不能视作V2.1逐动作通道速度修正实现。历史源码用于接口参考，没有启动或复用历史训练权重。

## 6. 开发接入所需区分

视觉网格来自Eagle投影特征，当前认知来自最后有效用户EOS；本体来自processor state，动作专家速度来自每步decoder动作切片。87项状态清单不意味着87个监督标签：当前可认证标签及known mask以parquet为准。23个动作通道、28个仿真DOF、82维观测和128维填充空间必须使用映射文件，不能互相替代。

以上提供真实接入位置，不决定V2.1最终特征、图连接学习方式、loss配方或网络结构。
