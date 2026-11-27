# 训练自动汇总

由 `scripts/training_archive.py` 按固定规则生成，无 LLM 参与。原始数据及 SHA-256 清单见各事件链接。

首次安装只补录指定历史 run；随后自动记录工作台新训练及已有训练的完成、暂停、失败、取消事件。
这里是最近一次归档终态；训练恢复后，等下一个终态再更新。闭环结果在结束后补入。

Loss 为训练记录中的 loss；V2.1 为 fm + 配方 state_weight × state。首尾均值各取最多100条记录，保留重复 step 并报告数量；缺失/非有限值不当作零。epoch 均值不代表完整 epoch。

| 设备 / run / arm | 归档状态 | optimizer step / 当前停止目标 | 首100 Loss | 末100 Loss | 原始记录数 |
| --- | --- | --- | --- | --- | --- |
| [a100/v1__test01__exp13__run001/memory_v1](runs/a100/v1__test01__exp13__run001/memory_v1/events/204b3706c90523df31cb-958f37778991/workbench.json) | completed | 16000 / 16000 | 0.018298337 | 0.010263368 | 16000 |

## v1__test01__exp13__run001 / memory_v1 / a100

证据：[清单](runs/a100/v1__test01__exp13__run001/memory_v1/events/204b3706c90523df31cb-958f37778991/MANIFEST.json)、[实际配方与统计](runs/a100/v1__test01__exp13__run001/memory_v1/events/204b3706c90523df31cb-958f37778991/workbench.json)、[最近终态](runs/a100/v1__test01__exp13__run001/memory_v1/latest.json)。

原始输出：`/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v1__test01__exp13__run001`；源码：`/nvme3/projects/memory_vla/deployments/v1__test01__exp13__run001__resume002`；源码 digest：`68b6d82b8ed2608046b86ab39c58442d22f122f5e46fc4e62c2dafc05324fc85`。

身份字段（直接沿用工作台，不补猜历史编号）：{"model_version": "V_1", "test_id": "test01", "experiment_id": "exp13", "training_id": "v1__test01__exp13__run001", "run": "v1__test01__exp13__run001"}

初始权重 / 恢复：{"checkpoint": "/nvme3/huggingface/train_model/GR00T/GR00T-N1.6-3B", "label": "官方 GR00T N1.6-3B · 通用预训练", "mode": "resume", "optimizer_restored": true, "resume_checkpoint": "/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v1__test01__exp13__run001/memory_v1/checkpoints/v1__test01__exp13__run001__step011000", "resume_record": "/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v1__test01__exp13__run001/config/web_resume_latest.json", "resume_step": 11000, "source_file": "/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v1__test01__exp13__run001/memory_v1/config/ddp.json"}

原始指标：已归档；step 范围 1–16000；重复 step 0；非递增转换 0；范围内缺步 0；不完整尾行 0。

预算记录：当前停止目标 16000；原始配置 steps 20000。终态 completed 只描述本次工作台目标。

### 实际记录配方

字段来源：["/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v1__test01__exp13__run001/config/web_launch.json", "/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v1__test01__exp13__run001/config/ddp_launch.json", "/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v1__test01__exp13__run001/memory_v1/config/ddp.json", "/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v1__test01__exp13__run001/config/web_resume_latest.json"]

| 参数 | 值 |
| --- | --- |
| checkpoint | /nvme3/huggingface/train_model/GR00T/GR00T-N1.6-3B |
| checkpoint_interval | 1000 |
| dataloader_workers | 4 |
| dataset | /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v1__test01__exp13__run001/dataset |
| disable_early_stopping | True |
| fine_tuning_mode | full |
| freeze_action_dit | False |
| freeze_action_projector | False |
| freeze_action_vlln | False |
| freeze_llm | False |
| freeze_memory | False |
| freeze_vision | False |
| freeze_vision_projector | False |
| frozen_eagle | False |
| fsdp_accumulation | retain |
| global_decisions_max | 256 |
| global_windows_per_step | 16 |
| gpus | 0,1,2,3 |
| gradient_accumulation_steps | 8 |
| gradient_checkpointing | False |
| history_sampling | random_ordered |
| image_augmentation | True |
| learning_rate | 2e-05 |
| llm_last_n_layers | 0 |
| lora_alpha | 32 |
| lora_dropout | 0 |
| lora_rank | 16 |
| lora_targets | llm,action_dit |
| max_grad_norm | 1 |
| memory_cognitive_qk_norm | none |
| memory_cognitive_residual_projection | False |
| memory_cognitive_residual_scope | eos |
| memory_conditioning | openvla |
| memory_length | 16 |
| memory_residual_readout | delta |
| memory_update_fused | False |
| microbatch | 8 |
| optimizer_eps | 1e-08 |
| original_steps | 20000 |
| scheduler | constant |
| seed | 42 |
| sharding | SHARD_GRAD_OP |
| steps | 20000 |
| stop_after_step | 16000 |
| tasks | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49] |
| train_episodes | 10000 |
| training_protocol | memoryvla_openvla_b1k_group16_raw_v2 |
| validation_enabled | False |
| validation_episodes | 0 |
| validation_gpu | 未记录 |
| validation_interval | 500 |
| warmup_steps | 0 |
| weight_decay | 0 |

### 指标覆盖

| 字段 | 有限记录 | 零值记录 | 缺失或非有限 | 均值 | 最小 / 最大 |
| --- | --- | --- | --- | --- | --- |
| body_flow_mse.base | 16000 | 0 | 0 | 0.046135908 | 0.021346877 / 0.10330422 |
| body_flow_mse.left_arm | 16000 | 0 | 0 | 0.0037072393 | 0.0010008509 / 0.0088993186 |
| body_flow_mse.left_gripper | 16000 | 0 | 0 | 0.014358659 | 0.0004159218 / 0.12042696 |
| body_flow_mse.right_arm | 16000 | 0 | 0 | 0.0037613074 | 0.0013743278 / 0.011758543 |
| body_flow_mse.right_gripper | 16000 | 0 | 0 | 0.021151277 | 0.00058703236 / 0.48197294 |
| body_flow_mse.torso | 16000 | 0 | 0 | 0.010984494 | 0.0032411045 / 0.041966953 |
| component_grad_norms.action_head.memory_ | 1606 | 0 | 14394 | 0.014034947 | 0.0063512973 / 0.67722178 |
| component_grad_norms.action_head.model | 1606 | 0 | 14394 | 0.017741879 | 0.010104267 / 0.33657992 |
| component_grad_norms.backbone | 1606 | 0 | 14394 | 0.035884107 | 0.012273719 / 0.84627712 |
| component_grad_norms.memory.cognitive | 1606 | 0 | 14394 | 0.0003467866 | 1.7388116e-05 / 0.24329323 |
| component_grad_norms.memory.perceptual | 1606 | 0 | 14394 | 0.017951811 | 0.0074211969 / 1.3431749 |
| decisions_per_second_end_to_end | 16000 | 0 | 0 | 27.396739 | 3.6985744 / 42.087169 |
| gradient_norm | 16000 | 0 | 0 | 0.09729311 | 0.040563975 / 3.5800807 |
| gradient_norm_after_clip | 16000 | 0 | 0 | 0.096931286 | 0.040563975 / 1 |
| learning_rate | 1606 | 0 | 14394 | 2e-05 | 2e-05 / 2e-05 |
| loss | 16000 | 0 | 0 | 0.01174502 | 0.0068790678 / 0.047464521 |
| memory_gradients.cognitive.gate_fusion_blocks.proj.bias | 1606 | 0 | 14394 | 1.7397563e-06 | 7.1420295e-08 / 0.0011871384 |
| memory_gradients.cognitive.gate_fusion_blocks.proj.weight | 1606 | 0 | 14394 | 0.00033285409 | 1.4074207e-05 / 0.23634218 |
| memory_gradients.cognitive.retrieval_blocks.0._fsdp_wrapped_module.attn_norm.bias | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.cognitive.retrieval_blocks.0._fsdp_wrapped_module.attn_norm.weight | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.cognitive.retrieval_blocks.0._fsdp_wrapped_module.ffn.0.bias | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.cognitive.retrieval_blocks.0._fsdp_wrapped_module.ffn.0.weight | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.cognitive.retrieval_blocks.0._fsdp_wrapped_module.ffn.2.bias | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.cognitive.retrieval_blocks.0._fsdp_wrapped_module.ffn.2.weight | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.cognitive.retrieval_blocks.0._fsdp_wrapped_module.ffn_norm.bias | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.cognitive.retrieval_blocks.0._fsdp_wrapped_module.ffn_norm.weight | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.cognitive.retrieval_blocks.0._fsdp_wrapped_module.k_proj.bias | 1606 | 0 | 14394 | 3.6864456e-11 | 2.714322e-12 / 1.0459651e-08 |
| memory_gradients.cognitive.retrieval_blocks.0._fsdp_wrapped_module.k_proj.weight | 1606 | 0 | 14394 | 7.7868023e-07 | 1.9706301e-08 / 0.00049827714 |
| memory_gradients.cognitive.retrieval_blocks.0._fsdp_wrapped_module.q_proj.bias | 1606 | 0 | 14394 | 5.8037601e-09 | 1.0233006e-10 / 4.2009115e-06 |
| memory_gradients.cognitive.retrieval_blocks.0._fsdp_wrapped_module.q_proj.weight | 1606 | 0 | 14394 | 7.8623734e-07 | 2.0229459e-08 / 0.00048670874 |
| memory_gradients.cognitive.retrieval_blocks.0._fsdp_wrapped_module.v_proj.bias | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.cognitive.retrieval_blocks.0._fsdp_wrapped_module.v_proj.weight | 1606 | 0 | 14394 | 1.4006497e-05 | 1.2328613e-06 / 0.0090356907 |
| memory_gradients.cognitive.retrieval_blocks.1._fsdp_wrapped_module.attn_norm.bias | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.cognitive.retrieval_blocks.1._fsdp_wrapped_module.attn_norm.weight | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.cognitive.retrieval_blocks.1._fsdp_wrapped_module.ffn.0.bias | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.cognitive.retrieval_blocks.1._fsdp_wrapped_module.ffn.0.weight | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.cognitive.retrieval_blocks.1._fsdp_wrapped_module.ffn.2.bias | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.cognitive.retrieval_blocks.1._fsdp_wrapped_module.ffn.2.weight | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.cognitive.retrieval_blocks.1._fsdp_wrapped_module.ffn_norm.bias | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.cognitive.retrieval_blocks.1._fsdp_wrapped_module.ffn_norm.weight | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.cognitive.retrieval_blocks.1._fsdp_wrapped_module.k_proj.bias | 1606 | 0 | 14394 | 7.622054e-11 | 4.9073475e-12 / 1.4745387e-08 |
| memory_gradients.cognitive.retrieval_blocks.1._fsdp_wrapped_module.k_proj.weight | 1606 | 0 | 14394 | 1.5928522e-06 | 3.9980918e-08 / 0.00063324609 |
| memory_gradients.cognitive.retrieval_blocks.1._fsdp_wrapped_module.q_proj.bias | 1606 | 0 | 14394 | 3.2966749e-08 | 8.4857144e-10 / 1.4269465e-05 |
| memory_gradients.cognitive.retrieval_blocks.1._fsdp_wrapped_module.q_proj.weight | 1606 | 0 | 14394 | 1.4952272e-06 | 3.8505053e-08 / 0.00064093049 |
| memory_gradients.cognitive.retrieval_blocks.1._fsdp_wrapped_module.v_proj.bias | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.cognitive.retrieval_blocks.1._fsdp_wrapped_module.v_proj.weight | 1606 | 0 | 14394 | 6.9641394e-05 | 7.285817e-06 / 0.043938149 |
| memory_gradients.cognitive.timestep_encoder.mlp.0.bias | 1606 | 0 | 14394 | 2.0262791e-09 | 1.3420566e-10 / 3.6676246e-07 |
| memory_gradients.cognitive.timestep_encoder.mlp.0.weight | 1606 | 0 | 14394 | 9.3684508e-08 | 6.5549348e-09 / 1.7708997e-05 |
| memory_gradients.cognitive.timestep_encoder.mlp.2.bias | 1606 | 0 | 14394 | 7.1256097e-11 | 5.0611737e-12 / 1.1522299e-08 |
| memory_gradients.cognitive.timestep_encoder.mlp.2.weight | 1606 | 0 | 14394 | 1.7901431e-07 | 1.2152374e-08 / 3.3687385e-05 |
| memory_gradients.perceptual.gate_fusion_blocks.proj.bias | 1606 | 0 | 14394 | 8.4269593e-05 | 3.4438155e-05 / 0.0044070305 |
| memory_gradients.perceptual.gate_fusion_blocks.proj.weight | 1606 | 0 | 14394 | 0.0011684852 | 0.00048447389 / 0.06698411 |
| memory_gradients.perceptual.retrieval_blocks.0._fsdp_wrapped_module.attn_norm.bias | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.perceptual.retrieval_blocks.0._fsdp_wrapped_module.attn_norm.weight | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.perceptual.retrieval_blocks.0._fsdp_wrapped_module.ffn.0.bias | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.perceptual.retrieval_blocks.0._fsdp_wrapped_module.ffn.0.weight | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.perceptual.retrieval_blocks.0._fsdp_wrapped_module.ffn.2.bias | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.perceptual.retrieval_blocks.0._fsdp_wrapped_module.ffn.2.weight | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.perceptual.retrieval_blocks.0._fsdp_wrapped_module.ffn_norm.bias | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.perceptual.retrieval_blocks.0._fsdp_wrapped_module.ffn_norm.weight | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.perceptual.retrieval_blocks.0._fsdp_wrapped_module.k_proj.bias | 1606 | 0 | 14394 | 8.4015769e-10 | 3.6489448e-10 / 2.839379e-09 |
| memory_gradients.perceptual.retrieval_blocks.0._fsdp_wrapped_module.k_proj.weight | 1606 | 0 | 14394 | 1.1325131e-05 | 2.7485894e-06 / 5.2741267e-05 |
| memory_gradients.perceptual.retrieval_blocks.0._fsdp_wrapped_module.q_proj.bias | 1606 | 0 | 14394 | 3.0525966e-06 | 1.0629018e-06 / 1.5850614e-05 |
| memory_gradients.perceptual.retrieval_blocks.0._fsdp_wrapped_module.q_proj.weight | 1606 | 0 | 14394 | 9.1517242e-06 | 2.53027e-06 / 3.9186016e-05 |
| memory_gradients.perceptual.retrieval_blocks.0._fsdp_wrapped_module.v_proj.bias | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.perceptual.retrieval_blocks.0._fsdp_wrapped_module.v_proj.weight | 1606 | 0 | 14394 | 0.0019870114 | 0.00068950176 / 0.13236961 |
| memory_gradients.perceptual.retrieval_blocks.1._fsdp_wrapped_module.attn_norm.bias | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.perceptual.retrieval_blocks.1._fsdp_wrapped_module.attn_norm.weight | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.perceptual.retrieval_blocks.1._fsdp_wrapped_module.ffn.0.bias | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.perceptual.retrieval_blocks.1._fsdp_wrapped_module.ffn.0.weight | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.perceptual.retrieval_blocks.1._fsdp_wrapped_module.ffn.2.bias | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.perceptual.retrieval_blocks.1._fsdp_wrapped_module.ffn.2.weight | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.perceptual.retrieval_blocks.1._fsdp_wrapped_module.ffn_norm.bias | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.perceptual.retrieval_blocks.1._fsdp_wrapped_module.ffn_norm.weight | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.perceptual.retrieval_blocks.1._fsdp_wrapped_module.k_proj.bias | 1606 | 0 | 14394 | 7.0086136e-10 | 3.5371758e-10 / 2.2241702e-09 |
| memory_gradients.perceptual.retrieval_blocks.1._fsdp_wrapped_module.k_proj.weight | 1606 | 0 | 14394 | 8.7264068e-06 | 2.8936729e-06 / 4.5961518e-05 |
| memory_gradients.perceptual.retrieval_blocks.1._fsdp_wrapped_module.q_proj.bias | 1606 | 0 | 14394 | 5.7657702e-07 | 2.0670041e-07 / 2.9165601e-06 |
| memory_gradients.perceptual.retrieval_blocks.1._fsdp_wrapped_module.q_proj.weight | 1606 | 0 | 14394 | 8.4309272e-06 | 2.8889215e-06 / 5.148265e-05 |
| memory_gradients.perceptual.retrieval_blocks.1._fsdp_wrapped_module.v_proj.bias | 0 | 0 | 16000 | 未记录 | 未记录 / 未记录 |
| memory_gradients.perceptual.retrieval_blocks.1._fsdp_wrapped_module.v_proj.weight | 1606 | 0 | 14394 | 0.00044701956 | 0.00019617287 / 0.025733588 |
| memory_gradients.perceptual.timestep_encoder.mlp.0.bias | 1606 | 0 | 14394 | 2.6030578e-07 | 3.545679e-08 / 3.2059384e-06 |
| memory_gradients.perceptual.timestep_encoder.mlp.0.weight | 1606 | 0 | 14394 | 4.2747538e-06 | 5.8721071e-07 / 4.6182191e-05 |
| memory_gradients.perceptual.timestep_encoder.mlp.2.bias | 1606 | 0 | 14394 | 3.1333851e-09 | 8.7195651e-10 / 2.4014629e-08 |
| memory_gradients.perceptual.timestep_encoder.mlp.2.weight | 1606 | 0 | 14394 | 4.9924942e-06 | 1.0510596e-06 / 4.6857498e-05 |
| memory_gradients.perceptual_encoder.excite.1.bias | 1606 | 0 | 14394 | 4.2625449e-05 | 1.4993604e-05 / 0.0023295539 |
| memory_gradients.perceptual_encoder.excite.1.weight | 1606 | 0 | 14394 | 0.00045077873 | 0.00020290818 / 0.027345436 |
| memory_gradients.perceptual_encoder.excite.3.bias | 1606 | 0 | 14394 | 0.00010810408 | 4.9648363e-05 / 0.0055278721 |
| memory_gradients.perceptual_encoder.excite.3.weight | 1606 | 0 | 14394 | 0.00012683639 | 6.3153864e-05 / 0.0080939578 |
| memory_gradients.perceptual_encoder.expand.weight | 1606 | 0 | 14394 | 0.0085097273 | 0.0038583309 / 0.43769652 |
| memory_gradients.perceptual_encoder.reduce.weight | 1606 | 0 | 14394 | 0.015144082 | 0.0059901718 / 1.2436821 |
| peak_allocated_gib | 16000 | 0 | 0 | 37.315464 | 14.271565 / 37.454773 |
| seconds | 16000 | 0 | 0 | 10.358076 | 6.0826139 / 69.215858 |

### 按原始 epoch 字段汇总

| epoch 原值 | 记录数 | 有限 Loss 均值 |
| --- | --- | --- |
| 0 | 625 | 0.015410647 |
| 1 | 625 | 0.013825724 |
| 2 | 625 | 0.013435059 |
| 3 | 625 | 0.012984534 |
| 4 | 625 | 0.012610434 |
| 5 | 625 | 0.012453778 |
| 6 | 625 | 0.012078032 |
| 7 | 625 | 0.011897155 |
| 8 | 625 | 0.011890788 |
| 9 | 625 | 0.011642271 |
| 10 | 625 | 0.011463126 |
| 11 | 625 | 0.011407739 |
| 12 | 625 | 0.011343611 |
| 13 | 625 | 0.01140452 |
| 14 | 625 | 0.011210119 |
| 15 | 625 | 0.011227215 |
| 16 | 625 | 0.011015365 |
| 17 | 625 | 0.010903672 |
| 18 | 625 | 0.010841212 |
| 19 | 625 | 0.011204489 |
| 20 | 625 | 0.011031738 |
| 21 | 625 | 0.010884838 |
| 22 | 625 | 0.010837297 |
| 23 | 625 | 0.010721675 |
| 24 | 625 | 0.010597326 |
| 25 | 375 | 0.010583586 |

### 工作台列出的 checkpoint

仅列元数据；权重与 optimizer tensor 留在训练设备。完整性标记见事件清单中的原始文件。

| optimizer step | 原始路径 |
| --- | --- |
| 16000 | /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v1__test01__exp13__run001/memory_v1/checkpoints/v1__test01__exp13__run001__step016000 |
| 15000 | /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v1__test01__exp13__run001/memory_v1/checkpoints/v1__test01__exp13__run001__step015000 |
| 14000 | /nvme2/memory_vla_exp13_retained_checkpoints/v1__test01__exp13__run001__step014000 |
| 13000 | /nvme2/memory_vla_exp13_retained_checkpoints/v1__test01__exp13__run001__step013000 |
| 12000 | /nvme2/memory_vla_exp13_retained_checkpoints/v1__test01__exp13__run001__step012000 |
| 11000 | /nvme2/memory_vla_exp13_retained_checkpoints/v1__test01__exp13__run001__step011000 |
| 10000 | /nvme2/memory_vla_exp13_retained_checkpoints/v1__test01__exp13__run001__step010000 |
| 9000 | /nvme2/memory_vla_exp13_retained_checkpoints/v1__test01__exp13__run001__step009000 |
| 8000 | /nvme2/memory_vla_exp13_retained_checkpoints/v1__test01__exp13__run001__step008000 |
| 7000 | /nvme2/memory_vla_exp13_retained_checkpoints/v1__test01__exp13__run001__step007000 |
| 6000 | /nvme2/memory_vla_exp13_retained_checkpoints/v1__test01__exp13__run001__step006000 |
| 4000 | /nvme2/memory_vla_exp13_retained_checkpoints/v1__test01__exp13__run001__step004000 |
| 2000 | /nvme2/memory_vla_exp13_retained_checkpoints/v1__test01__exp13__run001__step002000 |
| 46 | /nvme2/memory_vla_exp13_retained_checkpoints/v1__test01__exp13__run001__step000046 |
| 34 | /nvme2/memory_vla_exp13_retained_checkpoints/v1__test01__exp13__run001__step000034 |

### 闭环评估

- [原始结果](runs/a100/v1__test01__exp13__run001/memory_v1/evaluations/85718b166932cb57/results.json)：结果文件状态 complete，工作台状态 complete；完成 10/10；成功 0/10（分母为原始 official_episodes 字段）；checkpoint `/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v1__test01__exp13__run001/memory_v1/checkpoints/v1__test01__exp13__run001__step016000`。
