# 训练分析报告

由工作台归档程序生成；不含 AI 推断。缺失字段为未记录，不以零替代。

## 1. 实验身份与权重

| 字段 | 值 |
| --- | --- |
| run | v1__test01__exp13__run001 |
| device | a100 |
| arm | memory_v1 |
| status | completed |
| step | 16000 |
| 目标步数 | 16000 |
| 归档事件 | 204b3706c90523df31cb-958f37778991 |
| 归档时间 | 2026-11-27T06:20:13.960485+00:00 |
| 训练备注 | User confirmed 2026-11-25. Direct four-A100 start, no queue; resumable full checkpoints every 2000 steps. |
| 源码摘要 | 68b6d82b8ed2608046b86ab39c58442d22f122f5e46fc4e62c2dafc05324fc85 |
| 初始权重/续训 | {"checkpoint": "/nvme3/huggingface/train_model/GR00T/GR00T-N1.6-3B", "label": "官方 GR00T N1.6-3B · 通用预训练", "mode": "resume", "optimizer_restored": true, "resume_checkpoint": "/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v1__test01__exp13__run001/memory_v1/checkpoints/v1__test01__exp13__run001__step011000", "resume_record": "/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v1__test01__exp13__run001/config/web_resume_latest.json", "resume_step": 11000, "source_file": "/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v1__test01__exp13__run001/memory_v1/config/ddp.json"} |

## 2. 实际训练配置

配置来源：["/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v1__test01__exp13__run001/config/web_launch.json", "/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v1__test01__exp13__run001/config/ddp_launch.json", "/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v1__test01__exp13__run001/memory_v1/config/ddp.json", "/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v1__test01__exp13__run001/config/web_resume_latest.json"]

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

## 3. 实际运行与数据

| 字段 | 值 |
| --- | --- |
| prepared.mode | 未记录 |
| prepared.train_frames | 未记录 |
| prepared.val_frames | 未记录 |
| prepared.train_valid_anchors | 未记录 |
| prepared.accepted_cells | 未记录 |
| prepared.hashes | 未记录 |
| split | 未记录 |
| 初始权重分片 SHA-256 | 未记录 |
| 统计量来源 | 未记录 |

## 4. 训练指标

训练 Loss 为记录点/日志区间值；验证 Loss 单独列出。不同动作归一化、mask、噪声采样及数据划分下的 Loss 不宜直接比较。
首尾统计为最多100条日志记录；日志间隔可能大于1步，步号间隔不等于丢失训练更新。

| 字段 | 值 |
| --- | --- |
| records | 16000 |
| first_step | 1 |
| last_step | 16000 |
| duplicate_steps | 0 |
| nonincreasing_transitions | 0 |
| interior_missing_steps | 0 |
| incomplete_tail_records | 0 |
| finite_loss | 16000 |
| loss_first100 | 0.018298337 |
| loss_last100 | 0.010263368 |
| loss_min | 0.0068790678 |

### 每1000 optimizer steps 的训练记录均值

| 步数范围 | 记录数 | Loss | LR首值 | LR末值 | 梯度范数 | 底盘 | 躯干 | 左臂 | 右臂 | 左夹爪 | 右夹爪 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1–1000 | 1000 | 0.014896035 | 2e-05 | 2e-05 | 0.17480324 | 0.05664576 | 0.015424502 | 0.0043305626 | 0.0044294854 | 0.020020983 | 0.0296322 |
| 1001–2000 | 1000 | 0.013490831 | 2e-05 | 2e-05 | 0.13476223 | 0.05206309 | 0.013273414 | 0.0041319406 | 0.0041939412 | 0.017132778 | 0.025592238 |
| 2001–3000 | 1000 | 0.012679591 | 2e-05 | 2e-05 | 0.10778713 | 0.049697804 | 0.011961868 | 0.003965424 | 0.0040405332 | 0.015613952 | 0.023034055 |
| 3001–4000 | 1000 | 0.012412588 | 2e-05 | 2e-05 | 0.10818074 | 0.048590293 | 0.011607891 | 0.0038434101 | 0.0039246209 | 0.015496085 | 0.02341479 |
| 4001–5000 | 1000 | 0.011955555 | 2e-05 | 2e-05 | 0.096124144 | 0.046941102 | 0.011188589 | 0.0037802836 | 0.0038392518 | 0.014797204 | 0.02126616 |
| 5001–6000 | 1000 | 0.011784234 | 2e-05 | 2e-05 | 0.096238277 | 0.046466555 | 0.011010028 | 0.0036947102 | 0.003792167 | 0.014265767 | 0.020923705 |
| 6001–7000 | 1000 | 0.011533741 | 2e-05 | 2e-05 | 0.090369958 | 0.045462797 | 0.010749481 | 0.0037010253 | 0.0037663764 | 0.013325709 | 0.020292215 |
| 7001–8000 | 1000 | 0.01133794 | 2e-05 | 2e-05 | 0.088847035 | 0.045011599 | 0.010406395 | 0.0036175536 | 0.0036841668 | 0.013261049 | 0.019739155 |
| 8001–9000 | 1000 | 0.011400048 | 2e-05 | 2e-05 | 0.090386323 | 0.044640171 | 0.010508881 | 0.0036180786 | 0.003669388 | 0.014165877 | 0.021066934 |
| 9001–10000 | 1000 | 0.011187404 | 2e-05 | 2e-05 | 0.086883457 | 0.044066581 | 0.010223838 | 0.0035835808 | 0.0035952568 | 0.013580242 | 0.020383091 |
| 10001–11000 | 1000 | 0.010948212 | 2e-05 | 2e-05 | 0.083753493 | 0.043568282 | 0.0099806186 | 0.0034981151 | 0.003537767 | 0.013275143 | 0.018655249 |
| 11001–12000 | 1000 | 0.010878762 | 2e-05 | 2e-05 | 0.082283534 | 0.043106251 | 0.0099264732 | 0.0035285515 | 0.0034950249 | 0.013244389 | 0.018777458 |
| 12001–13000 | 1000 | 0.011175077 | 2e-05 | 2e-05 | 0.082323249 | 0.043802397 | 0.010312369 | 0.0036000485 | 0.003629862 | 0.013938729 | 0.019821997 |
| 13001–14000 | 1000 | 0.010890633 | 2e-05 | 2e-05 | 0.079215514 | 0.043218109 | 0.0098607514 | 0.0034729471 | 0.0035466083 | 0.012921265 | 0.019329069 |
| 14001–15000 | 1000 | 0.010757495 | 2e-05 | 2e-05 | 0.077903536 | 0.042677441 | 0.0097921067 | 0.0034613237 | 0.00349656 | 0.012704684 | 0.01881176 |
| 15001–16000 | 1000 | 0.010592173 | 2e-05 | 2e-05 | 0.076827897 | 0.042216296 | 0.0095246976 | 0.003488273 | 0.003539908 | 0.011994685 | 0.017680355 |

### 按实际 epoch 字段汇总

| epoch | 记录数 | Loss |
| --- | ---: | ---: |
| 0 | 625 | 0.015410647 |
| 1 | 625 | 0.013825724 |
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
| 2 | 625 | 0.013435059 |
| 20 | 625 | 0.011031738 |
| 21 | 625 | 0.010884838 |
| 22 | 625 | 0.010837297 |
| 23 | 625 | 0.010721675 |
| 24 | 625 | 0.010597326 |
| 25 | 375 | 0.010583586 |
| 3 | 625 | 0.012984534 |
| 4 | 625 | 0.012610434 |
| 5 | 625 | 0.012453778 |
| 6 | 625 | 0.012078032 |
| 7 | 625 | 0.011897155 |
| 8 | 625 | 0.011890788 |
| 9 | 625 | 0.011642271 |

### 验证指标（独立于训练记录）

| step | FM Loss | 样本数 |
| ---: | ---: | ---: |
| 未记录 | 未记录 | 未记录 |

### 指标覆盖

| 字段 | 有限记录 | 缺失/非有限 | 均值 | 最小 | 最大 |
| --- | ---: | ---: | ---: | ---: | ---: |
| body_flow_mse.base | 16000 | 0 | 0.046135908 | 0.021346877 | 0.10330422 |
| body_flow_mse.left_arm | 16000 | 0 | 0.0037072393 | 0.0010008509 | 0.0088993186 |
| body_flow_mse.left_gripper | 16000 | 0 | 0.014358659 | 0.0004159218 | 0.12042696 |
| body_flow_mse.right_arm | 16000 | 0 | 0.0037613074 | 0.0013743278 | 0.011758543 |
| body_flow_mse.right_gripper | 16000 | 0 | 0.021151277 | 0.00058703236 | 0.48197294 |
| body_flow_mse.torso | 16000 | 0 | 0.010984494 | 0.0032411045 | 0.041966953 |
| component_grad_norms.action_head.memory_ | 1606 | 14394 | 0.014034947 | 0.0063512973 | 0.67722178 |
| component_grad_norms.action_head.model | 1606 | 14394 | 0.017741879 | 0.010104267 | 0.33657992 |
| component_grad_norms.backbone | 1606 | 14394 | 0.035884107 | 0.012273719 | 0.84627712 |
| component_grad_norms.memory.cognitive | 1606 | 14394 | 0.0003467866 | 1.7388116e-05 | 0.24329323 |
| component_grad_norms.memory.perceptual | 1606 | 14394 | 0.017951811 | 0.0074211969 | 1.3431749 |
| decisions_per_second_end_to_end | 16000 | 0 | 27.396739 | 3.6985744 | 42.087169 |
| gradient_norm | 16000 | 0 | 0.09729311 | 0.040563975 | 3.5800807 |
| gradient_norm_after_clip | 16000 | 0 | 0.096931286 | 0.040563975 | 1 |
| learning_rate | 1606 | 14394 | 2e-05 | 2e-05 | 2e-05 |
| loss | 16000 | 0 | 0.01174502 | 0.0068790678 | 0.047464521 |
| memory_gradients.cognitive.gate_fusion_blocks.proj.bias | 1606 | 14394 | 1.7397563e-06 | 7.1420295e-08 | 0.0011871384 |
| memory_gradients.cognitive.gate_fusion_blocks.proj.weight | 1606 | 14394 | 0.00033285409 | 1.4074207e-05 | 0.23634218 |
| memory_gradients.cognitive.retrieval_blocks.0._fsdp_wrapped_module.attn_norm.bias | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.cognitive.retrieval_blocks.0._fsdp_wrapped_module.attn_norm.weight | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.cognitive.retrieval_blocks.0._fsdp_wrapped_module.ffn.0.bias | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.cognitive.retrieval_blocks.0._fsdp_wrapped_module.ffn.0.weight | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.cognitive.retrieval_blocks.0._fsdp_wrapped_module.ffn.2.bias | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.cognitive.retrieval_blocks.0._fsdp_wrapped_module.ffn.2.weight | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.cognitive.retrieval_blocks.0._fsdp_wrapped_module.ffn_norm.bias | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.cognitive.retrieval_blocks.0._fsdp_wrapped_module.ffn_norm.weight | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.cognitive.retrieval_blocks.0._fsdp_wrapped_module.k_proj.bias | 1606 | 14394 | 3.6864456e-11 | 2.714322e-12 | 1.0459651e-08 |
| memory_gradients.cognitive.retrieval_blocks.0._fsdp_wrapped_module.k_proj.weight | 1606 | 14394 | 7.7868023e-07 | 1.9706301e-08 | 0.00049827714 |
| memory_gradients.cognitive.retrieval_blocks.0._fsdp_wrapped_module.q_proj.bias | 1606 | 14394 | 5.8037601e-09 | 1.0233006e-10 | 4.2009115e-06 |
| memory_gradients.cognitive.retrieval_blocks.0._fsdp_wrapped_module.q_proj.weight | 1606 | 14394 | 7.8623734e-07 | 2.0229459e-08 | 0.00048670874 |
| memory_gradients.cognitive.retrieval_blocks.0._fsdp_wrapped_module.v_proj.bias | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.cognitive.retrieval_blocks.0._fsdp_wrapped_module.v_proj.weight | 1606 | 14394 | 1.4006497e-05 | 1.2328613e-06 | 0.0090356907 |
| memory_gradients.cognitive.retrieval_blocks.1._fsdp_wrapped_module.attn_norm.bias | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.cognitive.retrieval_blocks.1._fsdp_wrapped_module.attn_norm.weight | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.cognitive.retrieval_blocks.1._fsdp_wrapped_module.ffn.0.bias | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.cognitive.retrieval_blocks.1._fsdp_wrapped_module.ffn.0.weight | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.cognitive.retrieval_blocks.1._fsdp_wrapped_module.ffn.2.bias | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.cognitive.retrieval_blocks.1._fsdp_wrapped_module.ffn.2.weight | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.cognitive.retrieval_blocks.1._fsdp_wrapped_module.ffn_norm.bias | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.cognitive.retrieval_blocks.1._fsdp_wrapped_module.ffn_norm.weight | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.cognitive.retrieval_blocks.1._fsdp_wrapped_module.k_proj.bias | 1606 | 14394 | 7.622054e-11 | 4.9073475e-12 | 1.4745387e-08 |
| memory_gradients.cognitive.retrieval_blocks.1._fsdp_wrapped_module.k_proj.weight | 1606 | 14394 | 1.5928522e-06 | 3.9980918e-08 | 0.00063324609 |
| memory_gradients.cognitive.retrieval_blocks.1._fsdp_wrapped_module.q_proj.bias | 1606 | 14394 | 3.2966749e-08 | 8.4857144e-10 | 1.4269465e-05 |
| memory_gradients.cognitive.retrieval_blocks.1._fsdp_wrapped_module.q_proj.weight | 1606 | 14394 | 1.4952272e-06 | 3.8505053e-08 | 0.00064093049 |
| memory_gradients.cognitive.retrieval_blocks.1._fsdp_wrapped_module.v_proj.bias | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.cognitive.retrieval_blocks.1._fsdp_wrapped_module.v_proj.weight | 1606 | 14394 | 6.9641394e-05 | 7.285817e-06 | 0.043938149 |
| memory_gradients.cognitive.timestep_encoder.mlp.0.bias | 1606 | 14394 | 2.0262791e-09 | 1.3420566e-10 | 3.6676246e-07 |
| memory_gradients.cognitive.timestep_encoder.mlp.0.weight | 1606 | 14394 | 9.3684508e-08 | 6.5549348e-09 | 1.7708997e-05 |
| memory_gradients.cognitive.timestep_encoder.mlp.2.bias | 1606 | 14394 | 7.1256097e-11 | 5.0611737e-12 | 1.1522299e-08 |
| memory_gradients.cognitive.timestep_encoder.mlp.2.weight | 1606 | 14394 | 1.7901431e-07 | 1.2152374e-08 | 3.3687385e-05 |
| memory_gradients.perceptual.gate_fusion_blocks.proj.bias | 1606 | 14394 | 8.4269593e-05 | 3.4438155e-05 | 0.0044070305 |
| memory_gradients.perceptual.gate_fusion_blocks.proj.weight | 1606 | 14394 | 0.0011684852 | 0.00048447389 | 0.06698411 |
| memory_gradients.perceptual.retrieval_blocks.0._fsdp_wrapped_module.attn_norm.bias | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.perceptual.retrieval_blocks.0._fsdp_wrapped_module.attn_norm.weight | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.perceptual.retrieval_blocks.0._fsdp_wrapped_module.ffn.0.bias | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.perceptual.retrieval_blocks.0._fsdp_wrapped_module.ffn.0.weight | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.perceptual.retrieval_blocks.0._fsdp_wrapped_module.ffn.2.bias | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.perceptual.retrieval_blocks.0._fsdp_wrapped_module.ffn.2.weight | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.perceptual.retrieval_blocks.0._fsdp_wrapped_module.ffn_norm.bias | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.perceptual.retrieval_blocks.0._fsdp_wrapped_module.ffn_norm.weight | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.perceptual.retrieval_blocks.0._fsdp_wrapped_module.k_proj.bias | 1606 | 14394 | 8.4015769e-10 | 3.6489448e-10 | 2.839379e-09 |
| memory_gradients.perceptual.retrieval_blocks.0._fsdp_wrapped_module.k_proj.weight | 1606 | 14394 | 1.1325131e-05 | 2.7485894e-06 | 5.2741267e-05 |
| memory_gradients.perceptual.retrieval_blocks.0._fsdp_wrapped_module.q_proj.bias | 1606 | 14394 | 3.0525966e-06 | 1.0629018e-06 | 1.5850614e-05 |
| memory_gradients.perceptual.retrieval_blocks.0._fsdp_wrapped_module.q_proj.weight | 1606 | 14394 | 9.1517242e-06 | 2.53027e-06 | 3.9186016e-05 |
| memory_gradients.perceptual.retrieval_blocks.0._fsdp_wrapped_module.v_proj.bias | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.perceptual.retrieval_blocks.0._fsdp_wrapped_module.v_proj.weight | 1606 | 14394 | 0.0019870114 | 0.00068950176 | 0.13236961 |
| memory_gradients.perceptual.retrieval_blocks.1._fsdp_wrapped_module.attn_norm.bias | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.perceptual.retrieval_blocks.1._fsdp_wrapped_module.attn_norm.weight | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.perceptual.retrieval_blocks.1._fsdp_wrapped_module.ffn.0.bias | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.perceptual.retrieval_blocks.1._fsdp_wrapped_module.ffn.0.weight | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.perceptual.retrieval_blocks.1._fsdp_wrapped_module.ffn.2.bias | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.perceptual.retrieval_blocks.1._fsdp_wrapped_module.ffn.2.weight | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.perceptual.retrieval_blocks.1._fsdp_wrapped_module.ffn_norm.bias | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.perceptual.retrieval_blocks.1._fsdp_wrapped_module.ffn_norm.weight | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.perceptual.retrieval_blocks.1._fsdp_wrapped_module.k_proj.bias | 1606 | 14394 | 7.0086136e-10 | 3.5371758e-10 | 2.2241702e-09 |
| memory_gradients.perceptual.retrieval_blocks.1._fsdp_wrapped_module.k_proj.weight | 1606 | 14394 | 8.7264068e-06 | 2.8936729e-06 | 4.5961518e-05 |
| memory_gradients.perceptual.retrieval_blocks.1._fsdp_wrapped_module.q_proj.bias | 1606 | 14394 | 5.7657702e-07 | 2.0670041e-07 | 2.9165601e-06 |
| memory_gradients.perceptual.retrieval_blocks.1._fsdp_wrapped_module.q_proj.weight | 1606 | 14394 | 8.4309272e-06 | 2.8889215e-06 | 5.148265e-05 |
| memory_gradients.perceptual.retrieval_blocks.1._fsdp_wrapped_module.v_proj.bias | 0 | 16000 | 未记录 | 未记录 | 未记录 |
| memory_gradients.perceptual.retrieval_blocks.1._fsdp_wrapped_module.v_proj.weight | 1606 | 14394 | 0.00044701956 | 0.00019617287 | 0.025733588 |
| memory_gradients.perceptual.timestep_encoder.mlp.0.bias | 1606 | 14394 | 2.6030578e-07 | 3.545679e-08 | 3.2059384e-06 |
| memory_gradients.perceptual.timestep_encoder.mlp.0.weight | 1606 | 14394 | 4.2747538e-06 | 5.8721071e-07 | 4.6182191e-05 |
| memory_gradients.perceptual.timestep_encoder.mlp.2.bias | 1606 | 14394 | 3.1333851e-09 | 8.7195651e-10 | 2.4014629e-08 |
| memory_gradients.perceptual.timestep_encoder.mlp.2.weight | 1606 | 14394 | 4.9924942e-06 | 1.0510596e-06 | 4.6857498e-05 |
| memory_gradients.perceptual_encoder.excite.1.bias | 1606 | 14394 | 4.2625449e-05 | 1.4993604e-05 | 0.0023295539 |
| memory_gradients.perceptual_encoder.excite.1.weight | 1606 | 14394 | 0.00045077873 | 0.00020290818 | 0.027345436 |
| memory_gradients.perceptual_encoder.excite.3.bias | 1606 | 14394 | 0.00010810408 | 4.9648363e-05 | 0.0055278721 |
| memory_gradients.perceptual_encoder.excite.3.weight | 1606 | 14394 | 0.00012683639 | 6.3153864e-05 | 0.0080939578 |
| memory_gradients.perceptual_encoder.expand.weight | 1606 | 14394 | 0.0085097273 | 0.0038583309 | 0.43769652 |
| memory_gradients.perceptual_encoder.reduce.weight | 1606 | 14394 | 0.015144082 | 0.0059901718 | 1.2436821 |
| peak_allocated_gib | 16000 | 0 | 37.315464 | 14.271565 | 37.454773 |
| seconds | 16000 | 0 | 10.358076 | 6.0826139 | 69.215858 |

## 5. Checkpoint

| step | 路径 |
| ---: | --- |
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

## 6. 闭环评估（仅明确绑定本 run 的终态记录）

### evaluation/V_1/memory/task40_eval15/results.json

| 字段 | 值 |
| --- | --- |
| checkpoint | /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v1__test01__exp13__run001/memory_v1/checkpoints/v1__test01__exp13__run001__step016000 |
| 状态 | complete |
| 完成/计划 | [10, 10] |
| 成功/有效 | [0, 10] |
| Q Score | 0 |
| Task Progress | 22.5 |
| 实际评估配置 | {"mode": "single2", "preset": "official-eval", "model_name": "groot", "device": "local", "naming_schema": "groot-runs-v2", "evaluation_id": "task40_eval15", "policy_identity": {"version": "V_1", "method": "memory", "training_id": "v1__test01__exp13__run001", "train_number": 1, "checkpoint": "/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v1__test01__exp13__run001/memory_v1/checkpoints/v1__test01__exp13__run001__step016000", "checkpoint_step": "v1__test01__exp13__run001__step016000", "source_dir": "/nvme3/projects/memory_vla/deployments/v1__test01__exp13__run001__resume002", "source_digest": "68b6d82b8ed2608046b86ab39c58442d22f122f5e46fc4e62c2dafc05324fc85", "training_seed": 42, "policy_seed": "42", "history_mode": "normal", "task_id": "0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29_30_31_32_33_34_35_36_37_38_39_40_41_42_43_44_45_46_47_48_49", "pid": 3857420, "start_ticks": 1409493654, "host": "172.28.62.17", "port": 5557, "naming_schema": "groot-runs-v2", "task40_diagnostics": false, "task40_trace": false, "gpu_index": "3", "binding": "observed_process"}, "method": "memory", "version": "V_1", "group": "memory", "protocol": {"id": "b1k-groot-v1", "sha256": "620ecd00b89129120d3a8c9ffdaa4c7c44aa560991a9da3190e29b0bee940508"}, "run_tag": null, "task_id": 40, "n_envs_per_gpu": 2, "instances": {"0": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}, "base_seed": 0, "seed_source": "0", "seed_mode": "sequence", "episodes_per_gpu": 10, "synchronized_start": false, "policy_server": "172.28.62.17:5557", "action_chunk": 8, "record_video": true, "video_steps_per_render": 24, "video_fps": 20, "video_speedup": 16.0, "viewer_enabled": true, "task40_stage_trace": true, "runner_max_episode_steps": 999999999, "diagnostics": false, "wait_for_gpu": false, "gpu_wait_interval": 30.0, "appdata": {"0": "/nvme3/tmp/memory_vla/groot_runtime_gpu0/appdata"}, "invocation": ["evaluation/scripts/run_groot_eval.sh", "--gpu-index", "0", "--group", "memory", "--mode", "single2", "--model-name", "groot", "--policy-host", "172.28.62.17", "--policy-port", "5557", "--preset", "official-eval", "--seed", "0", "--seed-mode", "sequence", "--task", "40", "--version", "V_1", "--video-steps-per-render", "24", "--record-video", "--policy-identity", "{\"version\": \"V_1\", \"method\": \"memory\", \"training_id\": \"v1__test01__exp13__run001\", \"train_number\": 1, \"checkpoint\": \"/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v1__test01__exp13__run001/memory_v1/checkpoints/v1__test01__exp13__run001__step016000\", \"checkpoint_step\": \"v1__test01__exp13__run001__step016000\", \"source_dir\": \"/nvme3/projects/memory_vla/deployments/v1__test01__exp13__run001__resume002\", \"source_digest\": \"68b6d82b8ed2608046b86ab39c58442d22f122f5e46fc4e62c2dafc05324fc85\", \"training_seed\": 42, \"policy_seed\": \"42\", \"history_mode\": \"normal\", \"task_id\": \"0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29_30_31_32_33_34_35_36_37_38_39_40_41_42_43_44_45_46_47_48_49\", \"pid\": 3857420, \"start_ticks\": 1409493654, \"host\": \"172.28.62.17\", \"port\": 5557, \"naming_schema\": \"groot-runs-v2\", \"task40_diagnostics\": false, \"task40_trace\": false, \"gpu_index\": \"3\", \"binding\": \"observed_process\"}"]} |

| 回合 | instance | seed | 成功 | Q Score | Task Progress | 终止 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 0 | False | 0 | 37.5 | {"termination_conditions": {"timeout": {"done": true, "success": false}, "predicate": {"done": false, "success": false}}, "success": false, "goal_status": {"satisfied": [], "unsatisfied": [0, 1]}} |
| 2 | 5 | 1 | False | 0 | 12.5 | {"termination_conditions": {"timeout": {"done": true, "success": false}, "predicate": {"done": false, "success": false}}, "success": false, "goal_status": {"satisfied": [], "unsatisfied": [0, 1]}} |
| 3 | 6 | 2 | False | 0 | 37.5 | {"termination_conditions": {"timeout": {"done": true, "success": false}, "predicate": {"done": false, "success": false}}, "success": false, "goal_status": {"satisfied": [], "unsatisfied": [0, 1]}} |
| 4 | 1 | 1 | False | 0 | 12.5 | {"termination_conditions": {"timeout": {"done": true, "success": false}, "predicate": {"done": false, "success": false}}, "success": false, "goal_status": {"satisfied": [], "unsatisfied": [0, 1]}} |
| 5 | 2 | 2 | False | 0 | 12.5 | {"termination_conditions": {"timeout": {"done": true, "success": false}, "predicate": {"done": false, "success": false}}, "success": false, "goal_status": {"satisfied": [], "unsatisfied": [0, 1]}} |
| 6 | 7 | 3 | False | 0 | 37.5 | {"termination_conditions": {"timeout": {"done": true, "success": false}, "predicate": {"done": false, "success": false}}, "success": false, "goal_status": {"satisfied": [], "unsatisfied": [0, 1]}} |
| 7 | 3 | 3 | False | 0 | 12.5 | {"termination_conditions": {"timeout": {"done": true, "success": false}, "predicate": {"done": false, "success": false}}, "success": false, "goal_status": {"satisfied": [], "unsatisfied": [0, 1]}} |
| 8 | 8 | 4 | False | 0 | 12.5 | {"termination_conditions": {"timeout": {"done": true, "success": false}, "predicate": {"done": false, "success": false}}, "success": false, "goal_status": {"satisfied": [], "unsatisfied": [0, 1]}} |
| 9 | 4 | 4 | False | 0 | 12.5 | {"termination_conditions": {"timeout": {"done": true, "success": false}, "predicate": {"done": false, "success": false}}, "success": false, "goal_status": {"satisfied": [], "unsatisfied": [0, 1]}} |
| 10 | 9 | 5 | False | 0 | 37.5 | {"termination_conditions": {"timeout": {"done": true, "success": false}, "predicate": {"done": false, "success": false}}, "success": false, "goal_status": {"satisfied": [], "unsatisfied": [0, 1]}} |


## 7. 证据与局限

- [归档清单](events/204b3706c90523df31cb-958f37778991/MANIFEST.json)
- [工作台原始快照](events/204b3706c90523df31cb-958f37778991/workbench.json)
- 原始指标按清单中的 SHA-256 分片保存；日志只保存已声明范围。未记录的指标不作推断。
- 归一化审计、实际 processor 配置、源码、终态评估及逐回合数据以归档证据为准。
- [recorded/logs/controller.log](events/204b3706c90523df31cb-958f37778991/recorded/logs/controller.log)
- [recorded/logs/memory_v1_torchrun.log](events/204b3706c90523df31cb-958f37778991/recorded/logs/memory_v1_torchrun.log)
- [recorded/logs/policy_v1__test01__exp13__run001_v1__test01__exp13__run001__step016000_port5557.log](events/204b3706c90523df31cb-958f37778991/recorded/logs/policy_v1__test01__exp13__run001_v1__test01__exp13__run001__step016000_port5557.log)
