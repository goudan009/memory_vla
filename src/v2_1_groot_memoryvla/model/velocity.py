"""V1 audited action-head forward and Euler loop; only post-decoder residual insertion.
Original source blob 16664ca6b094c16674ea2474d0db97b14ae01833. No changes to V1 source or VLM conditions.
"""
import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

def training_velocity(self, backbone_output, action_input, refiner, cache, robot):
    """
    Forward pass through the action head.

    Args:
        backbone_output: Output from the backbone model containing:
            - backbone_features: [B, seq_len, backbone_embedding_dim]
            - backbone_attention_mask: [B, seq_len]
        action_input: Input containing:
            - state: [B, state_dim]
            - action: [B, action_horizon, action_dim] (during training)
            - embodiment_id: [B] (embodiment IDs)
            - action_mask: [B, action_horizon, action_dim]

    Returns:
        BatchFeature containing:
            - loss: action prediction loss
    """
    # Set frozen modules to eval
    self.set_frozen_modules_to_eval_mode()

    backbone_output = self.process_backbone_output(backbone_output)

    # Get vision and language embeddings.
    vl_embeds = backbone_output.backbone_features
    device = vl_embeds.device

    # Get embodiment ID.
    embodiment_id = action_input.embodiment_id

    # Embed state.
    state_features = self.state_encoder(action_input.state, embodiment_id)

    # Dropout state features.
    if self.state_dropout_prob > 0:
        do_dropout = (
            torch.rand(state_features.shape[0], device=state_features.device)
            < self.state_dropout_prob
        )
        do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
        state_features = state_features * (1 - do_dropout) + self.mask_token * do_dropout

    # Add Gaussian noise to state features.
    if self.training and self.state_additive_noise_scale > 0:
        print(
            f"Adding Gaussian noise to state features with scale {self.state_additive_noise_scale}"
        )
        noise = torch.randn_like(state_features) * self.state_additive_noise_scale
        state_features = state_features + noise

    # Embed noised action trajectory.
    actions = action_input.action
    noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
    t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
    t = t[:, None, None]  # shape (B,1,1) for broadcast

    noisy_trajectory = (1 - t) * noise + t * actions
    velocity = actions - noise

    # Convert (continuous) t -> discrete if needed
    t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
    action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

    # Maybe add position embedding.
    if self.config.add_pos_embed:
        pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
        pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
        action_features = action_features + pos_embs

    # Join vision, language, state and action embedding along sequence dimension.
    sa_embs = torch.cat((state_features, action_features), dim=1)
    vl_attn_mask = backbone_output.backbone_attention_mask
    fusion_kwargs = self._fusion_kwargs(backbone_output, action_features.shape[1])
    if self.openvla_memory:
        fusion_kwargs["perceptual_condition"] = self._perceptual_condition(backbone_output)

    if self.config.use_alternate_vl_dit:
        image_mask = backbone_output.image_mask
        backbone_attention_mask = backbone_output.backbone_attention_mask
        model_output, _ = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embeds,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=True,
            image_mask=image_mask,
            backbone_attention_mask=backbone_attention_mask,
            **fusion_kwargs,
        )
    else:
        model_output, _ = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embeds,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=True,
            **fusion_kwargs,
        )

    pred = self.action_decoder(model_output, embodiment_id)
    pred_actions = pred[:, -actions.shape[1] :]

    # Slice out only the action portion of pred and target.
    u0 = pred_actions
    pred_actions, delta = refiner(noisy_trajectory, u0, t[:,0,0], cache, robot, 1.)
    action_mask = action_input.action_mask
    action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
    loss = action_loss.sum() / (action_mask.sum() + 1e-6)

    outputs = {
        "loss": loss,
        "base_velocity": u0.detach(),
        "delta": delta.detach(),
        "state_logits": cache.logits,
        "fm_numerator": action_loss.float().sum(),
        "fm_count": action_mask.float().sum(),
        "action_loss": action_loss,
        "action_mask": action_mask,
        "backbone_features": vl_embeds,
        "state_features": state_features,
    }
    if self.memory_fusion_mode != "legacy":
        # Live, post-fusion DiT action features for optional state supervision.
        outputs["fusion_action_features"] = model_output[:, -actions.shape[1]:]
    return outputs

@torch.no_grad()
def inference_velocity(
    self,
    backbone_features: torch.Tensor,
    state_features: torch.Tensor,
    embodiment_id: torch.Tensor,
    backbone_output: BatchFeature,
    refiner, cache, robot, scale, initial_noise=None,
) -> BatchFeature:
    """
    Generate actions using the flow matching diffusion process.

    Args:
        backbone_features: [B, seq_len, backbone_embedding_dim]
        state_features: [B, state_horizon, input_embedding_dim]
        embodiment_id: [B] (embodiment IDs)
        backbone_output: Output from the backbone model
    """
    vl_embeds = backbone_features

    # Set initial actions as the sampled noise.
    batch_size = vl_embeds.shape[0]
    device = vl_embeds.device
    actions = torch.randn(
        size=(batch_size, self.config.action_horizon, self.action_dim),
        dtype=vl_embeds.dtype,
        device=device,
    )

    if initial_noise is not None:
        if initial_noise.shape != actions.shape: raise ValueError('bad initial noise')
        actions = initial_noise.to(actions)
    residual_stats = []
    dt = 1.0 / self.num_inference_timesteps
    fusion_kwargs = self._fusion_kwargs(backbone_output, self.action_horizon)
    # The upstream LIBERO runner uses cognition-only CFG at 1.5. Apply the
    # same guidance to GR00T velocity before its original Euler update.
    unconditional_kwargs = (self._fusion_kwargs(backbone_output, self.action_horizon, unconditional=True)
                            if self.openvla_memory else None)
    if unconditional_kwargs is not None:
        perception = self._perceptual_condition(backbone_output)
        fusion_kwargs["perceptual_condition"] = perception
        unconditional_kwargs["perceptual_condition"] = perception

    # Run denoising steps.
    for t in range(self.num_inference_timesteps):
        t_cont = t / float(self.num_inference_timesteps)  # e.g. goes 0, 1/N, 2/N, ...
        t_discretized = int(t_cont * self.num_timestep_buckets)

        # Embed noised action trajectory.
        timesteps_tensor = torch.full(
            size=(batch_size,), fill_value=t_discretized, device=device
        )
        action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
        # Add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)

        # Run model forward.
        if self.config.use_alternate_vl_dit:
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps_tensor,
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
                **fusion_kwargs,
            )
            if unconditional_kwargs is not None:
                unconditioned = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                    **unconditional_kwargs,
                )
        else:
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps_tensor,
                **fusion_kwargs,
            )
        pred = self.action_decoder(model_output, embodiment_id)

        pred_velocity = pred[:, -self.action_horizon :]
        if unconditional_kwargs is not None:
            unconditioned_velocity = self.action_decoder(unconditioned, embodiment_id)[:, -self.action_horizon:]
            pred_velocity = unconditioned_velocity + 1.5 * (pred_velocity - unconditioned_velocity)

        u0 = pred_velocity
        pred_velocity, delta = refiner(actions, u0, t_cont, cache, robot, scale)
        residual_stats.append((u0.detach(), delta.detach()))
        # Update actions using euler integration.
        actions = actions + dt * pred_velocity
    outputs = {
        "action_pred": actions,
        "residual_stats": residual_stats,
        "state_logits": cache.logits,
        "backbone_features": vl_embeds,
        "state_features": state_features,
    }
    if self.memory_fusion_mode != "legacy":
        outputs["fusion_action_features"] = model_output[:, -self.action_horizon:]
    return BatchFeature(data=outputs)
