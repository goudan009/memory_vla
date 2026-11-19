"""Dual memory conditions for GR00T; legacy checkpoints retain residual injection."""

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from .bank import BottleneckSE, MemoryBank


class MemoryAdapter(nn.Module):
    def __init__(self, vlm_dim=2048, perceptual_dim=256, capacity=16,
                 retrieval_layers=2, group_size=16, eos_token_id=151645, conditioning="residual",
                 cognitive_qk_norm="none", residual_identity_init=False, cognitive_residual_projection=False,
                 cognitive_residual_scope="eos", residual_readout="delta"):
        super().__init__()
        if conditioning not in ("direct", "residual") or capacity < 1:
            raise ValueError("Memory requires direct/residual conditioning and positive capacity")
        if cognitive_residual_projection and not (conditioning == "residual" and residual_identity_init):
            raise ValueError("Cognitive projection requires identity-initialized residual conditioning")
        if cognitive_residual_scope not in ("eos", "text"):
            raise ValueError("Cognitive residual scope must be eos or text")
        if residual_readout not in ("delta", "value") or (residual_readout == "value" and conditioning != "residual"):
            raise ValueError("Value readout requires residual conditioning; readout must be delta or value")
        self.conditioning = conditioning
        self.residual_readout = residual_readout
        self.residual_identity_init = residual_identity_init
        self.cognitive_residual_scope = cognitive_residual_scope
        self.group_size = group_size
        self.eos_token_id = eos_token_id
        self.perceptual_encoder = BottleneckSE(vlm_dim, 512, perceptual_dim)
        self.perceptual = MemoryBank(perceptual_dim, capacity, retrieval_layers)
        self.cognitive = MemoryBank(vlm_dim, capacity, retrieval_layers, qk_norm=cognitive_qk_norm)
        # Adapt the perceptual width to the existing GR00T action head.
        # Retain the original delta projection only for legacy checkpoints.
        if conditioning == "direct":
            self.perceptual_projection = nn.Linear(perceptual_dim, vlm_dim, bias=False)
        else:
            self.perceptual_residual = nn.Linear(perceptual_dim, vlm_dim, bias=False)
            if residual_identity_init:
                self.perceptual_residual._memory_zero_output = True
                nn.init.zeros_(self.perceptual_residual.weight)
                if cognitive_residual_projection:
                    self.cognitive_residual = nn.Linear(vlm_dim, vlm_dim, bias=False)
                    self.cognitive_residual._memory_zero_output = True
                    nn.init.zeros_(self.cognitive_residual.weight)
                else:
                    self.cognitive_residual_scale = nn.Parameter(torch.zeros(()))
        for module in self.modules():
            module._memory_v1 = True

    def reset(self, keys=None):
        self.perceptual.reset(keys)
        self.cognitive.reset(keys)

    def forward(self, backbone_output, input_ids, context):
        h = backbone_output["backbone_features"]
        visual = backbone_output["visual_features"]  # [B,cameras,grid_tokens,D]
        image_mask = backbone_output["image_mask"]
        attention = backbone_output["backbone_attention_mask"]
        b, cameras, n, d = visual.shape
        if len(context["episode_keys"]) != b or len(context["timesteps"]) != b:
            raise ValueError("Memory metadata must match the sample batch")
        group_mode = context.get("mode") == "group"
        chunk_mode = context.get("mode") == "train_sequence"
        history_mode = context.get('history_mode', 'normal')
        if history_mode not in ('normal', 'empty', 'other_episode'):
            raise ValueError('History control must be normal, empty or other_episode')
        if self.training and history_mode != 'normal':
            raise ValueError('History interventions are inference-only')
        empty_mode = context.get("mode") == "empty" or history_mode == 'empty'
        capture = context.get('memory_diagnostics') is True and not self.training
        self.last_diagnostics = (dict(schema_version=1,history_mode=history_mode,observations=[])
                                 if capture else None)
        keys = [tuple(k) if isinstance(k, list) else k for k in context['episode_keys']]
        donor_keys = []
        other_history = {}
        if history_mode == 'other_episode':
            if b < 2 or len(set(keys)) != b:
                raise ValueError('Other-episode control requires at least two distinct concurrent environments')
            donor_keys = keys[1:] + keys[:1]
            # Snapshot before any observation in this batch writes. Own banks
            # keep their real stream; only the read uses another episode.
            for name, bank in (('perceptual', self.perceptual), ('cognitive', self.cognitive)):
                other_history[name] = [
                    [(t, f) for t, f in bank.bank.get(donor, []) if t < int(now)]
                    for donor, now in zip(donor_keys, context['timesteps'])]
        self.last_history_control = dict(mode=history_mode, episode_keys=keys,
                                        donor_keys=donor_keys, timesteps=list(context['timesteps']))
        sequence_length = int(context.get("sequence_length", self.group_size))
        if self.training and not (group_mode or chunk_mode):
            raise ValueError("Training memory requires complete ordered groups")
        group_starts = set()
        if group_mode:
            lengths = context.get("group_lengths")
            if lengths is None:
                lengths = [sequence_length] * (b // max(sequence_length, 1))
            if not lengths or min(lengths) < 1 or sum(lengths) != b:
                raise ValueError("A training batch must contain complete memory groups")
            offset = 0
            for length in lengths:
                group_starts.add(offset)
                offset += length
            self.reset()
        if chunk_mode:
            keys = [tuple(k) if isinstance(k, list) else k for k in context["episode_keys"]]
            if len(set(keys)) != 1:
                raise ValueError("A training sequence chunk must stay in one Episode")
            if context["sequence_start"]:
                self.reset(keys)
            elif keys[0] not in self.cognitive.bank:
                raise ValueError("Missing sequence_start before a continuation chunk")
        # Last valid im_end is the user EOS after all three cameras and text;
        # neither the system EOS nor the final newline/padding is selected.
        eos = (input_ids == self.eos_token_id) & attention
        positions = torch.arange(h.shape[1], device=h.device).expand(b, -1)
        eos_indices = positions.masked_fill(~eos, -1).max(dim=1).values
        if (eos_indices < 0).any() or not torch.all(image_mask.sum(1) == cameras * n):
            raise ValueError("Missing user EOS or visual token/position mismatch")
        updated = []
        for i, (key, timestep) in enumerate(zip(context["episode_keys"], context["timesteps"])):
            key = tuple(key) if isinstance(key, list) else key
            if group_mode and i in group_starts:
                self.reset()
            if group_mode and i not in group_starts:
                previous = context["episode_keys"][i - 1]
                if key != (tuple(previous) if isinstance(previous, list) else previous):
                    raise ValueError("Memory group crosses an Episode boundary")
            # Use the same camera batch in training and online inference.
            # BF16 Conv2d can choose different kernels for B*3 versus 3 cameras.
            current_per = self.perceptual_encoder(visual[i]).reshape(1, cameras * n, -1)
            current_cog = h[i:i + 1, eos_indices[i]:eos_indices[i] + 1]
            per_options = {'history_override': other_history['perceptual'][i]} if other_history else {}
            cog_options = {'history_override': other_history['cognitive'][i]} if other_history else {}
            if capture:
                per_options['capture_diagnostics'] = True
                cog_options.update(capture_diagnostics=True,capture_attention=True)
            if self.residual_readout == "value" and not empty_mode:
                fused_per, perceptual_delta = self.perceptual.read_fuse_write(current_per, key, int(timestep), return_value_message=True, **per_options)
                fused_cog, cognitive_delta = self.cognitive.read_fuse_write(current_cog, key, int(timestep), return_value_message=True, **cog_options)
            else:
                fused_per = current_per if empty_mode else self.perceptual.read_fuse_write(current_per, key, int(timestep), **per_options)
                fused_cog = current_cog if empty_mode else self.cognitive.read_fuse_write(current_cog, key, int(timestep), **cog_options)
                perceptual_delta = fused_per - current_per
                cognitive_delta = fused_cog - current_cog
            diagnostic = None
            if capture:
                diagnostic = dict(write_source=key,read_source=donor_keys[i] if donor_keys else key,
                    timestep=int(timestep),history_read_enabled=not empty_mode,
                    perceptual=self.perceptual.last_diagnostics if not empty_mode else
                        dict(history_timesteps=[],history_slots=0,write_enabled=False),
                    cognitive=self.cognitive.last_diagnostics if not empty_mode else
                        dict(history_timesteps=[],history_slots=0,write_enabled=False))
            if self.conditioning == "direct":
                # Existing AlternateVLDiT image attention reads fused perception;
                # its text attention reads only the fused contextual EOS.
                updated.append(torch.cat((self.perceptual_projection(fused_per), fused_cog), dim=1).squeeze(0))
                if diagnostic is not None:
                    diagnostic['conditioning'] = 'direct'
                    self.last_diagnostics['observations'].append(diagnostic)
                continue
            # Preserve the whole contextual VLM sequence and both masks.
            delta = torch.zeros_like(h[i])
            delta[image_mask[i]] = self.perceptual_residual(perceptual_delta).squeeze(0)
            cognitive_delta = cognitive_delta.squeeze(0).squeeze(0)
            if self.residual_identity_init:
                cognitive_delta = (self.cognitive_residual(cognitive_delta) if hasattr(self, "cognitive_residual")
                                   else self.cognitive_residual_scale * cognitive_delta)
            if self.cognitive_residual_scope == "text":
                delta[attention[i] & ~image_mask[i]] = cognitive_delta
            else:
                delta[eos_indices[i]] = cognitive_delta
            updated.append(h[i] + delta)
            if diagnostic is not None:
                diagnostic.update(conditioning='residual',
                    residual_rms=float(delta.detach().float().square().mean().sqrt()),
                    input_rms=float(h[i].detach().float().square().mean().sqrt()),
                    cognitive_output_rms=float(cognitive_delta.detach().float().square().mean().sqrt()))
                self.last_diagnostics['observations'].append(diagnostic)
        if group_mode:
            self.reset()
        elif chunk_mode and context["sequence_end"]:
            self.reset(keys)  # End this Episode without clearing other Episodes.
        if self.conditioning == "direct":
            attention = torch.ones((b, cameras * n + 1), dtype=torch.bool, device=h.device)
            image_mask = attention.clone()
            image_mask[:, -1] = False
        return BatchFeature(data={"backbone_features": torch.stack(updated),
                                  "backbone_attention_mask": attention, "image_mask": image_mask})
