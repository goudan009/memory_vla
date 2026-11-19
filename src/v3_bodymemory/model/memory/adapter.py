"""Dual memory conditions for GR00T; legacy checkpoints retain residual injection."""

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from .bank import BottleneckSE, MemoryBank


class MemoryAdapter(nn.Module):
    def __init__(self, vlm_dim=2048, perceptual_dim=256, capacity=16,
                 retrieval_layers=2, group_size=16, eos_token_id=151645, conditioning="residual"):
        super().__init__()
        if conditioning not in ("direct", "residual") or capacity < 1:
            raise ValueError("Memory requires direct/residual conditioning and positive capacity")
        self.conditioning = conditioning
        self.group_size = group_size
        self.eos_token_id = eos_token_id
        self.perceptual_encoder = BottleneckSE(vlm_dim, 512, perceptual_dim)
        self.perceptual = MemoryBank(perceptual_dim, capacity, retrieval_layers)
        self.cognitive = MemoryBank(vlm_dim, capacity, retrieval_layers)
        # Adapt the perceptual width to the existing GR00T action head.
        # Retain the original delta projection only for legacy checkpoints.
        if conditioning == "direct":
            self.perceptual_projection = nn.Linear(perceptual_dim, vlm_dim, bias=False)
        else:
            self.perceptual_residual = nn.Linear(perceptual_dim, vlm_dim, bias=False)
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
        empty_mode = context.get("mode") == "empty"
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
            fused_per = current_per if empty_mode else self.perceptual.read_fuse_write(current_per, key, int(timestep))
            fused_cog = current_cog if empty_mode else self.cognitive.read_fuse_write(current_cog, key, int(timestep))
            if self.conditioning == "direct":
                # Existing AlternateVLDiT image attention reads fused perception;
                # its text attention reads only the fused contextual EOS.
                updated.append(torch.cat((self.perceptual_projection(fused_per), fused_cog), dim=1).squeeze(0))
                continue
            # Preserve the whole contextual VLM sequence and both masks.
            delta = torch.zeros_like(h[i])
            delta[image_mask[i]] = self.perceptual_residual(fused_per - current_per).squeeze(0)
            delta[eos_indices[i]] = (fused_cog - current_cog).squeeze(0).squeeze(0)
            updated.append(h[i] + delta)
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
