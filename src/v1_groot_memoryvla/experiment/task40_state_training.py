"""Task40 G3 state-only forward for use inside the caller's DDP module."""

from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from v1_groot_memoryvla.data.types import EmbodimentTag, MessageType, VLAStepData


LABEL_NAMES = ("attempted_while_incomplete", "door_completed", "open_now")
PANEL_KIND = "task40_state_only_tensor_panel"
BANK_NAMES = ("perceptual", "cognitive")


def load_state_panel(path):
    """Load the exported CPU tensors without needing the source traces or experiment scripts."""
    panel = torch.load(path, map_location="cpu", weights_only=True)
    if (panel.get("schema") != 1 or panel.get("kind") != PANEL_KIND
            or tuple(panel.get("label_names", ())) != LABEL_NAMES):
        raise ValueError("Expected a Task40 state-only tensor panel (schema 1)")
    return panel


class Task40StateHead(nn.Module):
    """Linear multi-label readout of the mean final DiT action token."""

    def __init__(self, feature_dim):
        super().__init__()
        self.projection = nn.Linear(feature_dim, len(LABEL_NAMES))

    def forward(self, action_features):
        pooled = action_features.float().mean(dim=1).to(self.projection.weight.dtype)
        return self.projection(pooled)


def masked_state_loss(logits, targets, target_mask):
    """Mean BCE over known labels; an all-unknown sample yields a differentiable zero."""
    loss = F.binary_cross_entropy_with_logits(
        logits.float().masked_select(target_mask), targets.float().masked_select(target_mask),
        reduction="sum",
    )
    return loss / target_mask.sum().clamp_min(1)


def processor_inputs(processor, sample):
    """Mirror archived R1Pro inference preprocessing, retaining the single-frame time axis."""
    if processor.training:
        raise ValueError("Use a separate eval-mode processor for Task40 state-only inputs")
    embodiment = EmbodimentTag.BEHAVIOR_R1_PRO
    modalities = processor.modality_configs[embodiment.value]
    raw = sample["observation"]
    step = VLAStepData(
        images={key: raw[f"video.{key}"][0].detach().cpu().numpy()
                for key in modalities["video"].modality_keys},
        states={key: raw[f"state.{key}"][0].detach().cpu().numpy()
                for key in modalities["state"].modality_keys},
        actions={}, text=sample["text"], embodiment=embodiment,
    )
    processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
    return processor.collator([processed])["inputs"]


@contextmanager
def _evaluation(module):
    # Restore mixed modes as well as the outer module's mode.
    modes = [(child, child.training) for child in module.modules()]
    try:
        module.eval()
        yield
    finally:
        for child, training in modes:
            child.training = training


def _replay_memory(model, sample, backbone_inputs, backbone_output):
    from .task21_formal import isolated_validation, restore_memory_runtime

    snapshot = sample["memory"]
    device = backbone_output["backbone_features"].device
    runtime = [snapshot[name] for name in BANK_NAMES]
    for name, state in zip(BANK_NAMES, runtime):
        if state["capacity"] != getattr(model.memory, name).capacity:
            raise ValueError(f"Archived {name} memory capacity differs from the model")
    # The online stream writes once, but only to temporary bank dictionaries.
    with isolated_validation(model), _evaluation(model.memory):
        restore_memory_runtime(model, runtime, device)
        return model.memory(backbone_output, backbone_inputs["input_ids"], sample["memory_context"])


def forward_state_only(model, processor, state_head, sample, *, noise_seed=0):
    """Replay one exported sample and return loss, logits, targets, and target_mask.

    Call inside the DDP-managed forward. The caller owns parameter freezing,
    optimization and checkpointing. Use the checkpoint's eval processor and
    the same device/precision context as the policy. The backbone runs without
    gradients; archived bank tensors are detached, but fusion and DiT are not.
    """
    head = model.action_head
    if model.memory is None or head.memory_fusion_mode == "legacy":
        raise ValueError("Task40 state-only training requires Route A or Route B memory fusion")
    inputs = processor_inputs(processor, sample)
    backbone_inputs, action_inputs = model.prepare_input(inputs)
    with _evaluation(model.backbone), torch.no_grad():
        backbone_output = model.backbone(backbone_inputs)
    backbone_output = _replay_memory(model, sample, backbone_inputs, backbone_output)

    device = backbone_output["backbone_features"].device
    cuda_devices = [device.index] if device.type == "cuda" else []
    # Use real denoising tokens with independent, repeatable noise. Other
    # training RNG streams and the head's modes survive this auxiliary forward.
    with _evaluation(head), torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(noise_seed)
        if cuda_devices:
            torch.cuda.default_generators[device.index].manual_seed(noise_seed)
        features = head._encode_features(backbone_output, action_inputs)
        output = head.get_action_with_features.__wrapped__(
            head, backbone_features=features.backbone_features, state_features=features.state_features,
            embodiment_id=action_inputs.embodiment_id, backbone_output=backbone_output,
        )

    logits = state_head(output["fusion_action_features"])
    targets = sample["targets"].to(device=logits.device, dtype=torch.float32).unsqueeze(0)
    target_mask = sample["target_mask"].to(device=logits.device, dtype=torch.bool).unsqueeze(0)
    return dict(loss=masked_state_loss(logits, targets, target_mask), logits=logits,
                targets=targets, target_mask=target_mask)
