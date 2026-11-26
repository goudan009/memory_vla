"""Report paired Task40 forward evidence from inside the workbench DDP preflight."""

from contextlib import ExitStack
from unittest.mock import patch

import torch

from .first_training import cache_sequence, restore, seeded, slice_features, snapshot
from .task21_formal import atomic_json


@torch.no_grad()
def reload_forward(model, processor, dataset, group_indices, output):
    """Exp13 checkpoint check: one causal group, no history interventions."""
    model.eval()
    was_training = processor.training
    processor.eval()
    try:
        target = cache_sequence(model, processor, dataset, group_indices[0], 4)
        model.memory.reset()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for index, timestep in enumerate(target["timesteps"]):
                conditioned = model.memory(slice_features(target["features"], index, index + 1),
                    target["input_ids"][index:index + 1].cuda(),
                    dict(mode="stream", episode_keys=["reload-check"], timesteps=[timestep]))
            inputs = slice_features(target["actions"], index, index + 1)
            seeded(42)
            loss = model.action_head(conditioned, inputs)["loss"]
            seeded(42)
            prediction = model.action_head.get_action(conditioned, inputs)["action_pred"]
        if not torch.isfinite(loss) or not torch.isfinite(prediction).all():
            raise RuntimeError("Non-finite checkpoint verification forward")
        atomic_json(output, dict(episode=target["episode"], frame_indices=target["frame_indices"],
            timesteps=target["timesteps"], seed=42, action_shape=list(prediction.shape),
            finite=True, purpose="checkpoint reload verification"))
        return prediction.float().cpu()
    finally:
        model.memory.reset()
        processor.train() if was_training else processor.eval()


def difference(left, right):
    delta = (left.float() - right.float()).abs()
    return dict(mae=float(delta.mean()), maximum=float(delta.max()),
                changed_fraction=float((delta > 1e-6).float().mean()))


def context_intervention(head, original, *, native_text=False, native_image=False):
    """Replace selected DiT cross-attention inputs for a diagnostic forward."""
    context = head.vlln(original["backbone_features"])
    image = original["image_mask"]
    valid = original["backbone_attention_mask"]
    hooks = ExitStack()
    for index, block in enumerate(head.model.transformer_blocks[::2]):
        text_block = index % head.model.attend_text_every_n_blocks == 0
        if not (native_text if text_block else native_image):
            continue
        mask = valid & (~image if text_block else image)

        def replace_context(_, args, kwargs, mask=mask):
            return args, dict(kwargs, encoder_hidden_states=context, encoder_attention_mask=mask)

        handle = block.register_forward_pre_hook(replace_context, with_kwargs=True)
        hooks.callback(handle.remove)
    return hooks


@torch.no_grad()
def forward_panel(model, processor, dataset, group_indices, output):
    model.eval()
    was_training = processor.training
    processor.eval()
    try:
        target, donor = (cache_sequence(model, processor, dataset, index, 4) for index in group_indices)
    finally:
        if was_training:
            processor.train()
    assert target["episode"] != donor["episode"]
    memory, head = model.memory, model.action_head
    anchor = len(target["timesteps"]) - 1
    timestep = target["timesteps"][anchor]

    def read(sequence, index, time, key="anchor", diagnostics=False):
        return memory(slice_features(sequence["features"], index, index+1),
                      sequence["input_ids"][index:index+1].cuda(),
                      dict(mode="stream", episode_keys=[key], timesteps=[time],
                           memory_diagnostics=diagnostics))

    def fill(sequence, count):
        memory.reset()
        # Rebase the donor to the identical history time grid.
        for index in range(count):
            read(sequence, index, timestep-anchor+index)
        return snapshot(memory)

    inputs = slice_features(target["actions"], anchor, anchor+1)
    fixed = slice_features(target["features"], anchor, anchor+1)
    copies = {}
    outcomes = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        normal = fill(target, anchor)
        wrong = fill(donor, anchor)
        # 16 real observations fill capacity. A subsequent stream observation
        # is the first read with 16 slots and triggers an adjacent merge.
        read(donor, anchor, timestep)
        read(target, anchor, timestep+1, diagnostics=True)
        merge = memory.last_diagnostics["observations"][0]
        for bank in (memory.perceptual, memory.cognitive):
            assert len(bank.bank["anchor"]) == 16 and bank.merge_count["anchor"] == 1

        for label, state in (("official", None), ("normal", normal), ("empty", None),
                             ("wrong_perception", (wrong[0], normal[1])),
                             ("wrong_cognition", (normal[0], wrong[1])), ("wrong_both", wrong),
                             ("native_context", normal), ("no_cognition", normal)):
            memory.reset() if state is None else restore(memory, state)
            head.openvla_memory = label != "official"
            if head.openvla_memory:
                conditioned = read(target, anchor, timestep, diagnostics=True)
                diagnostics = memory.last_diagnostics["observations"][0]
                copies[label] = {key: conditioned[key].float().cpu()
                                 for key in ("memory_perceptual_tokens", "memory_cognitive_token")}
                copies[label]["projected_perception"] = head._perceptual_condition(conditioned).float().cpu()
                copies[label]["projected_cognition"] = head._fusion_kwargs(
                    conditioned, head.action_horizon)["cognitive_condition"].float().cpu()
                assert torch.equal(conditioned["backbone_features"], fixed["backbone_features"])
                assert torch.equal(conditioned["image_mask"], fixed["image_mask"])
                assert torch.equal(conditioned["backbone_attention_mask"], fixed["backbone_attention_mask"])
            else:
                conditioned, diagnostics = fixed, None
                copies[label] = {}
            with ExitStack() as intervention:
                if label == "native_context":
                    intervention.enter_context(context_intervention(
                        head, fixed, native_image=True))
                elif label == "no_cognition":
                    intervention.enter_context(patch.object(head, "_fusion_kwargs", return_value={}))
                seeded(42)
                loss = head(conditioned, inputs)
                assert torch.equal(loss["action_mask"], inputs["action_mask"])
                seeded(42)
                prediction = head.get_action(conditioned, inputs)["action_pred"]
            assert prediction.shape == inputs["action"].shape
            mask = inputs["action_mask"][:, :8].bool()
            if not torch.isfinite(prediction).all() or not torch.isfinite(loss["loss"]):
                raise RuntimeError(f"Non-finite preflight result for {label}")
            copies[label]["actions"] = prediction.float().cpu()
            outcomes[label] = dict(flow_loss=float(loss["loss"]),
                                   first8_action_mae=difference(prediction[:, :8][mask],
                                                               inputs["action"][:, :8][mask])["mae"],
                                   valid_first8_labels=int(mask.sum()), diagnostics=diagnostics)
    memory.reset()
    assert not any(bank.bank for bank in (memory.perceptual, memory.cognitive))
    for label, outcome in outcomes.items():
        if label != "normal":
            outcome["difference_vs_normal"] = {
                key: difference(copies[label][key], copies["normal"][key])
                for key in copies[label].keys() & copies["normal"].keys()
            }
    torch.save(copies, output.with_suffix(".pt"))
    atomic_json(output, dict(target_episode=target["episode"], donor_episode=donor["episode"],
                             frames=target["frame_indices"], merge=merge, seed=42,
                             shape={key: list(tensor.shape) for key, tensor in copies["normal"].items()},
                             results=outcomes))
    return copies["normal"]["actions"]
