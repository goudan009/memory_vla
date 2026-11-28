"""Real-checkpoint acceptance before the Task40 fusion optimization loop."""

from contextlib import contextmanager

import torch

from .first_training import cache_sequence, restore, seeded, slice_features, snapshot


@contextmanager
def bypass_fusion(model):
    head = model.action_head
    mode = head.memory_fusion_mode
    hook = None
    if mode == "route_a":
        hook = model.memory.cognitive_fusion.register_forward_hook(
            lambda module, inputs, output: torch.zeros_like(output))
    else:
        head.memory_fusion_mode = "legacy"
    try:
        yield
    finally:
        if hook is not None:
            hook.remove()
        head.memory_fusion_mode = mode


def tensor_mae(a, b):
    return float((a.detach().float() - b.detach().float()).abs().mean())


def check_fusion(model, processor, dataset, args):
    """Use one chronologically ordered, action-labeled real demonstration."""
    model.eval()
    # The first ordered window must provide at least two prior memory slots.
    index = next(indices[0] for indices in dataset.by_episode.values() if indices)
    seq = cache_sequence(model, processor, dataset, index, args.microbatch,
                         augmentation_seed=args.seed)
    adapter, head = model.memory, model.action_head
    key = "exp10-real-check"
    seed = args.seed + 19007
    target = 3
    saved_memory = snapshot(adapter)
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    saved_parameters = {name: parameter.detach().clone() for name, parameter in trainable.items()}

    def features(i):
        return adapter(slice_features(seq["features"], i, i + 1),
                       seq["input_ids"][i:i + 1].cuda(),
                       {"mode": "stream", "episode_keys": [key], "timesteps": [seq["timesteps"][i]]})

    def replay(history):
        restore(adapter, history)
        return features(target)

    def predict(history):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            fused = replay(history)
            seeded(seed)
            result = head.get_action(fused, actions)
            return fused["backbone_features"], result["action_pred"]

    def backward(history):
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fused = replay(history)
            seeded(seed)
            loss = head(fused, actions)["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError("Task40 fusion preflight action loss is non-finite")
        loss.backward()
        grads = {name: float(parameter.grad.detach().float().norm()) if parameter.grad is not None else 0.
                 for name, parameter in trainable.items()}
        if not all(torch.isfinite(p.grad).all() for p in trainable.values() if p.grad is not None):
            raise RuntimeError("Task40 fusion preflight gradient is non-finite")
        return float(loss.detach()), grads

    try:
        adapter.reset()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(target):
                features(i)
        history = snapshot(adapter)
        cognition = history[1][0][key]
        if len(cognition) < 2:
            raise RuntimeError("Task40 fusion preflight requires two causal cognitive slots")
        no_cognition = [history[0], ({}, history[1][1], history[1][2])]
        actions = slice_features(seq["actions"], target, target + 1)
        with bypass_fusion(model):
            original_features, original_action = predict(history)
            _, original_empty_action = predict(no_cognition)
        fused_features, fused_action = predict(history)
        if not torch.equal(original_features, fused_features) or not torch.equal(original_action, fused_action):
            raise RuntimeError("Zero-initialized Task40 fusion does not reproduce the baseline")
        first_loss, first_grads = backward(history)
        projection = "value.weight" if head.memory_fusion_mode == "route_a" else "output.weight"
        name, = (name for name in trainable if name.endswith(".cognitive_fusion." + projection))
        if first_grads[name] <= 0:
            raise RuntimeError(f"Task40 action loss has no gradient at {name}")
        diagnostic_step = torch.optim.AdamW(trainable.values(), lr=args.learning_rate, weight_decay=0.)
        diagnostic_step.step()
        delta = tensor_mae(trainable[name], saved_parameters[name])
        if delta == 0.:
            raise RuntimeError(f"Task40 action loss did not update {name}")
        second_loss, second_grads = backward(history)
        required = ("gate.weight",) if head.memory_fusion_mode == "route_a" else (
            "query.weight", "key.weight", "value.weight")
        for suffix in required:
            if not any(value > 0 for k, value in second_grads.items() if k.endswith(suffix)):
                raise RuntimeError(f"Task40 action loss did not reach {suffix} after one update")
        _, updated_action = predict(history)
        _, updated_empty_action = predict(no_cognition)
        return {
            "episode": seq["episode"], "decision_in_window": target, "rng_seed": seed,
            "cognitive_history_slots": len(cognition), "baseline_features_exact": True,
            "baseline_action_exact": True, "initial_loss": first_loss, "after_update_loss": second_loss,
            "first_action_grad_norms": first_grads, "second_action_grad_norms": second_grads,
            "diagnostic_parameter_update_mae": delta,
            "diagnostic_normal_action_shift": tensor_mae(updated_action, original_action),
            "diagnostic_empty_cognitive_action_shift": tensor_mae(updated_empty_action, original_empty_action),
            "diagnostic_normal_vs_empty_action_mae": tensor_mae(updated_action, updated_empty_action),
            "diagnostic_update_reverted": True,
        }
    finally:
        with torch.no_grad():
            for name, parameter in trainable.items():
                parameter.copy_(saved_parameters[name])
        model.zero_grad(set_to_none=True)
        restore(adapter, saved_memory)
