"""Explicit trainable mask for the single joint V2 method."""


def configure_trainable_parameters(model):
    """Freeze all networks and original injection scale; train cognition only.

    The cognitive bank includes its existing Q/K/V, FFN, norms, time encoder,
    fusion gate, plus the new StateHead and zero-initialized state projections.
    The perceptual bank/encoder/projection and original residual scalar stay fixed.
    The frozen action expert must still run WITH autograd during training.
    """
    model.requires_grad_(False)
    model.memory.cognitive.requires_grad_(True)
    return parameter_manifest(model)


def parameter_manifest(model):
    return [dict(name=name, shape=list(p.shape), numel=p.numel(), trainable=p.requires_grad)
            for name, p in model.named_parameters()]
