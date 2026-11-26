import torch

from v3_groot_role_memory.evaluation import compare, numpy
from v3_groot_role_memory.config import ROLES


def test_bfloat16_tensor_converts_to_numpy_float32_without_value_change():
    value = torch.tensor([[0.5, -1.25]], dtype=torch.bfloat16)

    result = numpy(value)

    assert result.dtype.name == "float32"
    assert (result == value.float().numpy()).all()


def test_mechanism_comparison_aligns_shorter_v3_off_gate_trace():
    import numpy as np

    actions = np.zeros((1, 50, 4), dtype=np.float32)
    gates = np.zeros((1, 4, 50, 1), dtype=np.float32)
    normal = dict(actions=actions, message=np.zeros((1, 4, 8), dtype=np.float32),
                  velocity_delta=actions, velocity=actions, gates=gates,
                  loss_unreduced=actions)
    other = {key: value.copy() for key, value in normal.items()}
    other["gates"] = gates[:, :, :32].copy()
    reference = np.ones((1, 50, 4), dtype=np.float32)
    mask = np.ones_like(reference)
    maps = {role: [index] for index, role in enumerate(ROLES)}

    result = compare(normal, other, reference, mask, maps)

    assert result["gates_mae"] == 0.0
    assert result["roles"][ROLES[0]]["normal_reference_action_mae"] == 1.0
