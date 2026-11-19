"""CPU integration contracts using the local GR00T model/head inference methods.

Extract only those methods to avoid loading Eagle/Isaac/GPU dependencies. Tiny
backbone/DiT layers exercise the native inference control flow, not checkpoint
quality or full-model numerical equivalence.
"""

import ast
from collections import UserDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vlamemory import MemoryConfig
from vlamemory.integrations.groot import Gr00tMemoryModel, Gr00tMemoryPolicy


class Features(UserDict):
    def __init__(self, data=None, **kwargs):
        super().__init__(data, **kwargs)

    def __getattr__(self, key):
        try:
            return self.data[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _native_methods(class_name, names):
    root = Path(__file__).resolve().parents[1]
    path = root / "third_party/Isaac-GR00T/gr00t/model/gr00t_n1d6/gr00t_n1d6.py"
    source = ast.parse(path.read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in methods} == set(names)
    namespace = {"torch": torch, "BatchFeature": Features}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


class EmbodimentLinear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)

    def forward(self, x, embodiment):
        return self.linear(x) * (embodiment[:, None, None] + 1)


class ActionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 8)

    def forward(self, actions, timestep, embodiment):
        return self.linear(actions)


class TinyDiT(nn.Module):
    def forward(self, hidden_states, encoder_hidden_states, **kwargs):
        return hidden_states + encoder_hidden_states.mean(dim=1, keepdim=True)


class NativeHead(nn.Module):
    locals().update(_native_methods("Gr00tN1d6ActionHead", ["process_backbone_output", "_encode_features",
                                                           "get_action_with_features", "get_action"]))

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(backbone_embedding_dim=8, action_horizon=2,
                                      add_pos_embed=False, use_alternate_vl_dit=True)
        self.action_dim, self.action_horizon = 5, 2
        self.num_inference_timesteps, self.num_timestep_buckets = 2, 1000
        self.vlln = nn.LayerNorm(8)
        self.state_encoder = EmbodimentLinear(3, 8)
        self.action_encoder = ActionEncoder()
        self.model = TinyDiT()
        self.action_decoder = EmbodimentLinear(8, 5)


class TinyBackbone(nn.Module):
    def forward(self, inputs):
        self.last_output = Features({"backbone_features": inputs["features"].clone(),
                                     "backbone_attention_mask": inputs["mask"],
                                     "image_mask": inputs["image_mask"]})
        return self.last_output


class NativeModel(nn.Module):
    locals().update(_native_methods("Gr00tN1d6", ["get_action"]))

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()
        self.action_head = NativeHead()
        self.backbone = TinyBackbone()

    def prepare_input(self, inputs):
        return inputs, Features({"state": inputs["state"], "embodiment_id": inputs["embodiment_id"]})


class NativePolicy:
    def __init__(self):
        self.model = NativeModel().eval()
        self.fail_decode = False

    def get_action(self, observation, options=None):
        result = self.model.get_action(inputs=observation)
        if self.fail_decode:
            raise RuntimeError("decode failed")
        return result, {"native": True}

    def reset(self, options=None):
        return {"native_reset": True}

    def get_modality_config(self):
        return {"native": "three cameras"}


def _inputs(batch=1):
    return {"features": torch.arange(batch * 5 * 8).reshape(batch, 5, 8).float() / 100,
            "mask": torch.tensor([[True, True, True, False, False]]).expand(batch, -1),
            "image_mask": torch.ones(batch, 5, dtype=torch.bool),
            "state": torch.ones(batch, 1, 3), "embodiment_id": torch.arange(batch)}


def _options(session="gpu0", envs=(0,), episodes=("episode0",)):
    return {"session_id": session, "env_ids": list(envs), "episode_ids": list(episodes)}


def _config(kind="action_conditioned"):
    return MemoryConfig(memory_type=kind, num_action_groups=2, num_attention_heads=2,
                        retrieval_layers=1, use_timestep_pe=False)


@pytest.mark.parametrize("kind", ["none", "uniform", "action_conditioned"])
def test_native_action_path_exact_at_initialization_and_does_not_consume_rng(kind):
    native = NativeModel().eval()
    rng = torch.get_rng_state().clone()
    wrapped = Gr00tMemoryModel(native, _config(kind))
    assert torch.equal(torch.get_rng_state(), rng)
    wrapped.reset(_options())
    inputs = _inputs()
    original_features = inputs["features"].clone()
    for t in range(3):
        torch.manual_seed(41 + t)
        expected = native.get_action(inputs)
        torch.manual_seed(41 + t)
        actual = wrapped.get_action(inputs, memory_options=_options())
        assert torch.equal(expected["action_pred"], actual["action_pred"])
        assert actual["action_pred"].shape == (1, 2, 5)
    assert torch.equal(inputs["features"], original_features)
    assert native.backbone.last_output["backbone_attention_mask"] is inputs["mask"]
    assert native.backbone.last_output["image_mask"] is inputs["image_mask"]


def test_batched_envs_and_two_sessions_have_independent_histories_and_targeted_reset():
    policy = Gr00tMemoryPolicy(NativePolicy(), _config())
    a = _options(envs=(0, 1), episodes=("a0", "a1"))
    b = _options(session="gpu1")
    policy.reset(a)
    policy.reset(b)
    policy.get_action(_inputs(2), a)
    policy.get_action(_inputs(), b)
    model = policy.model
    a0, a1 = [model.sessions.episode_id("gpu0", i) for i in (0, 1)]
    b0 = model.sessions.episode_id("gpu1", 0)
    bank = model.memory.memory_bank
    # Padding excluded; observations are written once after native action decode.
    expected = _inputs(2)["features"][:, :3].mean(dim=1)
    for row, key in enumerate((a0, a1)):
        torch.testing.assert_close(bank.perceptual_bank.read(key)[0][1], expected[row:row + 1])
    policy.reset(_options(envs=(0,), episodes=("next",)))
    assert not bank.perceptual_bank.read(a0)
    assert len(bank.perceptual_bank.read(a1)) == 1
    assert len(bank.perceptual_bank.read(b0)) == 1
    assert not bank.cognitive_bank.bank
    assert all("bank" not in name for name in bank.state_dict())


def test_request_ownership_and_failed_decode_do_not_write_or_advance_memory():
    native = NativePolicy()
    policy = Gr00tMemoryPolicy(native, _config())
    opts = _options()
    with pytest.raises(ValueError, match="not active"):
        policy.get_action(_inputs(), opts)
    policy.reset(opts)
    native.fail_decode = True
    with pytest.raises(RuntimeError, match="decode failed"):
        policy.get_action(_inputs(), opts)
    model = policy.model
    key = model.sessions.episode_id("gpu0", 0)
    assert model._timesteps[key] == 0
    assert not model.memory.memory_bank.perceptual_bank.read(key)
    native.fail_decode = False
    policy.get_action(_inputs(), opts)
    assert model._timesteps[key] == 1
    with pytest.raises(ValueError, match="not active"):
        policy.get_action(_inputs(), _options(episodes=("different",)))
    with pytest.raises(ValueError, match="batch size"):
        policy.get_action(_inputs(2), opts)


def test_nonzero_memory_changes_only_history_case_and_keeps_padding_features():
    native = NativeModel().eval()
    wrapped = Gr00tMemoryModel(native, _config())
    opts, inputs = _options(), _inputs()
    wrapped.reset(opts)
    with torch.no_grad():
        wrapped.memory.residual.weight.copy_(torch.eye(8))
    torch.manual_seed(20)
    expected = native.get_action(inputs)["action_pred"]
    torch.manual_seed(20)
    first = wrapped.get_action(inputs, memory_options=opts)["action_pred"]
    assert torch.equal(first, expected)
    torch.manual_seed(20)
    second = wrapped.get_action(inputs, memory_options=opts)["action_pred"]
    assert not torch.equal(second, expected)
    # The copy passed to the native head does not overwrite the backbone result.
    assert torch.equal(native.backbone.last_output["backbone_features"], inputs["features"])
    wrapped.reset(opts)
    torch.manual_seed(20)
    after_reset = wrapped.get_action(inputs, memory_options=opts)["action_pred"]
    assert torch.equal(after_reset, expected)


def test_config_rejects_openpi_mode_or_hidden_width_mismatch():
    with pytest.raises(ValueError, match="OpenPI"):
        MemoryConfig(injection_mode="prefix-context")
    with pytest.raises(ValueError, match="backbone_embedding_dim"):
        Gr00tMemoryModel(NativeModel(), MemoryConfig(memory_type="uniform", model_dim=16))


def test_native_sim_wrapper_can_read_original_modality_metadata():
    class PolicyWrapper:
        def __init__(self, policy, strict=True):
            self.policy, self.strict = policy, strict

    path = Path(__file__).resolve().parents[1] / "third_party/Isaac-GR00T/gr00t/policy/gr00t_policy.py"
    cls = next(node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "Gr00tSimPolicyWrapper")
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name in ("__init__", "get_modality_config")]
    source = ast.parse("from __future__ import annotations")
    source.body.append(cls)
    namespace = {"PolicyWrapper": PolicyWrapper}
    exec(compile(source, str(path), "exec"), namespace)

    native = NativePolicy()
    native.modality_configs = {"language": SimpleNamespace(delta_indices=[0])}
    wrapped = Gr00tMemoryPolicy(native, _config())
    sim = namespace["Gr00tSimPolicyWrapper"](wrapped)
    assert sim.policy.modality_configs is native.modality_configs
    assert sim.get_modality_config() == native.get_modality_config()


def test_native_bfloat16_dtype_and_masked_residual_are_preserved():
    native = NativeModel().eval().to(dtype=torch.bfloat16)
    wrapped = Gr00tMemoryModel(native, _config())
    inputs = {key: value.to(torch.bfloat16) if value.is_floating_point() else value
              for key, value in _inputs().items()}
    opts = _options()
    wrapped.reset(opts)
    wrapped.get_action(inputs, memory_options=opts)
    key = wrapped.sessions.episode_id("gpu0", 0)
    with torch.no_grad():
        wrapped.memory.residual.bias.fill_(1)
    actual = wrapped.memory(inputs["features"], key, 1, inputs["mask"])
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual[:, 3:], inputs["features"][:, 3:])
    assert not torch.equal(actual[:, :3], inputs["features"][:, :3])
    wrapped.get_action(inputs, memory_options=opts)
    assert all(feature.dtype == torch.bfloat16 for _, feature in
               wrapped.memory.memory_bank.perceptual_bank.read(key))


def test_core_import_has_no_jax_flax_openpi_or_vendor_import():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-c", "import sys, vlamemory; "
                             "from vlamemory.integrations.groot import Gr00tMemoryPolicy; "
                             "assert not any(k in sys.modules for k in ('jax', 'flax', 'openpi', 'gr00t'))"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
