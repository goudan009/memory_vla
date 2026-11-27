import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fault_diagnostics', ROOT/'src/v1_groot_memoryvla/policy/action_diagnostics.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_paired_rpc_restores_rng_and_steps_and_records_raw_values(tmp_path):
    class Policy:
        memory_state = None
        model = SimpleNamespace(action_head=SimpleNamespace(num_inference_timesteps=4),
                                parameters=lambda: iter([torch.zeros(1)]))

        def get_action(self, observation, options):
            # With identical RNG, these calls must see identical noise.
            raw = torch.randn(1, 2, 3)
            decoded = {'base': np.clip(raw.numpy(), -1, 1)}
            self.action_diagnostic_hook(observation, raw, decoded, options)
            return decoded, {}

    policy = Policy()
    recorder = module.ActionDiagnostics(policy, tmp_path)
    policy.action_diagnostic_hook = recorder.record
    before = torch.get_rng_state().clone()
    result = recorder.diagnose({'video': {}, 'state': {}, 'language': {}}, seed=42)
    assert torch.equal(torch.get_rng_state(), before)
    assert policy.model.action_head.num_inference_timesteps == 4
    np.testing.assert_array_equal(result['4']['normalized_action'], result['16']['normalized_action'])
    assert len(list(tmp_path.glob('request_*.npz'))) == 3
    assert np.max(np.abs(result['4']['normalized_action'])) > 1


def test_diagnostic_failure_restores_steps_and_rng(tmp_path):
    class Policy:
        memory_state = None
        model = SimpleNamespace(action_head=SimpleNamespace(num_inference_timesteps=8),
                                parameters=lambda: iter([torch.zeros(1)]))

        def get_action(self, *args, **kwargs):
            torch.randn(3)
            raise RuntimeError('model failure')

    policy = Policy()
    recorder = module.ActionDiagnostics(policy, tmp_path)
    before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match='model failure'):
        recorder.diagnose({}, steps=[16])
    assert policy.model.action_head.num_inference_timesteps == 8
    assert torch.equal(before, torch.get_rng_state())
