"""Precision regression through the policy constructor, without HF/GPU weights."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


class Head(torch.nn.Linear):
    def set_trainable_parameters(self, *flags):
        self.requires_grad_(all(flags))


class FakeBase(torch.nn.Module):
    info = {}

    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(1, 1)
        self.action_head = Head(1, 1)
        with torch.no_grad():
            self.action_head.weight.fill_(.123456791)

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return cls(), cls.info


def policy_constructor():
    """Run the actual V2.1 constructor; fake only the external dependencies."""
    path = Path(__file__).resolve().parents[1] / 'policy.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Policy')
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__']
    class V1Policy:
        def __init__(self, *a, **kw):
            self.model = FakeBase().bfloat16()
    class BasePolicy:
        def __init__(self, strict=True): self.strict = strict
    class Processor:
        @classmethod
        def from_pretrained(cls, path): return cls()
        def eval(self): return self
        def get_modality_configs(self):
            return {'behavior_r1_pro': {'language': SimpleNamespace(modality_keys=['task'], delta_indices=[0])}}
        collator = staticmethod(lambda x: x)
    class Sidecar:
        @staticmethod
        def load_sidecar(base, path, device):
            return SimpleNamespace(base=base, memory=object())
    def load_base(*args, **kw):
        from v2_1_groot_memoryvla.loading import load_base as actual
        return actual(*args, **kw)
    env = dict(Path=Path, Gr00tPolicy=V1Policy, BasePolicy=BasePolicy,
               Integrated=Sidecar, MemoryState=lambda x: x, check_scale=float,
               Gr00tN1d6Processor=Processor, load_base=load_base)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), 'exec'), env)
    return env['Policy']


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'v1_groot_memoryvla.model.gr00t_n1d6.gr00t_n1d6',
                        SimpleNamespace(Gr00tN1d6=FakeBase))
    monkeypatch.setattr(FakeBase, 'info', {})
    (tmp_path / 'config.json').write_text(json.dumps({'memory_enabled': False}))
    (tmp_path / 'v2_1.json').write_text('{}')
    return tmp_path


def test_service_preserves_fp32_head_weights(checkpoint):
    policy = policy_constructor()(SimpleNamespace(value='behavior_r1_pro'), checkpoint, device='cpu')
    base = policy.model.base
    assert base.action_head.weight.dtype == torch.float32
    assert torch.equal(base.action_head.weight, torch.full_like(base.action_head.weight, .123456791))
    assert base.backbone.weight.dtype == torch.bfloat16


@pytest.mark.parametrize('key', ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'])
def test_shared_loader_rejects_checkpoint_mismatches(checkpoint, monkeypatch, key):
    from v2_1_groot_memoryvla.loading import load_base
    monkeypatch.setattr(FakeBase, 'info', {key: ['bad.weight']})
    with pytest.raises(ValueError, match='mismatch'):
        load_base(checkpoint, 'memory', device='cpu')


@pytest.mark.parametrize('stage', ['stage1', 'continued', 'memory'])
def test_shared_loader_stage_freezing(checkpoint, stage):
    from v2_1_groot_memoryvla.loading import load_base
    base = load_base(checkpoint, stage, device='cpu')
    assert base.action_head.weight.dtype == torch.float32
    assert base.action_head.weight.requires_grad == (stage != 'memory')
    assert base.action_head.training == (stage != 'memory')
    assert not base.backbone.weight.requires_grad
    assert not base.backbone.training
