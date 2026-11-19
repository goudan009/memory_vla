"""Trainability, real autograd, LoRA checkpoint and DDP contracts on small models."""
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from v1_groot_memoryvla.experiment.finetuning import apply_finetuning, components
from v1_groot_memoryvla.experiment.finetuning_options import resolve_finetuning
from v1_groot_memoryvla.experiment.first_training import slice_features, cache_sequence
from v1_groot_memoryvla.experiment.task21_ddp import TrainableStep, backward_windows
from v1_groot_memoryvla.model.modules.lora import LoRALinear, install_lora


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.model = nn.Module()
        self.model.language_model.model.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.model.vision_model = nn.Sequential(nn.Linear(4, 4))
        self.model.mlp1 = nn.Sequential(nn.Linear(4, 4))

    def forward(self, inputs):
        x = self.model.mlp1(self.model.vision_model(inputs['pixels']))
        for layer in self.model.language_model.model.layers:
            x = layer(x).tanh()
        return {'backbone_features': x.unsqueeze(1)}


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(add_pos_embed=False)
        self.state_dropout_prob = 0
        self.state_encoder = nn.Linear(4, 4)
        self.action_encoder = nn.Linear(4, 4)
        self.action_decoder = nn.Linear(4, 2)
        self.model = nn.Sequential(nn.Linear(4, 4))
        self.vlln = nn.LayerNorm(4)

    def forward(self, features, actions):
        x = self.vlln(features['backbone_features'].mean(1))
        x = self.action_decoder(self.model(self.state_encoder(x) + self.action_encoder(x)))
        return {'loss': (x - actions['target']).square().mean()}


class TinyModel(nn.Module):
    def __init__(self, spec=None):
        super().__init__()
        self.config = SimpleNamespace()
        self.backbone, self.action_head, self.memory = TinyBackbone(), TinyHead(), None
        if spec:
            self.config.finetuning_lora = spec
            install_lora(self, spec)


def args(**kw):
    return SimpleNamespace(arm='baseline', **kw)


@pytest.mark.parametrize('mode', ['action_full', 'full', 'custom', 'lora'])
def test_component_defaults_and_live_autograd(mode):
    torch.manual_seed(4)
    model = TinyModel()
    summary = apply_finetuning(model, args(fine_tuning_mode=mode))
    model.train()
    core = TrainableStep(model)
    inputs = {'pixels': torch.randn(3, 4)}
    if summary['frozen_eagle']:
        with torch.no_grad():
            inputs = model.backbone(inputs)
    loss = core(inputs, None, None, {'target': torch.randn(3, 2)}, 52)
    loss.backward()
    assert any(p.grad is not None for p in core.parameters())
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    if mode == 'lora':
        assert all(not m.weight.requires_grad for m in model.modules() if isinstance(m, LoRALinear))
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if n.endswith('lora_B'))
    assert summary['frozen_eagle'] == (mode in ('action_full', 'custom'))


def test_explicit_freeze_and_last_llm_layers():
    model = TinyModel()
    summary = apply_finetuning(model, args(fine_tuning_mode='full', freeze_vision=True,
        freeze_action_projector=True, llm_last_n_layers=1))
    assert summary['components']['vision']['trainable'] == 0
    assert summary['components']['action_projector']['trainable'] == 0
    assert not any(p.requires_grad for p in model.backbone.model.language_model.model.layers[0].parameters())
    assert all(p.requires_grad for p in model.backbone.model.language_model.model.layers[1].parameters())


def test_lora_initially_identity_and_checkpoint_has_exact_reload(tmp_path):
    torch.manual_seed(42)
    before = TinyModel().eval()
    model = deepcopy(before)
    apply_finetuning(model, args(fine_tuning_mode='lora', lora_rank=2, lora_alpha=4.))
    model.eval()
    x = {'pixels': torch.randn(4, 4)}
    target = {'target': torch.randn(4, 2)}
    original = before.action_head(before.backbone(x), target)['loss']
    loss = model.action_head(model.backbone(x), target)['loss']
    torch.testing.assert_close(loss, original, rtol=0, atol=0)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    loss.backward(); optimizer.step()
    for name, value in frozen.items():
        torch.testing.assert_close(dict(model.named_parameters())[name], value, rtol=0, atol=0)
    spec = model.config.finetuning_lora
    torch.save(model.state_dict(), tmp_path/'weights.pt')
    (tmp_path/'lora.json').write_text(json.dumps(spec))
    reloaded = TinyModel(json.loads((tmp_path/'lora.json').read_text())).eval()
    reloaded.load_state_dict(torch.load(tmp_path/'weights.pt', weights_only=True), strict=True)
    torch.testing.assert_close(reloaded.action_head(reloaded.backbone(x), target)['loss'],
                               model.action_head(model.backbone(x), target)['loss'], rtol=0, atol=0)
    # Reapplying identical settings must keep adapted weights, not reinitialize them.
    apply_finetuning(reloaded, args(fine_tuning_mode='lora', lora_rank=2, lora_alpha=4.))
    for name, value in model.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[name], value, rtol=0, atol=0)


def test_flattened_camera_inputs_follow_sample_slices():
    raw = dict(input_ids=torch.zeros(4, 6, dtype=torch.long), attention_mask=torch.ones(4, 6),
               pixel_values=[torch.tensor([i]) for i in range(12)])
    selected = slice_features(raw, 1, 3, 'cpu')
    assert len(selected['input_ids']) == 2
    assert [int(x) for x in selected['pixel_values']] == list(range(3, 9))


@pytest.mark.parametrize('mode', ['full', 'lora'])
def test_checkpoint_loader_preserves_fp32_trainable_backbone_weights(monkeypatch, mode):
    from v1_groot_memoryvla.experiment import task21_diagnostic as diagnostic
    model = TinyModel()
    apply_finetuning(model, args(fine_tuning_mode=mode))
    with torch.no_grad():
        for parameter in model.backbone.parameters():
            if parameter.requires_grad:
                parameter.fill_(.123456789)
    expected = {n: p.detach().clone() for n,p in model.backbone.named_parameters() if p.requires_grad}
    monkeypatch.setattr(diagnostic.Gr00tN1d6, 'from_pretrained', lambda *a, **kw:
        (deepcopy(model), dict(missing_keys=[], unexpected_keys=[], mismatched_keys=[])))
    monkeypatch.setattr(nn.Module, 'cuda', lambda self, *a, **kw: self)
    restored = diagnostic.load_model('/unused', False, 42, finetuning=args(fine_tuning_mode=mode))
    for name, parameter in restored.backbone.named_parameters():
        if name in expected:
            assert parameter.dtype == torch.float32
            torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
        else:
            assert parameter.dtype == torch.bfloat16


def test_trainable_backbone_cache_keeps_inputs_instead_of_detached_features(monkeypatch):
    class Dataset:
        processor = SimpleNamespace(collator=lambda steps: {'inputs': {
            'input_ids': torch.zeros(2, 3, dtype=torch.long), 'attention_mask': torch.ones(2, 3),
            'pixel_values': [torch.zeros(3, 4, 4)] * 6, 'state': torch.zeros(2, 1, 4),
            'action': torch.zeros(2, 2, 2), 'action_mask': torch.ones(2, 2, 2),
            'embodiment_id': torch.zeros(2, dtype=torch.long)}})
        def __getitem__(self, index):
            return dict(steps=[{}, {}], timesteps=[0, 1])
    dataset = Dataset()
    model = TinyModel().train()
    model.prepare_input = lambda batch: (batch, batch)
    model.backbone.forward = lambda *a: pytest.fail('Eagle must run inside DDP, not during input caching')
    result = cache_sequence(model, dataset.processor, dataset, 0, 1)
    assert set(result['features']) == {'input_ids', 'attention_mask', 'pixel_values'}
    assert len(result['features']['pixel_values']) == 6


@pytest.mark.parametrize('bad', [dict(fine_tuning_mode='wrong'), dict(lora_rank=0),
    dict(lora_alpha=float('nan')), dict(lora_dropout=1), dict(freeze_llm='false'),
    dict(llm_last_n_layers=-1), dict(lora_targets='action_vlln'),
    dict(fine_tuning_mode='lora', freeze_llm=True, freeze_action_dit=True)])
def test_invalid_options_rejected(bad):
    with pytest.raises(ValueError):
        resolve_finetuning(bad)


def _distributed_rank(rank, temporary, mode):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + str(Path(temporary)/'rendezvous'),
                            rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        torch.manual_seed(42)
        model = TinyModel()
        apply_finetuning(model, args(fine_tuning_mode=mode, freeze_vision=True, freeze_vision_projector=True))
        core = TrainableStep(model)
        ddp = DistributedDataParallel(core, find_unused_parameters=True)
        optimizer = torch.optim.SGD([p for p in core.parameters() if p.requires_grad], lr=.01)
        for i in range(2):
            optimizer.zero_grad(set_to_none=True)
            # Raw backbone tensors, with unequal Episode tail lengths per rank.
            torch.manual_seed(100 + rank + i * 10)
            n = 3 if rank == 0 else 1
            seq = dict(features={'pixels': torch.randn(n, 4)}, input_ids=torch.zeros(n, 2, dtype=torch.long),
                actions={'target': torch.randn(n, 2)}, timesteps=list(range(n)), episode_key=str(rank),
                episode_start=True, episode_end=True)
            backward_windows(ddp, [seq], [1000], 2, microbatch=2, device='cpu')
            optimizer.step()
        torch.save(model.state_dict(), Path(temporary)/f'rank{rank}.pt')
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('mode', ['full', 'lora'])
def test_backbone_training_ddp_syncs_across_unequal_tails(tmp_path, mode):
    mp.start_processes(_distributed_rank, args=(str(tmp_path), mode), nprocs=2, join=True, start_method='spawn')
    left, right = [torch.load(tmp_path/f'rank{rank}.pt', weights_only=True) for rank in range(2)]
    for name in left:
        torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
