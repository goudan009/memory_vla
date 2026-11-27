"""Replay cached real EOS through both retrieval implementations (no VLM needed)."""
import argparse
import json
import math
from pathlib import Path
import time

import torch

from v1_groot_memoryvla.model.memory.bank import MemoryBank


def run(weights, rows, norm, device):
    bank = MemoryBank(rows[0]['eos'].numel(), capacity=64, qk_norm=norm).to(device).eval()
    bank.load_state_dict(weights, strict=True)
    projected, events, active = {}, [], {}
    block = bank.retrieval_blocks[0]
    handles = [getattr(block, name).register_forward_hook(
        lambda module, inputs, out, name=name: projected.update({name: out.detach().float()}))
        for name in ('q_proj', 'k_proj')]
    def collect(module, inputs, output):
        scores = projected['q_proj'] @ projected['k_proj'].transpose(-1, -2) / math.sqrt(inputs[0].shape[-1])
        if scores.shape[-1] < 8:
            return
        p = scores.softmax(-1).flatten()
        events.append(dict(episode=active['episode'], timestep=active['timestep'],
            oldest_probability=float(p[0]), max_probability=float(p.max()),
            normalized_entropy=float(-(p * p.clamp_min(1e-30).log()).sum() / math.log(p.numel()))))
    handles.append(block.register_forward_hook(collect))
    gradients = []
    for row in rows:
        active.update(row)
        x = row['eos'].to(device).float().reshape(1, 1, -1)
        # Autocast matches training: FP32 parameters, BF16 projections on CUDA.
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            out = bank.read_fuse_write(x, row['episode'], row['timestep'])
            # Numerical gradient probe, NOT an action-training objective.
            loss = (out.float() - torch.roll(x, 1, -1)).square().mean()
        if out.requires_grad and row['timestep'] in (8, 32, 64, 127):
            loss.backward()
            gradients.append(float(block.q_proj.weight.grad.norm()))
            bank.zero_grad(set_to_none=True)
        assert torch.isfinite(out).all(), 'Nonfinite memory output'
    for handle in handles:
        handle.remove()
    assert events and gradients, 'Replay must include history >=8 and gradient probe timesteps'
    return dict(queries=len(events), oldest_probability_mean=sum(e['oldest_probability'] for e in events)/len(events),
        normalized_entropy_mean=sum(e['normalized_entropy'] for e in events)/len(events),
        saturated_fraction=sum(e['max_probability'] > .999 for e in events)/len(events),
        q_gradient_median=float(torch.tensor(gradients).median()), trace=events)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, required=True, help='Compact cognitive bank state_dict .pt')
    p.add_argument('--replay', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    torch.set_num_threads(2)
    started = time.monotonic()
    weights = torch.load(args.weights, map_location='cpu', weights_only=True)
    rows = torch.load(args.replay, map_location='cpu', weights_only=True)['eos']
    results = {norm: run(weights, rows, norm, torch.device(args.device)) for norm in ('none', 'layernorm')}
    fixed = results['layernorm']
    # Broad anti-collapse gate on this multi-episode diagnostic fixture. High
    # attention on an early cue can be correct; it is not banned per query.
    passed = fixed['saturated_fraction'] < .5 and fixed['q_gradient_median'] > 1e-8
    report = dict(scope='Real EOS retrieval/gradient regression; not policy success',
        weights=str(args.weights.resolve()), replay=str(args.replay.resolve()),
        elapsed_seconds=time.monotonic()-started, passed=passed, results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: {n: v for n, v in r.items() if n != 'trace'} for k, r in results.items()}, indent=2))
    if not passed:
        raise SystemExit('Retrieval collapse gate failed; inspect report before long training')


if __name__ == '__main__':
    main()
