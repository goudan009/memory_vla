"""Bounded real-feature/action-label training smoke; no simulator or live service.

The supplied replay must come from this checkpoint's frozen backbone and
processor. The bundled diagnostic cache uses the GR00T-N1.6-3B backbone.
Sparse cached decisions test plumbing/gradients, not full training quality.
"""
import argparse
import json
from pathlib import Path
import time

import torch
from transformers.feature_extraction_utils import BatchFeature

from v1_groot_memoryvla.experiment.task21_diagnostic import load_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--replay', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--arm', choices=['baseline', 'direct_empty', 'memory'], default='memory')
    p.add_argument('--qk-norm', choices=['none', 'layernorm'], default='layernorm')
    args = p.parse_args()
    if args.steps < 1:
        p.error('--steps must be positive')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.8)
    started = time.monotonic()
    data = torch.load(args.replay, map_location='cpu', weights_only=True)['samples']
    episodes = sorted({s['episode'] for s in data})
    assert len(episodes) >= 2
    training = {e: sorted([s for s in data if s['episode'] == e], key=lambda s: s['timestep'])
                for e in episodes[:-1]}
    panel = [s for s in data if s['episode'] == episodes[-1]]
    model = load_model(args.checkpoint, args.arm != 'baseline', 42,
        memory_conditioning='direct', memory_length=64, memory_cognitive_qk_norm=args.qk_norm)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-5, weight_decay=.01, foreach=False)

    def inputs(row):
        return (BatchFeature({k: v.cuda() for k, v in row['features'].items()}),
                row['input_ids'].cuda(), BatchFeature({k: v.cuda() for k, v in row['actions'].items()}))

    @torch.no_grad()
    def evaluate():
        model.eval()
        results = {}
        for condition in (['baseline'] if model.memory is None else ['normal', 'empty', 'wrong']):
            if model.memory is not None:
                model.memory.reset()
            errors, masks = 0., 0.
            for i, row in enumerate(panel):
                feat, ids, actions = inputs(row)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    if model.memory is not None:
                        context = dict(mode='empty' if condition == 'empty' else 'stream',
                                       episode_keys=['panel'], timesteps=[row['timestep']])
                        # Wrong history is produced by another episode's observations,
                        # with the same temporal slots; current observation stays real.
                        if condition == 'wrong' and i:
                            model.memory.reset()
                            donor = training[episodes[0]]
                            for j, previous in enumerate(panel[:i]):
                                df, di, _ = inputs(donor[j % len(donor)])
                                model.memory(df, di, dict(mode='stream', episode_keys=['panel'],
                                                          timesteps=[previous['timestep']]))
                        feat = model.memory(feat, ids, context)
                    torch.manual_seed(90000+row['timestep'])
                    pred = model.action_head.get_action(feat, actions)['action_pred'].float()
                    assert torch.isfinite(pred).all()
                    mask = actions.action_mask[:, :8].float()
                    errors += float(((pred[:, :8]-actions.action[:, :8]).abs()*mask).sum())
                    masks += float(mask.sum())
            results[condition] = errors/masks
        return results

    before = evaluate()
    rows = []
    for step in range(args.steps):
        episode = episodes[step % len(training)]
        samples = training[episode]
        model.action_head.train()
        if model.memory is not None:
            model.memory.reset()
            model.memory.train(args.arm == 'memory')
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for i, row in enumerate(samples):
            feat, ids, actions = inputs(row)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                if model.memory is not None:
                    feat = model.memory(feat, ids, dict(
                        mode='train_sequence' if args.arm == 'memory' else 'empty',
                        sequence_start=i == 0, sequence_end=i == len(samples)-1,
                        episode_keys=[episode], timesteps=[row['timestep']]))
                torch.manual_seed(42 + step*100 + i)
                loss = model.action_head(feat, actions)['loss']/len(samples)
            assert torch.isfinite(loss), f'Nonfinite loss at {step}'
            loss.backward()
            total += float(loss.detach())
        qgrad = None
        if args.arm == 'memory':
            grad = model.memory.cognitive.retrieval_blocks[0].q_proj.weight.grad
            assert grad is not None, 'Cognitive query disconnected from action loss'
            qgrad = float(grad.norm())
        norm = float(torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True))
        optimizer.step()
        row = dict(step=step+1, loss=total, gradient_norm=norm, cognitive_q_gradient=qgrad)
        rows.append(row)
        if step == 0 or (step+1) % 10 == 0:
            print(json.dumps(row), flush=True)
    after = evaluate()
    passed = args.arm != 'memory' or sum(r['cognitive_q_gradient'] > 1e-8 for r in rows) >= .8*len(rows)
    report = dict(scope='Real cached feature/action training smoke; sparse histories; NOT closed-loop or validation success',
        checkpoint=str(args.checkpoint.resolve()), replay=str(args.replay.resolve()), arm=args.arm,
        qk_norm=args.qk_norm, steps=args.steps, train_episodes=list(training), diagnostic_episode=episodes[-1],
        warning='Diagnostic episode was in the original 16k training set; these are not held-out results',
        before_mae_first8=before, after_mae_first8=after, training=rows, passed=passed,
        elapsed_seconds=time.monotonic()-started, peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'training'}, indent=2))
    if not passed:
        raise SystemExit('Cognitive query gradient disappeared; inspect report before long training')


if __name__ == '__main__':
    main()
