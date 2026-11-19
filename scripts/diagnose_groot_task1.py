"""Bounded original-GR00T diagnostics; no Memory imports or production edits.

Statistics use every available raw episode; prediction/overfit use existing
three-camera videos and the checkpoint's unchanged official processor/model.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path('/nvme3/projects/memory_vla')
SOURCE = Path('/nvme3/projects/GR00T/official-baseline-9b37aa1')
OUT = ROOT / 'outputs/diagnostics/task1_groot_b1k'
CHECKPOINT = ROOT / 'checkpoints/groot_n1d6_behavior1k'
RAW = ROOT / 'assets/datasets/b1k2025'
TASKS = (21, 22, 26)
GROUPS = {'base': (0, 3), 'torso': (3, 7), 'left_arm': (7, 14),
          'left_gripper': (14, 15), 'right_arm': (15, 22), 'right_gripper': (22, 23)}
STATE_SLICES = json.loads((ROOT / 'assets/datasets/b1k2025_groot_task21_diagnostic/meta/conversion.json').read_text())['state_slices_raw']
REFS = {'torso': 'trunk_qpos', 'left_arm': 'arm_left_qpos', 'right_arm': 'arm_right_qpos'}
CAMS = ('head', 'left_wrist', 'right_wrist')


def write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def dataset(task):
    suffix = 'diagnostic' if task == 21 else 'v1_200'
    return ROOT / f'assets/datasets/b1k2025_groot_task{task}_{suffix}'


def video(task, eid, cam):
    return dataset(task) / f'videos/task-{task:04d}/observation.images.rgb.{cam}_256_256/episode_{eid:08d}.mp4'


def low(task, eid):
    return pd.read_parquet(RAW / f'data/task-{task:04d}/episode_{eid:08d}.parquet', columns=['action', 'observation.state'])


def arrays(df):
    return np.stack(df['action']).astype(np.float32), np.stack(df['observation.state']).astype(np.float32)


def quant(x):
    return np.quantile(x, [.5, .9, .99], axis=0).tolist()


def audit():
    """Compare existing converted files to raw rows, without model inference."""
    selection = json.loads((OUT / 'selection.json').read_text())
    episodes = sorted({(r['task'], r['episode']) for r in selection['samples']})
    raw_tasks = {r['task_index']: r['task'] for r in map(json.loads, (RAW / 'meta/tasks.jsonl').read_text().splitlines())}
    raw_episodes = {r['episode_index']: r for r in map(json.loads, (RAW / 'meta/episodes.jsonl').read_text().splitlines())}
    rows = []
    for task, eid in episodes:
        raw = pd.read_parquet(RAW / f'data/task-{task:04d}/episode_{eid:08d}.parquet')
        converted = pd.read_parquet(dataset(task) / f'data/task-{task:04d}/episode_{eid:08d}.parquet')
        a, s = arrays(raw); ca, cs = arrays(converted)
        expected = np.concatenate([s[:, lo:hi] for lo, hi in STATE_SLICES.values()], axis=1)
        row = {'task': task, 'episode': eid, 'frames': len(raw),
               'action_max_error': float(np.abs(a - ca).max()),
               'state_max_error': float(np.abs(expected - cs).max()),
               'timestamp_max_error': float(np.abs(raw.timestamp.to_numpy() - converted.timestamp.to_numpy()).max()),
               'frame_index_exact': bool(np.array_equal(converted.frame_index.to_numpy(), np.arange(len(raw)))),
               'language_index_exact': bool(np.array_equal(raw.task_index, converted['annotation.human.coarse_action'])),
               'language_text_exact': all(raw_tasks[t] == raw_episodes[eid]['tasks'][0] for t in raw.task_index.unique()),
               'complete_training_metadata': all((dataset(task) / 'meta' / f).exists() for f in ('info.json', 'modality.json', 'tasks.jsonl'))}
        rows.append(row)
        print('CONVERSION_AUDIT', row, flush=True)
    write('conversion_audit.json', {'episodes': rows, 'state_slices': STATE_SLICES,
           'statistics_sha256': hashlib.sha256((CHECKPOINT / 'statistics.json').read_bytes()).hexdigest()})


def stats():
    stats_cp = json.loads((CHECKPOINT / 'statistics.json').read_text())['behavior_r1_pro']
    result, manifest, clip = {}, [], None
    for task in TASKS:
        values, relatives, spans, phases, episodes, rejected = [], [], [], [], [], []
        files = sorted((RAW / f'data/task-{task:04d}').glob('*.parquet'))
        available = []
        for cam in CAMS:
            available.append({int(p.stem.split('_')[-1]) for p in video(task, 0, cam).parent.glob('*.mp4')})
        complete = sorted(set.intersection(*available))
        picked = complete[:3]
        selected = []
        for i, path in enumerate(files):
            eid = int(path.stem.split('_')[-1])
            try:
                df = pd.read_parquet(path, columns=['action', 'observation.state'])
            except Exception as exc:
                rejected.append({'path': str(path), 'error': str(exc)})
                print('REJECT_CORRUPT_PARQUET', path, str(exc), flush=True)
                continue
            a, s = arrays(df)
            if not np.isfinite(a).all() or not np.isfinite(s).all():
                raise ValueError(f'Nonfinite dataset: {path}')
            idx = np.arange(0, len(a) - 32, 8)
            target = a[idx]
            delta = target.copy()
            for g, key in REFS.items():
                lo, hi = GROUPS[g]; sl, sh = STATE_SLICES[key]
                delta[:, lo:hi] -= s[idx, sl:sh]
            span = np.abs(a[idx + 31] - a[idx])
            # Operational proxies, not semantic grasp/place labels.
            base_on = np.max(np.abs(target[:, :3]), axis=1) > .05
            arm_on = np.maximum(np.max(span[:, 7:14], axis=1), np.max(span[:, 15:22], axis=1)) > .03
            close, opened = [], []
            near_close = np.zeros(len(idx), bool); near_open = near_close.copy()
            for gi in (14, 22):
                change = np.flatnonzero(np.diff(a[:, gi] > 0) != 0) + 1
                for t in change:
                    is_close = a[t, gi] < 0
                    (close if is_close else opened).append(int(t))
                    hit = (idx >= t - 16) & (idx <= t + 32)
                    if is_close: near_close |= hit
                    else: near_open |= hit
            manip = arm_on | near_close | near_open
            phase = np.stack([base_on & ~manip, manip & ~base_on, base_on & manip,
                              ~base_on & ~manip, near_close, near_open], axis=1)
            phases.append(phase); values.append(target); relatives.append(delta); spans.append(span)
            episodes.append({'episode': eid, 'frames': len(a), 'close_events': len(close), 'open_events': len(opened)})
            if eid in picked:
                score = np.maximum(np.max(span[:, 7:14], axis=1), np.max(span[:, 15:22], axis=1))
                candidates = idx[manip]
                rng = np.random.default_rng(7000 + eid)
                chosen = set(int(t) for t in rng.choice(candidates, min(10, len(candidates)), replace=False))
                for t in close[:3] + opened[:3]:
                    chosen.update(((t - 16) // 8 * 8, t // 8 * 8))
                chosen = {t for t in chosen if 0 <= t < len(a) - 32}
                for t in idx[np.argsort(score)[::-1]]:
                    if len(chosen) >= 20: break
                    chosen.add(int(t))
                chosen = sorted(chosen)[:20]
                selected.extend({'task': task, 'episode': eid, 'frame': t} for t in chosen)
                if task == 21 and clip is None:
                    for t in sorted(close):
                        if t < 160 or t + 192 >= len(a): continue
                        side = 'left' if a[t, 14] < a[t - 1, 14] else 'right'
                        key = f'eef_{side}_pos'
                        sl, sh = STATE_SLICES[key]
                        z = s[t:t + 192, sh - 1]
                        lift = float(z.max() - z[0])
                        reach = float(np.linalg.norm(s[t - 128, sl:sh] - s[t, sl:sh]))
                        if lift > .05 and reach > .05:
                            clip = {'task': task, 'episode': eid, 'close_frame': t, 'side': side,
                                    'start': t - 128, 'end': t + 192, 'relative_eef_lift_m': lift,
                                    'relative_eef_reach_m': reach,
                                    'frames': np.linspace(t - 128, t + 160, 16, dtype=int).tolist()}
                            break
            if i % 40 == 0: print('STAT', task, i, len(files), flush=True)
        a, d, span, ph = map(np.concatenate, (values, relatives, spans, phases))
        groups = {}
        for g, (lo, hi) in GROUPS.items():
            v = d[:, lo:hi]
            pars = stats_cp['relative_action'][g] if g in REFS else stats_cp['action'][g]
            mn, mx = np.asarray(pars['min']), np.asarray(pars['max'])
            if mn.ndim == 2:
                mn, mx = mn[0], mx[0]  # first-action statistics, not all 32 offsets flattened
            norm = 2 * (v - mn) / np.maximum(mx - mn, 1e-8) - 1
            threshold = .05 if g == 'base' or 'gripper' in g else .01
            groups[g] = {'target_mean': a[:, lo:hi].mean(0).tolist(), 'target_std': a[:, lo:hi].std(0).tolist(),
                         'target_min': a[:, lo:hi].min(0).tolist(), 'target_max': a[:, lo:hi].max(0).tolist(),
                         'abs_target_minus_current_p50_p90_p99': quant(np.abs(v)),
                         'span32_abs_p50_p90_p99': quant(span[:, lo:hi]),
                         'span32_near_static_fraction': float(np.mean(np.max(span[:, lo:hi], axis=1) < threshold)),
                         'span32_static_threshold': threshold,
                         'normalized_std': norm.std(0).tolist(), 'normalization_outside_range_fraction': float((np.abs(norm) > 1).mean())}
        result[str(task)] = {'episodes': len(episodes), 'rejected_episodes': rejected, 'frames': sum(e['frames'] for e in episodes),
                             'sampled_decisions_stride8': len(a), 'three_camera_episodes': len(complete),
                             'groups': groups, 'phase_proxy_fractions': dict(zip(
                                 ('navigation_only', 'manipulation_only', 'navigation_and_manipulation', 'rest',
                                  'near_close_event', 'near_open_event'), ph.mean(0).tolist())),
                             'episode_details': episodes}
        manifest.extend(selected)
        write('data_statistics.json', result)
        print('TASK_STATS_COMPLETE', task, flush=True)
    write('selection.json', {'samples': manifest, 'clip': clip,
                           'selection': '3 existing-camera episodes per task, 20 operation-enriched decisions each; seed=7000+episode',
                           'phase_definitions': 'base maxabs>.05; arm abs(target[t+31]-target[t]) max>.03rad; close/open windows[-16,+32]; proxies overlap'})


def setup():
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'; os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    sys.path.insert(0, str(SOURCE))
    import torch
    torch.set_num_threads(4)
    import gr00t.model  # registry
    return torch


def make_samples(processor):
    from gr00t.data.types import VLAStepData, EmbodimentTag
    from gr00t.utils.video_utils import get_frames_by_indices
    tag = EmbodimentTag.BEHAVIOR_R1_PRO
    selection = json.loads((OUT / 'selection.json').read_text())
    specs = [dict(r, purpose='operation_probe') for r in selection['samples']]
    clip = selection['clip']
    specs += [dict(task=clip['task'], episode=clip['episode'], frame=t, purpose='overfit_clip') for t in clip['frames']]
    episodes_meta = {r['episode_index']: r for r in map(json.loads, (RAW / 'meta/episodes.jsonl').read_text().splitlines())}
    samples = []
    for task, eid in sorted({(s['task'], s['episode']) for s in specs}):
        rows = [r for r in specs if r['task'] == task and r['episode'] == eid]
        frames = sorted({r['frame'] for r in rows})
        a, s = arrays(low(task, eid))
        images = {f'observation.images.rgb.{cam}_256_256': get_frames_by_indices(str(video(task, eid, cam)), frames, 'torchcodec') for cam in CAMS}
        for r in rows:
            t = r['frame']; j = frames.index(t)
            step = VLAStepData(images={k: [v[j]] for k, v in images.items()},
                states={k: s[t:t + 1, lo:hi].copy() for k, (lo, hi) in STATE_SLICES.items()},
                actions={k: a[t:t + 32, lo:hi].copy() for k, (lo, hi) in GROUPS.items()},
                text=episodes_meta[eid]['tasks'][0], embodiment=tag)
            processed = processor([{'type': 'episode_step', 'content': step}])
            samples.append({'meta': r, 'processed': processed, 'states': step.states,
                            'target': a[t:t + 32].copy(), 'text': step.text})
        print('PREPARED', task, eid, len(rows), flush=True)
    return samples


def predict(model, processor, sample, seed):
    import torch
    from gr00t.data.types import EmbodimentTag
    batch = processor.collator([sample['processed']])['inputs']
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        raw = model.get_action(batch)['action_pred'].float().cpu().numpy()
    decoded = processor.decode_action(raw, EmbodimentTag.BEHAVIOR_R1_PRO,
                                     {k: v[None] for k, v in sample['states'].items()})
    return np.concatenate([decoded[k][0] for k in GROUPS], axis=-1), raw[0]


def errors(pred, sample, raw):
    groups = {}
    for name, (lo, hi) in GROUPS.items():
        target = sample['target'][:, lo:hi]
        item = {'mae_first8': float(np.mean(np.abs(pred[:8, lo:hi] - target[:8]))),
                'mae_horizon32': float(np.mean(np.abs(pred[:32, lo:hi] - target))),
                'normalized_mae_first8': float(np.mean(np.abs(raw[:8, lo:hi] - np.asarray(sample['processed']['action'])[:8, lo:hi])))}
        if name in REFS:
            current = sample['states'][REFS[name]][-1]
            item.update(hold_current_mae_first8=float(np.mean(np.abs(target[:8] - current))),
                        target_motion_first8=float(np.mean(np.abs(target[:8] - current))),
                        prediction_motion_first8=float(np.mean(np.abs(pred[:8, lo:hi] - current))))
        if 'gripper' in name:
            item['sign_accuracy_first8'] = float(np.mean((pred[:8, lo:hi] > 0) == (target[:8] > 0)))
        groups[name] = item
    return groups


def plot_predictions(samples, predictions, suffix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for task in TASKS:
        inds = [i for i, s in enumerate(samples) if s['meta']['task'] == task and s['meta']['purpose'] == 'operation_probe']
        if not inds: continue
        fig, axes = plt.subplots(6, 1, figsize=(12, 12), constrained_layout=True)
        for ax, (g, (lo, hi)) in zip(axes, GROUPS.items()):
            gt = np.concatenate([samples[i]['target'][:8, lo:hi] for i in inds])
            pr = np.concatenate([predictions[i][:8, lo:hi] for i in inds])
            j = int(np.argmax(gt.std(0)))
            ax.plot(gt[:, j], label=f'demonstration joint {j}', lw=.9)
            ax.plot(pr[:, j], label='prediction', lw=.8, alpha=.8)
            ax.set_ylabel(g); ax.legend(loc='upper right')
        axes[-1].set_xlabel('Concatenated first-8 action chunks at sampled operation points (not a continuous rollout)')
        fig.suptitle(f'Task {task}: original B1K checkpoint / {suffix}')
        fig.savefig(OUT / f'task{task}_{suffix}.png', dpi=140); plt.close(fig)


def infer():
    torch = setup()
    from transformers import AutoModel, AutoProcessor
    from gr00t.data.types import EmbodimentTag
    processor = AutoProcessor.from_pretrained(CHECKPOINT); processor.eval()
    samples = make_samples(processor)
    torch.save(samples, OUT / 'samples.pt')
    model, loading = AutoModel.from_pretrained(CHECKPOINT, output_loading_info=True)
    assert not any(loading[k] for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')), loading
    model = model.cuda().bfloat16().eval().requires_grad_(False)
    rows, predictions, roundtrips, masks = [], [], [], []
    start = time.monotonic()
    for i, s in enumerate(samples):
        pred, raw = predict(model, processor, s, 91000 + i)
        predictions.append(pred)
        rows.append({**s['meta'], 'seed': 91000 + i, 'groups': errors(pred, s, raw)})
        restored = processor.decode_action(np.asarray(s['processed']['action'])[None], EmbodimentTag.BEHAVIOR_R1_PRO,
                                           {k: v[None] for k, v in s['states'].items()})
        roundtrips.append(float(np.max(np.abs(np.concatenate([restored[k][0] for k in GROUPS], -1) - s['target']))))
        masks.append(np.asarray(s['processed']['action_mask'])[:32, :23].sum(0).tolist())
        if i % 20 == 0: print('PREDICTION', i, len(samples), flush=True)
    summary = {}
    for task in TASKS:
        selected = [r for r in rows if r['task'] == task and r['purpose'] == 'operation_probe']
        summary[str(task)] = {g: {key: float(np.mean([r['groups'][g][key] for r in selected]))
                                 for key in selected[0]['groups'][g]} for g in GROUPS}
    write('open_loop.json', {'checkpoint': str(CHECKPOINT), 'source': str(SOURCE), 'loading_info': loading,
                            'samples': rows, 'summary': summary, 'roundtrip_max_error': max(roundtrips),
                            'mask_per_dimension_unique': np.unique(masks, axis=0).tolist(),
                            'seconds': time.monotonic() - start, 'peak_cuda_gib': torch.cuda.max_memory_allocated() / 2**30})
    np.savez_compressed(OUT / 'open_loop_arrays.npz', prediction=np.asarray(predictions),
                        target=np.asarray([s['target'] for s in samples]))
    plot_predictions(samples, predictions, 'initial')
    print('INFERENCE_COMPLETE', summary, flush=True)


def overfit(steps, lr=1e-4, subdir=None):
    global OUT
    samples_path = OUT / 'samples.pt'
    if subdir:
        OUT = OUT / subdir
        OUT.mkdir(parents=True, exist_ok=True)
    torch = setup()
    from transformers import AutoModel, AutoProcessor
    processor = AutoProcessor.from_pretrained(CHECKPOINT); processor.eval()
    samples = torch.load(samples_path, map_location='cpu', weights_only=False)
    samples = [s for s in samples if s['meta']['purpose'] == 'overfit_clip']
    torch.manual_seed(42)
    # Retain checkpoint/default trainability, including top four LLM layers.
    model = AutoModel.from_pretrained(CHECKPOINT).cuda().bfloat16()
    for p in model.parameters():
        if p.requires_grad: p.data = p.data.float()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-5)
    identity = {'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'backbone_trainable': sum(p.numel() for p in model.backbone.parameters() if p.requires_grad),
                'action_head_trainable': sum(p.numel() for p in model.action_head.parameters() if p.requires_grad),
                'samples': len(samples), 'steps': steps, 'batch_size': 2, 'learning_rate': lr,
                'augmentation': 'fixed eval preprocessed images for memorization diagnostic only',
                'scope': 'one real trajectory clip; unchanged original network and flow-matching loss; not formal training/evaluation'}
    history = []
    def evaluate(step):
        model.eval(); rows = []; preds = []
        for i, s in enumerate(samples):
            pred, raw = predict(model, processor, s, 120000 + i)
            rows.append(errors(pred, s, raw)); preds.append(pred)
        metrics = {g: {k: float(np.mean([r[g][k] for r in rows])) for k in rows[0][g]} for g in GROUPS}
        history.append({'step': step, 'groups': metrics})
        np.savez_compressed(OUT / f'overfit_step{step}.npz', prediction=np.asarray(preds), target=np.asarray([s['target'] for s in samples]))
        write('overfit.json', {'configuration': identity, 'history': history})
        print('OVERFIT_EVAL', step, json.dumps(metrics), flush=True)
    evaluate(0)
    start = time.monotonic(); logs = (OUT / 'overfit_updates.jsonl').open('w', buffering=1)
    for step in range(1, steps + 1):
        model.train(); opt.zero_grad(set_to_none=True)
        ids = [(2 * (step - 1) + j) % len(samples) for j in range(2)]
        batch = processor.collator([samples[i]['processed'] for i in ids])['inputs']
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(batch); loss = out['loss']
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        row = {'step': step, 'loss': float(loss.detach()), 'grad_norm': float(norm), 'elapsed_s': time.monotonic() - start}
        if step == 1 or step % 20 == 0:
            row['flow_loss_by_group'] = {g: float(out['action_loss'][:, :32, lo:hi].detach().sum() /
                                                   out['action_mask'][:, :32, lo:hi].sum())
                                        for g, (lo, hi) in GROUPS.items()}
        if step == 1:
            row['gradient_groups'] = {g: {'tensors': sum(p.grad is not None for p in part.parameters()),
                'norm': float(sum((p.grad.float().square().sum() for p in part.parameters() if p.grad is not None), torch.tensor(0., device='cuda')).sqrt())}
                for g, part in [('action_head', model.action_head), ('backbone', model.backbone)]}
        logs.write(json.dumps(row) + '\n')
        if step % 20 == 0: print('UPDATE', row, flush=True)
        if step in (50, 150, 300, steps): evaluate(step)
    logs.close()
    model.save_pretrained(OUT / 'overfit_checkpoint', safe_serialization=True)
    processor.save_pretrained(OUT / 'overfit_checkpoint')
    write('overfit.json', {'configuration': identity, 'history': history, 'elapsed_s': time.monotonic() - start,
                           'peak_cuda_gib': torch.cuda.max_memory_allocated() / 2**30})
    print('OVERFIT_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['audit', 'stats', 'infer', 'overfit']); p.add_argument('--steps', type=int, default=300)
    p.add_argument('--lr', type=float, default=1e-4); p.add_argument('--subdir')
    args = p.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    {'audit': audit, 'stats': stats, 'infer': infer, 'overfit': lambda: overfit(args.steps, args.lr, args.subdir)}[args.stage]()
