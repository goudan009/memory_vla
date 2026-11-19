"""Aggregate Task-2 evidence without treating flow MSE as policy success."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch


GROUPS = {'base': (0, 3), 'torso': (3, 7), 'left_arm': (7, 14),
          'left_gripper': (14, 15), 'right_arm': (15, 22), 'right_gripper': (22, 23)}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def average(rows, field):
    return float(np.mean([r[field] for r in rows])) if rows else None


def summarize(root, quiet=False):
    request = json.loads((root / 'request.json').read_text())
    stats = json.loads((Path(request['checkpoint']) / 'statistics.json').read_text())['behavior_r1_pro']
    manifest = json.loads((root / 'data_manifest.json').read_text())
    target_cache = {}
    for record in manifest:
        if record['split'] == 'validation':
            episode = torch.load(record['path'], map_location='cpu', weights_only=False)
            target_cache[record['episode']] = dict(
                target=episode['actions']['action'][:, :32, :23].float().numpy(),
                mask=episode['actions']['action_mask'][:, :32, :23].float().numpy())
            del episode
    scales = {}
    for group, (lo, hi) in GROUPS.items():
        source = stats['relative_action' if group in ('torso', 'left_arm', 'right_arm') else 'action'][group]
        scale = (np.asarray(source['max']) - np.asarray(source['min'])) / 2
        scales[group] = scale[:32] if scale.ndim == 2 else np.broadcast_to(scale, (32, hi - lo))
    ambiguous = json.loads((root / 'history_dependency_candidates.json').read_text())['pairs']
    candidate_keys = {(p['episode'], i) for p in ambiguous for i in (p['i'], p['j'])}
    result = dict(config=request, history_dependency_candidates=ambiguous, arms={})
    arms = ['official_plain', 'A_joint', 'B_frozen']
    arms += [a + '_history_content' for a in ('A_joint', 'B_frozen')
             if (root / (a + '_history_content') / 'complete.json').exists()]
    for arm in arms:
        rows = read_rows(root / arm / 'evaluation.jsonl')
        summaries = []
        by_step = sorted({r['step'] for r in rows})
        for step in by_step:
            predictions = np.load(root / arm / f'predictions_step{step:03d}.npz')
            for row in [r for r in rows if r['step'] == step]:
                episode, index = row['episode'], row['index']
                data = target_cache[episode]
                pred = predictions[f'{episode}_{index}_{row["condition"]}'][0]
                target, mask = data['target'][index], data['mask'][index]
                error = pred - target
                row['candidate'] = (episode, index) in candidate_keys
                for group, (lo, hi) in GROUPS.items():
                    raw_error = np.abs(error[:, lo:hi] * scales[group])
                    m = mask[:, lo:hi]
                    row['parts'][group]['raw_mae8'] = float((raw_error[:8] * m[:8]).sum() / m[:8].sum())
                    row['parts'][group]['raw_mae32'] = float((raw_error * m).sum() / m.sum())
                if step > 0 and (root / arm / 'predictions_step000.npz').exists():
                    initial = np.load(root / arm / 'predictions_step000.npz')
                    before = initial[f'{episode}_{index}_{row["condition"]}'][0]
                    row['step0_output_shift_mae8'] = float(np.abs(pred[:8] - before[:8]).mean())
            for subset in ('all', 'candidate', 'noncandidate'):
                for condition in sorted({r['condition'] for r in rows}):
                    chosen = [r for r in rows if r['step'] == step and r['condition'] == condition
                              and (subset == 'all' or r['candidate'] == (subset == 'candidate'))]
                    if not chosen:
                        continue
                    summary = dict(step=step, subset=subset, condition=condition, samples=len(chosen),
                        **{key: average(chosen, key) for key in ('flow_loss', 'mae8', 'mae32')},
                        parts={group: {key: float(np.mean([r['parts'][group][key] for r in chosen]))
                                       for key in ('flow_mse', 'mae8', 'mae32', 'raw_mae8', 'raw_mae32')}
                               for group in GROUPS})
                    for field in ('correct_output_shift_mae8', 'correct_output_shift_mae32',
                                  'correct_feature_shift_rms', 'step0_output_shift_mae8'):
                        if field in chosen[0]:
                            summary[field] = average(chosen, field)
                    summaries.append(summary)
        pairs = []
        for step in by_step:
            normal = {(r['episode'], r['index']): r for r in rows if r['step'] == step and r['condition'] == 'correct'}
            for condition in sorted({r['condition'] for r in rows} - {'correct'}):
                other = {(r['episode'], r['index']): r for r in rows if r['step'] == step and r['condition'] == condition}
                for subset in ('all', 'candidate'):
                    keys = sorted(set(normal) & set(other))
                    if subset == 'candidate':
                        keys = [k for k in keys if k in candidate_keys]
                    if not keys:
                        continue
                    changes = np.asarray([other[k]['mae8'] - normal[k]['mae8'] for k in keys])
                    # Clustered paired bootstrap at independent validation Episode level.
                    episodes = sorted({k[0] for k in keys})
                    by_episode = np.asarray([np.mean([other[k]['mae8'] - normal[k]['mae8'] for k in keys if k[0] == e]) for e in episodes])
                    rng = np.random.default_rng(42)
                    boot = by_episode[rng.integers(0, len(by_episode), (5000, len(by_episode)))].mean(1)
                    pairs.append(dict(step=step, subset=subset, comparator=condition,
                        correct_mae8_advantage_mean=float(changes.mean()),
                        correct_better_samples=int((changes > 0).sum()), samples=len(keys),
                        episode_mean_advantage=float(by_episode.mean()),
                        episode_bootstrap95=np.quantile(boot, [.025, .975]).tolist(), episodes=len(episodes),
                        per_episode={str(e): float(v) for e, v in zip(episodes, by_episode)}))
        arm_result = dict(evaluation=summaries, paired_history_comparison=pairs, rows=rows)
        if arm in ('A_joint', 'B_frozen'):
            training = read_rows(root / arm / 'training.jsonl')
            config = json.loads((root / arm / 'config.json').read_text())
            gradients = {name: float(np.median([r['gradients'].get(name, 0) for r in training]))
                         for name in set().union(*(r['gradients'].keys() for r in training))}
            arm_result.update(initialization_sha256=config['initialization_sha256'],
                optimizer_groups=config['optimizer_groups'],
                training_first10_mean=float(np.mean([r['loss'] for r in training[:10]])),
                training_last10_mean=float(np.mean([r['loss'] for r in training[-10:]])),
                training_part_mse={group: float(np.mean([r['part_flow_mse'][group] for r in training])) for group in GROUPS},
                gradient_norm_medians=gradients,
                parameter_updates=json.loads((root / arm / f'updates_step{request["steps"]:03d}.json').read_text())['modules'],
                optimizer_steps=len(training), total_seconds=sum(r['seconds'] for r in training))
        result['arms'][arm] = arm_result
    path = root / 'aggregated_results.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    compact = {arm: {k: v for k, v in data.items() if k not in ('rows', 'parameter_updates')}
               for arm, data in result['arms'].items()}
    if not quiet:
        print(json.dumps(compact, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/task2_memory_training_20261113'))
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    summarize(args.root, args.quiet)
