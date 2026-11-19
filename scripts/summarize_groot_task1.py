"""Plots and conditional metrics for the fixed task1 diagnostic artifacts."""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/nvme3/projects/memory_vla')
OUT = ROOT / 'outputs/diagnostics/task1_groot_b1k'
GROUPS = {'base': (0, 3), 'torso': (3, 7), 'left_arm': (7, 14),
          'left_gripper': (14, 15), 'right_arm': (15, 22), 'right_gripper': (22, 23)}


def main():
    report = json.loads((OUT / 'open_loop.json').read_text())
    arr = np.load(OUT / 'open_loop_arrays.npz')
    pred, target = arr['prediction'], arr['target']
    conditional = {}
    for task in (21, 22, 26):
        ids = [i for i, r in enumerate(report['samples']) if r['task'] == task and r['purpose'] == 'operation_probe']
        detail = {}
        for g in ('left_gripper', 'right_gripper'):
            lo, hi = GROUPS[g]
            switches = [i for i in ids if np.any(np.diff(target[i, :8, lo] > 0))]
            detail[g] = {'first8_switch_samples': len(switches),
                         'first8_switch_sign_accuracy': float(np.mean((pred[switches, :8, lo] > 0) == (target[switches, :8, lo] > 0))) if switches else None,
                         'non_switch_samples': len(ids) - len(switches)}
        conditional[str(task)] = detail
    (OUT / 'conditional_metrics.json').write_text(json.dumps(conditional, indent=2) + '\n')
    stats = json.loads((OUT / 'data_statistics.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    labels = ['navigation_only', 'manipulation_only', 'navigation_and_manipulation', 'rest']
    bottom = np.zeros(3)
    for label in labels:
        vals = np.array([stats[str(t)]['phase_proxy_fractions'][label] for t in (21, 22, 26)])
        axes[0].bar(['21', '22', '26'], vals * 100, bottom=bottom, label=label)
        bottom += vals * 100
    axes[0].set_ylabel('% sampled decisions'); axes[0].set_xlabel('Task'); axes[0].legend(fontsize=7)
    for i, task in enumerate((21, 22, 26)):
        vals = [stats[str(task)]['groups'][g]['span32_near_static_fraction'] * 100 for g in GROUPS]
        axes[1].bar(np.arange(6) + (i - 1) * .25, vals, .25, label=f'Task{task}')
    axes[1].set_xticks(np.arange(6), GROUPS, rotation=25); axes[1].set_ylabel('% near-static over 32 actions'); axes[1].legend()
    fig.savefig(OUT / 'data_distribution.png', dpi=150); plt.close(fig)
    selection = json.loads((OUT / 'selection.json').read_text())
    clip_rows = [r for r in report['samples'] if r['purpose'] == 'overfit_clip']
    times = np.array([r['frame'] for r in clip_rows])
    cases = []
    for folder, label in [(OUT, 'lr=1e-4'), (OUT / 'overfit_lr1e5', 'lr=1e-5')]:
        if not (folder / 'overfit.json').exists(): continue
        d = json.loads((folder / 'overfit.json').read_text())
        fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
        for ax, g in zip(axes.flat, GROUPS):
            ax.plot([h['step'] for h in d['history']], [h['groups'][g]['mae_horizon32'] for h in d['history']], marker='o')
            ax.set_title(g); ax.set_xlabel('Optimizer steps'); ax.set_ylabel('Decoded action MAE, horizon32')
        fig.suptitle(f'One-clip memorization: {label}, fixed prediction noise')
        fig.savefig(folder / 'overfit_errors.png', dpi=150); plt.close(fig)
        step = d['history'][-1]['step']
        if label == 'lr=1e-5' and step > 300 and (folder / 'overfit_step300.npz').exists():
            cases.append((label + ' step300', np.load(folder / 'overfit_step300.npz')['prediction']))
        if (folder / f'overfit_step{step}.npz').exists():
            cases.append((label + f' step{step}', np.load(folder / f'overfit_step{step}.npz')['prediction']))
    a = np.load(OUT / 'overfit_step0.npz'); gt = a['target']; initial = a['prediction']
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    for ax, (g, (lo, hi)) in zip(axes.flat, GROUPS.items()):
        j = lo + int(np.argmax(gt[:, 0, lo:hi].std(0)))
        ax.plot(times, gt[:, 0, j], '-o', label='demonstration', markersize=3)
        ax.plot(times, initial[:, 0, j], label='initial B1K')
        for label, pr in cases: ax.plot(times, pr[:, 0, j], label=label)
        ax.axvline(selection['clip']['close_frame'], color='grey', ls=':')
        ax.set_title(f'{g}, joint {j-lo}'); ax.set_xlabel('Demonstration frame (30Hz)'); ax.legend(fontsize=7)
    fig.savefig(OUT / 'overfit_action_curves.png', dpi=150); plt.close(fig)
    i = int(np.argmin(np.abs(times - (selection['clip']['close_frame'] - 16))))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    for ax, (lo, hi), name in zip(axes, [(15, 22), (22, 23)], ['right arm', 'right gripper']):
        j = lo + int(np.argmax(gt[i, :, lo:hi].std(0)))
        x = times[i] + np.arange(32)
        ax.plot(x, gt[i, :, j], label='demonstration'); ax.plot(x, initial[i, :, j], label='initial B1K')
        for label, pr in cases: ax.plot(x, pr[i, :, j], label=label)
        ax.axvline(selection['clip']['close_frame'], color='grey', ls=':'); ax.set_title(name); ax.legend(fontsize=7)
    fig.savefig(OUT / 'grasp_boundary_chunk.png', dpi=150); plt.close(fig)
    control = OUT / 'control'
    if (control / 'report.json').exists():
        steps = [json.loads(line) for line in (control / 'steps.jsonl').read_text().splitlines()]
        values = np.load(control / 'control_arrays.npz')
        ids = [i for i, row in enumerate(steps) if row['phase'] == 'demonstration']
        clip = selection['clip']
        raw_path = ROOT / f'assets/datasets/b1k2025/data/task-0021/episode_{clip["episode"]:08d}.parquet'
        raw = np.stack(pd.read_parquet(raw_path, columns=['action'])['action'])
        result = {'demonstration_command_raw_max_error': float(np.abs(values['commands'][ids] - raw[clip['start']:clip['end']]).max()),
                  'terminated_steps': sum(r['terminated'] for r in steps),
                  'truncated_steps': sum(r['truncated'] for r in steps),
                  'valid_field_values': sorted({str(r['valid']) for r in steps}),
                  'demo_tracking': {}}
        for name, lo, hi in [('left_arm', 7, 14), ('right_arm', 15, 22)]:
            delta = np.abs(values['commands'][ids, lo:hi] - values['actual'][ids, lo:hi])
            result['demo_tracking'][name] = {'mae_rad': float(delta.mean()), 'p95_rad': float(np.quantile(delta, .95)), 'max_rad': float(delta.max())}
        for phase in ('manual_close', 'manual_open'):
            last = [r for r in steps if r['phase'] == phase][-1]
            result[phase] = {side: last[f'{side}_gripper_qpos'] for side in ('left', 'right')}
        text = (OUT / 'control.log').read_text(errors='replace')
        result['asset_hash_mismatch_objects'] = sorted(set(re.findall(r'Object (\S+) was expected to have USD file hash', text)))
        (control / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
        for ax, (name, lo, hi) in zip(axes, [('left_arm', 7, 14), ('right_arm', 15, 22)]):
            j = lo + int(np.argmax(np.ptp(values['commands'][ids, lo:hi], axis=0)))
            ax.plot(np.arange(len(steps)) / 30, values['commands'][:, j], label='controller command', lw=1)
            ax.plot(np.arange(len(steps)) / 30, values['actual'][:, j], label='actual joint', ls='--', lw=1)
            ax.axvline(ids[0] / 30, color='grey', ls=':'); ax.set_title(f'{name}, joint {j-lo}'); ax.legend()
            ax.set_xlabel('Simulation seconds'); ax.set_ylabel('rad')
        fig.savefig(control / 'tracking.png', dpi=150); plt.close(fig)
    print(json.dumps(conditional, indent=2))


if __name__ == '__main__':
    main()
