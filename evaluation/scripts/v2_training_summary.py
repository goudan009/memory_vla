"""Archive the completed V2 run's measured training curves; no GPU work."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from benchmark_dashboard import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run / 'v2_training.json').read_text())
    rows = list(csv.DictReader((args.run / 'train_monitor.csv').open()))
    steps = np.array([int(row['step']) for row in rows])
    if not report['passed'] or not np.array_equal(steps, np.arange(1, 1001)):
        raise ValueError('Expected the completed unique 1000-step run')
    args.output.mkdir(parents=True, exist_ok=True)
    fields = ('loss', 'action_loss', 'state_loss', 'gradient_norm',
              'cognitive_q_gradient', 'seconds', 'peak_cuda_gib')
    values = {key: np.array([float(row[key]) for row in rows]) for key in fields}
    if not all(np.isfinite(value).all() for value in values.values()):
        raise ValueError('Nonfinite training measurements')
    summary = dict(job_id=args.run.name, optimizer_steps=len(rows),
        seen_samples=int(rows[-1]['seen_samples']),
        episodes_consumed=len({row['episode'] for row in rows}),
        known_decisions=sum(int(row['state_known']) for row in rows),
        positive_decisions=sum(int(row['state_positive']) for row in rows),
        scope='Fixed 1000-step budget; not a full pass over 180 training episodes',
        units='Chronological observation decisions, correlated within episodes',
        statistics={key: dict(mean=float(value.mean()), minimum=float(value.min()),
            maximum=float(value.max()), first100_mean=float(value[:100].mean()),
            last100_mean=float(value[-100:].mean())) for key, value in values.items()},
        training_report=report)
    atomic_json(args.output / 'training_summary.json', summary)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, key, title in zip(axes.flat,
            ('action_loss', 'state_loss', 'gradient_norm', 'seconds'),
            ('Action loss', 'Started state loss', 'Gradient norm before clipping', 'Seconds per optimizer step')):
        value = values[key]
        axis.plot(steps, value, alpha=.3, linewidth=.6, label='Measured')
        axis.plot(steps[24:], np.convolve(value, np.ones(25)/25, mode='valid'),
                  linewidth=1.2, label='25-step trailing mean')
        axis.set(xlabel='Optimizer step', ylabel=title)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle('V2 / test02 / exp01 / run001 — real-label training')
    fig.savefig(args.output / 'training_curves.png', dpi=180)
    fig.savefig(args.output / 'training_curves.pdf')
    plt.close(fig)
    print(json.dumps({key: summary[key] for key in ('optimizer_steps', 'seen_samples', 'episodes_consumed')}))


if __name__ == '__main__':
    main()
