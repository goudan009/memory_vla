"""Plot measured V2 state and mechanism diagnostics; does not change evaluation."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.run / 'state_predictions.jsonl').open()]
    known = [row for row in rows if row['known']]
    mechanism = list(csv.DictReader((args.output / 'mechanism_metrics.csv').open()))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for label, name in ((0, 'Not started'), (1, 'Started')):
        values = [row['probability'] for row in known if row['truth'] == label]
        axes[0, 0].hist(values, bins=np.linspace(0, .55, 75), density=True,
                        histtype='step', linewidth=1.5, label=f'{name} (n={len(values)})')
    axes[0, 0].axvline(.5, color='black', linestyle='--', label='Locked classifier threshold')
    axes[0, 0].set(xlabel='Predicted started probability', ylabel='Density', title='All predictions below threshold')
    axes[0, 0].legend(fontsize=8)
    episode = next(row['episode'] for row in known if row['truth'] == 1)
    trajectory = [row for row in rows if row['episode'] == episode]
    axes[0, 1].plot([r['frame'] for r in trajectory], [r['probability'] for r in trajectory], label='Prediction')
    axes[0, 1].scatter([r['frame'] for r in trajectory if r['known']],
                       [r['truth'] for r in trajectory if r['known']], s=3, label='Known physical label')
    axes[0, 1].set(xlabel='Observation frame', ylabel='Started probability / truth',
                   title=f'First held-out episode: {episode}', ylim=(-.05, 1.05))
    axes[0, 1].legend(fontsize=8)
    conditions = ['empty_cognitive', 'wrong_cognitive', 'state_guidance_off']
    names = ['Empty history', 'Wrong history', 'State guidance off']
    for index, (field, title) in enumerate((('normalized_actions_mae', 'Normalized candidate action MAE'),
                                           ('dit_condition_eos_mae', 'DiT condition MAE'))):
        groups = [[float(r[field]) for r in mechanism if r['condition'] == condition] for condition in conditions]
        axis = axes[1, index]
        axis.boxplot(groups, tick_labels=names, showfliers=False)
        axis.set(ylabel=title, title='100 correlated windows / 20 episodes')
        axis.tick_params(axis='x', labelsize=8)
        axis.grid(axis='y', alpha=.2)
    fig.suptitle('V2 / test02 / exp01 — state failure and measured mechanism response')
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / 'evaluation_diagnostics.png', dpi=180)
    fig.savefig(args.output / 'evaluation_diagnostics.pdf')
    plt.close(fig)
    print(f'Plotted {len(known)} labeled decisions; no threshold or protocol changes')


if __name__ == '__main__':
    main()
