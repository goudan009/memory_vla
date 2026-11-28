"""Plot recorded training loss without rerunning a model or including MAE."""

import argparse
import json
from pathlib import Path
from statistics import fmean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = {'baseline': '#2563eb', 'memory_v1': '#ea580c'}
    labels = {'baseline': 'GR00T without memory', 'memory_v1': 'MemoryVLA M1'}
    boundaries = set()
    for arm in colors:
        path = args.run / arm / 'metrics/metrics.jsonl'
        if not path.exists():
            continue
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        rows = sorted((row for row in records if row['kind'] == 'train'), key=lambda row: row['step'])
        if not rows or len({row['step'] for row in rows}) != len(rows):
            raise ValueError(f'Expected nonempty unique training steps: {path}')
        boundaries.update(rows[i]['step'] for i in range(1, len(rows))
                          if rows[i]['epoch'] != rows[i-1]['epoch'])
        for ax, block_size in zip(axes, (50, 500)):
            blocks = [rows[i:i+block_size] for i in range(0, len(rows), block_size)]
            ax.plot([block[-1]['step'] for block in blocks],
                    [fmean(row['loss'] for row in block) for block in blocks],
                    label=labels[arm], color=colors[arm], linewidth=1.5,
                    marker='o' if block_size == 500 else None, markersize=3)
            ax.set_title(f'Training loss: mean over {block_size} consecutive optimizer steps')
        print(f'{arm}: {len(rows)} steps, first 1000 mean={fmean(r["loss"] for r in rows[:1000]):.10f}, '
              f'last 1000 mean={fmean(r["loss"] for r in rows[-1000:]):.10f}')
    for ax in axes:
        for step in sorted(boundaries):
            ax.axvline(step, color='gray', linestyle='--', alpha=.7,
                       label=f'Next data pass starts ({step})')
        ax.set_ylabel('Flow-matching loss')
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    axes[-1].set_xlabel('Optimizer steps')
    split = json.loads((args.run / 'config/data_split.json').read_text())
    fig.suptitle(f'Task {split.get("task", 21)} training | loss measures fitting, not robot task success', fontsize=12)
    fig.tight_layout()
    output = args.run / 'plots/training_loss.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output.resolve())


if __name__ == '__main__':
    main()
