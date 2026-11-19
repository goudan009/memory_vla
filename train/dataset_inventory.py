"""Inspect converted B1K task files before registering A100 readiness."""

import json
from pathlib import Path


CAMERAS = ('head', 'left_wrist', 'right_wrist')


def inspect_task(dataset, task, split_path):
    dataset, split_path = Path(dataset), Path(split_path)
    required = ['info.json', 'modality.json', 'stats.json', 'relative_stats.json',
                'tasks.jsonl', 'episodes.jsonl']
    missing = [str(dataset/'meta'/name) for name in required if not (dataset/'meta'/name).is_file()]
    if not split_path.is_file():
        missing.append(str(split_path))
    if missing:
        return dict(task=task, ready=False, missing=missing)
    split = json.loads(split_path.read_text())
    assert split['task'] == task and split['seed'] == 42
    assert len(set(split['train'])) == len(split['train']) == 180
    assert len(set(split['validation'])) == len(split['validation']) == 20
    assert not set(split['train']) & set(split['validation'])
    episodes = [json.loads(line) for line in (dataset/'meta/episodes.jsonl').read_text().splitlines()]
    ids = {row['episode_index'] for row in episodes}
    assert len(episodes) == len(ids) == 200 and ids == set(split['train']+split['validation'])
    info = json.loads((dataset/'meta/info.json').read_text())
    assert info['fps'] == 30 and info['total_episodes'] == 200
    assert info['features']['observation.state']['shape'] == [82]
    assert info['features']['action']['shape'] == [23]
    files = []
    for eid in sorted(ids):
        paths = [dataset/f'data/task-{task:04d}/episode_{eid:08d}.parquet']
        paths.extend(dataset/f'videos/task-{task:04d}/observation.images.rgb.{camera}_256_256/episode_{eid:08d}.mp4'
                     for camera in CAMERAS)
        for path in paths:
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
            else:
                files.append(path)
    return dict(task=task, ready=not missing, missing=missing, dataset=str(dataset), split=str(split_path),
                episodes=200, parquet_files=sum(p.suffix=='.parquet' for p in files),
                video_files=sum(p.suffix=='.mp4' for p in files), bytes=sum(p.stat().st_size for p in files))


def remote_inventory(datasets, training, register=False):
    datasets, training = Path(datasets), Path(training)
    rows = []
    candidates = {}
    for path in sorted(training.glob('*/config/data_split.json')):
        split = json.loads(path.read_text())
        if type(split.get('task')) is int:
            candidates.setdefault(split['task'], path)
    for task in range(50):
        dataset = datasets/f'b1k2025_groot_task{task}_v1_200'
        split_path = dataset/'preparation/data_split.json'
        if not split_path.is_file():
            split_path = candidates.get(task, split_path)
        try:
            row = inspect_task(dataset, task, split_path)
            if row['ready'] and register:
                preparation = dataset/'preparation'
                preparation.mkdir(exist_ok=True)
                value = json.loads(split_path.read_text())
                split_path = preparation/'data_split.json'
                split_path.write_text(json.dumps(value, indent=2)+'\n')
                marker = dict(status='completed', task=task, dataset=str(dataset), split=str(split_path),
                              train_episodes=180, validation_episodes=20,
                              verification=dict(parquet_files=200, video_files=600, fps=30))
                pending = preparation/'ready.pending'
                pending.write_text(json.dumps(marker, indent=2)+'\n')
                pending.replace(preparation/'ready.json')
                row['split'] = str(split_path)
        except (AssertionError, KeyError, ValueError) as error:
            row = dict(task=task, ready=False, error=repr(error), missing=[])
        rows.append(row)
    return rows
