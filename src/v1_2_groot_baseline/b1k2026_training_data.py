"""Materialize explicit LeRobot v3 episode splits without changing downloads."""
import hashlib
import json
from pathlib import Path

from .b1k2026_data import CAMERAS, episode_rows, modality_metadata
from .contracts import inspect_upstream


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def validate_split(source, split):
    rows = episode_rows(source)
    if not isinstance(split, dict) or set(split) != {'train', 'validation'}:
        raise ValueError('split requires explicit train and validation episode IDs')
    groups = {}
    for name, ids in split.items():
        if not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('split IDs must be nonempty unique integer lists')
        groups[name] = set(ids)
    if groups['train'] & groups['validation']:
        raise ValueError('train/validation episode overlap')
    by_id = {r['episode_index']: r for r in rows}
    selected = groups['train'] | groups['validation']
    if not selected <= by_id.keys():
        raise ValueError('split references unknown episodes')
    task_sets = [{by_id[i]['task_index'] for i in groups[name]} for name in split]
    if task_sets[0] != task_sets[1]:
        raise ValueError('each selected task needs train and validation episodes')
    selected_tasks = task_sets[0]
    if selected != {r['episode_index'] for r in rows if r['task_index'] in selected_tasks}:
        raise ValueError('split must account for every downloaded episode of selected tasks')
    instances = [{(by_id[i]['task_index'], by_id[i]['task_instance_id']) for i in groups[name]} for name in split]
    if instances[0] & instances[1]:
        raise ValueError('train/validation scene-instance overlap')
    return {name: [by_id[i] for i in sorted(groups[name])] for name in split}


def native_statistics(view, upstream):
    import sys
    sys.path.insert(0, str(upstream))
    from gr00t.eval.eval_b1k_wrapper import load_modality_config
    load_modality_config(str(upstream / 'examples/b1k/r1pro.py'))
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.stats import generate_stats, generate_rel_stats
    generate_stats(view)
    generate_rel_stats(view, EmbodimentTag.NEW_EMBODIMENT)


def materialize(source, destination, split, upstream, language_mode):
    """Destination is an allocated, empty workbench resource directory.

    Physical parquet rows are filtered (the upstream stats code globs them).
    Videos retain original files and per-episode time ranges; no cross-split
    episode record is accessible to the native loader.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    source, destination, upstream = map(lambda p: Path(p).resolve(strict=True), (source, destination, upstream))
    if source == destination or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('derived split resource must be separate from source')
    if any(destination.iterdir()):
        raise ValueError('split destination must be empty; never overwrite an existing split')
    if language_mode not in ('native_tasks', 'task_description'):
        raise ValueError('explicit native_tasks or task_description language_mode required')
    source_identity = inspect_upstream(upstream)
    selected = validate_split(source, split)
    info = json.loads((source / 'meta/info.json').read_text())
    if info['codebase_version'] != 'v3.0' or info['fps'] != 30:
        raise ValueError('expected native LeRobot v3.0 / 30 Hz')
    task_table = pq.read_table(source / 'meta/tasks.parquet')
    descriptions = {r['task_index']: r for r in map(json.loads, (source / 'meta/tasks.jsonl').read_text().splitlines())}
    source_files = {}
    views = {}
    for name, records in selected.items():
        view = destination / name
        (view / 'meta/episodes/chunk-000').mkdir(parents=True)
        by_file = {}
        video_files = set()
        for record in records:
            relative = info['data_path'].format(chunk_index=record['data/chunk_index'], file_index=record['data/file_index'])
            by_file.setdefault(relative, []).append(record['episode_index'])
            for camera in CAMERAS.values():
                prefix = 'videos/' + camera + '/'
                video_files.add(info['video_path'].format(video_key=camera,
                    chunk_index=record[prefix + 'chunk_index'], file_index=record[prefix + 'file_index']))
        for relative, ids in by_file.items():
            original = source / relative
            table = pq.read_table(original)
            table = table.filter(pc.is_in(table['episode_index'], value_set=pa.array(ids, type=table['episode_index'].type)))
            if set(table['episode_index'].to_pylist()) != set(ids):
                raise ValueError('physical episode set differs from metadata: ' + relative)
            expected_length = sum(r['length'] for r in records if r['episode_index'] in ids)
            if len(table) != expected_length:
                raise ValueError('physical row count differs from split metadata')
            target = view / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, target)
            source_files[relative] = digest(original)
        for relative in sorted(video_files):
            original = (source / relative).resolve(strict=True)
            target = view / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(original)
        tasks = sorted({r['task_index'] for r in records})
        subset_tasks = task_table.filter(pc.is_in(task_table['task_index'], value_set=pa.array(tasks, type=task_table['task_index'].type)))
        if language_mode == 'task_description':
            subset_tasks = subset_tasks.set_column(subset_tasks.schema.get_field_index('task'), 'task',
                                                   pa.array([descriptions[t]['task'] for t in subset_tasks['task_index'].to_pylist()]))
        pq.write_table(subset_tasks, view / 'meta/tasks.parquet')
        (view / 'meta/tasks.jsonl').write_text('\n'.join(json.dumps(descriptions[t]) for t in tasks) + '\n')
        pq.write_table(pa.Table.from_pylist(records), view / 'meta/episodes/chunk-000/file-000.parquet')
        subset_info = dict(info, total_episodes=len(records), total_frames=sum(r['length'] for r in records),
                           total_tasks=len(tasks), total_videos=len(video_files), splits={'train': f'0:{len(records)}'})
        (view / 'meta/info.json').write_text(json.dumps(subset_info, indent=2))
        (view / 'meta/modality.json').write_text(json.dumps(modality_metadata(), indent=2))
        native_statistics(view, upstream)
        views[name] = dict(path=str(view), episodes=[r['episode_index'] for r in records],
            rows=subset_info['total_frames'], tasks=tasks,
            task_prompts={str(r['task_index']): r['task'] for r in subset_tasks.to_pylist()},
            videos={relative:str((source / relative).resolve(strict=True)) for relative in sorted(video_files)},
            files={str(p.relative_to(view)): digest(p) for p in [view / rel for rel in sorted(by_file)] +
                   [view / 'meta/info.json', view / 'meta/modality.json', view / 'meta/tasks.parquet',
                    view / 'meta/stats.json', view / 'meta/relative_stats.json', view / 'meta/episodes/chunk-000/file-000.parquet']},
            video_spans=[dict(episode_index=r['episode_index'],
                cameras={key: {k: r['videos/' + key + '/' + k] for k in
                     ('chunk_index', 'file_index', 'from_timestamp', 'to_timestamp')} for key in CAMERAS.values()}) for r in records])
    manifest = dict(benchmark='b1k2026', schema='native-v3-split-v1', source=str(source),
        upstream_revision=source_identity['revision'], source_files=source_files,
        language_mode=language_mode, skill_supervision=False, views=views,
        statistics_scope='each split independently; training uses train statistics only')
    (destination / 'data_manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def verify_manifest(path):
    path = Path(path).resolve(strict=True)
    manifest = json.loads(path.read_text())
    if manifest['benchmark'] != 'b1k2026' or manifest['schema'] != 'native-v3-split-v1':
        raise ValueError('expected B1K-2026 native split manifest')
    groups = []
    for name in ('train', 'validation'):
        view = manifest['views'][name]
        root = Path(view['path']).resolve(strict=True)
        if root != path.parent / name:
            raise ValueError('split view outside its allocated resource')
        actual_parquet = {p.relative_to(root).as_posix() for folder in ('data', 'meta/episodes')
                          for p in (root / folder).rglob('*.parquet')}
        expected_parquet = {p for p in view['files'] if p.startswith(('data/', 'meta/episodes/'))}
        if actual_parquet != expected_parquet:
            raise ValueError('unregistered parquet file in split view')
        for relative, original in view['videos'].items():
            if not (root / relative).is_file() or (root / relative).resolve() != Path(original):
                raise ValueError('split video target changed: ' + relative)
        for relative, expected in view['files'].items():
            target = (root / relative).resolve(strict=True)
            if not target.is_relative_to(root) or digest(target) != expected:
                raise ValueError('split file changed: ' + relative)
        actual = sorted(r['episode_index'] for r in episode_rows(root))
        if actual != sorted(view['episodes']):
            raise ValueError('split metadata changed')
        groups.append(set(actual))
    if groups[0] & groups[1]:
        raise ValueError('train/validation overlap')
    return manifest
