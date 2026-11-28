"""Read-only inspection of native B1K-2026 v3 data; no training defaults."""
import hashlib
import importlib.util
import json
from pathlib import Path

from .contracts import ACTION_GROUPS, STATE_GROUPS_2026, inspect_upstream

CAMERAS = {
    'head': 'observation.rgb.zed_link_camera_0',
    'left_wrist': 'observation.rgb.left_realsense_link_camera_0',
    'right_wrist': 'observation.rgb.right_realsense_link_camera_0',
}
TASKS = {0: 'turning_on_radio', 40: 'make_microwave_popcorn'}


def episode_rows(root):
    import pyarrow.parquet as pq
    paths = sorted((Path(root) / 'meta/episodes').glob('chunk-*/file-*.parquet'))
    if not paths:
        raise ValueError('missing v3 episode metadata')
    rows = [row for path in paths for row in pq.read_table(path).to_pylist()]
    if len({row['episode_index'] for row in rows}) != len(rows):
        raise ValueError('duplicate v3 episode metadata')
    return sorted(rows, key=lambda row: row['episode_index'])


def modality_metadata():
    return {
        'state': {key: dict(start=lo, end=hi) for key, (lo, hi) in STATE_GROUPS_2026.items()},
        'action': {key: dict(start=lo, end=hi) for key, (lo, hi) in ACTION_GROUPS.items()},
        'video': {key: dict(original_key=value) for key, value in CAMERAS.items()},
        'annotation': {'human.task_description': dict(original_key='task_index')},
    }


def validate_annotation(annotation):
    """Skill ranges use raw frame coordinates, not offsets from valid start."""
    metadata = annotation['meta_data']
    start, end = metadata['valid_duration']
    if not 0 <= start <= end or end - start != metadata['task_duration']:
        raise ValueError('invalid annotation valid_duration')
    for skill in annotation['skill_annotation']:
        lo, hi = skill['frame_duration']
        if not start <= lo <= hi <= end:
            raise ValueError('skill interval outside annotation valid_duration')
    return metadata['task_duration']


def audit_2026(dataset):
    """Check actual rows and referenced RGB/annotation files, without mutation.

    Whole-subset action statistics are audit measurements, never train-only
    normalization statistics. Annotation frame alignment is reported separately
    because the raw annotations and released trajectories use different lengths.
    """
    import av
    import numpy as np
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    root = Path(dataset).resolve(strict=True)
    info = json.loads((root / 'meta/info.json').read_text())
    if info.get('codebase_version') != 'v3.0' or info.get('fps') != 30:
        raise ValueError('expected B1K-2026 LeRobot v3.0 at 30 Hz')
    for key, dim in [('observation.state', 61), ('action', 23)]:
        if info['features'][key]['shape'] != [dim]:
            raise ValueError('incorrect physical tensor dimension: ' + key)
    rows = episode_rows(root)
    expected = {row['episode_index']: row for row in rows}
    descriptions = {row['task_index']: row for row in map(json.loads,
                    (root / 'meta/tasks.jsonl').read_text().splitlines())}
    native_tasks = {row['task_index']: row['task'] for row in
                    pq.read_table(root / 'meta/tasks.parquet').to_pylist()}
    if {row['task_index'] for row in rows} != set(TASKS):
        raise ValueError('expected scoped Task0/Task40 data')
    for task, name in TASKS.items():
        if descriptions[task]['task_name'] != name or not descriptions[task]['task'].strip():
            raise ValueError('task language metadata mismatch')
    files = sorted((root / 'data').glob('chunk-*/file-*.parquet'))
    seen = set()
    count = 0
    action_min, action_max = np.full(23, np.inf), np.full(23, -np.inf)
    action_sum, action_square_sum = np.zeros(23), np.zeros(23)
    for path in files:
        table = pq.read_table(path, columns=['episode_index', 'task_index', 'frame_index',
                              'index', 'timestamp', 'observation.state', 'action'])
        columns = {key: table[key].to_numpy(zero_copy_only=False) for key in
                   ['episode_index', 'task_index', 'frame_index', 'index', 'timestamp']}
        tensors = {}
        for key, dim in [('observation.state', 61), ('action', 23)]:
            col = table[key].combine_chunks()
            if not pc.all(pc.equal(pc.list_value_length(col), dim)).as_py():
                raise ValueError('incorrect row tensor width: ' + key)
            tensors[key] = col.flatten().to_numpy(zero_copy_only=False).reshape(-1, dim)
            if not np.isfinite(tensors[key]).all():
                raise ValueError('nonfinite state/action: ' + str(path))
        for eid in np.unique(columns['episode_index']):
            if int(eid) not in expected or int(eid) in seen:
                raise ValueError('unknown or repeated physical episode: ' + str(eid))
            seen.add(int(eid))
            row = expected[int(eid)]
            mask = columns['episode_index'] == eid
            length = int(row['length'])
            if int(mask.sum()) != length or length < 1:
                raise ValueError('episode length mismatch: ' + str(eid))
            expected_file = root / info['data_path'].format(
                chunk_index=row['data/chunk_index'], file_index=row['data/file_index'])
            if expected_file != path or row['dataset_to_index'] - row['dataset_from_index'] != length:
                raise ValueError('episode file/range mismatch: ' + str(eid))
            if not np.all(columns['task_index'][mask] == row['task_index']):
                raise ValueError('episode task mismatch: ' + str(eid))
            if not np.array_equal(columns['frame_index'][mask], np.arange(length)):
                raise ValueError('episode frame_index mismatch: ' + str(eid))
            if not np.array_equal(columns['index'][mask], np.arange(row['dataset_from_index'], row['dataset_to_index'])):
                raise ValueError('episode global index mismatch: ' + str(eid))
            if not np.allclose(columns['timestamp'][mask], np.arange(length) / 30, atol=2e-5, rtol=0):
                raise ValueError('episode timestamp mismatch: ' + str(eid))
        actions = tensors['action'].astype(np.float64)
        count += len(actions)
        action_min = np.minimum(action_min, actions.min(axis=0))
        action_max = np.maximum(action_max, actions.max(axis=0))
        action_sum += actions.sum(axis=0)
        action_square_sum += np.square(actions).sum(axis=0)
    if seen != set(expected):
        raise ValueError('physical episode set differs from metadata')
    videos = {}
    annotation_mismatches = []
    for row in rows:
        for key in CAMERAS.values():
            prefix = 'videos/' + key + '/'
            path = root / info['video_path'].format(video_key=key,
                chunk_index=row[prefix + 'chunk_index'], file_index=row[prefix + 'file_index'])
            if path not in videos:
                with av.open(str(path)) as container:
                    stream = container.streams.video[0]
                    if float(stream.average_rate) != 30 or list(info['features'][key]['shape'][:2]) != [stream.height, stream.width]:
                        raise ValueError('video rate/shape mismatch: ' + str(path))
                    videos[path] = int(stream.frames)
            start, end = float(row[prefix + 'from_timestamp']), float(row[prefix + 'to_timestamp'])
            if start < 0 or abs((end - start) * 30 - row['length']) > 0.01 or round(end * 30) > videos[path]:
                raise ValueError('video episode span mismatch: ' + str(row['episode_index']))
        annotation = json.loads((root / row['annotation_path']).read_text())
        duration = validate_annotation(annotation)
        if duration != row['length']:
            annotation_mismatches.append(dict(episode_index=row['episode_index'], annotation_frames=duration,
                                             trajectory_frames=row['length']))
    mean = action_sum / count
    return dict(status='PASS', scope='structural audit only; no training or model validation',
        dataset=str(root), episodes=len(rows), rows=count, state_dim=61, selected_state_dim=25, action_dim=23,
        data_files=len(files), rgb_files=len(videos), video_spans_checked=len(rows) * 3,
        tasks=[dict(task_index=task, task_name=name,
                    episodes=sum(row['task_index'] == task for row in rows),
                    instance_ids=sorted({row['task_instance_id'] for row in rows if row['task_index'] == task}),
                    native_loader_text=native_tasks[task], task_description=descriptions[task]['task'])
               for task, name in TASKS.items()],
        annotation_files=len(rows), annotation_length_mismatches=len(annotation_mismatches),
        annotation_mismatch_examples=annotation_mismatches[:3], annotation_frame_alignment_verified=False,
        action_audit_statistics=dict(scope='all downloaded rows; not training normalization', count=count,
            min=action_min.tolist(), max=action_max.tolist(), mean=mean.tolist(),
            std=np.sqrt(np.maximum(0, action_square_sum / count - np.square(mean))).tolist()),
        metadata_sha256={p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [root / 'meta/info.json', root / 'meta/tasks.jsonl', root / 'meta/tasks.parquet',
                      root / 'meta/stats.json', *sorted((root / 'meta/episodes').glob('*/*.parquet'))]},
        source_stats_count=json.loads((root / 'meta/stats.json').read_text())['action'].get('count'),
        source_modality_present=(root / 'meta/modality.json').is_file(), native_train_ready=False)


def probe_native_loader(view, upstream):
    """Run pinned native data/video extraction on a preallocated loader view.

    The view must already have the official modality file. This function neither
    prepares data nor overwrites language or statistics and never loads a model.
    """
    import sys
    root = Path(inspect_upstream(upstream)['root'])
    view = Path(view).resolve(strict=True)
    if json.loads((view / 'meta/modality.json').read_text()) != modality_metadata():
        raise ValueError('loader view must use the pinned official R1Pro modality')
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('v12_native_r1pro', root / 'examples/b1k/r1pro.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    loader = LeRobotEpisodeLoader(view, MODALITY_CONFIGS['new_embodiment'], video_backend='opencv',
                                 data_cache_size=1, video_cache_size=1)
    samples = []
    for task in TASKS:
        positions = [i for i, row in enumerate(loader.episodes_metadata) if row['task_index'] == task]
        for position in [positions[0], positions[-1]]:
            row = loader.episodes_metadata[position]
            indices = [0, row['length'] // 2, row['length'] - 1]
            frame = loader.load_episode(position, needed_video_indices=indices)
            images = {key: [list(frame['video.' + key].iloc[i].shape) for i in indices] for key in CAMERAS}
            samples.append(dict(episode_index=row['episode_index'], rows=len(frame), frames=indices,
                image_shapes=images, text=frame['language.annotation.human.task_description'].iloc[0],
                state_widths={key: len(frame['state.' + key].iloc[0]) for key in STATE_GROUPS_2026},
                action_widths={key: len(frame['action.' + key].iloc[0]) for key in ACTION_GROUPS}))
    return dict(status='PASS', upstream_revision=inspect_upstream(upstream)['revision'],
                view=str(view), episodes=len(loader), video_backend='opencv',
                decoded_frames=sum(len(s['frames']) * len(CAMERAS) for s in samples), samples=samples,
                model_forward_backward_verified=False, training_ready=False)
