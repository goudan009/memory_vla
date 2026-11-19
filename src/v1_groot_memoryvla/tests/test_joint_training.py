"""Joint sampling must preserve Episode order, task balance and resumability."""

from collections import Counter, defaultdict
import json
from types import SimpleNamespace

import pytest

from v1_groot_memoryvla.data.episode_split import validate_split
from v1_groot_memoryvla.experiment.task21_ddp import TaskBalancedPlan
from v1_groot_memoryvla.experiment.task21_formal import donor_episode


def joint_split():
    tasks = [0, 34, 40]
    return dict(tasks=tasks,
        train=[t*10000+e for t in tasks for e in range(180)],
        validation=[t*10000+e for t in tasks for e in range(180, 200)],
        episode_tasks={str(t*10000+e): t for t in tasks for e in range(200)})


def test_joint_split_preserves_per_task_episode_isolation():
    split = joint_split()
    mapping = validate_split(split)
    assert len(mapping) == 600
    split['validation'][0] = split['train'][0]
    with pytest.raises(AssertionError):
        validate_split(split)


@pytest.mark.parametrize('count',[0,2,40,100])
def test_custom_ratios_keep_whole_episodes_and_each_task_balanced(count):
    split=joint_split()
    split.update(train=[t*10000+e for t in split['tasks'] for e in range(200-count)],
        validation=[t*10000+e for t in split['tasks'] for e in range(200-count,200)],
        train_episodes_per_task=200-count,validation_episodes_per_task=count,validation_enabled=count>0)
    assert len(validate_split(split))==600
    split['train_episodes_per_task']=199-count
    with pytest.raises(AssertionError):
        validate_split(split)


def dataset_fixture():
    groups, by_episode, tasks = [], {}, {}
    for task in [0, 34, 40]:
        for e in range(6):
            eid = task*10000+e
            by_episode[eid] = []
            tasks[eid] = task
            for w in range(2+e+(task % 3)):
                by_episode[eid].append(len(groups))
                groups.append((0, eid, w*256))
    return SimpleNamespace(groups=groups, by_episode=by_episode, task_by_episode=tasks,
        order=lambda seed: [indices[w] for w in range(10) for indices in by_episode.values()
                            if w < len(indices)])


def test_task_balance_order_ownership_and_resume_across_different_task_passes():
    dataset = dataset_fixture()
    plan = TaskBalancedPlan(dataset, slots=4, seed=42)
    resumed = TaskBalancedPlan(dataset, slots=4, seed=42)
    positions, counts = defaultdict(list), Counter()
    owners = {}
    for position in range(180):
        progress = plan.progress(position)
        counts[progress['task']] += 1
        batch = plan.batch(position)
        if position >= 71:
            assert resumed.batch(position) == batch
        for slot, index in enumerate(batch):
            if index is None:
                continue
            _, eid, frame = dataset.groups[index]
            assert dataset.task_by_episode[eid] == progress['task']
            key = (progress['task'], progress['task_epoch'], eid)
            assert owners.setdefault(key, slot) == slot
            positions[key].append(frame)
    assert set(counts.values()) == {60}
    assert max(k[1] for k in positions) >= 3
    for frames in positions.values():
        assert frames == list(range(0, len(frames)*256, 256))


def test_wrong_history_is_independent_and_from_same_task():
    train = SimpleNamespace(by_episode={10: [0], 340010: [1], 400010: [2]},
        task_by_episode={10: 0, 340010: 34, 400010: 40})
    validation = SimpleNamespace(task_by_episode={340020: 34})
    assert donor_episode(train, validation, 340020) == 340010


def test_new_identity_keeps_all_three_task_ids(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'train'))
    from run_identity import training_identity
    root = tmp_path/'V_1_memory_task0_34_40_train01'
    (root/'config').mkdir(parents=True)
    (root/'config/data_split.json').write_text(json.dumps(joint_split()))
    assert training_identity(root, 'memory_v1')['task_id'] == '0_34_40'


def test_queue_waits_for_all_four_final_validations(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'train'))
    import joint_task_queue as queue
    directory = tmp_path/'queues/previous'
    jobs = []
    for task in [34, 40]:
        for arm in ['baseline', 'memory_v1']:
            root = tmp_path/f'{task}_{arm}'
            queue.write(root/'config/data_split.json', dict(task=task))
            queue.write(root/arm/'status.json', dict(training_complete=True, step=3000))
            queue.write(root/'status.json', dict(status='completed'))
            queue.write(root/arm/'validation_status.json', dict(status='completed', validated_steps=[3000]))
            queue.write(root/arm/'checkpoints/step-003000/config.json', {})
            jobs.append(dict(output=str(root), arm=arm))
    queue.write(directory/'status.json', dict(status='completed', runs=jobs))
    config = dict(after_queue=str(directory))
    assert queue.dependency_state(config)[0]
    queue.write(root/arm/'validation_status.json', dict(status='validating', validated_steps=[1000, 2000]))
    assert not queue.dependency_state(config)[0]
