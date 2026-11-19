"""Episode isolation shared by single-task and joint training entry points."""


def validate_split(split):
    tasks = split.get("tasks", [split.get("task")])
    assert tasks and None not in tasks and len(set(tasks)) == len(tasks)
    train, validation = split["train"], split["validation"]
    train_count = split.get("train_episodes_per_task", 180)
    validation_count = split.get("validation_episodes_per_task", 20)
    assert type(train_count) is int and type(validation_count) is int
    assert train_count > 0 and validation_count >= 0 and train_count+validation_count == 200
    assert split.get("validation_enabled", True) == (validation_count > 0)
    assert len(train) == len(set(train)) == train_count * len(tasks)
    assert len(validation) == len(set(validation)) == validation_count * len(tasks)
    assert not set(train) & set(validation), "Episode leakage"
    mapping = {int(e): int(t) for e, t in split.get("episode_tasks", {}).items()}
    if len(tasks) == 1 and not mapping:
        mapping = {e: tasks[0] for e in train + validation}
    assert set(mapping) == set(train + validation)
    assert set(mapping.values()) == set(tasks)
    for task in tasks:
        assert sum(mapping[e] == task for e in train) == train_count
        assert sum(mapping[e] == task for e in validation) == validation_count
    return mapping
