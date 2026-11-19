"""Single baseline runs use the same asynchronous controller as paired runs."""

import json
from types import SimpleNamespace

import pytest

from v3_bodymemory.experiment import launch_task21_ddp as launcher
from v3_bodymemory.experiment import task21_async


def arguments(tmp_path):
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"task": 0, "train": list(range(180)),
                                 "validation": list(range(180, 200))}))
    return SimpleNamespace(dataset=tmp_path / "data", checkpoint=tmp_path / "official",
        output=tmp_path / "run", split=split, arm="baseline", gpus=[1, 2, 3],
        paired=False, baseline_gpus=None, global_windows_per_step=3, steps=1000,
        microbatch=4, seed=42, learning_rate=1e-5, async_validation_gpu=1,
        baseline_validation_gpu=None, dry_run=False, adopt_running_memory=False,
        resume_step=0, allow_service_pids=[], train_only=False)


def test_single_baseline_dispatches_three_ranks_and_async_validation(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    observed = {}
    monkeypatch.setattr(launcher, "busy_gpus", lambda allowed: {0})
    monkeypatch.setattr(task21_async, "launch_async",
        lambda args, groups: observed.update(groups))
    launcher.launch(args)
    assert observed == {"baseline": [1, 2, 3]}
    plan = json.loads((args.output / "config/ddp_launch.json").read_text())
    assert plan["validation_parallel_with_training"]
    assert plan["global_decisions_per_step"] == 96
    argv = plan["commands"]["baseline"]["argv"]
    assert "--nproc-per-node=3" in argv and "--async-validation" in argv
    assert "--paired" not in argv


def test_separate_validator_gpu_is_checked_before_launch(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    args.async_validation_gpu = 0
    monkeypatch.setattr(launcher, "busy_gpus", lambda allowed: {0})
    with pytest.raises(AssertionError, match="Selected GPUs"):
        launcher.launch(args)
    assert not args.output.exists()


def test_completed_single_baseline_resumes_with_extended_budget(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    args.resume_step, args.steps = 1000, 5000
    config = args.output / "config"
    config.mkdir(parents=True)
    (config / "data_split.json").write_text(args.split.read_text())
    original = {k: str(v) if isinstance(v, type(tmp_path)) else v
                for k, v in vars(args).items()}
    original.update(world_size=3, steps=1000)
    arm = args.output / "baseline"
    (arm / "config").mkdir(parents=True)
    (arm / "config/ddp.json").write_text(json.dumps(original))
    checkpoint = arm / "checkpoints/step-001000"
    checkpoint.mkdir(parents=True)
    (checkpoint.parent / "latest.json").write_text(json.dumps(dict(step=1000,
        world_size=3, global_windows_per_step=3, seed=42, path=str(checkpoint))))
    (args.output / "status.json").write_text(json.dumps({"status": "completed"}))
    observed = {}
    monkeypatch.setattr(launcher, "busy_gpus", lambda allowed: {0})
    monkeypatch.setattr(task21_async, "launch_async", lambda args, groups: observed.update(groups))
    launcher.launch(args)
    assert observed == {"baseline": [1, 2, 3]}
    plan = json.loads((config / "ddp_resume_step-001000.json").read_text())
    assert plan["steps_each"] == 5000 and plan["resume_step"] == 1000
    argv = plan["commands"]["baseline"]["argv"]
    assert argv[-2:] == ["--resume-step", "1000"]
    assert json.loads((arm / "config/ddp.json").read_text())["steps"] == 1000


def test_resume_reopens_training_before_the_validator_starts(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    args.resume_step, args.steps = 1000, 5000
    status = args.output / "baseline/status.json"
    status.parent.mkdir(parents=True)
    status.write_text(json.dumps({"status": "completed", "step": 1000,
                                  "training_complete": True}))
    (args.output / "logs").mkdir()

    class ReachedValidator(Exception):
        pass

    class FinishedProcess:
        pid = 90000
        def poll(self):
            return 0

    def spawn(argv, **kwargs):
        if "torch.distributed.run" in argv:
            return FinishedProcess()
        state = json.loads(status.read_text())
        assert not state.get("training_complete", False)
        assert state["step"] == 1000 and state["total_steps"] == 5000
        raise ReachedValidator()

    monkeypatch.setattr(task21_async.subprocess, "Popen", spawn)
    with pytest.raises(ReachedValidator):
        task21_async.launch_async(args, {"baseline": [1, 2, 3]})


def test_training_curves_do_not_require_a_memory_arm(tmp_path):
    from v3_bodymemory.experiment.launch_task21_formal import plot
    metrics = tmp_path / "baseline/metrics/metrics.jsonl"
    metrics.parent.mkdir(parents=True)
    metrics.write_text(json.dumps({"kind": "train", "step": 1, "loss": .003,
        "decisions_per_second_end_to_end": 30, "peak_allocated_gib": 32})+"\n")
    (tmp_path / "plots").mkdir()
    plot(tmp_path)
    assert (tmp_path / "plots/training_curves.png").stat().st_size > 1000


def test_no_validation_launch_keeps_checkpoints_without_starting_validator(tmp_path,monkeypatch):
    args=arguments(tmp_path)
    args.no_validation=True
    args.async_validation_gpu=None
    args.split.write_text(json.dumps(dict(task=0,train=list(range(200)),validation=[],
        train_episodes_per_task=200,validation_episodes_per_task=0,validation_enabled=False)))
    launched=[]
    class Completed:
        pid=90000
        def wait(self):return 0
        def poll(self):return 0
    monkeypatch.setattr(launcher,'busy_gpus',lambda allowed:set())
    monkeypatch.setattr(launcher.subprocess,'Popen',lambda argv,**kw:launched.append(argv) or Completed())
    monkeypatch.setattr(launcher,'plot',lambda output:None)
    launcher.launch(args)
    assert len(launched)==1
    assert '--no-validation' in launched[0] and '--train-only' not in launched[0]
    assert '--async-validation' not in launched[0]
    assert json.loads((args.output/'selected_checkpoints.json').read_text())=={}
    assert json.loads((args.output/'status.json').read_text())['status']=='completed'
