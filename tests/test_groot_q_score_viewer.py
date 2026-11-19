"""Q must survive public result merges and remain distinct from task progress."""

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


output = load_module("groot_eval_output", ROOT / "scripts/groot_eval_output.py")
viewer = load_module("groot_result_viewer", ROOT / "evaluation/view_results.py")


@pytest.mark.parametrize("mode,expected_mean", [("official-eval", 0.25), ("fast-debug", None)])
def test_public_merge_counts_missing_q_and_excludes_invalid(tmp_path, mode, expected_mean):
    legacy = {"episode_index": 1, "success": False, "valid": True, "task_progress": 90, "sim_steps": 8}
    (tmp_path / "episodes.jsonl").write_text(json.dumps(legacy) + "\n")
    output.update_results(tmp_path, updates={"evaluation_mode": mode})
    for q_score, valid in [(0.0, True), (1.0, False), (0.5, True)]:
        output.update_results(tmp_path, episode={**legacy, "q_score": q_score, "valid": valid})
    # Status/per-GPU updates must preserve the journal-derived Q, including zero.
    summary = output.update_results(tmp_path, updates={"status": "complete", "per_gpu": {}})
    assert summary["mean_q_score"] == expected_mean
    assert summary["q_score_episodes"] == 2
    assert summary["missing_q_score_episodes"] == 1
    assert summary["completed_episodes"] == 3
    assert summary["mean_task_progress"] == 90
    assert summary["success_rate"] == (0.0 if mode == "official-eval" else None)
    records = output.read_records(tmp_path / "episodes.jsonl")
    assert "q_score" not in records[0]
    assert [record["q_score"] for record in records[1:]] == [0.0, 1.0, 0.5]


def test_viewer_uses_all_episodes_for_scores_and_episode_selection(tmp_path, capsys):
    run = tmp_path / "groot_task22_1"
    run.mkdir()
    (run / "results.json").write_text(json.dumps({
        "evaluation_mode": "official-eval", "status": "complete",
        "completed_episodes": 25, "requested_episodes": 25,
    }))
    records = [{"valid": True, "q_score": 1.0}] * 5 + [{"valid": True, "q_score": 0.0}] * 15 + [{"valid": True}] * 5
    (run / "episodes.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    loaded = viewer.load_runs(tmp_path)[0]
    assert len(loaded["_episodes"]) == 25
    assert loaded["mean_q_score"] == 0.25
    assert loaded["q_score_episodes"] == 20
    assert loaded["missing_q_score_episodes"] == 5
    viewer.print_report(tmp_path)
    report = capsys.readouterr().out
    assert "平均Q" in report and "0.2500" in report and "20/25" in report
    # Viewing legacy runs does not rewrite historical measurements.
    assert "mean_q_score" not in json.loads((run / "results.json").read_text())


def test_viewer_legacy_missing_q_remains_unknown(tmp_path):
    run = tmp_path / "groot_task22_1"
    run.mkdir()
    (run / "results.json").write_text(json.dumps({"evaluation_mode": "official-eval"}))
    (run / "episodes.jsonl").write_text(json.dumps({"valid": True, "task_progress": 100}) + "\n")
    loaded = viewer.load_runs(tmp_path)[0]
    assert loaded["mean_q_score"] is None
    assert loaded["q_score_episodes"] == 0
    assert loaded["missing_q_score_episodes"] == 1
