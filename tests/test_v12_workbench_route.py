import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evaluation" / "scripts"))
from groot_web_launch import LaunchManager
from v1_2_groot_baseline import contracts as c


def _fixture_checkpoint(root):
    (root / "config.json").write_text(json.dumps({
        "model_type": "Gr00tN1d7", "action_horizon": 40,
        "max_action_dim": 132, "max_state_dim": 132,
    }))
    modes = {
        k: {"modality_keys": list(v), "delta_indices": list(range(16)) if k == "action" else [0]}
        for k, v in {"state": c.STATE_GROUPS_2026, "action": c.ACTION_GROUPS,
                     "video": ["head", "left_wrist", "right_wrist"],
                     "language": ["annotation.human.task_description"]}.items()
    }
    modes["state"]["modality_keys"] = list(c.STATE_GROUPS_2026)
    modes["action"]["action_configs"] = [
        {"rep": "RELATIVE" if k in ("torso", "left_arm", "right_arm") else "ABSOLUTE",
         "type": "NON_EEF", "format": "DEFAULT",
         "state_key": k if k in ("torso", "left_arm", "right_arm") else None,
         "is_gripper": "gripper" in k}
        for k in c.ACTION_GROUPS
    ]
    (root / "processor_config.json").write_text(json.dumps({
        "processor_class": "Gr00tN1d7Processor", "processor_kwargs": {
            "modality_configs": {"new_embodiment": modes}, "max_action_horizon": 40,
            "max_action_dim": 132, "max_state_dim": 132, "use_relative_action": True,
        }}))
    stats = {}
    for kind, layout in (("state", c.STATE_GROUPS_2026), ("action", c.ACTION_GROUPS),
                         ("relative_action", {k: c.ACTION_GROUPS[k] for k in ("torso", "left_arm", "right_arm")})):
        stats[kind] = {k: {"min": [-1] * (b - a), "max": [1] * (b - a)} for k, (a, b) in layout.items()}
    (root / "statistics.json").write_text(json.dumps({"new_embodiment": stats}))
    (root / "model.safetensors").write_bytes(b"fixture")


def test_v12_preview_binds_native_route(tmp_path, monkeypatch):
    upstream = tmp_path / "upstream"
    (upstream / "examples/b1k").mkdir(parents=True)
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    _fixture_checkpoint(ckpt)
    python = tmp_path / "python"
    python.touch()
    behavior_python = tmp_path / "behavior-python"
    behavior_python.touch()
    monkeypatch.setattr(c, "inspect_upstream", lambda root: {"root": str(root), "revision": c.FORK_REVISION})
    monkeypatch.setenv("V12_GROOT_ROOT", str(upstream))
    monkeypatch.setenv("V12_GROOT_PYTHON", str(python))
    monkeypatch.setenv("V12_BEHAVIOR_ROOT", str(tmp_path / "behavior"))
    monkeypatch.setenv("V12_BEHAVIOR_PYTHON", str(behavior_python))
    monkeypatch.setenv("V12_CHECKPOINT", str(ckpt))
    monkeypatch.setenv("V12_CHECKPOINT_SHA256", c.checkpoint_digest(ckpt))
    (tmp_path / "behavior").mkdir()
    manager = LaunchManager(tmp_path, log_root=tmp_path / "logs")
    request = {"version": "V_1_2", "protocol": c.B1K2026, "instance": 3,
               "rollouts": 1, "gpu_index": 1, "notes": "V1.2 / N1.7 / B1K2026: official radio, RTC off"}
    prepared = manager.prepare(request)
    plan = prepared["plan"]
    assert plan["kind"] == c.B1K2026 and plan["instances"] == {"1": [3]}
    assert plan["notes"].startswith("V1.2 / N1.7")
    assert "run_groot_eval.sh" not in prepared["command"]
    assert "serve_b1k.py" in json.dumps(plan["policy_identity"])
    with pytest.raises(ValueError, match="B1K-2026"):
        manager.prepare({**request, "protocol": c.B1K2025})
