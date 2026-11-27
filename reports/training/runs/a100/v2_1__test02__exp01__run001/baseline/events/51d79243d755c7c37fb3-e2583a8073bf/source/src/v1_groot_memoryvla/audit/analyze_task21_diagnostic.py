"""Episode-level paired summaries; never treats noise repeats as samples."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


def compare(run):
    with (run / "metrics/action_by_episode.csv").open() as f:
        data = list(csv.DictReader(f))
    comparisons = []
    for subset in ("uniform", "panel", "history_candidates", "post_merge"):
        rows = [r for r in data if r["subset"] == subset]
        episodes = sorted({int(r["episode"]) for r in rows})
        def get(eid, arm, condition, metric):
            return float(next(r[metric] for r in rows if int(r["episode"]) == eid
                              and r["arm"] == arm and r["condition"] == condition))
        for ref_arm, condition in (("baseline", "normal"), ("memory_v1", "empty"), ("memory_v1", "wrong")):
            pairs = []
            for eid in episodes:
                n = get(eid, "memory_v1", "normal", "mae_horizon32")
                reference = get(eid, ref_arm, condition, "mae_horizon32")
                pairs.append({"episode": eid, "normal_mae": n, "reference_mae": reference,
                              "normal_minus_reference": n-reference,
                              "relative_improvement": (reference-n)/reference})
            comparisons.append({"subset": subset, "reference_arm": ref_arm, "reference_condition": condition,
                "independent_episodes": len(episodes), "normal_lower_error_episodes": sum(p["normal_minus_reference"] < 0 for p in pairs),
                "episode_mean_delta": float(np.mean([p["normal_minus_reference"] for p in pairs])),
                "episode_mean_relative_improvement": float(np.mean([p["relative_improvement"] for p in pairs])),
                "per_episode": pairs})
    result = {"sampling_unit": "Episode", "action_noise_seeds_per_observation": 3,
              "interpretation": "paired descriptive diagnosis; one training seed, no claim of repeatable success gains",
              "comparisons": comparisons}
    (run / "metrics/paired_comparisons.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def schema(checkpoint):
    result = {}
    for shard in checkpoint.glob("model*.safetensors"):
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            result.update({k: f.get_slice(k).get_shape() for k in f.keys()})
    return result


def training_audit(run):
    data = {}
    for arm in ("baseline", "memory_v1"):
        records = [json.loads(line) for line in (run / arm / "metrics/metrics.jsonl").read_text().splitlines()]
        data[arm] = [r for r in records if r["kind"] == "train"]
    a, b = data["baseline"], data["memory_v1"]
    assert [(r["step"], r["group_id"], r["episode"]) for r in a] == [
           (r["step"], r["group_id"], r["episode"]) for r in b]
    grads = {k: {"min": min(r["memory_gradients"][k] for r in b),
                 "max": max(r["memory_gradients"][k] for r in b),
                 "mean": float(np.mean([r["memory_gradients"][k] for r in b]))}
             for k in b[0]["memory_gradients"]}
    assert all(np.isfinite(v["min"]) and v["min"] > 0 for v in grads.values())
    assert all(r["merges_per_bank"] == 16 for r in b)
    roots = Path("src/v1_groot_memoryvla")
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(roots.rglob("*.py"))
                     if ".venv" not in p.parts}
    (run / "config/source_sha256.json").write_text(json.dumps(source_hashes, indent=2)+"\n")
    result = {"optimizer_steps_each": len(a), "matched_sequence_order_and_budget": True,
              "memory_gradients_all_nonzero_finite_at_every_step": True,
              "gradient_norms_before_clip": grads, "writes_per_sequence": 32,
              "merges_per_bank_per_sequence": 16,
              "throughput_and_memory": {arm: {
                  "mean_cached_decisions_per_second": float(np.mean([r["decisions_per_second"] for r in rows])),
                  "peak_allocated_gib": max(r["peak_allocated_gib"] for r in rows),
                  "peak_reserved_gib": max(r["peak_reserved_gib"] for r in rows),
                  "total_training_seconds": sum(r["seconds"] for r in rows)} for arm, rows in data.items()}}
    (run / "diagnostics/training_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    curves = []
    for arm in ("baseline", "memory_v1"):
        rec = [json.loads(line) for line in (run / arm / "metrics/metrics.jsonl").read_text().splitlines()]
        for r in rec:
            if r["kind"] == "eval":
                for eid, loss in r["validation_by_episode"].items():
                    curves.append({"arm": arm, "step": r["step"], "validation_episode": eid,
                                   "episode_validation_loss": loss,
                                   "episode_mean_validation_loss": r["validation_loss"],
                                   "train_fixed_loss": r["train_fixed_loss"]})
    with (run / "metrics/loss_by_episode.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    result = compare(a.run)
    result["training_audit"] = training_audit(a.run)
    original = schema(Path("outputs/v1_groot_memoryvla/validation/acceptance/checkpoint"))
    current = schema(a.run / "memory_v1/checkpoints/final")
    assert current == original
    baseline = schema(a.run / "baseline/checkpoints/final")
    assert baseline == {k: v for k, v in current.items() if not k.startswith("memory.")}
    result["unchanged_model_parameter_schema"] = True
    (a.run / "diagnostics/final_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    report_path = a.run / "report.json"
    report = json.loads(report_path.read_text())
    report["analysis_artifacts"] = {"paired_episode_comparisons": "metrics/paired_comparisons.json",
                                    "training_audit": "diagnostics/training_audit.json",
                                    "loss_by_episode": "metrics/loss_by_episode.csv",
                                    "network_schema": "diagnostics/final_audit.json"}
    findings = a.run / "diagnostics/findings.json"
    if findings.exists():
        report["findings"] = json.loads(findings.read_text())
    report_path.write_text(json.dumps(report, indent=2)+"\n")
    for r in result["comparisons"]:
        print(r["subset"], r["reference_arm"], r["reference_condition"],
              "wins", r["normal_lower_error_episodes"], "/", r["independent_episodes"],
              "relative improvement", r["episode_mean_relative_improvement"])


if __name__ == "__main__":
    main()
