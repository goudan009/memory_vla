#!/usr/bin/env python3
"""Execute archived Task40 observations on the existing PolicyServer.

Batch JSON contains `{"decisions": [{"target": "path/to/0007"}]}`.
Each entry can also specify `donor` and `kinematics` paths. Relative paths
resolve against --root. Pass --batch outside the new --output directory.

Use the evaluation Python environment with NumPy, Torch and the GR00T
PolicyClient. The workbench supplies the A100 endpoint and checkpoint identity
in its job metadata. The default matrix tests cognitive history removal, an
optional cognitive donor, and normal memory at 1x, 2x, 5x and 10x in EOS and
DiT conditions. Top-level scales, locations and detail can reduce that matrix.
Set paired_donor to compare the donor at every selected scale and location.
"""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import logging
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from task40_memory_instrumentation import (
    BANKS, SCHEMA_VERSION, apply_memory_delta, pack_memory, unpack_memory,
)


LOG = logging.getLogger("task40_forward")
ENDPOINT = "task40_frozen_forward"
PROBE_FIELDS = ("current_pooled", "cognitive_bank_pooled",
                "cognitive_readout_pooled", "dit_condition_pooled")


def read_json(path):
    return json.loads(path.read_text())


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value {type(value)}")


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, default=jsonable, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def resolve(root, value):
    return (root / value).resolve()


def slot_key(session):
    env, = session["env_ids"]
    episode, = session["episode_ids"]
    return str(session["session_id"]), str(env), str(episode)


def artifact(entry, record, name):
    manifest = record.get("memory_artifacts", {})
    check(name in manifest, f"{entry} has no recorded {name}.pt. An instrumented collection is required.")
    ref = manifest[name]
    filename = name + ".pt"
    check(ref["file"] == filename, f"{entry} references an unexpected {name} filename")
    payload = (entry / filename).read_bytes()
    check(sha256(payload).hexdigest() == ref["sha256"], f"{entry / filename} digest mismatch")
    return payload


def check_clock(snapshot, key, decision):
    expected = {key[:2]: (key, decision)} if decision or snapshot["slots"] else {}
    check(snapshot["slots"] == expected, f"Memory session or decision clock mismatch at {decision}")
    for name in BANKS:
        for field in ("bank", "last_written", "merge_count"):
            check(not (snapshot[name][field].keys() - {key}), f"Unexpected session in {name}.{field}")


def reconstruct(entry, record):
    """Use the nearest base and preceding deltas. The target delta is validated but excluded."""
    decision = record["decision"]
    key = slot_key(record["session"])
    check(entry.name.isdecimal() and int(entry.name) == decision and decision >= 0,
          f"Decision directory does not match decision.json: {entry}")
    # Collection uses at least four digits and extends the width at decision 10000.
    pending = []
    cursor = decision
    previous, metadata = entry, record
    while "memory_base" not in metadata.get("memory_artifacts", {}):
        check(cursor > 0, f"No memory_base.pt found in the trace ending at {entry}")
        cursor -= 1
        previous = entry.parent / f"{cursor:04d}"
        metadata = read_json(previous / "decision.json")
        check(metadata["decision"] == cursor and slot_key(metadata["session"]) == key,
              f"Discontinuous decision or session at {previous}")
        pending.append((previous, metadata))
    snapshot = unpack_memory(artifact(previous, metadata, "memory_base"))
    check_clock(snapshot, key, cursor)
    for directory, row in reversed(pending):
        snapshot = apply_memory_delta(snapshot, artifact(directory, row, "memory_delta"))
        check_clock(snapshot, key, row["decision"] + 1)
    check_clock(snapshot, key, decision)
    after = apply_memory_delta(snapshot, artifact(entry, record, "memory_delta"))
    check_clock(after, key, decision + 1)
    retrieval, = record["retrieval"]["observations"]
    for name in BANKS:
        times = [step for step, _ in snapshot[name]["bank"].get(key, ())]
        check(list(retrieval[name]["history_timesteps"]) == times,
              f"{entry} reconstructed {name} differs from the online retrieval clock")
    return snapshot, {"base": str(previous), "deltas_applied": len(pending)}


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def load_analysis(root):
    scripts = root / "Output/test_01/experiment_09/scripts"
    sys.path.insert(0, str(scripts))
    import compare_forward_artifacts
    return compare_forward_artifacts


def stage_for(entry, record):
    """Same-stage donor matching uses physical state through the pre-action step."""
    from audit_trajectories import reconstruct as reconstruct_labels, verify_state

    initial = read_json(entry.parent / "initial.json")["truth"]
    start = record["start_step"]
    check(initial["step"] <= start, f"Invalid physical start step at {entry}")
    rows = [initial]
    with (entry.parent / "truth.jsonl").open() as stream:
        for line in stream:
            if rows[-1]["step"] == start:
                break
            row = json.loads(line)
            check(row["step"] == rows[-1]["step"] + 1, f"Missing physical step at {entry}")
            rows.append(row)
    check(rows[-1]["step"] == start, f"Physical trace does not reach the pre-action step at {entry}")
    verify_state(rows[-1], record["state_before"], f"Pre-action state mismatch at {entry}")
    labels, _ = reconstruct_labels(rows, initial["hinge"], initial["step"] == 0, initial["step"])
    stage = {name: labels[start][name]
             for name in ("attempted_while_incomplete", "door_completed", "open_now", "attempt_exclusion")}
    return {**stage, "toggled_on": rows[-1]["ToggledOn"], "inside": rows[-1].get("Inside")}


def replay_snapshot(client, entry, record, cache):
    """Build a checkpoint-native stream from the archived prefix."""
    parent, decision = entry.parent, record["decision"]
    session, key = record["session"], slot_key(record["session"])
    if cache.get("parent") != parent or cache["decision"] > decision:
        client.reset()
        cache.clear()
        cache.update(parent=parent, decision=0, digest=sha256())
    for index in range(cache["decision"], decision):
        folder = parent / f"{index:04d}"
        meta = read_json(folder / "decision.json")
        check((meta["decision"], meta["start_step"], meta["end_step"], meta["session"]) ==
              (index, index * 8, (index + 1) * 8, session), f"Invalid replay history at {folder}")
        vision = folder / "observation.npz"
        cache["digest"].update(sha256(vision.read_bytes()).digest())
        cache["digest"].update(json.dumps(meta["language"], sort_keys=True).encode())
        client.call_endpoint("get_action", {"observation": load_arrays(vision) | meta["language"],
                                             "options": session})
    cache["decision"] = decision
    reply = client.call_endpoint("task40_memory_snapshot", {"session": session})
    snapshot = unpack_memory(reply["memory"])
    check_clock(snapshot, key, decision)
    return snapshot, {"replayed": decision, "input_sha256": cache["digest"].hexdigest()}


def donor_for(root, row, target, target_record, target_snapshot, stage, analysis, reconstruct_state=reconstruct):
    if not row.get("donor"):
        return None, None
    donor = resolve(root, row["donor"])
    donor_record = read_json(donor / "decision.json")
    target_key = slot_key(target_record["session"])
    donor_key = slot_key(donor_record["session"])
    check(donor.parent != target.parent and donor_key != target_key, "Donor must come from another episode")
    check(donor_record["decision"] == target_record["decision"], "Donor must match the target decision clock")
    donor_stage = stage_for(donor, donor_record)
    relation = row.get("donor_relation", "same_stage")
    check(stage["door_completed"] is not None and donor_stage["door_completed"] is not None,
          "Target and donor must have verified history through the pre-action step")
    if relation == "same_stage":
        check(donor_stage == stage, "Donor must match the target's verified pre-action Task40 stage")
    elif relation == "different_history":
        check(donor_stage["door_completed"] != stage["door_completed"]
              and donor_stage["open_now"] == stage["open_now"],
              "History-only donor must differ in previous opening and match current door truth")
    elif relation == "different_stage":
        check(donor_stage["door_completed"] != stage["door_completed"]
              or donor_stage["attempted_while_incomplete"] != stage["attempted_while_incomplete"],
              "Different-stage donor lacks a confirmed stage contrast")
    else:
        raise ValueError(f"Unsupported donor relation: {relation}")
    snapshot, lineage = reconstruct_state(donor, donor_record)
    original = target_snapshot["cognitive"]["bank"].get(target_key, [])
    replacement = snapshot["cognitive"]["bank"].get(donor_key, [])
    check(bool(original) and len(original) == len(replacement), "Donor cognitive occupancy must match and be nonzero")
    check(target_snapshot["cognitive"]["capacity"] == snapshot["cognitive"]["capacity"],
          "Donor cognitive bank capacity differs")
    contrasts = []
    for (target_time, base), (donor_time, other) in zip(original, replacement):
        check(base.shape == other.shape and base.dtype == other.dtype, "Donor slot shape or dtype differs")
        contrasts.append({"target_time": target_time, "donor_time": donor_time,
                          **analysis.array_contrast(base.float().numpy(), other.float().numpy())})
    check(any(item["changed_fraction"] for item in contrasts), "Donor cognitive content is identical to the target")
    return pack_memory(snapshot), {"source": str(donor), "lineage": lineage, "slots": contrasts,
                                   "stage": donor_stage, "relation": relation}


def conditions(scales, locations, use_donor, paired_donor=False):
    yield "A_normal", 1.0, "eos", (), ()
    yield "B_empty_cognitive", 1.0, "eos", ("cognitive",), ()
    if use_donor:
        yield "C_donor_cognitive", 1.0, "eos", (), ("cognitive",)
    for scale in scales:
        for location in locations:
            if (scale, location) != (1.0, "eos"):
                yield scale_name(location, scale), scale, location, (), ()
                if use_donor and paired_donor:
                    yield scale_name(location, scale, "C_donor"), scale, location, (), ("cognitive",)


def scale_name(location, scale, prefix="A"):
    label = str(int(scale)) if scale.is_integer() else str(scale)
    return f"{prefix}_{location}_{label}x"


def summarize_banks(snapshot, key):
    return {
        name: {"capacity": snapshot[name]["capacity"],
               "slot_times": [t for t, _ in snapshot[name]["bank"].get(key, ())],
               "merge_count": snapshot[name]["merge_count"].get(key, 0),
               "last_written": snapshot[name]["last_written"].get(key)}
        for name in BANKS
    }


def run_decision(args, row, folder, client, analysis, config, progress):
    source = resolve(args.root, row["target"])
    record = read_json(source / "decision.json")
    def native(entry, meta):
        return replay_snapshot(client, entry, meta, progress_cache)
    progress_cache = getattr(args, "replay_cache", {})
    reconstruct_state = native if config["replay_prefix"] else reconstruct
    snapshot, lineage = reconstruct_state(source, record)
    session, decision = record["session"], record["decision"]
    key = slot_key(session)
    bank_summary = summarize_banks(snapshot, key)
    memory = pack_memory(snapshot)
    rng = artifact(source, record, "rng_before")
    rng_values = unpack_memory(rng)
    stage = stage_for(source, record) if row.get("donor") else None
    donor, donor_info = donor_for(args.root, row, source, record, snapshot, stage, analysis, reconstruct_state)
    observation = load_arrays(source / "observation.npz")
    language = record["language"]
    check(sum(name.startswith("video.") for name in observation) == 3
          and any(name.startswith("state.") for name in observation)
          and language and all(name.startswith("annotation.") for name in language)
          and not (observation.keys() & language.keys()), f"Incomplete or invalid policy input at {source}")
    observation.update(language)
    online_actions = load_arrays(source / "proposed.npz")
    online_features = load_arrays(source / "model_features.npz")
    check("dit_condition_eos" in online_features and "normalized_actions" in online_features,
          f"{source} lacks the instrumented online DiT and action features")
    labels = list(conditions(config["scales"], config["locations"], donor is not None, config["paired_donor"]))
    rendered_inputs, renderer_sources = {}, {}
    if row.get("restore_rgb_frames"):
        runs = args.root / "Output/test_01/experiment_10/runs"
        cameras = {name for name in observation if name.startswith("video.")}
        for index, value in enumerate(row["restore_rgb_frames"], 1):
            frame = resolve(args.root, value)
            check(frame.is_relative_to(runs.resolve()) and frame.name == "render_06.npz",
                  "Restore RGB diagnostic requires a sixth-refresh frame from Experiment 10")
            recovered = load_arrays(frame)
            check(set(recovered) == observation.keys() - language.keys()
                  and all(np.array_equal(recovered[key], observation[key]) for key in recovered.keys() - cameras)
                  and all((recovered[key].shape, recovered[key].dtype) == (observation[key].shape, observation[key].dtype)
                          for key in cameras), "Recovered observation changed state or RGB format")
            name = f"restore_rgb_{index}"
            rendered_inputs[name] = observation | {key: recovered[key] for key in cameras}
            renderer_sources[name] = {"frame": str(frame), "sha256": sha256(frame.read_bytes()).hexdigest()}
        rendered_inputs["restore_rgb_1_repeat"] = rendered_inputs["restore_rgb_1"]
        labels = [labels[0]] + [(name, 1.0, "eos", (), ()) for name in rendered_inputs]
    source_metadata = {
        "source": str(source), "decision": decision, "session": session, "stage": stage,
        "start_step": record["start_step"], "seed": record.get("seed"), "instance": record.get("instance"),
        "memory_lineage": lineage, "banks": bank_summary, "donor": donor_info,
        "input_memory_sha256": sha256(memory).hexdigest(), "input_rng_sha256": sha256(rng).hexdigest(),
        "input_observation_sha256": sha256((source / "observation.npz").read_bytes()).hexdigest(),
        "language": language,
    }
    if renderer_sources:
        source_metadata["restore_rgb_sources"] = renderer_sources
    (folder / "memory_before.pt").write_bytes(memory)
    (folder / "rng_before.pt").write_bytes(rng)
    if donor is not None:
        (folder / "donor_memory_before.pt").write_bytes(donor)
    write_json(folder / "source.json", source_metadata)

    comparisons = {}
    variants = {}
    baseline_features = baseline_actions = baseline_rng = baseline_retrieval = None
    fusion_modes = set()
    for ordinal, (name, scale, location, empty, swap) in enumerate(labels, 1):
        progress.update(condition=name, condition_index=ordinal, conditions=len(labels))
        write_json(args.output / "metrics" / "active_stage.json", progress)
        LOG.info("decision=%s source=%s condition=%s", decision, source, name)
        request = {"observation": rendered_inputs.get(name, observation), "session": session, "memory": memory, "rng": rng,
                   "scale": scale, "location": location, "detail": config["detail"],
                   "empty_banks": list(empty), "donor_banks": list(swap)}
        if swap:
            request["donor_memory"] = donor
        response = client.call_endpoint(ENDPOINT, request)
        trace = response["trace"]
        fusion_modes.add(trace.get("fusion_mode", "legacy"))
        check(response["schema_version"] == SCHEMA_VERSION and response["decision"] == decision
              and trace["input_memory_sha256"] == source_metadata["input_memory_sha256"]
              and trace["input_rng_sha256"] == source_metadata["input_rng_sha256"]
              and (trace["scale_multiplier"], trace["location"], trace["detail"]) == (scale, location, config["detail"])
              and trace["empty_banks"] == list(empty) and trace["donor_banks"] == list(swap),
              f"RPC input or intervention differs at {name}")
        if swap:
            check(trace["donor_memory_sha256"] == sha256(donor).hexdigest(), "RPC donor digest differs")
        retrieval = response["retrieval"]
        observed, = retrieval["observations"]
        for bank in BANKS:
            check(list(observed[bank]["history_timesteps"])
                  == ([] if bank in empty else bank_summary[bank]["slot_times"]),
                  f"RPC retrieved a different set of {bank} time slots at {name}")
        features, actions = response["model_features"], response["actions"]
        check(actions.keys() == online_actions.keys()
              and all(actions[key].shape == array.shape for key, array in online_actions.items()),
              f"RPC and online action schemas differ at {name}")
        rng_after = response["rng_after"]
        output = folder / name
        output.mkdir()
        np.savez_compressed(output / "model_features.npz", **features)
        np.savez_compressed(output / "proposed.npz", **actions)
        (output / "rng_after.pt").write_bytes(rng_after)
        write_json(output / "trace.json", {"trace": trace, "retrieval": retrieval})
        rng_state = unpack_memory(rng_after)
        check(len(rng_state[3]) == len(rng_values[3]), "RPC CUDA RNG device count differs")
        if name == "A_normal":
            baseline_features, baseline_actions, baseline_rng = features, actions, rng_state
            baseline_retrieval = retrieval
            comparisons["online_to_A_normal"] = {
                **analysis.contrast(online_features, features, online_actions, actions),
                "retrieval": analysis.retrieval_contrast(record["retrieval"], retrieval),
            }
        else:
            comparisons[f"A_normal_to_{name}"] = {
                **analysis.contrast(baseline_features, features, baseline_actions, actions),
                "retrieval": analysis.retrieval_contrast(baseline_retrieval, retrieval),
            }
            fixed_fields = ("input_ids", "normalized_state", "dit_state_features", "embodiment_id")
            if name not in rendered_inputs:
                fixed_fields += ("pre_backbone_features_eos", "current_pooled")
            check(all(np.array_equal(baseline_features[field], features[field]) for field in fixed_fields),
                  f"Fixed model inputs differ at {name}")
        variants[name] = {
            "trace_file": f"{name}/trace.json",
            "rng_after_matches_normal": analysis.rng_equal(baseline_rng, rng_state),
            "cognitive_residual": analysis.residual_summary(features),
        }

    def compare_saved(first, second):
        left, right = folder / first, folder / second
        return {
            **analysis.contrast(load_arrays(left / "model_features.npz"), load_arrays(right / "model_features.npz"),
                                load_arrays(left / "proposed.npz"), load_arrays(right / "proposed.npz")),
            "retrieval": analysis.retrieval_contrast(read_json(left / "trace.json")["retrieval"],
                                                     read_json(right / "trace.json")["retrieval"]),
        }

    if donor is not None:
        first, second = "B_empty_cognitive", "C_donor_cognitive"
        comparisons[f"{first}_to_{second}"] = compare_saved(first, second)
        if config["paired_donor"]:
            for scale in config["scales"]:
                for location in config["locations"]:
                    if (scale, location) != (1.0, "eos"):
                        first, second = scale_name(location, scale), scale_name(location, scale, "C_donor")
                        comparisons[f"{first}_to_{second}"] = compare_saved(first, second)
    if rendered_inputs:
        for second in ("restore_rgb_1_repeat", "restore_rgb_2"):
            comparisons[f"restore_rgb_1_to_{second}"] = compare_saved("restore_rgb_1", second)
    if {"eos", "dit_condition"}.issubset(set(config["locations"])):
        for scale in config["scales"]:
            first = "A_normal" if scale == 1 else scale_name("eos", scale)
            second = scale_name("dit_condition", scale)
            comparisons[f"{first}_to_{second}"] = compare_saved(first, second)
    online = comparisons["online_to_A_normal"]
    # Online collection is compact. Compare every recorded field and allow additional full trace fields.
    online_exact = (analysis.all_equal(online["features"], allow_extra_right=True)
                    and analysis.all_equal(online["actions"]))
    check(len(fusion_modes) == 1, "A frozen batch changed the loaded fusion path")
    fusion_mode, = fusion_modes
    if config.get("allow_cross_checkpoint", False):
        check(fusion_mode in ("route_a", "route_b"), "Expected an Experiment 10 learned-fusion model")
        fixed_fields = ("input_ids", "normalized_state", "pre_backbone_features_eos",
                        "current_pooled", "dit_state_features", "embodiment_id")
        check(all(np.array_equal(online_features[field], baseline_features[field]) for field in fixed_fields),
              "Cross-checkpoint forward changed a frozen model input")
    else:
        check(fusion_mode in ("legacy", "openvla"), "Learned-fusion comparisons require allow_cross_checkpoint")
    results = {"source": str(source), "decision": decision, "stage": stage,
               "baseline_matches_online": online_exact, "fusion_mode": fusion_mode,
               "variants": variants, "comparisons": comparisons,
               "kinematics": analysis.action_targets(folder, source, row.get("kinematics"), args.root, record)}
    write_json(folder / "contrasts.json", results)
    if ((not online_exact and not config.get("allow_cross_checkpoint", False))
            or any(not row["rng_after_matches_normal"] for row in variants.values())):
        LOG.warning("decision=%s has an online baseline mismatch or differing post-forward RNG", decision)
    return {key: results[key] for key in ("source", "decision", "stage", "baseline_matches_online", "fusion_mode")} | {
        "metrics": str((folder / "contrasts.json").relative_to(args.output)),
        "all_rng_after_match": all(row["rng_after_matches_normal"] for row in variants.values()),
    }


def load_plan(path):
    plan = read_json(path)
    rows = plan["decisions"]
    check(isinstance(rows, list) and rows and all(isinstance(row.get("target"), str) for row in rows),
          "Batch must list at least one target decision")
    scales = [float(scale) for scale in plan.get("scales", [1, 2, 5, 10])]
    locations = plan.get("locations", ["eos", "dit_condition"])
    detail = plan.get("detail", "full")
    paired_donor = plan.get("paired_donor", False)
    cross_checkpoint = plan.get("allow_cross_checkpoint", False)
    replay_prefix = plan.get("replay_prefix", False)
    check(scales and len(scales) == len(set(scales)) and all(math.isfinite(s) and s >= 0 for s in scales),
          "Scales must be distinct, finite and nonnegative")
    check(isinstance(locations, list) and locations and len(locations) == len(set(locations))
          and set(locations) <= {"eos", "dit_condition"}, "Specify unique eos and/or dit_condition locations")
    check(detail in {"full", "compact"}, "Detail must be full or compact")
    check(type(paired_donor) is bool and (not paired_donor or all(row.get("donor") for row in rows)),
          "Paired donor grid requires a donor for every decision")
    check(type(cross_checkpoint) is bool and (not cross_checkpoint or (scales == [1.] and locations == ["eos"])),
          "Cross-checkpoint comparisons must use native scale and fusion location")
    check(type(replay_prefix) is bool and not (replay_prefix and cross_checkpoint),
          "Prefix replay and legacy cross-checkpoint mode are mutually exclusive")
    for row in rows:
        if "restore_rgb_frames" in row:
            frames = row["restore_rgb_frames"]
            check(isinstance(frames, list) and len(frames) == 2 and all(isinstance(path, str) for path in frames)
                  and frames[0] != frames[1] and not row.get("donor") and not cross_checkpoint
                  and scales == [1.] and locations == ["eos"],
                  "Restore RGB diagnostic requires two distinct frames and native normal memory")
    return {"decisions": rows, "scales": scales, "locations": locations, "detail": detail,
            "paired_donor": paired_donor, "allow_cross_checkpoint": cross_checkpoint, "replay_prefix": replay_prefix}


def connect(args):
    groot = Path(os.environ.get("GROOT_REPO", "/nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1"))
    sys.path.insert(0, str(groot))
    import zmq
    from gr00t.policy.server_client import PolicyClient

    client = PolicyClient(host=args.policy_host, port=args.port)
    client.socket.setsockopt(zmq.LINGER, 0)
    for option in (zmq.RCVTIMEO, zmq.SNDTIMEO):
        client.socket.setsockopt(option, args.rpc_timeout * 1000)
    return client


def export_probe(args, client, episodes, results, progress):
    metrics = args.output / "metrics"
    check(isinstance(episodes, list) and episodes, "Missing probe cohort")
    for ordinal, row in enumerate(episodes, 1):
        target = resolve(args.root, row["episode"])
        check(target.is_relative_to(args.root / "Output/test_01"), "Invalid probe episode path")
        decisions = row["decisions"]
        check(decisions and decisions == sorted(set(decisions)) and decisions[0] >= 0,
              "Probe decisions must be ordered and unique")
        client.reset()
        values = {field: [] for field in PROBE_FIELDS}
        selected = set(decisions)
        session = read_json(target / "0000/decision.json")["session"]
        digest = sha256()
        for index in range(decisions[-1] + 1):
            if index % 10 == 0:
                progress.update(stage="replaying_probe", index=ordinal, total=len(episodes),
                                episode=str(target), decision=index)
                write_json(metrics / "active_stage.json", progress)
            entry = target / f"{index:04d}"
            record = read_json(entry / "decision.json")
            check((record["decision"], record["start_step"], record["session"]) ==
                  (index, index * 8, session), f"Probe prefix discontinuity at {entry}")
            obs = entry / "observation.npz"
            digest.update(sha256(obs.read_bytes()).digest())
            digest.update(json.dumps(record["language"], sort_keys=True).encode())
            observation = load_arrays(obs) | record["language"]
            if index not in selected:
                client.call_endpoint("get_action", {"observation": observation, "options": session})
                continue
            response = client.call_endpoint("task40_trace_infer", {
                "observation": observation, "session": session, "decision": index, "detail": "compact"})
            check(response["schema_version"] == SCHEMA_VERSION and response["decision"] == index,
                  f"Probe clock mismatch at {entry}")
            for field in PROBE_FIELDS:
                vector = np.asarray(response["model_features"][field])
                check(vector.ndim == 1 and vector.size and np.isfinite(vector).all(),
                      f"Invalid probe field {field} at {entry}")
                values[field].append(vector)
        snapshot = unpack_memory(client.call_endpoint("task40_memory_snapshot", {"session": session})["memory"])
        check_clock(snapshot, slot_key(session), decisions[-1] + 1)
        name = f"probe_{ordinal:04d}.npz"
        np.savez_compressed(metrics / name, decisions=decisions,
                            **{field: np.stack(vectors) for field, vectors in values.items()})
        results["decisions"].append({"episode": str(target), "rows": len(decisions),
                                     "file": f"metrics/{name}", "prefix_sha256": digest.hexdigest()})
        progress["completed"] = ordinal
        LOG.info("probe episode %s/%s exported %s decisions from %s",
                 ordinal, len(episodes), len(decisions), target)


def main(args, client_factory=connect):
    args.root = args.root.resolve()
    args.output = args.output.resolve()
    exp09 = args.root / "Output/test_01/experiment_09"
    for parent in args.output.parents:
        if parent.parent == exp09.parent and parent.name.startswith("experiment_"):
            group = json.loads(args.experiment_plan).get("group", exp09.name)
            check(group in (exp09.name, "experiment_10", "experiment_11", "experiment_12") and parent.name == group,
                  "Frozen forward output must match its registered experiment")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "metrics"):
        (args.output / name).mkdir()
    with (args.output / "logs" / "forward.log").open("x", buffering=1) as stream:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        LOG.addHandler(handler)
        LOG.setLevel(logging.INFO)
        try:
            return execute(args, client_factory)
        finally:
            LOG.removeHandler(handler)


def execute(args, client_factory):
    results = {"status": "failed", "decisions": [], "started_at": now()}
    start = time.monotonic()
    metrics = args.output / "metrics"
    progress = {"stage": "loading_batch", "completed": 0}
    write_json(metrics / "active_stage.json", progress)
    try:
        raw = read_json(args.batch)
        if "probe_episodes" in raw:
            job = {**json.loads(args.experiment_plan or "{}"), "phase": "native_probe_export",
                   "endpoint": "task40_trace_infer", "batch": str(args.batch.resolve())}
            write_json(args.output / "job.json", job)
            results["configuration"] = job
            export_probe(args, client_factory(args), raw["probe_episodes"], results, progress)
            results["status"] = "completed"
            return 0
        configuration = load_plan(args.batch)
        for row in configuration["decisions"]:
            for name in ("target", "donor"):
                if row.get(name):
                    check(not args.output.is_relative_to(resolve(args.root, row[name]).parent),
                          "Frozen forward output must not be inside a source trajectory")
        job = {**json.loads(args.experiment_plan or "{}"), "kind": "task40_forward", "task_id": 40,
               "output": str(args.output), "batch": str(args.batch.resolve()), "phase": "frozen_forward",
               "policy_host": args.policy_host, "policy_port": args.port,
               "endpoint": ENDPOINT, "configuration": configuration, "started_at": results["started_at"]}
        write_json(args.output / "job.json", job)
        results["configuration"] = job
        analysis = load_analysis(args.root)
        client = client_factory(args)
        args.replay_cache = {}
        for ordinal, row in enumerate(configuration["decisions"], 1):
            progress.update(stage="frozen_forward", index=ordinal, total=len(configuration["decisions"]),
                            target=row["target"], condition=None)
            write_json(metrics / "active_stage.json", progress)
            folder = metrics / f"decision_{ordinal:04d}"
            folder.mkdir()
            results["decisions"].append(run_decision(args, row, folder, client, analysis, configuration, progress))
            progress["completed"] = len(results["decisions"])
        results["status"] = "completed"
        print(f"TASK40_FORWARD_COMPLETE {args.output}", flush=True)
        return 0
    except Exception as error:
        LOG.exception("Frozen-forward batch failed")
        results["error"] = f"{type(error).__name__}: {error}"
        print(results["error"], file=sys.stderr, flush=True)
        return 1
    finally:
        progress["stage"] = results["status"]
        results.update(finished_at=now(), elapsed_seconds=time.monotonic() - start)
        write_json(metrics / "active_stage.json", progress)
        write_json(args.output / "results.json", results)


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--batch", type=Path, required=True, help="JSON batch specification")
    parser.add_argument("--experiment-plan", help="JSON metadata supplied by LaunchManager")
    parser.add_argument("--rpc-timeout", type=int, default=300, help="Policy RPC timeout in seconds")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.rpc_timeout <= 0:
        parser.error("Specify a valid port and positive RPC timeout")
    return main(args)


if __name__ == "__main__":
    raise SystemExit(cli())
