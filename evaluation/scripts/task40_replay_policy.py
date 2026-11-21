"""Private, single-episode MemoryVLA policy server for the Task40 replay probe."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from v1_groot_memoryvla.data.embodiment_tags import EmbodimentTag
from v1_groot_memoryvla.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
from v1_groot_memoryvla.policy.server_client import PolicyServer


BANK_FIELDS = ("bank", "last_written", "merge_count")
BANKS = ("perceptual", "cognitive")


def memory_copy(state):
    return {"slots": dict(state.slots), **{
        name: {"bank": {key: [(step, tensor.detach().cpu().clone()) for step, tensor in slots]
                        for key, slots in getattr(state.memory, name).bank.items()},
               "last_written": dict(getattr(state.memory, name).last_written),
               "merge_count": dict(getattr(state.memory, name).merge_count),
               "capacity": getattr(state.memory, name).capacity}
        for name in BANKS}}


def compare_memory(expected, actual):
    result = {"clock_equal": expected["slots"] == actual["slots"]}
    for name in BANKS:
        first, second = expected[name], actual[name]
        metadata = all(first[field] == second[field] for field in ("capacity", "last_written", "merge_count"))
        times = [{key: [time for time, _ in history] for key, history in bank["bank"].items()}
                 for bank in (first, second)]
        structure = times[0] == times[1]
        pairs = [(left, right) for key in first["bank"]
                 for (_, left), (_, right) in zip(first["bank"][key], second["bank"][key]) ] if structure else []
        exact = structure and all(left.dtype == right.dtype and torch.equal(left, right) for left, right in pairs)
        error = max((float((left.float() - right.float()).abs().max()) for left, right in pairs), default=0.0)
        result[name] = {"metadata_equal": metadata, "times_equal": structure, "tensor_equal": exact,
                        "max_abs_error": error, "slots_compared": len(pairs),
                        "timesteps": list(times[0].values()), "capacity": first["capacity"],
                        "merges": list(first["merge_count"].values())}
    result["equal"] = result["clock_equal"] and all(
        result[name]["metadata_equal"] and result[name]["times_equal"] and result[name]["tensor_equal"] for name in BANKS)
    return result


def rng_snapshot():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())


def restore_rng(values):
    random.setstate(values[0])
    np.random.set_state(values[1])
    torch.set_rng_state(values[2])
    torch.cuda.set_rng_state_all(values[3])


def serve(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy = Gr00tPolicy(EmbodimentTag.BEHAVIOR_R1_PRO, str(args.checkpoint), device="cuda")
    state = policy.memory_state
    if state is None:
        raise RuntimeError("This probe requires a MemoryVLA checkpoint")
    wrapped = Gr00tSimPolicyWrapper(policy)
    server = PolicyServer(wrapped, host="127.0.0.1", port=args.port)
    initial_rng = rng_snapshot()

    def clock():
        return [{"key": list(key), "next_decision": next_step,
                 **{name: {"last_written": getattr(state.memory, name).last_written[key],
                           "slots": len(getattr(state.memory, name).bank[key]),
                           "merges": getattr(state.memory, name).merge_count.get(key, 0)} for name in BANKS}}
                for key, next_step in state.slots.values()]

    def checkpoint():
        torch.save(memory_copy(state), args.output / "memory_online.pt")
        torch.save(rng_snapshot(), args.output / "policy_rng.pt")
        return clock()

    def rebuild(decisions, repeat_number=None, intervention=None):
        source = args.source_run or args.output
        expected = torch.load(source / "memory_online.pt", weights_only=True)
        state.reset()
        restore_rng(initial_rng)
        for index in range(decisions):
            entry = source / "prefix" / f"{index:04d}"
            metadata = json.loads((entry / "decision.json").read_text())
            if metadata["decision"] != index:
                raise RuntimeError(f"Unexpected replay index for {entry}")
            with np.load(entry / "observation.npz", allow_pickle=False) as archive:
                observation = {key: archive[key] for key in archive.files}
            observation.update(metadata["language"])
            wrapped.get_action(observation, options=metadata["session"])
        rebuilt = memory_copy(state)
        comparison = compare_memory(expected, rebuilt)
        if args.source_run:
            number = repeat_number if repeat_number is not None else next(rebuild_numbers)
            if number not in (1, 2, 3):
                raise ValueError(f"Invalid repeat number: {number}")
            comparison_file = args.output / f"memory_compare_{number}.json"
        else:
            torch.save(rebuilt, args.output / "memory_rebuilt.pt")
            comparison_file = args.output / "memory_compare.json"
        comparison_file.write_text(json.dumps(comparison, indent=2) + "\n")
        if not comparison["equal"]:
            raise RuntimeError(f"Rebuilt bank differs from online bank; see {comparison_file}")
        if intervention:
            key, next_decision = next(iter(state.slots.values()))
            assert next_decision == decisions and intervention in ("A_normal", "B_empty", "C_wrong")
            branches = args.output / intervention
            if intervention == "C_wrong":
                from v1_groot_memoryvla.audit.train_official_batch import load_real_step
                from v1_groot_memoryvla.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

                loader = LeRobotEpisodeLoader(args.donor_dataset, policy.modality_configs, video_backend="pyav")
                donor_episode = loader.episodes_metadata[0]["episode_index"]
                donor_session = {"session_id": "task40-donor", "env_ids": [0], "episode_ids": [str(donor_episode)]}
                frames = list(range(0, 8 * decisions, 8))
                for frame in frames:
                    step = load_real_step(loader, donor_episode, frame)
                    policy.get_action({"video": {name: np.asarray(values)[None] for name, values in step.images.items()},
                                       "state": {name: np.asarray(values)[None] for name, values in step.states.items()},
                                       "language": {policy.language_key: [[step.text]]}}, options=donor_session)
                donor_key, _ = state.slots.pop(("task40-donor", "0"))
            history = {}
            for name in BANKS:
                bank = getattr(state.memory, name)
                if intervention == "B_empty":
                    bank.bank[key].clear()
                elif intervention == "C_wrong":
                    original, donor = bank.bank[key], bank.bank[donor_key]
                    assert len(original) == len(donor) == decisions
                    assert all(left.shape == right.shape and left.dtype == right.dtype
                               for (_, left), (_, right) in zip(original, donor, strict=True))
                    assert any(not torch.equal(left, right) for (_, left), (_, right) in zip(original, donor))
                    bank.bank[key] = [(time, feature) for (time, _), (_, feature) in zip(original, donor)]
                    bank.reset([donor_key])
                history[name] = {"capacity": bank.capacity, "times": [time for time, _ in bank.bank[key]]}
            args.output.joinpath(f"history_{intervention}.json").write_text(json.dumps({
                "clock": clock(), "history": history,
                "donor": {"dataset": str(args.donor_dataset), "episode": int(donor_episode), "frames": frames}
                if intervention == "C_wrong" else None}, indent=2) + "\n")
            capture.update(branch=branches, first=decisions)
        restore_rng(torch.load(source / "policy_rng.pt", weights_only=False))
        print(f"TASK40_MEMORY_MATCH {decisions}", flush=True)
        return clock()

    def repeat_fixed_input(repeats=5, rng_mode="restore"):
        if type(repeats) is not int or not 2 <= repeats <= 10:
            raise ValueError("repeats must be an integer from 2 to 10")
        if rng_mode not in ("restore", "advance"):
            raise ValueError("rng_mode must be restore or advance")
        source = args.source_run or args.output
        saved = torch.load(source / "sim_checkpoint.pt", map_location="cpu", weights_only=False)
        anchor = saved["decision"]
        with np.load(source / "anchor_live_observation.npz", allow_pickle=False) as archive:
            observation = {key: archive[key] for key in archive.files}
        metadata = json.loads((source / "prefix" / f"{anchor - 1:04d}" / "decision.json").read_text())
        observation.update(metadata["language"])
        expected = torch.load(source / "memory_online.pt", map_location="cpu", weights_only=True)
        outputs = []
        rng_cursor = torch.load(source / "policy_rng.pt", map_location="cpu", weights_only=False)
        for trial in range(repeats):
            state.reset()
            restore_rng(initial_rng)
            for index in range(anchor):
                entry = source / "prefix" / f"{index:04d}"
                prefix_meta = json.loads((entry / "decision.json").read_text())
                with np.load(entry / "observation.npz", allow_pickle=False) as archive:
                    prefix_observation = {key: archive[key] for key in archive.files}
                prefix_observation.update(prefix_meta["language"])
                wrapped.get_action(prefix_observation, options=prefix_meta["session"])
            rebuilt = memory_copy(state)
            memory_result = compare_memory(expected, rebuilt)
            if not memory_result["equal"]:
                raise RuntimeError("Fixed-input trial did not rebuild the saved memory exactly")
            if rng_mode == "restore":
                restore_rng(torch.load(source / "policy_rng.pt", map_location="cpu", weights_only=False))
            else:
                restore_rng(rng_cursor)
            action, _ = wrapped.get_action(observation, options=metadata["session"])
            outputs.append({key: value.copy() for key, value in action.items()})
            if rng_mode == "advance":
                rng_cursor = rng_snapshot()
            np.savez_compressed(args.output / f"fixed_input_{rng_mode}_{trial + 1:02d}.npz", **action)
            print(f"TASK40_FIXED_INPUT_REPEAT mode={rng_mode} {trial + 1} memory_equal=true", flush=True)
        differences = {}
        reference = outputs[0]
        for key in reference:
            differences[key] = {
                "max_abs": max(float(np.max(np.abs(reference[key] - item[key]))) for item in outputs[1:]),
                "mean_abs": max(float(np.mean(np.abs(reference[key] - item[key]))) for item in outputs[1:]),
                "exact_matches": sum(bool(np.array_equal(reference[key], item[key])) for item in outputs[1:]),
            }
        digest = hashlib.sha256(source.joinpath("anchor_live_observation.npz").read_bytes()).hexdigest()
        result = {"repeats": repeats, "rng_mode": rng_mode, "anchor_decision": anchor, "input_sha256": digest,
                  "memory_equal_each_trial": True, "action_differences_vs_first": differences}
        (args.output / f"fixed_input_comparison_{rng_mode}.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    capture = {}
    if args.donor_dataset:
        action_head = policy.model.action_head
        predict = action_head.get_action

        def capture_action_input(backbone_output, action_input):
            if capture and state.memory.last_diagnostics:
                timestep = state.memory.last_diagnostics["observations"][0]["timestep"]
                directory = capture["branch"] / f"{timestep:04d}"
                features = backbone_output.backbone_features.detach().float().cpu().numpy()
                payload = {"sha256": hashlib.sha256(features.tobytes()).hexdigest(), "shape": list(features.shape),
                           "rms": float(np.sqrt(np.mean(features ** 2)))}
                directory.joinpath("action_expert_input.json").write_text(json.dumps(payload) + "\n")
                if timestep == capture["first"]:
                    np.savez_compressed(directory / "action_expert_input.npz", features=features,
                                        mask=backbone_output.backbone_attention_mask.cpu().numpy(),
                                        image_mask=backbone_output.image_mask.cpu().numpy())
            return predict(backbone_output, action_input)

        action_head.get_action = capture_action_input

    rebuild_numbers = iter(range(1, 4))
    for name, callback in (("task40_clock", clock), ("task40_checkpoint", checkpoint),
                           ("task40_rebuild", rebuild), ("task40_fixed_input_repeat", repeat_fixed_input)):
        server.register_endpoint(name, callback, requires_input=name in ("task40_rebuild", "task40_fixed_input_repeat"))
    print("TASK40_POLICY_READY", args.checkpoint, flush=True)
    server.run()
    server.socket.close()
    server.context.term()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--donor-dataset", type=Path)
    parser.add_argument("--port", type=int, default=5570)
    parser.add_argument("--seed", type=int, default=22004)
    serve(parser.parse_args())
