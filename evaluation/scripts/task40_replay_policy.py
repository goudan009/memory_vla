"""Private, single-episode MemoryVLA policy server for the Task40 replay probe."""
import argparse
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

    def rebuild(decisions, repeat_number=None):
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
        restore_rng(torch.load(source / "policy_rng.pt", weights_only=False))
        print(f"TASK40_MEMORY_MATCH {decisions}", flush=True)
        return clock()

    rebuild_numbers = iter(range(1, 4))
    for name, callback in (("task40_clock", clock), ("task40_checkpoint", checkpoint), ("task40_rebuild", rebuild)):
        server.register_endpoint(name, callback, requires_input=name == "task40_rebuild")
    print("TASK40_POLICY_READY", args.checkpoint, flush=True)
    server.run()
    server.socket.close()
    server.context.term()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--port", type=int, default=5570)
    parser.add_argument("--seed", type=int, default=22004)
    serve(parser.parse_args())
