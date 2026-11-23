"""Opt-in instrumentation for the existing Task40 MemoryVLA PolicyServer.

Call register_task40_instrumentation(server, policy, wrapped) after the legacy
endpoint registrations. The server's serial dispatch owns one policy instance.
The RPCs use the existing task observation and session fields. All binary
artifacts use torch serialization and preserve the original bank dtypes.
"""

import hashlib
import io
import math
import random
import runpy
import sys
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import torch


BANKS = ("perceptual", "cognitive")
SCHEMA_VERSION = 1


def _pack(value):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _unpack(payload):
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)


def capture_rng():
    """Use the same four RNG sources and tuple format as the legacy service."""
    return random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()


def pack_rng(values):
    python, numpy, cpu, cuda = values
    generator, keys, *tail = numpy
    return _pack((python, (generator, torch.tensor(keys.astype(np.int64)), *tail), cpu, cuda))


def _restore_rng(values):
    python, numpy, cpu, cuda = values
    random.setstate(python)
    np.random.set_state(numpy)
    torch.set_rng_state(cpu)
    torch.cuda.set_rng_state_all(cuda)


def _unpack_rng(payload):
    python, (generator, keys, *tail), cpu, cuda = _unpack(payload)
    if len(cuda) != torch.cuda.device_count():
        raise ValueError("RNG artifact must match the number of visible policy GPUs")
    return python, (generator, keys.numpy().astype(np.uint32), *tail), cpu, cuda


def _key(session):
    envs = session.get("env_ids", [0])
    episodes = session.get("episode_ids", ["episode0"])
    if len(envs) != 1 or len(episodes) != 1:
        raise ValueError("Diagnostic calls take one environment and one episode")
    return str(session.get("session_id", "standalone")), str(envs[0]), str(episodes[0])


def _view(state, key):
    """Take immutable references to this stream's slots for an adjacent diff.

    MemoryBank._write appends or replaces tensors, including cosine merges.
    Copy the lists here so the upcoming append cannot mutate the before view.
    """
    slot = key[:2]
    cursor = state.slots.get(slot)
    result = {"slots": {slot: cursor} if cursor is not None and cursor[0] == key else {}}
    for name in BANKS:
        bank = getattr(state.memory, name)
        result[name] = {
            "capacity": bank.capacity,
            "bank": {key: list(bank.bank[key])} if key in bank.bank else {},
            "last_written": {key: bank.last_written[key]} if key in bank.last_written else {},
            "merge_count": {key: bank.merge_count[key]} if key in bank.merge_count else {},
        }
    return result


def _cpu_snapshot(view):
    return {
        **view,
        **{
            name: {
                **view[name],
                "bank": {
                    key: [(time, tensor.detach().to(device="cpu", copy=True)) for time, tensor in rows]
                    for key, rows in view[name]["bank"].items()
                },
            }
            for name in BANKS
        },
    }


def _clock(snapshot):
    """Metadata needed to check delta order, including absent versus empty banks."""
    return {
        "slots": snapshot["slots"],
        **{
            name: {
                **snapshot[name],
                "bank": {key: [time for time, _ in rows] for key, rows in snapshot[name]["bank"].items()},
            }
            for name in BANKS
        },
    }


def _make_delta(before, after, key):
    updates = {}
    for name in BANKS:
        old_rows = dict(before[name]["bank"].get(key, ()))
        new_rows = after[name]["bank"].get(key, ())
        updates[name] = {
            time: tensor.detach().to(device="cpu", copy=True)
            for time, tensor in new_rows
            if old_rows.get(time) is not tensor
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "key": key,
        "before": _clock(before),
        "after": _clock(after),
        "updates": updates,
    }


def apply_memory_delta(snapshot, payload):
    """Apply the serialized delta to a CPU snapshot and return its successor."""
    delta = _unpack(payload)
    if delta["schema_version"] != SCHEMA_VERSION or _clock(snapshot) != delta["before"]:
        raise ValueError("Memory delta must follow the matching snapshot or preceding delta")
    key = delta["key"]
    result = {"slots": delta["after"]["slots"]}
    for name in BANKS:
        metadata = delta["after"][name]
        features = dict(snapshot[name]["bank"].get(key, ())) | delta["updates"][name]
        result[name] = {
            **metadata,
            "bank": {k: [(time, features[time]) for time in times] for k, times in metadata["bank"].items()},
        }
    return result


def pack_memory(snapshot):
    """Serialize in the existing memory.pt format with original tensor dtypes."""
    return _pack(_cpu_snapshot(snapshot))


def unpack_memory(payload):
    """Load an existing memory.pt artifact or a snapshot from the diagnostic RPC."""
    return _unpack(payload)


def _condition_memory(snapshot, key, empty_banks, donor, donor_banks):
    if len(snapshot["slots"]) > 1 or snapshot["slots"].keys() - {key[:2]}:
        raise ValueError("Target snapshot must describe this single policy session")
    if snapshot["slots"] and snapshot["slots"][key[:2]][0] != key:
        raise ValueError("Session and episode must match the target snapshot")
    if (empty_banks | donor_banks) - set(BANKS) or empty_banks & donor_banks:
        raise ValueError("Choose distinct valid banks for empty and donor conditions")
    if donor_banks:
        (donor_key, donor_next), = donor["slots"].values()
        if donor_key == key or snapshot["slots"][key[:2]][1] != donor_next:
            raise ValueError("Donor must be a different episode at the same decision clock")
    result = {"slots": snapshot["slots"]}
    for name in BANKS:
        bank = snapshot[name]
        for field in ("bank", "last_written", "merge_count"):
            if bank[field].keys() - {key}:
                raise ValueError("Target snapshot contains another policy session")
        result[name] = bank
        if name in empty_banks:
            result[name] = {**bank, "bank": {key: []}}
        elif name in donor_banks:
            original = bank["bank"].get(key, ())
            alternative = donor[name]["bank"].get(donor_key, ())
            if not original or len(original) != len(alternative) or bank["capacity"] != donor[name]["capacity"]:
                raise ValueError(f"{name} donor must match the target's nonempty occupancy and capacity")
            if any(a.shape != b.shape or a.dtype != b.dtype for (_, a), (_, b) in zip(original, alternative)):
                raise ValueError(f"{name} donor tensor shapes and dtypes must match")
            result[name] = {**bank, "bank": {key: [(t, value) for (t, _), (_, value) in zip(original, alternative)]}}
    return result


def _array(tensor):
    if tensor.is_floating_point():
        tensor = tensor.float()
    return tensor.detach().cpu().numpy().copy()


def _pooled(tensor):
    return _array(tensor.float().reshape(-1, tensor.shape[-1]).mean(0))


class _Trace:
    """Scoped hooks on the deployed residual/value/EOS architecture."""

    def __init__(self, policy, scale, location, detail):
        self.adapter = policy.model.memory
        self.head = policy.model.action_head
        self.scale = scale
        self.location = location
        self.full = detail == "full"
        self.arrays = {}
        self.gated = None
        self.applied = None
        self.eos = None
        self.memory_calls = self.dit_calls = 0

    def _before_memory(self, _module, args):
        backbone, input_ids, context = args
        if self.memory_calls or len(context["episode_keys"]) != 1 or context.get("history_mode") != "normal":
            raise RuntimeError("Expected one normal MemoryVLA stream with one inference per request")
        self.memory_calls += 1
        attention = backbone["backbone_attention_mask"].bool()
        image = backbone["image_mask"].bool()
        indices = torch.where((input_ids[0] == self.adapter.eos_token_id) & attention[0])[0]
        if not indices.numel() or image[0, indices[-1]]:
            raise RuntimeError("Cannot identify the user EOS")
        index = int(indices[-1])
        self.eos_index, self.eos = index, slice(index, index + 1)
        features = backbone["backbone_features"]
        self.attention = attention
        self.base_eos = features[:, self.eos, :]
        self.arrays["input_ids"] = _array(input_ids)
        self.arrays["pre_backbone_attention_mask"] = _array(attention)
        self.arrays["pre_image_mask"] = _array(image)
        self.arrays["pre_backbone_features_eos"] = _array(self.base_eos)
        self.arrays["current_pooled"] = _pooled(features[attention])
        if self.full:
            self.arrays["pre_backbone_features"] = _array(features)

    def _readout(self, _module, _args, output):
        readout, attended_value = output
        self.arrays["cognitive_readout"] = _array(readout)
        self.arrays["cognitive_readout_pooled"] = _pooled(readout)
        self.arrays["cognitive_attended_value"] = _array(attended_value)

    def _gate(self, _module, _args, output):
        fused, self.gated = output
        self.arrays["cognitive_gated"] = _array(fused)
        self.arrays["cognitive_value_after_gate"] = _array(self.gated)

    def _after_memory(self, _module, _args, output):
        features = output["backbone_features"]
        native_eos = features[:, self.eos, :]
        parameter = self.adapter.cognitive_residual_scale
        native = parameter * self.gated[0, 0] if self.gated is not None else torch.zeros_like(self.base_eos[0, 0])
        expected_eos = self.base_eos + native.to(self.base_eos.dtype)
        fusion_mode = getattr(self.head, "memory_fusion_mode", "legacy")
        if fusion_mode == "route_a":
            if self.scale != 1 or self.location != "eos":
                raise ValueError("Use native scale and location to diagnose a learned fusion checkpoint")
            self.arrays["learned_eos_residual"] = _array(native_eos.float() - expected_eos.float())
        elif not torch.equal(native_eos, expected_eos):
            raise RuntimeError("Recorded cognitive residual does not reproduce the adapter's EOS")
        self.arrays["cognitive_native_scale"] = _array(parameter)
        self.arrays["cognitive_value_native_scaled"] = _array(native[None, None])
        self.arrays["expert_input_native_eos"] = _array(native_eos)
        self.applied = (native if self.scale == 1 or self.gated is None
                        else (parameter * self.scale) * self.gated[0, 0])[None, None]
        self.arrays["cognitive_value_scaled"] = _array(self.applied)
        if self.gated is not None and (self.scale != 1 or self.location == "dit_condition"):
            features = features.clone()
            features[:, self.eos, :] = (
                self.base_eos if self.location == "dit_condition"
                else self.base_eos + self.applied.to(self.base_eos.dtype)
            )
            output["backbone_features"] = features
        self.arrays["expert_input_eos"] = _array(features[:, self.eos, :])
        self.arrays["memory_residual_eos"] = self.arrays["expert_input_eos"] - self.arrays["pre_backbone_features_eos"]
        if self.full:
            self.arrays["expert_input"] = _array(features)
            self.arrays["memory_residual"] = self.arrays["expert_input"] - self.arrays["pre_backbone_features"]

    def install(self, stack):
        adapter = self.adapter
        if (adapter.training or self.head.training or
                (adapter.conditioning, adapter.residual_readout, adapter.cognitive_residual_scope)
                != ("residual", "value", "eos") or not hasattr(adapter, "cognitive_residual_scale")):
            raise ValueError("Instrumentation requires the frozen Task40 residual/value/EOS checkpoint")
        stack.callback(adapter.register_forward_pre_hook(self._before_memory).remove)
        stack.callback(adapter.cognitive.retrieval_blocks[-1].register_forward_hook(self._readout).remove)
        stack.callback(adapter.cognitive.gate_fusion_blocks.register_forward_hook(self._gate).remove)
        stack.callback(adapter.register_forward_hook(self._after_memory).remove)
        original_action = self.head.get_action
        original_dit = self.head.get_action_with_features

        def get_action(backbone_output, action_input):
            self.arrays["normalized_state"] = _array(action_input.state)
            output = original_action(backbone_output, action_input)
            self.arrays["normalized_actions"] = _array(output["action_pred"])
            if "fusion_action_features" in output:
                tokens = output["fusion_action_features"]
                self.arrays["dit_action_tokens_pooled"] = _pooled(tokens)
                if self.full:
                    self.arrays["dit_action_tokens"] = _array(tokens)
            return output

        def get_action_with_features(backbone_features, **kwargs):
            self.dit_calls += 1
            self.arrays["dit_state_features"] = _array(kwargs["state_features"])
            self.arrays["embodiment_id"] = _array(kwargs["embodiment_id"])
            before = backbone_features[:, self.eos, :]
            self.arrays["dit_condition_pre_intervention_eos"] = _array(before)
            if self.location == "dit_condition" and self.gated is not None:
                backbone_features = backbone_features.clone()
                backbone_features[:, self.eos, :] = before + self.applied.to(before.dtype)
            self.arrays["dit_condition_eos"] = _array(backbone_features[:, self.eos, :])
            self.arrays["dit_condition_pooled"] = _pooled(backbone_features[self.attention])
            if self.full:
                self.arrays["dit_condition"] = _array(backbone_features)
            return original_dit(backbone_features=backbone_features, **kwargs)

        stack.enter_context(patch.object(self.head, "get_action", get_action))
        stack.enter_context(patch.object(self.head, "get_action_with_features", get_action_with_features))


class _OpenVLATrace(_Trace):
    """Observe the independent cognitive and perceptual DiT inputs."""

    def _readout(self, _module, _args, output):
        self.arrays["cognitive_readout"] = _array(output)
        self.arrays["cognitive_readout_pooled"] = _pooled(output)

    def _gate(self, _module, _args, output):
        self.gated = output
        self.arrays["cognitive_gated"] = _array(output)

    def _after_memory(self, _module, _args, output):
        self.arrays["expert_input_eos"] = _array(output["backbone_features"][:, self.eos, :])
        self.arrays["cognitive_fused"] = _array(output["memory_cognitive_token"])
        if self.full:
            self.arrays["perceptual_fused"] = _array(output["memory_perceptual_tokens"])

    def install(self, stack):
        adapter, head = self.adapter, self.head
        if adapter.training or head.training or self.scale != 1 or self.location != "eos":
            raise ValueError("OpenVLA diagnostics require native inference without residual interventions")
        stack.callback(adapter.register_forward_pre_hook(self._before_memory).remove)
        stack.callback(adapter.cognitive.retrieval_blocks[-1].register_forward_hook(self._readout).remove)
        stack.callback(adapter.cognitive.gate_fusion_blocks.register_forward_hook(self._gate).remove)
        stack.callback(adapter.register_forward_hook(self._after_memory).remove)

        def first_condition(_module, _args, kwargs):
            if "dit_condition_eos" in self.arrays:
                return
            condition = kwargs["cognitive_condition"]
            timed = condition + head.model.timestep_encoder(kwargs["timestep"])[:, None]
            # Keep the shared comparison schema while recording the native
            # condition against which the cognitive probe is measured.
            self.arrays["dit_condition_eos"] = _array(timed)
            self.arrays["dit_condition_pooled"] = _pooled(timed)
            self.arrays["dit_cognitive_projection"] = _array(condition)
            perceptual = kwargs["encoder_hidden_states"]
            self.arrays["dit_perception_pooled"] = _pooled(perceptual)
            if self.full:
                self.arrays["dit_perception"] = _array(perceptual)

        stack.callback(head.model.register_forward_pre_hook(first_condition, with_kwargs=True).remove)
        original_action = head.get_action
        original_dit = head.get_action_with_features

        def get_action(backbone_output, action_input):
            self.arrays["normalized_state"] = _array(action_input.state)
            output = original_action(backbone_output, action_input)
            self.arrays["normalized_actions"] = _array(output["action_pred"])
            return output

        def get_action_with_features(backbone_features, **kwargs):
            self.dit_calls += 1
            self.arrays["dit_state_features"] = _array(kwargs["state_features"])
            self.arrays["embodiment_id"] = _array(kwargs["embodiment_id"])
            return original_dit(backbone_features=backbone_features, **kwargs)

        stack.enter_context(patch.object(head, "get_action", get_action))
        stack.enter_context(patch.object(head, "get_action_with_features", get_action_with_features))


@contextmanager
def _frozen_state(state, snapshot, rng, device):
    """Use temporary dictionaries so frozen requests never reset a live stream."""
    banks = [getattr(state.memory, name) for name in BANKS]
    slots = state.slots
    original = [(bank.bank, bank.last_written, bank.merge_count) for bank in banks]
    rng_before = capture_rng()
    observations = [
        (module, attr, vars(module).get(attr), attr in vars(module))
        for module in state.memory.modules()
        for attr in ("last_diagnostics", "last_history_control", "last_attention", "last_gate_mean")
    ]
    try:
        state.slots = dict(snapshot["slots"])
        for name, bank in zip(BANKS, banks):
            saved = snapshot[name]
            if saved["capacity"] != bank.capacity:
                raise ValueError(f"{name} capacity differs from the loaded checkpoint")
            bank.bank = {key: [(t, value.to(device=device, copy=True)) for t, value in rows]
                         for key, rows in saved["bank"].items()}
            bank.last_written = dict(saved["last_written"])
            bank.merge_count = dict(saved["merge_count"])
        _restore_rng(rng)
        yield
    finally:
        state.slots = slots
        for bank, (rows, last, merges) in zip(banks, original):
            bank.bank, bank.last_written, bank.merge_count = rows, last, merges
        for module, attr, value, existed in observations:
            if existed:
                setattr(module, attr, value)
            elif attr in vars(module):
                delattr(module, attr)
        _restore_rng(rng_before)


class Task40Instrumentation:
    def __init__(self, policy, wrapped):
        self.policy = policy
        self.wrapped = wrapped
        self.state = policy.memory_state
        if self.state is None or self.state.memory is not policy.model.memory:
            raise ValueError("Policy must expose its live MemoryVLA adapter")

    def snapshot(self, session):
        view = _view(self.state, _key(session))
        return {"schema_version": SCHEMA_VERSION, "memory": pack_memory(view), "rng": pack_rng(capture_rng())}

    def restore_memory(self, session, memory, rng, decision):
        """Install a single archived stream and its policy RNG on a dedicated worker."""
        key = _key(session)
        if type(decision) is not int or decision < 0:
            raise ValueError("Restore decision must be a nonnegative integer")
        archived = unpack_memory(memory)
        cursor = archived["slots"].get(key[:2])
        if len(archived["slots"]) != 1 or cursor != (key, decision) or type(cursor[1]) is not int:
            raise ValueError("Archive must contain exactly the requested session and decision")
        saved = _condition_memory(archived, key, set(), None, set())
        target_rng = _unpack_rng(rng)

        state = self.state
        original_slots = state.slots
        if any(slot != key[:2] or cursor[0] != key for slot, cursor in original_slots.items()):
            raise ValueError("Restore requires a dedicated worker without another live session")
        original_banks, restored_banks = [], []
        clock = {"key": list(key), "decision": decision}
        for name in BANKS:
            bank = getattr(state.memory, name)
            source = saved[name]
            if any(mapping.keys() - {key} for mapping in (bank.bank, bank.last_written, bank.merge_count)):
                raise ValueError("Restore requires a dedicated worker without another live session")
            rows = source["bank"].get(key, [])
            if source["capacity"] != bank.capacity or len(rows) > bank.capacity:
                raise ValueError(f"{name} capacity differs from the loaded checkpoint or is exceeded")
            times = [time for time, _ in rows]
            last = source["last_written"].get(key, -1)
            merges = source["merge_count"].get(key, 0)
            if (type(last) is not int or not -1 <= last < decision or type(merges) is not int or merges < 0
                    or any(type(time) is not int or not 0 <= time <= last for time in times)
                    or times != sorted(set(times))):
                raise ValueError(f"{name} contains invalid timestamps or merge metadata")
            clock[name] = {"times": times, "last_written": last, "merges": merges, "capacity": bank.capacity}
            original_banks.append((bank, bank.bank, bank.last_written, bank.merge_count))
            restored_banks.append((
                {episode: [(time, value.to(device=self.policy.model.device, copy=True)) for time, value in history]
                 for episode, history in source["bank"].items()},
                dict(source["last_written"]), dict(source["merge_count"]),
            ))

        original_rng = capture_rng()
        try:
            state.slots = dict(saved["slots"])
            for (bank, *_), restored in zip(original_banks, restored_banks):
                bank.bank, bank.last_written, bank.merge_count = restored
            _restore_rng(target_rng)
            # Digest the installed snapshot format rather than an archive's original ZIP container.
            installed = self.snapshot(session)
            return {"schema_version": SCHEMA_VERSION, "decision": decision, "clock": clock,
                    "memory_sha256": hashlib.sha256(installed["memory"]).hexdigest(),
                    "rng_sha256": hashlib.sha256(installed["rng"]).hexdigest()}
        except Exception:
            state.slots = original_slots
            for bank, rows, last_written, merge_count in original_banks:
                bank.bank, bank.last_written, bank.merge_count = rows, last_written, merge_count
            _restore_rng(original_rng)
            raise

    @torch.inference_mode()
    def _forward(self, observation, session, before, scale, location, detail):
        if detail not in ("compact", "full"):
            raise ValueError("detail must be compact or full")
        openvla = getattr(self.policy.model.action_head, "openvla_memory", False)
        trace_type = _OpenVLATrace if openvla else _Trace
        trace = trace_type(self.policy, scale, location, detail)
        with ExitStack() as stack:
            trace.install(stack)
            actions, info = self.wrapped.get_action(
                observation, options={**session, "memory_diagnostics": True, "memory_action_controls": False}
            )
        if trace.memory_calls != 1 or trace.dit_calls != 1 or "normalized_actions" not in trace.arrays:
            raise RuntimeError("Policy did not execute the expected adapter and action-head path")
        key = _key(session)
        history = before["cognitive"]["bank"].get(key, ())
        current = trace.arrays["current_pooled"]
        trace.arrays["cognitive_bank_pooled"] = (
            _pooled(torch.stack([feature for _, feature in history])) if history else np.zeros_like(current)
        )
        trace.arrays.setdefault("cognitive_readout_pooled", np.zeros_like(current))
        retrieval = info["memory_diagnostics"]
        record, = retrieval["observations"]
        for name in BANKS:
            expected_times = [t for t, _ in before[name]["bank"].get(key, ())]
            if list(record[name]["history_timesteps"]) != expected_times:
                raise RuntimeError(f"{name} retrieval does not match the recorded bank")
        return {
            "schema_version": SCHEMA_VERSION,
            "actions": actions,
            "model_features": trace.arrays,
            "retrieval": retrieval,
            "trace": {"user_eos_index": trace.eos_index, "cognitive_read": trace.gated is not None,
                      "fusion_mode": "openvla" if openvla else getattr(trace.head, "memory_fusion_mode", "legacy"),
                      "fusion_block": getattr(trace.head, "memory_fusion_block", None),
                      "scale_multiplier": scale, "location": location, "detail": detail},
        }

    def infer(self, observation, session, decision, include_base=False, detail="full"):
        key = _key(session)
        before = _view(self.state, key)
        expected = before["slots"].get(key[:2], (key, 0))[1]
        if decision != expected:
            raise ValueError(f"Expected next decision {expected}, got {decision}")
        rng = capture_rng()
        old_key = self.state.slots.get(key[:2], (key, 0))[0]
        previous = before if key == old_key else _view(self.state, old_key)
        try:
            response = self._forward(observation, session, before, 1.0, "eos", detail)
            after = _view(self.state, key)
            if after["slots"] != {key[:2]: (key, decision + 1)}:
                raise RuntimeError("Policy clock did not advance exactly once")
            response.update(decision=decision, rng_before=pack_rng(rng),
                            memory_delta=_pack(_make_delta(before, after, key)))
            if include_base or decision == 0:
                response["memory_base"] = pack_memory(before)
        except Exception:
            self.state.slots.pop(key[:2], None)
            self.state.slots.update(previous["slots"])
            for name in BANKS:
                bank = getattr(self.state.memory, name)
                bank.reset([key, old_key])
                for field in ("bank", "last_written", "merge_count"):
                    getattr(bank, field).update(previous[name][field])
            _restore_rng(rng)
            raise
        return response

    def frozen_forward(self, observation, session, memory, rng, scale=1.0, location="eos",
                       detail="full", empty_banks=(), donor_memory=None, donor_banks=()):
        scale = float(scale)
        if not math.isfinite(scale) or scale < 0 or location not in ("eos", "dit_condition"):
            raise ValueError("Specify a finite nonnegative scale and location eos or dit_condition")
        if donor_banks and donor_memory is None:
            raise ValueError("Donor intervention requires donor_memory")
        key = _key(session)
        target = _condition_memory(
            unpack_memory(memory), key, set(empty_banks),
            unpack_memory(donor_memory) if donor_banks else None, set(donor_banks),
        )
        with _frozen_state(self.state, target, _unpack_rng(rng), self.policy.model.device):
            before = _view(self.state, key)
            response = self._forward(deepcopy(observation), session, before, scale, location, detail)
            response["rng_after"] = pack_rng(capture_rng())
            response["decision"] = before["slots"].get(key[:2], (key, 0))[1]
            response["trace"].update(
                empty_banks=list(empty_banks), donor_banks=list(donor_banks),
                input_memory_sha256=hashlib.sha256(memory).hexdigest(),
                input_rng_sha256=hashlib.sha256(rng).hexdigest(),
            )
            if donor_banks:
                response["trace"]["donor_memory_sha256"] = hashlib.sha256(donor_memory).hexdigest()
        return response


def register_task40_instrumentation(server, policy, wrapped):
    extension = Task40Instrumentation(policy, wrapped)
    server.register_endpoint("task40_memory_snapshot", extension.snapshot)
    server.register_endpoint("task40_restore_memory", extension.restore_memory)
    server.register_endpoint("task40_trace_infer", extension.infer)
    server.register_endpoint("task40_frozen_forward", extension.frozen_forward)
    return extension


def register_restore_rng(server):
    def restore_rng(rng):
        _restore_rng(_unpack_rng(rng))
        return {"rng_sha256": hashlib.sha256(rng).hexdigest()}

    server.register_endpoint("task40_restore_rng", restore_rng)


def run_stateless_server():
    """Expose archived policy RNG on the official no-memory service."""
    from v1_groot_memoryvla.policy import server_client

    class RestorePolicyServer(server_client.PolicyServer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            register_restore_rng(self)

    with patch.object(server_client, "PolicyServer", RestorePolicyServer):
        runpy.run_module("v1_groot_memoryvla.serve", run_name="__main__")


def run_legacy_server():
    """Run the archived entry point with the same CLI and A100 runtime check."""
    from v1_groot_memoryvla.policy import server_client

    class InstrumentedPolicyServer(server_client.PolicyServer):
        def __init__(self, wrapped, *args, **kwargs):
            super().__init__(wrapped, *args, **kwargs)
            register_task40_instrumentation(self, wrapped.policy, wrapped)

    archive = sys.argv.pop(1)
    if sys.argv.pop(1) != "--task40-trace":
        raise SystemExit("Missing Task40 trace process marker")
    with patch.object(server_client, "PolicyServer", InstrumentedPolicyServer):
        runpy.run_path(archive, run_name="__main__")


def run_native_memory_server():
    """Install Task40 endpoints on the current MemoryVLA policy server."""
    from v1_groot_memoryvla.policy import server_client

    class InstrumentedPolicyServer(server_client.PolicyServer):
        def __init__(self, policy, *args, **kwargs):
            super().__init__(policy, *args, **kwargs)
            register_task40_instrumentation(self, policy.policy, policy)

    with patch.object(server_client, "PolicyServer", InstrumentedPolicyServer):
        runpy.run_module("v1_groot_memoryvla.serve", run_name="__main__")
