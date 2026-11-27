"""Opt-in observation/action boundary recorder for a concrete control fault.

No smoothing, action conversion, optimizer update or model parameter changes.
The extra RPC isolates Torch RNG and restores integration steps after each call.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch


class ActionDiagnostics:
    def __init__(self, policy, output):
        if policy.memory_state is not None:
            raise ValueError('Action fault diagnostic requires a no-memory policy')
        self.policy = policy
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.index = len(list(self.output.glob('request_*.npz')))
        self.last_raw = None

    def record(self, observation, normalized, decoded, options):
        self.last_raw = normalized.detach().float().cpu().numpy()
        arrays = {'normalized_action': self.last_raw}
        language = observation.get('language', {})
        for modality in ('video', 'state'):
            for name, value in observation[modality].items():
                arrays[modality + '.' + name] = np.asarray(value)
        for name, value in decoded.items():
            arrays['decoded.' + name] = value
        name = f'request_{self.index:06d}.npz'
        np.savez_compressed(self.output / name, **arrays)
        record = dict(index=self.index, file=name, language=language, options=options,
                      time=time.time(), inference_steps=self.policy.model.action_head.num_inference_timesteps,
                      normalized_sha256=hashlib.sha256(self.last_raw.tobytes()).hexdigest())
        with (self.output / 'requests.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        self.index += 1

    def diagnose(self, observation, seed=42, steps=(4, 8, 16)):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('invalid diagnostic seed')
        if not isinstance(steps, (list, tuple)) or not steps or any(type(n) is not int or n not in (4, 8, 16) for n in steps):
            raise ValueError('steps must contain 4/8/16')
        head = self.policy.model.action_head
        original_steps = head.num_inference_timesteps
        device = next(self.policy.model.parameters()).device
        devices = [device.index] if device.type == 'cuda' else []
        results = {}
        try:
            with torch.random.fork_rng(devices=devices):
                for n in steps:
                    torch.manual_seed(seed)
                    if devices:
                        torch.cuda.manual_seed(seed)
                    head.num_inference_timesteps = n
                    action, _ = self.policy.get_action(observation, options={'fault_diagnostic': True, 'seed': seed})
                    results[str(n)] = dict(action=action, normalized_action=self.last_raw.copy())
        finally:
            head.num_inference_timesteps = original_steps
        return results


def install(server, policy, output):
    recorder = ActionDiagnostics(policy, output)
    policy.action_diagnostic_hook = recorder.record
    server.register_endpoint('diagnose_actions', recorder.diagnose)
    return recorder
