"""Workbench-owned local policy endpoint for the locked V1/V3 comparison."""
import argparse
from pathlib import Path

import torch

from v1_groot_memoryvla.data.embodiment_tags import EmbodimentTag
from v1_groot_memoryvla.experiment.first_training import seeded
from v1_groot_memoryvla.experiment.task21_diagnostic import load_model
from v1_groot_memoryvla.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
from v1_groot_memoryvla.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
from v1_groot_memoryvla.policy.memory_state import MemoryState
from v1_groot_memoryvla.policy.policy import BasePolicy
from v1_groot_memoryvla.policy.server_client import PolicyServer
from v3_groot_role_memory.model import Gr00tN1d6
from task40_memory_instrumentation import register_task40_instrumentation, _Trace
from unittest.mock import patch


class RoleTrace(_Trace):
    def install(self, stack):
        super().install(stack)
        branch = getattr(self.head, 'role_velocity', None)
        if branch is None:
            return
        original = branch.forward
        def forward(velocity, hidden, memory, state, valid, mask=None):
            result = original(velocity, hidden, memory, state, valid, mask)
            self.arrays['v3_role_memory'] = memory.detach().float().cpu().numpy()
            self.arrays['v3_role_valid'] = valid.detach().float().cpu().numpy()
            self.arrays['v3_original_velocity'] = velocity.detach().float().cpu().numpy()
            self.arrays['v3_role_delta'] = branch.last['delta'].detach().float().cpu().numpy()
            for role, gate in branch.last['gates'].items():
                self.arrays['v3_gate_'+role] = gate.detach().float().cpu().numpy()
            return result
        stack.enter_context(patch.object(branch, 'forward', forward))



class ComparisonPolicy(Gr00tPolicy):
    def __init__(self, checkpoint):
        BasePolicy.__init__(self, strict=True)
        import json
        config = json.loads((checkpoint / 'config.json').read_text())
        if config['model_type'] == 'Gr00tN1d6V3':
            model = Gr00tN1d6.from_pretrained(checkpoint, torch_dtype=torch.float32,
                transformers_loading_kwargs={'trust_remote_code': True, 'local_files_only': True})
            model.backbone.bfloat16()
            model.cuda().eval()
        else:
            model = load_model(checkpoint, True, 42)
            model.backbone.cuda()
        self.model = model
        self.memory_state = MemoryState(model.memory)
        self.history_mode = 'normal'
        self.processor = Gr00tN1d6Processor.from_pretrained(checkpoint,
            transformers_loading_kwargs={'trust_remote_code': True, 'local_files_only': True})
        self.processor.eval()
        self.embodiment_tag = EmbodimentTag('behavior_r1_pro')
        self.modality_configs = self.processor.get_modality_configs()[self.embodiment_tag.value]
        self.collate_fn = self.processor.collator
        self.language_key, = self.modality_configs['language'].modality_keys

    def _get_action(self, observation, options=None):
        # Match the training/reload arithmetic; do not round learned FP32 memory
        # weights by using the legacy policy's blanket model.bfloat16().
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return super()._get_action(observation, options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    import task40_memory_instrumentation
    task40_memory_instrumentation._Trace = RoleTrace
    policy = ComparisonPolicy(args.checkpoint)
    wrapped = Gr00tSimPolicyWrapper(policy)
    server = PolicyServer(wrapped, host='127.0.0.1', port=args.port)
    register_task40_instrumentation(server, policy, wrapped)
    seeded(42)
    print('V3_COMPARISON_POLICY_READY', args.checkpoint, args.port, flush=True)
    server.run()


if __name__ == '__main__':
    main()
