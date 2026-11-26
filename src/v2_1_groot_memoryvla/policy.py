"""V2.1 policy using the existing V1 RPC and processor contract."""
from pathlib import Path
from typing import Any
import numpy as np
import torch
from v1_groot_memoryvla.policy.gr00t_policy import Gr00tPolicy,_rec_to_dtype
from v1_groot_memoryvla.policy.memory_state import MemoryState
from v1_groot_memoryvla.data.types import MessageType
from .model.integration import Integrated
from .config import check_scale
from .data.windows import raw_from_states

class Policy(Gr00tPolicy):
    def __init__(self,embodiment_tag,model_path,*,device,strict=True,history_mode='normal',residual_scale=1.):
        if not (Path(model_path)/'v2_1.json').is_file():
            raise ValueError('Expected V2.1 sidecar; refusing a baseline fallback')
        if embodiment_tag.value!='behavior_r1_pro':
            raise ValueError('V2.1 currently supports the audited R1Pro mapping')
        super().__init__(embodiment_tag,model_path,device=device,strict=strict,history_mode='normal')
        self.model=Integrated.load_sidecar(self.model,model_path,device)
        self.device_type=next(self.model.base.parameters()).device.type
        self.model.residual_scale=check_scale(residual_scale)
        self.memory_state=MemoryState(self.model.memory)
        if history_mode not in ('normal','empty','wrong','other_episode'):
            raise ValueError('invalid history mode')
        self.history_mode=history_mode

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Internal method to compute actions from observations.

        Pipeline:
        1. Unbatch observations into individual samples
        2. Convert each to VLAStepData and process
        3. Collate into model input batch
        4. Run model inference
        5. Decode and unnormalize actions

        Args:
            observation: Batched observation dictionary
            options: session_id, env_ids and episode_ids from the shared evaluator

        Returns:
            Tuple of (actions_dict, info_dict)
        """
        # Step 1: Split batched observation into individual observations
        unbatched_observations = self._unbatch_observation(observation)
        processed_inputs = []

        # Step 2: Process each observation through the VLA processor
        states = []
        for obs in unbatched_observations:
            vla_step_data = self._to_vla_step_data(obs)
            states.append(vla_step_data.states)  # dict[str, np.ndarray[np.float32, (T, D)]]
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed_inputs.append(self.processor(messages))

        # Step 3: Collate processed inputs into a single batch for model
        collated_inputs = self.collate_fn(processed_inputs)
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)
        if self.memory_state is not None:
            context = self.memory_state.context(len(unbatched_observations), options)
            context['history_mode'] = getattr(self, 'history_mode', 'normal')
            context['memory_diagnostics'] = (options or {}).get('memory_diagnostics') is True
            if (options or {}).get('memory_action_controls'):
                raise ValueError('Use V2.1 offline diagnostic, not the V1 controls')
            if context['history_mode'] in ('wrong','other_episode'):
                context['donor_keys'] = (options or {}).get('donor_keys')
            collated_inputs["inputs"]["memory_context"] = context

        # Step 4: Run model inference to predict actions
        raw_q = torch.from_numpy(np.stack([raw_from_states(s) for s in states]))
        model_inputs = dict(collated_inputs['inputs'])
        model_inputs.pop('memory_context', None)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.device_type=='cuda'):
            model_pred = self.model.get_action(model_inputs, raw_q, context)
        if self.memory_state is not None:
            self.memory_state.advance(context)
        normalized_action = model_pred["action_pred"].float()

        # Step 5: Decode actions from normalized space back to physical units
        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)  # (B, T, D)
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )

        # Cast all actions to float32 for consistency
        casted_action = {
            key: value.astype(np.float32) for key, value in unnormalized_action.items()
        }
        info = {}
        if self.memory_state is not None and context.get('memory_diagnostics'):
            info['memory_diagnostics'] = self.model.memory.last_diagnostics
        info['residual_scale'] = self.model.residual_scale
        if (options or {}).get('memory_diagnostics'):
            info['state_scores'] = model_pred['state_logits'].float().sigmoid().cpu().tolist()
        return casted_action, info
