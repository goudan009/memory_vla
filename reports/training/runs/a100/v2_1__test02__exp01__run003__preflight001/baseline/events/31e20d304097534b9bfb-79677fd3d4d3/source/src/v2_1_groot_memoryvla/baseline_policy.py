"""V2_1 no-memory serving with the training checkpoint's FP32 action expert."""
import torch

from v1_groot_memoryvla.policy.gr00t_policy import Gr00tPolicy
from v1_groot_memoryvla.policy.policy import BasePolicy
from v1_groot_memoryvla.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor

from .loading import load_base


class BaselinePolicy(Gr00tPolicy):
    def __init__(self, embodiment_tag, model_path, *, device, strict=True, history_mode='normal', **kwargs):
        if history_mode != 'normal':
            raise ValueError('No-memory V2_1 checkpoint requires normal history mode')
        if embodiment_tag.value != 'behavior_r1_pro':
            raise ValueError('V2_1 currently supports the audited R1Pro mapping')
        BasePolicy.__init__(self, strict=strict)
        self.model = load_base(model_path, 'memory', device=device).eval()
        self.processor = Gr00tN1d6Processor.from_pretrained(model_path)
        self.processor.eval()
        self.embodiment_tag = embodiment_tag
        self.modality_configs = self.processor.get_modality_configs()[embodiment_tag.value]
        self.collate_fn = self.processor.collator
        language = self.modality_configs['language']
        if len(language.modality_keys) != 1 or len(language.delta_indices) != 1:
            raise ValueError('Only one language key and delta index are supported')
        self.language_key = language.modality_keys[0]
        self.memory_state = None
        self.history_mode = 'normal'
        self.device_type = next(self.model.parameters()).device.type

    def _get_action(self, observation, options=None):
        with torch.autocast(self.device_type, dtype=torch.bfloat16,
                            enabled=self.device_type in ('cuda', 'cpu')):
            return super()._get_action(observation, options)
