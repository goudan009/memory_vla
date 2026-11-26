"""Serializable V2 additions to the unchanged GR00T/V1 configuration."""

from dataclasses import dataclass
import math

from v1_groot_memoryvla.configs.model.gr00t_n1d6 import Gr00tN1d6Config as V1Config
from .model.state import validate_labels


@dataclass(init=False)
class Gr00tN1d6Config(V1Config):
    model_type: str = "Gr00tN1d6V2"
    memory_enabled: bool = True
    memory_state_labels: tuple = ()  # Must explicitly select labels after audit.
    memory_state_guidance: bool = True
    memory_event_low: float = .25
    memory_event_high: float = .75
    memory_recent_slots: int = 2
    memory_state_loss_weight: float | None = None  # Calibrate on real batches, then lock.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Empty labels are allowed for HF's default-config serialization, but
        # constructing an actual StateHead requires an explicit nonempty list.
        if self.memory_state_labels:
            self.memory_state_labels = validate_labels(self.memory_state_labels)
        if self.memory_state_loss_weight is not None and (
            not math.isfinite(self.memory_state_loss_weight) or self.memory_state_loss_weight <= 0
        ):
            raise ValueError("State loss weight must be finite and positive")


def adapter_options(config):
    return dict(state_labels=config.memory_state_labels,
                guidance_enabled=config.memory_state_guidance,
                event_low=config.memory_event_low, event_high=config.memory_event_high,
                recent_slots=config.memory_recent_slots)
