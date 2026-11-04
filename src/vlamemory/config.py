"""
Memory module configuration for π0.5 integration.

This defines the configuration options for memory injection,
allowing switchable memory types and injection modes via config.
"""

from dataclasses import dataclass
from typing import Literal, Optional, List


@dataclass
class MemoryConfig:
    """Configuration for memory module integration."""

    # Memory type selection
    memory_type: Literal['none', 'uniform', 'action_conditioned'] = 'none'

    # Memory bank parameters
    mem_length: int = 16
    model_dim: int = 2048
    retrieval_layers: int = 2
    num_attention_heads: int = 8
    use_timestep_pe: bool = True
    consolidate_type: Literal['tome', 'fifo'] = 'tome'

    # Action-conditioned specific parameters
    num_action_groups: int = 4
    use_kinematic_prior: bool = False
    action_dims_per_group: Optional[List[int]] = None
    kinematic_descriptors: Optional[List[List[float]]] = None
    fusion_type: Literal['gate', 'concat', 'add'] = 'gate'

    # Token output configuration
    num_mem_tokens: Optional[int] = None  # None = auto (5 for action_conditioned, 1 for uniform)

    # Injection mode
    injection_mode: Literal['prefix-context', 'modulation', 'memory-expert'] = 'prefix-context'
    use_projection: bool = False

    def __post_init__(self):
        """Validate and set defaults."""
        # Set default token count based on memory type
        if self.num_mem_tokens is None:
            if self.memory_type == 'action_conditioned':
                # 1 global + G group-specific tokens
                self.num_mem_tokens = 1 + self.num_action_groups
            elif self.memory_type == 'uniform':
                # Match action-conditioned for fair comparison
                self.num_mem_tokens = 1 + self.num_action_groups
            else:
                self.num_mem_tokens = 0

        # Validate action-conditioned parameters
        if self.memory_type == 'action_conditioned':
            if self.use_kinematic_prior:
                if self.action_dims_per_group is None:
                    raise ValueError("action_dims_per_group required when use_kinematic_prior=True")
                if len(self.action_dims_per_group) != self.num_action_groups:
                    raise ValueError(
                        f"action_dims_per_group length {len(self.action_dims_per_group)} "
                        f"must match num_action_groups {self.num_action_groups}"
                    )
