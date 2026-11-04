"""
Memory modulation interface for π0.5 integration.

This module handles:
1. Converting memory bank output to prefix tokens
2. Injecting into π0.5's embed_prefix at position 968
3. Updating masks and position IDs
4. Supporting different injection modes (prefix-context/modulation/memory-expert)
"""

from typing import Tuple, Literal, Optional
import jax
import jax.numpy as jnp
from flax import nnx

from .memory import ActionConditionedMemoryBank, UniformMemoryBank


class MemoryToPrefix(nnx.Module):
    """
    Convert memory retrieval results to prefix tokens.

    For ActionConditionedMemoryBank:
        - Input: global_context (B, 1, D), group_contexts (B, G, 1, D)
        - Output: (B, N_mem_tokens, D) where N_mem_tokens = 1 + G (or configured)

    For UniformMemoryBank:
        - Input: memory_context (B, 1, D)
        - Output: (B, N_mem_tokens, D) where N_mem_tokens = 1
    """

    def __init__(
        self,
        model_dim: int,
        num_mem_tokens: Optional[int] = 5,
        use_projection: bool = False,
        *,
        rngs: nnx.Rngs
    ):
        """
        Args:
            model_dim: Model hidden dimension (2048 for π0.5)
            num_mem_tokens: Number of memory tokens to output, or None for auto (1+G)
            use_projection: Whether to apply additional linear projection
        """
        self.model_dim = model_dim
        self.num_mem_tokens = num_mem_tokens
        self.use_projection = use_projection

        if use_projection:
            self.proj = nnx.Linear(model_dim, model_dim, rngs=rngs)
        else:
            self.proj = None

    def from_action_conditioned(
        self,
        global_context: jnp.ndarray,
        group_contexts: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Convert action-conditioned retrieval to prefix tokens.

        Args:
            global_context: (B, 1, D)
            group_contexts: (B, G, 1, D)

        Returns:
            memory_tokens: (B, N_mem_tokens, D)
        """
        B, _, D = global_context.shape
        G = group_contexts.shape[1]

        # Reshape group_contexts: (B, G, 1, D) -> (B, G, D)
        group_contexts_reshaped = group_contexts.squeeze(2)

        # Concatenate global + groups: (B, 1+G, D)
        memory_tokens = jnp.concatenate([global_context, group_contexts_reshaped], axis=1)

        if self.num_mem_tokens is not None and self.num_mem_tokens != (1 + G):
            # Adjust to desired number of tokens
            if self.num_mem_tokens < (1 + G):
                # Truncate
                memory_tokens = memory_tokens[:, :self.num_mem_tokens, :]
            else:
                # Pad with zeros
                pad_width = ((0, 0), (0, self.num_mem_tokens - (1 + G)), (0, 0))
                memory_tokens = jnp.pad(memory_tokens, pad_width, mode='constant')

        if self.use_projection and self.proj is not None:
            # Apply projection to each token
            B, N, D = memory_tokens.shape
            memory_tokens_flat = memory_tokens.reshape(B * N, D)
            memory_tokens_flat = self.proj(memory_tokens_flat)
            memory_tokens = memory_tokens_flat.reshape(B, N, D)

        return memory_tokens

    def from_uniform(
        self,
        memory_context: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Convert uniform retrieval to prefix tokens.

        Args:
            memory_context: (B, 1, D)

        Returns:
            memory_tokens: (B, N_mem_tokens, D)
        """
        B, _, D = memory_context.shape

        if self.num_mem_tokens is not None and self.num_mem_tokens > 1:
            # Expand to desired number of tokens by repeating
            memory_tokens = jnp.tile(memory_context, (1, self.num_mem_tokens, 1))
        else:
            memory_tokens = memory_context

        if self.use_projection and self.proj is not None:
            # Apply projection
            B, N, D = memory_tokens.shape
            memory_tokens_flat = memory_tokens.reshape(B * N, D)
            memory_tokens_flat = self.proj(memory_tokens_flat)
            memory_tokens = memory_tokens_flat.reshape(B, N, D)

        return memory_tokens


InjectionMode = Literal['prefix-context', 'modulation', 'memory-expert']


class MemoryModulation(nnx.Module):
    """
    Full memory modulation interface for π0.5.

    Usage:
    1. Initialize with memory bank (action-conditioned or uniform)
    2. Call inject_prefix() to extend prefix from 968 to 968+N tokens
    3. Returns modified (embeddings, input_mask, ar_mask, position_ids)
    """

    def __init__(
        self,
        memory_bank: nnx.Module,  # ActionConditionedMemoryBank or UniformMemoryBank
        model_dim: int = 2048,
        num_mem_tokens: Optional[int] = 5,
        injection_mode: InjectionMode = 'prefix-context',
        use_projection: bool = False,
        *,
        rngs: nnx.Rngs
    ):
        """
        Args:
            memory_bank: Memory retrieval module (ActionConditionedMemoryBank or UniformMemoryBank)
            model_dim: Model hidden dimension (2048 for π0.5)
            num_mem_tokens: Number of memory tokens, or None for auto
            injection_mode: How to inject memory ('prefix-context', 'modulation', 'memory-expert')
            use_projection: Whether to apply projection to memory tokens
        """
        self.memory_bank = memory_bank
        self.model_dim = model_dim
        self.injection_mode = injection_mode
        self.is_action_conditioned = isinstance(memory_bank, ActionConditionedMemoryBank)

        # Memory-to-prefix converter
        self.mem_to_prefix = MemoryToPrefix(
            model_dim=model_dim,
            num_mem_tokens=num_mem_tokens,
            use_projection=use_projection,
            rngs=rngs
        )

        # Mode-specific layers
        if injection_mode == 'modulation':
            # Modulation: multiplicative gating
            self.gate = nnx.Sequential(
                nnx.Linear(model_dim, model_dim, rngs=rngs),
                nnx.sigmoid
            )
            self.action_scale = nnx.Linear(model_dim, model_dim, rngs=rngs)
            self.action_shift = nnx.Linear(model_dim, model_dim, rngs=rngs)
        elif injection_mode == 'memory-expert':
            # Memory expert: separate transformer layer
            # Simplified: just add a linear projection
            self.expert_proj = nnx.Linear(model_dim, model_dim, rngs=rngs)
            self.action_expert_proj = nnx.Linear(model_dim, model_dim, rngs=rngs)
        else:
            # prefix-context: no additional layers needed
            pass

    def inject_prefix(
        self,
        prefix_embeddings: jnp.ndarray,
        input_mask: jnp.ndarray,
        ar_mask: jnp.ndarray,
        episode_id: str,
        current_timestep: int,
        position_ids: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Inject memory tokens into π0.5 prefix.

        Args:
            prefix_embeddings: Original prefix (B, 968, D)
            input_mask: Original input mask (B, 968)
            ar_mask: Original autoregressive mask (B, 968)
            episode_id: Episode identifier for memory bank lookup
            current_timestep: Current timestep
            position_ids: Optional position IDs (B, 968)

        Returns:
            new_embeddings: Extended prefix (B, 968+N, D)
            new_input_mask: Extended mask (B, 968+N)
            new_ar_mask: Extended AR mask (B, 968+N)
            new_position_ids: Extended position IDs (B, 968+N) or None
        """
        B, S_orig, D = prefix_embeddings.shape
        assert S_orig == 968, f"Expected prefix length 968, got {S_orig}"
        assert D == self.model_dim

        # Retrieve memory
        if self.is_action_conditioned:
            global_context, group_contexts = self.memory_bank.retrieve(
                prefix_embeddings, episode_id, current_timestep
            )
            memory_tokens = self.mem_to_prefix.from_action_conditioned(
                global_context, group_contexts
            )
        else:
            memory_context = self.memory_bank.retrieve(
                prefix_embeddings, episode_id, current_timestep
            )
            memory_tokens = self.mem_to_prefix.from_uniform(memory_context)

        N_mem = memory_tokens.shape[1]

        # Apply injection mode
        if self.injection_mode == 'prefix-context':
            # Simply concatenate at the end of prefix
            new_embeddings = jnp.concatenate([prefix_embeddings, memory_tokens], axis=1)

        elif self.injection_mode == 'modulation':
            # Modulate prefix with memory
            # Broadcast memory across prefix tokens
            memory_aggregated = jnp.mean(memory_tokens, axis=1, keepdims=True)  # (B, 1, D)
            gate_values = self.gate(memory_aggregated)  # (B, 1, D)
            gate_values = jnp.tile(gate_values, (1, S_orig, 1))  # (B, 968, D)

            modulated_prefix = prefix_embeddings * gate_values
            new_embeddings = jnp.concatenate([modulated_prefix, memory_tokens], axis=1)

        elif self.injection_mode == 'memory-expert':
            # Memory expert: add expert-processed memory
            expert_memory = self.expert_proj(memory_tokens)  # (B, N_mem, D)
            new_embeddings = jnp.concatenate([prefix_embeddings, expert_memory], axis=1)

        else:
            raise ValueError(f"Unknown injection_mode: {self.injection_mode}")

        # Extend masks
        # Memory tokens are always visible (mask=1, ar_mask=0 for bidirectional attention)
        mem_input_mask = jnp.ones((B, N_mem), dtype=input_mask.dtype)
        mem_ar_mask = jnp.zeros((*ar_mask.shape[:-1], N_mem), dtype=ar_mask.dtype)

        new_input_mask = jnp.concatenate([input_mask, mem_input_mask], axis=1)
        new_ar_mask = jnp.concatenate([ar_mask, mem_ar_mask], axis=-1)

        # Extend position IDs if provided
        if position_ids is not None:
            # Memory tokens continue from position 968
            max_pos = jnp.max(position_ids, axis=1, keepdims=True)  # (B, 1)
            mem_position_ids = max_pos + jnp.arange(1, N_mem + 1)[None, :]  # (B, N_mem)
            new_position_ids = jnp.concatenate([position_ids, mem_position_ids], axis=1)
        else:
            new_position_ids = None

        return new_embeddings, new_input_mask, new_ar_mask, new_position_ids

    def reset(self, episode_id: str):
        """Reset memory bank for new episode."""
        self.memory_bank.reset(episode_id)

    def modulate_action_expert(
        self,
        action_hidden: jnp.ndarray,
        episode_id: str,
        current_timestep: int,
    ) -> jnp.ndarray:
        """Apply retrieved group contexts to shared action-expert states.

        Args:
            action_hidden: Shared action-expert states, (B, T_action, D).
        Returns:
            Group-conditioned hidden states, (B, G, T_action, D).
        """
        if action_hidden.ndim != 3 or action_hidden.shape[-1] != self.model_dim:
            raise ValueError(f"action_hidden must have shape (B, T, {self.model_dim})")
        if not self.is_action_conditioned:
            raise ValueError("Action-expert group modulation requires ActionConditionedMemoryBank")
        _, group_contexts = self.memory_bank.retrieve(action_hidden, episode_id, current_timestep)
        context = group_contexts.squeeze(2)
        if self.injection_mode == 'modulation':
            scale = jnp.tanh(self.action_scale(context))[:, :, None, :]
            shift = self.action_shift(context)[:, :, None, :]
            return action_hidden[:, None, :, :] * (1 + scale) + shift
        if self.injection_mode == 'memory-expert':
            expert = self.action_expert_proj(context)[:, :, None, :]
            return action_hidden[:, None, :, :] + expert
        return action_hidden[:, None, :, :] + context[:, :, None, :]

    def write(
        self,
        episode_id: str,
        feat: jnp.ndarray,
        timestep: int,
        memory_type: str = 'cognitive',
    ):
        """Write a cognitive or perceptual entry to the configured bank."""
        if self.is_action_conditioned:
            self.memory_bank.write(episode_id, feat, timestep, memory_type=memory_type)
        elif memory_type != 'cognitive':
            raise ValueError("UniformMemoryBank accepts only the shared cognitive memory stream")
        else:
            self.memory_bank.write(episode_id, feat, timestep)
