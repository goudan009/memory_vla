"""
Uniform (non-action-conditioned) memory retrieval baseline.

This is the baseline comparison for paper §5.3. Can be switched via config:
    memory_type: 'action_conditioned' | 'uniform'

No action-group conditioning, single global retrieval only.
"""

from typing import List, Tuple, Optional
import jax
import jax.numpy as jnp
from flax import nnx

from .base import BaseMemoryBank, TimestepEmbedder, CrossAttentionBlock


class UniformMemoryBank(nnx.Module):
    """
    Uniform memory retrieval without action-subspace conditioning.

    Architecture:
    1. Shared lightweight memory bank (BaseMemoryBank)
    2. Single global query: q = W_q @ h_t
    3. Cross-attention retrieval from bank
    4. Output: single memory context for all action groups

    This is the baseline that does NOT use action-group structure.
    """

    def __init__(
        self,
        model_dim: int,
        mem_length: int = 16,
        retrieval_layers: int = 2,
        num_attention_heads: int = 8,
        use_timestep_pe: bool = True,
        consolidate_type: str = 'tome',
        num_output_tokens: int = 1,
        *,
        rngs: nnx.Rngs
    ):
        """
        Args:
            model_dim: Model hidden dimension (D_model, 2048 for π0.5)
            mem_length: Maximum number of memory entries (K in paper)
            retrieval_layers: Number of cross-attention layers
            num_attention_heads: Number of attention heads
            use_timestep_pe: Whether to add timestep positional encoding
            consolidate_type: Memory consolidation strategy ('tome' or 'fifo')
            num_output_tokens: Number of tokens to output (for fair comparison with action-conditioned)
        """
        self.model_dim = model_dim
        self.mem_length = mem_length
        self.num_output_tokens = num_output_tokens

        # Shared memory bank
        self.bank = BaseMemoryBank(mem_length=mem_length, consolidate_type=consolidate_type)

        # Query projection
        self.query_proj = nnx.Linear(model_dim, model_dim, rngs=rngs)

        # Cross-attention retrieval blocks
        self.retrieval = nnx.Sequential(
            *[CrossAttentionBlock(model_dim, num_attention_heads, rngs=rngs)
              for _ in range(retrieval_layers)]
        )

        # Timestep positional encoding
        self.use_timestep_pe = use_timestep_pe
        if use_timestep_pe:
            self.timestep_embedder = TimestepEmbedder(model_dim, rngs=rngs)

    def reset(self, episode_id: str):
        """Reset memory bank for a new episode."""
        self.bank.reset(episode_id)

    def write(self, episode_id: str, feat: jnp.ndarray, timestep: int):
        """
        Write a memory entry to the bank.

        Args:
            episode_id: Episode identifier
            feat: Memory feature (N_mem, D)
            timestep: Current timestep
        """
        self.bank.write(episode_id, feat, timestep)

    def retrieve(
        self,
        h_t: jnp.ndarray,
        episode_id: str,
        current_timestep: int
    ) -> jnp.ndarray:
        """
        Retrieve uniform memory context for current state.

        Args:
            h_t: Current state representation (B, N_state, D)
            episode_id: Episode identifier
            current_timestep: Current timestep

        Returns:
            memory_context: (B, num_output_tokens, D) - Uniform memory context
        """
        B, N_state, D = h_t.shape
        assert D == self.model_dim

        # Get memory entries from bank
        if episode_id not in self.bank.bank or len(self.bank.bank[episode_id]) == 0:
            # No memory yet, return zeros with correct number of tokens
            memory_context = jnp.zeros((B, self.num_output_tokens, D), dtype=h_t.dtype)
            return memory_context

        # Retrieve memory entries
        memory_entries = self.bank.bank[episode_id]
        timesteps = jnp.array([t for t, _ in memory_entries], dtype=jnp.int32)
        memory_feats = jnp.stack([f for _, f in memory_entries], axis=0)  # (T, N_mem, D)
        T, N_mem, _ = memory_feats.shape

        # Flatten for batch processing: (T*N_mem, D)
        memory_flat = memory_feats.reshape(T * N_mem, D)

        # Add timestep positional encoding
        if self.use_timestep_pe:
            # Expand timesteps to match each token: (T,) -> (T, N_mem) -> (T*N_mem,)
            timesteps_expanded = jnp.repeat(timesteps, N_mem)
            timestep_pe = self.timestep_embedder(timesteps_expanded)  # (T*N_mem, D)
            memory_flat = memory_flat + timestep_pe

        # Reshape for cross-attention: batch over B
        memory_kv = jnp.tile(memory_flat[None, :, :], (B, 1, 1))  # (B, T*N_mem, D)

        # Aggregate h_t for query: mean pooling
        h_t_pooled = jnp.mean(h_t, axis=1, keepdims=True)  # (B, 1, D)
        query = self.query_proj(h_t_pooled)  # (B, 1, D)

        # Cross-attention retrieval
        memory_context = query
        for layer in self.retrieval.layers:
            memory_context = layer(memory_context, memory_kv, memory_kv)  # (B, 1, D)

        # Expand to match number of output tokens if needed
        if self.num_output_tokens > 1:
            # Replicate the single retrieved context to num_output_tokens
            # memory_context is (B, 1, D), tile to (B, num_output_tokens, D)
            memory_context = jnp.repeat(memory_context, self.num_output_tokens, axis=1)

        return memory_context

    def __call__(
        self,
        h_t: jnp.ndarray,
        episode_id: str,
        current_timestep: int
    ) -> jnp.ndarray:
        """Convenience wrapper for retrieve()."""
        return self.retrieve(h_t, episode_id, current_timestep)
