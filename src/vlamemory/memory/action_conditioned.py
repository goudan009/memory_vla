"""
Action-subspace conditioned memory retrieval for π0.5.

This module implements the core contribution of the paper (§4.3):
- Shared lightweight memory bank (§4.2)
- Action-subspace conditioned queries (§4.3)
- Global + group-specific retrieval (§3.5)
- Memory modulation interface for π0.5 (§4.4)

Input shapes:
    - Current state h_t: (B, N_state, D_model) where D_model=2048 for π0.5
    - Memory bank: List of (timestep, (N_mem, D_model)) entries per episode
    - Action group descriptors: (G, D_group) where G is number of action groups

Output shapes:
    - Global memory context: (B, N_mem_tokens, D_model)
    - Group-specific contexts: (B, G, N_mem_tokens, D_model)
    - Combined for prefix injection: (B, N_total_mem_tokens, D_model)
      where N_total_mem_tokens = N_mem_tokens * (1 + G) or fused

Reference: paper.md §4.2-4.4
"""
from typing import Optional, List, Tuple, Dict
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

from .base import BaseMemoryBank, CrossAttentionBlock, GateFusion, TimestepEmbedder


class ActionGroupDescriptor(nnx.Module):
    """
    Action group descriptor combining learnable embedding with kinematic structure.

    For mobile bimanual robots, typical groups are:
    - Group 0: Base (mobile platform: x, y, theta)
    - Group 1: Torso/Trunk (height/tilt)
    - Group 2: Left arm + gripper
    - Group 3: Right arm + gripper

    Args:
        num_groups: Number of action groups (G)
        embedding_dim: Dimension of learnable group embedding
        use_kinematic_prior: Whether to incorporate kinematic structure k_g
        action_dims_per_group: List of action dimensions for each group (for k_g)
    """

    def __init__(
        self,
        num_groups: int,
        embedding_dim: int,
        use_kinematic_prior: bool = False,
        action_dims_per_group: Optional[List[int]] = None,
        kinematic_descriptors: Optional[List[List[float]]] = None,
        *,
        rngs: nnx.Rngs
    ):
        self.num_groups = num_groups
        self.embedding_dim = embedding_dim
        self.use_kinematic_prior = use_kinematic_prior

        # Learnable group embeddings e_g
        self.group_embeddings = nnx.Embed(num_groups, embedding_dim, rngs=rngs)

        if use_kinematic_prior:
            assert action_dims_per_group is not None
            assert len(action_dims_per_group) == num_groups
            if kinematic_descriptors is None or len(kinematic_descriptors) != num_groups:
                raise ValueError("kinematic_descriptors must provide one structural descriptor per group")
            self.action_dims = jnp.array(action_dims_per_group, dtype=jnp.float32)
            descriptor_widths = {len(descriptor) for descriptor in kinematic_descriptors}
            if len(descriptor_widths) != 1 or not descriptor_widths or 0 in descriptor_widths:
                raise ValueError("kinematic_descriptors must be a non-empty rectangular sequence")
            self.kinematic_features = jnp.asarray(kinematic_descriptors, dtype=jnp.float32)
            self.kinematic_proj = nnx.Linear(descriptor_widths.pop() + 1, embedding_dim, rngs=rngs)
        else:
            self.action_dims = None
            self.kinematic_features = None
            self.kinematic_proj = None

    def __call__(self, group_ids: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Get action group descriptors c_g = [e_g; k_g].

        Args:
            group_ids: (G,) array of group indices, or None to return all groups

        Returns:
            (G, D_descriptor) group descriptors
        """
        if group_ids is None:
            group_ids = jnp.arange(self.num_groups)

        # Learnable embeddings
        e_g = self.group_embeddings(group_ids)  # (G, embedding_dim)

        if self.use_kinematic_prior and self.kinematic_proj is not None:
            dims = self.action_dims[group_ids]  # (G,)
            dims_normalized = dims / jnp.sum(self.action_dims)  # Normalize to [0, 1]
            structure = self.kinematic_features[group_ids]
            k_features = jnp.concatenate([structure, dims_normalized[:, None]], axis=-1)
            k_g = self.kinematic_proj(k_features)  # (G, embedding_dim)

            # Concatenate e_g and k_g
            c_g = jnp.concatenate([e_g, k_g], axis=-1)  # (G, 2*embedding_dim)
        else:
            c_g = e_g  # (G, embedding_dim)

        return c_g


class ActionConditionedMemoryBank(nnx.Module):
    """
    Action-subspace conditioned memory retrieval (paper §4.2-4.4).

    Architecture:
    1. Shared lightweight memory bank (BaseMemoryBank)
    2. Global query: q_global = W_q_global @ h_t
    3. Group queries: q_g = W_q_group @ [h_t; c_g] for each action group g
    4. Cross-attention retrieval from shared bank
    5. Fusion: z_g = Fuse(r_global, r_g)
    6. Output: memory tokens ready for prefix injection

    Shape conventions:
        B: batch size
        N_state: number of state tokens (e.g., 968 for π0.5 prefix)
        D: model dimension (2048 for π0.5)
        G: number of action groups
        T: number of memory entries in bank
        N_mem: number of tokens per memory entry
    """

    def __init__(
        self,
        model_dim: int,
        num_groups: int,
        mem_length: int = 16,
        retrieval_layers: int = 2,
        num_attention_heads: int = 8,
        use_timestep_pe: bool = True,
        use_kinematic_prior: bool = False,
        action_dims_per_group: Optional[List[int]] = None,
        kinematic_descriptors: Optional[List[List[float]]] = None,
        consolidate_type: str = 'tome',
        fusion_type: str = 'gate',
        *,
        rngs: nnx.Rngs
    ):
        """
        Args:
            model_dim: Model hidden dimension (D_model, 2048 for π0.5)
            num_groups: Number of action groups (G)
            mem_length: Maximum number of memory entries (K in paper)
            retrieval_layers: Number of cross-attention layers
            num_attention_heads: Number of attention heads
            use_timestep_pe: Whether to add timestep positional encoding
            use_kinematic_prior: Whether to use kinematic features in group descriptors
            action_dims_per_group: Action dimensions for each group (for kinematic prior)
            consolidate_type: Memory consolidation strategy ('tome' or 'fifo')
            fusion_type: Fusion strategy for global+group memory ('gate', 'concat', or 'add')
        """
        self.model_dim = model_dim
        self.num_groups = num_groups
        self.mem_length = mem_length
        self.fusion_type = fusion_type

        # Shared memory bank
        self.bank = BaseMemoryBank(mem_length=mem_length, consolidate_type=consolidate_type)
        self.perceptual_bank = BaseMemoryBank(mem_length=mem_length, consolidate_type=consolidate_type)
        self.cognitive_bank = self.bank

        # Action group descriptors
        group_emb_dim = model_dim // 4  # Smaller embedding dimension
        self.group_descriptor = ActionGroupDescriptor(
            num_groups=num_groups,
            embedding_dim=group_emb_dim,
            use_kinematic_prior=use_kinematic_prior,
            action_dims_per_group=action_dims_per_group,
            kinematic_descriptors=kinematic_descriptors,
            rngs=rngs
        )

        descriptor_dim = group_emb_dim * 2 if use_kinematic_prior else group_emb_dim
        # Query projections
        self.global_query_proj = nnx.Linear(model_dim, model_dim, rngs=rngs)
        self.perceptual_global_query_proj = nnx.Linear(model_dim, model_dim, rngs=rngs)
        self.perceptual_group_query_proj = nnx.Linear(model_dim + descriptor_dim, model_dim, rngs=rngs)

        # Group query projects [h_t; c_g] -> query
        self.group_query_proj = nnx.Linear(model_dim + descriptor_dim, model_dim, rngs=rngs)

        # Cross-attention retrieval blocks
        self.global_retrieval = nnx.Sequential(
            *[CrossAttentionBlock(model_dim, num_attention_heads, rngs=rngs)
              for _ in range(retrieval_layers)]
        )
        self.group_retrieval = nnx.Sequential(
            *[CrossAttentionBlock(model_dim, num_attention_heads, rngs=rngs)
              for _ in range(retrieval_layers)]
        )
        self.perceptual_global_retrieval = nnx.Sequential(
            *[CrossAttentionBlock(model_dim, num_attention_heads, rngs=rngs)
              for _ in range(retrieval_layers)]
        )
        self.perceptual_group_retrieval = nnx.Sequential(
            *[CrossAttentionBlock(model_dim, num_attention_heads, rngs=rngs)
              for _ in range(retrieval_layers)]
        )

        # Timestep positional encoding
        self.use_timestep_pe = use_timestep_pe
        if use_timestep_pe:
            self.timestep_embedder = TimestepEmbedder(model_dim, rngs=rngs)

        # Fusion module
        if fusion_type == 'gate':
            self.fusion = GateFusion(model_dim, rngs=rngs)
        elif fusion_type == 'concat':
            # Project concatenated features back to model_dim
            self.fusion = nnx.Linear(model_dim * 2, model_dim, rngs=rngs)
        elif fusion_type == 'add':
            self.fusion = None  # Simple addition
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        self.memory_type_fusion = nnx.Linear(model_dim * 3, model_dim, rngs=rngs)

    def reset(self, episode_id: str):
        """Reset memory bank for a new episode."""
        self.bank.reset(episode_id)
        self.perceptual_bank.reset(episode_id)

    def write(self, episode_id: str, feat: jnp.ndarray, timestep: int, memory_type: str = 'cognitive'):
        """
        Write a memory entry to the bank.

        Args:
            episode_id: Episode identifier
            feat: Memory feature (N_mem, D)
            timestep: Current timestep
        """
        if memory_type == 'cognitive':
            self.cognitive_bank.write(episode_id, feat, timestep)
        elif memory_type == 'perceptual':
            self.perceptual_bank.write(episode_id, feat, timestep)
        else:
            raise ValueError(f"Unknown memory_type: {memory_type}")

    def retrieve(
        self,
        h_t: jnp.ndarray,
        episode_id: str,
        current_timestep: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Retrieve memory contexts for current state.

        Args:
            h_t: Current state representation (B, N_state, D)
            episode_id: Episode identifier
            current_timestep: Current timestep

        Returns:
            global_context: (B, N_retrieved, D) - Global memory context
            group_contexts: (B, G, N_retrieved, D) - Per-group memory contexts
        """
        B, N_state, D = h_t.shape
        assert D == self.model_dim

        # Empty episodes preserve the established zero-context behavior.
        if not self.bank.bank.get(episode_id) and not self.perceptual_bank.bank.get(episode_id):
            # No memory yet, return zeros
            N_retrieved = 1  # At least one token
            global_context = jnp.zeros((B, N_retrieved, D), dtype=h_t.dtype)
            group_contexts = jnp.zeros((B, self.num_groups, N_retrieved, D), dtype=h_t.dtype)
            return global_context, group_contexts

        # Retrieve memory entries
        memory_entries = self.bank.bank.get(episode_id) or self.perceptual_bank.bank[episode_id]
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

        # === Global retrieval ===
        # Aggregate h_t for query: mean pooling
        h_t_pooled = jnp.mean(h_t, axis=1, keepdims=True)  # (B, 1, D)
        q_global = self.global_query_proj(h_t_pooled)  # (B, 1, D)

        # Cross-attention retrieval
        r_global = q_global
        for layer in self.global_retrieval.layers:
            r_global = layer(r_global, memory_kv, memory_kv)  # (B, 1, D)

        # === Group-specific retrieval ===
        # Get group descriptors
        c_g = self.group_descriptor()  # (G, D_descriptor)
        G = c_g.shape[0]

        # Expand h_t_pooled for each group: (B, G, 1, D)
        h_t_expanded = jnp.tile(h_t_pooled[:, None, :, :], (1, G, 1, 1))  # (B, G, 1, D)

        # Expand c_g for batch: (B, G, D_descriptor)
        c_g_expanded = jnp.tile(c_g[None, :, :], (B, 1, 1))

        # Concatenate h_t and c_g: (B, G, D + D_descriptor)
        h_t_pooled_per_group = h_t_expanded.squeeze(2)  # (B, G, D)
        query_input = jnp.concatenate([h_t_pooled_per_group, c_g_expanded], axis=-1)

        # Flatten for projection: (B*G, D+D_descriptor)
        query_input_flat = query_input.reshape(B * G, -1)
        q_group_flat = self.group_query_proj(query_input_flat)  # (B*G, D)
        q_group = q_group_flat.reshape(B, G, 1, D)  # (B, G, 1, D)

        # Cross-attention for each group (vectorized over G)
        # Expand memory_kv: (B, G, T*N_mem, D)
        memory_kv_expanded = jnp.tile(memory_kv[:, None, :, :], (1, G, 1, 1))

        # Flatten batch and groups for layer processing
        q_group_flat = q_group.reshape(B * G, 1, D)
        memory_kv_flat = memory_kv_expanded.reshape(B * G, T * N_mem, D)

        r_group_flat = q_group_flat
        for layer in self.group_retrieval.layers:
            r_group_flat = layer(r_group_flat, memory_kv_flat, memory_kv_flat)  # (B*G, 1, D)

        r_group = r_group_flat.reshape(B, G, 1, D)  # (B, G, 1, D)

        # Perceptual entries use their own shared bank and query projections.
        perceptual_entries = self.perceptual_bank.bank.get(episode_id)
        if perceptual_entries:
            p_timesteps = jnp.array([t for t, _ in perceptual_entries], dtype=jnp.int32)
            p_feats = jnp.stack([f for _, f in perceptual_entries], axis=0)
            p_t, p_tokens, _ = p_feats.shape
            p_flat = p_feats.reshape(p_t * p_tokens, D)
            if self.use_timestep_pe:
                p_flat = p_flat + self.timestep_embedder(jnp.repeat(p_timesteps, p_tokens))
            p_kv = jnp.tile(p_flat[None, :, :], (B, 1, 1))
            p_global = self.perceptual_global_query_proj(h_t_pooled)
            for layer in self.perceptual_global_retrieval.layers:
                p_global = layer(p_global, p_kv, p_kv)
            p_group_input = jnp.concatenate([h_t_pooled_per_group, c_g_expanded], axis=-1).reshape(B * G, -1)
            p_query = self.perceptual_group_query_proj(p_group_input).reshape(B * G, 1, D)
            p_group_kv = jnp.tile(p_kv[:, None, :, :], (1, G, 1, 1)).reshape(B * G, p_t * p_tokens, D)
            for layer in self.perceptual_group_retrieval.layers:
                p_query = layer(p_query, p_group_kv, p_group_kv)
            p_group = p_query.reshape(B, G, 1, D)
        else:
            p_global = jnp.zeros_like(r_global)
            p_group = jnp.zeros_like(r_group)

        # === Fusion: combine r_global and r_group ===
        # Expand r_global for each group: (B, G, 1, D)
        r_global_expanded = jnp.tile(r_global[:, None, :, :], (1, G, 1, 1))

        if self.fusion_type == 'gate':
            # Flatten for gate fusion
            r_global_flat = r_global_expanded.reshape(B * G, 1, D)
            r_group_flat_fused = r_group.reshape(B * G, 1, D)
            z_g_flat = self.fusion(r_global_flat, r_group_flat_fused)  # (B*G, 1, D)
            z_g = z_g_flat.reshape(B, G, 1, D)
        elif self.fusion_type == 'concat':
            # Concatenate and project
            concat = jnp.concatenate([r_global_expanded, r_group], axis=-1)  # (B, G, 1, 2D)
            z_g = self.fusion(concat)  # (B, G, 1, D)
        elif self.fusion_type == 'add':
            z_g = r_global_expanded + r_group  # (B, G, 1, D)
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        # Output shapes
        global_context = r_global  # (B, 1, D)
        p_global_expanded = jnp.tile(p_global[:, None, :, :], (1, G, 1, 1))
        group_contexts = self.memory_type_fusion(
            jnp.concatenate([z_g, p_group, p_global_expanded], axis=-1)
        )  # (B, G, 1, D)

        return global_context, group_contexts

    def __call__(
        self,
        h_t: jnp.ndarray,
        episode_id: str,
        current_timestep: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Convenience wrapper for retrieve()."""
        return self.retrieve(h_t, episode_id, current_timestep)
