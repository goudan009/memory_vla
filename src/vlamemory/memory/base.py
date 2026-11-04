"""
Base memory bank implementations adapted from MemoryVLA for π0.5.

Key changes from MemoryVLA:
- Adapted for π0.5's PaliGemma width (2048 instead of 4096)
- Designed to work with JAX/Flax instead of PyTorch
- Interface designed for prefix token injection into π0.5

Reference: github.com/shihao1895/MemoryVLA
"""
from typing import Optional, Dict, List, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx


class TimestepEmbedder(nnx.Module):
    """Timestep embedding module using sinusoidal positional encoding."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, *, rngs: nnx.Rngs):
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nnx.Sequential(
            nnx.Linear(frequency_embedding_size, hidden_size, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_size, hidden_size, rngs=rngs),
        )

    def timestep_embedding(self, t: jnp.ndarray, max_period: int = 10000) -> jnp.ndarray:
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: (T,) array of timesteps
            max_period: controls the minimum frequency of the embeddings

        Returns:
            (T, frequency_embedding_size) embeddings
        """
        half = self.frequency_embedding_size // 2
        freqs = jnp.exp(
            -jnp.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half
        )
        args = t[:, None] * freqs[None, :]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        return embedding

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            t: (T,) timestep indices

        Returns:
            (T, hidden_size) embedded timesteps
        """
        t_freq = self.timestep_embedding(t)
        t_emb = self.mlp(t_freq)
        return t_emb


class CrossAttentionBlock(nnx.Module):
    """Cross-attention block for memory retrieval."""

    def __init__(self, dim: int, num_heads: int = 8, *, rngs: nnx.Rngs):
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.head_dim = dim // num_heads

        self.q_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.k_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.v_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.out_proj = nnx.Linear(dim, dim, rngs=rngs)

        self.norm_q = nnx.LayerNorm(dim, rngs=rngs)
        self.norm_kv = nnx.LayerNorm(dim, rngs=rngs)

    def __call__(self, query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
        """
        Cross-attention: query attends to key-value pairs.

        Args:
            query: (B, N_q, D)
            key: (B, N_kv, D)
            value: (B, N_kv, D)

        Returns:
            (B, N_q, D) attended output
        """
        B, N_q, D = query.shape
        _, N_kv, _ = key.shape

        # Normalize
        query = self.norm_q(query)
        key = self.norm_kv(key)
        value = self.norm_kv(value)

        # Project and reshape to (B, num_heads, N, head_dim)
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = jnp.sqrt(self.head_dim)
        attn_weights = jax.nn.softmax(jnp.einsum('bhqd,bhkd->bhqk', q, k) / scale, axis=-1)
        attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)

        # Reshape and project back
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, N_q, D)
        output = self.out_proj(attn_output)

        return query + output  # Residual connection


class GateFusion(nnx.Module):
    """Gate fusion mechanism to combine working memory and retrieved episode memory."""

    def __init__(self, dim: int, *, rngs: nnx.Rngs):
        self.dim = dim
        self.gate_proj = nnx.Linear(dim * 2, dim, rngs=rngs)
        self.norm = nnx.LayerNorm(dim, rngs=rngs)

    def __call__(self, working_mem: jnp.ndarray, episode_mem: jnp.ndarray) -> jnp.ndarray:
        """
        Fuse working memory with retrieved episode memory using gating.

        Args:
            working_mem: (B, N, D) current state representation
            episode_mem: (B, N, D) retrieved memory

        Returns:
            (B, N, D) fused representation
        """
        concat = jnp.concatenate([working_mem, episode_mem], axis=-1)
        gate = jax.nn.sigmoid(self.gate_proj(concat))
        fused = gate * working_mem + (1 - gate) * episode_mem
        return self.norm(fused)


class BaseMemoryBank:
    """
    Base memory bank storing compressed history representations.

    Adapted from MemoryVLA's CogMemBank/PerMemBank structure but simplified
    for π0.5 integration. Uses episode-aware storage with fixed capacity.

    Attributes:
        bank: Dict[episode_id -> List[(timestep, features)]]
        mem_length: Maximum number of memory entries per episode
        consolidate_type: 'fifo' or 'tome' (Token Merge)
    """

    def __init__(
        self,
        mem_length: int = 16,
        consolidate_type: str = 'tome',
    ):
        assert consolidate_type in ('fifo', 'tome')
        self.mem_length = mem_length
        self.consolidate_type = consolidate_type
        self.bank: Dict[str, List[Tuple[int, jnp.ndarray]]] = {}

    def reset(self, episode_id: Optional[str] = None):
        """Clear one episode, or all episodes when no ID is provided."""
        if episode_id is None:
            self.bank = {}
        else:
            self.clear_episode(episode_id)

    def clear_episode(self, episode_id: str):
        """Remove a specific episode from the bank."""
        self.bank.pop(episode_id, None)

    def _consolidate_with_token_merge(self, episode_id: str):
        """
        Merge the most similar adjacent memory entries using cosine similarity.

        This is the ToME (Token Merging) strategy from MemoryVLA.
        """
        bank = self.bank.get(episode_id, [])
        T = len(bank)
        if T < 2:
            return

        feats = [feat for (_, feat) in bank]

        # Compute pairwise cosine similarities
        sims = []
        for i in range(T - 1):
            f1 = feats[i].reshape(-1)
            f2 = feats[i + 1].reshape(-1)
            sim = jnp.dot(f1, f2) / (jnp.linalg.norm(f1) * jnp.linalg.norm(f2) + 1e-8)
            sims.append(float(sim))

        # Find the most similar pair
        idx_max = int(np.argmax(sims))

        # Merge them
        timestep_i, feat_i = bank[idx_max]
        timestep_j, feat_j = bank[idx_max + 1]
        fused_feat = 0.5 * (feat_i + feat_j)

        # Update bank
        bank[idx_max] = (timestep_i, fused_feat)
        bank.pop(idx_max + 1)

    def write(self, episode_id: str, feat: jnp.ndarray, timestep: int):
        """
        Write a new memory entry to the bank.

        Args:
            episode_id: Episode identifier
            feat: (N, D) or (D,) feature representation
            timestep: Current timestep within episode
        """
        if episode_id not in self.bank:
            self.bank[episode_id] = []

        # Store a copy to avoid mutation
        self.bank[episode_id].append((timestep, jnp.array(feat)))

        # Consolidate if over capacity
        while len(self.bank[episode_id]) > self.mem_length:
            if self.consolidate_type == 'fifo':
                self.bank[episode_id] = self.bank[episode_id][-self.mem_length:]
            elif self.consolidate_type == 'tome':
                self._consolidate_with_token_merge(episode_id)

    def read(self, episode_id: str) -> Optional[List[Tuple[int, jnp.ndarray]]]:
        """
        Read all memory entries for an episode.

        Args:
            episode_id: Episode identifier

        Returns:
            List of (timestep, features) tuples, or None if episode not in bank
        """
        return self.bank.get(episode_id)
