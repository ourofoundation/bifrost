"""
Transformer components for BIFROST model.

This module contains the transformer blocks and related components
used in the BIFROST architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.

    This implementation uses PyTorch's built-in multi-head attention
    with causal masking for autoregressive generation.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Single multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,  # Use batch_first format
        )

        # Output projection (handled internally by MultiheadAttention)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass through multi-head attention.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
            is_causal: Whether to use causal attention (for autoregressive)

        Returns:
            Attention output (batch_size, seq_len, d_model)
        """
        # Use PyTorch's built-in multi-head attention
        attn_output, _ = self.attention(
            query=x,
            key=x,
            value=x,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
        )

        return self.dropout(attn_output)

    # --- Incremental/KV-cache utilities ---
    def _qkv_from_in_proj(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project inputs to Q, K, V using the internal MultiheadAttention weights.

        Args:
            x: (batch, seq, embed_dim)

        Returns:
            Tuple (q, k, v) each shaped (batch, seq, embed_dim)
        """
        mha: nn.MultiheadAttention = self.attention
        w = mha.in_proj_weight  # (3E, E)
        b = mha.in_proj_bias  # (3E,)
        E = self.d_model
        w_q, w_k, w_v = w[:E, :], w[E : 2 * E, :], w[2 * E :, :]
        b_q = b[:E] if b is not None else None
        b_k = b[E : 2 * E] if b is not None else None
        b_v = b[2 * E :] if b is not None else None

        # x: (B, S, E) -> (B, S, E)
        q = torch.matmul(x, w_q.T)
        k = torch.matmul(x, w_k.T)
        v = torch.matmul(x, w_v.T)
        if b is not None:
            q = q + b_q
            k = k + b_k
            v = v + b_v
        return q, k, v

    def compute_kv_cache(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute K,V projections for a full sequence to initialize cache.

        Args:
            x: (batch, seq, embed_dim)

        Returns:
            k_cache, v_cache: (batch, n_heads, seq, head_dim)
        """
        q, k, v = self._qkv_from_in_proj(x)
        B, S, E = k.shape
        H = self.n_heads
        D = self.head_dim
        # reshape to heads
        k = k.view(B, S, H, D).transpose(1, 2)  # (B, H, S, D)
        v = v.view(B, S, H, D).transpose(1, 2)  # (B, H, S, D)
        return k, v

    def incremental_step(
        self,
        x_t: torch.Tensor,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute attention output for a single timestep using cached K,V.

        Args:
            x_t: (batch, 1, embed_dim) current token input
            past_k: (batch, n_heads, seq, head_dim)
            past_v: (batch, n_heads, seq, head_dim)

        Returns:
            attn_out_t: (batch, 1, embed_dim)
            new_k: (batch, n_heads, seq+1, head_dim)
            new_v: (batch, n_heads, seq+1, head_dim)
        """
        B, S1, E = x_t.shape
        H = self.n_heads
        D = self.head_dim

        # Project current step
        q_t, k_t, v_t = self._qkv_from_in_proj(x_t)  # (B,1,E)
        # reshape heads
        q_t = q_t.view(B, S1, H, D).transpose(1, 2)  # (B,H,1,D)
        k_t = k_t.view(B, S1, H, D).transpose(1, 2)  # (B,H,1,D)
        v_t = v_t.view(B, S1, H, D).transpose(1, 2)  # (B,H,1,D)

        # Append to cache
        new_k = torch.cat([past_k, k_t], dim=2) if past_k is not None else k_t
        new_v = torch.cat([past_v, v_t], dim=2) if past_v is not None else v_t

        # Scaled dot-product attention for single query position
        # attn_weights: (B,H,1,seq+1)
        scale = 1.0 / math.sqrt(D)
        attn_scores = torch.matmul(q_t, new_k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # attention output: (B,H,1,D)
        context = torch.matmul(attn_weights, new_v)
        # Merge heads -> (B,1,E)
        context = context.transpose(1, 2).contiguous().view(B, S1, E)

        # Output projection uses MHA's out_proj
        out = self.attention.out_proj(context)
        out = self.dropout(out)
        return out, new_k, new_v


class FeedForwardNetwork(nn.Module):
    """
    Feed-forward network used in transformer blocks.

    This consists of two linear layers with a GELU activation and dropout
    in between, following the standard transformer architecture.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize feed-forward network.

        Args:
            d_model: Model dimension
            d_ff: Hidden dimension of feed-forward network
            dropout: Dropout probability
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # Initialize weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feed-forward network.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Single transformer block consisting of multi-head attention and feed-forward network.

    This follows the standard transformer architecture with pre-layer normalization
    and residual connections.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize transformer block.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Hidden dimension of feed-forward network
            dropout: Dropout probability
        """
        super().__init__()

        # Multi-head attention sublayer
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.attention_norm = nn.LayerNorm(d_model)

        # Feed-forward network sublayer
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        self.feed_forward_norm = nn.LayerNorm(d_model)

        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Multi-head attention with residual connection and layer norm
        attn_output = self.attention(
            x, mask=mask, key_padding_mask=key_padding_mask, is_causal=False
        )
        x = self.attention_norm(x + self.dropout(attn_output))

        # Feed-forward network with residual connection and layer norm
        ffn_output = self.feed_forward(x)
        x = self.feed_forward_norm(x + self.dropout(ffn_output))

        return x

    def incremental_forward(
        self,
        x_t: torch.Tensor,
        cache: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Incremental forward for a single timestep with KV cache.

        Args:
            x_t: (batch, 1, d_model) input at this layer
            cache: (k, v) where each is (batch, n_heads, seq, head_dim)

        Returns:
            y_t: (batch, 1, d_model) output at this layer
            new_cache: updated (k, v)
        """
        k, v = cache
        attn_out_t, new_k, new_v = self.attention.incremental_step(x_t, k, v)
        # Residual and norm
        x_after_attn = self.attention_norm(x_t + self.dropout(attn_out_t))
        # FFN with residual and norm
        ffn_out = self.feed_forward(x_after_attn)
        y_t = self.feed_forward_norm(x_after_attn + self.dropout(ffn_out))
        return y_t, (new_k, new_v)


class BIFROSTTransformer(nn.Module):
    """
    Stack of transformer blocks for BIFROST.

    This module contains multiple transformer blocks stacked sequentially
    to form the core of the BIFROST architecture.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 16,
        n_layers: int = 16,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        """
        Initialize BIFROST transformer.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Hidden dimension of feed-forward network
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        # Stack of transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        # Final layer normalization
        self.final_norm = nn.LayerNorm(d_model)

        # Store parameters
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through transformer stack.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Pass through each transformer block
        for block in self.blocks:
            x = block(x, mask=mask, key_padding_mask=key_padding_mask)

        # Apply final layer normalization
        x = self.final_norm(x)

        return x

    def build_kv_caches(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]:
        """
        Run a full forward pass and initialize per-layer KV caches from inputs of each layer.

        Args:
            x: (batch, seq, d_model)
        Returns:
            hidden_states: (batch, seq, d_model)
            caches: tuple of (k,v) per layer, each (batch, n_heads, seq, head_dim)
        """
        caches = []
        for block in self.blocks:
            # Build K,V cache from current layer input x
            k, v = block.attention.compute_kv_cache(x)
            caches.append((k, v))
            # Standard forward for this layer
            x = block(x, mask=mask, key_padding_mask=key_padding_mask)
        x = self.final_norm(x)
        return x, tuple(caches)

    def incremental_forward_step(
        self,
        x_t: torch.Tensor,
        caches: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    ) -> Tuple[torch.Tensor, Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]:
        """
        Process a single timestep through all layers using KV caches.

        Args:
            x_t: (batch, 1, d_model)
            caches: tuple of (k,v) per layer

        Returns:
            y_t: (batch, 1, d_model)
            new_caches: updated caches
        """
        new_caches = []
        for block, kv in zip(self.blocks, caches):
            x_t, kv = block.incremental_forward(x_t, kv)
            new_caches.append(kv)
        y_t = self.final_norm(x_t)
        return y_t, tuple(new_caches)

    def get_attention_weights(
        self, x: torch.Tensor, layer_idx: int = 0
    ) -> torch.Tensor:
        """
        Get attention weights from a specific layer (for analysis).

        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            layer_idx: Index of layer to get attention from

        Returns:
            Attention weights (batch_size, n_heads, seq_len, seq_len)
        """
        if layer_idx >= self.n_layers:
            raise ValueError(f"Layer index {layer_idx} out of range")

        block = self.blocks[layer_idx]
        return block.attention.attention(x, x, x)[1]


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Create causal attention mask for autoregressive generation.

    Args:
        seq_len: Sequence length
        device: Device to create mask on

    Returns:
        Causal mask (seq_len, seq_len) where upper triangle is -inf
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask


def create_padding_mask(seq_len: int, pad_token_id: int = 0) -> torch.Tensor:
    """
    Create padding mask from token IDs.

    Args:
        seq_len: Sequence length
        pad_token_id: ID of padding token

    Returns:
        Padding mask (seq_len, seq_len) where padding positions are -inf
    """
    # This would be used with actual token sequences containing padding
    # For now, return None (no padding)
    return None
