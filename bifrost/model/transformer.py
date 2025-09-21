"""
Transformer components for BIFROST model.

This module contains the transformer blocks and related components
used in the BIFROST architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


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
            query=x, key=x, value=x, attn_mask=mask, is_causal=is_causal
        )

        return self.dropout(attn_output)


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
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
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
        attn_output = self.attention(x, mask=mask, is_causal=False)
        x = self.attention_norm(x + self.dropout(attn_output))

        # Feed-forward network with residual connection and layer norm
        ffn_output = self.feed_forward(x)
        x = self.feed_forward_norm(x + self.dropout(ffn_output))

        return x


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
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
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
            x = block(x, mask=mask)

        # Apply final layer normalization
        x = self.final_norm(x)

        return x

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
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    if device is not None:
        mask = mask.to(device)

    # Convert to float and set upper triangle to -inf
    mask = mask.float().masked_fill(mask == 1, float("-inf"))
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
