"""
Embedding layers for BIFROST model.

This module contains the embedding layers that handle both discrete tokens
and continuous values, along with positional encoding and token type embeddings.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models.

    This adds position information to token embeddings so the model can
    distinguish the position of tokens in the sequence.
    """

    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, won't be updated during training)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ContinuousValueEncoder(nn.Module):
    """
    Encoder for continuous values in the token sequence.

    This module transforms continuous values (coordinates, lattice parameters)
    into the model's embedding space.
    """

    def __init__(self, d_model: int, hidden_dim: int = 64):
        """
        Initialize continuous value encoder.

        Args:
            d_model: Model dimension
            hidden_dim: Hidden dimension for the encoder network
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, d_model)
        )

        # Initialize weights
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode continuous values.

        Args:
            x: Tensor of shape (batch_size, seq_len) or (batch_size, seq_len, 1)

        Returns:
            Encoded tensor of shape (batch_size, seq_len, d_model)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # Add feature dimension

        return self.encoder(x)


class TokenTypeEmbedding(nn.Module):
    """
    Embedding layer for token types.

    This helps the model distinguish between different types of tokens
    (properties, elements, coordinates, etc.).
    """

    def __init__(self, num_types: int, d_model: int):
        """
        Initialize token type embeddings.

        Args:
            num_types: Number of token types
            d_model: Model dimension
        """
        super().__init__()
        self.embedding = nn.Embedding(num_types, d_model)

        # Initialize with small values
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, token_types: torch.Tensor) -> torch.Tensor:
        """
        Get token type embeddings.

        Args:
            token_types: Tensor of token type indices (batch_size, seq_len)

        Returns:
            Token type embeddings (batch_size, seq_len, d_model)
        """
        return self.embedding(token_types)


class BIFROSTEmbedding(nn.Module):
    """
    Main embedding layer for BIFROST that handles both discrete and continuous tokens.

    This layer combines:
    - Discrete token embeddings
    - Continuous value encoding
    - Positional encoding
    - Token type embeddings
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        max_seq_len: int = 512,
        num_token_types: int = 7,
        continuous_hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        """
        Initialize BIFROST embedding layer.

        Args:
            vocab_size: Size of discrete token vocabulary
            d_model: Model dimension
            max_seq_len: Maximum sequence length
            num_token_types: Number of token types
            continuous_hidden_dim: Hidden dimension for continuous encoder
            dropout: Dropout probability
        """
        super().__init__()

        # Discrete token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.token_embedding.weight, std=0.02)

        # Continuous value encoder
        self.continuous_encoder = ContinuousValueEncoder(d_model, continuous_hidden_dim)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Token type embeddings
        self.token_type_embedding = TokenTypeEmbedding(num_token_types, d_model)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Store model dimension
        self.d_model = d_model

    def forward(
        self,
        token_ids: torch.Tensor,
        token_types: torch.Tensor,
        continuous_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through embedding layer.

        Args:
            token_ids: Discrete token IDs (batch_size, seq_len)
            token_types: Token type indices (batch_size, seq_len)
            continuous_mask: Mask indicating continuous tokens (batch_size, seq_len)

        Returns:
            Embedded tokens (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = token_ids.size()

        # Prepare discrete token ids by masking out continuous positions to a safe index (0)
        # Ensure dtype is long for embedding lookup
        discrete_token_ids = token_ids
        if discrete_token_ids.dtype != torch.long:
            # Cast only after masking to avoid large/invalid indices from floats
            discrete_token_ids = discrete_token_ids.clone()
        discrete_token_ids = discrete_token_ids.masked_fill(continuous_mask, 0).long()

        # Create embeddings for discrete tokens (safe lookup)
        discrete_embeddings = self.token_embedding(discrete_token_ids)

        # Create embeddings for continuous tokens using original numeric values
        continuous_embeddings = self.continuous_encoder(token_ids.float())

        # Combine based on mask
        embeddings = torch.where(
            continuous_mask.unsqueeze(-1), continuous_embeddings, discrete_embeddings
        )

        # Add token type embeddings
        type_embeddings = self.token_type_embedding(token_types)
        embeddings = embeddings + type_embeddings

        # Add positional encoding
        embeddings = self.positional_encoding(embeddings)

        # Apply layer normalization and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


def create_continuous_mask(token_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Create mask indicating which tokens are continuous values.

    Args:
        token_ids: Token IDs (batch_size, seq_len)
        vocab_size: Size of discrete vocabulary

    Returns:
        Mask tensor (batch_size, seq_len) where True indicates continuous tokens
    """
    # Continuous tokens are represented as values >= vocab_size
    return token_ids >= vocab_size


def get_token_type_id(token_type: str, type_mapping: dict) -> int:
    """
    Get token type ID from string.

    Args:
        token_type: String token type
        type_mapping: Dictionary mapping type names to IDs

    Returns:
        Token type ID
    """
    return type_mapping.get(token_type, 0)  # Default to 0 if not found
