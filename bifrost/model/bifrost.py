"""
Main BIFROST model implementation.

This module contains the complete BIFROST model that combines embeddings,
transformer blocks, and output heads for crystal structure generation.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any
import math

from .embeddings import BIFROSTEmbedding, create_continuous_mask
from .transformer import BIFROSTTransformer
from .heads import BIFROSTHeads


class BIFROST(nn.Module):
    """
    Main BIFROST model for crystal structure generation with property conditioning.

    This model uses an autoregressive transformer architecture to generate
    crystal structures as sequences of tokens, with properties specified
    as discrete bins in a prefix.
    """

    def __init__(
        self,
        vocab_size: int = 1430,
        d_model: int = 512,
        n_heads: int = 16,
        n_layers: int = 16,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        num_token_types: int = 7,
    ):
        """
        Initialize BIFROST model.

        Args:
            vocab_size: Size of discrete token vocabulary
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Hidden dimension of feed-forward network
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            num_token_types: Number of token types
        """
        super().__init__()

        # Model components
        self.embeddings = BIFROSTEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            num_token_types=num_token_types,
            dropout=dropout,
        )

        self.transformer = BIFROSTTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        self.heads = BIFROSTHeads(d_model, vocab_size)

        # Store configuration
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_tokens: torch.Tensor,
        token_types: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through BIFROST.

        Args:
            input_tokens: Input token IDs (batch_size, seq_len)
            token_types: Token type indices (batch_size, seq_len)
            attention_mask: Optional attention mask (batch_size, seq_len)

        Returns:
            Tuple of (discrete_logits, continuous_params, type_probs)
        """
        # Create continuous mask from token types: 1 indicates continuous
        continuous_mask = token_types == 1

        # Generate embeddings
        embeddings = self.embeddings(input_tokens, token_types, continuous_mask)

        # Create causal mask for autoregressive generation
        seq_len = embeddings.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=embeddings.device), diagonal=1
        )
        causal_mask = causal_mask.float().masked_fill(causal_mask == 1, float("-inf"))

        # Pass through transformer
        hidden_states = self.transformer(embeddings, mask=causal_mask)

        # Generate predictions
        discrete_logits, continuous_params, type_probs = self.heads(hidden_states)

        return discrete_logits, continuous_params, type_probs

    def compute_loss(
        self,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        token_types: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute training loss.

        Args:
            input_tokens: Input token IDs (batch_size, seq_len)
            target_tokens: Target token IDs (batch_size, seq_len)
            token_types: Token type indices (batch_size, seq_len)
            attention_mask: Optional attention mask (batch_size, seq_len)

        Returns:
            Tuple of (total_loss, loss_components)
        """
        # Create continuous mask from token types
        continuous_mask = token_types == 1

        # Generate embeddings
        embeddings = self.embeddings(input_tokens, token_types, continuous_mask)

        # Create causal mask for autoregressive generation
        seq_len = embeddings.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=embeddings.device), diagonal=1
        )
        causal_mask = causal_mask.float().masked_fill(causal_mask == 1, float("-inf"))

        # Pass through transformer
        hidden_states = self.transformer(embeddings, mask=causal_mask)

        # Generate predictions
        discrete_logits, continuous_params, type_probs = self.heads(hidden_states)

        # Derive target types by shifting input token types by one position
        target_types = torch.zeros_like(token_types)
        target_types[:, :-1] = token_types[:, 1:]
        # Create continuous mask for targets from derived target types
        target_continuous_mask = target_types == 1

        # Compute loss using heads
        total_loss, discrete_loss, continuous_loss = self.heads.compute_loss(
            hidden_states, target_tokens, target_types.float(), target_continuous_mask
        )

        # Return loss and components
        loss_components = {
            "total": total_loss,
            "discrete": discrete_loss,
            "continuous": continuous_loss,
            "type_prediction": torch.tensor(0.0),  # Placeholder
        }

        return total_loss, loss_components

    def generate(
        self,
        prefix_tokens: torch.Tensor,
        prefix_types: torch.Tensor,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate token sequence autoregressively.

        Args:
            prefix_tokens: Initial token sequence (batch_size, prefix_len)
            prefix_types: Token types for prefix (batch_size, prefix_len)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            eos_token_id: End-of-sequence token ID

        Returns:
            Tuple of (generated_tokens, generated_types)
        """
        if eos_token_id is None:
            # Use EOS token from vocabulary (assumed to be vocab_size - 1)
            eos_token_id = self.vocab_size - 1

        batch_size = prefix_tokens.size(0)
        device = prefix_tokens.device

        # Initialize generated sequence with prefix
        # Tokens kept as float to support continuous values
        generated_tokens = prefix_tokens.clone().float()
        generated_types = prefix_types.clone()

        # Generate tokens one by one
        for _ in range(max_length - prefix_tokens.size(1)):

            # Forward pass on current sequence to get hidden states
            # (reuse internal forward pieces to avoid running heads twice)
            # Create continuous mask from token types
            cont_mask = generated_types == 1
            embeddings = self.embeddings(generated_tokens, generated_types, cont_mask)
            seq_len = embeddings.size(1)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=embeddings.device), diagonal=1
            )
            causal_mask = causal_mask.float().masked_fill(
                causal_mask == 1, float("-inf")
            )
            hidden_states = self.transformer(embeddings, mask=causal_mask)

            # Get head outputs
            discrete_logits, continuous_params, type_probs = self.heads(hidden_states)

            # Predict next token from outputs
            next_token, is_discrete = self.heads.predict_next_token_from_outputs(
                discrete_logits,
                continuous_params,
                type_probs,
                temperature,
                top_k,
                top_p,
            )

            # Convert to appropriate format (elementwise masks)
            # Ensure generated token types follow convention: 0 = discrete, 1 = continuous
            token_type = torch.zeros_like(is_discrete, dtype=torch.long)
            token_type[~is_discrete] = 1

            # Append to generated sequence
            generated_tokens = torch.cat(
                [generated_tokens, next_token.unsqueeze(1)], dim=1
            )
            generated_types = torch.cat(
                [generated_types, token_type.unsqueeze(1)], dim=1
            )

            # Check for EOS token
            if (next_token == eos_token_id).all():
                break

        return generated_tokens, generated_types

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size(self) -> Dict[str, int]:
        """Get model size information."""
        total_params = self.get_num_parameters()

        # Estimate memory usage (rough approximation)
        param_memory = total_params * 4  # 4 bytes per float32 parameter
        buffer_memory = sum(buf.numel() * 4 for buf in self.buffers())

        return {
            "total_parameters": total_params,
            "trainable_parameters": total_params,
            "parameter_memory_mb": param_memory / (1024 * 1024),
            "buffer_memory_mb": buffer_memory / (1024 * 1024),
            "total_memory_mb": (param_memory + buffer_memory) / (1024 * 1024),
        }


def create_bifrost_model(config: Dict[str, Any]) -> BIFROST:
    """
    Create BIFROST model from configuration dictionary.

    Args:
        config: Model configuration dictionary

    Returns:
        Configured BIFROST model
    """
    return BIFROST(
        vocab_size=config.get("vocab_size", 1430),
        d_model=config.get("d_model", 512),
        n_heads=config.get("n_heads", 16),
        n_layers=config.get("n_layers", 16),
        d_ff=config.get("d_ff", 2048),
        dropout=config.get("dropout", 0.1),
        max_seq_len=config.get("max_seq_len", 512),
        num_token_types=config.get("num_token_types", 7),
    )
