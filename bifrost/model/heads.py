"""
Output heads for BIFROST model.

This module contains the output heads for predicting both discrete tokens
and continuous values, along with token type prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DiscreteHead(nn.Module):
    """
    Output head for predicting discrete tokens.

    This head takes transformer outputs and predicts probabilities
    over the discrete token vocabulary.
    """

    def __init__(self, d_model: int, vocab_size: int):
        """
        Initialize discrete output head.

        Args:
            d_model: Model dimension
            vocab_size: Size of discrete token vocabulary
        """
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )

        # Initialize weights
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through discrete head.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            Logits over vocabulary (batch_size, seq_len, vocab_size)
        """
        return self.head(x)


class ContinuousHead(nn.Module):
    """
    Output head for predicting continuous values.

    This head predicts both mean and log variance for a Gaussian
    distribution over continuous values.
    """

    def __init__(self, d_model: int, output_dim: int = 2):
        """
        Initialize continuous output head.

        Args:
            d_model: Model dimension
            output_dim: Output dimension (2 for mean + log_var)
        """
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),  # mean, log_var
        )

        # Initialize weights
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through continuous head.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            Parameters (batch_size, seq_len, output_dim)
        """
        return self.head(x)

    def sample(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Sample from Gaussian distribution defined by head outputs.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            temperature: Sampling temperature

        Returns:
            Sampled values (batch_size, seq_len, 1)
        """
        params = self.forward(x)  # (batch_size, seq_len, 2)
        mean, log_var = params.split(1, dim=-1)  # Split into mean and log_var

        # Compute standard deviation
        std = torch.exp(0.5 * log_var)

        # Sample from Gaussian: mean + std * epsilon
        epsilon = torch.randn_like(mean)
        samples = mean + temperature * std * epsilon

        return samples


class TokenTypePredictor(nn.Module):
    """
    Predicts whether next token should be discrete or continuous.

    This is a binary classifier that helps the model decide whether
    to use the discrete head or continuous head for the next token.
    """

    def __init__(self, d_model: int):
        """
        Initialize token type predictor.

        Args:
            d_model: Model dimension
        """
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1)
        )

        # Initialize weights
        for layer in self.predictor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through token type predictor.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            Probabilities of being discrete token (batch_size, seq_len, 1)
        """
        logits = self.predictor(x)  # (batch_size, seq_len, 1)
        # Return raw logits; callers can apply sigmoid as needed
        return logits


class BIFROSTHeads(nn.Module):
    """
    Combined output heads for BIFROST.

    This module combines all output heads and provides a unified
    interface for making predictions.
    """

    def __init__(self, d_model: int, vocab_size: int):
        """
        Initialize BIFROST output heads.

        Args:
            d_model: Model dimension
            vocab_size: Size of discrete token vocabulary
        """
        super().__init__()

        # Output heads
        self.discrete_head = DiscreteHead(d_model, vocab_size)
        self.continuous_head = ContinuousHead(d_model)
        self.token_type_predictor = TokenTypePredictor(d_model)

        # Store vocab size for convenience
        self.vocab_size = vocab_size
        self.d_model = d_model

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through all heads.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            Tuple of (discrete_logits, continuous_params, type_probs)
        """
        discrete_logits = self.discrete_head(x)
        continuous_params = self.continuous_head(x)
        type_probs = self.token_type_predictor(x)

        return discrete_logits, continuous_params, type_probs

    def predict_next_token(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next token (discrete or continuous) given input.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter

        Returns:
            Tuple of (token_id, is_discrete) where token_id is the predicted token
        """
        # Get predictions from all heads
        discrete_logits, continuous_params, type_probs = self.forward(x)

        # Get the last position predictions (autoregressive)
        discrete_logits = discrete_logits[:, -1]  # (batch_size, vocab_size)
        continuous_params = continuous_params[:, -1]  # (batch_size, 2)
        type_probs = type_probs[:, -1]  # (batch_size, 1)

        # Decide whether to predict discrete or continuous token
        # Convention: target type 1 = continuous, 0 = discrete
        # Therefore, probability > 0.5 => continuous; else discrete
        is_continuous = torch.sigmoid(type_probs).squeeze(-1) > 0.5  # (batch_size,)
        is_discrete = ~is_continuous

        # Initialize output tensor as float to support continuous values
        batch_size = x.size(0)
        token_ids = torch.zeros(batch_size, dtype=torch.float, device=x.device)

        # Sample discrete tokens
        discrete_mask = is_discrete
        if discrete_mask.any():
            logits = discrete_logits[discrete_mask] / temperature

            # Apply top-k filtering if specified
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Apply top-p (nucleus) filtering if specified
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample from filtered distribution
            probs = F.softmax(logits, dim=-1)
            sampled_indices = torch.multinomial(probs, 1).squeeze(-1)
            token_ids[discrete_mask] = sampled_indices.float()

        # Sample continuous tokens
        continuous_mask = ~is_discrete
        if continuous_mask.any():
            mean, log_var = continuous_params[continuous_mask].split(1, dim=-1)
            std = torch.exp(0.5 * log_var)

            # Sample from Gaussian
            epsilon = torch.randn_like(mean)
            sampled_values = mean + temperature * std * epsilon
            token_ids[continuous_mask] = sampled_values.squeeze(-1)

        return token_ids, is_discrete

    def predict_next_token_from_outputs(
        self,
        discrete_logits: torch.Tensor,
        continuous_params: torch.Tensor,
        type_probs: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next token using already computed head outputs for last position.

        Args:
            discrete_logits: (batch_size, 1, vocab_size)
            continuous_params: (batch_size, 1, 2)
            type_probs: (batch_size, 1, 1)
        Returns:
            Tuple of (token_value, is_discrete_mask)
        """
        # Squeeze last time dimension
        logits = discrete_logits[:, -1]
        params = continuous_params[:, -1]
        tprobs = type_probs[:, -1]

        # Convention: 1 = continuous target
        is_continuous = torch.sigmoid(tprobs).squeeze(-1) > 0.5  # (batch,)
        is_discrete = ~is_continuous

        batch_size = logits.size(0)
        token_vals = torch.zeros(batch_size, dtype=torch.float, device=logits.device)

        # Discrete sampling
        if is_discrete.any():
            dlogits = logits[is_discrete] / max(temperature, 1e-6)
            if top_k is not None:
                thresh = torch.topk(dlogits, min(top_k, dlogits.size(-1)))[0][
                    ..., -1, None
                ]
                dlogits = dlogits.masked_fill(dlogits < thresh, float("-inf"))
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(dlogits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1, index=sorted_indices, src=sorted_indices_to_remove
                )
                dlogits = dlogits.masked_fill(indices_to_remove, float("-inf"))
            probs = torch.softmax(dlogits, dim=-1)
            sampled = torch.multinomial(probs, 1).squeeze(-1)
            token_vals[is_discrete] = sampled.float()

        # Continuous sampling
        if (~is_discrete).any():
            mean, log_var = params[~is_discrete].split(1, dim=-1)
            std = torch.exp(0.5 * log_var)
            epsilon = torch.randn_like(mean)
            samples = mean + temperature * std * epsilon
            token_vals[~is_discrete] = samples.squeeze(-1)

        return token_vals, is_discrete

    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        target_tokens: torch.Tensor,
        target_types: torch.Tensor,
        continuous_mask: torch.Tensor,
        ignore_index: int = -100,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss for discrete and continuous predictions.

        Args:
            hidden_states: Model outputs (batch_size, seq_len, d_model)
            target_tokens: Target token IDs (batch_size, seq_len)
            target_types: Target token types (batch_size, seq_len)
            continuous_mask: Mask indicating continuous targets (batch_size, seq_len)
            ignore_index: Index to ignore in loss computation

        Returns:
            Tuple of (total_loss, discrete_loss, continuous_loss)
        """
        # Get predictions
        discrete_logits, continuous_params, type_probs = self.forward(hidden_states)

        # Compute discrete loss (cross-entropy)
        discrete_mask = ~continuous_mask
        if discrete_mask.any():
            discrete_logits_flat = discrete_logits[discrete_mask]
            target_tokens_flat = target_tokens[discrete_mask].long()

            # Ignore padding tokens
            valid_mask = target_tokens_flat != ignore_index
            if valid_mask.any():
                discrete_loss = F.cross_entropy(
                    discrete_logits_flat[valid_mask],
                    target_tokens_flat[valid_mask],
                    reduction="mean",
                )
            else:
                discrete_loss = torch.tensor(0.0, device=hidden_states.device)
        else:
            discrete_loss = torch.tensor(0.0, device=hidden_states.device)

        # Compute continuous loss (Gaussian negative log likelihood)
        if continuous_mask.any():
            continuous_params_flat = continuous_params[continuous_mask]
            target_tokens_flat = target_tokens[continuous_mask].float()

            mean, log_var = continuous_params_flat.split(1, dim=-1)

            # Gaussian NLL loss
            var = torch.exp(log_var)
            log_likelihood = -0.5 * (
                torch.log(2 * torch.pi * var)
                + (target_tokens_flat.unsqueeze(-1) - mean) ** 2 / var
            )
            continuous_loss = -log_likelihood.mean()
        else:
            continuous_loss = torch.tensor(0.0, device=hidden_states.device)

        # Compute token type loss (binary cross-entropy)
        target_type_probs = target_types.float()
        type_loss = F.binary_cross_entropy_with_logits(
            type_probs.squeeze(-1), target_type_probs, reduction="mean"
        )

        # Combine losses
        total_loss = discrete_loss + continuous_loss + type_loss

        return total_loss, discrete_loss, continuous_loss
