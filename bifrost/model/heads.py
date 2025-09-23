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

        # Clamp variance for stability
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)

        # Compute standard deviation
        std = torch.exp(0.5 * log_var)
        std = torch.clamp(std, min=1e-6)

        # Sample from Gaussian: mean + std * epsilon
        epsilon = torch.randn_like(mean)
        samples = mean + temperature * std * epsilon

        return samples


class TokenTypePredictor(nn.Module):
    """
    Predicts the next token's type among 7 classes.

    Classes follow the tokenizer mapping:
    0=PROPERTY, 1=ELEMENT, 2=COUNT, 3=SPACEGROUP, 4=WYCKOFF, 5=COORDINATE, 6=LATTICE.
    """

    def __init__(self, d_model: int):
        """
        Initialize token type predictor.

        Args:
            d_model: Model dimension
        """
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 7),
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
            Logits over 7 token types (batch_size, seq_len, 7)
        """
        logits = self.predictor(x)  # (batch_size, seq_len, 1)
        # Return raw logits; callers can apply softmax as needed
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

        # Optional mask: [7, vocab_size] booleans indicating which discrete tokens
        # are allowed for each predicted type id (0..6). If None, no masking.
        self.type_token_mask: Optional[torch.Tensor] = None

        # Loss weighting (defaults can be overridden by trainer)
        self.w_disc: float = 1.0
        self.w_cont: float = 1.0
        self.w_type: float = 1.0

    def set_type_token_mask(self, mask: torch.Tensor):
        """Set a [7, vocab_size] boolean mask for type-conditioned decoding."""
        self.type_token_mask = mask.to(next(self.parameters()).device)

    def set_loss_weights(
        self, w_disc: float = 1.0, w_cont: float = 1.0, w_type: float = 1.0
    ):
        """Configure loss term weights for combining total loss."""
        self.w_disc = float(w_disc)
        self.w_cont = float(w_cont)
        self.w_type = float(w_type)

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
        type_logits = self.token_type_predictor(x)

        return discrete_logits, continuous_params, type_logits

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
            Tuple of (token_value, predicted_type_id)
        """
        # Get predictions from all heads
        discrete_logits, continuous_params, type_logits = self.forward(x)

        # Get the last position predictions (autoregressive)
        discrete_logits = discrete_logits[:, -1]  # (batch_size, vocab_size)
        continuous_params = continuous_params[:, -1]  # (batch_size, 2)
        type_logits = type_logits[:, -1]  # (batch_size, 7)

        # Predict next token type (argmax)
        predicted_types = torch.argmax(torch.softmax(type_logits, dim=-1), dim=-1)
        is_discrete = predicted_types <= 4

        # Initialize output tensor as float to support continuous values
        batch_size = x.size(0)
        token_ids = torch.zeros(batch_size, dtype=torch.float, device=x.device)

        # Sample discrete tokens
        discrete_mask = is_discrete
        if discrete_mask.any():
            logits = discrete_logits[discrete_mask] / max(temperature, 1e-6)

            # Apply type-based vocabulary masking if available
            if self.type_token_mask is not None:
                t_ids = predicted_types[discrete_mask]
                masks = self.type_token_mask[t_ids]
                logits = logits.masked_fill(~masks, float("-inf"))

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

        return token_ids, predicted_types

    def predict_next_token_from_outputs(
        self,
        discrete_logits: torch.Tensor,
        continuous_params: torch.Tensor,
        type_logits: torch.Tensor,
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
            Tuple of (token_value, predicted_type_ids)
        """
        # Squeeze last time dimension
        logits = discrete_logits[:, -1]
        params = continuous_params[:, -1]
        tlogits = type_logits[:, -1]

        # Predict type ids
        predicted_types = torch.argmax(
            torch.softmax(tlogits, dim=-1), dim=-1
        )  # (batch,)
        is_discrete = predicted_types <= 4

        batch_size = logits.size(0)
        token_vals = torch.zeros(batch_size, dtype=torch.float, device=logits.device)

        # Discrete sampling
        if is_discrete.any():
            dlogits = logits[is_discrete] / max(temperature, 1e-6)

            # Apply type-based vocabulary masking if available
            if self.type_token_mask is not None:
                t_ids = predicted_types[is_discrete]
                masks = self.type_token_mask[t_ids]
                dlogits = dlogits.masked_fill(~masks, float("-inf"))
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
            log_var = torch.clamp(log_var, min=-10.0, max=10.0)
            std = torch.exp(0.5 * log_var)
            std = torch.clamp(std, min=1e-6)
            epsilon = torch.randn_like(mean)
            samples = mean + temperature * std * epsilon
            token_vals[~is_discrete] = samples.squeeze(-1)
        return token_vals, predicted_types

    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        target_tokens: torch.Tensor,
        target_types: torch.Tensor,
        continuous_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss for discrete and continuous predictions, plus token-type loss.

        Args:
            hidden_states: Model outputs (batch_size, seq_len, d_model)
            target_tokens: Target token IDs (batch_size, seq_len)
            target_types: Target token types (batch_size, seq_len)
            continuous_mask: Mask indicating continuous targets (batch_size, seq_len)
            attention_mask: Mask of valid (non-padding) positions (batch_size, seq_len)
            ignore_index: Index to ignore in loss computation

        Returns:
            Tuple of (total_loss, discrete_loss, continuous_loss, type_loss)
        """
        # Get predictions
        discrete_logits, continuous_params, type_probs = self.forward(hidden_states)

        # Build a validity mask to exclude padded positions from all loss terms
        # If no attention_mask provided, treat all positions as valid
        if attention_mask is not None:
            valid_positions = attention_mask.bool()
        else:
            valid_positions = torch.ones_like(target_tokens, dtype=torch.bool)

        # Compute discrete loss (cross-entropy) on valid, discrete positions only
        discrete_positions = (~continuous_mask) & valid_positions
        if discrete_positions.any():
            discrete_logits_flat = discrete_logits[discrete_positions]
            target_tokens_flat = target_tokens[discrete_positions].long()

            # Optional: also respect ignore_index if present in targets
            keep_mask = target_tokens_flat != ignore_index
            if keep_mask.any():
                discrete_loss = F.cross_entropy(
                    discrete_logits_flat[keep_mask],
                    target_tokens_flat[keep_mask],
                    reduction="mean",
                    label_smoothing=0.1,
                )
            else:
                discrete_loss = torch.tensor(0.0, device=hidden_states.device)
        else:
            discrete_loss = torch.tensor(0.0, device=hidden_states.device)

        # Compute continuous loss (Gaussian negative log likelihood) on valid, continuous positions only
        continuous_positions = continuous_mask & valid_positions
        if continuous_positions.any():
            continuous_params_flat = continuous_params[continuous_positions]
            target_tokens_flat = target_tokens[continuous_positions].float()

            mean, log_var = continuous_params_flat.split(1, dim=-1)

            # Clamp for stability
            log_var = torch.clamp(log_var, min=-10.0, max=10.0)

            # Gaussian NLL loss with variance floor
            var = torch.exp(log_var)
            var = torch.clamp(var, min=1e-6)
            log_likelihood = -0.5 * (
                torch.log(2 * torch.pi * var)
                + (target_tokens_flat.unsqueeze(-1) - mean) ** 2 / var
            )
            continuous_loss = -log_likelihood.mean()
        else:
            continuous_loss = torch.tensor(0.0, device=hidden_states.device)

        # Compute token type loss (cross-entropy over 7 classes) on valid positions only
        if valid_positions.any():
            type_logits_full = type_probs[valid_positions]  # (num_valid, 7)
            target_types_flat = target_types[valid_positions].long()  # (num_valid,)
            type_loss = F.cross_entropy(
                type_logits_full, target_types_flat, reduction="mean"
            )
        else:
            type_loss = torch.tensor(0.0, device=hidden_states.device)

        # Combine losses with weights
        total_loss = (
            self.w_disc * discrete_loss
            + self.w_cont * continuous_loss
            + self.w_type * type_loss
        )

        return total_loss, discrete_loss, continuous_loss, type_loss
