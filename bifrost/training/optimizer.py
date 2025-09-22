"""
Optimizers and learning rate schedules for BIFROST training.

This module provides optimized training configurations including
learning rate schedules, weight decay, and mixed precision training.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, LinearLR
from typing import Dict, Any, Optional, Union, List
import math
import logging


class BIFROSTOptimizer:
    """
    Optimizer configuration for BIFROST training.

    This class handles the creation of optimizers and learning rate schedulers
    optimized for the BIFROST model architecture.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 10000,
        total_steps: int = 100000,
        scheduler_type: str = "one_cycle",
    ):
        """
        Initialize optimizer configuration.

        Args:
            model: BIFROST model
            learning_rate: Peak learning rate
            weight_decay: Weight decay factor
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            scheduler_type: Type of learning rate schedule
        """
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.scheduler_type = scheduler_type

        # Logger (configured by caller/CLI or trainer)
        self.setup_logging()

        # Create parameter groups with different weight decay
        self.param_groups = self._create_parameter_groups()

        # Create optimizer
        self.optimizer = self._create_optimizer()

        # Create learning rate scheduler
        self.scheduler = self._create_scheduler()

    def setup_logging(self):
        """Initialize module logger; configuration is handled by the application."""
        self.logger = logging.getLogger(__name__)

    def _create_parameter_groups(self) -> List[Dict[str, Any]]:
        """
        Create parameter groups with different weight decay settings.

        Returns:
            List of parameter group configurations
        """
        # Separate parameters that should have different weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if any(nd in name for nd in no_decay):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {
                "params": decay_params,
                "weight_decay": self.weight_decay,
                "lr": self.learning_rate,
            },
            {"params": no_decay_params, "weight_decay": 0.0, "lr": self.learning_rate},
        ]

        return param_groups

    def _create_optimizer(self) -> AdamW:
        """
        Create AdamW optimizer with appropriate settings.

        Returns:
            Configured AdamW optimizer
        """
        return AdamW(
            self.param_groups,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
            fused=True if torch.cuda.is_available() else False,
        )

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Create learning rate scheduler.

        Returns:
            Configured learning rate scheduler
        """
        if self.scheduler_type == "one_cycle":
            # Ensure valid total steps and warmup ratio even for short runs
            total_steps = max(2, int(self.total_steps))
            warmup_steps = int(self.warmup_steps)
            # Clamp warmup_steps to be within [1, total_steps - 1]
            warmup_steps = max(1, min(warmup_steps, total_steps - 1))
            warmup_pct = warmup_steps / float(total_steps)
            # Torch expects 0 < pct_start < 1
            warmup_pct = max(1.0 / total_steps, min(0.9, warmup_pct))

            self.logger.info(f"Total steps: {total_steps}")
            self.logger.info(f"Warmup steps: {warmup_steps}")
            self.logger.info(f"Warmup percentage: {warmup_pct}")

            return OneCycleLR(
                self.optimizer,
                max_lr=self.learning_rate,
                total_steps=total_steps,
                pct_start=warmup_pct,
                anneal_strategy="cos",
                div_factor=25.0,  # initial_lr = max_lr / div_factor
                final_div_factor=1.0,
            )

        elif self.scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer, T_max=self.total_steps, eta_min=self.learning_rate * 0.1
            )

        elif self.scheduler_type == "linear":
            return LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.warmup_steps,
            )

        else:
            # No scheduler
            return None

    def step(self):
        """Perform optimizer step and scheduler step."""
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()

    def get_current_lr(self) -> float:
        """
        Get current learning rate.

        Returns:
            Current learning rate
        """
        return self.optimizer.param_groups[0]["lr"]

    def get_lr_schedule(self) -> Dict[str, Any]:
        """
        Get learning rate schedule information.

        Returns:
            Dictionary with schedule information
        """
        return {
            "scheduler_type": self.scheduler_type,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "current_lr": self.get_current_lr(),
        }


class GradientScaler:
    """
    Gradient scaling for mixed precision training.

    This class handles automatic mixed precision training and gradient scaling
    to prevent underflow/overflow issues.
    """

    def __init__(self, enabled: bool = True, init_scale: float = 65536.0):
        """
        Initialize gradient scaler.

        Args:
            enabled: Whether to enable mixed precision training
            init_scale: Initial gradient scale
        """
        self.enabled = enabled and torch.cuda.is_available()
        self.scaler = (
            torch.cuda.amp.GradScaler(enabled=self.enabled, init_scale=init_scale)
            if self.enabled
            else None
        )

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss for mixed precision training.

        Args:
            loss: Loss tensor

        Returns:
            Scaled loss tensor
        """
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def step(self, optimizer: torch.optim.Optimizer):
        """
        Perform optimizer step with gradient scaling.

        Args:
            optimizer: Optimizer to step
        """
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def unscale_(self, optimizer: torch.optim.Optimizer):
        """
        Unscale gradients.

        Args:
            optimizer: Optimizer whose gradients to unscale
        """
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)


class OptimizerFactory:
    """
    Factory class for creating optimizers and schedulers.

    This provides a convenient interface for creating training configurations
    with sensible defaults for BIFROST.
    """

    @staticmethod
    def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> BIFROSTOptimizer:
        """
        Create BIFROST optimizer from configuration.

        Args:
            model: BIFROST model
            config: Optimizer configuration

        Returns:
            Configured BIFROST optimizer
        """
        return BIFROSTOptimizer(
            model=model,
            learning_rate=config.get("learning_rate", 2e-4),
            weight_decay=config.get("weight_decay", 0.01),
            warmup_steps=config.get("warmup_steps", 10000),
            total_steps=config.get("total_steps", 100000),
            scheduler_type=config.get("scheduler_type", "one_cycle"),
        )

    @staticmethod
    def create_gradient_scaler(config: Dict[str, Any]) -> GradientScaler:
        """
        Create gradient scaler from configuration.

        Args:
            config: Training configuration

        Returns:
            Configured gradient scaler
        """
        return GradientScaler(
            enabled=config.get("mixed_precision", True),
            init_scale=config.get("init_scale", 65536.0),
        )


# Default training configurations
TRAINING_CONFIGS = {
    "default": {
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "warmup_steps": 10000,
        "scheduler_type": "one_cycle",
        "mixed_precision": True,
        "gradient_clip": 1.0,
        "batch_size": 256,
    },
    "large_scale": {
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "warmup_steps": 20000,
        "scheduler_type": "one_cycle",
        "mixed_precision": True,
        "gradient_clip": 1.0,
        "batch_size": 128,
    },
    "debug": {
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "warmup_steps": 1000,
        "scheduler_type": "linear",
        "mixed_precision": False,
        "gradient_clip": 1.0,
        "batch_size": 32,
    },
}


def get_training_config(name: str = "default") -> Dict[str, Any]:
    """
    Get predefined training configuration.

    Args:
        name: Configuration name ('default', 'large_scale', 'debug')

    Returns:
        Training configuration dictionary
    """
    if name not in TRAINING_CONFIGS:
        raise ValueError(f"Unknown training config: {name}")

    return TRAINING_CONFIGS[name].copy()
