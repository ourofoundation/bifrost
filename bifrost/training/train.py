"""
Main training loop for BIFROST model.

This module provides the main training loop with support for curriculum learning,
mixed precision training, gradient clipping, and comprehensive logging.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import logging
from typing import Dict, Any, Optional, Tuple, List, Callable
from pathlib import Path
import json
from torch.utils.tensorboard import SummaryWriter

from ..model import BIFROST
from .curriculum import CurriculumManager
from .optimizer import BIFROSTOptimizer, GradientScaler, OptimizerFactory


class TrainingStats:
    """
    Tracks training statistics and metrics.

    This class maintains running statistics for loss components,
    learning rates, and other training metrics.
    """

    def __init__(self):
        """Initialize training statistics."""
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.epoch = 0
        self.step = 0
        self.loss_history = []
        self.lr_history = []
        self.component_history = {
            "discrete_loss": [],
            "continuous_loss": [],
            "type_prediction_loss": [],
            "total_loss": [],
        }
        self.epoch_times = []
        self.batch_times = []

    def update(
        self, loss: float, components: Dict[str, float], lr: float, batch_time: float
    ):
        """
        Update statistics with new batch information.

        Args:
            loss: Total loss value
            components: Dictionary of loss components
            lr: Current learning rate
            batch_time: Time taken for batch
        """
        self.step += 1
        self.loss_history.append(loss)
        self.lr_history.append(lr)

        for key, value in components.items():
            if key in self.component_history:
                self.component_history[key].append(value)

        self.batch_times.append(batch_time)

    def update_epoch(self):
        """Update epoch-level statistics."""
        self.epoch += 1
        self.epoch_times.append(sum(self.batch_times))
        self.batch_times.clear()

    def get_epoch_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the current epoch.

        Returns:
            Dictionary with epoch summary
        """
        if not self.loss_history:
            return {}

        return {
            "epoch": self.epoch,
            "avg_loss": sum(self.loss_history) / len(self.loss_history),
            "min_loss": min(self.loss_history),
            "max_loss": max(self.loss_history),
            "avg_lr": sum(self.lr_history) / len(self.lr_history),
            "component_avgs": {
                key: sum(values) / len(values) if values else 0.0
                for key, values in self.component_history.items()
            },
        }

    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get overall training summary.

        Returns:
            Dictionary with training summary
        """
        summary = {
            "total_steps": self.step,
            "total_epochs": self.epoch,
            "overall_stats": {},
        }

        if self.loss_history:
            summary["overall_stats"] = {
                "avg_loss": sum(self.loss_history) / len(self.loss_history),
                "min_loss": min(self.loss_history),
                "max_loss": max(self.loss_history),
                "final_loss": self.loss_history[-1],
            }

        return summary


class BIFROSTTrainer:
    """
    Main trainer class for BIFROST model.

    This class handles the complete training loop including curriculum learning,
    mixed precision training, checkpointing, and evaluation.
    """

    def __init__(
        self,
        model: BIFROST,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        config: Dict[str, Any] = None,
    ):
        """
        Initialize BIFROST trainer.

        Args:
            model: BIFROST model to train
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            config: Training configuration
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Set up configuration
        self.config = config or {}
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Move model to device
        self.model.to(self.device)

        # Configure loss weights if provided
        w_disc = self.config.get("w_disc", 1.0)
        w_cont = self.config.get("w_cont", 1.0)
        w_type = self.config.get("w_type", 0.2)
        try:
            self.model.heads.set_loss_weights(
                w_disc=w_disc, w_cont=w_cont, w_type=w_type
            )
        except Exception:
            pass

        # Set up training components
        self.optimizer = OptimizerFactory.create_optimizer(model, self.config)
        self.gradient_scaler = OptimizerFactory.create_gradient_scaler(self.config)
        self.curriculum_manager = CurriculumManager(
            train_dataloader.dataset,
            enable_curriculum=self.config.get("enable_curriculum", True),
        )

        # Training statistics
        self.stats = TrainingStats()

        # Starting epoch offset (set when resuming from checkpoint)
        self.start_epoch = 0

        # Logging setup
        self.setup_logging()

        # TensorBoard setup
        self.tb_writer: Optional[SummaryWriter] = None
        self.tb_log_dir: Optional[Path] = None
        if self.config.get("tensorboard", False):
            base_log_dir = Path(self.config.get("tensorboard_log_dir", "runs/"))
            base_log_dir.mkdir(parents=True, exist_ok=True)
            # Generate or use provided run name
            provided_run_name = self.config.get("run_name")
            if provided_run_name is None:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                # Create a compact model size identifier based on parameter count
                model_size_info = self.model.get_model_size()
                total_params = model_size_info["total_parameters"]
                if total_params < 8_000_000:
                    size_suffix = "small"
                elif total_params < 150_000_000:
                    size_suffix = "base"
                else:
                    size_suffix = "large"
                provided_run_name = f"{timestamp}-{size_suffix}"
            self.tb_log_dir = base_log_dir / provided_run_name
            self.tb_log_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(self.tb_log_dir))
            # Persist hyperparameters snapshot early for convenience
            try:
                hparams_snapshot = {
                    "run_name": provided_run_name,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "device": str(self.device),
                    "learning_rate": self.config.get("learning_rate"),
                    "weight_decay": self.config.get("weight_decay"),
                    "warmup_steps": self.config.get("warmup_steps"),
                    "scheduler_type": str(self.config.get("scheduler_type")),
                    "mixed_precision": bool(self.config.get("mixed_precision", True)),
                    "gradient_clip": self.config.get("gradient_clip"),
                    "batch_size": self.config.get("batch_size"),
                    "enable_curriculum": bool(
                        self.config.get("enable_curriculum", False)
                    ),
                    "train_dataset_size": (
                        len(self.train_dataloader.dataset)
                        if hasattr(self.train_dataloader, "dataset")
                        else None
                    ),
                    "val_dataset_size": (
                        len(self.val_dataloader.dataset)
                        if (
                            self.val_dataloader is not None
                            and hasattr(self.val_dataloader, "dataset")
                        )
                        else None
                    ),
                    "model_total_parameters": self.model.get_model_size().get(
                        "total_parameters"
                    ),
                }
                with open(self.tb_log_dir / "hparams.json", "w") as f:
                    json.dump(hparams_snapshot, f, indent=2)
                self.logger.info(f"TensorBoard logging to {self.tb_log_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to write hparams.json: {e}")

        # Checkpointing
        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.logger.info(f"BIFROST Trainer initialized on {self.device}")
        self.logger.info(f"Model parameters: {model.get_num_parameters():,}")

    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform single training step.

        Args:
            batch: Batch of training data

        Returns:
            Dictionary with step results
        """
        start_time = time.time()

        # Move batch to device
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass with mixed precision
        with torch.amp.autocast("cuda", enabled=self.gradient_scaler.enabled):
            loss, loss_components = self.model.compute_loss(
                batch["input_tokens"],
                batch["target_tokens"],
                batch["token_types"],
                batch["attention_mask"],
            )

        # Scale loss and compute gradients
        scaled_loss = self.gradient_scaler.scale(loss)
        scaled_loss.backward()

        # Unscale gradients for clipping
        self.gradient_scaler.unscale_(self.optimizer.optimizer)

        # Gradient clipping
        gradient_clip = self.config.get("gradient_clip", 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)

        # Optimizer step
        self.gradient_scaler.step(self.optimizer.optimizer)
        self.optimizer.step()

        # Update statistics
        batch_time = time.time() - start_time
        lr = self.optimizer.get_current_lr()

        self.stats.update(loss.item(), loss_components, lr, batch_time)

        # TensorBoard step logging
        if self.tb_writer is not None:
            global_step = self.stats.step
            self.tb_writer.add_scalar("train/loss", loss.item(), global_step)
            self.tb_writer.add_scalar("train/lr", lr, global_step)
            self.tb_writer.add_scalar("train/batch_time", batch_time, global_step)
            for comp_key, comp_val in loss_components.items():
                self.tb_writer.add_scalar(f"train/{comp_key}", comp_val, global_step)

        return {
            "loss": loss.item(),
            "lr": lr,
            "batch_time": batch_time,
            **loss_components,
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with epoch results
        """
        self.model.train()
        self.curriculum_manager.step_epoch()

        epoch_start_time = time.time()
        self.logger.info(f"Starting epoch {epoch + 1}")

        for batch_idx, batch in enumerate(self.train_dataloader):
            # Training step
            step_results = self.train_step(batch)

            # Log progress
            if batch_idx % self.config.get("log_interval", 100) == 0:
                self.logger.info(
                    f"Epoch {epoch + 1}, Batch {batch_idx}, "
                    f"Loss: {step_results['loss']:.4f}, "
                    f"LR: {step_results['lr']:.2e}"
                )

        # Update epoch statistics
        self.stats.update_epoch()
        epoch_time = time.time() - epoch_start_time

        # Get epoch summary
        epoch_summary = self.stats.get_epoch_summary()
        epoch_summary["epoch_time"] = epoch_time
        epoch_summary["curriculum_info"] = self.curriculum_manager.get_current_info()

        self.logger.info(
            f"Epoch {epoch + 1} completed. "
            f"Avg Loss: {epoch_summary['avg_loss']:.4f}, "
            f"Time: {epoch_time:.2f}s"
        )

        # TensorBoard epoch logging
        if self.tb_writer is not None and epoch_summary:
            epoch_index = epoch + 1
            self.tb_writer.add_scalar(
                "epoch/avg_loss", epoch_summary.get("avg_loss", 0.0), epoch_index
            )
            self.tb_writer.add_scalar(
                "epoch/min_loss", epoch_summary.get("min_loss", 0.0), epoch_index
            )
            self.tb_writer.add_scalar(
                "epoch/max_loss", epoch_summary.get("max_loss", 0.0), epoch_index
            )
            self.tb_writer.add_scalar(
                "epoch/avg_lr", epoch_summary.get("avg_lr", 0.0), epoch_index
            )
            component_avgs = epoch_summary.get("component_avgs", {})
            for comp_key, comp_val in component_avgs.items():
                self.tb_writer.add_scalar(f"epoch/{comp_key}", comp_val, epoch_index)

        return epoch_summary

    def validate(self) -> Dict[str, float]:
        """
        Run validation on validation set.

        Returns:
            Dictionary with validation metrics
        """
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                # Move batch to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Forward pass
                with torch.amp.autocast("cuda", enabled=self.gradient_scaler.enabled):
                    loss, _ = self.model.compute_loss(
                        batch["input_tokens"],
                        batch["target_tokens"],
                        batch["token_types"],
                        batch["attention_mask"],
                    )

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        self.logger.info(f"Validation Loss: {avg_loss:.4f}")
        # TensorBoard validation logging
        if self.tb_writer is not None:
            # Use current epoch as x-axis if available, else global step
            step_or_epoch = (
                self.stats.epoch if self.stats.epoch > 0 else self.stats.step
            )
            self.tb_writer.add_scalar("val/loss", avg_loss, step_or_epoch)
        return {"val_loss": avg_loss}

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            metrics: Training metrics
        """
        # Get model size information for filename
        model_size_info = self.model.get_model_size()
        total_params = model_size_info["total_parameters"]

        # Create a compact model size identifier based on parameter count
        if total_params < 8_000_000:  # Less than 50M parameters
            size_suffix = "small"
        elif total_params < 150_000_000:  # Less than 150M parameters
            size_suffix = "base"
        else:  # 150M+ parameters
            size_suffix = "large"

        checkpoint_path = (
            self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}_{size_suffix}.pt"
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.optimizer.scheduler.state_dict()
                if self.optimizer.scheduler
                else None
            ),
            "scaler_state_dict": (
                self.gradient_scaler.scaler.state_dict()
                if self.gradient_scaler.scaler
                else None
            ),
            "stats": self.stats.get_training_summary(),
            "metrics": metrics,
            "config": self.config,
        }

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Epoch number from checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if checkpoint["scheduler_state_dict"] and self.optimizer.scheduler:
            self.optimizer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if checkpoint["scaler_state_dict"] and self.gradient_scaler.scaler:
            self.gradient_scaler.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Restore statistics
        self.stats = TrainingStats()
        self.stats.loss_history = checkpoint.get("stats", {}).get("loss_history", [])
        self.stats.lr_history = checkpoint.get("stats", {}).get("lr_history", [])

        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        # Continue training from the next epoch after the checkpoint
        self.start_epoch = checkpoint.get("epoch", 0)
        return self.start_epoch

    def train(
        self, num_epochs: int, save_interval: int = 10, eval_interval: int = 1
    ) -> Dict[str, Any]:
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs to train
            save_interval: Save checkpoint every N epochs
            eval_interval: Run evaluation every N epochs

        Returns:
            Dictionary with training results
        """
        self.logger.info("Starting training...")

        training_results = {
            "epoch_summaries": [],
            "validation_results": [],
            "final_stats": None,
        }

        try:
            for relative_epoch in range(num_epochs):
                # Compute global epoch index so numbering continues when resuming
                epoch = self.start_epoch + relative_epoch
                # Train for one epoch
                epoch_summary = self.train_epoch(epoch)

                # Add to results
                training_results["epoch_summaries"].append(epoch_summary)

                # Run validation
                if (epoch + 1) % eval_interval == 0:
                    val_results = self.validate()
                    training_results["validation_results"].append(
                        {"epoch": epoch + 1, **val_results}
                    )

                # Save checkpoint
                if (epoch + 1) % save_interval == 0:
                    self.save_checkpoint(epoch, epoch_summary)

            # Final validation
            final_val_results = self.validate()
            training_results["final_validation"] = final_val_results

            # Final statistics
            training_results["final_stats"] = self.stats.get_training_summary()

            self.logger.info("Training completed successfully!")

            # TensorBoard: log hparams and close
            if self.tb_writer is not None:
                hparams = {
                    "learning_rate": self.config.get("learning_rate"),
                    "weight_decay": self.config.get("weight_decay"),
                    "warmup_steps": self.config.get("warmup_steps"),
                    "scheduler_type": str(self.config.get("scheduler_type")),
                    "mixed_precision": bool(self.config.get("mixed_precision", True)),
                    "gradient_clip": self.config.get("gradient_clip"),
                    "batch_size": self.config.get("batch_size"),
                    "enable_curriculum": bool(
                        self.config.get("enable_curriculum", False)
                    ),
                }
                final_metrics = {}
                overall = training_results.get("final_stats", {}).get(
                    "overall_stats", {}
                )
                if overall:
                    final_metrics = {
                        "hparam/final_loss": overall.get("final_loss", 0.0),
                        "hparam/avg_loss": overall.get("avg_loss", 0.0),
                    }
                self.tb_writer.add_hparams(
                    hparam_dict=hparams, metric_dict=final_metrics
                )
                self.tb_writer.flush()
                self.tb_writer.close()

        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            self.save_checkpoint(epoch, self.stats.get_epoch_summary())

        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            self.save_checkpoint(epoch, self.stats.get_epoch_summary())
            raise

        return training_results

    def get_training_info(self) -> Dict[str, Any]:
        """
        Get current training information.

        Returns:
            Dictionary with training information
        """
        return {
            "epoch": self.stats.epoch,
            "step": self.stats.step,
            "current_lr": self.optimizer.get_current_lr(),
            "curriculum_info": self.curriculum_manager.get_current_info(),
            "device": str(self.device),
            "model_size": self.model.get_model_size(),
        }


def create_trainer(
    model: BIFROST,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    config: Optional[Dict[str, Any]] = None,
) -> BIFROSTTrainer:
    """
    Create BIFROST trainer with default configuration.

    Args:
        model: BIFROST model
        train_dataloader: Training data loader
        val_dataloader: Optional validation data loader
        config: Optional training configuration

    Returns:
        Configured BIFROST trainer
    """
    if config is None:
        config = {
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "warmup_steps": 10000,
            "scheduler_type": "one_cycle",
            "mixed_precision": True,
            "gradient_clip": 1.0,
            "log_interval": 100,
            "enable_curriculum": True,
            "checkpoint_dir": "checkpoints",
        }

    return BIFROSTTrainer(model, train_dataloader, val_dataloader, config)
