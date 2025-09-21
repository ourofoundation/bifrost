"""
BIFROST training components.

This package contains all the components needed for training
the BIFROST crystal structure generation model.
"""

from .curriculum import CurriculumScheduler, CurriculumManager, CurriculumDataset
from .optimizer import (
    BIFROSTOptimizer,
    GradientScaler,
    OptimizerFactory,
    get_training_config,
    TRAINING_CONFIGS,
)
from .train import BIFROSTTrainer, TrainingStats, create_trainer

__all__ = [
    "CurriculumScheduler",
    "CurriculumManager",
    "CurriculumDataset",
    "BIFROSTOptimizer",
    "GradientScaler",
    "OptimizerFactory",
    "get_training_config",
    "TRAINING_CONFIGS",
    "BIFROSTTrainer",
    "TrainingStats",
    "create_trainer",
]
