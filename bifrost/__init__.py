"""
BIFROST: Crystal Structure Generation with Property Conditioning

BIFROST is an autoregressive transformer model for generating 3D crystal structures
conditioned on target material properties. It uses a sequence-based representation
of crystal structures and learns to generate chemically valid, stable structures.

Main Components:
- model: Core BIFROST model architecture
- data: Data pipeline for tokenization and dataset handling
- training: Training loop with curriculum learning and optimization
- generation: Structure generation and inference
- evaluation: Model evaluation and benchmarking tools
- utils: Utility functions for chemistry and crystallography

Example usage:
    from bifrost.model import BIFROST
    from bifrost.config import create_model_config

    # Create model
    config = create_model_config('base')
    model = BIFROST(**config)

    # Set up training data and trainer
    # ... (see example.py for full example)
"""

__version__ = "1.0.0"
__author__ = "BIFROST Team"

# Import main components for easy access
from .model import BIFROST, create_bifrost_model
from .config import (
    create_model_config,
    create_training_config,
    create_generation_config,
    example_training_setup,
    example_generation_setup,
)

__all__ = [
    "BIFROST",
    "create_bifrost_model",
    "create_model_config",
    "create_training_config",
    "create_generation_config",
    "example_training_setup",
    "example_generation_setup",
]
