"""
Configuration and example usage for BIFROST.

This module provides configuration templates and example code
for using the BIFROST crystal structure generation model.
"""

from typing import Dict, Any, Optional
from pathlib import Path


# Default configurations
DEFAULT_MODEL_CONFIG = {
    # "vocab_size": 1430,
    "d_model": 512,
    "n_heads": 16,
    "n_layers": 16,
    "d_ff": 2048,
    "dropout": 0.1,
    "max_seq_len": 512,
    "num_token_types": 7,
}

DEFAULT_TRAINING_CONFIG = {
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "warmup_steps": 10000,
    "scheduler_type": "one_cycle",
    "mixed_precision": True,
    "gradient_clip": 1.0,
    "batch_size": 256,
    "log_interval": 100,
    "enable_curriculum": False,
    "checkpoint_dir": "checkpoints",
    # TensorBoard
    "tensorboard": False,
    "tensorboard_log_dir": "runs",
}

DEFAULT_GENERATION_CONFIG = {
    "temperature": 0.8,
    "top_k": None,
    "top_p": None,
    "max_length": 512,
    "eos_token_id": None,  # Will be set to vocab_size - 1
}

# Model size presets
MODEL_PRESETS = {
    "small": {
        "d_model": 256,
        "n_heads": 8,
        "n_layers": 8,
        "d_ff": 1024,
        "dropout": 0.1,
        "max_seq_len": 128,
    },
    "base": {
        "d_model": 256,
        "n_heads": 8,
        "n_layers": 16,
        "d_ff": 2048,
        "dropout": 0.1,
        "max_seq_len": 256,
    },
    "large": {
        "d_model": 768,
        "n_heads": 16,
        "n_layers": 24,
        "d_ff": 3072,
        "dropout": 0.1,
        "max_seq_len": 512,
    },
}

# Training configurations
TRAINING_PRESETS = {
    "debug": {
        "learning_rate": 1e-3,
        "batch_size": 32,
        "warmup_steps": 1000,
        "mixed_precision": True,
        "enable_curriculum": False,
    },
    "default": DEFAULT_TRAINING_CONFIG,
    "large_scale": {
        "learning_rate": 1e-4,
        "batch_size": 128,
        "warmup_steps": 20000,
        "mixed_precision": True,
        "enable_curriculum": False,
    },
}


def create_model_config(
    size: str = "base", custom_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create model configuration.

    Args:
        size: Model size preset ('small', 'base', 'large')
        custom_config: Custom configuration overrides

    Returns:
        Complete model configuration
    """
    config = DEFAULT_MODEL_CONFIG.copy()

    if size in MODEL_PRESETS:
        config.update(MODEL_PRESETS[size])

    if custom_config:
        config.update(custom_config)

    return config


def create_training_config(
    preset: str = "default", custom_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create training configuration.

    Args:
        preset: Training preset ('debug', 'default', 'large_scale')
        custom_config: Custom configuration overrides

    Returns:
        Complete training configuration
    """
    if preset not in TRAINING_PRESETS:
        raise ValueError(f"Unknown training preset: {preset}")

    config = TRAINING_PRESETS[preset].copy()

    if custom_config:
        config.update(custom_config)

    return config


def create_generation_config(
    custom_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create generation configuration.

    Args:
        custom_config: Custom configuration overrides

    Returns:
        Complete generation configuration
    """
    config = DEFAULT_GENERATION_CONFIG.copy()

    if custom_config:
        config.update(custom_config)

    return config


# Example usage functions
def example_training_setup():
    """
    Example of setting up BIFROST for training.

    Returns:
        Dictionary with example setup
    """
    return {
        "model_config": create_model_config("base"),
        "training_config": create_training_config("default"),
        "data_config": {
            "max_seq_len": 512,
            "property_dropout": 0.3,
            "property_removal": 0.1,
        },
    }


def example_generation_setup():
    """
    Example of setting up BIFROST for generation.

    Returns:
        Dictionary with example setup
    """
    return {
        "model_config": create_model_config("base"),
        "generation_config": create_generation_config(
            {"temperature": 0.8, "top_k": 50, "max_length": 512}
        ),
        "property_targets": {
            "band_gap": 2.5,
            "density": 3.0,
            "energy_above_hull": 0.02,
        },
    }


def save_config(config: Dict[str, Any], filepath: str):
    """
    Save configuration to file.

    Args:
        config: Configuration dictionary
        filepath: Path to save configuration
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        import json

        json.dump(config, f, indent=2)


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from file.

    Args:
        filepath: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(filepath, "r") as f:
        import json

        return json.load(f)
