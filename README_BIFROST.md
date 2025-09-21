# BIFROST: Crystal Structure Generation with Property Conditioning

BIFROST is an autoregressive transformer model for generating 3D crystal structures conditioned on target material properties. It uses a sequence-based representation of crystal structures and learns to generate chemically valid, stable structures.

## Overview

BIFROST addresses the challenge of generating crystal structures with specific material properties by using a transformer-based autoregressive model. The key innovation is representing crystal structures as linear sequences of tokens that include:

- **Property conditioning**: Target properties as discrete bins in a prefix
- **Composition**: Chemical elements and their stoichiometric ratios
- **Symmetry**: Space group and Wyckoff positions
- **Structure**: Atomic coordinates and lattice parameters

## Architecture

### Core Components

1. **BIFROST Model** (`bifrost/model/bifrost.py`)
   - Main model combining embeddings, transformer blocks, and output heads
   - Supports both discrete tokens and continuous values
   - Property-conditioned generation via prefix tokens

2. **Data Pipeline** (`bifrost/data/`)
   - **Properties** (`properties.py`): Property discretization into bins
   - **Tokenizer** (`tokenizer.py`): Structure ↔ token sequence conversion
   - **Dataset** (`dataset.py`): PyTorch datasets with curriculum learning support

3. **Training** (`bifrost/training/`)
   - **Curriculum Learning** (`curriculum.py`): Progressive training from simple to complex structures
   - **Optimization** (`optimizer.py`): AdamW with OneCycleLR scheduling
   - **Training Loop** (`train.py`): Complete training pipeline with mixed precision

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd bifrost

# Install dependencies
pip install torch transformers

# Optional: install additional dependencies for full functionality
pip install pymatgen  # For crystal structure manipulation
pip install torch-geometric  # For advanced structure processing
```

## Quick Start

### Basic Usage

```python
import torch
from bifrost.model import BIFROST
from bifrost.config import create_model_config

# Create model
config = create_model_config('small')
model = BIFROST(**config)

# Set up training data (example)
structures = [
    {
        'composition': [('Li', 2), ('Fe', 1), ('P', 1), ('O', 4)],
        'space_group': 62,
        'wyckoff_positions': [...],
        'lattice': {...},
        'properties': {'bandgap': 3.5, 'density': 3.5}
    }
]

# Train model (see example.py for complete training setup)
# trainer = create_trainer(model, train_dataloader)
# results = trainer.train(num_epochs=100)
```

### Configuration

BIFROST provides several configuration presets:

```python
from bifrost.config import create_model_config, create_training_config

# Model configurations
small_config = create_model_config('small')    # 256M parameters
base_config = create_model_config('base')      # 512M parameters
large_config = create_model_config('large')    # 768M parameters

# Training configurations
debug_config = create_training_config('debug')      # Quick debugging
default_config = create_training_config('default')  # Standard training
large_config = create_training_config('large_scale') # Large-scale training
```

## Features

### Property-Conditioned Generation

Generate structures with target properties:

```python
# Target properties
property_targets = {
    'bandgap': 2.5,      # eV
    'density': 3.0,      # g/cm³
    'ehull': 0.02        # eV/atom
}

# Generate structure conditioned on these properties
# (Generation pipeline implementation needed)
```

### Curriculum Learning

Progressive training from simple to complex structures:

1. **Level 0**: Simple structures (≤5 elements, ≤5 Wyckoff sites)
2. **Level 1**: Medium complexity (≤10 elements, ≤10 Wyckoff sites)
3. **Level 2**: Full complexity (all structures)

### Mixed Precision Training

Automatic mixed precision support for faster training:

```python
config = create_training_config('default')
config['mixed_precision'] = True  # Enable FP16 training
```

## File Structure

```
bifrost/
├── model/                 # Core model components
│   ├── __init__.py
│   ├── bifrost.py         # Main BIFROST model
│   ├── embeddings.py      # Token and positional embeddings
│   ├── transformer.py     # Transformer blocks
│   └── heads.py           # Output heads for prediction
├── data/                  # Data pipeline
│   ├── __init__.py
│   ├── properties.py      # Property discretization
│   ├── tokenizer.py       # Structure ↔ token conversion
│   └── dataset.py         # PyTorch datasets
├── training/              # Training pipeline
│   ├── __init__.py
│   ├── curriculum.py      # Curriculum learning
│   ├── optimizer.py       # Optimization and scheduling
│   └── train.py           # Training loop
├── config.py              # Configuration and presets
├── example.py             # Example usage
└── __init__.py            # Package initialization
```

## Implementation Details

### Tokenization Strategy

Crystal structures are converted to sequences:

```
[PROPERTY_PREFIX] [SEP] [COMPOSITION] [SYMMETRY] [POSITIONS] [LATTICE] [EOS]
```

Where:
- **PROPERTY_PREFIX**: Discretized property tokens (e.g., `BANDGAP_MED`, `DENSITY_LOW`)
- **COMPOSITION**: Element and count tokens (e.g., `Li`, `COUNT_2`, `Fe`, `COUNT_1`)
- **SYMMETRY**: Space group tokens (e.g., `SPACE_62`)
- **POSITIONS**: Wyckoff positions with coordinates
- **LATTICE**: Continuous lattice parameters (a, b, c, α, β, γ)

### Property Discretization

Continuous properties are binned into discrete levels:

```python
property_bins = {
    'bandgap': {
        'thresholds': [0.5, 2.0, 4.0],  # eV
        'tokens': ['BANDGAP_NONE', 'BANDGAP_LOW', 'BANDGAP_MED', 'BANDGAP_HIGH']
    },
    'density': {
        'thresholds': [2.0, 4.0, 8.0],  # g/cm³
        'tokens': ['DENSITY_NONE', 'DENSITY_LOW', 'DENSITY_MED', 'DENSITY_HIGH']
    }
}
```

### Model Architecture

- **Embedding Layer**: Handles both discrete tokens and continuous values
- **Positional Encoding**: Sinusoidal encoding for sequence positions
- **Transformer Blocks**: Standard transformer with causal attention
- **Output Heads**: Separate heads for discrete/continuous prediction and token type classification

## Training

### Hyperparameters

```python
hyperparameters = {
    # Model
    'd_model': 512,
    'n_heads': 16,
    'n_layers': 16,
    'd_ff': 2048,
    'dropout': 0.1,
    'vocab_size': 1430,

    # Training
    'batch_size': 256,
    'learning_rate': 2e-4,
    'warmup_steps': 10000,
    'weight_decay': 0.01,
    'gradient_clip': 1.0,

    # Properties
    'property_dropout': 0.3,
    'n_property_bins': 4,
}
```

### Data Requirements

- **Training data**: ~2.4M crystal structures from Materials Project, Alexandria, WBM
- **Property filtering**: Structures with Ehull < 0.25 eV/atom
- **Sequence length**: Maximum 512 tokens
- **Curriculum**: Progressive complexity increase

## Generation

### Inference Process

1. **Property Conditioning**: Convert target properties to discrete tokens
2. **Prefix Construction**: Build property prefix sequence
3. **Autoregressive Generation**: Generate structure tokens one by one
4. **Constraint Enforcement**: Apply chemical and structural constraints
5. **Post-processing**: Structure relaxation and validation

### Sampling Strategies

- **Temperature Sampling**: Control generation diversity
- **Top-k Sampling**: Limit to top-k most probable tokens
- **Top-p (Nucleus) Sampling**: Sample from smallest set with cumulative probability ≥ p

## Evaluation

### Metrics

- **Validity**: Percentage of generated structures passing constraint checks
- **Uniqueness**: Percentage of unique structures after deduplication
- **Novelty**: Percentage not found in training set
- **Stability**: Percentage within 0.1 eV/atom of convex hull
- **Property Accuracy**: MAE vs DFT-calculated values

### Benchmarking

Comprehensive evaluation suite for:
- Generation quality assessment
- Property targeting accuracy
- Diversity and coverage analysis
- Computational efficiency

## Contributing

To contribute to BIFROST:

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Ensure code follows project style
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use BIFROST in your research, please cite:

```bibtex
@article{bifrost2024,
    title={BIFROST: Crystal Structure Generation with Property Conditioning},
    author={BIFROST Team},
    journal={arXiv preprint},
    year={2024}
}
```

## Acknowledgments

- Materials Project for crystal structure data
- PyTorch team for the deep learning framework
- Research community for transformer architectures and crystal generation methods

## Contact

For questions, issues, or contributions, please open an issue on GitHub or contact the development team.
