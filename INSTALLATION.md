# BIFROST Installation Guide

This guide explains how to install and use the BIFROST crystal structure generation library.

## Quick Installation

### Option 1: Install from Source (Recommended for Development)

```bash
# Clone the repository
git clone <repository-url>
cd bifrost

# Install in development mode
pip install -e .

# Test installation
python test_install.py
```

### Option 2: Install Minimal Requirements

```bash
# Install core dependencies only
pip install torch>=2.0.0 numpy>=1.21.0 typing-extensions>=4.0.0

# Add BIFROST to Python path or install in development mode
pip install -e .
```

### Option 3: Install with Materials Science Support

```bash
# Install core + materials science libraries
pip install -e ".[materials]"

# This includes: pymatgen, matminer, ase, spglib
```

## Package Structure

```
bifrost/
├── bifrost/              # Main package
│   ├── model/           # Core model components
│   ├── data/            # Data processing
│   ├── training/        # Training infrastructure
│   └── config.py        # Configuration utilities
├── pyproject.toml       # Modern Python packaging
├── setup.py            # Legacy compatibility
├── requirements.txt     # Dependency specification
└── test_install.py     # Installation verification
```

## Installation Verification

After installation, run the test script to verify everything works:

```bash
python test_install.py
```

You should see output like:
```
============================================================
BIFROST Installation Test
============================================================
Testing BIFROST import...
✓ All imports successful

Testing model creation...
✓ Model created with 7,236,889 parameters

Testing basic functionality...
✓ Forward pass successful
  - Discrete logits shape: torch.Size([2, 5, 1430])
  - Continuous params shape: torch.Size([2, 5, 2])
  - Type probabilities shape: torch.Size([2, 5, 1])

Testing data processing...
✓ Property discretization successful
  - Original: {'bandgap': 2.5, 'density': 3.0, 'ehull': 0.02, 'formation_energy': -2.5}
  - Discretized: {'bandgap': 'BANDGAP_MED', 'density': 'DENSITY_LOW', 'ehull': 'EHULL_LOW', 'formation_energy': 'FORM_NONE'}

============================================================
Test Results: 4/4 passed
✓ All tests passed! BIFROST is ready to use.
```

## Usage Examples

### Basic Model Creation

```python
from bifrost.model import BIFROST, create_bifrost_model, get_bifrost_config

# Create a small model
config = get_bifrost_config("small")
model = create_bifrost_model(config)

print(f"Model has {model.get_num_parameters():,} parameters")
```

### Training with Sample Data

```python
# Run the training demo
python train_demo.py
```

### Using Different Model Sizes

```python
# Small model (7M parameters) - fast training, lower quality
small_config = get_bifrost_config("small")

# Base model (512M parameters) - balanced performance
base_config = get_bifrost_config("base")

# Large model (768M parameters) - best quality, slower training
large_config = get_bifrost_config("large")
```

## Dependency Management

### Core Dependencies (Always Required)

- **torch >= 2.0.0**: Deep learning framework
- **numpy >= 1.21.0**: Numerical computations
- **typing-extensions >= 4.0.0**: Type hints support

### Optional Dependencies

Install based on your needs:

```bash
# Materials science integration
pip install pymatgen matminer ase spglib

# Advanced structure processing
pip install torch-geometric

# Visualization
pip install matplotlib plotly seaborn

# Performance optimization
pip install numba cython
```

## Configuration Files

### pyproject.toml

Modern Python packaging configuration with:
- Project metadata
- Dependency specifications
- Build configuration
- Development tools configuration
- Entry points for command-line scripts

### requirements.txt

Traditional requirements file with:
- Core dependencies
- Optional dependencies (commented out)
- Development dependencies (commented out)

## Development Setup

For contributors and developers:

```bash
# Install in development mode with all optional dependencies
pip install -e ".[all]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black .
isort .

# Type checking
mypy bifrost/

# Lint code
flake8 bifrost/
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure BIFROST is installed in development mode:
   ```bash
   pip install -e .
   ```

2. **Missing Dependencies**: Install required packages:
   ```bash
   pip install torch numpy typing-extensions
   ```

3. **CUDA Issues**: If you have CUDA, ensure PyTorch is installed with CUDA support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Getting Help

- Check the test script output: `python test_install.py`
- Read the documentation: `README_BIFROST.md`
- Look at examples: `example.py`, `train_demo.py`
- Check issues: GitHub Issues

## Next Steps

After successful installation:

1. **Run the demo**: `python train_demo.py`
2. **Explore examples**: Look at `example.py` for usage patterns
3. **Train your model**: Use `train_mp.py` with real Materials Project data
4. **Customize**: Modify configurations in `config.py`

BIFROST is now ready for crystal structure generation and materials discovery research! 🎉
