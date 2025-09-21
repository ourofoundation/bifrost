"""
Generation utilities for BIFROST.

Utility functions for property conditioning and structure generation.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path


def create_property_prefix(
    target_properties: Dict[str, Union[float, str]], tokenizer=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create property prefix tokens for generation.

    Args:
        target_properties: Dict of property_name -> value/bin_name
        tokenizer: BIFROST tokenizer instance

    Returns:
        Tuple of (prefix_tokens, prefix_types)
    """
    if tokenizer is None:
        from ..data.tokenizer import tokenizer

    # Convert property values to tokens
    property_tokens = []
    property_types = []

    from ..data.properties import get_property_bins

    property_bins = get_property_bins()

    for prop_name, value in target_properties.items():
        if prop_name not in property_bins:
            print(f"Warning: Unknown property '{prop_name}', skipping")
            continue

        if isinstance(value, str):
            # Assume it's already a bin name (e.g., "BANDGAP_MED")
            bin_name = value.replace(f"{prop_name.upper()}_", "").lower()
            token_name = value
        else:
            # Convert numeric value to bin
            from ..data.properties import property_binner

            bin_name = property_binner.get_property_bin(prop_name, value)
            token_name = property_binner.get_property_token(prop_name, bin_name)

        # Get token ID
        token_id = tokenizer.vocab.get(token_name, tokenizer.vocab["UNK"])
        property_tokens.append(token_id)
        property_types.append(tokenizer.token_types["PROPERTY"])

    if not property_tokens:
        raise ValueError("No valid properties specified")

    # Convert to tensors
    prefix_tokens = torch.tensor([property_tokens], dtype=torch.long)
    prefix_types = torch.tensor([property_types], dtype=torch.long)

    return prefix_tokens, prefix_types


def sample_structure(
    model,
    target_properties: Dict[str, Union[float, str]],
    tokenizer=None,
    max_length: int = 512,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Sample a single crystal structure with property conditioning.

    Args:
        model: BIFROST model instance
        target_properties: Target properties for conditioning
        tokenizer: BIFROST tokenizer instance
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        device: Device to run generation on

    Returns:
        Generated crystal structure dictionary
    """
    if tokenizer is None:
        from ..data.tokenizer import tokenizer

    if device is None:
        device = next(model.parameters()).device

    # Create property prefix
    prefix_tokens, prefix_types = create_property_prefix(target_properties, tokenizer)

    # Generate token sequence
    with torch.no_grad():
        generated_tokens, generated_types = model.generate(
            prefix_tokens.to(device),
            prefix_types.to(device),
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    # Decode structure
    from .decoder import StructureDecoder

    decoder = StructureDecoder(tokenizer)

    tokens = generated_tokens[0].cpu().numpy()
    types = generated_types[0].cpu().numpy()

    structure = decoder.decode_structure(tokens, types)

    if structure:
        structure["generated_properties"] = target_properties.copy()

    return structure


def validate_property_ranges(
    property_ranges: Dict[str, Tuple[float, float]],
) -> Dict[str, Tuple[float, float]]:
    """
    Validate and adjust property ranges to reasonable values.

    Args:
        property_ranges: Dict of property_name -> (min, max) ranges

    Returns:
        Validated property ranges
    """
    from ..data.properties import get_property_bins

    property_bins = get_property_bins()
    validated_ranges = {}

    for prop_name, (min_val, max_val) in property_ranges.items():
        if prop_name not in property_bins:
            print(f"Warning: Unknown property '{prop_name}', skipping")
            continue

        # Get typical ranges for this property
        thresholds = property_bins[prop_name]["thresholds"]

        # Adjust ranges to be within reasonable bounds
        if prop_name == "bandgap":
            min_val = max(0.0, min_val)
            max_val = min(10.0, max_val)
        elif prop_name == "density":
            min_val = max(0.1, min_val)
            max_val = min(20.0, max_val)
        elif prop_name == "ehull":
            min_val = max(0.0, min_val)
            max_val = min(1.0, max_val)
        elif prop_name == "formation_energy":
            min_val = max(-10.0, min_val)
            max_val = min(5.0, max_val)
        elif prop_name == "bulk_modulus":
            min_val = max(0.0, min_val)
            max_val = min(500.0, max_val)

        validated_ranges[prop_name] = (min_val, max_val)

    return validated_ranges


def get_property_examples() -> Dict[str, Dict[str, Any]]:
    """Get example property configurations for different use cases."""
    examples = {
        "semiconductor": {
            "bandgap": 2.0,
            "ehull": 0.05,
            "density": 4.0,
        },
        "metal": {
            "bandgap": 0.0,
            "ehull": 0.0,
            "density": 8.0,
        },
        "insulator": {
            "bandgap": 5.0,
            "ehull": 0.1,
            "density": 3.0,
        },
        "lightweight": {
            "density": 2.0,
            "formation_energy": -1.0,
        },
        "high_density": {
            "density": 10.0,
            "bulk_modulus": 200.0,
        },
    }

    return examples


def save_structures(
    structures: List[Dict[str, Any]],
    output_path: Union[str, Path],
    format: str = "json",
) -> None:
    """
    Save generated structures to file.

    Args:
        structures: List of crystal structure dictionaries
        output_path: Path to save file
        format: Output format ('json', 'yaml')
    """
    import json
    import yaml

    output_path = Path(output_path)

    if format.lower() == "json":
        with open(output_path, "w") as f:
            json.dump(structures, f, indent=2, default=str)
    elif format.lower() == "yaml":
        with open(output_path, "w") as f:
            yaml.dump(structures, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"✓ Saved {len(structures)} structures to {output_path}")


def load_structures(
    input_path: Union[str, Path], format: str = "json"
) -> List[Dict[str, Any]]:
    """
    Load structures from file.

    Args:
        input_path: Path to load file from
        format: Input format ('json', 'yaml')

    Returns:
        List of crystal structure dictionaries
    """
    import json
    import yaml

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    if format.lower() == "json":
        with open(input_path, "r") as f:
            structures = json.load(f)
    elif format.lower() == "yaml":
        with open(input_path, "r") as f:
            structures = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported format: {format}")

    return structures
