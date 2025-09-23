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
        if prop_name == "band_gap":
            min_val = max(0.0, min_val)
            max_val = min(10.0, max_val)
        elif prop_name == "density":
            min_val = max(0.1, min_val)
            max_val = min(20.0, max_val)
        elif prop_name == "energy_above_hull":
            min_val = max(0.0, min_val)
            max_val = min(1.0, max_val)
        elif prop_name == "formation_energy_per_atom":
            min_val = max(-10.0, min_val)
            max_val = min(5.0, max_val)
        elif prop_name == "bulk_modulus":
            min_val = max(0.0, min_val)
            max_val = min(500.0, max_val)
        elif prop_name == "shear_modulus":
            min_val = max(0.0, min_val)
            max_val = min(500.0, max_val)
        elif prop_name == "efermi":
            min_val = max(0.0, min_val)
            max_val = min(10.0, max_val)
        elif prop_name == "total_magnetization":
            min_val = max(0.0, min_val)
            max_val = min(10.0, max_val)

        validated_ranges[prop_name] = (min_val, max_val)

    return validated_ranges


def get_property_examples() -> Dict[str, Dict[str, Any]]:
    """Get example property configurations for different use cases."""
    examples = {
        "semiconductor": {
            "band_gap": 2.0,
            "energy_above_hull": 0.05,
            "density": 4.0,
        },
        "metal": {
            "band_gap": 0.0,
            "energy_above_hull": 0.0,
            "density": 8.0,
        },
        "insulator": {
            "band_gap": 5.0,
            "energy_above_hull": 0.1,
            "density": 3.0,
        },
        "lightweight": {
            "density": 2.0,
            "formation_energy_per_atom": -1.0,
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
    elif format.lower() == "cif":
        # Prefer robust CIF writing via pymatgen
        try:
            from pymatgen.core import Lattice, Structure as PMGStructure
            from pymatgen.io.cif import CifWriter
        except Exception as e:
            raise ImportError(
                "pymatgen is required for CIF export. Install with `pip install pymatgen`."
            ) from e

        def _pmg_structure_from_dict(s: Dict[str, Any]) -> PMGStructure:
            lattice_dict = s.get("lattice", {})
            a = float(lattice_dict.get("a", 1.0))
            b = float(lattice_dict.get("b", 1.0))
            c = float(lattice_dict.get("c", 1.0))
            alpha = float(lattice_dict.get("alpha", 90.0))
            beta = float(lattice_dict.get("beta", 90.0))
            gamma = float(lattice_dict.get("gamma", 90.0))
            lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)

            wyckoffs = s.get("wyckoff_positions", [])
            if not wyckoffs:
                raise ValueError(
                    "Cannot write CIF: structure has no atomic positions (wyckoff_positions)."
                )

            species: List[str] = []
            frac_coords: List[List[float]] = []
            for pos in wyckoffs:
                elem = pos.get("element")
                coords = pos.get("coordinates", [0.0, 0.0, 0.0])
                if elem is None or len(coords) < 3:
                    continue
                x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
                # Ensure fractional range [0,1)
                x = x % 1.0
                y = y % 1.0
                z = z % 1.0
                species.append(str(elem))
                frac_coords.append([x, y, z])

            if not species:
                raise ValueError(
                    "Cannot write CIF: no valid atomic sites parsed from wyckoff_positions."
                )

            pmg_struct = PMGStructure(
                lattice, species, frac_coords, coords_are_cartesian=False
            )
            return pmg_struct

        def _write_single(target_path: Path, s: Dict[str, Any]) -> None:
            pmg_struct = _pmg_structure_from_dict(s)
            writer = CifWriter(pmg_struct)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            writer.write_file(str(target_path))

        # Determine output strategy
        if len(structures) == 0:
            print("✓ Saved 0 structures to", output_path)
        elif len(structures) == 1:
            target = output_path
            if target.suffix.lower() != ".cif":
                target = target.with_suffix(".cif")
            _write_single(target, structures[0])
            print(f"✓ Saved 1 structure to {target}")
        else:
            if output_path.suffix and output_path.suffix.lower() == ".cif":
                base_dir = output_path.parent
                base_name = output_path.stem
            elif output_path.is_dir() or (not output_path.suffix):
                base_dir = output_path
                base_name = output_path.stem or "structure"
            else:
                base_dir = output_path.parent
                base_name = output_path.stem or "structure"

            base_dir.mkdir(parents=True, exist_ok=True)
            written = 0
            for idx, s in enumerate(structures, 1):
                sid = s.get("structure_id")
                filename = f"{base_name}_{idx:03d}.cif" if not sid else f"{sid}.cif"
                target = base_dir / filename
                _write_single(target, s)
                written += 1
            print(f"✓ Saved {written} structures to {base_dir}")
    else:
        raise ValueError(f"Unsupported format: {format}")

    if format.lower() in ("json", "yaml"):
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
