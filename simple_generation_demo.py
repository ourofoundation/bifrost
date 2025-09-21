#!/usr/bin/env python3
"""
Simple BIFROST Generation Demo.

A simplified demonstration of crystal structure generation with property conditioning.
"""

import torch
import random
import numpy as np
from typing import Dict, Any, List

from bifrost.model import create_bifrost_model, get_bifrost_config
from bifrost.data.tokenizer import tokenizer
from bifrost.data.properties import discretize_structure_properties
from bifrost.generation.decoder import StructureDecoder


def create_simple_structure(property_values: Dict[str, float]) -> Dict[str, Any]:
    """Create a simple structure with given properties."""
    return {
        "structure_id": "demo_structure",
        "composition": [("Li", 2), ("Fe", 1), ("P", 1), ("O", 4)],
        "space_group": 62,
        "wyckoff_positions": [
            {"element": "Li", "wyckoff": "4c", "coordinates": [0.0, 0.5, 0.0]},
            {"element": "Fe", "wyckoff": "4c", "coordinates": [0.28, 0.25, 0.97]},
            {"element": "P", "wyckoff": "4c", "coordinates": [0.09, 0.25, 0.42]},
            {"element": "O", "wyckoff": "4c", "coordinates": [0.10, 0.25, 0.74]},
        ],
        "lattice": {
            "a": 10.33,
            "b": 6.01,
            "c": 4.69,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        },
        "properties": property_values,
    }


def generate_structure_with_properties(
    model, target_properties: Dict[str, float], max_length: int = 100
) -> Dict[str, Any]:
    """
    Generate a crystal structure with target properties.

    This is a simplified version that demonstrates the concept.
    """
    print("Generating structure with properties:", target_properties)

    # For now, just return a structure with the target properties
    # In a full implementation, this would use the model's generate method
    structure = create_simple_structure(target_properties)

    # Add some variation to the lattice parameters based on properties
    if "density" in target_properties:
        density = target_properties["density"]
        # Adjust lattice parameters based on density
        base_a = 10.33
        base_b = 6.01
        base_c = 4.69

        # Higher density = smaller lattice parameters
        scale_factor = 1.0 / (density / 4.0)  # Normalize around density=4
        structure["lattice"]["a"] = base_a * scale_factor
        structure["lattice"]["b"] = base_b * scale_factor
        structure["lattice"]["c"] = base_c * scale_factor

    return structure


def main():
    """Demonstrate BIFROST generation."""
    print("=" * 60)
    print("BIFROST Generation Demo")
    print("=" * 60)

    # Create model
    config = get_bifrost_config("small")
    model = create_bifrost_model(config)
    model.eval()

    print(f"✓ Loaded model with {model.get_num_parameters():,} parameters")

    # Create decoder
    decoder = StructureDecoder(tokenizer)

    # Example property configurations
    examples = {
        "semiconductor": {"bandgap": 2.0, "density": 4.0, "ehull": 0.05},
        "metal": {"bandgap": 0.0, "density": 8.0, "ehull": 0.0},
        "insulator": {"bandgap": 5.0, "density": 3.0, "ehull": 0.1},
        "lightweight": {"density": 2.0, "formation_energy": -1.0},
    }

    print("\nGenerating structures with different property profiles...")

    generated_structures = []

    for name, properties in examples.items():
        print(f"\n--- {name.upper()} ---")

        # Generate structure
        structure = generate_structure_with_properties(model, properties)

        print(f"✓ Generated structure: {structure['structure_id']}")
        print(f"  Composition: {structure['composition']}")
        print(f"  Space group: {structure['space_group']}")
        print(f"  Properties: {structure['properties']}")
        a_val = structure["lattice"]["a"]
        b_val = structure["lattice"]["b"]
        c_val = structure["lattice"]["c"]
        print(f"  Lattice: a={a_val:.3f}, b={b_val:.3f}, c={c_val:.3f}")

        generated_structures.append(structure)

    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)

    print(f"Generated {len(generated_structures)} structures:")

    for i, structure in enumerate(generated_structures):
        print(f"\n{i+1}. {structure['structure_id']}")
        print(f"   Properties: {structure['properties']}")
        print(f"   Lattice: {structure['lattice']}")

    print("\nBIFROST generation demonstration completed!")
    print("\nNext steps:")
    print("- Implement full autoregressive generation")
    print("- Add property-conditioned sampling")
    print("- Create structure validation and optimization")
    print("- Add support for multiple structure types")


if __name__ == "__main__":
    main()
