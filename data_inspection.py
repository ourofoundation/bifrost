#!/usr/bin/env python3
"""
Data inspection utilities for BIFROST.

Modes:
  - validate: Check dataset Wyckoff labels vs tokenizer mapping
  - inspect:  Show sample tokenization and vocabulary info

Usage:
  python data_inspection.py --mode validate --dataset my_dataset.json
  python data_inspection.py --mode inspect
"""

from __future__ import annotations
import argparse
import json
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, List


def load_mapping(mapping_path: str) -> Dict[str, str]:
    with open(mapping_path, "r") as f:
        return json.load(f)


def validate_dataset(dataset_path: str, mapping_path: str) -> None:
    with open(dataset_path, "r") as f:
        data: List[Dict[str, Any]] = json.load(f)

    mapping = load_mapping(mapping_path)
    invalid_count = 0
    total_sites = 0
    invalid_by_sg = Counter()
    invalid_examples_by_sg: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for entry in data:
        sg = int(entry.get("space_group", -1))
        for site in entry.get("wyckoff_positions", []):
            total_sites += 1
            wy = site.get("wyckoff")
            key = f"{sg}_{wy}"
            if mapping.get(key) is None:
                invalid_count += 1
                invalid_by_sg[sg] += 1
                if len(invalid_examples_by_sg[sg]) < 5:
                    invalid_examples_by_sg[sg].append(
                        {
                            "structure_id": entry.get("structure_id"),
                            "wyckoff": wy,
                            "composition": entry.get("composition"),
                        }
                    )

    print(f"Total sites: {total_sites}")
    print(f"Invalid sites: {invalid_count}")
    if invalid_count:
        print("Top offending space groups (first 10):")
        for sg, cnt in invalid_by_sg.most_common(10):
            print(f"  SG {sg}: {cnt} invalid sites")
            for ex in invalid_examples_by_sg[sg]:
                print(f"    - {ex['structure_id']}: wyckoff={ex['wyckoff']}")


import sys
import os
import json
from typing import Dict, Any, List, Tuple
import numpy as np

# Add bifrost to path
sys.path.append("/Users/mmoderwell/ouro/bifrost")

from bifrost.data.tokenizer import BIFROSTTokenizer, tokenizer
from bifrost.data.dataset import CrystalStructureDataset, load_sample_dataset
from bifrost.data.properties import discretize_structure_properties, property_binner


def inspect_raw_data():
    """Examine the raw crystal structure data format."""
    print("=" * 60)
    print("RAW DATA STRUCTURE INSPECTION")
    print("=" * 60)

    # Load sample data
    sample_structures = load_sample_dataset("dummy_path")  # Uses built-in sample data

    print(f"Number of sample structures: {len(sample_structures)}")
    print()

    # Examine first structure in detail
    structure = sample_structures[0]
    print("First structure details:")
    print(f"  Structure ID: {structure['structure_id']}")
    print(f"  Composition: {structure['composition']}")
    print(f"  Space group: {structure['space_group']}")
    print(f"  Number of Wyckoff positions: {len(structure['wyckoff_positions'])}")
    print(f"  Lattice parameters: {structure['lattice']}")
    print(f"  Properties: {structure['properties']}")
    print()

    # Show Wyckoff positions in detail
    print("Wyckoff positions:")
    for i, pos in enumerate(structure["wyckoff_positions"]):
        print(
            f"  {i+1}. Element: {pos['element']}, Wyckoff: {pos['wyckoff']}, Coords: {pos['coordinates']}"
        )
    print()


def inspect_tokenization():
    """Show how structures are tokenized."""
    print("=" * 60)
    print("TOKENIZATION PROCESS INSPECTION")
    print("=" * 60)

    # Load sample data
    sample_structures = load_sample_dataset("dummy_path")
    structure = sample_structures[0]

    print("Original structure:")
    print(f"  Composition: {structure['composition']}")
    print(f"  Space group: {structure['space_group']}")
    print(f"  Properties: {structure['properties']}")
    print()

    # Tokenize the structure
    tokens, token_types = tokenizer.encode_structure(structure)

    print(f"Tokenized sequence length: {len(tokens)} tokens")
    print(f"Token types length: {len(token_types)} types")
    print()

    # Show token breakdown
    print("Token sequence breakdown:")
    print("  Special tokens:")
    print(f"    PAD: {tokenizer.vocab['PAD']}, UNK: {tokenizer.vocab['UNK']}")
    print(f"    SEP: {tokenizer.vocab['SEP']}, EOS: {tokenizer.vocab['EOS']}")
    print(f"    BOS: {tokenizer.vocab['BOS']}, MASK: {tokenizer.vocab['MASK']}")
    print()

    # Decode properties
    if "properties" in structure:
        discretized = discretize_structure_properties(structure["properties"])
        print("Property discretization:")
        for prop, value in structure["properties"].items():
            token = discretized.get(prop, "NONE")
            print(f"  {prop}: {value} -> {token}")
        print()

    # Show composition tokens
    print("Composition tokenization:")
    comp_tokens = tokenizer._encode_composition(structure["composition"])
    element_tokens = comp_tokens[0][: len(structure["composition"])]  # Elements
    count_tokens = comp_tokens[0][len(structure["composition"]) :]  # Counts

    for i, (elem, count) in enumerate(structure["composition"]):
        elem_token = element_tokens[i] if i < len(element_tokens) else "UNK"
        count_token = count_tokens[i] if i < len(count_tokens) else "UNK"
        print(
            f"  {elem} ({count}x) -> Element token: {elem_token}, Count token: {count_token}"
        )
    print()

    # Show space group tokenization
    sg_tokens = tokenizer._encode_space_group(structure["space_group"])
    print(f"Space group {structure['space_group']} -> Token: {sg_tokens[0][0]}")
    print()

    # Show lattice tokenization
    lattice_tokens = tokenizer._encode_lattice(structure["lattice"])
    print("Lattice parameters tokenization:")
    lattice_params = ["a", "b", "c", "alpha", "beta", "gamma"]
    for i, param in enumerate(lattice_params):
        if i < len(lattice_tokens[0]):
            token_val = lattice_tokens[0][i]
            token_type = lattice_tokens[1][i]
            print(
                f"  {param}: {structure['lattice'][param]} -> Token: {token_val} (type: {token_type})"
            )
    print()

    # Show Wyckoff tokenization
    wyck_tokens = tokenizer._encode_wyckoff_positions(
        structure["wyckoff_positions"], structure["space_group"]
    )
    print("Wyckoff positions tokenization:")
    for i, pos in enumerate(structure["wyckoff_positions"]):
        start_idx = i * 5  # wyckoff + element + 3 coords
        if start_idx + 4 < len(wyck_tokens[0]):
            wyck_token = wyck_tokens[0][start_idx]
            elem_token = wyck_tokens[0][start_idx + 1]
            coords = wyck_tokens[0][start_idx + 2 : start_idx + 5]
            coord_types = wyck_tokens[1][start_idx + 2 : start_idx + 5]
            wyck_name = tokenizer.reverse_vocab.get(
                int(wyck_token), f"UNK_{int(wyck_token)}"
            )
            elem_name = tokenizer.reverse_vocab.get(
                int(elem_token), f"UNK_{int(elem_token)}"
            )
            print(
                f"  {pos['element']} at {pos['wyckoff']} -> Wyckoff token: {wyck_token:.0f} ({wyck_name}), Element token: {elem_token:.0f} ({elem_name})"
            )
            print(
                f"    Coordinates: {pos['coordinates']} -> Tokens: {coords} (types: {coord_types})"
            )
    print()


def inspect_full_token_sequence():
    """Show the complete token sequence for a structure."""
    print("=" * 60)
    print("COMPLETE TOKEN SEQUENCE INSPECTION")
    print("=" * 60)

    # Load sample data
    sample_structures = load_sample_dataset("dummy_path")
    structure = sample_structures[0]

    # Get full token sequence
    tokens, token_types = tokenizer.encode_structure(structure)

    print(f"Full token sequence ({len(tokens)} tokens):")
    print("Format: token_id(token_type): 'decoded_name' [actual_value]")
    print()

    # Break down the sequence
    i = 0
    while i < len(tokens):
        token_id = tokens[i]
        token_type = token_types[i]

        # Decode token name
        token_name = tokenizer.reverse_vocab.get(int(token_id), f"UNK_{int(token_id)}")

        # Format output
        if token_type == 0:  # discrete
            print(f"  {i:3d}: {token_id:4.0f}({token_type}): '{token_name}'")
        else:  # continuous
            actual_value = tokens[
                i
            ]  # For continuous tokens, the value is the token itself
            print(f"  {i:3d}: {token_id:7.3f}({token_type}): CONTINUOUS_VALUE")

        i += 1

    print()
    print("Token type legend:")
    print("  0: Discrete tokens (elements, counts, space groups, etc.)")
    print("  1: Continuous tokens (coordinates, lattice parameters)")


def inspect_vocabulary():
    """Examine the tokenizer vocabulary."""
    print("=" * 60)
    print("VOCABULARY INSPECTION")
    print("=" * 60)

    vocab = tokenizer.vocab
    reverse_vocab = tokenizer.reverse_vocab

    print(f"Total vocabulary size: {len(vocab)} tokens")
    print()

    # Count tokens by type
    token_counts = {}
    for token_name in vocab.keys():
        if token_name.startswith("PAD"):
            token_counts["Special"] = token_counts.get("Special", 0) + 1
        elif token_name.startswith("UNK"):
            token_counts["Special"] = token_counts.get("Special", 0) + 1
        elif token_name.startswith("MASK"):
            token_counts["Special"] = token_counts.get("Special", 0) + 1
        elif token_name.startswith("SEP"):
            token_counts["Special"] = token_counts.get("Special", 0) + 1
        elif token_name.startswith("EOS"):
            token_counts["Special"] = token_counts.get("Special", 0) + 1
        elif token_name.startswith("BOS"):
            token_counts["Special"] = token_counts.get("Special", 0) + 1
        elif token_name.startswith("COUNT_"):
            token_counts["Counts"] = token_counts.get("Counts", 0) + 1
        elif token_name.startswith("SPACE_"):
            token_counts["Space Groups"] = token_counts.get("Space Groups", 0) + 1
        elif token_name.startswith("WYCK_"):
            token_counts["Wyckoff"] = token_counts.get("Wyckoff", 0) + 1
        elif len(token_name) <= 2 and token_name.isalpha():
            token_counts["Elements"] = token_counts.get("Elements", 0) + 1
        elif any(
            token_name.startswith(prefix)
            for prefix in ["BANDGAP_", "DENSITY_", "EHULL_", "FORM_", "BULK_"]
        ):
            token_counts["Properties"] = token_counts.get("Properties", 0) + 1
        else:
            token_counts["Other"] = token_counts.get("Other", 0) + 1

    print("Token distribution:")
    for category, count in token_counts.items():
        print(f"  {category}: {count} tokens")
    print()

    # Show some examples
    print("Sample tokens by category:")

    # Elements
    elements = [name for name in vocab.keys() if len(name) <= 2 and name.isalpha()][:10]
    print(f"  Elements (first 10): {elements}")

    # Space groups
    space_groups = [name for name in vocab.keys() if name.startswith("SPACE_")][:5]
    print(f"  Space groups (first 5): {space_groups}")

    # Counts
    counts = [name for name in vocab.keys() if name.startswith("COUNT_")][:5]
    print(f"  Counts (first 5): {counts}")

    # Properties
    properties = [
        name
        for name in vocab.keys()
        if any(
            name.startswith(prefix)
            for prefix in ["BANDGAP_", "DENSITY_", "EHULL_", "FORM_", "BULK_"]
        )
    ]
    print(f"  Properties: {properties[:5]}")
    print()


def inspect_dataset_processing():
    """Show how the dataset processes structures."""
    print("=" * 60)
    print("DATASET PROCESSING INSPECTION")
    print("=" * 60)

    # Load sample data
    sample_structures = load_sample_dataset("dummy_path")

    # Create dataset
    dataset = CrystalStructureDataset(
        sample_structures,
        tokenizer,
        max_seq_len=512,
        property_dropout=0.3,
        property_removal=0.1,
        curriculum_level=0,
    )

    print(f"Dataset size: {len(dataset)} structures")
    print()

    # Get first example
    example = dataset[0]

    print("First training example:")
    print(f"  Input tokens shape: {example['input_tokens'].shape}")
    print(f"  Target tokens shape: {example['target_tokens'].shape}")
    print(f"  Token types shape: {example['token_types'].shape}")
    print(f"  Attention mask shape: {example['attention_mask'].shape}")
    print()

    # Show token values
    input_tokens = example["input_tokens"]
    target_tokens = example["target_tokens"]
    token_types = example["token_types"]

    print("Sample input tokens (first 20):")
    for i in range(min(20, len(input_tokens))):
        token_val = input_tokens[i].item()
        token_type = token_types[i].item()
        if token_type == 0:  # discrete
            token_name = tokenizer.reverse_vocab.get(
                int(token_val), f"UNK_{int(token_val)}"
            )
            print(f"    {i:2d}: {token_val:6.0f} (discrete) -> '{token_name}'")
        else:  # continuous
            print(f"    {i:2d}: {token_val:6.3f} (continuous)")
    print()

    print("Sample target tokens (first 20):")
    for i in range(min(20, len(target_tokens))):
        token_val = target_tokens[i].item()
        print(f"    {i:2d}: {token_val:6.3f}")
    print()

    # Show property statistics
    stats = dataset.get_property_statistics()
    print("Property statistics:")
    print(f"  Total structures: {stats['total_structures']}")
    print(f"  Properties per structure: {stats['properties_per_structure']}")
    print(f"  Property value ranges: {stats['property_values']}")


def inspect_property_discretization():
    """Show how continuous properties are discretized."""
    print("=" * 60)
    print("PROPERTY DISCRETIZATION INSPECTION")
    print("=" * 60)

    # Sample property values
    sample_properties = {
        "band_gap": [0.0, 1.0, 2.5, 3.0, 5.0],
        "density": [1.0, 3.0, 5.0, 7.0, 10.0],
        "energy_above_hull": [0.0, 0.02, 0.06, 0.08, 0.15],
        "formation_energy_per_atom": [-3.0, -1.5, -0.5, 0.0, 1.0],
    }

    print("Property discretization examples:")
    print("Format: value -> bin -> token")
    print()

    for prop_name, values in sample_properties.items():
        print(f"{prop_name.upper()} discretization:")
        for value in values:
            bin_name = property_binner.get_property_bin(prop_name, value)
            token = property_binner.get_property_token(prop_name, bin_name)
            print(f"  {value:6.2f} -> {bin_name:8s} -> {token}")
        print()

    # Show all property bins
    print("All property bins:")
    for prop_name, config in property_binner.property_bins.items():
        print(f"  {prop_name}:")
        print(f"    Thresholds: {config['thresholds']}")
        print(f"    Tokens: {config['tokens']}")
        print(f"    Description: {config['description']}")
        print()


def run_inspection():
    """Run the full interactive inspection output."""
    print("BIFROST Training Data and Tokenization Inspection")
    print("=" * 60)
    print()

    try:
        # Run all inspections
        inspect_raw_data()
        inspect_tokenization()
        inspect_full_token_sequence()
        inspect_vocabulary()
        inspect_dataset_processing()
        inspect_property_discretization()

        print("=" * 60)
        print("INSPECTION COMPLETE")
        print("=" * 60)
        print()
        print("Summary:")
        print(
            "- BIFROST uses a hybrid tokenization approach with discrete and continuous tokens"
        )
        print(
            "- Discrete tokens include elements, space groups, Wyckoff positions, and property bins"
        )
        print("- Continuous tokens represent coordinates and lattice parameters")
        print(
            "- Properties are discretized into bins (NONE, LOW, MED, HIGH) for conditioning"
        )
        print(
            "- The model learns to predict both discrete structural tokens and continuous values"
        )

    except Exception as e:
        print(f"Inspection failed with error: {e}")
        import traceback

        traceback.print_exc()


def main() -> None:
    parser = argparse.ArgumentParser(description="BIFROST data inspection")
    parser.add_argument(
        "--mode",
        choices=["validate", "inspect"],
        default="validate",
        help="Run validation or interactive inspection",
    )
    parser.add_argument("--dataset", help="Path to dataset JSON (for validate mode)")
    parser.add_argument(
        "--mapping",
        default="bifrost/data/wyckoff_map_sg_letter_to_orbit.json",
        help="Path to sg-letter->orbit mapping JSON (for validate mode)",
    )

    args = parser.parse_args()

    if args.mode == "validate":
        if not args.dataset:
            parser.error("--dataset is required in validate mode")
        validate_dataset(args.dataset, args.mapping)
    else:
        run_inspection()


if __name__ == "__main__":
    main()
