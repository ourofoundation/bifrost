#!/usr/bin/env python3
"""
BIFROST Generation CLI.

Command-line interface for generating crystal structures with property conditioning.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

from .generation import BIFROSTGenerator, get_property_examples, save_structures


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Generate crystal structures with property conditioning using BIFROST",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate structures with specific properties
  bifrost-generate --properties '{"band_gap": 2.0, "density": 4.0}' --num-samples 10

  # Generate with property ranges
  bifrost-generate --ranges '{"band_gap": [1.0, 3.0], "ehull": [0.0, 0.1]}' --num-samples 5

  # Use example configurations
  bifrost-generate --example semiconductor --num-samples 20

  # Generate with custom model
  bifrost-generate --model-path checkpoints/model.pt --properties '{"band_gap": 3.0}'
        """,
    )

    parser.add_argument(
        "--model-path", type=str, help="Path to trained model checkpoint (optional)"
    )

    parser.add_argument(
        "--model-config",
        type=str,
        default="small",
        choices=["small", "medium", "large"],
        help="Model configuration to use (default: small)",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        help="Device to run generation on (default: auto-detect)",
    )

    # Property specification options
    prop_group = parser.add_mutually_exclusive_group()

    prop_group.add_argument(
        "--properties",
        type=str,
        help='JSON string of target properties, e.g., \'{"band_gap": 2.0, "density": 4.0}\'',
    )

    prop_group.add_argument(
        "--ranges",
        type=str,
        help='JSON string of property ranges, e.g., \'{"band_gap": [1.0, 3.0], "density": [3.0, 5.0]}\'',
    )

    prop_group.add_argument(
        "--example",
        type=str,
        choices=["semiconductor", "metal", "insulator", "lightweight", "high_density"],
        help="Use predefined property configuration",
    )

    # Generation parameters
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of structures to generate (default: 1)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for generation (default: 1)",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum sequence length for generation (defaults to model's max_seq_len)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1)",
    )

    parser.add_argument("--top-k", type=int, help="Top-k sampling parameter (optional)")

    parser.add_argument(
        "--top-p", type=float, help="Top-p (nucleus) sampling parameter (optional)"
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default="generated_structures.cif",
        help="Output file path (default: generated_structures.cif)",
    )

    parser.add_argument(
        "--format",
        type=str,
        default="cif",
        choices=["json", "yaml", "cif"],
        help="Output format (default: cif)",
    )

    # Debug/inspection options
    parser.add_argument(
        "--print-sequences",
        action="store_true",
        help="Print raw generated token/type sequences for inspection",
    )
    parser.add_argument(
        "--print-decoded",
        action="store_true",
        help="When printing sequences, also show decoded token names",
    )

    # Information options
    parser.add_argument(
        "--list-properties",
        action="store_true",
        help="List available properties for conditioning",
    )

    parser.add_argument(
        "--list-examples",
        action="store_true",
        help="List example property configurations",
    )

    return parser


def parse_properties(properties_str: str) -> Dict[str, Any]:
    """Parse properties JSON string."""
    try:
        return json.loads(properties_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for properties: {e}")


def parse_ranges(ranges_str: str) -> Dict[str, List[float]]:
    """Parse property ranges JSON string."""
    try:
        ranges = json.loads(ranges_str)
        # Convert lists to tuples for consistency
        return {k: tuple(v) for k, v in ranges.items()}
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for ranges: {e}")


def main():
    """Main CLI function."""
    parser = create_parser()
    args = parser.parse_args()

    # Handle information requests
    if args.list_properties or args.list_examples:
        if args.list_properties:
            print("Available properties for conditioning:")
            print("=" * 50)

            # Create a temporary generator to get property info
            try:
                generator = BIFROSTGenerator(model_config="small")
                properties = generator.get_available_properties()

                for prop_name, prop_info in properties.items():
                    print(f"\n{prop_name.upper()}:")
                    print(f"  Description: {prop_info['description']}")
                    print(f"  Thresholds: {prop_info['thresholds']}")
                    print(f"  Tokens: {prop_info['tokens']}")

            except Exception as e:
                print(f"Error loading properties: {e}")
                sys.exit(1)

        if args.list_examples:
            print("\nExample property configurations:")
            print("=" * 50)
            examples = get_property_examples()

            for name, props in examples.items():
                print(f"\n{name}:")
                for prop, value in props.items():
                    print(f"  {prop}: {value}")

        return

    # Check if property specification is provided for generation
    if not (args.properties or args.ranges or args.example):
        parser.error(
            "One of --properties, --ranges, or --example must be specified for generation"
        )

    # Create generator
    try:
        generator = BIFROSTGenerator(
            model_path=args.model_path,
            model_config=args.model_config,
            device=args.device,
        )
        # Model param count is printed within BIFROSTGenerator
    except Exception as e:
        print(f"Error initializing generator: {e}")
        sys.exit(1)

    # Determine target properties
    target_properties = None

    if args.example:
        examples = get_property_examples()
        if args.example not in examples:
            print(f"Unknown example: {args.example}")
            print(f"Available examples: {list(examples.keys())}")
            sys.exit(1)
        target_properties = examples[args.example]
        print(f"Using example configuration: {args.example}")

    elif args.properties:
        try:
            target_properties = parse_properties(args.properties)
            print(f"Using specified properties: {target_properties}")
        except ValueError as e:
            print(f"Error parsing properties: {e}")
            sys.exit(1)

    elif args.ranges:
        try:
            property_ranges = parse_ranges(args.ranges)
            print(f"Using property ranges: {property_ranges}")
        except ValueError as e:
            print(f"Error parsing ranges: {e}")
            sys.exit(1)

    # Validate properties
    if target_properties:
        available_props = generator.get_available_properties()
        invalid_props = set(target_properties.keys()) - set(available_props.keys())
        if invalid_props:
            print(f"Warning: Unknown properties: {invalid_props}")
            print("Available properties:", list(available_props.keys()))

    # Use model's max_seq_len as default if not specified
    max_length = args.max_length
    if max_length is None:
        max_length = generator.model.max_seq_len
        print(f"Using model's max sequence length: {max_length}")

    # Validate max_length doesn't exceed model's capability
    if max_length > generator.model.max_seq_len:
        print(
            f"Warning: Requested max_length ({max_length}) exceeds model's maximum ({generator.model.max_seq_len})"
        )
        print(f"Reducing max_length to {generator.model.max_seq_len}")
        max_length = generator.model.max_seq_len

    # Generate sequences or structures
    try:
        if args.print_sequences:
            sequences = generator.generate_sequences(
                target_properties or (list(get_property_examples().values())[0]),
                num_samples=args.num_samples,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                batch_size=args.batch_size,
                max_length=max_length,
            )
            # Print a compact view of sequences
            for idx, s in enumerate(sequences):
                print(f"\nSequence {idx+1}:")
                print(f"  tokens: {s['tokens']}")
                # print(f"  types: {s['types']}")
                if args.print_decoded:
                    decoded = generator.decode_tokens(s["tokens"])
                    print(f"  decoded: {decoded}")
            print(f"\nPrinted {len(sequences)} sequences.")
            return
        elif args.ranges:
            structures = generator.generate_with_property_ranges(
                property_ranges,
                num_samples=args.num_samples,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                batch_size=args.batch_size,
                max_length=max_length,
            )
        else:
            structures = generator.generate(
                target_properties,
                num_samples=args.num_samples,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                batch_size=args.batch_size,
                max_length=max_length,
            )

    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Save results
    if structures:
        try:
            save_structures(structures, args.output, args.format)
            print("✓ Generation completed successfully!")
            print(f"Generated {len(structures)} structures")
            print(f"Saved to: {args.output}")

            # Print summary of generated structures
            print("\nGenerated structures summary:")
            print("-" * 40)
            for i, structure in enumerate(structures[:3]):  # Show first 3
                print(f"Structure {i+1}:")
                print(f"  ID: {structure.get('structure_id', 'N/A')}")
                print(f"  Composition: {structure.get('composition', [])}")
                print(f"  Space group: {structure.get('space_group', 'N/A')}")
                if structure.get("lattice"):
                    lattice = structure["lattice"]
                    a_val = lattice.get("a", "N/A")
                    b_val = lattice.get("b", "N/A")
                    c_val = lattice.get("c", "N/A")
                    print(f"  Lattice: a={a_val:.3f}, b={b_val:.3f}, c={c_val:.3f}")
                print(f"  Wyckoff positions: {structure.get('wyckoff_positions', [])}")

            if len(structures) > 3:
                print(f"... and {len(structures) - 3} more structures")

        except Exception as e:
            print(f"Error saving results: {e}")
            sys.exit(1)
    else:
        print("No structures were generated.")
        sys.exit(1)


if __name__ == "__main__":
    main()
