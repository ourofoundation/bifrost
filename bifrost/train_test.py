#!/usr/bin/env python3
"""
BIFROST Training Demo

This script demonstrates training the BIFROST model with sample Materials Project data.
"""

import torch
import random
from typing import Dict, Any, List

# Import BIFROST components
from bifrost.model import BIFROST, create_bifrost_model, get_bifrost_config
from bifrost.data.tokenizer import tokenizer
from bifrost.data.dataset import CrystalStructureDataset, create_dataloader
from bifrost.data.properties import discretize_structure_properties
from bifrost.training import create_trainer, get_training_config


def create_sample_data(num_samples: int = 50) -> List[Dict[str, Any]]:
    """Create sample Materials Project-style data."""
    print(f"Creating {num_samples} sample structures...")

    # Sample structures
    sample_structures = [
        {
            "structure_id": "mp-1234",
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
            "properties": {
                "bandgap": 3.5,
                "density": 3.5,
                "ehull": 0.05,
                "formation_energy": -2.3,
            },
        },
        {
            "structure_id": "mp-5678",
            "composition": [("Na", 1), ("Cl", 1)],
            "space_group": 225,
            "wyckoff_positions": [
                {"element": "Na", "wyckoff": "4a", "coordinates": [0.0, 0.0, 0.0]},
                {"element": "Cl", "wyckoff": "4b", "coordinates": [0.5, 0.5, 0.5]},
            ],
            "lattice": {
                "a": 5.64,
                "b": 5.64,
                "c": 5.64,
                "alpha": 90.0,
                "beta": 90.0,
                "gamma": 90.0,
            },
            "properties": {
                "bandgap": 5.2,
                "density": 2.16,
                "ehull": 0.0,
                "formation_energy": -4.1,
            },
        },
    ]

    # Generate additional structures
    structures = []
    for i in range(num_samples):
        base_structure = random.choice(sample_structures)
        variation = {
            "bandgap": max(
                0, base_structure["properties"]["bandgap"] + random.gauss(0, 0.5)
            ),
            "density": max(
                0, base_structure["properties"]["density"] + random.gauss(0, 0.2)
            ),
            "ehull": max(
                0, base_structure["properties"]["ehull"] + random.gauss(0, 0.1)
            ),
            "formation_energy": base_structure["properties"]["formation_energy"]
            + random.gauss(0, 0.3),
        }

        structure = base_structure.copy()
        structure["structure_id"] = f"mp-{i+10000}"
        structure["properties"] = variation
        structures.append(structure)

    print(f"✓ Created {len(structures)} structures")
    return structures


def main():
    """Main training demo."""
    print("=" * 60)
    print("BIFROST Training Demo")
    print("=" * 60)

    # Setup
    print("Setting up training...")

    # Create model
    config = get_bifrost_config("small")
    # Set vocab size to tokenizer vocab to reflect updated wyckoff/lattice tokens
    config["vocab_size"] = tokenizer.get_vocab_size()
    model = create_bifrost_model(config)
    print(f"✓ Created model with {model.get_num_parameters():,} parameters")

    # Create data
    structures = create_sample_data(200)
    random.shuffle(structures)

    train_structures = structures[:160]  # 80% for training
    val_structures = structures[160:180]  # 10% for validation
    test_structures = structures[180:]  # 10% for testing

    # Create datasets
    train_dataset = CrystalStructureDataset(
        train_structures, tokenizer, max_seq_len=512
    )

    val_dataset = CrystalStructureDataset(val_structures, tokenizer, max_seq_len=512)

    print(f"✓ Created datasets: {len(train_dataset)} train, {len(val_dataset)} val")

    # Create dataloaders
    train_dataloader = create_dataloader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = create_dataloader(val_dataset, batch_size=8, shuffle=False)

    # Create trainer
    training_config = get_training_config("debug")
    trainer = create_trainer(model, train_dataloader, val_dataloader, training_config)
    print("✓ Created trainer")

    # Train for a few epochs
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    try:
        results = trainer.train(num_epochs=15, save_interval=5, eval_interval=5)

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)

        if results and "final_stats" in results:
            final_stats = results["final_stats"]
            print(f"Final loss: {final_stats['overall_stats']['final_loss']:.4f}")

        print("\nBIFROST model is ready for crystal structure generation!")
        print("You can now:")
        print("- Generate structures with property conditioning")
        print("- Fine-tune on larger datasets")
        print("- Use for materials discovery research")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
