#!/usr/bin/env python3
"""
BIFROST Training Demo

This script demonstrates training the BIFROST model with sample Materials Project data.
"""

import torch
import json
import random
from typing import Dict, Any, List
import os

# Import BIFROST components
from bifrost.model import BIFROST, create_bifrost_model
from bifrost.data.tokenizer import tokenizer
from bifrost.data.dataset import CrystalStructureDataset, create_dataloader
from bifrost.data.properties import discretize_structure_properties
from bifrost.training import create_trainer, get_training_config
from bifrost.config import create_model_config


def main():
    """Main training demo."""
    print("=" * 60)
    print("BIFROST Training Demo")
    print("=" * 60)

    # Setup
    print("Setting up training...")

    # Create model
    config = create_model_config("small")
    max_seq_len = config["max_seq_len"]
    # create_model_config
    # Set vocab size to tokenizer vocab to reflect updated wyckoff/lattice tokens
    config["vocab_size"] = tokenizer.get_vocab_size()
    model = create_bifrost_model(config)
    print(f"✓ Created model with {model.get_num_parameters():,} parameters")

    # Create datasets
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "data/mp/mp_dataset.json"), "r") as f:
        structures = json.load(f)

    len_structures = len(structures)
    print(f"✓ Found {len_structures} structures")
    train_structures = structures[0 : int(len_structures * 0.8)]  # 80% for training
    val_structures = structures[-int(len_structures * 0.2) :]  # 20% for validation

    train_dataset = CrystalStructureDataset(
        train_structures, tokenizer, max_seq_len=max_seq_len
    )
    val_dataset = CrystalStructureDataset(
        val_structures, tokenizer, max_seq_len=max_seq_len
    )

    print(f"✓ Created datasets: {len(train_dataset)} train, {len(val_dataset)} val")
    print(f"✓ Max sequence length: {max_seq_len}")

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
        results = trainer.train(num_epochs=10, save_interval=5, eval_interval=5)

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)

        if results and "final_stats" in results:
            final_stats = results["final_stats"]
            print(f"Final loss: {final_stats['overall_stats']['final_loss']:.4f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
