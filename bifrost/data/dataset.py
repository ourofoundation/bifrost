"""
PyTorch Dataset for BIFROST crystal structure training.

This module provides dataset classes for loading and processing crystal structure
data for training the BIFROST model.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import random
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging


class CrystalStructureDataset(Dataset):
    """
    PyTorch Dataset for crystal structures.

    This dataset handles loading crystal structure data and converting it
    to the format expected by BIFROST for training.
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        max_seq_len: int = 512,
        property_dropout: float = 0.3,
        property_removal: float = 0.1,
        curriculum_level: int = 0,
    ):
        """
        Initialize dataset.

        Args:
            data: List of structure dictionaries
            tokenizer: BIFROST tokenizer instance
            max_seq_len: Maximum sequence length
            property_dropout: Probability of dropping individual properties
            property_removal: Probability of masking properties entirely
            curriculum_level: Level of complexity (0=simple, 1=medium, 2=complex)
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.property_dropout = property_dropout
        self.property_removal = property_removal
        self.curriculum_level = curriculum_level

        # Logger (configured by application)
        self.logger = logging.getLogger(__name__)

        # Drop structures that contain invalid space-group/Wyckoff combinations
        self.data = self._filter_invalid_wyckoff(data)

        # Apply curriculum filtering
        self.data = self._apply_curriculum_filtering(self.data)

    def setup_logging(self):
        """Deprecated: logger configured by application."""
        self.logger = logging.getLogger(__name__)

    def _filter_invalid_wyckoff(
        self, data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove structures containing any Wyckoff labels not present in mapping.

        This prevents tokenizer errors like: "Wyckoff position 8g not found for space group 70".
        """
        mapping = getattr(self.tokenizer, "wyckoff_mapping", None)
        if not isinstance(mapping, dict) or not mapping:
            # If mapping unavailable, keep data unchanged
            return data

        filtered: List[Dict[str, Any]] = []
        removed: List[Dict[str, Any]] = []
        for structure in data:
            sg = int(structure.get("space_group", -1))
            wyckoff_positions = structure.get("wyckoff_positions", [])
            ok = True
            for pos in wyckoff_positions:
                wy = pos.get("wyckoff")
                key = f"{sg}_{wy}"
                if mapping.get(key) is None:
                    ok = False
                    break
            if ok:
                filtered.append(structure)
            else:
                removed.append(structure)
        self.logger.info(
            f"Removed {len(removed)} structures due to invalid Wyckoff positions"
        )
        return filtered

    def _apply_curriculum_filtering(
        self, data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply curriculum-based filtering to the dataset."""
        if self.curriculum_level == 0:
            # Simple structures: max 5 unique elements, max 5 Wyckoff sites
            filtered = []
            for structure in data:
                composition = structure.get("composition", [])
                wyckoff_positions = structure.get("wyckoff_positions", [])

                unique_elements = len(set(elem for elem, _ in composition))
                n_sites = len(wyckoff_positions)

                if unique_elements <= 5 and n_sites <= 5:
                    filtered.append(structure)
            return filtered

        elif self.curriculum_level == 1:
            # Medium complexity: max 10 unique elements, max 10 Wyckoff sites
            filtered = []
            for structure in data:
                composition = structure.get("composition", [])
                wyckoff_positions = structure.get("wyckoff_positions", [])

                unique_elements = len(set(elem for elem, _ in composition))
                n_sites = len(wyckoff_positions)

                if unique_elements <= 10 and n_sites <= 10:
                    filtered.append(structure)
            return filtered

        else:
            # Full complexity
            return data

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example."""
        structure = self.data[idx]

        # Apply property dropout and removal
        processed_structure = self._apply_property_dropout(structure.copy())

        # Encode to tokens
        tokens, token_types = self.tokenizer.encode_structure(processed_structure)

        # Truncate if too long
        if len(tokens) > self.max_seq_len:
            tokens = tokens[: self.max_seq_len]
            token_types = token_types[: self.max_seq_len]

        # Convert to tensors
        # Use float dtype to preserve continuous values; discrete handling is done in the model
        input_tokens = torch.tensor(tokens[:-1], dtype=torch.float)  # Remove last token
        target_tokens = torch.tensor(tokens[1:], dtype=torch.float)  # Shift by 1
        token_types = torch.tensor(token_types[:-1], dtype=torch.long)

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones_like(input_tokens)

        return {
            "input_tokens": input_tokens,
            "target_tokens": target_tokens,
            "token_types": token_types,
            "attention_mask": attention_mask,
            "structure_id": structure.get("structure_id", f"struct_{idx}"),
        }

    def _apply_property_dropout(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Apply property dropout and masking during training."""
        if "properties" not in structure:
            return structure

        processed_properties = {}
        available_props = list(structure["properties"].keys())

        for prop_name in available_props:
            if random.random() < self.property_dropout:
                # Keep this property
                if random.random() > self.property_removal:
                    # Use actual value
                    processed_properties[prop_name] = structure["properties"][prop_name]
                else:
                    # Mask this property (set to None to trigger masking)
                    processed_properties[prop_name] = None
            # Otherwise drop this property entirely

        structure["properties"] = processed_properties
        return structure

    def get_property_statistics(self) -> Dict[str, Any]:
        """Get statistics about properties in the dataset."""
        stats = {
            "total_structures": len(self.data),
            "properties_per_structure": {},
            "property_values": {},
        }

        # Count properties
        prop_counts = {}
        prop_values = {}

        for structure in self.data:
            if "properties" in structure:
                for prop_name, value in structure["properties"].items():
                    prop_counts[prop_name] = prop_counts.get(prop_name, 0) + 1
                    if prop_name not in prop_values:
                        prop_values[prop_name] = []
                    prop_values[prop_name].append(value)

        stats["properties_per_structure"] = prop_counts

        # Calculate value statistics
        for prop_name, values in prop_values.items():
            stats["property_values"][prop_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }

        return stats


class PropertyConditionedDataset(CrystalStructureDataset):
    """
    Dataset variant that ensures certain properties are always present.

    This is useful for generation tasks where specific property conditioning
    is required.
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        required_properties: Dict[str, float],
        max_seq_len: int = 512,
        **kwargs,
    ):
        """
        Initialize with required properties.

        Args:
            data: List of structure dictionaries
            tokenizer: BIFROST tokenizer instance
            required_properties: Properties that must be present in every example
            max_seq_len: Maximum sequence length
        """
        self.required_properties = required_properties
        super().__init__(data, tokenizer, max_seq_len, **kwargs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get example with guaranteed property conditioning."""
        structure = self.data[idx].copy()

        # Ensure required properties are present
        if "properties" not in structure:
            structure["properties"] = {}

        for prop_name, prop_value in self.required_properties.items():
            structure["properties"][prop_name] = prop_value

        # Apply standard processing
        return super().__getitem__(idx)


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create PyTorch DataLoader for the dataset.

    Args:
        dataset: CrystalStructureDataset instance
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory (for GPU training)

    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_batch,
    )


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader to handle variable-length sequences.

    Args:
        batch: List of individual examples

    Returns:
        Batched tensors with appropriate padding
    """
    # Find maximum sequence length in batch
    max_len = max(len(item["input_tokens"]) for item in batch)

    # Initialize batch tensors
    batch_size = len(batch)
    # Preserve float dtype for tokens to support continuous values
    input_tokens = torch.zeros(batch_size, max_len, dtype=torch.float)
    target_tokens = torch.zeros(batch_size, max_len, dtype=torch.float)
    token_types = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    structure_ids = []

    for i, item in enumerate(batch):
        seq_len = len(item["input_tokens"])

        # Copy sequences
        input_tokens[i, :seq_len] = item["input_tokens"]
        target_tokens[i, :seq_len] = item["target_tokens"]
        token_types[i, :seq_len] = item["token_types"]
        attention_mask[i, :seq_len] = item["attention_mask"]

        structure_ids.append(item["structure_id"])

    return {
        "input_tokens": input_tokens,
        "target_tokens": target_tokens,
        "token_types": token_types,
        "attention_mask": attention_mask,
        "structure_ids": structure_ids,
    }


def load_sample_dataset(filepath: str) -> List[Dict[str, Any]]:
    """
    Load sample dataset from file (placeholder for actual data loading).

    Args:
        filepath: Path to dataset file

    Returns:
        List of structure dictionaries
    """
    # This is a placeholder - in practice, you would load from
    # Materials Project, Alexandria, or other crystal databases

    # Example structure format
    sample_structures = [
        {
            "structure_id": "mp-1234",
            "composition": [("Li", 1), ("Fe", 1), ("P", 1), ("O", 4)],
            "space_group": 62,
            "wyckoff_positions": [
                {"element": "Li", "wyckoff": "4c", "coordinates": [0.0, 0.0, 0.0]},
                {"element": "Fe", "wyckoff": "4c", "coordinates": [0.28, 0.25, 0.97]},
                {"element": "P", "wyckoff": "4c", "coordinates": [0.09, 0.25, 0.42]},
                {"element": "O", "wyckoff": "4c", "coordinates": [0.10, 0.25, 0.74]},
            ],
            "lattice": {
                "a": 10.3,
                "b": 6.0,
                "c": 4.7,
                "alpha": 90,
                "beta": 90,
                "gamma": 90,
            },
            "properties": {
                "band_gap": 2.5,
                "density": 3.2,
                "energy_above_hull": 0.02,
                "formation_energy_per_atom": -2.5,
            },
        }
    ]

    return sample_structures
