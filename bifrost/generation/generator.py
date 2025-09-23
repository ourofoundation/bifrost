"""
BIFROST Generator.

High-level interface for generating crystal structures with property conditioning.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

from ..model import BIFROST, create_bifrost_model
from ..data.tokenizer import tokenizer
from ..data.properties import discretize_structure_properties, get_property_bins
from ..config import create_model_config


class BIFROSTGenerator:
    """
    High-level generator for crystal structures with property conditioning.

    This class provides an easy-to-use interface for generating crystal structures
    using trained BIFROST models, with support for conditioning on target properties.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        model_config: str = "small",
        device: Optional[str] = None,
    ):
        """
        Initialize BIFROST generator.

        Args:
            model_path: Path to trained model checkpoint (optional)
            model_config: Model configuration name ('small', 'medium', 'large')
            device: Device to run generation on ('cpu', 'cuda', 'mps')
        """
        # Setup device
        self.device = device or torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        print(f"Using device: {self.device}")

        # Create model
        config = create_model_config(model_config)
        # Always align vocab size with tokenizer so Wyckoff/lattice vocab matches
        config["vocab_size"] = tokenizer.get_vocab_size()

        self.model = create_bifrost_model(config)
        self.model.to(self.device)
        self.model.eval()

        # Load checkpoint if provided
        if model_path:
            self.load_checkpoint(model_path)

        # Configure type-conditioned vocab mask for decoding
        try:
            self._configure_type_token_mask()
        except Exception as e:
            print(f"Warning: failed to configure type token mask: {e}")

        # Get property information
        self.property_bins = get_property_bins()

        print(
            f"✓ BIFROST Generator initialized with {self.model.get_num_parameters():,} parameters"
        )

    def _configure_type_token_mask(self) -> None:
        """Build and set a [7, vocab_size] boolean mask for type-conditioned decoding.

        Rows 0..4 (PROPERTY, ELEMENT, COUNT, SPACEGROUP, WYCKOFF) specify which discrete
        vocab ids are allowed when that type is predicted. Rows 5..6 (COORDINATE, LATTICE)
        are not used for discrete sampling and are left all False.
        """
        vocab = tokenizer.vocab
        reverse_vocab = {v: k for k, v in vocab.items()}
        vocab_size = len(vocab)

        # Precompute property tokens from bins
        prop_bins = get_property_bins()
        property_token_names = set()
        for info in prop_bins.values():
            for t in info.get("tokens", []):
                property_token_names.add(t)

        # Build sets by pattern
        wyck_ids = set()
        space_ids = set()
        count_ids = set()
        property_ids = set()
        element_ids = set()

        special_ids = set(tokenizer.special_tokens.values())

        for tok_id, name in reverse_vocab.items():
            if tok_id in special_ids:
                continue
            if name.startswith("WYCK_"):
                wyck_ids.add(tok_id)
            elif name.startswith("SPACE_"):
                space_ids.add(tok_id)
            elif name.startswith("COUNT_"):
                count_ids.add(tok_id)
            elif name in property_token_names:
                property_ids.add(tok_id)
            else:
                # Treat all other known discrete tokens as elements
                element_ids.add(tok_id)

        import torch as _torch

        mask = _torch.zeros(7, vocab_size, dtype=_torch.bool, device=self.device)
        # PROPERTY=0, ELEMENT=1, COUNT=2, SPACEGROUP=3, WYCKOFF=4
        if property_ids:
            mask[0, list(property_ids)] = True
        if element_ids:
            mask[1, list(element_ids)] = True
        if count_ids:
            mask[2, list(count_ids)] = True
        if space_ids:
            mask[3, list(space_ids)] = True
        if wyck_ids:
            mask[4, list(wyck_ids)] = True

        # Set on model heads (will be respected during sampling)
        try:
            self.model.heads.set_type_token_mask(mask)
        except Exception:
            # Backwards compatibility if heads do not implement setter
            if hasattr(self.model.heads, "type_token_mask"):
                self.model.heads.type_token_mask = mask

    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """
        Load trained model checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✓ Loaded checkpoint from {checkpoint_path}")

    def create_property_prefix(
        self, target_properties: Dict[str, Union[float, str]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create property prefix tokens for generation.

        Args:
            target_properties: Dict of property_name -> value/bin_name

        Returns:
            Tuple of (prefix_tokens, prefix_types)
        """
        # Convert property values to tokens
        property_tokens = []
        property_types = []

        for prop_name, value in target_properties.items():
            if prop_name not in self.property_bins:
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

        # Append separator token to indicate end of property prefix
        sep_id = tokenizer.special_tokens.get("SEP")
        if sep_id is not None:
            property_tokens.append(sep_id)
            property_types.append(
                tokenizer.token_types["PROPERTY"]
            )  # treat as discrete

        # Convert to tensors
        prefix_tokens = torch.tensor(
            [property_tokens], dtype=torch.long, device=self.device
        )
        prefix_types = torch.tensor(
            [property_types], dtype=torch.long, device=self.device
        )

        return prefix_tokens, prefix_types

    def generate(
        self,
        target_properties: Dict[str, Union[float, str]],
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        num_samples: int = 1,
        batch_size: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Generate crystal structures with property conditioning.

        Args:
            target_properties: Target properties to condition generation on
            max_length: Maximum sequence length for generation (defaults to model's max_seq_len)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            num_samples: Number of structures to generate
            batch_size: Batch size for generation

        Returns:
            List of generated crystal structures
        """
        # Use model's max_seq_len as default
        if max_length is None:
            max_length = self.model.max_seq_len

        # Validate max_length doesn't exceed model's capability
        if max_length > self.model.max_seq_len:
            print(
                f"Warning: Requested max_length ({max_length}) exceeds model's maximum ({self.model.max_seq_len})"
            )
            print(f"Reducing max_length to {self.model.max_seq_len}")
            max_length = self.model.max_seq_len

        print("Generating crystal structures...")
        print(f"Target properties: {target_properties}")

        # Create property prefix
        prefix_tokens, prefix_types = self.create_property_prefix(target_properties)

        generated_structures = []

        # Generate in batches
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            current_batch_size = batch_end - batch_start

            print(f"Generating batch {batch_start + 1}-{batch_end}/{num_samples}")

            # Expand prefix for batch
            batch_prefix_tokens = prefix_tokens.repeat(current_batch_size, 1)
            batch_prefix_types = prefix_types.repeat(current_batch_size, 1)

            # Generate token sequences
            with torch.no_grad():
                generated_tokens, generated_types = self.model.generate(
                    batch_prefix_tokens,
                    batch_prefix_types,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )

            # Decode each structure in the batch
            for i in range(current_batch_size):
                tokens = generated_tokens[i].cpu().numpy()
                types = generated_types[i].cpu().numpy()

                try:
                    structure = tokenizer.decode_structure(tokens, types)
                    if structure:
                        structure["generated_properties"] = target_properties.copy()
                        # Attach raw and decoded sequences, including predicted token types
                        try:
                            structure["sequence_tokens"] = tokens.tolist()
                            structure["sequence_types"] = types.astype(int).tolist()
                            # Human-readable token names for discrete entries
                            structure["decoded_tokens"] = self.decode_tokens(
                                tokens.tolist()
                            )
                            # Map type ids to names using tokenizer mapping
                            type_id_to_name = {
                                v: k for k, v in tokenizer.token_types.items()
                            }
                            structure["sequence_type_names"] = [
                                type_id_to_name.get(int(t), str(int(t))) for t in types
                            ]
                        except Exception:
                            pass
                        generated_structures.append(structure)
                except Exception as e:
                    print(f"Warning: Failed to decode structure: {e}")
                    continue

        print(f"✓ Generated {len(generated_structures)}/{num_samples} structures")
        return generated_structures

    def generate_sequences(
        self,
        target_properties: Dict[str, Union[float, str]],
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        num_samples: int = 1,
        batch_size: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Generate raw token/type sequences with property conditioning.

        Args:
            target_properties: Target properties to condition generation on
            max_length: Maximum sequence length for generation (defaults to model's max_seq_len)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            num_samples: Number of sequences to generate
            batch_size: Batch size for generation

        Returns:
            List of dicts with keys: 'tokens' (List[float]) and 'types' (List[int])
        """
        # Use model's max_seq_len as default
        if max_length is None:
            max_length = self.model.max_seq_len

        # Validate max_length doesn't exceed model's capability
        if max_length > self.model.max_seq_len:
            print(
                f"Warning: Requested max_length ({max_length}) exceeds model's maximum ({self.model.max_seq_len})"
            )
            print(f"Reducing max_length to {self.model.max_seq_len}")
            max_length = self.model.max_seq_len
        print("Generating raw sequences...")
        print(f"Target properties: {target_properties}")

        prefix_tokens, prefix_types = self.create_property_prefix(target_properties)

        sequences: List[Dict[str, Any]] = []

        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            current_batch_size = batch_end - batch_start

            print(f"Sampling batch {batch_start + 1}-{batch_end}/{num_samples}")

            batch_prefix_tokens = prefix_tokens.repeat(current_batch_size, 1)
            batch_prefix_types = prefix_types.repeat(current_batch_size, 1)

            with torch.no_grad():
                generated_tokens, generated_types = self.model.generate(
                    batch_prefix_tokens,
                    batch_prefix_types,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )

            for i in range(current_batch_size):
                tok = generated_tokens[i].detach().cpu().numpy().tolist()
                typ = generated_types[i].detach().cpu().numpy().tolist()
                # Also include human-readable names for types and tokens
                try:
                    type_id_to_name = {v: k for k, v in tokenizer.token_types.items()}
                    sequences.append(
                        {
                            "tokens": tok,
                            "types": typ,
                            "decoded_tokens": self.decode_tokens(tok),
                            "type_names": [
                                type_id_to_name.get(int(t), str(int(t))) for t in typ
                            ],
                        }
                    )
                except Exception:
                    sequences.append({"tokens": tok, "types": typ})

        print(f"✓ Sampled {len(sequences)} sequences")
        return sequences

    def decode_tokens(self, tokens: List[float]) -> List[str]:
        """
        Convert token ids (possibly floats for continuous) to human-readable strings.

        Discrete ids are mapped using the tokenizer's reverse vocab; continuous
        values are formatted as floats.
        """
        reverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        decoded: List[str] = []
        for val in tokens:
            try:
                int_val = int(val)
                if abs(val - int_val) < 1e-6 and int_val in reverse_vocab:
                    decoded.append(reverse_vocab[int_val])
                elif abs(val - int_val) < 1e-6:
                    decoded.append(str(int_val))
                else:
                    decoded.append(f"{float(val):.4f}")
            except Exception:
                decoded.append(str(val))
        return decoded

    def generate_with_property_ranges(
        self,
        property_ranges: Dict[str, Tuple[float, float]],
        num_samples: int = 10,
        **generation_kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate structures with properties sampled from ranges.

        Args:
            property_ranges: Dict of property_name -> (min, max) ranges
            num_samples: Number of structures to generate
            **generation_kwargs: Additional generation arguments

        Returns:
            List of generated structures with sampled properties
        """
        print(
            f"Generating {num_samples} structures with property ranges: {property_ranges}"
        )

        generated_structures = []

        for i in range(num_samples):
            # Sample properties from ranges
            target_properties = {}
            for prop_name, (min_val, max_val) in property_ranges.items():
                if prop_name not in self.property_bins:
                    continue

                value = np.random.uniform(min_val, max_val)
                target_properties[prop_name] = value

            # Generate structure with these properties
            structures = self.generate(
                target_properties, num_samples=1, **generation_kwargs
            )

            if structures:
                generated_structures.extend(structures)

        return generated_structures

    def get_available_properties(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available properties for conditioning."""
        return self.property_bins.copy()

    def get_property_examples(self) -> Dict[str, List[str]]:
        """Get example property values for each property type."""
        examples = {}

        for prop_name, prop_info in self.property_bins.items():
            examples[prop_name] = {
                "description": prop_info["description"],
                "thresholds": prop_info["thresholds"],
                "example_bins": prop_info["tokens"],
                "units": self._get_property_units(prop_name),
            }

        return examples

    def _get_property_units(self, prop_name: str) -> str:
        """Get units for a property."""
        units = {
            "band_gap": "eV",
            "density": "g/cm³",
            "energy_above_hull": "eV/atom",
            "bulk_modulus": "GPa",
            "efermi": "eV",
            "total_magnetization": "μB/atom",
            "formation_energy_per_atom": "eV/atom",
            "shear_modulus": "GPa",
        }
        return units.get(prop_name, "unknown")
