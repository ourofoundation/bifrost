"""
Structure Decoder.

Converts token sequences back to crystal structures.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import re


class StructureDecoder:
    """
    Decodes token sequences into crystal structures.

    This class handles the conversion from BIFROST token sequences back to
    structured crystal representations.
    """

    def __init__(self, tokenizer):
        """
        Initialize decoder.

        Args:
            tokenizer: BIFROST tokenizer instance
        """
        self.tokenizer = tokenizer
        self.reverse_vocab = {v: k for k, v in tokenizer.vocab.items()}

        # Token type mappings (reverse of tokenizer.token_types)
        self.type_names = {
            0: "PROPERTY",
            1: "ELEMENT",
            2: "COUNT",
            3: "SPACEGROUP",
            4: "WYCKOFF",
            5: "COORDINATE",
            6: "LATTICE",
        }

    def decode_structure(
        self, tokens: np.ndarray, token_types: np.ndarray, max_elements: int = 20
    ) -> Optional[Dict[str, Any]]:
        """
        Decode token sequence into crystal structure.

        Args:
            tokens: Token ID sequence
            token_types: Token type sequence
            max_elements: Maximum number of elements to consider

        Returns:
            Dictionary representing crystal structure, or None if decoding fails
        """
        # Remove special tokens and padding
        tokens, token_types = self._clean_sequence(tokens, token_types)

        if len(tokens) == 0:
            return None

        # Parse different sections of the sequence
        structure = {
            "structure_id": f"generated_{hash(tuple(tokens)) % 10000}",
            "composition": [],
            "wyckoff_positions": [],
            "lattice": {},
            "properties": {},
            "space_group": None,
        }

        # Heuristic parsing independent of token_types, using vocab prefixes and continuous values
        def is_continuous_value(val: float) -> bool:
            as_int = int(val)
            return abs(val - as_int) > 1e-6 or (as_int not in self.reverse_vocab)

        i = 0
        last_wyck: Optional[str] = None
        while i < len(tokens):
            raw = tokens[i]
            # Continuous values are handled later
            if is_continuous_value(raw):
                i += 1
                continue

            tid = int(raw)
            tname = self.reverse_vocab.get(tid, "UNK")

            if tname.startswith("SPACE_"):
                try:
                    structure["space_group"] = int(tname.split("_")[1])
                except Exception:
                    pass
                i += 1
                continue

            if tname.startswith("COUNT_") and structure["composition"]:
                try:
                    count = int(tname.split("_")[1])
                    elem, _ = structure["composition"][-1]
                    structure["composition"][-1] = (elem, count)
                except Exception:
                    pass
                i += 1
                continue

            if tname.startswith("WYCK_"):
                last_wyck = tname.replace("WYCK_", "")
                i += 1
                # Element + 3 coords
                if i < len(tokens) and not is_continuous_value(tokens[i]):
                    elem_name = self.reverse_vocab.get(int(tokens[i]), "UNK")
                    coords: List[float] = []
                    j = i + 1
                    while j < len(tokens) and len(coords) < 3:
                        if is_continuous_value(tokens[j]):
                            coords.append(self._decode_continuous_token(tokens[j]))
                        j += 1
                    if elem_name != "UNK" and len(coords) == 3:
                        structure["wyckoff_positions"].append(
                            {
                                "element": elem_name,
                                "wyckoff": last_wyck,
                                "coordinates": coords,
                            }
                        )
                    i = j
                    continue
                continue

            # Element symbol
            if tname.isalpha() and len(tname) <= 2:
                structure["composition"].append((tname, 1))
                i += 1
                continue

            i += 1

        # Lattice: take last 6 continuous values as a,b,c,alpha,beta,gamma
        cont_vals = [
            self._decode_continuous_token(v) for v in tokens if is_continuous_value(v)
        ]
        if len(cont_vals) >= 6:
            a, b, c, alpha, beta, gamma = cont_vals[-6:]
            structure["lattice"] = {
                "a": float(a),
                "b": float(b),
                "c": float(c),
                "alpha": float(alpha),
                "beta": float(beta),
                "gamma": float(gamma),
            }

        # Validate structure
        if not self._validate_structure(structure):
            return None

        return structure

    def _clean_sequence(
        self, tokens: np.ndarray, token_types: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove special tokens and padding from sequence."""
        # Remove padding and special tokens
        special_tokens = [
            self.tokenizer.special_tokens["PAD"],
            self.tokenizer.special_tokens["UNK"],
            self.tokenizer.special_tokens["EOS"],
            self.tokenizer.special_tokens["BOS"],
        ]

        mask = ~np.isin(tokens, special_tokens)
        cleaned_tokens = tokens[mask]
        cleaned_types = (
            token_types[mask]
            if len(token_types) == len(tokens)
            else np.zeros_like(cleaned_tokens)
        )

        return cleaned_tokens, cleaned_types

    def _parse_wyckoff_section(
        self, tokens: np.ndarray, token_types: np.ndarray, start_idx: int
    ) -> List[Dict[str, Any]]:
        """Deprecated: handled in decode_structure heuristics."""
        return []

    def _parse_lattice_section(
        self, tokens: np.ndarray, token_types: np.ndarray, start_idx: int
    ) -> Dict[str, float]:
        """Deprecated: handled in decode_structure heuristics."""
        return {}

    def _decode_continuous_token(self, token_val: float) -> float:
        """Convert continuous token back to float (identity for now)."""
        try:
            return float(token_val)
        except Exception:
            return 0.0

    def _validate_structure(self, structure: Dict[str, Any]) -> bool:
        """Lenient validity: require at least non-empty composition."""
        return bool(structure.get("composition"))

    def decode_batch(
        self, batch_tokens: torch.Tensor, batch_types: torch.Tensor, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Decode batch of token sequences.

        Args:
            batch_tokens: Batch of token sequences (batch_size, seq_len)
            batch_types: Batch of token type sequences (batch_size, seq_len)
            **kwargs: Additional arguments for decode_structure

        Returns:
            List of decoded structures
        """
        structures = []

        for i in range(batch_tokens.size(0)):
            tokens = batch_tokens[i].cpu().numpy()
            types = batch_types[i].cpu().numpy()

            structure = self.decode_structure(tokens, types, **kwargs)
            if structure:
                structures.append(structure)

        return structures
