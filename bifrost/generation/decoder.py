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
        """Delegate to tokenizer's decode_structure for backward compatibility."""
        try:
            return self.tokenizer.decode_structure(
                tokens, token_types, max_elements=max_elements
            )
        except TypeError:
            # Fallback if signature changes
            return self.tokenizer.decode_structure(tokens, token_types)

    def _clean_sequence(
        self, tokens: np.ndarray, token_types: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Deprecated: tokenizer handles cleaning; kept for API compatibility."""
        return tokens, token_types

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
        """Deprecated: tokenizer provides implementation."""
        try:
            return float(token_val)
        except Exception:
            return 0.0

    def _validate_structure(self, structure: Dict[str, Any]) -> bool:
        """Deprecated: tokenizer validates; keep minimal behavior."""
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
