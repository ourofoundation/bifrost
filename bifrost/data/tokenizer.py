"""
Tokenizer for BIFROST crystal structure sequences.

This module handles the conversion between crystal structures and token sequences,
including vocabulary management for discrete tokens and handling of continuous values.
"""

import json
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np


class BIFROSTTokenizer:
    """
    Tokenizer for converting crystal structures to/from token sequences.

    The tokenizer handles:
    - Discrete tokens (elements, space groups, Wyckoff positions, property bins)
    - Continuous values (coordinates, lattice parameters)
    - Token type classification (discrete vs continuous)
    - Sequence construction and parsing
    """

    def __init__(self, vocab_file: Optional[str] = None):
        """Initialize tokenizer with vocabulary."""
        # Define special tokens
        self.special_tokens = {
            "PAD": 0,
            "UNK": 1,
            "MASK": 2,
            "SEP": 3,
            "EOS": 4,
            "BOS": 5,
        }

        # Token type mappings
        self.token_types = {
            "PROPERTY": 0,  # Property bin tokens (BANDGAP_MED, etc.)
            "ELEMENT": 1,  # Chemical elements
            "COUNT": 2,  # Stoichiometric counts (1-20)
            "SPACEGROUP": 3,  # Space group numbers
            "WYCKOFF": 4,  # Wyckoff position labels
            "COORDINATE": 5,  # x,y,z coordinates
            "LATTICE": 6,  # a,b,c,α,β,γ lattice parameters
        }

        # Initialize vocabulary
        self.vocab = self._build_vocabulary()
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        # Continuous value ranges
        self.continuous_ranges = {
            "coordinate": (0.0, 1.0),  # x,y,z coordinates
            "lattice_length": (1.0, 100.0),  # a,b,c in Angstroms
            "lattice_angle": (30.0, 150.0),  # α,β,γ in degrees
        }

        # Optional Wyckoff mapping (space_group + label -> orbit-id)
        # Loaded from wyckoff_map_sg_letter_to_orbit.json if available
        self.wyckoff_mapping: Dict[str, str] = {}
        self.sg_allowed_orbit_token_ids: Dict[int, set] = {}
        try:
            import os as _os
            import json as _json

            def _resolve_data_file(filename: str) -> Optional[str]:
                here = _os.path.dirname(__file__)
                cand1 = _os.path.join(here, filename)
                if _os.path.exists(cand1):
                    return cand1
                # fallback: project root two levels up
                cand2 = _os.path.join(here, "..", "..", filename)
                if _os.path.exists(cand2):
                    return cand2
                return None

            mapping_path = _resolve_data_file("wyckoff_map_sg_letter_to_orbit.json")
            if mapping_path:
                with open(mapping_path, "r") as f:
                    self.wyckoff_mapping = _json.load(f)
        except Exception:
            self.wyckoff_mapping = {}

    def _build_vocabulary(self) -> Dict[str, int]:
        """Build the complete vocabulary dictionary."""
        vocab = self.special_tokens.copy()

        # Add elements (H to Og, 103 elements)
        elements = [
            "H",
            "He",
            "Li",
            "Be",
            "B",
            "C",
            "N",
            "O",
            "F",
            "Ne",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "Cl",
            "Ar",
            "K",
            "Ca",
            "Sc",
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Zn",
            "Ga",
            "Ge",
            "As",
            "Se",
            "Br",
            "Kr",
            "Rb",
            "Sr",
            "Y",
            "Zr",
            "Nb",
            "Mo",
            "Tc",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Cd",
            "In",
            "Sn",
            "Sb",
            "Te",
            "I",
            "Xe",
            "Cs",
            "Ba",
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
            "Hf",
            "Ta",
            "W",
            "Re",
            "Os",
            "Ir",
            "Pt",
            "Au",
            "Hg",
            "Tl",
            "Pb",
            "Bi",
            "Po",
            "At",
            "Rn",
            "Fr",
            "Ra",
            "Ac",
            "Th",
            "Pa",
            "U",
            "Np",
            "Pu",
            "Am",
            "Cm",
            "Bk",
            "Cf",
            "Es",
            "Fm",
            "Md",
            "No",
            "Lr",
            "Rf",
            "Db",
            "Sg",
            "Bh",
            "Hs",
            "Mt",
            "Ds",
            "Rg",
            "Cn",
            "Nh",
            "Fl",
            "Mc",
            "Lv",
            "Ts",
            "Og",
        ]
        next_id = max(vocab.values()) + 1 if vocab else 0
        for i, elem in enumerate(elements, next_id):
            vocab[elem] = i

        # Add stoichiometric counts (1-20)
        next_id = max(vocab.values()) + 1
        for i in range(1, 21):
            vocab[f"COUNT_{i}"] = next_id + i - 1

        # Add space groups (1-230)
        next_id = max(vocab.values()) + 1
        for i in range(1, 231):
            vocab[f"SPACE_{i}"] = next_id + i - 1

        # Add Wyckoff positions (comprehensive set)
        # Load complete Wyckoff data from generated file
        try:
            import json
            import os

            # Try to load from complete_token_data.json if it exists
            here = os.path.dirname(__file__)
            orbits_token_data_path = os.path.join(
                here, "wyckoff_orbits_token_data.json"
            )
            if os.path.exists(orbits_token_data_path):
                with open(orbits_token_data_path, "r") as f:
                    token_data = json.load(f)

                # tokens with prefix ORBIT_
                wyckoff_positions = token_data.get("wyckoff_positions", [])

                # Silent load; counts available if needed for debugging

                # Add Wyckoff positions
                next_id = max(vocab.values()) + 1 if vocab else 0
                for i, wyck in enumerate(wyckoff_positions, next_id):
                    vocab[f"WYCK_{wyck}"] = i

                # Note: coordinate and lattice tokens are deprecated and not loaded

            else:
                # Raise an error
                raise FileNotFoundError(
                    "wyckoff_orbits_token_data.json not found, please run generate_wyckoff_data.py"
                )

        except Exception as e:
            print(f"Error loading token data: {e}")
            raise e

        # Add property tokens (from properties.py)
        from .properties import get_property_bins

        property_bins = get_property_bins()
        for prop_config in property_bins.values():
            for token in prop_config["tokens"]:
                if token not in vocab:
                    next_id = max(vocab.values()) + 1 if vocab else 0
                    vocab[token] = next_id

        return vocab

        # Note: Unreachable; kept for structure

    def encode_structure(
        self, structure_data: Dict[str, Any]
    ) -> Tuple[List[int], List[int]]:
        """
        Convert structure data to token sequence.

        Args:
            structure_data: Dictionary containing structure information
                - composition: List[Tuple[element, count]]
                - space_group: int
                - wyckoff_positions: List[Dict] with element, wyckoff, coords
                - lattice: Dict with a, b, c, alpha, beta, gamma
                - properties: Dict of property_name -> value (optional)

        Returns:
            Tuple of (tokens, token_types) where tokens are encoded as integers
            and token_types indicate 0=discrete, 1=continuous
        """
        tokens = []
        token_types = []

        # Add property prefix if available
        if "properties" in structure_data:
            prop_tokens = self._encode_properties(structure_data["properties"])
            tokens.extend(prop_tokens[0])
            token_types.extend(prop_tokens[1])

        # Add separator (treat as PROPERTY/DISCRETE type)
        tokens.append(self.vocab["SEP"])
        token_types.append(self.token_types["PROPERTY"])  # 0

        # Add composition
        comp_tokens = self._encode_composition(structure_data["composition"])
        tokens.extend(comp_tokens[0])
        token_types.extend(comp_tokens[1])

        # Add space group
        sg_tokens = self._encode_space_group(structure_data["space_group"])
        tokens.extend(sg_tokens[0])
        token_types.extend(sg_tokens[1])

        # Add Wyckoff positions
        wyck_tokens = self._encode_wyckoff_positions(
            structure_data["wyckoff_positions"], structure_data["space_group"]
        )
        tokens.extend(wyck_tokens[0])
        token_types.extend(wyck_tokens[1])

        # Add lattice parameters
        lattice_tokens = self._encode_lattice(structure_data["lattice"])
        tokens.extend(lattice_tokens[0])
        token_types.extend(lattice_tokens[1])

        # Add end of sequence
        tokens.append(self.vocab["EOS"])
        token_types.append(self.token_types["PROPERTY"])  # treat EOS as discrete type

        return tokens, token_types

    def _encode_properties(
        self, properties: Dict[str, float]
    ) -> Tuple[List[int], List[int]]:
        """Encode property prefix."""
        from .properties import discretize_structure_properties

        tokens = []
        token_types = []

        discretized = discretize_structure_properties(properties)

        # Sort properties for consistent ordering
        for prop_name in sorted(discretized.keys()):
            prop_token = discretized[prop_name]
            tokens.append(self.vocab.get(prop_token, self.vocab["UNK"]))
            token_types.append(self.token_types["PROPERTY"])  # 0

        return tokens, token_types

    def _encode_composition(
        self, composition: List[Tuple[str, int]]
    ) -> Tuple[List[int], List[int]]:
        """Encode chemical composition."""
        tokens = []
        token_types = []

        for element, count in composition:
            # Element token
            tokens.append(self.vocab.get(element, self.vocab["UNK"]))
            token_types.append(self.token_types["ELEMENT"])  # 1

            # Count token
            count_token = f"COUNT_{count}"
            tokens.append(self.vocab.get(count_token, self.vocab["UNK"]))
            token_types.append(self.token_types["COUNT"])  # 2

        return tokens, token_types

    def _encode_space_group(self, space_group: int) -> Tuple[List[int], List[int]]:
        """Encode space group."""
        sg_token = f"SPACE_{space_group}"
        return [self.vocab.get(sg_token, self.vocab["UNK"])], [
            self.token_types["SPACEGROUP"]
        ]

    def _encode_wyckoff_positions(
        self, wyckoff_positions: List[Dict], space_group: int
    ) -> Tuple[List[int], List[int]]:
        """Encode Wyckoff positions with elements and coordinates."""
        tokens = []
        token_types = []

        for pos in wyckoff_positions:
            # Wyckoff position token
            wyckoff = pos["wyckoff"]
            wyck_token = None
            key = f"{space_group}_{wyckoff}"
            orbit_id = self.wyckoff_mapping.get(key)

            if orbit_id is None:
                raise ValueError(
                    f"Wyckoff position {wyckoff} not found for space group {space_group}"
                )

            wyck_token = f"WYCK_{orbit_id}"
            tokens.append(self.vocab.get(wyck_token, self.vocab["UNK"]))
            token_types.append(self.token_types["WYCKOFF"])  # 4

            # Element token
            element = pos["element"]
            tokens.append(self.vocab.get(element, self.vocab["UNK"]))
            token_types.append(self.token_types["ELEMENT"])  # 1

            # Coordinates (x, y, z)
            coords = pos["coordinates"]
            for coord in coords:
                tokens.append(coord)  # continuous value
                token_types.append(self.token_types["COORDINATE"])  # 5

        return tokens, token_types

    def is_valid_wyckoff_for_space_group(
        self, wyckoff_token_id: int, space_group: int
    ) -> bool:
        """Check if an encoded Wyckoff orbit token is valid for a space group.

        Requires that mapping and vocab are available. Returns True if unknown.
        """
        if not self.sg_allowed_orbit_token_ids:
            # Build cache lazily
            try:
                from typing import Set

                sg_to_allowed: Dict[int, Set[int]] = {}
                for key, orbit_id in self.wyckoff_mapping.items():
                    try:
                        sg_str, _ = key.split("_", 1)
                        sg = int(sg_str)
                        token_name = f"WYCK_{orbit_id}"
                        tok_id = self.vocab.get(token_name)
                        if tok_id is None:
                            continue
                        sg_to_allowed.setdefault(sg, set()).add(tok_id)
                    except Exception:
                        continue
                self.sg_allowed_orbit_token_ids = sg_to_allowed
            except Exception:
                return True
        allowed = self.sg_allowed_orbit_token_ids.get(space_group)
        if not allowed:
            return True
        return wyckoff_token_id in allowed

    def _encode_lattice(self, lattice: Dict[str, float]) -> Tuple[List[int], List[int]]:
        """Encode lattice parameters."""
        tokens = []
        token_types = []

        # Lengths: a, b, c
        for param in ["a", "b", "c"]:
            if param in lattice:
                tokens.append(lattice[param])  # continuous value
                token_types.append(self.token_types["LATTICE"])  # 6

        # Angles: alpha, beta, gamma
        for param in ["alpha", "beta", "gamma"]:
            if param in lattice:
                tokens.append(lattice[param])  # continuous value
                token_types.append(self.token_types["LATTICE"])  # 6

        return tokens, token_types

    def decode_structure(
        self,
        tokens: np.ndarray,
        token_types: np.ndarray,
        max_elements: int = 20,
    ) -> Optional[Dict[str, Any]]:
        """Decode token and type sequences into a crystal structure.

        Args:
            tokens: Token ID sequence (numpy array)
            token_types: Token type sequence (numpy array)
            max_elements: Maximum number of element entries to consider

        Returns:
            Parsed structure dictionary or None if decoding fails.
        """
        # Keep a raw copy to locate EOS boundary for lattice parsing
        raw_tokens = np.array(tokens)
        # Remove special tokens and padding for main parsing
        tokens, token_types = self._clean_sequence(tokens, token_types)

        if len(tokens) == 0:
            return None

        structure: Dict[str, Any] = {
            "structure_id": f"generated_{hash(tuple(tokens)) % 10000}",
            "composition": [],
            "wyckoff_positions": [],
            "lattice": {},
            "properties": {},
            "space_group": None,
        }

        reverse_vocab = self.reverse_vocab

        def is_continuous_value(val: float) -> bool:
            as_int = int(val)
            return abs(val - as_int) > 1e-6 or (as_int not in reverse_vocab)

        # Determine start of structural tokens using SEP if present
        sep_idx: Optional[int] = None
        try:
            sep_token_id = self.vocab.get("SEP")
            for idx, tok in enumerate(tokens):
                as_int = int(tok)
                if abs(tok - as_int) < 1e-6 and as_int == sep_token_id:
                    sep_idx = idx
                    break
        except Exception:
            sep_idx = None

        i = (sep_idx + 1) if sep_idx is not None else 0
        encountered_space_group: bool = False

        # Parse sequentially by token types without forcing substitutions
        while i < len(tokens):
            ttype = int(token_types[i]) if i < len(token_types) else -1
            val = tokens[i]

            # SPACEGROUP
            if ttype == self.token_types["SPACEGROUP"]:
                try:
                    name = reverse_vocab.get(int(val), "")
                    if name.startswith("SPACE_"):
                        structure["space_group"] = int(name.split("_")[1])
                        encountered_space_group = True
                except Exception:
                    pass
                i += 1
                continue

            # COMPOSITION before SPACEGROUP: ELEMENT optionally followed by COUNT
            if not encountered_space_group:
                if ttype == self.token_types["ELEMENT"]:
                    try:
                        elem = reverse_vocab.get(int(val), "")
                        if (
                            elem
                            and elem.isalpha()
                            and len(elem) <= 2
                            and len(structure["composition"]) < max_elements
                        ):
                            structure["composition"].append((elem, 1))
                            # Optional COUNT follows
                            if (
                                i + 1 < len(tokens)
                                and int(token_types[i + 1]) == self.token_types["COUNT"]
                            ):
                                try:
                                    cnt_name = reverse_vocab.get(
                                        int(tokens[i + 1]), "COUNT_1"
                                    )
                                    cnt = (
                                        int(cnt_name.split("_")[1])
                                        if cnt_name.startswith("COUNT_")
                                        else 1
                                    )
                                    structure["composition"][-1] = (elem, cnt)
                                    i += 2
                                    continue
                                except Exception:
                                    pass
                    except Exception:
                        pass
                i += 1
                continue

            # After SPACEGROUP: WYCKOFF -> ELEMENT -> 3 * COORDINATE
            if ttype == self.token_types["WYCKOFF"]:
                wyck = ""
                try:
                    name = reverse_vocab.get(int(val), "")
                    if name.startswith("WYCK_"):
                        wyck = name.replace("WYCK_", "")
                except Exception:
                    wyck = ""

                j = i + 1
                # Require ELEMENT next
                if (
                    j < len(tokens)
                    and int(token_types[j]) == self.token_types["ELEMENT"]
                ):
                    elem_name = reverse_vocab.get(int(tokens[j]), None)
                    k = j + 1
                    coords: List[float] = []
                    for _ in range(3):
                        if (
                            k < len(tokens)
                            and int(token_types[k]) == self.token_types["COORDINATE"]
                        ):
                            coords.append(
                                float(self._decode_continuous_token(tokens[k])) % 1.0
                            )
                            k += 1
                        else:
                            break
                    if elem_name and len(coords) == 3 and wyck:
                        structure["wyckoff_positions"].append(
                            {
                                "element": elem_name,
                                "wyckoff": wyck,
                                "coordinates": coords,
                            }
                        )
                        i = k
                        continue
                # If pattern not satisfied fully, advance by one without forcing
                i += 1
                continue

            # Default: advance
            i += 1

        # Lattice: take the last 6 LATTICE-typed values BEFORE the first EOS token
        try:
            eos_id = self.special_tokens["EOS"]
            eos_indices = np.where(raw_tokens == eos_id)[0]
            cutoff_idx = (
                int(eos_indices[0]) if len(eos_indices) > 0 else len(raw_tokens)
            )
        except Exception:
            cutoff_idx = len(raw_tokens)

        lattice_vals: List[float] = []
        # Use token_types to select only LATTICE values up to cutoff
        for idx in range(min(cutoff_idx, len(token_types))):
            try:
                if int(token_types[idx]) == self.token_types["LATTICE"]:
                    lattice_vals.append(
                        float(self._decode_continuous_token(raw_tokens[idx]))
                    )
            except Exception:
                continue

        if len(lattice_vals) >= 6:
            a, b, c, alpha, beta, gamma = lattice_vals[-6:]
            structure["lattice"] = {
                "a": float(a),
                "b": float(b),
                "c": float(c),
                "alpha": float(alpha),
                "beta": float(beta),
                "gamma": float(gamma),
            }

        if not self._validate_structure(structure):
            return None

        return structure

    def decode_sequence(
        self, tokens: List[float], token_types: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Convenience method to decode a single sequence.

        Accepts Python lists, converts to numpy, and delegates to decode_structure.
        If token_types is not provided, a zero vector is used.
        """
        np_tokens = np.array(tokens)
        if token_types is None:
            np_types = np.zeros_like(np_tokens)
        else:
            np_types = np.array(token_types)
        result = self.decode_structure(np_tokens, np_types)
        return result or {
            "composition": [],
            "space_group": None,
            "wyckoff_positions": [],
            "lattice": {},
            "properties": {},
        }

    def _clean_sequence(
        self, tokens: np.ndarray, token_types: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove special tokens and padding from a sequence."""
        special_tokens = [
            self.special_tokens["PAD"],
            self.special_tokens["UNK"],
            self.special_tokens["EOS"],
            self.special_tokens["BOS"],
        ]
        mask = ~np.isin(tokens, special_tokens)
        cleaned_tokens = tokens[mask]
        cleaned_types = (
            token_types[mask]
            if len(token_types) == len(tokens)
            else np.zeros_like(cleaned_tokens)
        )
        return cleaned_tokens, cleaned_types

    def _decode_continuous_token(self, token_val: float) -> float:
        """Convert continuous token back to float (identity for now)."""
        try:
            return float(token_val)
        except Exception:
            return 0.0

    def _validate_structure(self, structure: Dict[str, Any]) -> bool:
        """Lenient validity: require at least non-empty composition."""
        return bool(structure.get("composition"))

    def is_discrete_token(self, token: Union[str, int]) -> bool:
        """Check if a token represents a discrete value."""
        if isinstance(token, str):
            return token in self.vocab and token not in self.special_tokens
        else:
            # For integer tokens, check if index exists in reverse vocab
            return token in self.reverse_vocab

    def get_vocab_size(self) -> int:
        """Get the size of the vocabulary."""
        return len(self.vocab)

    def save_vocab(self, filepath: str):
        """Save vocabulary to file."""
        with open(filepath, "w") as f:
            json.dump(self.vocab, f, indent=2)

    def load_vocab(self, filepath: str):
        """Load vocabulary from file."""
        with open(filepath, "r") as f:
            self.vocab = json.load(f)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def get_token_types(self) -> Dict[str, int]:
        """Get token type mappings."""
        return self.token_types.copy()


# Global tokenizer instance
tokenizer = BIFROSTTokenizer()
