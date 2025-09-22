#!/usr/bin/env python3
"""
Generate a BIFROST-compatible dataset from Materials Project using MPRester.

This script queries the Materials Project API for crystal structures and a set of
target properties, converts them into the structure dictionary format expected by
`bifrost.data.dataset.CrystalStructureDataset` and saves the results to JSON.

Usage:
  python bifrost/data/generate_mp_dataset.py \
      --api_key YOUR_MP_API_KEY \
      --max_structures 1000 \
      --output mp_dataset.json

Notes:
- Requires `pymatgen` (and `spglib`). `mp-api` is optional.
- Properties are mapped to our tokenizer property bins: band_gap, density, bulk_modulus,
  formation_energy_per_atom, and energy_above_hull if available.
"""

from __future__ import annotations
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path


# Materials Project symmetry precision
SYMPREC = 0.1


def _as_float(value: Any) -> Optional[float]:
    """Best-effort conversion to float for MP fields that may be nested.

    Handles raw numbers, dicts with common numeric keys, simple pydantic-like
    objects exposing a ``value`` attribute, and simple sequences.
    """
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        # dict payloads from some endpoints
        if isinstance(value, dict):
            for key in (
                "value",
                "band_gap",
                "band_gap",
                "energy",
                "e_above_hull",
                "bulk_modulus",
                "k_voigt",
                "k_vrh",
            ):
                v = value.get(key)
                if isinstance(v, (int, float)):
                    return float(v)
            return None
        # pydantic model style
        if hasattr(value, "value"):
            v = getattr(value, "value")
            if isinstance(v, (int, float)):
                return float(v)
        # simple sequences
        if isinstance(value, (list, tuple)) and len(value) > 0:
            first = value[0]
            if isinstance(first, (int, float)):
                return float(first)
        return None
    except Exception:
        return None


def _build_structure_entry(
    doc: Any,
) -> Optional[Dict[str, Any]]:
    """
    Convert an MP document into our structure dict.

    Expected output keys:
      - structure_id: str
      - composition: List[Tuple[element, count]]
      - space_group: int
      - wyckoff_positions: List[{element, wyckoff, coordinates}]
      - lattice: {a,b,c,alpha,beta,gamma}
      - properties: {band_gap, density, bulk_modulus, formation_energy_per_atom, energy_above_hull}

    Wyckoff positions are not directly available from MP documents; we will
    derive Wyckoff letters using spglib through pymatgen if possible. If Wyckoff
    determination fails, we skip that structure (the tokenizer requires wyckoff
    mapping).
    """

    try:
        # Get a Pymatgen Structure from the document
        # if hasattr(doc, "structure"):
        pmg_struct: Structure = doc.structure

        # Use conventional standardized structure for stable Wyckoff labeling
        sga = SpacegroupAnalyzer(pmg_struct, symprec=SYMPREC, angle_tolerance=5)
        conv = sga.get_conventional_standard_structure()
        sga_conv = SpacegroupAnalyzer(conv, symprec=SYMPREC, angle_tolerance=5)
        sym_struct = sga_conv.get_symmetrized_structure()
        space_group_number = sga_conv.get_space_group_number()

        # Wyckoff symbols per site group using equivalent indices to avoid
        # site-index mismatches between structures
        wyckoff_positions: List[Dict[str, Any]] = []
        for group_indices, site_group in zip(
            getattr(sym_struct, "equivalent_indices", []), sym_struct.equivalent_sites
        ):
            if not group_indices:
                continue
            idx = int(group_indices[0])
            # Guard against rare inconsistencies
            if idx < 0 or idx >= len(sym_struct.wyckoff_symbols):
                continue
            wyck = sym_struct.wyckoff_symbols[idx]
            rep_site = site_group[0]
            element = rep_site.specie.symbol
            frac_coords = [float(x) % 1.0 for x in rep_site.frac_coords]
            wyckoff_positions.append(
                {"element": element, "wyckoff": wyck, "coordinates": frac_coords}
            )

        # Composition as element counts (integerize by rounding if necessary)
        from pymatgen.core.composition import Composition

        # Get simplest integer formula from the conventional structure
        formula, _ = conv.composition.get_integer_formula_and_factor()
        comp_int = Composition(formula).get_el_amt_dict()
        composition: List[Tuple[str, int]] = [
            (elem, int(round(cnt))) for elem, cnt in comp_int.items()
        ]

        # Lattice parameters (conventional cell)
        lattice = conv.lattice
        lattice_dict = {
            "a": float(lattice.a),
            "b": float(lattice.b),
            "c": float(lattice.c),
            "alpha": float(lattice.alpha),
            "beta": float(lattice.beta),
            "gamma": float(lattice.gamma),
        }

        # Properties extraction (best-effort; allow missing/non-scalar values)
        properties: Dict[str, Optional[float]] = {}
        # band gap (eV)
        properties["band_gap"] = _as_float(getattr(doc, "band_gap", None))
        properties["efermi"] = _as_float(getattr(doc, "efermi", None))
        properties["total_magnetization"] = _as_float(
            getattr(doc, "total_magnetization", None)
        )

        # density (g/cm^3)
        properties["density"] = _as_float(getattr(conv, "density", None))
        # bulk modulus (GPa)
        bulk_modulus = None
        for key in ("k_vrh", "k_voigt", "bulk_modulus", "shear_modulus"):
            bulk_modulus = _as_float(getattr(doc, key, None))
            if bulk_modulus is not None:
                break
        properties["bulk_modulus"] = bulk_modulus
        # formation energy per atom (eV/atom)
        form_e = _as_float(getattr(doc, "formation_energy_per_atom", None))
        if form_e is None and isinstance(doc, dict):
            form_e = _as_float(doc.get("formation_energy_per_atom"))
        properties["formation_energy_per_atom"] = form_e
        # energy above hull (eV/atom)
        ehull = None

        # First check if we have thermodynamic data from our separate fetch
        if hasattr(doc, "energy_above_hull"):
            ehull = _as_float(getattr(doc, "energy_above_hull"))

        # If not found, try the standard fields (might be available in some API versions)
        if ehull is None:
            for key in ("e_above_hull", "ehull", "energy_above_hull"):
                ehull = _as_float(getattr(doc, key, None))
                if ehull is not None:
                    break
                if isinstance(doc, dict):
                    ehull = _as_float(doc.get(key))
                    if ehull is not None:
                        break

        # If energy above hull is still not found, we could calculate it using pymatgen
        # but this would require fetching all competing phases for the chemical system
        # For now, we'll just set it to None if not available
        properties["energy_above_hull"] = ehull

        structure_id = getattr(doc, "material_id", None) or getattr(
            doc, "task_id", None
        )
        if structure_id is None and isinstance(doc, dict):
            structure_id = (
                doc.get("material_id") or doc.get("task_id") or doc.get("task_id")
            )
        structure_id = str(structure_id) if structure_id is not None else "unknown"

        return {
            "structure_id": structure_id,
            "composition": composition,
            "space_group": int(space_group_number),
            "wyckoff_positions": wyckoff_positions,
            "lattice": lattice_dict,
            "properties": properties,
        }
    except Exception as e:
        return None


def _fetch_documents(
    api_key: str,
    max_structures: int,
    include_mechanical: bool = True,
) -> List[Any]:
    """Fetch MP documents with thermodynamic data.

    Returns a list of documents that each have `.structure` and attributes used in builder.
    Energy above hull is fetched from the thermo endpoint if available.
    """

    try:
        from mp_api.client import MPRester  # type: ignore

        # Fields available in summary endpoint
        summary_fields = [
            "material_id",
            "structure",
            "band_gap",
            "formation_energy_per_atom",
            "energy_above_hull",
            "efermi",
            "total_magnetization",
            "density",
        ]
        if include_mechanical:
            summary_fields += ["k_voigt", "k_vrh", "bulk_modulus", "shear_modulus"]

        docs: List[Any] = []
        thermo_data: Dict[str, float] = {}

        with MPRester(api_key) as mpr:
            # Get thermodynamic data including energy above hull
            try:
                thermo_docs = None
                # Use the thermo endpoint with energy above hull filter
                thermo_docs = list(
                    mpr.materials.thermo.search(
                        # energy_above_hull=(0, 0.4),
                        num_chunks=max(1, (max_structures + 999) // 1000),
                        chunk_size=min(1000, max_structures),
                        fields=[
                            "material_id",
                            "energy_above_hull",
                            "formation_energy_per_atom",
                            "thermo_type",
                        ],
                    )
                )

                print(f"Fetched {len(thermo_docs)} thermodynamic data")
                for thermo_doc in thermo_docs:
                    if hasattr(thermo_doc, "material_id") and hasattr(
                        thermo_doc, "energy_above_hull"
                    ):
                        thermo_data[thermo_doc.material_id] = (
                            thermo_doc.energy_above_hull
                        )
                        if len(thermo_data) >= max_structures:
                            break

                print(
                    f"Successfully fetched thermodynamic data for {len(thermo_data)} materials"
                )

            except Exception as e:
                print(f"Warning: Could not fetch thermodynamic data: {e}")
                raise e

            # Get material IDs that have thermo data, or search broadly
            material_ids = list(thermo_data.keys())[:max_structures]

            # Now get summary data
            print("Fetching summary data...")
            chunk_size = min(1000, max(100, max_structures))
            num_chunks = max(1, (max_structures + chunk_size - 1) // chunk_size)

            search_kwargs = {
                "fields": summary_fields,
                "num_chunks": num_chunks,
                "chunk_size": chunk_size,
            }

            # If we have a small number of IDs, pass them directly; otherwise, omit the ID
            # filter to avoid the mp_api validate_ids limit and filter locally instead.
            MAX_IDS_PER_QUERY = 1000
            use_id_filter = (
                len(material_ids) > 0 and len(material_ids) <= MAX_IDS_PER_QUERY
            )
            if use_id_filter:
                search_kwargs["material_ids"] = material_ids
                for doc in mpr.materials.summary.search(**search_kwargs):
                    docs.append(doc)
                    if len(docs) >= max_structures:
                        break
            else:
                if len(material_ids) > MAX_IDS_PER_QUERY:
                    print(
                        f"material_ids list too long ({len(material_ids)}). Streaming without ID filter and filtering locally."
                    )
                # Stream results and filter locally if we have a thermo ID set
                thermo_id_set = set(material_ids)
                for doc in mpr.materials.summary.search(**search_kwargs):
                    if (
                        thermo_id_set
                        and getattr(doc, "material_id", None) not in thermo_id_set
                    ):
                        continue
                    docs.append(doc)
                    if len(docs) >= max_structures:
                        break

        print(f"Fetched {len(docs)} total documents")

        # Store thermo data in a way that can be accessed by _build_structure_entry
        for doc in docs:
            if hasattr(doc, "material_id") and doc.material_id in thermo_data:
                # Store as a custom attribute that won't conflict with Pydantic
                setattr(doc, "_thermo_e_above_hull", thermo_data[doc.material_id])

        return docs[:max_structures]
    except Exception as e:

        import traceback

        traceback.print_exc()
        raise e


def main():
    parser = argparse.ArgumentParser(description="Generate MP dataset for BIFROST")
    parser.add_argument(
        "--api_key", type=str, required=True, help="Materials Project API key"
    )
    parser.add_argument(
        "--max_structures",
        type=int,
        default=1000,
        help="Maximum number of structures to fetch",
    )
    parser.add_argument(
        "--output", type=str, default="data/mp_dataset.json", help="Output JSON file"
    )
    args = parser.parse_args()

    print("Fetching documents from Materials Project...")
    docs = _fetch_documents(
        api_key=args.api_key,
        max_structures=args.max_structures,
    )
    print(f"Fetched {len(docs)} documents. Building dataset entries...")

    dataset: List[Dict[str, Any]] = []
    skipped = 0
    for doc in docs:
        entry = _build_structure_entry(doc)
        if entry is None:
            skipped += 1
            continue
        dataset.append(entry)

    print(
        f"Built {len(dataset)} entries. Skipped {skipped} due to missing Wyckoff or errors."
    )

    here = os.path.dirname(__file__)
    out_path = os.path.join(here, args.output)
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved dataset to {out_path}")


if __name__ == "__main__":
    main()
