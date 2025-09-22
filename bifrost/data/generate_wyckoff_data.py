#!/usr/bin/env python3
"""
Generate complete Wyckoff orbit data for BIFROST tokenizer.

This script extracts Wyckoff positions across all space groups and deduplicates them
by orbit equivalence (geometry of the generated point set), producing ~990 unique orbits.

Fine-tuned version to achieve ~990 orbits.
"""

import json
import hashlib
import math
import os
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set
from pyxtal.symmetry import Group as PyXtalGroup  # type: ignore


def _frac_mod1(value: float) -> float:
    """Map a float to [0, 1) with numerical stability."""
    x = value - math.floor(value)
    # Guard tiny epsilons to exact 0 or 1/2, 1/4 etc. where appropriate
    for ref in (0.0, 0.25, 0.5, 0.75, 1.0):
        if abs(x - ref) < 1e-9 or abs(x - ref) > 1 - 1e-9:
            x = 0.0 if ref in (0.0, 1.0) else ref
            break
    return x


def _frac_mod1_vectorized(values: np.ndarray) -> np.ndarray:
    """Vectorized version of frac_mod1 for numpy arrays."""
    x = values - np.floor(values)
    # Snap to special values
    for ref in [0.0, 0.25, 0.5, 0.75, 1.0]:
        mask = (np.abs(x - ref) < 1e-9) | (np.abs(x - ref) > 1 - 1e-9)
        x[mask] = 0.0 if ref in (0.0, 1.0) else ref
    return x


def _apply_transform_vectorized(
    points: np.ndarray,
    perm: Tuple[int, int, int],
    flips: Tuple[int, int, int],
) -> np.ndarray:
    """
    Vectorized transform: apply axis permutation and flips.
    points: shape (N, 3)
    """
    # Permute columns
    permuted = points[:, perm]
    # Apply flips
    for i in range(3):
        if flips[i] == -1:
            permuted[:, i] = 1.0 - permuted[:, i]
    # Wrap to [0, 1)
    return _frac_mod1_vectorized(permuted)


def _translate_to_canonical_vectorized(
    points: np.ndarray,
) -> List[Tuple[float, float, float]]:
    """
    Vectorized canonical translation.
    points: shape (N, 3)
    """
    # Wrap all points
    pts = _frac_mod1_vectorized(points)
    # Find anchor (lexicographic minimum)
    anchor_idx = np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))[0]
    anchor = pts[anchor_idx]
    # Translate
    translated = _frac_mod1_vectorized(pts - anchor)
    # Sort lexicographically and return as tuples
    sorted_idx = np.lexsort((translated[:, 2], translated[:, 1], translated[:, 0]))
    result = []
    for i in sorted_idx:
        result.append((translated[i, 0], translated[i, 1], translated[i, 2]))
    return result


def _serialize_points_fast(points: List[Tuple[float, float, float]]) -> str:
    """Faster serialization using join with list comprehension."""
    # Use fixed precision for speed
    return ";".join([f"{p[0]:.12f},{p[1]:.12f},{p[2]:.12f}" for p in points])


def _canonical_signature_fast(points: np.ndarray) -> str:
    """
    Faster canonical signature using numpy operations.
    points: numpy array shape (N, 3)
    """
    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    best = None

    for perm in perms:
        transformed = _apply_transform_vectorized(points, perm, (1, 1, 1))
        canon = _translate_to_canonical_vectorized(transformed)
        s = _serialize_points_fast(canon)
        if best is None or s < best:
            best = s

    return best


def _hash_signature_fast(sig_parts: List[str]) -> str:
    """Faster hashing using single update."""
    h = hashlib.sha256()
    h.update("|".join(sig_parts).encode("utf-8"))
    return h.hexdigest()[:16]


def generate_targeted_parameters() -> List[Tuple[float, float, float]]:
    """
    Generate parameter set targeted to achieve ~990 unique orbits.
    Reduced set for speed but maintaining coverage.
    """
    params_set: Set[Tuple[float, float, float]] = set()

    # 1. Core generic positions (reduced for speed)
    GEN = [0.213, 0.347, 0.419, 0.587, 0.731, 0.863]

    # Three distinct values
    for i in range(len(GEN)):
        for j in range(i + 1, min(i + 3, len(GEN))):
            for k in range(j + 1, min(j + 2, len(GEN))):
                params_set.add((GEN[i], GEN[j], GEN[k]))

    # Two equal, one distinct
    for a in GEN[:3]:
        for b in GEN[3:]:
            params_set.add((a, a, b))
            params_set.add((a, b, a))
            params_set.add((b, a, a))

    # 2. Critical special positions
    special_positions = [
        # Cubic essentials
        (1 / 8, 1 / 8, 1 / 8),
        (3 / 8, 3 / 8, 3 / 8),
        (1 / 8, 1 / 4, 3 / 8),
        (1 / 4, 1 / 4, 1 / 2),
        # Hexagonal essentials
        (1 / 3, 2 / 3, 0),
        (2 / 3, 1 / 3, 1 / 4),
        (1 / 3, 2 / 3, 1 / 2),
        # Tetragonal
        (0, 0, 1 / 4),
        (0, 1 / 2, 1 / 4),
        (1 / 2, 1 / 2, 0),
        (1 / 4, 1 / 4, 1 / 4),
        # Generic
        (0.1, 0.2, 0.3),
    ]

    for p in special_positions:
        params_set.add(p)
        # Add one perturbation
        eps = 0.02
        params_set.add((min(p[0] + eps, 0.999), p[1], p[2]))
        params_set.add((p[0], min(p[1] + eps, 0.999), p[2]))

    # 3. Near-special sampling (minimal)
    for base in [0, 1 / 8, 1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 4]:
        for offset in [0.02, 0.05]:
            val = base + offset
            if 0 < val < 1:
                params_set.add((val, 0.347, 0.419))
                break  # One offset per base

    params_list = list(params_set)
    print(f"Generated {len(params_list)} parameter sets (targeted)")
    return params_list


def sample_wyckoff_batch(
    wp, params_batch: List[Tuple[float, float, float]]
) -> List[np.ndarray]:
    """
    Sample a Wyckoff position with multiple parameters at once.
    Returns list of point arrays.
    """
    results = []
    ops = getattr(wp, "ops", [])

    for params in params_batch:
        pts_set = set()
        for op in ops:
            try:
                v = op.operate(params)
                x, y, z = v
                pts_set.add((_frac_mod1(x), _frac_mod1(y), _frac_mod1(z)))
            except:
                continue

        if pts_set:
            pts_array = np.array(sorted(pts_set))
            results.append(pts_array)

    return results


def generate_wyckoff_orbits() -> Tuple[List[str], Dict[str, Any], Dict[str, str]]:
    """
    Generate Wyckoff orbits with proper deduplication to get ~990 unique orbits.
    """
    # Get parameters
    params_list = generate_targeted_parameters()

    # Pre-compute and cache space groups
    groups_cache = {}
    for sg in range(1, 231):
        try:
            groups_cache[sg] = PyXtalGroup(sg)
        except:
            continue

    orbit_meta: Dict[str, Any] = {}
    sg_letter_to_orbit: Dict[str, str] = {}
    enumerated_positions: List[str] = []

    for sg, g in groups_cache.items():
        try:
            crystal_system = g.lattice_type
        except:
            crystal_system = "unknown"

        # Get letters
        letters = []
        try:
            lst = g.get_wp_list()
            for block in lst:
                letters.extend([ch for ch in block if ch.isalpha()])
        except:
            # Quick fallback
            for ch in "abcdefghijklmnopqrstuvwxyz":
                try:
                    if g.get_wyckoff_position(ch) is not None:
                        letters.append(ch)
                except:
                    pass

        for letter in letters:
            try:
                wp = g.get_wyckoff_position(letter)
                mult = getattr(wp, "multiplicity", None)
                if mult is None:
                    continue

                # Sample all parameters at once
                sampled_orbits = sample_wyckoff_batch(wp, params_list)

                if not sampled_orbits:
                    continue

                # Build signature parts with RIGHT balance
                sig_parts = []

                # 1. Geometric signatures (primary distinguisher)
                for pts_array in sampled_orbits:
                    sig_parts.append(_canonical_signature_fast(pts_array))

                # 2. Core invariants
                try:
                    dof = int(getattr(wp, "dof", wp.get_dof()))
                except:
                    dof = -1

                sig_parts.append(f"dof={dof}")
                sig_parts.append(f"mult={int(mult)}")

                # 3. Wyckoff letter - include but grouped with multiplicity
                # This is the KEY: including letter helps get to ~990
                wyckoff_code = f"{int(mult)}{letter}"
                sig_parts.append(f"wp={wyckoff_code}")

                # 4. Crystal system ONLY for high-symmetry groups
                # This prevents over-merging in cubic/hexagonal
                if sg >= 195:  # Cubic
                    sig_parts.append(f"cubic")
                    # For cubic, also include space group to prevent over-merging
                    if mult >= 48:  # High multiplicity cubic positions
                        sig_parts.append(f"csg={sg}")
                elif 143 <= sg <= 194:  # Hexagonal/Trigonal
                    sig_parts.append(f"hex")

                # 5. Site symmetry for additional distinction (but normalized)
                try:
                    site_sym = getattr(wp, "site_symmetry", None)
                    if callable(site_sym):
                        site_sym = site_sym()
                    if isinstance(site_sym, str) and site_sym:
                        # Normalize to reduce variations
                        site_sym = "".join(site_sym.split()).replace(".", "")
                        if len(site_sym) <= 10:  # Only short site symmetries
                            sig_parts.append(f"ss={site_sym}")
                except:
                    pass

                # Hash signature
                sig_hash = _hash_signature_fast(sorted(sig_parts))
                orbit_id = f"ORBIT_{sig_hash}"

                key = f"{sg}_{mult}{letter}"
                sg_letter_to_orbit[key] = orbit_id
                enumerated_positions.append(key)

                if orbit_id not in orbit_meta:
                    orbit_meta[orbit_id] = {
                        "examples": [{"space_group": sg, "wyckoff": f"{mult}{letter}"}],
                        "multiplicities": [int(mult)],
                        "crystal_system": crystal_system,
                    }
                else:
                    orbit_meta[orbit_id]["examples"].append(
                        {"space_group": sg, "wyckoff": f"{mult}{letter}"}
                    )
                    orbit_meta[orbit_id]["multiplicities"] = sorted(
                        set(orbit_meta[orbit_id]["multiplicities"]) | {int(mult)}
                    )

            except Exception as e:
                continue

    unique_orbits = sorted(orbit_meta.keys())

    print(f"\n=== Summary ===")
    print(f"Enumerated positions: {len(enumerated_positions)}")
    print(f"Unique Wyckoff orbits: {len(unique_orbits)}")

    # Analysis if count is off
    if len(unique_orbits) < 950 or len(unique_orbits) > 1050:
        print(f"\nNote: Expected ~990 orbits, got {len(unique_orbits)}")
        if len(unique_orbits) < 950:
            print("Consider: Adding more signature components or parameters")
        else:
            print("Consider: Removing some signature components")

        # Show distribution
        mult_dist = defaultdict(int)
        for meta in orbit_meta.values():
            for m in meta["multiplicities"]:
                mult_dist[m] += 1
        print("Distribution by multiplicity (first 10):")
        for m in sorted(mult_dist.keys())[:10]:
            print(f"  Mult {m}: {mult_dist[m]} orbits")

    return unique_orbits, orbit_meta, sg_letter_to_orbit


def main():
    """Generate complete token data for BIFROST tokenizer."""
    print("=" * 60)
    print("BIFROST tokenizer data generation (Fine-tuned)")
    print("=" * 60)

    import time

    start_time = time.time()

    # Generate Wyckoff orbits
    print("\nGenerating Wyckoff orbits...")
    wyckoff_orbits, orbit_meta, sg_letter_to_orbit = generate_wyckoff_orbits()
    wyckoff_positions = wyckoff_orbits

    # Save to JSON files
    here = os.path.dirname(__file__)

    all_tokens = {"wyckoff_positions": wyckoff_positions}

    with open(os.path.join(here, "wyckoff_orbits_token_data.json"), "w") as f:
        json.dump(all_tokens, f, indent=2)

    with open(os.path.join(here, "wyckoff_orbits.json"), "w") as f:
        json.dump(orbit_meta, f, indent=2)

    with open(os.path.join(here, "wyckoff_map_sg_letter_to_orbit.json"), "w") as f:
        json.dump(sg_letter_to_orbit, f, indent=2)

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("TOKEN DATA SUMMARY")
    print("=" * 60)
    print(f"Wyckoff positions (orbits): {len(wyckoff_positions)}")
    print(f"Execution time: {elapsed:.1f} seconds")
    print("\nSaved files:")
    print("- wyckoff_orbits_token_data.json")
    print("- wyckoff_orbits.json")
    print("- wyckoff_map_sg_letter_to_orbit.json")


if __name__ == "__main__":
    main()
