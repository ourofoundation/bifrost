#!/usr/bin/env python3
"""
Generate complete Wyckoff orbit data for BIFROST tokenizer.

This script extracts Wyckoff positions across all space groups and deduplicates them
by orbit equivalence (geometry of the generated point set), producing ~990 unique orbits.

It prefers the pyxtal library for accurate Wyckoff generators. If pyxtal is unavailable,
it will fall back to a simplified synthetic list (not recommended).
"""

import json
import hashlib
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Any
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


def _apply_transform(
    points: List[Tuple[float, float, float]],
    perm: Tuple[int, int, int],
    flips: Tuple[int, int, int],
) -> List[Tuple[float, float, float]]:
    """
    Apply axis permutation and flips (x -> 1-x if flip == -1) to a set of fractional points.
    """
    out: List[Tuple[float, float, float]] = []
    for p in points:
        q = [p[0], p[1], p[2]]
        r = [q[perm[0]], q[perm[1]], q[perm[2]]]
        for i in range(3):
            r[i] = _frac_mod1(1.0 - r[i]) if flips[i] == -1 else _frac_mod1(r[i])
        out.append((r[0], r[1], r[2]))
    return out


def _translate_to_canonical(
    points: List[Tuple[float, float, float]],
) -> List[Tuple[float, float, float]]:
    """
    Make the set translation-invariant by subtracting the lexicographically minimal point and wrapping mod 1.
    """
    pts = [(_frac_mod1(x), _frac_mod1(y), _frac_mod1(z)) for (x, y, z) in points]
    anchor = min(pts)
    out: List[Tuple[float, float, float]] = []
    for x, y, z in pts:
        out.append(
            (
                _frac_mod1(x - anchor[0]),
                _frac_mod1(y - anchor[1]),
                _frac_mod1(z - anchor[2]),
            )
        )
    out.sort()
    return out


def _serialize_points(
    points: List[Tuple[float, float, float]], precision: int = 8
) -> str:
    fmt = f"{{:.{precision}f}}"
    return ";".join(
        [
            "%s,%s,%s" % (fmt.format(p[0]), fmt.format(p[1]), fmt.format(p[2]))
            for p in points
        ]
    )


def _canonical_signature(points: List[Tuple[float, float, float]]) -> str:
    """
    Compute a canonical, translation/axis-permutation/flip invariant signature for a point set.
    """
    perms: List[Tuple[int, int, int]] = [
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ]
    flips: List[Tuple[int, int, int]] = []
    for fx in (1, -1):
        for fy in (1, -1):
            for fz in (1, -1):
                flips.append((fx, fy, fz))

    best: str = None  # type: ignore
    for perm in perms:
        for flip in flips:
            transformed = _apply_transform(points, perm, flip)
            canon = _translate_to_canonical(transformed)
            s = _serialize_points(canon)
            if best is None or s < best:
                best = s
    assert best is not None
    return best


def _hash_signature(sig_parts: List[str]) -> str:
    h = hashlib.sha256()
    for s in sig_parts:
        h.update(s.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:16]


def generate_wyckoff_orbits() -> Tuple[List[str], Dict[str, Any], Dict[str, str]]:
    """
    Generate unique Wyckoff orbit IDs by deduplicating across space groups using pyxtal generators.

    Returns:
        - List of unique orbit token strings (e.g., "ORBIT_abcdef1234567890")
        - Dict mapping orbit_id -> metadata (examples, multiplicity set)
        - Dict mapping f"{sg}_{mult}{letter}" -> orbit_id
    """
    # Parameter assignments for signature stability (expanded coverage)
    params_list: List[Tuple[float, float, float]] = []

    def add_param(a: float, b: float, c: float):
        t = (float(a), float(b), float(c))
        if t not in params_list:
            params_list.append(t)

    # Key float anchors
    F = [0.271828, 0.314159, 0.161803, 0.707107, 0.123457]
    # Rational fractions commonly appearing in Wyckoff descriptions
    R = [
        0.0,
        1.0 / 6.0,
        1.0 / 4.0,
        1.0 / 3.0,
        1.0 / 2.0,
        2.0 / 3.0,
        3.0 / 4.0,
        5.0 / 6.0,
    ]

    # Equal-value regimes
    for r in [0.0, 1.0 / 4.0, 1.0 / 3.0, 1.0 / 2.0, 2.0 / 3.0, 3.0 / 4.0]:
        add_param(r, r, r)
    for f in [F[0]]:
        add_param(f, f, f)

    # Distinct-value regimes (various triples from F)
    add_param(F[0], F[1], F[2])
    add_param(F[1], F[2], F[3])
    add_param(F[2], F[3], F[4])
    add_param(F[0], F[2], F[4])

    # Two-equal-one-distinct using rational + float
    for r in [0.0, 1.0 / 4.0, 1.0 / 2.0, 3.0 / 4.0]:
        for f in [F[0], F[1], F[2]]:
            add_param(r, r, f)
            add_param(r, f, r)
            add_param(f, r, r)

    # Boundary/plane-centered cases
    add_param(0.0, 0.37, 0.0)
    add_param(0.5, 0.13, 0.5)
    add_param(0.25, 0.75, 0.0)
    add_param(0.0, 0.0, 0.0)
    add_param(0.5, 0.5, 0.5)

    # Rational distinct combinations
    add_param(1.0 / 3.0, 2.0 / 3.0, 0.0)
    add_param(1.0 / 6.0, 5.0 / 6.0, 1.0 / 2.0)
    add_param(1.0 / 4.0, 3.0 / 4.0, 1.0 / 2.0)
    add_param(1.0 / 3.0, 1.0 / 2.0, 2.0 / 3.0)

    orbit_meta: Dict[str, Any] = {}
    sg_letter_to_orbit: Dict[str, str] = {}
    enumerated_positions: List[str] = []  # track all (sg, mult+letter)

    for sg in range(1, 231):
        try:
            g = PyXtalGroup(sg)
        except Exception as e:
            print(f"Warning: failed to create PyXtal Group for SG {sg}: {e}")
            continue

        # pyxtal Group.Wyckoff_positions is typically a list of Wyckoff position objects
        wyckoff_list = getattr(g, "Wyckoff_positions", None)
        if not wyckoff_list:
            # Some groups may represent positions differently; skip if unavailable
            print(f"Warning: no Wyckoff positions available for SG {sg}")
            continue

        # pyxtal Group.get_wp_list() returns letters by multiplicity classes; use get_wyckoff_position(letter)
        # We iterate letters available via group API
        try:
            letters = []
            # Prefer pyxtal-provided list of available letters per group
            try:
                lst = g.get_wp_list()  # e.g., ['abcd', 'ef', ...]
                for block in lst:
                    for ch in block:
                        if ch.isalpha():
                            letters.append(ch)
            except Exception:
                print(f"Warning: no WP list available for SG {sg}")
                # Fallback discovery across alphabet + 'alpha'
                for ch in list("abcdefghijklmnopqrstuvwxyz") + ["alpha", "α"]:
                    try:
                        wp = g.get_wyckoff_position(ch)
                        if wp is not None:
                            letters.append(ch)
                    except Exception:
                        continue
        except Exception:
            letters = []

        for letter in letters:
            try:
                wp = g.get_wyckoff_position(letter)
                mult = getattr(wp, "multiplicity", None)
                if mult is None:
                    continue

                # Build orbits by applying symmetry ops to multiple parameter points
                def sample(
                    points_params: Tuple[float, float, float],
                ) -> List[Tuple[float, float, float]]:
                    pts: List[Tuple[float, float, float]] = []
                    for op in getattr(wp, "ops", []):
                        try:
                            v = op.operate(points_params)
                        except Exception:
                            try:
                                v = op.operate(tuple(points_params))
                            except Exception:
                                continue
                        x, y, z = v
                        pts.append((_frac_mod1(x), _frac_mod1(y), _frac_mod1(z)))
                    # unique
                    pts = sorted(set(pts))
                    return pts

                sig_parts: List[str] = []
                any_pts = False
                for p in params_list:
                    pts = sample(p)
                    if pts:
                        any_pts = True
                        sig_parts.append(_canonical_signature(pts))

                if not any_pts:
                    continue

                # Include invariants to avoid over-merging
                # 1) DOF
                try:
                    dof = int(getattr(wp, "dof", wp.get_dof()))  # type: ignore[attr-defined]
                except Exception:
                    dof = -1
                sig_parts.append(f"dof={dof}")
                # 2) Wyckoff label shape (e.g., '4c')
                try:
                    label = str(wp.get_label())
                except Exception:
                    label = f"{mult}{letter}"
                sig_parts.append(f"label={label}")
                # 3) Site-symmetry euclidean ops signature (count + types via rotation matrices)
                try:
                    eops = wp.get_euclidean_ops()
                    # Serialize rotation parts only for stability
                    rot_sigs = []
                    for op in eops:
                        R = op.rotation_matrix
                        rot_sigs.append(
                            ",".join(
                                [
                                    "%d" % int(round(R[i, j]))
                                    for i in range(3)
                                    for j in range(3)
                                ]
                            )
                        )
                    rot_sigs = sorted(set(rot_sigs))
                    sig_parts.append(f"eops={len(eops)}:[" + ";".join(rot_sigs) + "]")
                except Exception:
                    pass

                # Final signature
                sig_hash = _hash_signature(sorted(sig_parts))
                orbit_id = f"ORBIT_{sig_hash}"

                key = f"{sg}_{mult}{letter}"
                sg_letter_to_orbit[key] = orbit_id
                enumerated_positions.append(key)

                if orbit_id not in orbit_meta:
                    orbit_meta[orbit_id] = {
                        "examples": [{"space_group": sg, "wyckoff": f"{mult}{letter}"}],
                        "multiplicities": sorted({int(mult)}),
                    }
                else:
                    orbit_meta[orbit_id]["examples"].append(
                        {"space_group": sg, "wyckoff": f"{mult}{letter}"}
                    )
                    ms = set(orbit_meta[orbit_id]["multiplicities"]) | {int(mult)}
                    orbit_meta[orbit_id]["multiplicities"] = sorted(ms)

            except Exception as e:
                # Be robust to odd cases; continue
                print(f"Warning: failed processing SG {sg} letter {letter}: {e}")
                continue

    unique_orbits = sorted(list(orbit_meta.keys()))
    print(f"Enumerated positions (sg, mult+letter): {len(enumerated_positions)}")
    print(f"Identified {len(unique_orbits)} unique Wyckoff orbits")
    return unique_orbits, orbit_meta, sg_letter_to_orbit


def main():
    """Generate complete token data for BIFROST tokenizer."""
    print("=" * 60)
    print("BIFROST tokenizer data generation")
    print("=" * 60)

    # Generate Wyckoff orbits
    print("Generating Wyckoff orbits (deduplicated across space groups)...")
    wyckoff_orbits, orbit_meta, sg_letter_to_orbit = generate_wyckoff_orbits()
    wyckoff_positions = wyckoff_orbits

    # Coordinate and lattice tokens are no longer generated (continuous values)
    coordinate_tokens: List[str] = []
    lattice_tokens: List[str] = []

    # Combine all tokens
    all_tokens = {"wyckoff_positions": wyckoff_positions}

    # Save to JSON file
    here = os.path.dirname(__file__)
    with open(os.path.join(here, "wyckoff_orbits_token_data.json"), "w") as f:
        json.dump(all_tokens, f, indent=2)
    # Save orbit metadata and mapping for tokenizer use
    with open(os.path.join(here, "wyckoff_orbits.json"), "w") as f:
        json.dump(orbit_meta, f, indent=2)
    with open(os.path.join(here, "wyckoff_map_sg_letter_to_orbit.json"), "w") as f:
        json.dump(sg_letter_to_orbit, f, indent=2)

    print("\n" + "=" * 60)
    print("TOKEN DATA SUMMARY")
    print("=" * 60)
    print(f"Wyckoff positions (orbits): {len(wyckoff_positions)}")
    print("\nSaved to: wyckoff_orbits_token_data.json")
    print("Also saved: wyckoff_orbits.json, wyckoff_map_sg_letter_to_orbit.json")


if __name__ == "__main__":
    main()
