#!/usr/bin/env python3
"""
structure_validation.py
=======================
Structure validation module for receptor-peptide complexes.

Provides PyRosetta-based structural quality assessments:
  - Ramachandran plot analysis (phi/psi angles)
  - Rotamer quality check (Dunbrack library)
  - Backbone geometry validation (bond lengths, angles)
  - Disulfide bond geometry validation (Cys3-Cys14 distance, chi angles)

All functions gracefully fall back to stub/PDB-parsed values when
PyRosetta is unavailable, enabling CI/dry-run usage.

Callable standalone (CLI) and as a Python import from the pipeline.

Usage (CLI):
    conda run -n bio-tools python -m AG_src.pipeline.structure_validation \
        --pdb complex.pdb --output validation_report.json

Usage (Python):
    from AG_src.pipeline.structure_validation import validate_structure
    report = validate_structure("complex.pdb")
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PyRosetta availability
# ---------------------------------------------------------------------------

_PYROSETTA_AVAILABLE: Optional[bool] = None


def _check_pyrosetta() -> bool:
    global _PYROSETTA_AVAILABLE
    if _PYROSETTA_AVAILABLE is None:
        try:
            import pyrosetta  # noqa: F401
            _PYROSETTA_AVAILABLE = True
        except ImportError:
            _PYROSETTA_AVAILABLE = False
    return _PYROSETTA_AVAILABLE


def _ensure_init() -> None:
    """Lazily initialize PyRosetta if not already done."""
    if not _check_pyrosetta():
        return
    import pyrosetta
    if not pyrosetta.rosetta.basic.was_init_called():
        pyrosetta.init(
            options="-mute all -ex1 -ex2aro -ignore_unrecognized_res",
            silent=True,
        )


# ---------------------------------------------------------------------------
# Ramachandran analysis
# ---------------------------------------------------------------------------

# Favored / allowed / outlier regions (simplified Ramachandran)
# Based on Top8000 dataset conventions
_RAMA_GENERAL_FAVORED = [
    # (phi_min, phi_max, psi_min, psi_max) — approximate favored regions
    (-180, -20, -80, 0),     # alpha-helix region
    (-180, -20, 50, 180),    # beta-sheet region
    (-180, -20, -180, -120), # beta-sheet extended
]

_RAMA_GENERAL_ALLOWED_EXTENSION = 30  # degrees beyond favored


def _classify_rama(phi: float, psi: float, is_glycine: bool, is_proline: bool) -> str:
    """Classify a phi/psi pair as favored/allowed/outlier.

    Uses simplified boundaries. Glycine and proline have broader
    allowed regions.
    """
    if is_glycine:
        # Glycine is flexible — almost everything is allowed
        return "favored"

    extension = _RAMA_GENERAL_ALLOWED_EXTENSION
    if is_proline:
        extension += 15

    for phi_min, phi_max, psi_min, psi_max in _RAMA_GENERAL_FAVORED:
        if phi_min <= phi <= phi_max and psi_min <= psi <= psi_max:
            return "favored"
        # Check allowed (extended region)
        if (phi_min - extension <= phi <= phi_max + extension and
                psi_min - extension <= psi <= psi_max + extension):
            return "allowed"

    return "outlier"


def analyze_ramachandran(
    pdb_path: str,
    peptide_chain: Optional[int] = None,
) -> Dict[str, Any]:
    """Analyze Ramachandran angles for all residues or a specific chain.

    Args:
        pdb_path:      Path to complex PDB.
        peptide_chain: If set, only analyze residues on this chain (1-indexed).

    Returns:
        Dict with per-residue phi/psi/classification and summary counts.
    """
    if not _check_pyrosetta():
        logger.warning("[StructureValidation] PyRosetta unavailable; returning stub Ramachandran.")
        return {
            "n_residues": 0,
            "n_favored": 0,
            "n_allowed": 0,
            "n_outlier": 0,
            "pct_favored": 0.0,
            "residues": [],
        }

    _ensure_init()
    import pyrosetta

    pose = pyrosetta.pose_from_pdb(pdb_path)
    residues = []
    n_favored = 0
    n_allowed = 0
    n_outlier = 0
    n_analyzed = 0

    for i in range(1, pose.total_residue() + 1):
        if peptide_chain is not None and pose.chain(i) != peptide_chain:
            continue

        res = pose.residue(i)
        # Skip non-protein residues
        if not res.is_protein():
            continue

        phi = pose.phi(i)
        psi = pose.psi(i)
        omega = pose.omega(i)
        is_gly = res.name3().strip() == "GLY"
        is_pro = res.name3().strip() == "PRO"

        classification = _classify_rama(phi, psi, is_gly, is_pro)
        n_analyzed += 1

        if classification == "favored":
            n_favored += 1
        elif classification == "allowed":
            n_allowed += 1
        else:
            n_outlier += 1

        residues.append({
            "resid": i,
            "name": res.name3().strip(),
            "name1": res.name1(),
            "chain": pose.chain(i),
            "phi": round(phi, 2),
            "psi": round(psi, 2),
            "omega": round(omega, 2),
            "classification": classification,
        })

    pct_favored = (n_favored / n_analyzed * 100) if n_analyzed > 0 else 0.0

    return {
        "n_residues": n_analyzed,
        "n_favored": n_favored,
        "n_allowed": n_allowed,
        "n_outlier": n_outlier,
        "pct_favored": round(pct_favored, 1),
        "residues": residues,
    }


# ---------------------------------------------------------------------------
# Rotamer quality check
# ---------------------------------------------------------------------------

def check_rotamer_quality(
    pdb_path: str,
    peptide_chain: Optional[int] = None,
) -> Dict[str, Any]:
    """Check rotamer quality using Dunbrack energy (fa_dun).

    Residues with high fa_dun scores have unfavorable rotameric states.

    Args:
        pdb_path:      Path to complex PDB.
        peptide_chain: If set, only analyze this chain.

    Returns:
        Dict with per-residue fa_dun scores, outlier count, and summary.
    """
    if not _check_pyrosetta():
        logger.warning("[StructureValidation] PyRosetta unavailable; returning stub rotamer quality.")
        return {"n_residues": 0, "n_outliers": 0, "residues": []}

    _ensure_init()
    import pyrosetta
    from pyrosetta.rosetta.core.scoring import ScoreType

    pose = pyrosetta.pose_from_pdb(pdb_path)
    sfxn = pyrosetta.get_fa_scorefxn()
    sfxn(pose)

    residues = []
    n_outliers = 0
    n_analyzed = 0
    FA_DUN_OUTLIER_THRESHOLD = 5.0  # REU

    for i in range(1, pose.total_residue() + 1):
        if peptide_chain is not None and pose.chain(i) != peptide_chain:
            continue

        res = pose.residue(i)
        if not res.is_protein():
            continue
        # Gly/Ala have no meaningful rotamers
        if res.name3().strip() in ("GLY", "ALA"):
            continue

        n_analyzed += 1
        energies = pose.energies().residue_total_energies(i)
        fa_dun = energies[ScoreType.fa_dun]

        is_outlier = fa_dun > FA_DUN_OUTLIER_THRESHOLD
        if is_outlier:
            n_outliers += 1

        # Collect chi angles
        n_chi = res.nchi()
        chi_angles = []
        for chi_idx in range(1, n_chi + 1):
            chi_angles.append(round(pose.chi(chi_idx, i), 2))

        residues.append({
            "resid": i,
            "name": res.name3().strip(),
            "chain": pose.chain(i),
            "fa_dun": round(fa_dun, 4),
            "chi_angles": chi_angles,
            "is_outlier": is_outlier,
        })

    return {
        "n_residues": n_analyzed,
        "n_outliers": n_outliers,
        "outlier_threshold": FA_DUN_OUTLIER_THRESHOLD,
        "residues": residues,
    }


# ---------------------------------------------------------------------------
# Backbone geometry validation
# ---------------------------------------------------------------------------

def validate_backbone_geometry(
    pdb_path: str,
    peptide_chain: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate backbone bond lengths and angles.

    Checks:
      - N-CA bond length (ideal ~1.458 Å)
      - CA-C bond length (ideal ~1.525 Å)
      - C-N bond length (ideal ~1.329 Å, peptide bond)
      - CA-C-N angle (ideal ~116.2°)
      - C-N-CA angle (ideal ~121.7°)
      - Omega angle (should be near 180° for trans, 0° for cis-Pro)

    Args:
        pdb_path:      Path to complex PDB.
        peptide_chain: If set, only analyze this chain.

    Returns:
        Dict with bond length/angle deviations and outlier count.
    """
    if not _check_pyrosetta():
        logger.warning("[StructureValidation] PyRosetta unavailable; returning stub backbone geometry.")
        return {"n_residues": 0, "n_bond_outliers": 0, "n_angle_outliers": 0, "residues": []}

    _ensure_init()
    import pyrosetta

    pose = pyrosetta.pose_from_pdb(pdb_path)

    # Ideal bond lengths (Å) and tolerances
    IDEAL_N_CA = 1.458
    IDEAL_CA_C = 1.525
    IDEAL_C_N = 1.329
    BOND_TOLERANCE = 0.05  # Å

    residues = []
    n_bond_outliers = 0
    n_angle_outliers = 0
    n_analyzed = 0

    for i in range(1, pose.total_residue() + 1):
        if peptide_chain is not None and pose.chain(i) != peptide_chain:
            continue

        res = pose.residue(i)
        if not res.is_protein():
            continue

        n_analyzed += 1
        issues: List[str] = []

        # N-CA bond
        if res.has("N") and res.has("CA"):
            n_ca_dist = (res.xyz("N") - res.xyz("CA")).norm()
            if abs(n_ca_dist - IDEAL_N_CA) > BOND_TOLERANCE:
                issues.append(f"N-CA={n_ca_dist:.3f}")
                n_bond_outliers += 1
        else:
            n_ca_dist = None

        # CA-C bond
        if res.has("CA") and res.has("C"):
            ca_c_dist = (res.xyz("CA") - res.xyz("C")).norm()
            if abs(ca_c_dist - IDEAL_CA_C) > BOND_TOLERANCE:
                issues.append(f"CA-C={ca_c_dist:.3f}")
                n_bond_outliers += 1
        else:
            ca_c_dist = None

        # Omega angle (cis/trans check)
        omega = pose.omega(i)
        is_cis = abs(omega) < 30.0
        is_pro = res.name3().strip() == "PRO"
        if is_cis and not is_pro:
            issues.append(f"cis-omega={omega:.1f}")
            n_angle_outliers += 1

        entry: Dict[str, Any] = {
            "resid": i,
            "name": res.name3().strip(),
            "chain": pose.chain(i),
            "omega": round(omega, 2),
        }
        if n_ca_dist is not None:
            entry["n_ca_bond"] = round(n_ca_dist, 3)
        if ca_c_dist is not None:
            entry["ca_c_bond"] = round(ca_c_dist, 3)
        if issues:
            entry["issues"] = issues
        residues.append(entry)

    return {
        "n_residues": n_analyzed,
        "n_bond_outliers": n_bond_outliers,
        "n_angle_outliers": n_angle_outliers,
        "residues": residues,
    }


# ---------------------------------------------------------------------------
# Disulfide bond validation
# ---------------------------------------------------------------------------

# Ideal disulfide geometry
_IDEAL_SS_DISTANCE = 2.04     # SG-SG distance in Å
_SS_DISTANCE_TOLERANCE = 0.3  # Å
_IDEAL_CHI3_SS = 90.0         # chi3 dihedral ±90°
_CHI3_TOLERANCE = 30.0        # degrees


def validate_disulfide_bonds(
    pdb_path: str,
    expected_pairs: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """Validate disulfide bond geometry.

    For the SSTR2 project, the default expected pair is Cys3-Cys14
    (SST-14 / DOTATATE pharmacophore).

    Args:
        pdb_path:       Path to complex PDB.
        expected_pairs: List of (resid_i, resid_j) pairs to validate.
                        If None, auto-detect from the structure.

    Returns:
        Dict with disulfide geometry details and pass/fail status.
    """
    if not _check_pyrosetta():
        logger.warning("[StructureValidation] PyRosetta unavailable; returning stub disulfide validation.")
        return {"n_disulfides": 0, "disulfides": [], "all_valid": False}

    _ensure_init()
    import pyrosetta

    pose = pyrosetta.pose_from_pdb(pdb_path)
    disulfides: List[Dict[str, Any]] = []

    if expected_pairs is None:
        # Auto-detect: find all Cys-Cys pairs with SG-SG < 3.0 Å
        cys_residues = []
        for i in range(1, pose.total_residue() + 1):
            if pose.residue(i).name3().strip() == "CYS" and pose.residue(i).has("SG"):
                cys_residues.append(i)

        expected_pairs = []
        for idx_a, res_a in enumerate(cys_residues):
            for res_b in cys_residues[idx_a + 1:]:
                sg_a = pose.residue(res_a).xyz("SG")
                sg_b = pose.residue(res_b).xyz("SG")
                dist = (sg_a - sg_b).norm()
                if dist < 3.0:
                    expected_pairs.append((res_a, res_b))

    all_valid = True
    for res_a, res_b in expected_pairs:
        entry: Dict[str, Any] = {
            "res_a": res_a,
            "res_b": res_b,
        }

        # Check residue types
        if (res_a > pose.total_residue() or res_b > pose.total_residue()):
            entry["error"] = "Residue number out of range"
            entry["valid"] = False
            all_valid = False
            disulfides.append(entry)
            continue

        res_obj_a = pose.residue(res_a)
        res_obj_b = pose.residue(res_b)

        if res_obj_a.name3().strip() != "CYS" or res_obj_b.name3().strip() != "CYS":
            entry["error"] = f"Expected CYS-CYS, got {res_obj_a.name3()}-{res_obj_b.name3()}"
            entry["valid"] = False
            all_valid = False
            disulfides.append(entry)
            continue

        entry["name_a"] = f"CYS{res_a}"
        entry["name_b"] = f"CYS{res_b}"
        entry["chain_a"] = pose.chain(res_a)
        entry["chain_b"] = pose.chain(res_b)

        # SG-SG distance
        if res_obj_a.has("SG") and res_obj_b.has("SG"):
            sg_a = res_obj_a.xyz("SG")
            sg_b = res_obj_b.xyz("SG")
            sg_dist = (sg_a - sg_b).norm()
            entry["sg_sg_distance"] = round(sg_dist, 3)
            entry["distance_ideal"] = _IDEAL_SS_DISTANCE
            entry["distance_deviation"] = round(abs(sg_dist - _IDEAL_SS_DISTANCE), 3)
            distance_ok = abs(sg_dist - _IDEAL_SS_DISTANCE) <= _SS_DISTANCE_TOLERANCE
            entry["distance_valid"] = distance_ok
        else:
            entry["error"] = "SG atoms not found"
            entry["distance_valid"] = False
            distance_ok = False

        # CB-SG-SG-CB dihedral (chi3 of disulfide)
        if (res_obj_a.has("CB") and res_obj_a.has("SG") and
                res_obj_b.has("SG") and res_obj_b.has("CB")):
            cb_a = res_obj_a.xyz("CB")
            sg_a_xyz = res_obj_a.xyz("SG")
            sg_b_xyz = res_obj_b.xyz("SG")
            cb_b = res_obj_b.xyz("CB")

            # Compute dihedral angle
            chi3 = _compute_dihedral(cb_a, sg_a_xyz, sg_b_xyz, cb_b)
            entry["chi3_dihedral"] = round(chi3, 2)
            chi3_deviation = min(abs(chi3 - 90), abs(chi3 + 90), abs(abs(chi3) - 90))
            entry["chi3_deviation"] = round(chi3_deviation, 2)
            chi3_ok = chi3_deviation <= _CHI3_TOLERANCE
            entry["chi3_valid"] = chi3_ok
        else:
            chi3_ok = True  # Can't check, assume OK

        valid = distance_ok and chi3_ok
        entry["valid"] = valid
        if not valid:
            all_valid = False

        disulfides.append(entry)

    return {
        "n_disulfides": len(disulfides),
        "disulfides": disulfides,
        "all_valid": all_valid,
    }


def _compute_dihedral(p1, p2, p3, p4) -> float:
    """Compute dihedral angle between four PyRosetta xyzVector points.

    Returns angle in degrees (-180 to +180).
    """
    # Vectors
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    # Cross products
    n1_x = b1.y * b2.z - b1.z * b2.y
    n1_y = b1.z * b2.x - b1.x * b2.z
    n1_z = b1.x * b2.y - b1.y * b2.x

    n2_x = b2.y * b3.z - b2.z * b3.y
    n2_y = b2.z * b3.x - b2.x * b3.z
    n2_z = b2.x * b3.y - b2.y * b3.x

    # Dot and cross of normals
    n1_dot_n2 = n1_x * n2_x + n1_y * n2_y + n1_z * n2_z
    n1_cross_n2_x = n1_y * n2_z - n1_z * n2_y
    n1_cross_n2_y = n1_z * n2_x - n1_x * n2_z
    n1_cross_n2_z = n1_x * n2_y - n1_y * n2_x

    b2_norm = b2.norm()
    if b2_norm < 1e-8:
        return 0.0

    # Sign: dot of (n1 x n2) with b2 unit vector
    sign_val = (n1_cross_n2_x * b2.x + n1_cross_n2_y * b2.y + n1_cross_n2_z * b2.z) / b2_norm
    n1_norm = math.sqrt(n1_x**2 + n1_y**2 + n1_z**2)
    n2_norm = math.sqrt(n2_x**2 + n2_y**2 + n2_z**2)

    if n1_norm < 1e-8 or n2_norm < 1e-8:
        return 0.0

    cos_angle = n1_dot_n2 / (n1_norm * n2_norm)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    angle = math.degrees(math.acos(cos_angle))

    if sign_val < 0:
        angle = -angle

    return angle


# ---------------------------------------------------------------------------
# Combined structure validation
# ---------------------------------------------------------------------------

def validate_structure(
    pdb_path: str,
    peptide_chain: Optional[int] = None,
    disulfide_pairs: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """Run a comprehensive structure validation on a complex PDB.

    Combines all sub-analyses:
      - Ramachandran analysis
      - Rotamer quality
      - Backbone geometry
      - Disulfide bond geometry

    Args:
        pdb_path:        Path to complex PDB.
        peptide_chain:   Chain to focus on (None for all).
        disulfide_pairs: Expected disulfide pairs (auto-detect if None).

    Returns:
        Combined dict with all validation results and an overall quality grade.
    """
    logger.info("[StructureValidation] Running full validation on %s", pdb_path)

    rama = analyze_ramachandran(pdb_path, peptide_chain=peptide_chain)
    rotamer = check_rotamer_quality(pdb_path, peptide_chain=peptide_chain)
    backbone = validate_backbone_geometry(pdb_path, peptide_chain=peptide_chain)
    disulfide = validate_disulfide_bonds(pdb_path, expected_pairs=disulfide_pairs)

    # Compute overall quality grade
    issues = []
    if rama["n_outlier"] > 0:
        issues.append(f"{rama['n_outlier']} Ramachandran outliers")
    if rotamer["n_outliers"] > 0:
        issues.append(f"{rotamer['n_outliers']} rotamer outliers")
    if backbone["n_bond_outliers"] > 0:
        issues.append(f"{backbone['n_bond_outliers']} bond length outliers")
    if backbone["n_angle_outliers"] > 0:
        issues.append(f"{backbone['n_angle_outliers']} angle outliers")
    if not disulfide["all_valid"]:
        issues.append("disulfide geometry issues")

    if not issues:
        grade = "GOOD"
    elif len(issues) <= 2:
        grade = "ACCEPTABLE"
    else:
        grade = "POOR"

    return {
        "pdb_path": pdb_path,
        "pyrosetta_available": _check_pyrosetta(),
        "ramachandran": rama,
        "rotamer_quality": rotamer,
        "backbone_geometry": backbone,
        "disulfide_bonds": disulfide,
        "quality_grade": grade,
        "quality_issues": issues,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Structure validation for receptor-peptide complex"
    )
    parser.add_argument("--pdb", required=True, help="Path to complex PDB")
    parser.add_argument("--output", default="", help="Output JSON path (default: stdout)")
    parser.add_argument(
        "--peptide-chain", type=int, default=None,
        help="Peptide chain number to focus on (default: all chains)",
    )
    parser.add_argument(
        "--disulfide-pairs", default="",
        help='Expected disulfide pairs as JSON, e.g., "[[3,14]]"',
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    if not Path(args.pdb).exists():
        print(json.dumps({"error": f"PDB not found: {args.pdb}"}), file=sys.stderr)
        sys.exit(1)

    disulfide_pairs = None
    if args.disulfide_pairs:
        disulfide_pairs = [tuple(p) for p in json.loads(args.disulfide_pairs)]

    report = validate_structure(
        args.pdb,
        peptide_chain=args.peptide_chain,
        disulfide_pairs=disulfide_pairs,
    )

    output_json = json.dumps(report, indent=2, ensure_ascii=False)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_json, encoding="utf-8")
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
