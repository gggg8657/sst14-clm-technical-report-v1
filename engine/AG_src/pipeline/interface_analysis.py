#!/usr/bin/env python3
"""
interface_analysis.py
=====================
Interface analysis module for receptor-peptide complexes.

Provides PyRosetta-based analyses of the binding interface:
  - Buried Surface Area (BSA / dSASA) calculation
  - Hydrogen bond counting at interface
  - Salt bridge detection (e.g., K9-D122)
  - Per-residue energy decomposition
  - Contact map generation

All functions gracefully fall back to stub values when PyRosetta is
unavailable, enabling CI/dry-run usage.

Callable standalone (CLI) and as a Python import from the pipeline.

Usage (CLI):
    conda run -n bio-tools python -m AG_src.pipeline.interface_analysis \
        --pdb complex.pdb --output interface_report.json

Usage (Python):
    from AG_src.pipeline.interface_analysis import analyze_interface
    report = analyze_interface("complex.pdb")
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
# Buried Surface Area (dSASA)
# ---------------------------------------------------------------------------

def compute_buried_surface_area(
    pdb_path: str,
    jump_id: int = 1,
) -> Dict[str, float]:
    """Compute buried surface area (dSASA) at the interface.

    Uses InterfaceAnalyzerMover to get:
      - interface_delta_sasa: total BSA (Å²)
      - interface_dG: interface binding energy (kcal/mol)

    Args:
        pdb_path: Path to receptor-peptide complex PDB.
        jump_id:  Jump ID separating receptor from peptide (default 1).

    Returns:
        Dict with 'delta_sasa', 'interface_dG', 'num_interface_residues'.
    """
    if not _check_pyrosetta():
        logger.warning("[InterfaceAnalysis] PyRosetta unavailable; returning stub BSA.")
        return {"delta_sasa": 0.0, "interface_dG": 0.0, "num_interface_residues": 0}

    _ensure_init()
    import pyrosetta
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

    pose = pyrosetta.pose_from_pdb(pdb_path)
    sfxn = pyrosetta.get_fa_scorefxn()
    sfxn(pose)

    iam = InterfaceAnalyzerMover(jump_id)
    iam.set_pack_input(True)
    iam.set_pack_separated(True)
    iam.apply(pose)

    return {
        "delta_sasa": round(iam.get_interface_delta_sasa(), 2),
        "interface_dG": round(iam.get_interface_dG(), 4),
        "num_interface_residues": iam.get_num_interface_residues(),
    }


# ---------------------------------------------------------------------------
# Hydrogen bond counting
# ---------------------------------------------------------------------------

def count_interface_hbonds(
    pdb_path: str,
    distance_cutoff: float = 8.0,
) -> Dict[str, Any]:
    """Count hydrogen bonds at the receptor-peptide interface.

    Args:
        pdb_path:        Path to complex PDB.
        distance_cutoff: CA-CA distance cutoff for interface residues (Å).

    Returns:
        Dict with 'total_hbonds', 'interface_hbonds', and per-pair details.
    """
    if not _check_pyrosetta():
        logger.warning("[InterfaceAnalysis] PyRosetta unavailable; returning stub hbond count.")
        return {"total_hbonds": 0, "interface_hbonds": 0, "hbond_details": []}

    _ensure_init()
    import pyrosetta
    from pyrosetta.rosetta.core.scoring.hbonds import HBondSet

    pose = pyrosetta.pose_from_pdb(pdb_path)
    sfxn = pyrosetta.get_fa_scorefxn()
    sfxn(pose)

    hbond_set = HBondSet()
    hbond_set.setup_for_residue_pair_energies(pose, False, False)

    total_hbonds = hbond_set.nhbonds()
    interface_hbonds = []

    for hb_idx in range(1, total_hbonds + 1):
        hb = hbond_set.hbond(hb_idx)
        don_res = hb.don_res()
        acc_res = hb.acc_res()
        don_chain = pose.chain(don_res)
        acc_chain = pose.chain(acc_res)

        # Interface = different chains
        if don_chain != acc_chain:
            interface_hbonds.append({
                "donor_res": don_res,
                "donor_name": pose.residue(don_res).name3(),
                "donor_chain": don_chain,
                "acceptor_res": acc_res,
                "acceptor_name": pose.residue(acc_res).name3(),
                "acceptor_chain": acc_chain,
                "energy": round(hb.energy(), 4),
            })

    return {
        "total_hbonds": total_hbonds,
        "interface_hbonds": len(interface_hbonds),
        "hbond_details": interface_hbonds,
    }


# ---------------------------------------------------------------------------
# Salt bridge detection
# ---------------------------------------------------------------------------

_POSITIVE_ATOMS = {"LYS": ["NZ"], "ARG": ["NH1", "NH2", "NE"]}
_NEGATIVE_ATOMS = {"ASP": ["OD1", "OD2"], "GLU": ["OE1", "OE2"]}

_SALT_BRIDGE_CUTOFF = 4.0  # Angstroms


def detect_salt_bridges(
    pdb_path: str,
    cutoff: float = _SALT_BRIDGE_CUTOFF,
) -> Dict[str, Any]:
    """Detect salt bridges at the receptor-peptide interface.

    A salt bridge is defined as a contact between a positively charged
    atom (Lys NZ, Arg NH1/NH2/NE) and a negatively charged atom
    (Asp OD1/OD2, Glu OE1/OE2) within cutoff distance, across chains.

    Args:
        pdb_path: Path to complex PDB.
        cutoff:   Distance cutoff for salt bridge (Å, default 4.0).

    Returns:
        Dict with 'n_salt_bridges' and list of salt bridge details.
    """
    if not _check_pyrosetta():
        logger.warning("[InterfaceAnalysis] PyRosetta unavailable; returning stub salt bridges.")
        return {"n_salt_bridges": 0, "salt_bridges": []}

    _ensure_init()
    import pyrosetta

    pose = pyrosetta.pose_from_pdb(pdb_path)
    salt_bridges: List[Dict[str, Any]] = []

    # Collect positive/negative residues with their atom coordinates
    pos_atoms: List[Tuple[int, str, Any]] = []  # (resid, atom_name, xyz)
    neg_atoms: List[Tuple[int, str, Any]] = []

    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        res_name = res.name3().strip()

        if res_name in _POSITIVE_ATOMS:
            for atom_name in _POSITIVE_ATOMS[res_name]:
                if res.has(atom_name):
                    pos_atoms.append((i, atom_name, res.xyz(atom_name)))

        if res_name in _NEGATIVE_ATOMS:
            for atom_name in _NEGATIVE_ATOMS[res_name]:
                if res.has(atom_name):
                    neg_atoms.append((i, atom_name, res.xyz(atom_name)))

    # Check all positive-negative pairs across chains
    for pos_resid, pos_atom, pos_xyz in pos_atoms:
        pos_chain = pose.chain(pos_resid)
        for neg_resid, neg_atom, neg_xyz in neg_atoms:
            neg_chain = pose.chain(neg_resid)
            if pos_chain == neg_chain:
                continue
            dist = (pos_xyz - neg_xyz).norm()
            if dist <= cutoff:
                salt_bridges.append({
                    "positive_res": pos_resid,
                    "positive_name": pose.residue(pos_resid).name3().strip(),
                    "positive_atom": pos_atom,
                    "positive_chain": pos_chain,
                    "negative_res": neg_resid,
                    "negative_name": pose.residue(neg_resid).name3().strip(),
                    "negative_atom": neg_atom,
                    "negative_chain": neg_chain,
                    "distance": round(dist, 3),
                })

    return {
        "n_salt_bridges": len(salt_bridges),
        "salt_bridges": salt_bridges,
    }


# ---------------------------------------------------------------------------
# Per-residue energy decomposition
# ---------------------------------------------------------------------------

def per_residue_energy(
    pdb_path: str,
    score_function: str = "ref2015",
) -> Dict[str, Any]:
    """Compute per-residue energy decomposition.

    Args:
        pdb_path:       Path to complex PDB.
        score_function: Score function name ("ref2015" or "beta_nov16").

    Returns:
        Dict with 'residues' list (each having resid, name, chain, total_energy,
        and per-term breakdown) and 'total_score'.
    """
    if not _check_pyrosetta():
        logger.warning("[InterfaceAnalysis] PyRosetta unavailable; returning stub energy decomposition.")
        return {"total_score": 0.0, "residues": []}

    _ensure_init()
    import pyrosetta
    from pyrosetta.rosetta.core.scoring import ScoreType

    pose = pyrosetta.pose_from_pdb(pdb_path)

    if score_function == "beta_nov16":
        sfxn = pyrosetta.create_score_function("beta_nov16")
    else:
        sfxn = pyrosetta.get_fa_scorefxn()

    total_score = sfxn(pose)

    key_terms = [
        ScoreType.fa_atr,
        ScoreType.fa_rep,
        ScoreType.fa_sol,
        ScoreType.fa_elec,
        ScoreType.hbond_sr_bb,
        ScoreType.hbond_lr_bb,
        ScoreType.hbond_bb_sc,
        ScoreType.hbond_sc,
        ScoreType.rama_prepro,
        ScoreType.fa_dun,
    ]

    residues = []
    for i in range(1, pose.total_residue() + 1):
        total_e = pose.energies().residue_total_energy(i)
        energies = pose.energies().residue_total_energies(i)

        term_dict = {}
        for st in key_terms:
            val = energies[st]
            if abs(val) > 1e-6:
                term_dict[st.name] = round(val, 4)

        residues.append({
            "resid": i,
            "name": pose.residue(i).name3().strip(),
            "name1": pose.residue(i).name1(),
            "chain": pose.chain(i),
            "total_energy": round(total_e, 4),
            "terms": term_dict,
        })

    return {
        "total_score": round(total_score, 4),
        "score_function": score_function,
        "residues": residues,
    }


# ---------------------------------------------------------------------------
# Contact map generation
# ---------------------------------------------------------------------------

def generate_contact_map(
    pdb_path: str,
    distance_cutoff: float = 8.0,
    interface_only: bool = True,
) -> Dict[str, Any]:
    """Generate a residue-residue contact map.

    For interface_only=True, only reports contacts between different chains.

    Args:
        pdb_path:        Path to complex PDB.
        distance_cutoff: CB-CB distance cutoff for contacts (Å).
        interface_only:  If True, only report inter-chain contacts.

    Returns:
        Dict with 'contacts' list and 'n_contacts'.
    """
    if not _check_pyrosetta():
        logger.warning("[InterfaceAnalysis] PyRosetta unavailable; returning stub contact map.")
        return {"n_contacts": 0, "contacts": []}

    _ensure_init()
    import pyrosetta

    pose = pyrosetta.pose_from_pdb(pdb_path)
    contacts: List[Dict[str, Any]] = []
    n_res = pose.total_residue()

    for i in range(1, n_res + 1):
        res_i = pose.residue(i)
        chain_i = pose.chain(i)
        # Use CB (or CA for Gly)
        atom_i = "CB" if res_i.has("CB") else "CA"
        xyz_i = res_i.xyz(atom_i)

        for j in range(i + 1, n_res + 1):
            chain_j = pose.chain(j)
            if interface_only and chain_i == chain_j:
                continue

            res_j = pose.residue(j)
            atom_j = "CB" if res_j.has("CB") else "CA"
            xyz_j = res_j.xyz(atom_j)

            dist = (xyz_i - xyz_j).norm()
            if dist <= distance_cutoff:
                contacts.append({
                    "res_i": i,
                    "name_i": res_i.name3().strip(),
                    "chain_i": chain_i,
                    "res_j": j,
                    "name_j": res_j.name3().strip(),
                    "chain_j": chain_j,
                    "distance": round(dist, 3),
                })

    return {
        "n_contacts": len(contacts),
        "distance_cutoff": distance_cutoff,
        "interface_only": interface_only,
        "contacts": contacts,
    }


# ---------------------------------------------------------------------------
# Combined interface analysis
# ---------------------------------------------------------------------------

def analyze_interface(
    pdb_path: str,
    score_function: str = "ref2015",
    contact_cutoff: float = 8.0,
    salt_bridge_cutoff: float = 4.0,
) -> Dict[str, Any]:
    """Run a comprehensive interface analysis on a complex PDB.

    Combines all sub-analyses into a single report:
      - Buried Surface Area
      - Hydrogen bonds
      - Salt bridges
      - Per-residue energy decomposition
      - Interface contact map

    Args:
        pdb_path:           Path to complex PDB.
        score_function:     "ref2015" or "beta_nov16".
        contact_cutoff:     CB-CB distance cutoff for contacts.
        salt_bridge_cutoff: Distance cutoff for salt bridges.

    Returns:
        Combined dict with all analysis results.
    """
    logger.info("[InterfaceAnalysis] Running full analysis on %s", pdb_path)

    bsa = compute_buried_surface_area(pdb_path)
    hbonds = count_interface_hbonds(pdb_path)
    salt = detect_salt_bridges(pdb_path, cutoff=salt_bridge_cutoff)
    energy = per_residue_energy(pdb_path, score_function=score_function)
    contacts = generate_contact_map(
        pdb_path, distance_cutoff=contact_cutoff, interface_only=True
    )

    return {
        "pdb_path": pdb_path,
        "pyrosetta_available": _check_pyrosetta(),
        "buried_surface_area": bsa,
        "hydrogen_bonds": hbonds,
        "salt_bridges": salt,
        "per_residue_energy": energy,
        "contact_map": contacts,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interface analysis for receptor-peptide complex"
    )
    parser.add_argument("--pdb", required=True, help="Path to complex PDB")
    parser.add_argument("--output", default="", help="Output JSON path (default: stdout)")
    parser.add_argument(
        "--score-function", default="ref2015",
        choices=["ref2015", "beta_nov16"],
        help="Score function to use (default: ref2015)",
    )
    parser.add_argument(
        "--contact-cutoff", type=float, default=8.0,
        help="Contact distance cutoff in Angstroms (default: 8.0)",
    )
    parser.add_argument(
        "--salt-bridge-cutoff", type=float, default=4.0,
        help="Salt bridge distance cutoff in Angstroms (default: 4.0)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    if not Path(args.pdb).exists():
        print(json.dumps({"error": f"PDB not found: {args.pdb}"}), file=sys.stderr)
        sys.exit(1)

    report = analyze_interface(
        args.pdb,
        score_function=args.score_function,
        contact_cutoff=args.contact_cutoff,
        salt_bridge_cutoff=args.salt_bridge_cutoff,
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
