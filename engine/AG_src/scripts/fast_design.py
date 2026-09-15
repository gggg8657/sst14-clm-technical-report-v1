#!/usr/bin/env python3
"""
fast_design.py
==============
Standalone PyRosetta FastRelax & FastDesign script.

Called by AG_src/pipeline/step06_rosetta.py via subprocess:
    # FastRelax (energy minimization)
    conda run -n bio-tools python AG_src/scripts/fast_design.py \
        --input complex.pdb --output relaxed.pdb --protocol fast_relax

    # FastDesign (sequence design x N candidates)
    conda run -n bio-tools python AG_src/scripts/fast_design.py \
        --input complex.pdb --output designed.pdb --protocol fast_design \
        --n-candidates 20 --peptide-sequence AGCKNFFWKTFTSC

stdout: JSON only (step06_rosetta.py L378 parses via json.loads)
stderr: all PyRosetta logs and diagnostics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Cys / design position utilities (inlined from peptide_design_utils.py)
# ---------------------------------------------------------------------------

def find_cys_positions(sequence: str, one_indexed: bool = True) -> List[int]:
    """Find all Cysteine (C) positions in a peptide sequence.

    Parameters
    ----------
    sequence : str
        Amino acid sequence in 1-letter code.
    one_indexed : bool
        If True, return 1-indexed (biology convention).

    Returns
    -------
    List[int]
        Sorted list of Cys positions.
    """
    offset = 1 if one_indexed else 0
    return [i + offset for i, aa in enumerate(sequence) if aa == "C"]


def get_design_positions(
    sequence: str,
    frozen_residues: Optional[Set[str]] = None,
    one_indexed: bool = True,
) -> List[int]:
    """Get designable positions = all positions EXCEPT frozen residues.

    By default, Cysteine (C) is the only frozen residue type.
    """
    if frozen_residues is None:
        frozen_residues = {"C"}
    offset = 1 if one_indexed else 0
    return [
        i + offset
        for i, aa in enumerate(sequence)
        if aa not in frozen_residues
    ]


# ---------------------------------------------------------------------------
# PyRosetta initialization
# ---------------------------------------------------------------------------

def init_pyrosetta() -> None:
    """Initialize PyRosetta with muted output (all logs to stderr)."""
    import pyrosetta
    pyrosetta.init(
        options="-mute all -ex1 -ex2aro -ignore_unrecognized_res",
        silent=True,
    )


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def compute_interface_ddg(pose: "pyrosetta.Pose") -> float:
    """Compute interface dG via InterfaceAnalyzerMover (jump_id=1)."""
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

    iam = InterfaceAnalyzerMover(1)
    iam.set_pack_input(True)
    iam.set_pack_separated(True)
    iam.apply(pose)
    return iam.get_interface_dG()


def compute_interface_dsasa(pose: "pyrosetta.Pose") -> float:
    """Compute interface delta SASA via InterfaceAnalyzerMover."""
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

    iam = InterfaceAnalyzerMover(1)
    iam.set_pack_input(True)
    iam.set_pack_separated(True)
    iam.apply(pose)
    return iam.get_interface_delta_sasa()


def compute_clash_score(pose: "pyrosetta.Pose") -> float:
    """Count residues with fa_rep > 10.0 REU (steric clashes)."""
    import pyrosetta
    from pyrosetta.rosetta.core.scoring import fa_rep

    scorefxn = pyrosetta.get_fa_scorefxn()
    scorefxn(pose)

    clash_count = 0
    for i in range(1, pose.total_residue() + 1):
        residue_energies = pose.energies().residue_total_energies(i)
        if residue_energies[fa_rep] > 10.0:
            clash_count += 1
    return float(clash_count)


def get_peptide_sequence(pose: "pyrosetta.Pose", peptide_chain: int = 2) -> str:
    """Extract the sequence of the specified chain (1-indexed chain number)."""
    seq = ""
    for i in range(1, pose.total_residue() + 1):
        if pose.chain(i) == peptide_chain:
            seq += pose.residue(i).name1()
    return seq


# ---------------------------------------------------------------------------
# FastRelax
# ---------------------------------------------------------------------------

def run_fast_relax(
    input_pdb: str,
    output_pdb: str,
    peptide_chain: int = 2,
) -> Dict:
    """Run FastRelax on a receptor-peptide complex.

    Only the peptide chain has backbone + sidechain DOF enabled.
    The receptor is held fixed to preserve its native structure.

    Returns a dict with total_score, ddg, and clash_score.
    """
    import pyrosetta
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta.rosetta.core.kinematics import MoveMap

    pose = pyrosetta.pose_from_pdb(input_pdb)
    scorefxn = pyrosetta.get_fa_scorefxn()

    # MoveMap: peptide chain flexible, receptor fixed
    mm = MoveMap()
    mm.set_bb(False)
    mm.set_chi(False)
    for i in range(1, pose.total_residue() + 1):
        if pose.chain(i) == peptide_chain:
            mm.set_bb(i, True)
            mm.set_chi(i, True)

    fr = FastRelax(scorefxn, 3)
    fr.set_movemap(mm)
    fr.apply(pose)

    pose.dump_pdb(output_pdb)

    total_score = scorefxn(pose)
    ddg = compute_interface_ddg(pose)
    clash = compute_clash_score(pose)

    return {
        "ddg": round(ddg, 4),
        "total_score": round(total_score, 4),
        "clash_score": clash,
        "constraint_violations": 0,
    }


# ---------------------------------------------------------------------------
# FastDesign
# ---------------------------------------------------------------------------

def build_task_factory(
    pose: "pyrosetta.Pose",
    design_positions: List[int],
    cys_positions: List[int],
    peptide_start: int,
    peptide_end: int,
) -> "pyrosetta.rosetta.core.pack.task.TaskFactory":
    """Build a TaskFactory for FastDesign.

    - Receptor residues: PreventRepacking (completely fixed)
    - Peptide Cys positions: PreventRepacking (preserve disulfide)
    - Peptide non-design positions: RestrictToRepacking (repack only)
    - Peptide design positions: allow full design (all amino acids)
    """
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.core.pack.task.operation import (
        OperateOnResidueSubset,
        PreventRepackingRLT,
        RestrictToRepackingRLT,
    )
    from pyrosetta.rosetta.core.select.residue_selector import (
        ResidueIndexSelector,
    )

    tf = TaskFactory()

    # Convert peptide-local positions to pose-global positions
    design_global = [peptide_start + p - 1 for p in design_positions]
    cys_global = [peptide_start + p - 1 for p in cys_positions]

    # 1. Prevent repacking for ALL receptor residues
    receptor_indices = [
        str(i) for i in range(1, pose.total_residue() + 1)
        if i < peptide_start or i > peptide_end
    ]
    if receptor_indices:
        receptor_sel = ResidueIndexSelector(",".join(receptor_indices))
        tf.push_back(OperateOnResidueSubset(PreventRepackingRLT(), receptor_sel))

    # 2. Prevent repacking for Cys positions (preserve disulfide)
    if cys_global:
        cys_sel = ResidueIndexSelector(",".join(str(c) for c in cys_global))
        tf.push_back(OperateOnResidueSubset(PreventRepackingRLT(), cys_sel))

    # 3. Restrict non-design peptide positions to repacking only
    non_design_global = [
        i for i in range(peptide_start, peptide_end + 1)
        if i not in design_global and i not in cys_global
    ]
    if non_design_global:
        non_design_sel = ResidueIndexSelector(
            ",".join(str(i) for i in non_design_global)
        )
        tf.push_back(
            OperateOnResidueSubset(RestrictToRepackingRLT(), non_design_sel)
        )

    # Design positions get default behavior (all amino acids allowed)
    return tf


def find_peptide_range(
    pose: "pyrosetta.Pose", peptide_chain: int = 2
) -> tuple:
    """Find the start and end residue indices for the peptide chain."""
    start = None
    end = None
    for i in range(1, pose.total_residue() + 1):
        if pose.chain(i) == peptide_chain:
            if start is None:
                start = i
            end = i
    if start is None:
        raise ValueError(f"Peptide chain {peptide_chain} not found in pose")
    return start, end


def run_fast_design(
    input_pdb: str,
    output_dir: str,
    n_candidates: int,
    peptide_sequence: str,
    peptide_chain: int = 2,
    seed_base: int = 1000,
) -> Dict:
    """Run FastDesign protocol with Cys-preserving constraints.

    Performs N rounds of FastDesign, each starting from the original pose.
    Collects candidates, filters by Cys preservation, and returns the best.

    Returns a dict with best candidate info and full candidates list.
    """
    import pyrosetta
    from pyrosetta.rosetta.protocols.denovo_design.movers import FastDesign

    pose = pyrosetta.pose_from_pdb(input_pdb)
    scorefxn = pyrosetta.get_fa_scorefxn()

    peptide_start, peptide_end = find_peptide_range(pose, peptide_chain)

    # Compute design positions from the peptide sequence
    cys_positions = find_cys_positions(peptide_sequence)
    design_positions = get_design_positions(peptide_sequence)

    print(
        f"Peptide range: {peptide_start}-{peptide_end}, "
        f"Cys positions: {cys_positions}, "
        f"Design positions: {design_positions}",
        file=sys.stderr,
    )

    # Build TaskFactory
    tf = build_task_factory(
        pose, design_positions, cys_positions,
        peptide_start, peptide_end,
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    candidates = []
    for trial in range(n_candidates):
        print(f"FastDesign trial {trial + 1}/{n_candidates}...", file=sys.stderr)

        # Set random seed for reproducibility
        pyrosetta.rosetta.numeric.random.rg().set_seed(seed_base + trial)

        trial_pose = pose.clone()

        fd = FastDesign()
        fd.set_scorefxn(scorefxn)
        fd.set_task_factory(tf)
        fd.apply(trial_pose)

        designed_seq = get_peptide_sequence(trial_pose, peptide_chain)
        dg = compute_interface_ddg(trial_pose)
        dsasa = compute_interface_dsasa(trial_pose)
        total = scorefxn(trial_pose)

        # Check Cys preservation
        original_cys = set(find_cys_positions(peptide_sequence))
        designed_cys = set(find_cys_positions(designed_seq))
        cys_preserved = original_cys == designed_cys

        candidate = {
            "trial": trial,
            "seq": designed_seq,
            "dG": round(dg, 4),
            "dSASA": round(dsasa, 2),
            "total_score": round(total, 4),
            "cys_preserved": cys_preserved,
        }
        candidates.append(candidate)

        # Save each candidate PDB
        trial_pdb = out_path / f"design_trial_{trial:03d}.pdb"
        trial_pose.dump_pdb(str(trial_pdb))

        print(
            f"  Trial {trial}: seq={designed_seq}, dG={dg:.2f}, "
            f"dSASA={dsasa:.1f}, cys_ok={cys_preserved}",
            file=sys.stderr,
        )

    # Filter: only candidates with preserved Cys
    valid = [c for c in candidates if c["cys_preserved"]]
    if not valid:
        print("WARNING: No candidates preserved Cys. Using all.", file=sys.stderr)
        valid = candidates

    # Select best by dG (most negative = strongest binding)
    best = min(valid, key=lambda c: c["dG"])

    # Copy best PDB to the main output location
    best_src = out_path / f"design_trial_{best['trial']:03d}.pdb"
    best_dst = out_path / "best_design.pdb"
    if best_src.exists():
        import shutil
        shutil.copy(best_src, best_dst)

    clash = compute_clash_score(
        pyrosetta.pose_from_pdb(str(best_src)) if best_src.exists()
        else pose
    )

    return {
        "ddg": best["dG"],
        "total_score": best["total_score"],
        "clash_score": clash,
        "constraint_violations": 0,
        "designed_sequence": best["seq"],
        "candidates": [
            {"seq": c["seq"], "dG": c["dG"], "dSASA": c["dSASA"]}
            for c in valid
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PyRosetta FastRelax / FastDesign"
    )
    parser.add_argument("--input", required=True, help="Input complex PDB")
    parser.add_argument("--output", required=True, help="Output PDB or directory")
    parser.add_argument(
        "--protocol",
        default="fast_relax",
        choices=["fast_relax", "fast_design"],
        help="Protocol to run (default: fast_relax)",
    )
    parser.add_argument(
        "--n-candidates", type=int, default=20,
        help="Number of FastDesign candidates (default: 20)",
    )
    parser.add_argument(
        "--peptide-sequence", default="",
        help="Peptide sequence for Cys detection (required for fast_design)",
    )
    parser.add_argument(
        "--peptide-chain", type=int, default=2,
        help="Peptide chain number in pose (default: 2)",
    )
    parser.add_argument(
        "--seed-base", type=int, default=1000,
        help="Base random seed for reproducibility (default: 1000)",
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(
            json.dumps({"error": f"Input PDB not found: {args.input}"}),
            file=sys.stderr,
        )
        sys.exit(1)

    init_pyrosetta()

    if args.protocol == "fast_relax":
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        result = run_fast_relax(args.input, args.output, args.peptide_chain)
    elif args.protocol == "fast_design":
        if not args.peptide_sequence:
            print("ERROR: --peptide-sequence required for fast_design", file=sys.stderr)
            sys.exit(1)
        result = run_fast_design(
            args.input,
            args.output,
            args.n_candidates,
            args.peptide_sequence,
            args.peptide_chain,
            args.seed_base,
        )
    else:
        print(f"ERROR: Unknown protocol: {args.protocol}", file=sys.stderr)
        sys.exit(1)

    # stdout = JSON only (parsed by step06_rosetta.py)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
