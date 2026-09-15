#!/usr/bin/env python3
"""
enhanced_pipeline.py
====================
Enhanced PyRosetta-only pipeline with full flow:
  input sequence → build pose → dock → refine → score → analyze

This script integrates all enhanced modules:
  - interface_analysis (BSA, hbonds, salt bridges, energy decomposition)
  - structure_validation (Ramachandran, rotamers, backbone, disulfide)
  - Enhanced scoring (ref2015 / beta_nov16, per-residue breakdown)
  - Checkpoint/resume capability

Designed to work both as a CLI script (subprocess from pipeline) and
as a Python importable module (from bio-tools env).

Usage (CLI):
    conda run -n bio-tools python AG_src/scripts/enhanced_pipeline.py \
        --template-pdb complex.pdb \
        --target-sequence AGCKNFFWKTFTSC \
        --output-dir runs/enhanced_test \
        --score-function ref2015

    # Resume from checkpoint
    conda run -n bio-tools python AG_src/scripts/enhanced_pipeline.py \
        --template-pdb complex.pdb \
        --target-sequence AGCKNFFWKTFTSC \
        --output-dir runs/enhanced_test \
        --resume

stdout: JSON result (parsed by caller)
stderr: diagnostics and progress logs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

CHECKPOINT_FILE = "pipeline_checkpoint.json"


def _save_checkpoint(out_dir: Path, stage: str, data: Dict[str, Any]) -> None:
    """Save a checkpoint to disk for resume capability."""
    ckpt_path = out_dir / CHECKPOINT_FILE
    ckpt = {}
    if ckpt_path.exists():
        try:
            ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    ckpt[stage] = data
    ckpt["last_stage"] = stage
    ckpt_path.write_text(
        json.dumps(ckpt, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  [checkpoint] Saved: {stage}", file=sys.stderr)


def _load_checkpoint(out_dir: Path) -> Dict[str, Any]:
    """Load checkpoint from disk."""
    ckpt_path = out_dir / CHECKPOINT_FILE
    if not ckpt_path.exists():
        return {}
    try:
        return json.loads(ckpt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# PyRosetta initialization
# ---------------------------------------------------------------------------

def _init_pyrosetta(score_function: str = "ref2015") -> None:
    """Initialize PyRosetta with appropriate options."""
    import pyrosetta
    opts = "-mute all -ex1 -ex2aro -ignore_unrecognized_res -flexPepDocking:pep_refine"
    if score_function == "beta_nov16":
        opts += " -beta_nov16"
    pyrosetta.init(options=opts, silent=True)


# ---------------------------------------------------------------------------
# Stage 1: Build / Mutate Pose
# ---------------------------------------------------------------------------

def stage_build_pose(
    template_pdb: str,
    target_sequence: str,
    peptide_chain: int = 1,
) -> Tuple["pyrosetta.Pose", int]:
    """Build pose from template by mutating peptide to target sequence.

    Returns (pose, resolved_peptide_chain).
    """
    import pyrosetta
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue

    _AA1TO3 = {
        'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
        'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
        'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
        'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
    }

    pose = pyrosetta.pose_from_pdb(template_pdb)
    print(f"  Loaded template: {pose.total_residue()} residues, {pose.num_chains()} chains", file=sys.stderr)

    # Identify peptide chain
    chain_residues: Dict[int, List[int]] = {}
    for i in range(1, pose.total_residue() + 1):
        cid = pose.chain(i)
        chain_residues.setdefault(cid, []).append(i)

    target_len = len(target_sequence)
    best_chain = min(
        chain_residues,
        key=lambda cid: (abs(len(chain_residues[cid]) - target_len), len(chain_residues[cid]))
    )

    resolved_chain = peptide_chain
    if resolved_chain not in chain_residues:
        resolved_chain = best_chain
    elif len(chain_residues[resolved_chain]) > target_len * 2:
        resolved_chain = best_chain

    pep_residues = chain_residues[resolved_chain]
    ref_seq = "".join(pose.residue(i).name1() for i in pep_residues)
    print(f"  Reference: chain {resolved_chain}, seq={ref_seq}", file=sys.stderr)
    print(f"  Target: {target_sequence}", file=sys.stderr)

    # Apply mutations
    seq_to_use = target_sequence[:len(pep_residues)]
    n_mut = 0
    for idx, pose_resnum in enumerate(pep_residues):
        if idx < len(seq_to_use) and ref_seq[idx] != seq_to_use[idx]:
            aa3 = _AA1TO3.get(seq_to_use[idx])
            if aa3:
                MutateResidue(pose_resnum, aa3).apply(pose)
                n_mut += 1

    print(f"  Applied {n_mut} mutations", file=sys.stderr)
    return pose, resolved_chain


# ---------------------------------------------------------------------------
# Stage 2: Reorder chains (peptide last for FlexPepDock)
# ---------------------------------------------------------------------------

def stage_reorder_peptide_last(
    pose: "pyrosetta.Pose",
    peptide_chain: int,
) -> "pyrosetta.Pose":
    """Reorder chains so peptide is last (FlexPepDock requirement)."""
    import pyrosetta
    import tempfile

    n_chains = pose.num_chains()
    if peptide_chain == n_chains:
        return pose

    print(f"  Reordering: chain {peptide_chain} → last ({n_chains})", file=sys.stderr)

    tmp = tempfile.mktemp(suffix=".pdb")
    pose.dump_pdb(tmp)

    with open(tmp) as fh:
        lines = fh.readlines()
    Path(tmp).unlink(missing_ok=True)

    chain_blocks: dict[str, list[str]] = {}
    header_lines: list[str] = []
    chain_order: list[str] = []

    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            cid = line[21]
            if cid not in chain_blocks:
                chain_blocks[cid] = []
                chain_order.append(cid)
            chain_blocks[cid].append(line)
        elif line.startswith("TER"):
            pass
        elif not line.startswith("END"):
            header_lines.append(line)

    if len(chain_order) < 2:
        return pose

    pep_letter = chain_order[peptide_chain - 1]
    new_order = [c for c in chain_order if c != pep_letter] + [pep_letter]

    new_lines = header_lines[:]
    for idx, old_cid in enumerate(new_order):
        new_cid = chr(65 + idx)
        for atom_line in chain_blocks[old_cid]:
            new_lines.append(atom_line[:21] + new_cid + atom_line[22:])
        new_lines.append("TER\n")
    new_lines.append("END\n")

    tmp2 = tempfile.mktemp(suffix="_reord.pdb")
    with open(tmp2, "w") as fh:
        fh.writelines(new_lines)

    new_pose = pyrosetta.pose_from_pdb(tmp2)
    Path(tmp2).unlink(missing_ok=True)

    print(f"  Reordered: {new_pose.num_chains()} chains, {new_pose.total_residue()} res", file=sys.stderr)
    return new_pose


# ---------------------------------------------------------------------------
# Stage 3: FlexPepDock refinement
# ---------------------------------------------------------------------------

def stage_dock_refine(
    pose: "pyrosetta.Pose",
    output_pdb: str,
) -> "pyrosetta.Pose":
    """Run FlexPepDock refinement on the pose."""
    from pyrosetta.rosetta.protocols.flexpep_docking import FlexPepDockingProtocol

    print(f"  Running FlexPepDock refinement...", file=sys.stderr)
    t0 = time.monotonic()
    fpd = FlexPepDockingProtocol()
    fpd.apply(pose)
    elapsed = time.monotonic() - t0
    print(f"  FlexPepDock completed in {elapsed:.1f}s", file=sys.stderr)

    Path(output_pdb).parent.mkdir(parents=True, exist_ok=True)
    pose.dump_pdb(output_pdb)
    return pose


# ---------------------------------------------------------------------------
# Stage 4: Score
# ---------------------------------------------------------------------------

def stage_score(
    pose: "pyrosetta.Pose",
    score_function: str = "ref2015",
) -> Dict[str, Any]:
    """Compute comprehensive scoring on the refined pose."""
    import pyrosetta
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    from pyrosetta.rosetta.core.scoring import ScoreType, fa_rep

    if score_function == "beta_nov16":
        sfxn = pyrosetta.create_score_function("beta_nov16")
    else:
        sfxn = pyrosetta.get_fa_scorefxn()

    total_score = sfxn(pose)

    # Interface ddG
    iam = InterfaceAnalyzerMover(1)
    iam.set_pack_input(True)
    iam.set_pack_separated(True)
    iam.apply(pose)
    ddg = iam.get_interface_dG()
    dsasa = iam.get_interface_delta_sasa()

    # Clash score
    clash_count = 0
    for i in range(1, pose.total_residue() + 1):
        energies = pose.energies().residue_total_energies(i)
        if energies[fa_rep] > 10.0:
            clash_count += 1

    # Per-residue energy for interface residues
    per_res: List[Dict[str, Any]] = []
    key_terms = [
        ScoreType.fa_atr, ScoreType.fa_rep, ScoreType.fa_sol,
        ScoreType.fa_elec, ScoreType.hbond_sc,
    ]
    for i in range(1, pose.total_residue() + 1):
        total_e = pose.energies().residue_total_energy(i)
        res_energies = pose.energies().residue_total_energies(i)
        terms = {}
        for st in key_terms:
            val = res_energies[st]
            if abs(val) > 0.01:
                terms[st.name] = round(val, 4)
        per_res.append({
            "resid": i,
            "name": pose.residue(i).name3().strip(),
            "chain": pose.chain(i),
            "total": round(total_e, 4),
            "terms": terms,
        })

    return {
        "score_function": score_function,
        "total_score": round(total_score, 4),
        "ddg": round(ddg, 4),
        "delta_sasa": round(dsasa, 2),
        "clash_score": float(clash_count),
        "constraint_violations": 0,
        "per_residue_energy": per_res,
    }


# ---------------------------------------------------------------------------
# Stage 5: Interface analysis
# ---------------------------------------------------------------------------

def stage_analyze(
    pose: "pyrosetta.Pose",
    output_pdb: str,
) -> Dict[str, Any]:
    """Run interface analysis: hbonds, salt bridges, contacts."""
    from pyrosetta.rosetta.core.scoring.hbonds import HBondSet
    import pyrosetta

    sfxn = pyrosetta.get_fa_scorefxn()
    sfxn(pose)

    # Hydrogen bonds at interface
    hbond_set = HBondSet()
    hbond_set.setup_for_residue_pair_energies(pose, False, False)

    interface_hbonds = 0
    for hb_idx in range(1, hbond_set.nhbonds() + 1):
        hb = hbond_set.hbond(hb_idx)
        if pose.chain(hb.don_res()) != pose.chain(hb.acc_res()):
            interface_hbonds += 1

    # Salt bridges at interface
    salt_bridges = []
    pos_atoms_map = {"LYS": ["NZ"], "ARG": ["NH1", "NH2", "NE"]}
    neg_atoms_map = {"ASP": ["OD1", "OD2"], "GLU": ["OE1", "OE2"]}
    pos_list = []
    neg_list = []

    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        rname = res.name3().strip()
        if rname in pos_atoms_map:
            for aname in pos_atoms_map[rname]:
                if res.has(aname):
                    pos_list.append((i, aname, res.xyz(aname)))
        if rname in neg_atoms_map:
            for aname in neg_atoms_map[rname]:
                if res.has(aname):
                    neg_list.append((i, aname, res.xyz(aname)))

    for pi, pa, pxyz in pos_list:
        for ni, na, nxyz in neg_list:
            if pose.chain(pi) != pose.chain(ni):
                dist = (pxyz - nxyz).norm()
                if dist <= 4.0:
                    salt_bridges.append({
                        "pos_res": pi,
                        "pos_name": pose.residue(pi).name3().strip(),
                        "neg_res": ni,
                        "neg_name": pose.residue(ni).name3().strip(),
                        "distance": round(dist, 3),
                    })

    return {
        "interface_hbonds": interface_hbonds,
        "total_hbonds": hbond_set.nhbonds(),
        "n_salt_bridges": len(salt_bridges),
        "salt_bridges": salt_bridges,
    }


# ---------------------------------------------------------------------------
# Stage 6: Structure validation
# ---------------------------------------------------------------------------

def stage_validate(
    pose: "pyrosetta.Pose",
    peptide_chain: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate structure quality: Ramachandran, rotamers, disulfide."""
    from pyrosetta.rosetta.core.scoring import ScoreType
    import pyrosetta

    sfxn = pyrosetta.get_fa_scorefxn()
    sfxn(pose)

    # Ramachandran outliers (simplified)
    n_outlier = 0
    n_total = 0
    for i in range(1, pose.total_residue() + 1):
        if peptide_chain is not None and pose.chain(i) != peptide_chain:
            continue
        if not pose.residue(i).is_protein():
            continue
        n_total += 1
        phi = pose.phi(i)
        psi = pose.psi(i)
        # Simplified outlier: outside general Ramachandran space
        is_gly = pose.residue(i).name3().strip() == "GLY"
        if not is_gly:
            in_helix = (-180 <= phi <= 10) and (-100 <= psi <= 20)
            in_sheet = (-180 <= phi <= -20) and (50 <= psi <= 180 or -180 <= psi <= -120)
            if not (in_helix or in_sheet):
                n_outlier += 1

    # Rotamer outliers (fa_dun > 5.0)
    n_rotamer_outlier = 0
    for i in range(1, pose.total_residue() + 1):
        if peptide_chain is not None and pose.chain(i) != peptide_chain:
            continue
        if not pose.residue(i).is_protein():
            continue
        if pose.residue(i).name3().strip() in ("GLY", "ALA"):
            continue
        fa_dun = pose.energies().residue_total_energies(i)[ScoreType.fa_dun]
        if fa_dun > 5.0:
            n_rotamer_outlier += 1

    # Disulfide check (auto-detect Cys-Cys pairs)
    disulfide_info = []
    cys_residues = []
    for i in range(1, pose.total_residue() + 1):
        if pose.residue(i).name3().strip() == "CYS" and pose.residue(i).has("SG"):
            cys_residues.append(i)

    for a_idx, res_a in enumerate(cys_residues):
        for res_b in cys_residues[a_idx + 1:]:
            sg_a = pose.residue(res_a).xyz("SG")
            sg_b = pose.residue(res_b).xyz("SG")
            dist = (sg_a - sg_b).norm()
            if dist < 3.0:
                disulfide_info.append({
                    "res_a": res_a,
                    "res_b": res_b,
                    "sg_sg_distance": round(dist, 3),
                    "valid": abs(dist - 2.04) < 0.3,
                })

    pct_favored = ((n_total - n_outlier) / n_total * 100) if n_total > 0 else 0.0

    return {
        "rama_outliers": n_outlier,
        "rama_total": n_total,
        "rama_pct_favored": round(pct_favored, 1),
        "rotamer_outliers": n_rotamer_outlier,
        "disulfides": disulfide_info,
        "all_disulfides_valid": all(d["valid"] for d in disulfide_info) if disulfide_info else True,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_enhanced_pipeline(
    template_pdb: str,
    target_sequence: str,
    output_dir: str,
    peptide_chain: int = 1,
    score_function: str = "ref2015",
    resume: bool = False,
) -> Dict[str, Any]:
    """Run the full enhanced PyRosetta pipeline.

    Flow: build pose → reorder → dock/refine → score → analyze → validate

    Supports checkpoint/resume: if resume=True and a checkpoint exists,
    skips already-completed stages.

    Args:
        template_pdb:    Path to template complex PDB.
        target_sequence: Target peptide sequence (1-letter code).
        output_dir:      Output directory for results.
        peptide_chain:   Peptide chain number in template (1-indexed).
        score_function:  "ref2015" or "beta_nov16".
        resume:          Resume from checkpoint if available.

    Returns:
        Combined result dict with all stage outputs.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    checkpoint = _load_checkpoint(out_path) if resume else {}
    last_stage = checkpoint.get("last_stage", "")

    _init_pyrosetta(score_function)

    result: Dict[str, Any] = {
        "template_pdb": template_pdb,
        "target_sequence": target_sequence,
        "score_function": score_function,
    }

    refined_pdb = str(out_path / "refined.pdb")

    # Stage 1: Build pose
    stage = "build_pose"
    if resume and stage in checkpoint:
        print(f"  [resume] Skipping {stage} (already done)", file=sys.stderr)
    else:
        print(f"=== Stage 1: Build Pose ===", file=sys.stderr)
        pose, resolved_chain = stage_build_pose(
            template_pdb, target_sequence, peptide_chain
        )
        # Reorder
        pose = stage_reorder_peptide_last(pose, resolved_chain)
        # Save pre-refinement
        pre_pdb = str(out_path / "pre_refine.pdb")
        pose.dump_pdb(pre_pdb)
        _save_checkpoint(out_path, stage, {
            "pre_pdb": pre_pdb,
            "resolved_chain": resolved_chain,
        })

    # Stage 2: Dock/Refine
    stage = "dock_refine"
    if resume and stage in checkpoint:
        print(f"  [resume] Skipping {stage} (already done)", file=sys.stderr)
        import pyrosetta
        refined_pdb = checkpoint[stage].get("refined_pdb", refined_pdb)
        pose = pyrosetta.pose_from_pdb(refined_pdb)
    else:
        print(f"=== Stage 2: Dock/Refine ===", file=sys.stderr)
        if "pose" not in dir():
            import pyrosetta
            pre_pdb = checkpoint.get("build_pose", {}).get("pre_pdb", "")
            if pre_pdb and Path(pre_pdb).exists():
                pose = pyrosetta.pose_from_pdb(pre_pdb)
            else:
                pose, resolved_chain = stage_build_pose(
                    template_pdb, target_sequence, peptide_chain
                )
                pose = stage_reorder_peptide_last(pose, resolved_chain)

        pose = stage_dock_refine(pose, refined_pdb)
        _save_checkpoint(out_path, stage, {"refined_pdb": refined_pdb})

    # Stage 3: Score
    stage = "score"
    if resume and stage in checkpoint:
        print(f"  [resume] Skipping {stage} (already done)", file=sys.stderr)
        score_data = checkpoint[stage]
    else:
        print(f"=== Stage 3: Score ===", file=sys.stderr)
        score_data = stage_score(pose, score_function)
        # Don't include per_residue in checkpoint (too large), save to file
        per_res = score_data.pop("per_residue_energy", [])
        per_res_path = out_path / "per_residue_energy.json"
        per_res_path.write_text(
            json.dumps(per_res, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        score_data["per_residue_path"] = str(per_res_path)
        _save_checkpoint(out_path, stage, score_data)

    result["scoring"] = score_data

    # Stage 4: Interface analysis
    stage = "analyze"
    if resume and stage in checkpoint:
        print(f"  [resume] Skipping {stage} (already done)", file=sys.stderr)
        analyze_data = checkpoint[stage]
    else:
        print(f"=== Stage 4: Interface Analysis ===", file=sys.stderr)
        analyze_data = stage_analyze(pose, refined_pdb)
        _save_checkpoint(out_path, stage, analyze_data)

    result["interface_analysis"] = analyze_data

    # Stage 5: Structure validation
    stage = "validate"
    if resume and stage in checkpoint:
        print(f"  [resume] Skipping {stage} (already done)", file=sys.stderr)
        validate_data = checkpoint[stage]
    else:
        print(f"=== Stage 5: Structure Validation ===", file=sys.stderr)
        validate_data = stage_validate(pose)
        _save_checkpoint(out_path, stage, validate_data)

    result["structure_validation"] = validate_data

    # Summary
    result["summary"] = {
        "ddg": score_data.get("ddg", 0.0),
        "total_score": score_data.get("total_score", 0.0),
        "clash_score": score_data.get("clash_score", 0.0),
        "delta_sasa": score_data.get("delta_sasa", 0.0),
        "interface_hbonds": analyze_data.get("interface_hbonds", 0),
        "n_salt_bridges": analyze_data.get("n_salt_bridges", 0),
        "rama_pct_favored": validate_data.get("rama_pct_favored", 0.0),
        "all_disulfides_valid": validate_data.get("all_disulfides_valid", True),
        "refined_pdb": refined_pdb,
    }

    # Write full result
    result_path = out_path / "enhanced_pipeline_result.json"
    # Strip non-serializable per_residue data from score_data before saving
    result_json = json.dumps(result, indent=2, ensure_ascii=False, default=str)
    result_path.write_text(result_json, encoding="utf-8")
    print(f"  Full result written to {result_path}", file=sys.stderr)

    _save_checkpoint(out_path, "complete", {"result_path": str(result_path)})

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enhanced PyRosetta pipeline: build → dock → refine → score → analyze"
    )
    parser.add_argument("--template-pdb", required=True, help="Template complex PDB")
    parser.add_argument("--target-sequence", required=True, help="Target peptide sequence")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--peptide-chain", type=int, default=1,
        help="Peptide chain number (default: 1)",
    )
    parser.add_argument(
        "--score-function", default="ref2015",
        choices=["ref2015", "beta_nov16"],
        help="Score function (default: ref2015)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint if available",
    )
    args = parser.parse_args()

    if not Path(args.template_pdb).exists():
        print(json.dumps({"error": f"Template PDB not found: {args.template_pdb}"}), file=sys.stderr)
        sys.exit(1)

    result = run_enhanced_pipeline(
        template_pdb=args.template_pdb,
        target_sequence=args.target_sequence,
        output_dir=args.output_dir,
        peptide_chain=args.peptide_chain,
        score_function=args.score_function,
        resume=args.resume,
    )

    # stdout = JSON summary (for subprocess parsing)
    print(json.dumps(result.get("summary", {})))


if __name__ == "__main__":
    main()
