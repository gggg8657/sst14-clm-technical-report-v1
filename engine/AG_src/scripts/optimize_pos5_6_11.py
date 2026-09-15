#!/usr/bin/env python3
"""
optimize_pos5_6_11.py
=====================
Systematic optimization of SST-14 positions 5, 6, 11 based on analog2 (best binder).

Analog2 baseline: AGCK[F][D]FWKT[I]TSC (N5F, F6D, F11I) → ddG = -14.855 REU

Strategy:
- Position 5: aromatic/hydrophobic residues (contact region)
- Position 6: acidic/polar residues (internal stabilization via K4 interaction)
- Position 11: aliphatic/small hydrophobic (contact region, avoid steric clash)

Constraints preserved:
- FWKT pharmacophore (pos 7-10) — NEVER modified
- Cys3-Cys14 disulfide — NEVER modified
- K4 positive charge — NEVER modified

Uses pharmacological filters to pre-screen before docking.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NATIVE_SEQ = "AGCKNFFWKTFTSC"
TEMPLATE = list(NATIVE_SEQ)  # mutable base

# Frozen positions (0-indexed): 0,1,2,3 (A,G,C,K), 6,7,8,9 (F,W,K,T), 12,13 (S,C)
# Variable positions: 4 (pos5), 5 (pos6), 10 (pos11)

# Position 5 candidates: aromatic + hydrophobic (analog2 had F here)
POS5_CANDIDATES = ['F', 'Y', 'W', 'L', 'I', 'V', 'M']

# Position 6 candidates: acidic + polar (analog2 had D here, analog1 had E)
POS6_CANDIDATES = ['D', 'E', 'N', 'Q', 'S', 'T', 'H']

# Position 11 candidates: aliphatic + small (analog2 had I, avoid F which clashes in analog3)
POS11_CANDIDATES = ['I', 'V', 'L', 'A', 'T', 'M', 'S']

# Pharmacological filter thresholds
FILTERS = {
    "gravy_max": 0.5,
    "instability_max": 50,  # relaxed for screening
    "disulfide_required": True,
    "fwkt_required": True,
    "k9_required": True,
}


def build_sequence(pos5: str, pos6: str, pos11: str) -> str:
    """Build SST-14 variant with substitutions at positions 5, 6, 11."""
    seq = list(NATIVE_SEQ)
    seq[4] = pos5   # position 5 (0-indexed: 4)
    seq[5] = pos6   # position 6 (0-indexed: 5)
    seq[10] = pos11  # position 11 (0-indexed: 10)
    return "".join(seq)


def pre_filter(sequence: str, pharma) -> Tuple[bool, str]:
    """Apply pharmacological filters before docking."""
    props = pharma.calculate_all(sequence)

    # Hard rules
    rules = props.get("structural_rules", {})
    all_rules = rules.get("rules", {})

    if FILTERS["fwkt_required"] and not all_rules.get("fwkt_pharmacophore", {}).get("pass", False):
        return False, "FWKT pharmacophore broken"
    if FILTERS["k9_required"] and not all_rules.get("k9_salt_bridge", {}).get("pass", False):
        return False, "K9 salt bridge broken"
    if FILTERS["disulfide_required"] and not all_rules.get("cys3_cys14_disulfide", {}).get("pass", False):
        return False, "Disulfide broken"

    # Soft filters
    if props.get("gravy", 0) > FILTERS["gravy_max"]:
        return False, f"GRAVY {props['gravy']:.3f} > {FILTERS['gravy_max']}"
    if props.get("instability_index", 0) > FILTERS["instability_max"]:
        return False, f"II {props['instability_index']:.1f} > {FILTERS['instability_max']}"

    return True, "PASS"


def run_docking(template_pdb: str, sequence: str, output_dir: Path, score_function: str = "ref2015") -> Dict:
    """Run PyRosetta FlexPepDock for a single sequence."""
    import pyrosetta
    pyrosetta.init(
        "-mute all "
        "-corrections::beta_nov16 true "
        "-in:file:fullatom "
        "-ignore_unrecognized_res true "
        "-detect_disulf true"
    )
    from pyrosetta import pose_from_pdb, Pose
    from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
    from pyrosetta.rosetta.protocols.flexpep_docking import FlexPepDockingProtocol
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
    from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking, IncludeCurrent
    from pyrosetta.rosetta.protocols.rigid import RigidBodyTransMover
    from pyrosetta.rosetta.core.scoring.hbonds import HBondSet

    aa_map = {
        'A':'ALA','C':'CYS','D':'ASP','E':'GLU','F':'PHE','G':'GLY','H':'HIS',
        'I':'ILE','K':'LYS','L':'LEU','M':'MET','N':'ASN','P':'PRO','Q':'GLN',
        'R':'ARG','S':'SER','T':'THR','V':'VAL','W':'TRP','Y':'TYR',
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    if result_path.exists():
        try:
            cached = json.loads(result_path.read_text())
            if cached.get("complete"):
                return cached
        except json.JSONDecodeError:
            pass

    t0 = time.time()
    pose = pose_from_pdb(template_pdb)
    original = pose.chain_sequence(1)

    # Mutate
    mutations = []
    for i, (o, t) in enumerate(zip(original, sequence)):
        if o != t:
            resid = pose.chain_begin(1) + i
            MutateResidue(resid, aa_map[t]).apply(pose)
            mutations.append(f"{o}{i+1}{t}")

    # Pack
    sfxn = ScoreFunctionFactory.create_score_function(score_function)
    tf = TaskFactory()
    tf.push_back(RestrictToRepacking())
    tf.push_back(IncludeCurrent())
    packer = PackRotamersMover(sfxn)
    packer.task_factory(tf)
    packer.apply(pose)

    # FlexPepDock
    FlexPepDockingProtocol().apply(pose)
    pose.dump_pdb(str(output_dir / "refined.pdb"))

    # ddG
    bound_pose = Pose(pose)
    bound_score = sfxn(bound_pose)

    unbound_pose = Pose(pose)
    trans = RigidBodyTransMover(unbound_pose, 1)
    trans.step_size(500.0)
    trans.apply(unbound_pose)
    packer2 = PackRotamersMover(sfxn)
    packer2.task_factory(tf)
    packer2.apply(unbound_pose)
    unbound_score = sfxn(unbound_pose)
    ddg = bound_score - unbound_score

    # H-bonds
    hbset = HBondSet()
    pose.update_residue_neighbors()
    hbset.setup_for_residue_pair_energies(pose, False, False)
    chain1_end = pose.chain_end(1)
    interface_hbonds = sum(
        1 for idx in range(1, hbset.nhbonds()+1)
        if (hbset.hbond(idx).don_res() <= chain1_end) != (hbset.hbond(idx).acc_res() <= chain1_end)
    )

    # Salt bridges
    n_salt = 0
    for i in range(pose.chain_begin(1), pose.chain_end(1)+1):
        ri = pose.residue(i)
        if ri.name1() not in ('K','R'): continue
        for a in range(1, ri.natoms()+1):
            an = ri.atom_name(a).strip()
            if an not in ('NZ','NH1','NH2'): continue
            for j in range(pose.chain_begin(2), pose.chain_end(2)+1):
                rj = pose.residue(j)
                if rj.name1() not in ('D','E'): continue
                for b in range(1, rj.natoms()+1):
                    bn = rj.atom_name(b).strip()
                    if bn not in ('OD1','OD2','OE1','OE2'): continue
                    if ri.xyz(a).distance(rj.xyz(b)) < 4.0:
                        n_salt += 1

    elapsed = time.time() - t0
    result = {
        "sequence": sequence, "mutations": mutations,
        "ddg": round(ddg, 3), "bound_score": round(bound_score, 3),
        "unbound_score": round(unbound_score, 3),
        "interface_hbonds": interface_hbonds, "n_salt_bridges": n_salt,
        "elapsed_sec": round(elapsed, 1), "complete": True,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="Position 5,6,11 combinatorial optimization")
    parser.add_argument("--template-pdb", required=True)
    parser.add_argument("--output-dir", default="runs/pos5_6_11_optimization")
    parser.add_argument("--top-n", type=int, default=20, help="Top N candidates to dock after filtering")
    parser.add_argument("--dock-all", action="store_true", help="Dock all filtered candidates (no top-N limit)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Generate all combinations and pre-filter
    print("=" * 70, file=sys.stderr)
    print("  PHASE 1: Combinatorial Generation + Pharmacological Pre-Filter", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    # Import pharma module
    script_dir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(script_dir.parent))
    from AG_src.pipeline.pharma_properties import PharmaProperties
    pharma = PharmaProperties(reference_seq=NATIVE_SEQ)

    all_combos = list(itertools.product(POS5_CANDIDATES, POS6_CANDIDATES, POS11_CANDIDATES))
    total = len(all_combos)
    print(f"  Total combinations: {len(POS5_CANDIDATES)}×{len(POS6_CANDIDATES)}×{len(POS11_CANDIDATES)} = {total}", file=sys.stderr)

    passed = []
    failed = []
    pharma_results = {}

    for i, (p5, p6, p11) in enumerate(all_combos):
        seq = build_sequence(p5, p6, p11)
        ok, reason = pre_filter(seq, pharma)
        props = pharma.calculate_all(seq)
        pharma_results[seq] = props

        if ok:
            passed.append((seq, p5, p6, p11, props))
        else:
            failed.append((seq, reason))

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{total}] Screened... {len(passed)} passed", file=sys.stderr)

    print(f"\n  Pre-filter results: {len(passed)} PASSED / {len(failed)} FAILED out of {total}", file=sys.stderr)

    # Save pharma screening
    screening_path = out_dir / "pharma_screening.json"
    screening_data = {
        "total_combinations": total,
        "passed": len(passed),
        "failed": len(failed),
        "pos5_candidates": POS5_CANDIDATES,
        "pos6_candidates": POS6_CANDIDATES,
        "pos11_candidates": POS11_CANDIDATES,
        "filters": FILTERS,
        "passed_sequences": [{"sequence": s, "pos5": p5, "pos6": p6, "pos11": p11} for s, p5, p6, p11, _ in passed],
        "failed_sequences": [{"sequence": s, "reason": r} for s, r in failed[:20]],
    }
    screening_path.write_text(json.dumps(screening_data, indent=2), encoding="utf-8")

    # Phase 2: Rank by pharma properties (Boman + low GRAVY) and pick top-N
    # Sort by: Boman (desc) as proxy for receptor-binding potential
    passed.sort(key=lambda x: -x[4].get("boman_index", 0))

    if not args.dock_all:
        to_dock = passed[:args.top_n]
    else:
        to_dock = passed

    print(f"\n  Selected {len(to_dock)} candidates for docking", file=sys.stderr)

    # Phase 3: PyRosetta docking
    print("\n" + "=" * 70, file=sys.stderr)
    print("  PHASE 2: PyRosetta FlexPepDock", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    dock_results = {}
    for idx, (seq, p5, p6, p11, props) in enumerate(to_dock, 1):
        label = f"p5{p5}_p6{p6}_p11{p11}"
        print(f"\n  [{idx}/{len(to_dock)}] {label}: {seq}", file=sys.stderr)

        dock_dir = out_dir / f"{label}_{seq}"
        try:
            result = run_docking(args.template_pdb, seq, dock_dir)
            result["pos5"] = p5
            result["pos6"] = p6
            result["pos11"] = p11
            result["label"] = label
            result["pharma"] = {
                "gravy": props.get("gravy"),
                "boman_index": props.get("boman_index"),
                "instability_index": props.get("instability_index"),
                "net_charge_ph74": props.get("net_charge_ph74"),
            }
            dock_results[seq] = result
            print(f"  ✓ ddG = {result['ddg']:.3f} REU, H-bonds = {result['interface_hbonds']}, "
                  f"Salt bridges = {result['n_salt_bridges']}", file=sys.stderr)
        except Exception as e:
            print(f"  ✗ FAILED: {e}", file=sys.stderr)
            dock_results[seq] = {"sequence": seq, "error": str(e), "label": label}

    # Phase 4: Rank and report
    print("\n" + "=" * 70, file=sys.stderr)
    print("  PHASE 3: Ranking & Report", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    # Reference values
    analog2_ddg = -14.855
    native_ddg = -6.173

    successful = {s: r for s, r in dock_results.items() if r.get("complete")}
    ranked = sorted(successful.items(), key=lambda x: x[1]["ddg"])

    # Report
    lines = ["# Position 5,6,11 Combinatorial Optimization Report\n"]
    lines.append(f"**Date**: 2026-02-27  ")
    lines.append(f"**Baseline**: analog2 (AGCKFDFWKTITSC, ddG = {analog2_ddg:.3f})  ")
    lines.append(f"**Native**: SST-14 (AGCKNFFWKTFTSC, ddG = {native_ddg:.3f})  ")
    lines.append(f"**Combinations**: {total} total → {len(passed)} filtered → {len(to_dock)} docked\n")

    lines.append("## Top Results\n")
    lines.append("| Rank | Sequence | Pos5 | Pos6 | Pos11 | ddG | vs analog2 | vs native | H-bonds | Salt Br | GRAVY | Boman |")
    lines.append("|------|----------|------|------|-------|-----|-----------|-----------|---------|---------|-------|-------|")

    for rank, (seq, r) in enumerate(ranked[:30], 1):
        ddg = r["ddg"]
        vs_a2 = ddg - analog2_ddg
        vs_nat = ddg - native_ddg
        ph = r.get("pharma", {})
        lines.append(
            f"| {rank} | `{seq}` | {r['pos5']} | {r['pos6']} | {r['pos11']} | "
            f"{ddg:.3f} | {vs_a2:+.3f} | {vs_nat:+.3f} | {r['interface_hbonds']} | "
            f"{r['n_salt_bridges']} | {ph.get('gravy',0):.3f} | {ph.get('boman_index',0):.3f} |"
        )

    lines.append(f"\n## Summary\n")
    lines.append(f"- **Best candidate**: `{ranked[0][0]}` (ddG = {ranked[0][1]['ddg']:.3f})" if ranked else "- No results")
    if len(ranked) >= 2:
        lines.append(f"- **Second best**: `{ranked[1][0]}` (ddG = {ranked[1][1]['ddg']:.3f})")
    lines.append(f"- **Candidates beating analog2**: {sum(1 for _,r in ranked if r['ddg'] < analog2_ddg)}")
    lines.append(f"- **Candidates beating native**: {sum(1 for _,r in ranked if r['ddg'] < native_ddg)}")

    report_path = out_dir / "optimization_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # Save full results JSON
    full_results = {
        "metadata": {
            "total_combinations": total,
            "passed_filter": len(passed),
            "docked": len(to_dock),
            "successful": len(successful),
            "analog2_ddg": analog2_ddg,
            "native_ddg": native_ddg,
        },
        "results": [
            {**r, "rank": i+1}
            for i, (_, r) in enumerate(ranked)
        ]
    }
    (out_dir / "optimization_results.json").write_text(json.dumps(full_results, indent=2), encoding="utf-8")

    # stdout: top 10 summary
    top10 = [
        {"rank": i+1, "sequence": seq, "ddg": r["ddg"],
         "vs_analog2": round(r["ddg"] - analog2_ddg, 3),
         "pos5": r["pos5"], "pos6": r["pos6"], "pos11": r["pos11"]}
        for i, (seq, r) in enumerate(ranked[:10])
    ]
    print(json.dumps({"top10": top10, "total_docked": len(successful)}, indent=2))


if __name__ == "__main__":
    main()
