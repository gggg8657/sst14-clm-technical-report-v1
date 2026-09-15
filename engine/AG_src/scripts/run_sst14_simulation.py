#!/usr/bin/env python3
"""
run_sst14_simulation.py
=======================
Run PyRosetta FlexPepDock simulations for SST-14 native + 5 analogs against SSTR2.
Then compute pharmacological properties for each sequence.

Usage:
    conda run -n bio-tools python AG_src/scripts/run_sst14_simulation.py \
        --template-pdb data/fold_test1/fold_test1_model_0.pdb \
        --output-dir runs/sst14_analogs_sim

Output: runs/sst14_analogs_sim/
    ├── native_AGCKNFFWKTFTSC/
    │   ├── enhanced_pipeline_result.json
    │   ├── refined.pdb
    │   └── pipeline_checkpoint.json
    ├── analog1_AGCKYEFWKTVTSC/
    │   └── ...
    ├── ...
    ├── pharma_properties.json      (all 13 methods × 6 sequences)
    ├── comparison_report.json      (ΔΔG, rankings)
    └── summary_report.md           (human-readable)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------

SEQUENCES = {
    "native":  "AGCKNFFWKTFTSC",
    "analog1": "AGCKYEFWKTVTSC",  # N5Y, F6E, F11V
    "analog2": "AGCKFDFWKTITSC",  # N5F, F6D, F11I
    "analog3": "AGCFIFFWKTFTSC",  # K4F, N5I
    "analog4": "AGCKHFFWHTFTSC",  # N5H, K9H (pH-selective)
    "analog5": "YGCKNFFWKTFTST",  # A1Y, C14T (linear)
}

MUTATIONS_DESC = {
    "native":  "Native SST-14 (reference)",
    "analog1": "N5Y, F6E, F11V — acidic substitution, π-π stacking lost",
    "analog2": "N5F, F6D, F11I — similar to #1, Asp instead of Glu",
    "analog3": "K4F, N5I — K4 positive charge lost, highly hydrophobic",
    "analog4": "N5H, K9H — pH-selective (His pKa ~6.0), K9-D122 salt bridge broken at pH 7.4",
    "analog5": "A1Y, C14T — disulfide bond destroyed, linear form",
}


# ---------------------------------------------------------------------------
# PyRosetta simulation for a single sequence
# ---------------------------------------------------------------------------

def run_single_simulation(
    template_pdb: str,
    sequence: str,
    output_dir: Path,
    score_function: str = "ref2015",
    resume: bool = False,
) -> Dict[str, Any]:
    """Run enhanced pipeline for one peptide sequence."""
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
    from pyrosetta.rosetta.core.pose import setPoseExtraScore
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "simulation_result.json"

    # Check resume
    if resume and result_path.exists():
        try:
            cached = json.loads(result_path.read_text())
            if cached.get("complete"):
                print(f"  [resume] {sequence} already complete, skipping", file=sys.stderr)
                return cached
        except json.JSONDecodeError:
            pass

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  Simulating: {sequence}", file=sys.stderr)
    print(f"  Output: {output_dir}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    t0 = time.time()

    # --- Stage 1: Load template ---
    print("  [1/6] Loading template...", file=sys.stderr)
    pose = pose_from_pdb(template_pdb)
    original_seq = pose.chain_sequence(1)
    print(f"         Template peptide: {original_seq} ({len(original_seq)} aa)", file=sys.stderr)
    print(f"         Target peptide:   {sequence} ({len(sequence)} aa)", file=sys.stderr)

    # --- Stage 2: Mutate peptide chain ---
    print("  [2/6] Mutating peptide residues...", file=sys.stderr)
    aa_map = {
        'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
        'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
        'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
        'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
    }
    mutations_applied = []
    chain1_start = pose.chain_begin(1)
    for i, (orig, target) in enumerate(zip(original_seq, sequence)):
        if orig != target:
            resid = chain1_start + i
            mut = MutateResidue(resid, aa_map[target])
            mut.apply(pose)
            mutations_applied.append(f"{orig}{i+1}{target}")
            print(f"         Mutated: {orig}{i+1} → {target}", file=sys.stderr)

    if not mutations_applied:
        print("         No mutations needed (native sequence)", file=sys.stderr)

    # Save pre-refinement pose
    pre_refine_pdb = str(output_dir / "pre_refinement.pdb")
    pose.dump_pdb(pre_refine_pdb)

    # --- Stage 3: Pre-packing (side-chain optimization) ---
    print("  [3/6] Pre-packing side chains...", file=sys.stderr)
    sfxn = ScoreFunctionFactory.create_score_function(score_function)

    from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.core.pack.task.operation import (
        RestrictToRepacking,
        IncludeCurrent,
    )
    tf = TaskFactory()
    tf.push_back(RestrictToRepacking())
    tf.push_back(IncludeCurrent())
    packer = PackRotamersMover(sfxn)
    packer.task_factory(tf)
    packer.apply(pose)

    pre_dock_score = sfxn(pose)
    print(f"         Pre-dock score: {pre_dock_score:.2f} REU", file=sys.stderr)

    # --- Stage 4: FlexPepDock refinement ---
    print("  [4/6] FlexPepDock refinement (this takes time)...", file=sys.stderr)
    t_dock = time.time()

    flexpep = FlexPepDockingProtocol()
    flexpep.apply(pose)

    post_dock_score = sfxn(pose)
    dock_elapsed = time.time() - t_dock
    print(f"         Post-dock score: {post_dock_score:.2f} REU ({dock_elapsed:.1f}s)", file=sys.stderr)

    # Save refined pose
    refined_pdb = str(output_dir / "refined.pdb")
    pose.dump_pdb(refined_pdb)

    # --- Stage 5: Scoring & Energy analysis ---
    print("  [5/6] Computing binding energy (ddG)...", file=sys.stderr)

    # Interface ddG via separation
    bound_pose = Pose(pose)
    bound_score = sfxn(bound_pose)

    # Separate chains
    from pyrosetta.rosetta.protocols.rigid import RigidBodyTransMover
    unbound_pose = Pose(pose)
    jump_id = 1
    trans = RigidBodyTransMover(unbound_pose, jump_id)
    trans.step_size(500.0)
    trans.apply(unbound_pose)

    # Repack after separation
    packer2 = PackRotamersMover(sfxn)
    packer2.task_factory(tf)
    packer2.apply(unbound_pose)
    unbound_score = sfxn(unbound_pose)

    ddg = bound_score - unbound_score
    print(f"         Bound: {bound_score:.2f}, Unbound: {unbound_score:.2f}", file=sys.stderr)
    print(f"         ddG: {ddg:.2f} REU", file=sys.stderr)

    # Per-residue energy for peptide chain
    per_residue = {}
    emap = pose.energies()
    for i in range(pose.chain_begin(1), pose.chain_end(1) + 1):
        resname = pose.residue(i).name3()
        total_e = emap.residue_total_energy(i)
        per_residue[f"{resname}{i}"] = round(total_e, 3)

    # Clash score (fa_rep)
    from pyrosetta.rosetta.core.scoring import ScoreType
    clash_score = pose.energies().total_energies()[ScoreType.fa_rep]

    # --- Stage 6: Interface analysis ---
    print("  [6/6] Interface analysis...", file=sys.stderr)

    # H-bonds
    from pyrosetta.rosetta.core.scoring.hbonds import HBondSet
    hbset = HBondSet()
    pose.update_residue_neighbors()
    hbset.setup_for_residue_pair_energies(pose, False, False)

    chain1_end = pose.chain_end(1)
    interface_hbonds = 0
    hbond_details = []
    for hb_idx in range(1, hbset.nhbonds() + 1):
        hb = hbset.hbond(hb_idx)
        don_res = hb.don_res()
        acc_res = hb.acc_res()
        don_in_pep = don_res <= chain1_end
        acc_in_pep = acc_res <= chain1_end
        if don_in_pep != acc_in_pep:  # cross-chain
            interface_hbonds += 1
            hbond_details.append({
                "donor": f"{pose.residue(don_res).name3()}{don_res}",
                "acceptor": f"{pose.residue(acc_res).name3()}{acc_res}",
                "energy": round(hb.energy(), 3),
            })

    # Salt bridges (K/R NZ/NH ↔ D/E OD/OE, <4.0 Å)
    salt_bridges = []
    for i in range(pose.chain_begin(1), pose.chain_end(1) + 1):
        res_i = pose.residue(i)
        if res_i.name1() not in ('K', 'R'):
            continue
        pos_atoms = []
        for a in range(1, res_i.natoms() + 1):
            aname = res_i.atom_name(a).strip()
            if aname in ('NZ', 'NH1', 'NH2'):
                pos_atoms.append((a, aname))

        for j in range(pose.chain_begin(2), pose.chain_end(2) + 1):
            res_j = pose.residue(j)
            if res_j.name1() not in ('D', 'E'):
                continue
            neg_atoms = []
            for a in range(1, res_j.natoms() + 1):
                aname = res_j.atom_name(a).strip()
                if aname in ('OD1', 'OD2', 'OE1', 'OE2'):
                    neg_atoms.append((a, aname))

            for pa, pa_name in pos_atoms:
                for na, na_name in neg_atoms:
                    dist = res_i.xyz(pa).distance(res_j.xyz(na))
                    if dist < 4.0:
                        salt_bridges.append({
                            "positive": f"{res_i.name3()}{i}.{pa_name}",
                            "negative": f"{res_j.name3()}{j}.{na_name}",
                            "distance": round(dist, 2),
                        })

    elapsed = time.time() - t0
    print(f"\n  Total elapsed: {elapsed:.1f}s", file=sys.stderr)

    result = {
        "sequence": sequence,
        "mutations": mutations_applied,
        "pre_dock_score": round(pre_dock_score, 3),
        "post_dock_score": round(post_dock_score, 3),
        "ddg": round(ddg, 3),
        "bound_score": round(bound_score, 3),
        "unbound_score": round(unbound_score, 3),
        "clash_score": round(clash_score, 3),
        "per_residue_energy": per_residue,
        "interface_hbonds": interface_hbonds,
        "hbond_details": hbond_details[:20],
        "salt_bridges": salt_bridges,
        "n_salt_bridges": len(salt_bridges),
        "refined_pdb": refined_pdb,
        "score_function": score_function,
        "elapsed_sec": round(elapsed, 1),
        "complete": True,
    }

    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


# ---------------------------------------------------------------------------
# Pharmacological properties (pure Python, no PyRosetta needed)
# ---------------------------------------------------------------------------

def run_pharma_analysis(sequences: Dict[str, str]) -> Dict[str, Any]:
    """Run pharmacological property analysis for all sequences."""
    # Add parent to path for import
    script_dir = Path(__file__).resolve().parent
    ag_src = script_dir.parent
    sys.path.insert(0, str(ag_src.parent))

    from AG_src.pipeline.pharma_properties import PharmaProperties

    pp = PharmaProperties(reference_seq="AGCKNFFWKTFTSC")
    results = {}
    for name, seq in sequences.items():
        print(f"  Pharma analysis: {name} ({seq})", file=sys.stderr)
        results[name] = pp.calculate_all(seq)
    return results


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

def generate_comparison(
    sim_results: Dict[str, Dict],
    pharma_results: Dict[str, Dict],
    output_dir: Path,
) -> None:
    """Generate comparison report across all analogs."""
    native_ddg = sim_results.get("native", {}).get("ddg", 0.0)

    comparison = []
    for name, seq in SEQUENCES.items():
        sim = sim_results.get(name, {})
        pharma = pharma_results.get(name, {})
        rules = pharma.get("structural_rules", {})

        entry = {
            "name": name,
            "sequence": seq,
            "description": MUTATIONS_DESC[name],
            "ddg": sim.get("ddg", "N/A"),
            "delta_ddg": round(sim.get("ddg", 0.0) - native_ddg, 3) if isinstance(sim.get("ddg"), (int, float)) else "N/A",
            "interface_hbonds": sim.get("interface_hbonds", "N/A"),
            "n_salt_bridges": sim.get("n_salt_bridges", "N/A"),
            "clash_score": sim.get("clash_score", "N/A"),
            "elapsed_sec": sim.get("elapsed_sec", "N/A"),
            "gravy": pharma.get("gravy", "N/A"),
            "boman_index": pharma.get("boman_index", "N/A"),
            "instability_index": pharma.get("instability_index", "N/A"),
            "pi": pharma.get("isoelectric_point", "N/A"),
            "charge_ph74": pharma.get("net_charge_ph7.4", "N/A"),
            "charge_ph65": pharma.get("net_charge_ph6.5", "N/A"),
            "fwkt_conserved": rules.get("fwkt_pharmacophore", "N/A"),
            "disulfide_intact": rules.get("cys3_cys14_disulfide", "N/A"),
            "k9_salt_bridge": rules.get("k9_d122_salt_bridge", "N/A"),
        }
        comparison.append(entry)

    # Save JSON
    comp_path = output_dir / "comparison_report.json"
    comp_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")

    # Generate markdown summary
    md_lines = [
        "# SST-14 Analog Simulation Report",
        f"\nTemplate: AlphaFold3 SSTR2-SST14 complex",
        f"Score function: ref2015",
        f"Reference ddG (native): {native_ddg:.3f} REU\n",
        "## Results Summary\n",
        "| Analog | Sequence | ddG (REU) | ΔΔG | H-bonds | Salt Bridges | FWKT | SS Bond | K9 Bridge |",
        "|--------|----------|-----------|-----|---------|--------------|------|---------|-----------|",
    ]
    for e in comparison:
        ddg_str = f"{e['ddg']:.3f}" if isinstance(e['ddg'], (int, float)) else str(e['ddg'])
        delta_str = f"{e['delta_ddg']:+.3f}" if isinstance(e['delta_ddg'], (int, float)) else str(e['delta_ddg'])
        md_lines.append(
            f"| {e['name']} | `{e['sequence']}` | {ddg_str} | {delta_str} | "
            f"{e['interface_hbonds']} | {e['n_salt_bridges']} | "
            f"{e['fwkt_conserved']} | {e['disulfide_intact']} | {e['k9_salt_bridge']} |"
        )

    md_lines.extend([
        "\n## Pharmacological Properties\n",
        "| Analog | GRAVY | Boman | II | pI | Charge(7.4) | Charge(6.5) |",
        "|--------|-------|-------|-----|-----|-------------|-------------|",
    ])
    for e in comparison:
        def fmt(v):
            return f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
        md_lines.append(
            f"| {e['name']} | {fmt(e['gravy'])} | {fmt(e['boman_index'])} | "
            f"{fmt(e['instability_index'])} | {fmt(e['pi'])} | "
            f"{fmt(e['charge_ph74'])} | {fmt(e['charge_ph65'])} |"
        )

    md_lines.extend([
        "\n## Mutation Details\n",
    ])
    for name, desc in MUTATIONS_DESC.items():
        md_lines.append(f"- **{name}**: {desc}")

    md_path = output_dir / "summary_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\n  Reports written to {output_dir}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SST-14 analog simulation batch runner")
    parser.add_argument("--template-pdb", required=True, help="Template SSTR2-SST14 complex PDB")
    parser.add_argument("--output-dir", default="runs/sst14_analogs_sim", help="Output directory")
    parser.add_argument("--score-function", default="ref2015", choices=["ref2015", "beta_nov16"])
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--sequences", nargs="*", default=list(SEQUENCES.keys()),
        help=f"Which sequences to run (default: all). Options: {list(SEQUENCES.keys())}",
    )
    args = parser.parse_args()

    template = Path(args.template_pdb)
    if not template.exists():
        print(f"ERROR: Template PDB not found: {template}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Run pharmacological analysis (no PyRosetta needed, fast) ---
    print("\n" + "=" * 60, file=sys.stderr)
    print("  PHASE 1: Pharmacological Property Analysis", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    seqs_to_run = {k: SEQUENCES[k] for k in args.sequences if k in SEQUENCES}
    pharma_results = run_pharma_analysis(seqs_to_run)
    pharma_path = output_dir / "pharma_properties.json"
    pharma_path.write_text(json.dumps(pharma_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Pharma results saved to {pharma_path}", file=sys.stderr)

    # --- Run PyRosetta simulations ---
    print("\n" + "=" * 60, file=sys.stderr)
    print("  PHASE 2: PyRosetta FlexPepDock Simulations", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    sim_results = {}
    total = len(seqs_to_run)
    for idx, (name, seq) in enumerate(seqs_to_run.items(), 1):
        print(f"\n  [{idx}/{total}] {name}: {seq}", file=sys.stderr)
        sim_dir = output_dir / f"{name}_{seq}"
        try:
            result = run_single_simulation(
                template_pdb=str(template),
                sequence=seq,
                output_dir=sim_dir,
                score_function=args.score_function,
                resume=args.resume,
            )
            sim_results[name] = result
            print(f"  ✓ {name}: ddG = {result['ddg']:.3f} REU, "
                  f"H-bonds = {result['interface_hbonds']}, "
                  f"Salt bridges = {result['n_salt_bridges']}", file=sys.stderr)
        except Exception as e:
            print(f"  ✗ {name}: FAILED — {e}", file=sys.stderr)
            sim_results[name] = {"sequence": seq, "error": str(e)}

    # --- Generate comparison report ---
    print("\n" + "=" * 60, file=sys.stderr)
    print("  PHASE 3: Comparison Report", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    generate_comparison(sim_results, pharma_results, output_dir)

    # stdout: summary JSON
    summary = {
        "total_sequences": total,
        "completed": sum(1 for r in sim_results.values() if r.get("complete")),
        "failed": sum(1 for r in sim_results.values() if "error" in r),
        "native_ddg": sim_results.get("native", {}).get("ddg", "N/A"),
        "results": {
            name: {
                "ddg": r.get("ddg", "N/A"),
                "delta_ddg": round(r.get("ddg", 0.0) - sim_results.get("native", {}).get("ddg", 0.0), 3)
                if isinstance(r.get("ddg"), (int, float)) and isinstance(sim_results.get("native", {}).get("ddg"), (int, float))
                else "N/A",
                "interface_hbonds": r.get("interface_hbonds", "N/A"),
                "n_salt_bridges": r.get("n_salt_bridges", "N/A"),
            }
            for name, r in sim_results.items()
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
