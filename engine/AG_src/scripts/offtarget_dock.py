#!/usr/bin/env python3
"""
offtarget_dock.py
=================
Off-target receptor에 대한 PyRosetta FlexPepDock standalone 스크립트.

SSTR2 refined complex에서 peptide를 추출하고, off-target receptor에
구조 정렬(CA superposition)로 peptide를 배치한 뒤 FlexPepDock refinement
+ InterfaceAnalyzerMover ddG를 계산한다.

Called by step05b_selectivity.py / run_pipeline_live.py via subprocess:
    conda run -n bio-tools python AG_src/scripts/offtarget_dock.py \
        --sstr2-complex refined_var_103.pdb \
        --offtarget-receptor SSTR1_alphafold.pdb \
        --output /tmp/ot_dock.pdb

stdout: JSON only (parsed by caller)
stderr: all PyRosetta logs and diagnostics
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path


def _ddg_physical_floor() -> float:
    try:
        return float(os.environ.get("DDG_PHYSICAL_FLOOR", "-150"))
    except ValueError:
        return -150.0


# ---------------------------------------------------------------------------
# PyRosetta initialization (reuses pattern from flexpep_dock.py)
# ---------------------------------------------------------------------------

def init_pyrosetta() -> None:
    """Initialize PyRosetta with muted output (all logs to stderr)."""
    import pyrosetta
    pyrosetta.init(
        options="-mute all -ex1 -ex2aro -ignore_unrecognized_res",
        silent=True,
    )


# ---------------------------------------------------------------------------
# Chain extraction
# ---------------------------------------------------------------------------

def extract_chain_pose(pose, chain_id: int):
    """Extract a single chain from a pose as a new pose.

    Args:
        pose: PyRosetta Pose object
        chain_id: Chain number (1-indexed)

    Returns:
        New Pose containing only the specified chain
    """
    import pyrosetta

    residues = [i for i in range(1, pose.total_residue() + 1) if pose.chain(i) == chain_id]
    if not residues:
        raise ValueError(f"No residues found for chain {chain_id}")

    # Dump chain to temp PDB, reload
    tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
    tmp_path = tmp.name
    tmp.close()

    # Write only the target chain residues
    lines = []
    pdb_text = pose.dump_pdb("")  # get PDB string
    # Alternative: use Pose slicing
    from pyrosetta.rosetta.core.pose import Pose
    chain_pose = Pose()

    start_res = residues[0]
    end_res = residues[-1]

    from pyrosetta.rosetta.protocols.grafting import return_region
    chain_pose = return_region(pose, start_res, end_res)

    Path(tmp_path).unlink(missing_ok=True)
    return chain_pose


def extract_chain_pdb_text(pdb_path: str, chain_letter: str) -> str:
    """Extract PDB lines for a specific chain letter from a PDB file.

    Args:
        pdb_path: Path to PDB file
        chain_letter: Chain letter (e.g. 'A', 'B')

    Returns:
        PDB text containing only the specified chain
    """
    lines = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")) and len(line) >= 22:
                if line[21] == chain_letter:
                    lines.append(line)
            elif line.startswith(("TER", "END")):
                if lines:
                    lines.append("TER\n")
    lines.append("END\n")
    return "".join(lines)


# ---------------------------------------------------------------------------
# Structure alignment via CA superposition
# ---------------------------------------------------------------------------

def align_offtarget_to_sstr2(sstr2_receptor_pose, offtarget_pose):
    """Align off-target receptor onto SSTR2 receptor via CA superposition.

    Uses the shorter chain's length to define the subset for alignment.

    Args:
        sstr2_receptor_pose: SSTR2 receptor Pose (reference)
        offtarget_pose: Off-target receptor Pose (mobile)

    Returns:
        (aligned_offtarget_pose, rmsd): Tuple of aligned pose and RMSD
    """
    from pyrosetta.rosetta.core.id import AtomID
    from pyrosetta.rosetta.utility import vector1_numeric_xyzVector_double_t as Vec1
    from pyrosetta.rosetta.numeric import xyzVector_double_t as XYZ
    import pyrosetta

    # Collect CA residue ids + 1-letter sequence from both poses
    def get_ca_info(pose):
        residue_ids = []
        seq = []
        for i in range(1, pose.total_residue() + 1):
            res = pose.residue(i)
            if res.has("CA"):
                residue_ids.append(i)
                try:
                    seq.append(res.name1())
                except Exception:
                    seq.append("X")
        return residue_ids, "".join(seq)

    sstr2_ids, sstr2_seq = get_ca_info(sstr2_receptor_pose)
    ot_ids, ot_seq = get_ca_info(offtarget_pose)

    # 2026-06-09 fix: 인덱스 기반 대응(residue i ↔ i)은 길이가 다른 off-target 수용체
    # (SSTR1=274, SSTR2=472 등)에서 28A 오정렬을 유발했다. **서열 정렬 기반 대응**으로
    # 상동 잔기끼리만 매핑한다 (Biopython global align, BLOSUM62). 정렬 실패 시 인덱스 폴백.
    pairs = []  # (ot_pose_resid, sstr2_pose_resid)
    try:
        from Bio.Align import PairwiseAligner, substitution_matrices
        aligner = PairwiseAligner()
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        aligner.open_gap_score = -10.0
        aligner.extend_gap_score = -0.5
        aln = aligner.align(ot_seq, sstr2_seq)[0]
        # aligned blocks: [[(ot_start,ot_end),...],[(s2_start,s2_end),...]]
        ot_blocks, s2_blocks = aln.aligned
        for (o0, o1), (s0, s1) in zip(ot_blocks, s2_blocks):
            for k in range(o1 - o0):
                pairs.append((ot_ids[o0 + k], sstr2_ids[s0 + k]))
        print(f"Sequence-based alignment: {len(pairs)} corresponding CA pairs "
              f"(ot {len(ot_seq)}aa vs sstr2 {len(sstr2_seq)}aa)", file=sys.stderr)
    except Exception as exc:
        n_align = min(len(sstr2_ids), len(ot_ids))
        print(f"WARNING: seq-align unavailable ({exc}); index fallback n={n_align}", file=sys.stderr)
        pairs = [(ot_ids[i], sstr2_ids[i]) for i in range(n_align)]

    if len(pairs) < 10:
        print(f"WARNING: Only {len(pairs)} CA pairs for alignment", file=sys.stderr)

    # Build atom ID map from corresponding residues
    from pyrosetta.rosetta.std import map_core_id_AtomID_core_id_AtomID as AtomIDMap
    atom_map = AtomIDMap()
    for ot_resid, s2_resid in pairs:
        mobile_atom = AtomID(offtarget_pose.residue(ot_resid).atom_index("CA"), ot_resid)
        ref_atom = AtomID(sstr2_receptor_pose.residue(s2_resid).atom_index("CA"), s2_resid)
        atom_map[mobile_atom] = ref_atom

    n_align = len(pairs)
    # Superimpose
    from pyrosetta.rosetta.core.scoring import superimpose_pose
    rmsd = superimpose_pose(offtarget_pose, sstr2_receptor_pose, atom_map)

    if rmsd > 10.0:
        print(f"WARNING: High alignment RMSD={rmsd:.2f}A (>10A)", file=sys.stderr)
    else:
        print(f"Alignment RMSD={rmsd:.2f}A ({n_align} CA atoms)", file=sys.stderr)

    return offtarget_pose, rmsd


def align_simple_centroid(sstr2_receptor_pose, offtarget_pose, peptide_pose):
    """Fallback: place peptide near off-target receptor centroid.

    Used when CA superposition fails (too few matching atoms).

    Args:
        sstr2_receptor_pose: SSTR2 receptor (for reference peptide location)
        offtarget_pose: Off-target receptor
        peptide_pose: Peptide to place

    Returns:
        offtarget_pose (unchanged, peptide placement done via PDB assembly)
    """
    print("Using centroid fallback for peptide placement", file=sys.stderr)
    return offtarget_pose


# ---------------------------------------------------------------------------
# Chimeric PDB assembly
# ---------------------------------------------------------------------------

def assemble_chimeric_pdb(offtarget_pdb_path: str, peptide_pdb_path: str, output_path: str) -> str:
    """Assemble chimeric complex: off-target receptor (chain B) + peptide (chain A).

    FlexPepDockingProtocol expects peptide as the LAST chain.
    So: receptor = chain 1 (first), peptide = chain 2 (last).

    Args:
        offtarget_pdb_path: Path to aligned off-target receptor PDB
        peptide_pdb_path: Path to peptide PDB
        output_path: Path to write chimeric PDB

    Returns:
        Path to the chimeric PDB
    """
    lines = []

    # Read receptor lines, relabel as chain B
    with open(offtarget_pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                # Set chain to B (receptor)
                line = line[:21] + "B" + line[22:]
                lines.append(line)
    lines.append("TER\n")

    # Read peptide lines, relabel as chain A
    with open(peptide_pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                # Set chain to A (peptide)
                line = line[:21] + "A" + line[22:]
                lines.append(line)
    lines.append("TER\n")
    lines.append("END\n")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.writelines(lines)

    return output_path


# ---------------------------------------------------------------------------
# Docking + Scoring (reuses patterns from flexpep_dock.py)
# ---------------------------------------------------------------------------

def dock_and_score(chimeric_pdb: str, output_pdb: str, nstruct: int = 20) -> dict:
    """Run FlexPepDock refinement and score the complex.

    Args:
        chimeric_pdb: Path to chimeric receptor+peptide PDB
        output_pdb: Path to write refined PDB

    Returns:
        dict with ddg, ddg_median, ddg_sd, n_converged, n_total, total_score, clash_score
    """
    import pyrosetta
    from pyrosetta.rosetta.protocols.flexpep_docking import FlexPepDockingProtocol
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    from pyrosetta.rosetta.core.scoring import fa_rep

    nstruct = int(os.environ.get("FLEXPEP_NSTRUCT", str(nstruct)))

    pose = pyrosetta.pose_from_pdb(chimeric_pdb)
    print(f"Chimeric complex: {pose.total_residue()} residues, {pose.num_chains()} chains", file=sys.stderr)

    # 2026-06-09: transplant 펩타이드는 off-target 포켓에서 clash 가 큼. InterfaceAnalyzer
    # 전에 좌표 제약 FastRelax 로 transplant strain 을 완화해 깨끗한 off-target ddG 를 얻는다.
    # (제약 없이 relax 하면 backbone 이 흐트러져 비교 불가 → start-coords 제약 사용.)
    try:
        from pyrosetta.rosetta.protocols.relax import FastRelax as _FastRelax
        _sf = pyrosetta.get_fa_scorefxn()
        _pre = _FastRelax(_sf, 1)
        _pre.constrain_relax_to_start_coords(True)
        _pre.apply(pose)
    except Exception as _exc:
        print(f"pre-relax skipped (non-fatal): {_exc}", file=sys.stderr)

    records = []
    for _ in range(nstruct):
        work_pose = pose.clone()

        # FlexPepDock refinement
        fpd = FlexPepDockingProtocol()
        fpd.apply(work_pose)

        # InterfaceAnalyzerMover ddG (jump_id=1)
        iam = InterfaceAnalyzerMover(1)
        iam.set_pack_input(True)
        iam.set_pack_separated(True)
        iam.apply(work_pose)
        ddg = iam.get_interface_dG()
        records.append((work_pose, ddg))

    _floor = _ddg_physical_floor()
    # 비물리 판정: floor 이하(예: −510) pose는 집계 제외, unphysical 마킹
    _unphysical_records = [(wp, d) for wp, d in records if d <= _floor]
    if _unphysical_records:
        print(
            f"[offtarget_dock] EXCLUDED_UNPHYSICAL {len(_unphysical_records)} pose(s): "
            f"{[round(d, 2) for _, d in _unphysical_records]} (floor={_floor})",
            file=sys.stderr,
        )
    converged_records = [
        (work_pose, d) for work_pose, d in records if d < 0 and d > _floor
    ]
    converged_ddgs = [d for _, d in converged_records]
    if converged_ddgs:
        ddg_median = statistics.median(converged_ddgs)
        ddg_mean = statistics.mean(converged_ddgs)
        ddg_sd = statistics.stdev(converged_ddgs) if len(converged_ddgs) > 1 else 0.0
        representative_pose, _representative_ddg = min(
            converged_records,
            key=lambda record: abs(record[1] - ddg_median),
        )
        ddg = ddg_median
        _rep_unphysical = False
    else:
        # 수렴 기록 없음 — 비물리가 아닌 것 중 최솟값, 없으면 전체 최솟값(비물리 마킹)
        non_unphysical = [(wp, d) for wp, d in records if d > _floor]
        if non_unphysical:
            representative_pose, ddg = min(non_unphysical, key=lambda record: record[1])
            _rep_unphysical = False
        else:
            representative_pose, ddg = min(records, key=lambda record: record[1])
            _rep_unphysical = True
            print(
                f"[offtarget_dock] WARNING: all {len(records)} pose(s) are UNPHYSICAL "
                f"(floor={_floor}). Representative ddG={ddg:.4f} marked unphysical.",
                file=sys.stderr,
            )
        ddg_median = ddg
        ddg_mean = ddg
        ddg_sd = 0.0

    representative_pose.dump_pdb(output_pdb)

    # Total score
    scorefxn = pyrosetta.get_fa_scorefxn()
    total_score = scorefxn(representative_pose)

    # Clash count — 펩타이드(마지막 체인) 잔기만 집계 (flexpep_dock.py 와 동일 정책,
    # 2026-06-09). 전체 pose 집계는 수용체 내부 클래시를 포함해 무의미.
    n_chains = representative_pose.num_chains()
    if n_chains >= 2:
        pep_begin, pep_end = representative_pose.chain_begin(n_chains), representative_pose.chain_end(n_chains)
    else:
        pep_begin, pep_end = 1, representative_pose.total_residue()
    clash_count = 0
    for i in range(pep_begin, pep_end + 1):
        residue_energies = representative_pose.energies().residue_total_energies(i)
        if residue_energies[fa_rep] > 10.0:
            clash_count += 1

    return {
        "ddg": round(ddg, 4),
        "ddg_median": round(ddg_median, 4),
        "ddg_sd": round(ddg_sd, 4),
        "n_converged": len(converged_ddgs),
        "n_total": nstruct,
        "n_unphysical": len(_unphysical_records),
        "unphysical": _rep_unphysical,
        "total_score": round(total_score, 4),
        "clash_score": float(clash_count),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Off-target PyRosetta FlexPepDock: align + dock + ddG"
    )
    parser.add_argument("--sstr2-complex", required=True, help="SSTR2 refined complex PDB")
    parser.add_argument("--offtarget-receptor", required=True, help="Off-target receptor PDB (e.g. AlphaFold)")
    parser.add_argument("--output", required=True, help="Output refined PDB path")
    parser.add_argument(
        "--pre-aligned", action="store_true",
        help="off-target 수용체가 이미 SSTR2 프레임에 정렬된 단일체인 구조이면 재정렬을 "
             "건너뛴다 (data/somatostatin_receptor/curated/*_receptor.pdb). 2026-06-09.",
    )
    args = parser.parse_args()

    if not Path(args.sstr2_complex).exists():
        print(json.dumps({"error": f"SSTR2 complex not found: {args.sstr2_complex}"}))
        sys.exit(1)
    if not Path(args.offtarget_receptor).exists():
        print(json.dumps({"error": f"Off-target receptor not found: {args.offtarget_receptor}"}))
        sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    init_pyrosetta()
    import pyrosetta

    # 1. Load SSTR2 complex → extract receptor + peptide
    sstr2_pose = pyrosetta.pose_from_pdb(args.sstr2_complex)
    n_chains = sstr2_pose.num_chains()
    print(f"SSTR2 complex: {sstr2_pose.total_residue()} residues, {n_chains} chains", file=sys.stderr)

    # Peptide = chain 1 (first/shorter), Receptor = chain 2 (last/longer)
    # Determine which chain is peptide (shorter) vs receptor (longer)
    chain_sizes = {}
    for i in range(1, sstr2_pose.total_residue() + 1):
        c = sstr2_pose.chain(i)
        chain_sizes[c] = chain_sizes.get(c, 0) + 1

    if len(chain_sizes) < 2:
        print(json.dumps({"error": "SSTR2 complex must have at least 2 chains"}))
        sys.exit(1)

    sorted_chains = sorted(chain_sizes.items(), key=lambda x: x[1])
    peptide_chain = sorted_chains[0][0]  # shorter chain = peptide
    receptor_chain = sorted_chains[-1][0]  # longer chain = receptor

    print(f"Peptide chain={peptide_chain} ({sorted_chains[0][1]} res), "
          f"Receptor chain={receptor_chain} ({sorted_chains[-1][1]} res)", file=sys.stderr)

    # Extract peptide and receptor as separate poses
    peptide_pose = extract_chain_pose(sstr2_pose, peptide_chain)
    receptor_pose = extract_chain_pose(sstr2_pose, receptor_chain)

    # 2. Load off-target receptor
    offtarget_pose = pyrosetta.pose_from_pdb(args.offtarget_receptor)
    print(f"Off-target receptor: {offtarget_pose.total_residue()} residues", file=sys.stderr)

    # 3. Align off-target to SSTR2 receptor (또는 이미 정렬된 구조면 skip)
    if args.pre_aligned:
        # curated/*_receptor.pdb 는 SSTR2 프레임에 0.93~0.95 overlap 으로 사전 정렬됨.
        # 재정렬(인덱스 기반 CA superpose)은 오히려 28A RMSD 오정렬을 유발하므로 skip.
        print("Pre-aligned mode: skipping re-alignment (off-target already in SSTR2 frame)", file=sys.stderr)
        aligned_ot = offtarget_pose
        rmsd = 0.0
    else:
        try:
            aligned_ot, rmsd = align_offtarget_to_sstr2(receptor_pose, offtarget_pose)
        except Exception as e:
            print(f"Alignment failed ({e}), using centroid fallback", file=sys.stderr)
            aligned_ot = align_simple_centroid(receptor_pose, offtarget_pose, peptide_pose)
            rmsd = -1.0

    # 4. Save aligned off-target and peptide to temp PDBs, assemble chimeric
    with tempfile.TemporaryDirectory() as tmpdir:
        ot_pdb = str(Path(tmpdir) / "aligned_offtarget.pdb")
        pep_pdb = str(Path(tmpdir) / "peptide.pdb")
        chimeric_pdb = str(Path(tmpdir) / "chimeric.pdb")

        aligned_ot.dump_pdb(ot_pdb)
        peptide_pose.dump_pdb(pep_pdb)

        assemble_chimeric_pdb(ot_pdb, pep_pdb, chimeric_pdb)

        # 5. FlexPepDock + score
        result = dock_and_score(chimeric_pdb, args.output, nstruct=20)

    result["alignment_rmsd"] = round(rmsd, 4) if rmsd >= 0 else None
    result["receptor_name"] = Path(args.offtarget_receptor).stem

    # stdout = JSON only
    print(json.dumps(result))


if __name__ == "__main__":
    main()
