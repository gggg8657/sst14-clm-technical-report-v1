#!/usr/bin/env python3
"""
flexpep_dock.py
===============
Standalone PyRosetta FlexPepDock refinement + InterfaceAnalyzer ddG script.

Called by AG_src/pipeline/step06_rosetta.py via subprocess:
    # Mode 1: Direct complex PDB input
    conda run -n bio-tools python AG_src/scripts/flexpep_dock.py \
        --input complex.pdb --output refined.pdb --protocol flexpep_refine

    # Mode 2: Reference complex + MutateResidue (preferred for variants)
    conda run -n bio-tools python AG_src/scripts/flexpep_dock.py \
        --input complex.pdb --output refined.pdb --protocol flexpep_refine \
        --reference-complex ref.pdb --target-sequence SGCKNFFWKTFTCA \
        --peptide-chain 1

stdout: JSON only (step06_rosetta.py L378 parses via json.loads)
stderr: all PyRosetta logs and diagnostics
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _ddg_physical_floor() -> float:
    try:
        return float(os.environ.get("DDG_PHYSICAL_FLOOR", "-150"))
    except ValueError:
        return -150.0


# ---------------------------------------------------------------------------
# PyRosetta initialization
# ---------------------------------------------------------------------------

def init_pyrosetta() -> None:
    """Initialize PyRosetta with muted output (all logs to stderr)."""
    import pyrosetta
    pyrosetta.init(
        options=(
            "-mute all -ex1 -ex2aro -ignore_unrecognized_res"
            " -flexPepDocking:pep_refine"
            " -constraints:cst_fa_weight 1.0"
        ),
        silent=True,
    )


# ---------------------------------------------------------------------------
# Complex preparation via MutateResidue
# ---------------------------------------------------------------------------

def prepare_complex_by_mutation(
    reference_pdb: str,
    target_sequence: str,
    peptide_chain: int = 1,
) -> Tuple["pyrosetta.Pose", int]:
    """Load a reference complex and mutate the peptide chain to the target sequence.

    This preserves the backbone conformation from the reference complex
    (e.g., AlphaFold3 prediction) and only changes sidechains.
    Much more reliable than assembling from separate PDBs.

    Args:
        reference_pdb: Path to the reference receptor-peptide complex.
        target_sequence: Target amino acid sequence (1-letter code).
        peptide_chain: Chain number of the peptide in the reference (1-indexed).

    Returns:
        (Pose with mutated peptide ready for FlexPepDock, resolved peptide chain index).
    """
    import pyrosetta
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue

    # 1-letter to 3-letter amino acid code mapping (MutateResidue requires 3-letter)
    _AA1TO3 = {
        'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
        'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
        'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
        'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
    }

    # D-amino acid 최소 지원: 소문자 1-letter → Rosetta fa_standard D-ResidueType
    # 3-letter 이름. 2026-07-01 conda env `bio-tools` (PyRosetta 2026.20) 에서
    # ChemicalManager.residue_type_set("fa_standard").has_name() 로 전수 검증 완료
    # (scratchpad/daa_verify_all.py) — 전부 "D"+표준 3-letter 형식이며
    # MutateResidue 적용 후 residue.type().is_d_aa() == True 확인됨.
    # Gly는 achiral(중심 탄소에 치환기 2개뿐)이므로 D/L 구분이 없어 L과 동일한 GLY 사용.
    _AA1TO3_D = {
        'a': 'DALA', 'c': 'DCYS', 'd': 'DASP', 'e': 'DGLU', 'f': 'DPHE',
        'g': 'GLY', 'h': 'DHIS', 'i': 'DILE', 'k': 'DLYS', 'l': 'DLEU',
        'm': 'DMET', 'n': 'DASN', 'p': 'DPRO', 'q': 'DGLN', 'r': 'DARG',
        's': 'DSER', 't': 'DTHR', 'v': 'DVAL', 'w': 'DTRP', 'y': 'DTYR',
    }

    pose = pyrosetta.pose_from_pdb(reference_pdb)

    # Collect chain residues and sequences from pose (chain index: 1-indexed)
    chain_residues: Dict[int, List[int]] = {}
    for i in range(1, pose.total_residue() + 1):
        cid = pose.chain(i)
        chain_residues.setdefault(cid, []).append(i)

    if not chain_residues:
        print("WARNING: No chains found in reference pose", file=sys.stderr)
        return pose, peptide_chain

    chain_lengths = {cid: len(res) for cid, res in chain_residues.items()}
    target_len = len(target_sequence)
    best_chain = min(chain_lengths, key=lambda cid: (abs(chain_lengths[cid] - target_len), chain_lengths[cid]))

    resolved_chain = peptide_chain
    if resolved_chain not in chain_residues:
        print(
            f"WARNING: Requested chain {peptide_chain} not found. Auto-selected chain {best_chain}.",
            file=sys.stderr,
        )
        resolved_chain = best_chain
    else:
        req_len = chain_lengths[resolved_chain]
        # If requested chain is far longer than target peptide, prefer the closest chain.
        if req_len > max(target_len + 10, target_len * 2) and best_chain != resolved_chain:
            print(
                f"WARNING: Requested chain {peptide_chain} length={req_len} mismatches target_len={target_len}. "
                f"Auto-switched to chain {best_chain} length={chain_lengths[best_chain]}.",
                file=sys.stderr,
            )
            resolved_chain = best_chain

    pep_residues = chain_residues[resolved_chain]

    if not pep_residues:
        print(
            f"WARNING: No residues found for chain {peptide_chain}",
            file=sys.stderr,
        )
        return pose, resolved_chain

    # Get reference peptide sequence
    ref_seq = "".join(pose.residue(i).name1() for i in pep_residues)
    print(
        f"Reference peptide (chain {resolved_chain}): {ref_seq} ({len(pep_residues)} residues)",
        file=sys.stderr,
    )
    print(f"Target sequence: {target_sequence}", file=sys.stderr)

    # Apply mutations where sequences differ (use 3-letter codes)
    # 소문자 1-letter는 D-amino acid로 해석(target_sequence 표기 관례:
    # 대문자=L-aa, 소문자=D-aa). L-aa 경로(대문자)는 기존 동작 완전 보존.
    seq_to_use = target_sequence[:len(pep_residues)]
    n_mutations = 0
    n_d_mutations = 0
    for idx, pose_resnum in enumerate(pep_residues):
        if idx < len(seq_to_use) and ref_seq[idx] != seq_to_use[idx]:
            target_char = seq_to_use[idx]
            is_d = target_char.islower()
            aa3 = _AA1TO3_D.get(target_char) if is_d else _AA1TO3.get(target_char)
            if aa3 is None:
                # 2026-07-01: silent skip은 오귀속 위험(치환 안 됐는데 성공한 것처럼
                # 보고됨) — unknown AA는 명시적 예외로 reject.
                raise ValueError(
                    f"Unknown amino acid code '{target_char}' at target_sequence "
                    f"index {idx} (pose_resnum={pose_resnum}); "
                    f"supported L-aa: {sorted(_AA1TO3)}, D-aa (lowercase): "
                    f"{sorted(_AA1TO3_D)}"
                )
            mutator = MutateResidue(pose_resnum, aa3)
            mutator.apply(pose)
            n_mutations += 1
            if is_d:
                n_d_mutations += 1
            print(
                f"  Mutated pos {pose_resnum}: {ref_seq[idx]} -> {target_char} ({aa3})"
                f"{' [D-aa]' if is_d else ''}",
                file=sys.stderr,
            )

    print(
        f"Applied {n_mutations} mutations ({n_d_mutations} D-aa)",
        file=sys.stderr,
    )
    return pose, resolved_chain


# ---------------------------------------------------------------------------
# Chain reordering (FlexPepDock expects peptide as LAST chain)
# ---------------------------------------------------------------------------

def reorder_peptide_last(
    pose: "pyrosetta.Pose", peptide_chain: int = 1,
) -> "pyrosetta.Pose":
    """Reorder pose so peptide chain is LAST (required by FlexPepDockingProtocol).

    Uses PDB text manipulation (most robust across PyRosetta versions):
    dump → reorder chain blocks → reload.
    """
    import pyrosetta
    import tempfile

    n_chains = pose.num_chains()
    if peptide_chain == n_chains:
        print(f"  Peptide already last chain ({peptide_chain}/{n_chains})", file=sys.stderr)
        return pose

    print(
        f"  Reordering: peptide chain {peptide_chain} → last (chain {n_chains})",
        file=sys.stderr,
    )

    # Dump current pose to PDB text
    tmp_orig = tempfile.mktemp(suffix=".pdb")
    pose.dump_pdb(tmp_orig)

    with open(tmp_orig) as fh:
        lines = fh.readlines()
    Path(tmp_orig).unlink(missing_ok=True)

    # Collect ATOM/HETATM/TER lines grouped by chain letter
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
            pass  # We'll add TER between chains ourselves
        elif not line.startswith("END"):
            header_lines.append(line)

    if len(chain_order) < 2:
        print("  WARNING: Only 1 chain found, skipping reorder", file=sys.stderr)
        return pose

    # Peptide chain letter (0-indexed from chain_order)
    pep_letter = chain_order[peptide_chain - 1]

    # Build new order: non-peptide chains first, then peptide last
    new_order = [c for c in chain_order if c != pep_letter] + [pep_letter]

    # Reletter chains A, B, C, ... and write new PDB
    new_lines = header_lines[:]
    for idx, old_cid in enumerate(new_order):
        new_cid = chr(65 + idx)  # A, B, C, ...
        for atom_line in chain_blocks[old_cid]:
            new_lines.append(atom_line[:21] + new_cid + atom_line[22:])
        new_lines.append("TER\n")
    new_lines.append("END\n")

    tmp_reord = tempfile.mktemp(suffix="_reord.pdb")
    with open(tmp_reord, "w") as fh:
        fh.writelines(new_lines)

    new_pose = pyrosetta.pose_from_pdb(tmp_reord)
    Path(tmp_reord).unlink(missing_ok=True)

    print(
        f"  Reordered: {new_pose.num_chains()} chains, "
        f"{new_pose.total_residue()} residues "
        f"(chain order: {' → '.join(new_order)})",
        file=sys.stderr,
    )
    return new_pose


# ---------------------------------------------------------------------------
# Disulfide constraint helpers
# ---------------------------------------------------------------------------

def _find_peptide_cys_residues(pose: "pyrosetta.Pose") -> List[int]:
    """Find Cys residue numbers in the last chain (peptide after reordering)."""
    n_chains = pose.num_chains()
    cys_residues = []
    for i in range(1, pose.total_residue() + 1):
        if pose.chain(i) == n_chains and pose.residue(i).name1() == "C":
            cys_residues.append(i)
    return cys_residues


def _add_disulfide_constraint(
    pose: "pyrosetta.Pose",
    cys1_resnum: int,
    cys2_resnum: int,
) -> None:
    """Add SG-SG AtomPairConstraint for a disulfide bond (2.05 A, sd=0.3)."""
    from pyrosetta.rosetta.core.scoring.constraints import AtomPairConstraint
    from pyrosetta.rosetta.core.scoring.func import HarmonicFunc
    from pyrosetta.rosetta.core.id import AtomID

    sg1 = AtomID(pose.residue(cys1_resnum).atom_index("SG"), cys1_resnum)
    sg2 = AtomID(pose.residue(cys2_resnum).atom_index("SG"), cys2_resnum)
    func = HarmonicFunc(2.05, 0.3)
    constraint = AtomPairConstraint(sg1, sg2, func)
    pose.add_constraint(constraint)
    print(
        f"  [disulfide] Added SG-SG constraint: res {cys1_resnum} <-> res {cys2_resnum} "
        f"(harmonic 2.05 A, sd=0.3)",
        file=sys.stderr,
    )


def _check_disulfide_distance(
    pose: "pyrosetta.Pose",
    cys1_resnum: int,
    cys2_resnum: int,
) -> Tuple[bool, float]:
    """Check SG-SG distance after refinement. Returns (intact, distance)."""
    sg1_xyz = pose.residue(cys1_resnum).xyz("SG")
    sg2_xyz = pose.residue(cys2_resnum).xyz("SG")
    distance = sg1_xyz.distance(sg2_xyz)
    intact = distance < 3.0
    print(
        f"  [disulfide] Post-refinement SG-SG distance: {distance:.3f} A "
        f"({'INTACT' if intact else 'BROKEN'})",
        file=sys.stderr,
    )
    return intact, distance


# ---------------------------------------------------------------------------
# FlexPepDock protocols
# ---------------------------------------------------------------------------

def run_flexpep_refine_pose(
    pose: "pyrosetta.Pose", output_pdb: str, nstruct: Optional[int] = None
) -> Tuple["pyrosetta.Pose", Dict]:
    """Run FlexPepDock refinement on an existing Pose.

    Runs nstruct independent refinements from the input pose and reports robust
    ddG statistics over converged (ddG < 0) structures. If nstruct is 1, the
    input pose is refined in-place, preserving the previous single-structure
    behavior.

    Returns (refined_pose, info) where info may contain disulfide_intact and
    sg_sg_distance if exactly 2 Cys found in the peptide, plus ddG statistics.
    """
    from pyrosetta.rosetta.protocols.flexpep_docking import FlexPepDockingProtocol

    if nstruct is None:
        env_nstruct = os.environ.get("FLEXPEP_NSTRUCT", "5")
        try:
            nstruct = int(env_nstruct)
        except ValueError:
            print(
                f"WARNING: Invalid FLEXPEP_NSTRUCT={env_nstruct!r}; using 8",
                file=sys.stderr,
            )
            nstruct = 8
    nstruct = max(1, int(nstruct))

    refined_records = []
    converged_ddgs: List[float] = []

    print(f"Running FlexPepDock refine nstruct={nstruct}", file=sys.stderr)
    _floor_val = _ddg_physical_floor()
    for i in range(nstruct):
        work_pose = pose if nstruct == 1 else pose.clone()
        run_info: Dict = {}
        cys_residues = _find_peptide_cys_residues(work_pose)

        # Use PyRosetta built-in disulfide detection instead of manual AtomPairConstraint.
        # Manual constraint after PDB dump→reload chain reordering can corrupt AtomIDs
        # and cause segfaults in FlexPepDockingProtocol.apply().
        if len(cys_residues) == 2:
            try:
                work_pose.conformation().detect_disulfides()
                print(
                    f"  [nstruct {i + 1}/{nstruct}] [disulfide] "
                    f"Auto-detected disulfides via conformation().detect_disulfides()",
                    file=sys.stderr,
                )
            except Exception as exc:
                print(
                    f"  [nstruct {i + 1}/{nstruct}] [disulfide] "
                    f"detect_disulfides() failed ({exc}), falling back to manual constraint",
                    file=sys.stderr,
                )
                try:
                    _add_disulfide_constraint(work_pose, cys_residues[0], cys_residues[1])
                except Exception as exc2:
                    print(
                        f"  [nstruct {i + 1}/{nstruct}] [disulfide] "
                        f"Manual constraint also failed ({exc2}), proceeding without "
                        f"disulfide constraint",
                        file=sys.stderr,
                    )

        fpd = FlexPepDockingProtocol()
        fpd.apply(work_pose)
        ddg = compute_interface_ddg(work_pose.clone())

        if len(cys_residues) == 2:
            intact, dist = _check_disulfide_distance(work_pose, cys_residues[0], cys_residues[1])
            run_info["disulfide_intact"] = intact
            run_info["sg_sg_distance"] = round(dist, 4)

        # 비물리 판정: floor 이하(예: −510)는 unphysical 마킹, 집계 제외
        _is_unphysical = ddg <= _floor_val
        run_info["unphysical"] = _is_unphysical
        refined_records.append((work_pose, ddg, run_info))
        _physical = ddg < 0 and not _is_unphysical
        if _physical:
            converged_ddgs.append(ddg)
        elif _is_unphysical:
            print(
                f"  [nstruct {i + 1}/{nstruct}] ddG={ddg:.4f} EXCLUDED_UNPHYSICAL (floor={_floor_val})",
                file=sys.stderr,
            )
        print(
            f"  [nstruct {i + 1}/{nstruct}] ddG={ddg:.4f} "
            f"({'converged' if _physical else 'excluded'})",
            file=sys.stderr,
        )

    _n_unphysical = sum(1 for _, ddg_r, _ in refined_records if ddg_r <= _floor_val)
    stats_info: Dict = {
        "ddg_median": None,
        "ddg_mean": None,
        "ddg_sd": None,
        "ddg_min": None,
        "n_converged": len(converged_ddgs),
        "n_total": nstruct,
        "n_unphysical": _n_unphysical,
        "converged": bool(converged_ddgs),
        "ddg": None,
    }

    if converged_ddgs:
        ddg_median = statistics.median(converged_ddgs)
        stats_info.update(
            {
                "ddg_median": round(ddg_median, 4),
                "ddg_mean": round(statistics.mean(converged_ddgs), 4),
                "ddg_sd": round(statistics.stdev(converged_ddgs), 4)
                if len(converged_ddgs) > 1
                else 0.0,
                "ddg_min": round(min(converged_ddgs), 4),
                "ddg": round(ddg_median, 4),
            }
        )
        # 대표 pose: 수렴(physical, ddG<0)한 것 중 median에 가장 가까운 pose 선택
        # unphysical pose는 대표 선택 후보에서 제외
        physical_records = [
            record for record in refined_records
            if record[1] < 0 and not record[2].get("unphysical", False)
        ]
        if physical_records:
            representative_pose, _, representative_info = min(
                physical_records,
                key=lambda record: abs(record[1] - ddg_median),
            )
        else:
            # fallback: 모든 수렴 기록 중 median 근접 (이미 converged_ddgs에 포함됨)
            representative_pose, _, representative_info = min(
                refined_records,
                key=lambda record: abs(record[1] - ddg_median),
            )
    else:
        # 수렴 기록 없음 — unphysical이 아닌 것 중 최솟값, 없으면 전체 최솟값
        non_unphysical = [r for r in refined_records if not r[2].get("unphysical", False)]
        if non_unphysical:
            representative_pose, _, representative_info = min(
                non_unphysical,
                key=lambda record: record[1],
            )
        else:
            representative_pose, _, representative_info = min(
                refined_records,
                key=lambda record: record[1],
            )

    stats_info.update(representative_info)
    # ★ FLEXPEP_SAVE_ALL_POSES=1 이면 개별 pose 전량 저장(수렴/비수렴 표기). 기본 off(다른 호출부 무영향).
    if os.environ.get("FLEXPEP_SAVE_ALL_POSES") == "1":
        from pathlib import Path as _P
        base = _P(output_pdb); stem = base.with_suffix("")
        saved = []
        for _i, (_pose, _ddg, _rinfo) in enumerate(refined_records, 1):
            _conv = "conv" if (_ddg < 0 and not _rinfo.get("unphysical", False)) else "excl"
            _pp = f"{stem}_pose{_i:02d}_{_conv}_ddg{round(_ddg,2)}.pdb"
            try:
                _pose.dump_pdb(_pp); saved.append(_pp)
            except Exception as _e:
                print(f"  [save-all] pose{_i} 저장실패: {_e}", file=sys.stderr)
        stats_info["all_pose_pdbs"] = saved
    representative_pose.dump_pdb(output_pdb)
    return representative_pose, stats_info


def run_flexpep_refine(
    input_pdb: str, output_pdb: str, nstruct: Optional[int] = None
) -> Tuple["pyrosetta.Pose", Dict]:
    """Run FlexPepDock refinement protocol on a receptor-peptide complex."""
    import pyrosetta

    pose = pyrosetta.pose_from_pdb(input_pdb)
    return run_flexpep_refine_pose(pose, output_pdb, nstruct=nstruct)


def run_flexpep_abinitio_pose(
    pose: "pyrosetta.Pose", output_pdb: str
) -> Tuple["pyrosetta.Pose", Dict]:
    """Run FlexPepDock ab-initio on an existing Pose.

    SS bond 처리: refine 모드(run_flexpep_refine_pose)와 동일하게
    detect_disulfides() 를 먼저 시도하고 실패 시 manual AtomPairConstraint 로 폴백.
    FlexPepDockingProtocol.apply() segfault 도 잡아 None 반환 대신 예외 재전파.

    Returns (refined_pose, disulfide_info).
    """
    from pyrosetta.rosetta.protocols.flexpep_docking import FlexPepDockingProtocol

    disulfide_info: Dict = {}
    cys_residues = _find_peptide_cys_residues(pose)

    # --- SS bond 처리: detect_disulfides() 우선, manual fallback ---
    if len(cys_residues) == 2:
        try:
            pose.conformation().detect_disulfides()
            print(
                "[ab-initio] [disulfide] Auto-detected disulfides via "
                "conformation().detect_disulfides()",
                file=sys.stderr,
            )
        except Exception as exc:
            print(
                f"[ab-initio] [disulfide] detect_disulfides() failed ({exc}), "
                "falling back to manual constraint",
                file=sys.stderr,
            )
            try:
                _add_disulfide_constraint(pose, cys_residues[0], cys_residues[1])
            except Exception as exc2:
                print(
                    f"[ab-initio] [disulfide] Manual constraint also failed ({exc2}), "
                    "proceeding without disulfide constraint",
                    file=sys.stderr,
                )

    # --- FlexPepDock ab-initio (segfault/RuntimeError 예외처리) ---
    try:
        fpd = FlexPepDockingProtocol()
        fpd.set_lowres_preoptimize(True)
        fpd.apply(pose)
    except Exception as exc:
        # segfault 는 Python 레벨에서 잡을 수 없지만 RuntimeError / ValueError 는 잡힌다.
        print(
            f"[ab-initio] FlexPepDockingProtocol.apply() raised {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise

    if len(cys_residues) == 2:
        intact, dist = _check_disulfide_distance(pose, cys_residues[0], cys_residues[1])
        disulfide_info["disulfide_intact"] = intact
        disulfide_info["sg_sg_distance"] = round(dist, 4)

    pose.dump_pdb(output_pdb)
    return pose, disulfide_info


def run_flexpep_abinitio(input_pdb: str, output_pdb: str) -> Tuple["pyrosetta.Pose", Dict]:
    """Run FlexPepDock ab-initio protocol from PDB file."""
    import pyrosetta

    pose = pyrosetta.pose_from_pdb(input_pdb)
    return run_flexpep_abinitio_pose(pose, output_pdb)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_interface_ddg(pose: "pyrosetta.Pose") -> float:
    """Compute interface dG using InterfaceAnalyzerMover.

    Uses jump_id=1 to separate receptor (chain A) from peptide (chain B).
    Returns dG in kcal/mol (more negative = stronger binding).
    """
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

    iam = InterfaceAnalyzerMover(1)
    iam.set_pack_input(True)
    iam.set_pack_separated(True)
    iam.apply(pose)
    return iam.get_interface_dG()


def compute_total_score(pose: "pyrosetta.Pose") -> float:
    """Compute total Rosetta energy score."""
    import pyrosetta
    scorefxn = pyrosetta.get_fa_scorefxn()
    return scorefxn(pose)


def compute_clash_score(pose: "pyrosetta.Pose") -> float:
    """Count PEPTIDE residues with high fa_rep (peptide-receptor steric clashes).

    Returns count of residues in the LAST chain (peptide — FlexPepDock reorders
    peptide last) where fa_rep > 10.0 REU.

    2026-06-09 fix: 이전 구현은 전체 pose(수용체+펩타이드, ~486 잔기)를 세어
    GPCR 수용체 내부 클래시까지 포함 → native·변이체 모두 clash~50 으로 QC
    게이트(<=10) 전원 탈락. 펩타이드 바인딩 QC 목적상 **펩타이드 체인 잔기만**
    세는 것이 옳다(펩타이드 잔기의 high fa_rep = 펩타이드-수용체 interface clash).
    """
    import pyrosetta
    from pyrosetta.rosetta.core.scoring import fa_rep

    scorefxn = pyrosetta.get_fa_scorefxn()
    scorefxn(pose)

    # 펩타이드 = 마지막 체인 (reorder_peptide_last 로 보장됨)
    n_chains = pose.num_chains()
    if n_chains >= 2:
        pep_begin = pose.chain_begin(n_chains)
        pep_end = pose.chain_end(n_chains)
    else:  # 단일 체인 폴백: 전체
        pep_begin, pep_end = 1, pose.total_residue()

    clash_count = 0
    for i in range(pep_begin, pep_end + 1):
        residue_energies = pose.energies().residue_total_energies(i)
        if residue_energies[fa_rep] > 10.0:
            clash_count += 1
    return float(clash_count)


# ---------------------------------------------------------------------------
# Pocket contact scoring
# ---------------------------------------------------------------------------

# SSTR2 binding pocket residue IDs (CA-CA distance threshold 8 Å)
# 출처: data/somatostatin_receptor/binding_pocket_SSTR2.json
_DEFAULT_POCKET_RESIDUE_IDS = [208, 209, 272, 273, 276]

# SST-14 pharmacophore positions (0-indexed from peptide start):
#   pos7=F, pos8=W, pos9=K, pos10=T  (AGCKNFFWKTFTSC, 1-indexed 8,9,10,11)
_PHARMACOPHORE_POSITIONS_0IDX = [7, 8, 9, 10]  # 0-indexed within peptide


def compute_pocket_contacts(
    pose: "pyrosetta.Pose",
    pocket_residue_ids: Optional[List[int]] = None,
    ca_distance_threshold: float = 8.0,
) -> int:
    """Count CA-CA contacts between peptide pharmacophore residues and receptor pocket.

    펩타이드 pharmacophore (pos7=F, pos8=W, pos9=K, pos10=T, 0-indexed) 잔기와
    SSTR2 binding pocket 잔기(기본값: 208,209,272,273,276) 간 CA-CA 거리 ≤ 8Å 접촉 수.

    0이면 펩타이드가 포켓 외부에 있다는 artifact 신호.

    Args:
        pose: 정제된 receptor+peptide complex Pose (peptide = 마지막 체인).
        pocket_residue_ids: 수용체 포켓 잔기 번호 목록 (Pose 기준 1-indexed).
                            None이면 _DEFAULT_POCKET_RESIDUE_IDS 사용.
        ca_distance_threshold: CA-CA 거리 임계값 (Å, 기본 8.0).

    Returns:
        int: 접촉 쌍 수 (0이면 포켓 외부).
    """
    if pocket_residue_ids is None:
        pocket_residue_ids = _DEFAULT_POCKET_RESIDUE_IDS

    n_chains = pose.num_chains()
    if n_chains < 2:
        print(
            "[pocket_contacts] WARNING: pose has <2 chains; returning 0",
            file=sys.stderr,
        )
        return 0

    # 펩타이드 = 마지막 체인
    pep_begin = pose.chain_begin(n_chains)
    pep_end = pose.chain_end(n_chains)
    pep_len = pep_end - pep_begin + 1

    # pharmacophore 잔기 pose 번호 계산 (0-indexed → pose 번호)
    pharma_pose_ids = []
    for idx in _PHARMACOPHORE_POSITIONS_0IDX:
        pose_id = pep_begin + idx
        if pose_id <= pep_end:
            pharma_pose_ids.append(pose_id)
        else:
            print(
                f"[pocket_contacts] WARNING: pharmacophore pos {idx} out of peptide range "
                f"({pep_begin}..{pep_end}, len={pep_len}), skipped",
                file=sys.stderr,
            )

    if not pharma_pose_ids:
        print(
            "[pocket_contacts] WARNING: no valid pharmacophore residues; returning 0",
            file=sys.stderr,
        )
        return 0

    # 포켓 잔기 유효성 필터 (존재하는 잔기만)
    valid_pocket_ids = []
    for rid in pocket_residue_ids:
        if 1 <= rid <= pose.total_residue():
            valid_pocket_ids.append(rid)
        else:
            print(
                f"[pocket_contacts] WARNING: pocket residue {rid} out of pose range "
                f"(total={pose.total_residue()}), skipped",
                file=sys.stderr,
            )

    if not valid_pocket_ids:
        print(
            "[pocket_contacts] WARNING: no valid pocket residues; returning 0",
            file=sys.stderr,
        )
        return 0

    # CA-CA 거리 계산
    contact_count = 0
    for pep_id in pharma_pose_ids:
        pep_res = pose.residue(pep_id)
        if not pep_res.has("CA"):
            continue
        pep_ca = pep_res.xyz("CA")
        for pocket_id in valid_pocket_ids:
            pocket_res = pose.residue(pocket_id)
            if not pocket_res.has("CA"):
                continue
            pocket_ca = pocket_res.xyz("CA")
            dist = pep_ca.distance(pocket_ca)
            if dist <= ca_distance_threshold:
                contact_count += 1

    print(
        f"[pocket_contacts] pharmacophore={pharma_pose_ids} vs pocket={valid_pocket_ids} "
        f"→ contacts={contact_count} (threshold={ca_distance_threshold}Å)",
        file=sys.stderr,
    )
    return contact_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PyRosetta FlexPepDock refinement + ddG calculation"
    )
    parser.add_argument("--input", required=True, help="Input complex PDB path")
    parser.add_argument("--output", required=True, help="Output refined PDB path")
    parser.add_argument(
        "--protocol",
        default="flexpep_refine",
        choices=["flexpep_refine", "flexpep_abinitio"],
        help="Docking protocol (default: flexpep_refine)",
    )
    parser.add_argument(
        "--reference-complex", default="",
        help="Reference complex PDB for MutateResidue approach (preferred)",
    )
    parser.add_argument(
        "--target-sequence", default="",
        help="Target peptide sequence for MutateResidue (used with --reference-complex)",
    )
    parser.add_argument(
        "--peptide-chain", type=int, default=1,
        help="Peptide chain number in reference complex (default: 1)",
    )
    parser.add_argument(
        "--nstruct",
        type=int,
        default=None,
        help="Number of independent FlexPepDock refine runs (default: FLEXPEP_NSTRUCT or 8)",
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(
            json.dumps({"error": f"Input PDB not found: {args.input}"}),
            file=sys.stderr,
        )
        sys.exit(1)

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Initialize PyRosetta (logs go to stderr via -mute all)
    init_pyrosetta()

    # Decide input mode: reference+mutation vs. direct PDB
    use_reference = (
        args.reference_complex
        and args.target_sequence
        and Path(args.reference_complex).exists()
    )

    disulfide_info: Dict = {}

    if use_reference:
        print(
            f"Mode: MutateResidue from reference complex",
            file=sys.stderr,
        )
        pose, resolved_chain = prepare_complex_by_mutation(
            args.reference_complex,
            args.target_sequence,
            args.peptide_chain,
        )
        # Reorder so peptide is LAST chain (FlexPepDock requirement)
        pose = reorder_peptide_last(pose, resolved_chain)

        # Score BEFORE refinement
        pre_score = compute_total_score(pose)
        print(f"Pre-refinement total_score: {pre_score:.2f}", file=sys.stderr)

        # Run FlexPepDock on the mutated pose
        if args.protocol == "flexpep_abinitio":
            pose, disulfide_info = run_flexpep_abinitio_pose(pose, args.output)
        else:
            pose, disulfide_info = run_flexpep_refine_pose(
                pose,
                args.output,
                nstruct=args.nstruct,
            )
    else:
        # Direct PDB mode
        import pyrosetta as _pyr
        pose = _pyr.pose_from_pdb(args.input)
        # Reorder so peptide is LAST chain (FlexPepDock requirement)
        pose = reorder_peptide_last(pose, args.peptide_chain)
        pre_score = compute_total_score(pose)
        print(f"Pre-refinement total_score: {pre_score:.2f}", file=sys.stderr)

        if args.protocol == "flexpep_abinitio":
            pose, disulfide_info = run_flexpep_abinitio_pose(pose, args.output)
        else:
            pose, disulfide_info = run_flexpep_refine_pose(
                pose,
                args.output,
                nstruct=args.nstruct,
            )

    # Score the refined pose
    total_score = compute_total_score(pose)
    ddg = disulfide_info["ddg"] if "ddg" in disulfide_info else round(compute_interface_ddg(pose), 4)
    clash = compute_clash_score(pose)
    score_delta = total_score - pre_score

    # Binding pocket contact score — pharmacophore(pos7=F,8=W,9=K,10=T) vs receptor pocket CA-CA ≤8Å
    # 0이면 펩타이드가 포켓 외부에 위치하는 artifact 신호 (scoring_pipeline 소비 키)
    pocket_contacts = compute_pocket_contacts(pose)

    print(f"Post-refinement total_score: {total_score:.2f} (delta={score_delta:.2f})", file=sys.stderr)

    # 대표 ddG의 비물리 여부 판정 (floor 이하는 raw 기록에서도 unphysical 마킹)
    _final_floor = _ddg_physical_floor()
    _ddg_unphysical: bool = (ddg is not None and ddg <= _final_floor)
    if _ddg_unphysical:
        print(
            f"WARNING: final representative ddG={ddg:.4f} is UNPHYSICAL (floor={_final_floor}). "
            f"Marking unphysical=True in output.",
            file=sys.stderr,
        )

    result = {
        "ddg": ddg,
        "total_score": round(total_score, 4),
        "pre_score": round(pre_score, 4),
        "score_delta": round(score_delta, 4),
        "clash_score": clash,
        "constraint_violations": 0,
        "pocket_contacts": pocket_contacts,
        "unphysical": _ddg_unphysical,
    }
    if disulfide_info:
        result.update(disulfide_info)
        # disulfide_info.update 이후에도 unphysical 플래그 보존
        result["unphysical"] = _ddg_unphysical

    # stdout = JSON only (parsed by step06_rosetta.py)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
