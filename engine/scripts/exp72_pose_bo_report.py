#!/usr/bin/env python3
"""exp72 Pose BO (rigid 6D) 결과 분석 — 발표자료용 findings/metrics 생성.

Part 1: BO 궤적 분석 (18서열 bo_history.jsonl 파싱)
Part 2: best pose 구조 메트릭 (FWKT 접촉, SS 거리, native/init 대비 RMSD)

전부 실제 파일에서만 읽음 — 숫자 생성/추정 없음.
"""
import json
import glob
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BO_DIR = REPO / "runs/exp72_analysis/pose_bo_rigid"
POOL_WORK = REPO / "runs/exp72_analysis/pool_work"
OUT_DIR = REPO / "runs/exp72_analysis/pose_bo_presentation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NATIVE_SEQ = "AGCKNFFWKTFTSC"
POCKET_RESIDUE_IDS = [208, 209, 272, 273, 276]  # receptor(chain B), PDB numbering
FWKT_POS = [7, 8, 9, 10]  # peptide(chain A), 1-indexed PDB numbering (F,W,K,T)


# ---------------------------------------------------------------------------
# 순수 PDB 파서 (PyRosetta 미사용 — 좌표 비교만 필요하므로 텍스트 파싱으로 재구현)
# ---------------------------------------------------------------------------

def parse_pdb(path):
    """Return dict: {(chain, resnum): {"resname": str, atomname: np.array([x,y,z])}}"""
    atoms = {}
    with open(path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atomname = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21].strip()
            try:
                resnum = int(line[22:26])
            except ValueError:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            key = (chain, resnum)
            if key not in atoms:
                atoms[key] = {"resname": resname}
            atoms[key][atomname] = np.array([x, y, z])
    return atoms


def get_ca_dict(atoms, chain):
    """{resnum: xyz} for CA atoms in given chain."""
    out = {}
    for (ch, resnum), d in atoms.items():
        if ch == chain and "CA" in d:
            out[resnum] = d["CA"]
    return out


def kabsch_fit(P, Q):
    """Superpose P onto Q (both Nx3 arrays, matched order). Returns (R, t) s.t. P@R.T + t ~ Q."""
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = Q.mean(axis=0) - R @ P.mean(axis=0)
    return R, t


def apply_transform(R, t, X):
    return (R @ X.T).T + t


def superpose_receptor_and_get_peptide_rmsd(mobile_path, ref_path, receptor_chain="B", peptide_chain="A"):
    """수용체(chain B) CA로 mobile→ref 좌표계 정렬 후, 펩타이드(chain A) CA RMSD 계산.

    Returns (rmsd, n_common_receptor_atoms, n_common_peptide_atoms) or (None, 0, 0) on failure.
    """
    mobile = parse_pdb(mobile_path)
    ref = parse_pdb(ref_path)
    mob_rec = get_ca_dict(mobile, receptor_chain)
    ref_rec = get_ca_dict(ref, receptor_chain)
    common_rec = sorted(set(mob_rec) & set(ref_rec))
    if len(common_rec) < 10:
        return None, len(common_rec), 0
    P = np.array([mob_rec[r] for r in common_rec])
    Q = np.array([ref_rec[r] for r in common_rec])
    R, t = kabsch_fit(P, Q)

    mob_pep = get_ca_dict(mobile, peptide_chain)
    ref_pep = get_ca_dict(ref, peptide_chain)
    common_pep = sorted(set(mob_pep) & set(ref_pep))
    if not common_pep:
        return None, len(common_rec), 0
    Pp = np.array([mob_pep[r] for r in common_pep])
    Qp = np.array([ref_pep[r] for r in common_pep])
    Pp_t = apply_transform(R, t, Pp)
    rmsd = float(np.sqrt(np.mean(np.sum((Pp_t - Qp) ** 2, axis=1))))
    return rmsd, len(common_rec), len(common_pep)


def fwkt_pocket_contacts(atoms, peptide_chain="A", receptor_chain="B",
                          fwkt_pos=FWKT_POS, pocket_ids=POCKET_RESIDUE_IDS):
    """FWKT(peptide 7-10) CA - pocket(receptor) CA 접촉 수 @ 8/10/12A. 및 최소거리."""
    pep_ca = get_ca_dict(atoms, peptide_chain)
    rec_ca = get_ca_dict(atoms, receptor_chain)
    dists = []
    for p in fwkt_pos:
        if p not in pep_ca:
            continue
        for r in pocket_ids:
            if r not in rec_ca:
                continue
            d = float(np.linalg.norm(pep_ca[p] - rec_ca[r]))
            dists.append(d)
    if not dists:
        return {"min_dist": None, "n_contacts_8": None, "n_contacts_10": None, "n_contacts_12": None, "n_pairs": 0}
    dists = np.array(dists)
    return {
        "min_dist": round(float(dists.min()), 3),
        "n_contacts_8": int((dists <= 8.0).sum()),
        "n_contacts_10": int((dists <= 10.0).sum()),
        "n_contacts_12": int((dists <= 12.0).sum()),
        "n_pairs": int(len(dists)),
    }


def disulfide_distance(atoms, peptide_chain="A"):
    """peptide chain 내 CYS 잔기 쌍 SG-SG 거리. 서열 내 실제 CYS 잔기 번호를 사용(가정 없음)."""
    cys_resnums = sorted(rn for (ch, rn), d in atoms.items() if ch == peptide_chain and d.get("resname") == "CYS")
    if len(cys_resnums) != 2:
        return {"cys_resnums": cys_resnums, "sg_sg_distance": None, "intact": None}
    r1, r2 = cys_resnums
    sg1 = atoms.get((peptide_chain, r1), {}).get("SG")
    sg2 = atoms.get((peptide_chain, r2), {}).get("SG")
    if sg1 is None or sg2 is None:
        return {"cys_resnums": cys_resnums, "sg_sg_distance": None, "intact": None}
    d = float(np.linalg.norm(sg1 - sg2))
    return {"cys_resnums": cys_resnums, "sg_sg_distance": round(d, 3), "intact": bool(d < 3.0)}


def peptide_com_vs_pocket_centroid(atoms, peptide_chain="A", receptor_chain="B", pocket_ids=POCKET_RESIDUE_IDS):
    pep_ca = get_ca_dict(atoms, peptide_chain)
    rec_ca = get_ca_dict(atoms, receptor_chain)
    if not pep_ca:
        return {"com_dist_to_pocket_centroid": None}
    pep_com = np.mean(list(pep_ca.values()), axis=0)
    pocket_xyz = [rec_ca[r] for r in pocket_ids if r in rec_ca]
    if not pocket_xyz:
        return {"com_dist_to_pocket_centroid": None}
    pocket_centroid = np.mean(pocket_xyz, axis=0)
    return {"com_dist_to_pocket_centroid": round(float(np.linalg.norm(pep_com - pocket_centroid)), 3)}


# ---------------------------------------------------------------------------
# Part 1: BO trajectory
# ---------------------------------------------------------------------------

def load_history(seq):
    fp = BO_DIR / f"{seq}_bo_history.jsonl"
    recs = []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def analyze_trajectory(seq, recs):
    n = len(recs)
    ddgs = [r.get("ddg_flex") for r in recs]
    fs = [r.get("f") for r in recs]
    ss_ok = [r.get("ss_ok") for r in recs]
    fwkt_ok = [r.get("fwkt_ok") for r in recs]
    dg_mm = [r.get("dg_bind_mm") for r in recs]

    n_none_ddg = sum(1 for x in ddgs if x is None)
    valid = [(i, r) for i, r in enumerate(recs) if r.get("ddg_flex") is not None]

    init_ddg = recs[0].get("ddg_flex") if recs else None
    init_f = recs[0].get("f") if recs else None

    if valid:
        best_ddg_idx, best_ddg_rec = min(valid, key=lambda ir: ir[1]["ddg_flex"])
    else:
        best_ddg_idx, best_ddg_rec = None, None

    f_valid = [(i, r) for i, r in enumerate(recs) if r.get("f") is not None]
    if f_valid:
        best_f_idx, best_f_rec = max(f_valid, key=lambda ir: ir[1]["f"])
    else:
        best_f_idx, best_f_rec = None, None

    improve_vs_init = None
    if init_ddg is not None and best_ddg_rec is not None:
        improve_vs_init = round(init_ddg - best_ddg_rec["ddg_flex"], 4)  # positive = better (more negative ddG)

    dg_mm_coverage = sum(1 for x in dg_mm if x is not None)
    ss_ok_mean = sum(x for x in ss_ok if x is not None) / len(ss_ok) if ss_ok else None
    fwkt_ok_mean = sum(x for x in fwkt_ok if x is not None) / len(fwkt_ok) if fwkt_ok else None

    return {
        "sequence": seq,
        "n_iter_total": n,
        "n_ddg_null": n_none_ddg,
        "n_ddg_valid": len(valid),
        "init_iter": 0,
        "init_ddg": init_ddg,
        "init_f": init_f,
        "best_ddg": best_ddg_rec["ddg_flex"] if best_ddg_rec else None,
        "best_ddg_iter": best_ddg_rec["iter"] if best_ddg_rec else None,
        "best_ddg_pdb": best_ddg_rec["refined_pdb"] if best_ddg_rec else None,
        "best_f": best_f_rec["f"] if best_f_rec else None,
        "best_f_iter": best_f_rec["iter"] if best_f_rec else None,
        "best_f_ddg": best_f_rec.get("ddg_flex") if best_f_rec else None,
        "best_f_fwkt_ok": best_f_rec.get("fwkt_ok") if best_f_rec else None,
        "best_f_pdb": best_f_rec["refined_pdb"] if best_f_rec else None,
        "improvement_ddg_reu": improve_vs_init,  # positive REU = BO improved binding vs iter0
        "ss_ok_mean": ss_ok_mean,
        "fwkt_ok_mean": fwkt_ok_mean,
        "dg_bind_mm_coverage": dg_mm_coverage,
        "dg_bind_mm_total": len(dg_mm),
    }


def main():
    summary_fp = BO_DIR / "bo_summary.jsonl"
    seqs = []
    with open(summary_fp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seqs.append(json.loads(line)["sequence"])

    metrics = {"sequences": {}}

    print(f"[part1] {len(seqs)} sequences")
    for seq in seqs:
        recs = load_history(seq)
        traj = analyze_trajectory(seq, recs)
        metrics["sequences"][seq] = {"trajectory": traj}
        print(f"  {seq}: init_ddg={traj['init_ddg']} best_ddg={traj['best_ddg']}"
              f"@iter{traj['best_ddg_iter']} best_f_iter={traj['best_f_iter']}"
              f" improve={traj['improvement_ddg_reu']}")

    # Part 2: structural metrics on BO-best (by f, matching bo_summary.jsonl definition) pose
    print("[part2] structural metrics")
    native_source = POOL_WORK / f"{NATIVE_SEQ}.pdb"
    assert native_source.exists(), f"native source missing: {native_source}"

    for seq, entry in metrics["sequences"].items():
        traj = entry["trajectory"]
        source_pdb = POOL_WORK / f"{seq}.pdb"
        best_f_pdb = traj.get("best_f_pdb")
        best_ddg_pdb = traj.get("best_ddg_pdb")

        struct = {}
        if best_f_pdb and Path(best_f_pdb).exists():
            atoms_best = parse_pdb(best_f_pdb)
            struct["fwkt_contacts_best"] = fwkt_pocket_contacts(atoms_best)
            struct["disulfide_best"] = disulfide_distance(atoms_best)
            struct["com_vs_pocket_best"] = peptide_com_vs_pocket_centroid(atoms_best)

            if source_pdb.exists():
                rmsd, n_rec, n_pep = superpose_receptor_and_get_peptide_rmsd(best_f_pdb, str(source_pdb))
                struct["rmsd_vs_init_source"] = {
                    "rmsd": round(rmsd, 3) if rmsd is not None else None,
                    "n_receptor_ca_aligned": n_rec, "n_peptide_ca_compared": n_pep,
                }
            rmsd_n, n_rec_n, n_pep_n = superpose_receptor_and_get_peptide_rmsd(best_f_pdb, str(native_source))
            struct["rmsd_vs_native"] = {
                "rmsd": round(rmsd_n, 3) if rmsd_n is not None else None,
                "n_receptor_ca_aligned": n_rec_n, "n_peptide_ca_compared": n_pep_n,
            }
        else:
            struct["error"] = f"best_f_pdb missing or not found: {best_f_pdb}"

        # also baseline (source, pre-BO) structural metrics for comparison table
        if source_pdb.exists():
            atoms_src = parse_pdb(source_pdb)
            struct["fwkt_contacts_source"] = fwkt_pocket_contacts(atoms_src)
            struct["disulfide_source"] = disulfide_distance(atoms_src)

        entry["structure"] = struct
        print(f"  {seq}: fwkt8={struct.get('fwkt_contacts_best', {}).get('n_contacts_8')}"
              f" ss_dist={struct.get('disulfide_best', {}).get('sg_sg_distance')}"
              f" rmsd_native={struct.get('rmsd_vs_native', {}).get('rmsd')}"
              f" rmsd_init={struct.get('rmsd_vs_init_source', {}).get('rmsd')}")

    # native's own structural baseline (for reference row in table)
    atoms_native = parse_pdb(native_source)
    metrics["native_reference"] = {
        "source_pdb": str(native_source),
        "fwkt_contacts": fwkt_pocket_contacts(atoms_native),
        "disulfide": disulfide_distance(atoms_native),
        "com_vs_pocket": peptide_com_vs_pocket_centroid(atoms_native),
    }

    out_fp = OUT_DIR / "metrics.json"
    with out_fp.open("w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_fp}")


if __name__ == "__main__":
    main()
