#!/usr/bin/env python3
"""EXP72 V3 — shortlist 선택성(Δmargin) 도킹. 재정렬 off-target(biology 세트) 사용.

목적: nstruct20 재도킹된 shortlist(양 arm) + native 를 SSTR1/3/4/5(재정렬 RMSD<2Å)에 도킹해
Δmargin(= min(offtarget_ddg) − sstr2_ddg, 양수=SSTR2 선택적) 산출. 검증축을 ddG→다목적 확장.

핵심: 코어 multiobjective.py 미편집 — screen_selectivity(offtarget_receptors=재정렬dict) 명시 전달.
native_selectivity_baseline도 신규 수용체로 재산출(구 값 무효).
출력: runs/exp72_system/v3_selectivity.jsonl (재개가능). 샤딩 --shard-index/--shard-count.
NO MOCK: 실제 off-target 도킹.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

TEMPLATE = str(REPO / "data/somatostatin_receptor/SSTR2_SST14_complex_boltz_1.pdb")
NATIVE = "AGCKNFFWKTFTSC"
NATIVE_DDG = -20.28
ALIGN = REPO / "_workspace/05_engineer-backend_offtarget_receptors_seqalign"
REALIGNED = {
    "SSTR1": str(ALIGN / "SSTR1_receptor_9ik8_seqalign.pdb"),
    "SSTR3": str(ALIGN / "SSTR3_receptor_8xir_seqalign.pdb"),
    "SSTR4": str(ALIGN / "SSTR4_receptor_7xmt_seqalign.pdb"),
    "SSTR5": str(ALIGN / "SSTR5_receptor_8zbj_seqalign.pdb"),
}


def _num(x):
    return x if isinstance(x, (int, float)) else None


def load_shortlist(src_jsonl=None, pdb_dir=None):
    """재도킹된 후보(status ok) + native. sstr2 복합체 PDB·on_target_ddg 포함.

    기본은 shortlist(phase1_confirm/phase1_work). --targets/--pdb-dir 로 풀 winner 등
    임의 jsonl(sequence/ddg_median/disulfide_flag 포함) + 대표 복합체 PDB 디렉토리 지정 가능(funnel용).
    마진 로직·필드 저장은 불변 — 입력 소스만 일반화."""
    out = []
    p = REPO / (src_jsonl or "runs/exp72_system/phase1_confirm.jsonl")
    work = REPO / (pdb_dir or "runs/exp72_system/phase1_work")
    for line in p.open(errors="ignore"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("status") != "ok" or r.get("ddg_median") is None:
            continue
        pdb = work / f"{r['sequence']}.pdb"
        if pdb.exists():
            out.append({"sequence": r["sequence"], "arm": r.get("arm") or r.get("provenance"),
                        "pdb": str(pdb), "on_target_ddg": r["ddg_median"],
                        "disulfide_flag": r.get("disulfide_flag")})
    # native (template 자체가 native 복합체) — home-advantage 보정 기준선(동일 수용체세트)
    out.append({"sequence": NATIVE, "arm": "native", "pdb": TEMPLATE,
                "on_target_ddg": NATIVE_DDG, "disulfide_flag": "normal"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default="runs/exp72_system/v3_selectivity.jsonl")
    ap.add_argument("--targets", default=None, help="임의 jsonl(풀 winner 등) — 기본 shortlist")
    ap.add_argument("--pdb-dir", default=None, help="대표 복합체 PDB 디렉토리 — 기본 phase1_work")
    args = ap.parse_args()

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.open(errors="ignore"):
            try:
                r = json.loads(line)
                # 실제 Δmargin 있는 것만 완료 처리 (빈 offtarget=결측은 재실행)
                if r.get("delta_margin") is not None:
                    done.add(r["sequence"])
            except Exception:
                pass

    targets = load_shortlist(args.targets, args.pdb_dir)
    mine = [t for i, t in enumerate(targets) if i % args.shard_count == args.shard_index
            and t["sequence"] not in done]
    print(f"[v3 shard {args.shard_index}/{args.shard_count}] 처리 {len(mine)} "
          f"(전체 {len(targets)}, 완료 {len(done)}) off-target=재정렬 biology세트", file=sys.stderr, flush=True)

    from pyrosetta_flow.multiobjective import screen_selectivity

    for t in mine:
        rec = {"sequence": t["sequence"], "arm": t["arm"], "on_target_ddg": t["on_target_ddg"],
               "disulfide_flag": t["disulfide_flag"], "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "offtarget_set": "biology_realigned(9IK8/8XIR/7XMT/8ZBJ)"}
        try:
            sel = screen_selectivity(sstr2_complex_pdb=t["pdb"], on_target_ddg=t["on_target_ddg"],
                                     offtarget_receptors=REALIGNED, conda_env="bio-tools",
                                     timeout=args.timeout)
            rec["delta_margin"] = _num(sel.get("selectivity_margin"))
            ot = sel.get("offtarget_ddg")
            # offtarget_ddg 를 순수 스칼라 dict 로 정규화 (SelectivityResult 등 비직렬화 객체 차단)
            if isinstance(ot, dict):
                rec["offtarget_ddg"] = {k: _num(v) for k, v in ot.items()}
            else:
                rec["offtarget_ddg"] = _num(ot)
            rec["status"] = "ok"
        except Exception as exc:
            rec["status"] = "error"; rec["error"] = f"{type(exc).__name__}: {exc}"
        with out_path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")   # default=str: 잔여 비직렬화 방어
        print(f"  [{t['arm']}] {t['sequence']} Δmargin={rec.get('delta_margin')} "
              f"offtarget={rec.get('offtarget_ddg')} status={rec['status']}", file=sys.stderr, flush=True)

    print(f"[v3 shard {args.shard_index}] DONE", file=sys.stderr)


if __name__ == "__main__":
    main()
