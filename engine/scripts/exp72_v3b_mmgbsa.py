#!/usr/bin/env python3
"""EXP72 V3b — shortlist MM-GBSA (정밀 결합에너지, 검증축 확장).

mmgbsa env 로 실행. 데몬의 검증된 함수 재사용:
  _build_complex_pdb_by_mutation(seq) → A(수용체)/B(펩타이드) 복합체 (체인 안전) → _run_one → dg_bind.
대상: phase1_confirm(status ok) shortlist + native. 도킹보다 빠름(~12s/pose).
출력: runs/exp72_system/v3b_mmgbsa.jsonl (재개가능).
NO MOCK: 실제 OpenMM MM-GBSA. 실패=fail-closed(None).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

NATIVE = "AGCKNFFWKTFTSC"


def load_targets(src=None):
    """기본 shortlist(phase1_confirm) 또는 --targets jsonl(풀 winner 등) 로드.
    각 후보의 (sequence, arm/provenance) 튜플 반환. 중복 dedup, native 자동 포함."""
    seqs = []
    seen = set()
    p = REPO / (src or "runs/exp72_system/phase1_confirm.jsonl")
    for line in p.open(errors="ignore"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        # phase1_confirm 은 status ok 요구, 풀 winner 는 status 없음(전량 견고)
        if src is not None or r.get("status") == "ok":
            s = r["sequence"]
            if s not in seen:
                seqs.append((s, r.get("arm") or r.get("provenance"))); seen.add(s)
    if NATIVE not in seen:
        seqs.append((NATIVE, "native"))
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=None, help="임의 jsonl(풀 winner 등, 기본 shortlist)")
    ap.add_argument("--out", default="runs/exp72_system/v3b_mmgbsa.jsonl")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    args = ap.parse_args()

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.open(errors="ignore"):
            try:
                r = json.loads(line)
                if r.get("dg_bind") is not None:
                    done.add(r["sequence"])
            except Exception:
                pass

    import mmgbsa_daemon as md   # 데몬 함수 재사용 (import 시 main 미실행)

    all_targets = load_targets(args.targets)
    # 샤딩 적용 후 재개 필터
    mine = [(s, a) for i, (s, a) in enumerate(all_targets)
            if i % args.shard_count == args.shard_index and s not in done]
    targets = mine
    print(f"[v3b mmgbsa shard {args.shard_index}/{args.shard_count}] 처리 {len(targets)} "
          f"(전체 {len(all_targets)}, 완료 {len(done)})", file=sys.stderr, flush=True)

    for seq, arm in targets:
        rec = {"sequence": seq, "arm": arm, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            pdb = md._build_complex_pdb_by_mutation(seq)   # A/B 체인 복합체(mutated_approx)
            if pdb is None:
                rec["dg_bind"] = None; rec["error"] = "complex build 실패"
            else:
                res = md._run_one(pdb, seq, seq, "mutated_approx", source="silo_b")
                if res:
                    rec.update({"dg_bind": res.get("dg_bind"), "dg_bind_sd": res.get("dg_bind_sd"),
                                "n_snapshots": res.get("n_snapshots"),
                                "ss_bond_verified": res.get("ss_bond_verified"),
                                "structure_source": "mutated_approx"})
                else:
                    rec["dg_bind"] = None; rec["error"] = "mmgbsa 실패(fail-closed)"
        except Exception as exc:
            rec["dg_bind"] = None; rec["error"] = f"{type(exc).__name__}: {exc}"
        with out_path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  [{arm}] {seq} dG_bind={rec.get('dg_bind')} {rec.get('error','')}", file=sys.stderr, flush=True)

    print("[v3b mmgbsa] DONE", file=sys.stderr)


if __name__ == "__main__":
    main()
