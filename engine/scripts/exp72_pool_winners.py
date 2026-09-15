#!/usr/bin/env python3
"""풀 재도킹(pool_redock.jsonl)에서 선택성 funnel 대상 winner 선정.

기준(정직·보수적):
  - native robust(-20.28) 능가 (ddg_median < NATIVE)
  - 이황화결합 정상형성 (disulfide_intact and disulfide_flag == 'normal')  ← compressed(과도refine) 아티팩트 제외
  - floppy 아님 (ddg_sd < SD_MAX)
  - 수렴 pose 충분 (n_converged >= NCONV_MIN)
정렬 = ddg_median 오름차순(강결합 우선), 상한 CAP(연산비용). 제외수·상한은 로그로 명시(무단절단 금지).
출력 = winner jsonl (v3_selectivity --targets 입력형식: pool_redock 레코드 그대로).
"""
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
POOL = REPO / "runs/exp72_analysis/pool_redock.jsonl"
POOL_WORK = REPO / "runs/exp72_analysis/pool_work"
OUT = REPO / "_workspace/EXP72_POOL_WINNERS.json"

NATIVE = -20.28   # native robust median (canonical, controlled nstruct10)
SD_MAX = 15.0
NCONV_MIN = 3
CAP = 0           # 0=무제한(견고 winner 전량, 무단절단 방지). 양수면 상한(강결합 우선 top N)


def main():
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else CAP
    recs = [json.loads(l) for l in POOL.open(errors="ignore") if l.strip()]
    total = len(recs)
    # 대표 복합체 PDB 존재하는 것만(선택성 입력 필수)
    def has_pdb(s):
        return (POOL_WORK / f"{s}.pdb").exists()

    beat = [r for r in recs if r.get("ddg_median") is not None and r["ddg_median"] < NATIVE]
    ss_ok = [r for r in beat if r.get("disulfide_intact") and r.get("disulfide_flag") == "normal"]
    firm = [r for r in ss_ok if (r.get("ddg_sd") or 99) < SD_MAX and (r.get("n_converged") or 0) >= NCONV_MIN]
    with_pdb = [r for r in firm if has_pdb(r["sequence"])]
    with_pdb.sort(key=lambda r: r["ddg_median"])
    winners = with_pdb if cap <= 0 else with_pdb[:cap]

    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in winners) + "\n")

    # 무단절단 금지 — 각 단계 통과/탈락 수 명시
    print(f"[pool-winners] 풀 {total}건 → native능가 {len(beat)} → 이황정상 {len(ss_ok)} "
          f"→ 견고(sd<{SD_MAX},conv>={NCONV_MIN}) {len(firm)} → PDB존재 {len(with_pdb)} "
          f"→ winner {len(winners)} (CAP {cap}, 초과탈락 {max(0,len(with_pdb)-cap)})", flush=True)
    if winners:
        prov = {}
        for r in winners:
            prov[r.get("provenance", "?")] = prov.get(r.get("provenance", "?"), 0) + 1
        print(f"  winner provenance: {prov}", flush=True)
        print(f"  best median {winners[0]['ddg_median']:.2f} ({winners[0]['sequence']}) "
              f"~ cutoff {winners[-1]['ddg_median']:.2f}", flush=True)
    print(f"  → {OUT}", flush=True)


if __name__ == "__main__":
    main()
