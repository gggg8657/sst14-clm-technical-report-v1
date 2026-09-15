#!/usr/bin/env bash
# 풀 winner 선택성 funnel (세션독립): 풀 재도킹(2583) 완료 대기 → winner 선정
#   → v3_selectivity(off-target nstruct2, home-adv 보정은 보고단계) → 마커.
# 완료판정=레코드 카운트. 로그: runs/exp72_analysis/pool_funnel.log
# 정지=touch _workspace/STOP_POOL_FUNNEL
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
BIO=/home/dongjukim/miniforge3/envs/bio-tools/bin/python
LOG=runs/exp72_analysis/pool_funnel.log
PIN="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1"
SHARDS=24
say(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
lines(){ wc -l < "$1" 2>/dev/null || echo 0; }

say "POOL FUNNEL start — 풀 완료(2583) 대기"
# 풀 완료 = pool_redock.jsonl 2583 레코드
while [ "$(lines runs/exp72_analysis/pool_redock.jsonl)" -lt 2583 ]; do
  [ -f _workspace/STOP_POOL_FUNNEL ] && { say "STOP 요청 — 종료"; exit 0; }
  sleep 300
done
say "풀 완료 ($(lines runs/exp72_analysis/pool_redock.jsonl)) → winner 선정"

# winner 선정 (견고 winner 전량, 무단절단 없음)
$BIO scripts/exp72_pool_winners.py >> "$LOG" 2>&1
NW=$(lines _workspace/EXP72_POOL_WINNERS.json)
say "winner $NW 선정 → 선택성 도킹 ($SHARDS 샤드)"

# 선택성 (off-target nstruct2, 풀 winner 대상, pool_work 대표복합체)
rm -f runs/exp72_analysis/pool_selectivity.jsonl
for i in $(seq 0 $((SHARDS-1))); do
  setsid env $PIN FLEXPEP_NSTRUCT=2 "$BIO" scripts/exp72_v3_selectivity.py \
    --shard-index "$i" --shard-count "$SHARDS" --timeout 7200 \
    --targets _workspace/EXP72_POOL_WINNERS.json \
    --pdb-dir runs/exp72_analysis/pool_work \
    --out runs/exp72_analysis/pool_selectivity.jsonl \
    > "runs/exp72_analysis/pool_sel_logs/shard_${i}.log" 2>&1 < /dev/null & sleep 0.3
done

# 완료 대기 = winner 수만큼 delta_margin 산출(+native 1)
TARGET=$((NW+1))
while [ "$($BIO -c "import json;print(sum(1 for l in open('runs/exp72_analysis/pool_selectivity.jsonl') if l.strip() and json.loads(l).get('delta_margin') is not None))" 2>/dev/null || echo 0)" -lt "$TARGET" ]; do
  [ -f _workspace/STOP_POOL_FUNNEL ] && { say "STOP 요청 — 종료"; exit 0; }
  sleep 120
done
say "선택성 완료 → 마커 생성(보고서는 native 보정 후 별도 발행)"
touch _workspace/EXP72_POOL_FUNNEL_DONE
say "POOL FUNNEL DONE — winner $NW 선택성 완결"
