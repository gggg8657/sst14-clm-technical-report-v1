#!/usr/bin/env bash
# MM-GBSA 풀 전체(non-winner 포함, 2,552 unique) 재채점 — 세션독립.
# pool_redock.jsonl 의 status 필드는 v3b_mmgbsa 가 --targets 로 받아 처리.
# 기존 pool_mmgbsa.jsonl 에서 완료분은 스킵(재개 가능).
# 정지 = touch _workspace/STOP_POOL_MMGBSA_FULL
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
MMG=/home/dongjukim/miniforge3/envs/mmgbsa/bin/python
LOG=runs/exp72_analysis/pool_mmgbsa_full.log
OUT=runs/exp72_analysis/pool_mmgbsa.jsonl   # 기존 파일에 append (재개)
SRC=runs/exp72_analysis/pool_redock.jsonl
SHARDS=8
PIN="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1"
say(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

say "POOL MMGBSA FULL start (2,552 unique 대상, 기완료 스킵)"
mkdir -p runs/exp72_analysis/pool_mmgbsa_full_logs

for i in $(seq 0 $((SHARDS-1))); do
  setsid nice -n 5 env $PIN "$MMG" scripts/exp72_v3b_mmgbsa.py \
    --targets "$SRC" --out "$OUT" \
    --shard-index "$i" --shard-count "$SHARDS" \
    > "runs/exp72_analysis/pool_mmgbsa_full_logs/shard_${i}.log" 2>&1 < /dev/null & sleep 0.3
done

# 완료 대기 = pool_redock 고유 서열 수 도달
TARGET=$($MMG -c "import json;s=set(json.loads(l)['sequence'] for l in open('$SRC') if l.strip());print(len(s))")
while true; do
  [ -f _workspace/STOP_POOL_MMGBSA_FULL ] && { say "STOP 요청 종료"; exit 0; }
  DONE=$($MMG -c "import json;s=set(json.loads(l)['sequence'] for l in open('$OUT') if l.strip() and json.loads(l).get('dg_bind') is not None);print(len(s))")
  [ "$DONE" -ge "$TARGET" ] && break
  sleep 180
done
say "POOL MMGBSA FULL DONE — $DONE / $TARGET"
touch _workspace/EXP72_POOL_MMGBSA_FULL_DONE
