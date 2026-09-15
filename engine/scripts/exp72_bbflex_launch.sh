#!/usr/bin/env bash
# Tier 1 18 후보 BB-flex 재도킹 - 세션독립 8샤드
# 정지 = touch _workspace/STOP_BBFLEX
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
BIO=/home/dongjukim/miniforge3/envs/bio-tools/bin/python
LOG=runs/exp72_analysis/bbflex_tier1.log
SHARDS=8
PIN="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1"
say(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
mkdir -p runs/exp72_analysis/bbflex_logs runs/exp72_analysis/bbflex_poses

say "BBFLEX start — Tier 1 18 nstruct 3 (backbone-flex lowres_preoptimize)"
for i in $(seq 0 $((SHARDS-1))); do
  setsid nice -n 5 env $PIN FLEXPEP_NSTRUCT=3 "$BIO" scripts/exp72_bbflex_tier1.py \
    --shard-index "$i" --shard-count "$SHARDS" \
    > "runs/exp72_analysis/bbflex_logs/shard_${i}.log" 2>&1 < /dev/null & sleep 0.3
done
say "BBFLEX 8 샤드 기동 완료"

# 완료 대기 = 18 records
while true; do
  [ -f _workspace/STOP_BBFLEX ] && { say "STOP 요청 종료"; exit 0; }
  DONE=$(wc -l < runs/exp72_analysis/bbflex_tier1.jsonl 2>/dev/null || echo 0)
  [ "$DONE" -ge 18 ] && break
  sleep 300
done
say "BBFLEX DONE — $DONE records"
touch _workspace/EXP72_BBFLEX_DONE
