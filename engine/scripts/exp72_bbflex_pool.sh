#!/usr/bin/env bash
# 풀 전체 2,552 unique 후보 BB-flex 재도킹 — 24 샤드 세션독립
# 예상: nstruct 3 × ~10 min/pose × 2552 = 76,560 CPU min / 24샤드 = ~53h wall
# 정지 = touch _workspace/STOP_BBFLEX_POOL
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
BIO=/home/dongjukim/miniforge3/envs/bio-tools/bin/python
LOG=runs/exp72_analysis/bbflex_pool.log
OUT=runs/exp72_analysis/bbflex_tier1.jsonl   # 기존 파일 유지, 이어서 append (dedup은 스크립트 done set)
SHARDS=24
PIN="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1"
say(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
mkdir -p runs/exp72_analysis/bbflex_pool_logs runs/exp72_analysis/bbflex_poses

TARGET_N=$($BIO -c "
import json,os,sys; sys.path.insert(0,'scripts')
from exp72_bbflex_tier1 import load_pool_seqs
print(len(load_pool_seqs()))")
say "BBFLEX POOL start — 풀 전체 $TARGET_N 후보, nstruct 3, 24 샤드 (BB-flex lowres_preoptimize)"

for i in $(seq 0 $((SHARDS-1))); do
  setsid nice -n 5 env $PIN FLEXPEP_NSTRUCT=3 "$BIO" scripts/exp72_bbflex_tier1.py \
    --source pool --out "$OUT" \
    --shard-index "$i" --shard-count "$SHARDS" \
    > "runs/exp72_analysis/bbflex_pool_logs/shard_${i}.log" 2>&1 < /dev/null & sleep 0.3
done
say "BBFLEX POOL $SHARDS 샤드 기동 완료 (target $TARGET_N)"

# 완료 대기
while true; do
  [ -f _workspace/STOP_BBFLEX_POOL ] && { say "STOP 요청 종료"; exit 0; }
  DONE=$(wc -l < "$OUT" 2>/dev/null || echo 0)
  [ "$DONE" -ge "$TARGET_N" ] && break
  sleep 600
done
say "BBFLEX POOL DONE — $DONE / $TARGET_N"
touch _workspace/EXP72_BBFLEX_POOL_DONE
