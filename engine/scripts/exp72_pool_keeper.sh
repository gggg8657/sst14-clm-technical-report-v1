#!/usr/bin/env bash
# 풀 재도킹 self-heal: 5분마다 샤드<100이면 120 재기동(resume으로 완료분 skip). 정지=touch _workspace/STOP_POOL_KEEPER
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
BIO=/home/dongjukim/miniforge3/envs/bio-tools/bin/python
while true; do
  [ -f _workspace/STOP_POOL_KEEPER ] && exit 0
  N=$(ps -eo comm,args|grep '[e]xp72_pool_redock'|grep -c python)
  DONE=$(wc -l < runs/exp72_analysis/pool_redock.jsonl 2>/dev/null||echo 0)
  if [ "$DONE" -ge 2583 ]; then echo "$(date -u +%FT%TZ) POOL COMPLETE $DONE" >> runs/exp72_analysis/pool_keeper.log; exit 0; fi
  if [ "$N" -lt 100 ]; then
    echo "$(date -u +%FT%TZ) 샤드 $N<100 → 120 재기동(done=$DONE)" >> runs/exp72_analysis/pool_keeper.log
    for i in $(seq 0 119); do
      setsid env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 FLEXPEP_NSTRUCT=10 \
        FLEXPEP_SAVE_ALL_POSES=1 \
        "$BIO" scripts/exp72_pool_redock.py --shard-index "$i" --shard-count 120 \
        > "runs/exp72_analysis/pool_logs/shard_${i}.log" 2>&1 < /dev/null &
      sleep 0.1
    done
  fi
  sleep 300
done
