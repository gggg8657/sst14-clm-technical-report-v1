#!/usr/bin/env bash
# 풀 winner 533에 MM-GBSA on-target 재채점 (세션독립). 서열에서 복합체 재구성 → OpenMM.
# 완료판정 = 레코드 카운트 >= winner+1(+native). 로그: runs/exp72_analysis/pool_mmgbsa.log
# 정지 = touch _workspace/STOP_POOL_MMGBSA
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
MMG=/home/dongjukim/miniforge3/envs/mmgbsa/bin/python
LOG=runs/exp72_analysis/pool_mmgbsa.log
OUT=runs/exp72_analysis/pool_mmgbsa.jsonl
WINNERS=_workspace/EXP72_POOL_WINNERS.json
SHARDS=6   # mmgbsa_daemon은 GPU 1개 사용, 샤드는 서열 dedup 병렬(GPU 자원경합 최소)
PIN="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1"

say(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

say "POOL MMGBSA start — winner $(wc -l < $WINNERS) + native"
mkdir -p runs/exp72_analysis/pool_mmgbsa_logs

# 샤드 병렬 기동 (setsid, GPU 공유)
for i in $(seq 0 $((SHARDS-1))); do
  setsid nice -n 5 env $PIN "$MMG" scripts/exp72_v3b_mmgbsa.py \
    --targets "$WINNERS" --out "$OUT" \
    --shard-index "$i" --shard-count "$SHARDS" \
    > "runs/exp72_analysis/pool_mmgbsa_logs/shard_${i}.log" 2>&1 < /dev/null & sleep 0.3
done

# 완료 대기 = winner + native = 534
TARGET=$(( $(wc -l < $WINNERS) + 1 ))
while true; do
  [ -f _workspace/STOP_POOL_MMGBSA ] && { say "STOP 요청 종료"; exit 0; }
  DONE=$($MMG -c "import json,os;f='$OUT';print(sum(1 for l in open(f) if l.strip() and json.loads(l).get('dg_bind') is not None)) if os.path.exists(f) else print(0)")
  [ "$DONE" -ge "$TARGET" ] && break
  sleep 120
done
say "POOL MMGBSA DONE — $DONE/$TARGET"
touch _workspace/EXP72_POOL_MMGBSA_DONE
