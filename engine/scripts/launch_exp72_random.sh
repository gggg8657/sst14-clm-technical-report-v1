#!/usr/bin/env bash
# 72h 랜덤 베이스라인 arm — N개 병렬 워커(서로 다른 seed)를 setsid 분리 기동.
# 각 워커는 공유 experiment_log.jsonl 에 O_APPEND(라인<4KB=원자적) 기록.
# 정지: touch _workspace/STOP_RANDOM_BASELINE  (전 워커 graceful 종료)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
BIO=/home/dongjukim/miniforge3/envs/bio-tools/bin/python
N="${1:-8}"                       # 병렬 워커 수
MAXSEC="${2:-259200}"             # 72h
NSTRUCT="${FLEXPEP_NSTRUCT:-5}"
OUT=runs/exp72_random
mkdir -p "$OUT/logs"
rm -f _workspace/STOP_RANDOM_BASELINE
echo "[exp72_random] N=$N workers, nstruct=$NSTRUCT, ${MAXSEC}s, out=$OUT"
for i in $(seq 0 $((N-1))); do
  SEED=$((20260708 + i))
  setsid "$BIO" scripts/random_baseline_dock.py \
    --out-dir "$OUT" --max-seconds "$MAXSEC" --nstruct "$NSTRUCT" --seed "$SEED" \
    --stop-file _workspace/STOP_RANDOM_BASELINE \
    > "$OUT/logs/worker_${i}.log" 2>&1 < /dev/null &
  echo "  worker $i seed=$SEED PID=$!"
  sleep 1
done
echo "[exp72_random] 전 워커 기동 완료. 정지: touch _workspace/STOP_RANDOM_BASELINE"
