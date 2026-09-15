#!/usr/bin/env bash
# Phase1 확증 재도킹 N샤드 병렬 (setsid 세션독립). 정지=STOP 파일 없음(완료시 자연종료).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
BIO=/home/dongjukim/miniforge3/envs/bio-tools/bin/python
N="${1:-8}"; NSTRUCT="${2:-20}"
mkdir -p runs/exp72_system/phase1_logs
echo "[phase1] N=$N shards, nstruct=$NSTRUCT"
for i in $(seq 0 $((N-1))); do
  setsid "$BIO" scripts/exp72_phase1_confirm.py --nstruct "$NSTRUCT" \
    --shard-index "$i" --shard-count "$N" \
    > "runs/exp72_system/phase1_logs/shard_${i}.log" 2>&1 < /dev/null &
  echo "  shard $i PID=$!"; sleep 1
done
echo "[phase1] 전 샤드 기동. 진행=tail runs/exp72_system/phase1_confirm.jsonl"
