#!/usr/bin/env bash
# CV 파이프라인 백본 기동 — N 워커 setsid 세션독립(PPID=1).
# 각 워커는 hashlib.md5(seq)%N == worker_id 후보 담당(균형 분할).
# 정지: touch _workspace/STOP_CV_PIPELINE (모든 워커 graceful).
# 로그: runs/cv_analysis/logs/worker_N.log
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
BIO=/home/dongjukim/miniforge3/envs/bio-tools/bin/python
N="${1:-4}"
NRUNS="${2:-5}"
NSTRUCT="${FLEXPEP_NSTRUCT:-5}"
OUT=runs/cv_analysis
mkdir -p "$OUT/logs"
rm -f _workspace/STOP_CV_PIPELINE
TOTAL_POSES=$((NRUNS * NSTRUCT))
echo "[cv-pipeline] N=$N workers, n_runs=$NRUNS × nstruct=$NSTRUCT = $TOTAL_POSES PDB/candidate (25 poses 전부 저장)"
for i in $(seq 0 $((N-1))); do
  setsid "$BIO" scripts/cv_pipeline.py \
    --worker-id "$i" --n-workers "$N" --n-runs "$NRUNS" --nstruct "$NSTRUCT" \
    > "$OUT/logs/worker_${i}.log" 2>&1 < /dev/null &
  echo "  worker $i PID=$!"
  sleep 1
done
echo "[cv-pipeline] 기동 완료. 정지: touch _workspace/STOP_CV_PIPELINE"
