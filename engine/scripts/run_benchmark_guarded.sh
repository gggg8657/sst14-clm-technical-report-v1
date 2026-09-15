#!/bin/bash
# 자가검증 세션독립 벤치마크 러너.
# 1) 스모크(8조건 nstruct=1) → 2) pocket_known clash 재발 없으면 → 3) 전체 실행.
# 스모크에서 pocket_known이 여전히 unphysical(clash)이면 전체 실행 안 함(쓰레기 방지).
# setsid nohup 으로 실행 → 세션 닫아도 완주. 결과 JSON은 autopush가 푸시.
set -u
REPO=/home/dongjukim/Documents/workspace/tmp/SST14-M_scr/AgenticAI4SCIENCE_pyrosetta_track/repos/ai4sci-kaeri
PY=/home/dongjukim/miniforge3/envs/bio-tools/bin/python
cd "$REPO" || exit 1

SMOKE_JSON=runs/pyrosetta_flow/dock_method_benchmark_smoke.json
FULL_JSON=runs/pyrosetta_flow/dock_method_benchmark.json
LOG=runs/pyrosetta_flow/benchmark_full.log
SMOKE_ERR=runs/pyrosetta_flow/benchmark_smoke.stderr.log

ts() { date +'%Y-%m-%dT%H:%M:%S%z'; }
echo "[$(ts)] === guarded benchmark 시작 ===" | tee -a "$LOG"

# ── 1) 스모크 ──
echo "[$(ts)] 스모크(8조건 nstruct=1) 실행..." | tee -a "$LOG"
$PY scripts/dock_method_benchmark.py --smoke --output-json "$SMOKE_JSON" > /dev/null 2> "$SMOKE_ERR"
SMOKE_RC=$?
echo "[$(ts)] 스모크 종료 rc=$SMOKE_RC" | tee -a "$LOG"

# ── 2) 게이트: pocket_known 조건이 clash(unphysical=True)면 중단 ──
BAD=$(grep -E "pocket_known#[0-9]+\] OK" "$SMOKE_ERR" | grep -c "unphysical=True")
LOCAL_BAD=$(grep -E "local_refine#[0-9]+\] OK" "$SMOKE_ERR" | grep -c "unphysical=True")
echo "[$(ts)] 스모크 게이트: pocket_known unphysical=$BAD, local_refine unphysical=$LOCAL_BAD" | tee -a "$LOG"
# pocket_known 결과 요약 로그
grep -E "(pocket_known|local_refine|blind)#[0-9]+\] OK" "$SMOKE_ERR" | tail -12 | tee -a "$LOG"

if [ "$BAD" -gt 0 ]; then
  echo "[$(ts)] ❌ 스모크 실패: pocket_known clash 재발($BAD건) — 전체 실행 중단(쓰레기 방지)." | tee -a "$LOG"
  exit 2
fi

# ── 3) 전체 실행 (2서열 × 8조건 × nstruct5 = 80 runs) ──
echo "[$(ts)] ✅ 스모크 통과 → 전체 80 runs 실행 시작(수 시간 소요)..." | tee -a "$LOG"
$PY scripts/dock_method_benchmark.py --output-json "$FULL_JSON" >> "$LOG" 2>&1
FULL_RC=$?
echo "[$(ts)] === 전체 종료 rc=$FULL_RC → $FULL_JSON ===" | tee -a "$LOG"
