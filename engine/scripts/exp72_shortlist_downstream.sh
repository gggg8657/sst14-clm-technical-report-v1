#!/usr/bin/env bash
# shortlist 다운스트림 (세션독립): S1(재도킹) 완료 대기 → S2 선택성 off-target → S3 MM-GBSA
#   → S4 선택성 집계+테이블+전통공문서 보고서. 완료판정=레코드 카운트(프로세스 레이스 회피).
# 풀은 별도 병행. 로그: runs/exp72_analysis/downstream.log
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
BIO=/home/dongjukim/miniforge3/envs/bio-tools/bin/python
MMG=/home/dongjukim/miniforge3/envs/mmgbsa/bin/python
LOG=runs/exp72_analysis/downstream.log
PIN="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1"
say(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
lines(){ wc -l < "$1" 2>/dev/null || echo 0; }

say "DOWNSTREAM start — S1 완료 대기"
# S1 완료 = phase1_confirm.jsonl 17 레코드
while [ "$(lines runs/exp72_system/phase1_confirm.jsonl)" -lt 17 ]; do sleep 120; done
say "S1 완료 ($(lines runs/exp72_system/phase1_confirm.jsonl)) → S2 off-target"

# S2 off-target 선택성 (pose 저장, 16샤드) — 완료=offtarget_full.jsonl 17
rm -f runs/exp72_system/offtarget_full.jsonl; rm -rf runs/exp72_system/offtarget_poses; mkdir -p runs/exp72_system/offtarget_poses
for i in $(seq 0 15); do
  setsid env $PIN "$BIO" scripts/exp72_offtarget_full.py --shard-index "$i" --shard-count 16 --timeout 7200 \
    > "runs/exp72_system/offtarget_logs/ds_shard_${i}.log" 2>&1 < /dev/null & sleep 0.3
done
while [ "$(lines runs/exp72_system/offtarget_full.jsonl)" -lt 17 ]; do sleep 120; done
say "S2 완료 → S3 MM-GBSA"

# S3 MM-GBSA (on-target v3b + off-target)
rm -f runs/exp72_system/v3b_mmgbsa.jsonl runs/exp72_system/offtarget_mmgbsa.jsonl
$MMG scripts/exp72_v3b_mmgbsa.py >> "$LOG" 2>&1
$MMG scripts/exp72_offtarget_mmgbsa.py >> "$LOG" 2>&1
say "S3 완료 → S4 선택성집계+테이블+보고서"

# S4 선택성 집계(v3, nstruct2) → 완료=17 real Δmargin, 그다음 테이블+보고서
rm -f runs/exp72_system/v3_selectivity.jsonl
for i in $(seq 0 15); do
  setsid env $PIN FLEXPEP_NSTRUCT=2 "$BIO" scripts/exp72_v3_selectivity.py --shard-index "$i" --shard-count 16 --timeout 7200 \
    > "runs/exp72_system/v3_logs/ds_shard_${i}.log" 2>&1 < /dev/null & sleep 0.3
done
while [ "$($BIO -c "import json;print(sum(1 for l in open('runs/exp72_system/v3_selectivity.jsonl') if l.strip() and json.loads(l).get('delta_margin') is not None))" 2>/dev/null || echo 0)" -lt 17 ]; do sleep 120; done
say "S4 선택성 완료 → 테이블+보고서 생성"
$BIO scripts/exp72_shortlist_table.py >> "$LOG" 2>&1
$BIO scripts/exp72_report.py >> "$LOG" 2>&1
touch _workspace/EXP72_SHORTLIST_POSESAVE_DONE
say "DOWNSTREAM DONE — shortlist 완전본(pose저장+전통공문서 보고서) 완성"
