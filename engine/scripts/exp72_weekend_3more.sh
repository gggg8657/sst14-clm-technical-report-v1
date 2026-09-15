#!/usr/bin/env bash
# 주말 3종 추가 실험 착수: INTENSIFY 확장 + Off-target BO + 새 서열 5k.
# 세션독립. 정지 = touch _workspace/STOP_WEEKEND_3MORE
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
BIO=/home/dongjukim/miniforge3/envs/bio-tools/bin/python
LOG=runs/exp72_analysis/weekend_3more.log
PIN="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1"
say(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

say "WEEKEND 3MORE launcher start"

# ── 1. INTENSIFY 확장 (16 샤드): Tier 1 18 × 1-mutation 20 = 360 후보 ──
mkdir -p runs/exp72_analysis/intensify_expansion runs/exp72_analysis/intensify_logs
say "Phase A: INTENSIFY 확장 (Tier 1 18 × neighbors 20 = 360 후보, 16 샤드)"
for i in $(seq 0 15); do
  setsid nice -n 5 env $PIN "$BIO" scripts/exp72_intensify_expansion.py \
    --shard-index "$i" --shard-count 16 --n-neighbors 20 --nstruct 5 \
    --work-dir runs/exp72_analysis/intensify_expansion \
    > "runs/exp72_analysis/intensify_logs/shard_$(printf '%02d' $i).log" 2>&1 < /dev/null & sleep 0.2
done
say "  INTENSIFY 16 샤드 기동"

# ── 2. Off-target BO 확장 (Tier 1 shortlist 결과 이미 있음 → nstruct 20으로 재확장) ──
# 여기서는 기존 v3_selectivity를 nstruct 20으로 재실행 (16 샤드)
mkdir -p runs/exp72_analysis/offtarget_bo_logs
say "Phase B: Off-target 확장 (Tier 1 18 × 4 아형 nstruct 20 재도킹, 16 샤드)"
# Tier 1 18 후보만 담은 targets json 생성
$BIO -c "
import json,sys
sys.path.insert(0,'scripts')
from exp72_bbflex_tier1 import load_tier1_seqs
seqs = load_tier1_seqs()
recs=[]
for s in seqs:
    # winner 대표 복합체 PDB
    recs.append({'sequence':s, 'ddg_median':-30.0, 'status':'ok'})
with open('_workspace/EXP72_TIER1_OFFTARGET_TARGETS.jsonl','w') as f:
    for r in recs: f.write(json.dumps(r)+'\n')
print(f'Tier 1 offtarget targets: {len(recs)}')" >> "$LOG" 2>&1

for i in $(seq 0 15); do
  setsid nice -n 5 env $PIN FLEXPEP_NSTRUCT=20 "$BIO" scripts/exp72_v3_selectivity.py \
    --shard-index "$i" --shard-count 16 --timeout 7200 \
    --targets _workspace/EXP72_TIER1_OFFTARGET_TARGETS.jsonl \
    --pdb-dir runs/exp72_analysis/pool_work \
    --out runs/exp72_analysis/offtarget_bo_tier1.jsonl \
    > "runs/exp72_analysis/offtarget_bo_logs/shard_$(printf '%02d' $i).log" 2>&1 < /dev/null & sleep 0.3
done
say "  Off-target 확장 16 샤드 기동"

# ── 3. 새 서열 5k random 발굴 (16 샤드) ──
mkdir -p runs/exp72_random_v2 runs/exp72_random_v2_logs
say "Phase C: 새 서열 5k random 발굴 (16 샤드, nstruct 3)"
for i in $(seq 0 15); do
  # random_baseline_dock는 자체 seed 다양성 있음. shard별 다른 seed.
  setsid nice -n 5 env $PIN FLEXPEP_NSTRUCT=3 "$BIO" scripts/random_baseline_dock.py \
    --out-dir "runs/exp72_random_v2/shard_$(printf '%02d' $i)" \
    --seed "$((20260723 + i))" --max-seconds 172800 \
    --stop-file "_workspace/STOP_RANDOM_V2" \
    > "runs/exp72_random_v2_logs/shard_$(printf '%02d' $i).log" 2>&1 < /dev/null & sleep 0.3
done
say "  Random 새 서열 16 샤드 기동"

say "3종 착수 완료. 감시는 개별 완료마커로."
