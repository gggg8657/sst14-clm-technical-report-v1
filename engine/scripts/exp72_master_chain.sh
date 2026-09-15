#!/usr/bin/env bash
# EXP72 마스터 체인 (세션독립): pose 전량 저장 코드로 shortlist 완전 재실행 → 풀 전체 재도킹.
# FLEXPEP_SAVE_ALL_POSES=1 로 모든 도킹의 개별 pose PDB 저장. setsid 로 기동 → 세션 꺼져도 완주.
# 로그: runs/exp72_analysis/master_chain.log
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
BIO=/home/dongjukim/miniforge3/envs/bio-tools/bin/python
MMG=/home/dongjukim/miniforge3/envs/mmgbsa/bin/python
LOG=runs/exp72_analysis/master_chain.log
PIN="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 FLEXPEP_SAVE_ALL_POSES=1"
say(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
wait_gone(){ while ps -eo comm,args|grep "[e]xp72_$1"|grep -q python; do sleep 120; done; }

say "MASTER CHAIN start (pose 전량 저장)"

# ── Stage S1: shortlist 재도킹 (nstruct20, pose 저장) ──
rm -f runs/exp72_system/phase1_confirm.jsonl
rm -rf runs/exp72_system/phase1_work; mkdir -p runs/exp72_system/phase1_work
say "S1 shortlist 재도킹 시작 (nstruct20, 17샤드, pose저장)"
for i in $(seq 0 16); do
  setsid env $PIN FLEXPEP_NSTRUCT=20 "$BIO" scripts/exp72_phase1_confirm.py --shard-index "$i" --shard-count 17 \
    > "runs/exp72_system/phase1_logs/re_shard_${i}.log" 2>&1 < /dev/null & sleep 0.3
done
wait_gone phase1_confirm.py
say "S1 완료 ($(wc -l < runs/exp72_system/phase1_confirm.jsonl 2>/dev/null) 후보, pose $(ls runs/exp72_system/phase1_work/*_pose*.pdb 2>/dev/null|wc -l)개)"

# ── Stage S2: shortlist 선택성 off-target (pose 저장) ──
rm -f runs/exp72_system/offtarget_full.jsonl; rm -rf runs/exp72_system/offtarget_poses; mkdir -p runs/exp72_system/offtarget_poses
say "S2 off-target 도킹 시작 (16샤드, timeout120분)"
for i in $(seq 0 15); do
  setsid env $PIN "$BIO" scripts/exp72_offtarget_full.py --shard-index "$i" --shard-count 16 --timeout 7200 \
    > "runs/exp72_system/offtarget_logs/mc_shard_${i}.log" 2>&1 < /dev/null & sleep 0.3
done
wait_gone offtarget_full.py
say "S2 완료 (성공 subtype $(grep -h 'SSTR[0-9]:' runs/exp72_system/offtarget_logs/mc_shard_*.log 2>/dev/null|grep -cvE failed))"

# ── Stage S3: shortlist MM-GBSA (on-target v3b + off-target Stage2) ──
say "S3 MM-GBSA 시작"
rm -f runs/exp72_system/v3b_mmgbsa.jsonl runs/exp72_system/offtarget_mmgbsa.jsonl
$MMG scripts/exp72_v3b_mmgbsa.py >> "$LOG" 2>&1
$MMG scripts/exp72_offtarget_mmgbsa.py >> "$LOG" 2>&1
say "S3 완료"

# ── Stage S4: shortlist 선택성 v3(집계 Δmargin) + 테이블 + 보고서 ──
say "S4 선택성 집계 + 테이블 + 보고서"
rm -f runs/exp72_system/v3_selectivity.jsonl
for i in $(seq 0 15); do
  setsid env $PIN FLEXPEP_NSTRUCT=2 "$BIO" scripts/exp72_v3_selectivity.py --shard-index "$i" --shard-count 16 --timeout 7200 \
    > "runs/exp72_system/v3_logs/mc_shard_${i}.log" 2>&1 < /dev/null & sleep 0.3
done
wait_gone v3_selectivity.py
$BIO scripts/exp72_shortlist_table.py >> "$LOG" 2>&1
$BIO scripts/exp72_report.py >> "$LOG" 2>&1
touch _workspace/EXP72_SHORTLIST_POSESAVE_DONE
say "S4 완료 — shortlist 완전 재실행+pose저장 끝. 마커 생성."

# ── Stage P1: 풀 전체 재도킹 (nstruct10, pose 저장, 120샤드) ──
rm -f runs/exp72_analysis/pool_redock.jsonl
rm -rf runs/exp72_analysis/pool_work; mkdir -p runs/exp72_analysis/pool_work
say "P1 풀 재도킹 시작 (2583, nstruct10, 120샤드, pose저장)"
for i in $(seq 0 119); do
  setsid env $PIN FLEXPEP_NSTRUCT=10 "$BIO" scripts/exp72_pool_redock.py --shard-index "$i" --shard-count 120 \
    > "runs/exp72_analysis/pool_logs/mc_shard_${i}.log" 2>&1 < /dev/null & sleep 0.1
done
say "P1 풀 재도킹 기동 완료 (120샤드). 이후 승자 선택성/MMGBSA는 별도 funnel."
touch _workspace/EXP72_MASTER_CHAIN_S_DONE
say "MASTER CHAIN: shortlist 완료 + 풀 재도킹 가동. (풀 완료는 별도 모니터)"
