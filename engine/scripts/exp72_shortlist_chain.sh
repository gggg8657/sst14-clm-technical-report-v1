#!/usr/bin/env bash
# shortlist 심화 자율 체인 (세션독립): Stage1 완료 대기 → Stage2 MM-GBSA → 완전 테이블 빌드.
# setsid 로 기동하면 세션 꺼져도 끝까지 자동 완주. 로그=runs/exp72_system/chain.log
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
BIO=/home/dongjukim/miniforge3/envs/bio-tools/bin/python
MMG=/home/dongjukim/miniforge3/envs/mmgbsa/bin/python
LOG=runs/exp72_system/chain.log
say(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

say "CHAIN start — Stage1 완료 대기"
# 1) Stage1 (off-target 개별 도킹) 완료 대기
while ps -eo comm,args | grep '[e]xp72_offtarget_full' | grep -q python; do sleep 120; done
say "Stage1 완료 → Stage2 MM-GBSA 착수"

# 2) Stage2: off-target 포즈 MM-GBSA (정합성). mmgbsa env.
setsid env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$MMG" scripts/exp72_offtarget_mmgbsa.py >> "$LOG" 2>&1 < /dev/null
say "Stage2 완료 → 완전 테이블 빌드"

# 3) 완전 테이블
"$BIO" scripts/exp72_shortlist_table.py >> "$LOG" 2>&1
say "CHAIN DONE — _workspace/EXP72_SHORTLIST_FULL_TABLE.md 생성"
touch _workspace/EXP72_SHORTLIST_CHAIN_DONE
