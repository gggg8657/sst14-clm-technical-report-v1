#!/usr/bin/env bash
# EXP72 후속 검증 watchdog — MM-GBSA + BB-flex 샤드 자동 재기동.
# 세션독립(setsid PPID=1). 완료마커 도달 시 자동 종료.
# 정지 = touch _workspace/STOP_EXP72_WATCHDOG
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
BIO=/home/dongjukim/miniforge3/envs/bio-tools/bin/python
MMG=/home/dongjukim/miniforge3/envs/mmgbsa/bin/python
LOG=runs/exp72_analysis/watchdog.log
PIN="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1"
INTERVAL=300  # 5분 간격

say(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

count_shards(){
  # /proc 정밀 매칭 (comm=python or bash)
  "$BIO" - "$1" "$2" <<'PY'
import os,glob,sys
pat,pref=sys.argv[1],sys.argv[2]
n=0
for d in glob.glob('/proc/[0-9]*'):
    try:
        cl=open(d+'/cmdline','rb').read().decode('utf8','ignore')
        if pat in cl and open(d+'/comm').read().startswith(pref): n+=1
    except: pass
print(n)
PY
}

say "WATCHDOG start (5분 간격)"

while true; do
  [ -f _workspace/STOP_EXP72_WATCHDOG ] && { say "STOP 요청 종료"; exit 0; }

  # ── MM-GBSA 8샤드 ──
  if [ ! -f _workspace/EXP72_POOL_MMGBSA_FULL_DONE ]; then
    N=$(count_shards "exp72_v3b_mmgbsa" "python")
    if [ "$N" -lt 8 ]; then
      say "MM-GBSA 샤드 $N<8 → 부족분 재기동"
      TARGET=8
      for i in $(seq 0 $((TARGET-1))); do
        RUNNING=$("$BIO" - "$i" <<'PY'
import os,glob,sys
i=sys.argv[1]
for d in glob.glob('/proc/[0-9]*'):
    try:
        cl=open(d+'/cmdline','rb').read().decode('utf8','ignore')
        if 'exp72_v3b_mmgbsa' in cl and f'--shard-index {i} ' in cl+' ':
            print('yes'); sys.exit()
    except: pass
print('no')
PY
)
        if [ "$RUNNING" = "no" ]; then
          setsid nice -n 5 env $PIN "$MMG" scripts/exp72_v3b_mmgbsa.py \
            --targets runs/exp72_analysis/pool_redock.jsonl \
            --out runs/exp72_analysis/pool_mmgbsa.jsonl \
            --shard-index "$i" --shard-count 8 \
            >> "runs/exp72_analysis/pool_mmgbsa_full_logs/shard_${i}.log" 2>&1 < /dev/null &
          say "  MM-GBSA shard $i 재기동"
        fi
      done
    fi
  fi

  # ── BB-flex 24샤드 ──
  if [ ! -f _workspace/EXP72_BBFLEX_POOL_DONE ]; then
    N=$(count_shards "exp72_bbflex_tier1" "python")
    if [ "$N" -lt 24 ]; then
      say "BB-flex 샤드 $N<24 → 부족분 재기동"
      TARGET=24
      for i in $(seq 0 $((TARGET-1))); do
        RUNNING=$("$BIO" - "$i" <<'PY'
import os,glob,sys
i=sys.argv[1]
for d in glob.glob('/proc/[0-9]*'):
    try:
        cl=open(d+'/cmdline','rb').read().decode('utf8','ignore')
        if 'exp72_bbflex_tier1' in cl and f'--shard-index {i} ' in cl+' ':
            print('yes'); sys.exit()
    except: pass
print('no')
PY
)
        if [ "$RUNNING" = "no" ]; then
          setsid nice -n 5 env $PIN FLEXPEP_NSTRUCT=3 "$BIO" scripts/exp72_bbflex_tier1.py \
            --source pool --out runs/exp72_analysis/bbflex_tier1.jsonl \
            --shard-index "$i" --shard-count 24 \
            >> "runs/exp72_analysis/bbflex_pool_logs/shard_${i}.log" 2>&1 < /dev/null &
          say "  BB-flex shard $i 재기동"
        fi
      done
    fi
  fi

  # ── 완료 판정 (두 마커 모두 도달) ──
  if [ -f _workspace/EXP72_POOL_MMGBSA_FULL_DONE ] && [ -f _workspace/EXP72_BBFLEX_POOL_DONE ]; then
    say "두 검증 완료 마커 확인 → watchdog 종료"
    exit 0
  fi

  sleep $INTERVAL
done
