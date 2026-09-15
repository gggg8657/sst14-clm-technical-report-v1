#!/usr/bin/env bash
# 주말 자율 실험 오케스트레이터.
# 착수 순서: Random pose sampling (32 shards) → 리소스 여유 판단해 Winner BO / INTENSIFY / Off-target 추가
# 세션독립 (setsid PPID=1). 정지 = touch _workspace/STOP_WEEKEND
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
BIO=/home/dongjukim/miniforge3/envs/bio-tools/bin/python
LOG=runs/exp72_analysis/weekend.log
PIN="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1"
say(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

mkdir -p runs/exp72_analysis/lhs_pose_data runs/exp72_analysis/lhs_logs
say "WEEKEND launcher start"

# ── Random pose sampling (LHS): 32 샤드 ──
LHS_SHARDS=32
say "Phase 1: Random pose sampling (LHS) $LHS_SHARDS shards"
for i in $(seq 0 $((LHS_SHARDS-1))); do
  setsid nice -n 5 env $PIN "$BIO" scripts/exp72_random_pose_sampling.py \
    --shard-index "$i" --shard-count "$LHS_SHARDS" --n-pose 30 \
    --work-dir runs/exp72_analysis/lhs_pose_data \
    > "runs/exp72_analysis/lhs_logs/shard_${i:02d}.log" 2>&1 < /dev/null & sleep 0.3
done
say "LHS $LHS_SHARDS shards 기동"

# 5분 후 load 확인 후 다음 phase 착수
sleep 300
LOAD=$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)
say "5분 경과 load=$LOAD (target < 150)"

if [ "$LOAD" -lt 150 ]; then
  say "Phase 2: 여유 있음 → 향후 확장 실험 트리거 가능 (별도 launcher)"
fi

# 완료 감시 (LHS 16k pose 목표: 533 unique 서열 × 30 pose = 15,990)
TARGET=15990
say "완료 대기: $TARGET pose"
while true; do
  [ -f _workspace/STOP_WEEKEND ] && { say "STOP 요청 종료"; exit 0; }
  # shard별 jsonl 총합
  DONE=$($BIO -c "
import glob,json
n=0
for f in glob.glob('runs/exp72_analysis/lhs_pose_data/lhs_pose_shard_*.jsonl'):
    for l in open(f):
        if l.strip():
            try:
                d=json.loads(l)
                if d.get('status')=='ok': n+=1
            except: pass
print(n)")
  say "  진척 $DONE / $TARGET"
  [ "$DONE" -ge "$TARGET" ] && break
  sleep 3600
done
touch _workspace/EXP72_WEEKEND_LHS_DONE
say "WEEKEND LHS DONE — $DONE pose 생산 완료"
