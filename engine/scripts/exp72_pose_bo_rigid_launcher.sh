#!/usr/bin/env bash
# Rigid 6D BO 프로토 검증 통과 후 Tier 1 18종 자동 착수.
# 프로토 완료(10 iter) 대기 → best_f 검증 → Tier 1 8샤드 세션독립 기동.
# 정지 = touch _workspace/STOP_POSE_BO_TIER1
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
BIO=/home/dongjukim/miniforge3/envs/bio-tools/bin/python
LOG=runs/exp72_analysis/pose_bo_rigid_tier1.log
PROTO=runs/exp72_analysis/pose_bo_rigid_proto/AGCKNFFWKTFTSC_bo_history.jsonl
PIN="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1"
SHARDS=8
say(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

mkdir -p runs/exp72_analysis/pose_bo_rigid runs/exp72_analysis/pose_bo_rigid_logs
say "TIER1 (rigid) launcher start — 프로토(10 iter) 대기"

# 프로토 완료 대기
while true; do
  [ -f _workspace/STOP_POSE_BO_TIER1 ] && { say "STOP 요청 종료"; exit 0; }
  n=$(wc -l < "$PROTO" 2>/dev/null || echo 0)
  [ "$n" -ge 10 ] && break
  sleep 60
done
say "프로토 완료 ($n iter). best_f 검증"

BEST=$($BIO -c "
import json
best=None; err=0
for l in open('$PROTO'):
    if l.strip():
        d=json.loads(l)
        if d.get('f') is not None:
            if best is None or d['f']>best: best=d['f']
        if d.get('error') or d.get('error_flex'): err+=1
print(f'{best if best is not None else 0} {err}')")
BF=$(echo "$BEST" | awk '{print $1}')
ERR=$(echo "$BEST" | awk '{print $2}')
say "프로토 결과 best_f=$BF errors=$ERR/10"

# 정상성 판정: best_f > 0.15 (SS/FWKT만은 f=0.2, ddG 실측 없이는 통과 어려움 → 최소 ddG 유입 필요)
OK=$($BIO -c "print(1 if float('$BF')>0.15 and int('$ERR')<8 else 0)")
if [ "$OK" != "1" ]; then
  say "❌ 프로토 검증 실패 (best_f=$BF, err=$ERR) — Tier 1 착수 취소"
  touch _workspace/EXP72_POSE_BO_PROTO_FAIL
  exit 1
fi

say "✅ 프로토 통과 → Tier 1 18종 8샤드 착수 (rigid, MM-GBSA 포함)"
for i in $(seq 0 $((SHARDS-1))); do
  setsid nice -n 5 env $PIN "$BIO" scripts/exp72_pose_bo_rigid.py \
    --shard-index "$i" --shard-count "$SHARDS" \
    --n-iter 100 --n-initial 10 \
    --work-dir runs/exp72_analysis/pose_bo_rigid \
    > "runs/exp72_analysis/pose_bo_rigid_logs/shard_${i}.log" 2>&1 < /dev/null & sleep 0.5
done
say "Tier 1 $SHARDS 샤드 기동 완료"

while true; do
  [ -f _workspace/STOP_POSE_BO_TIER1 ] && { say "STOP 요청 종료"; exit 0; }
  n=$(wc -l < runs/exp72_analysis/pose_bo_rigid/bo_summary.jsonl 2>/dev/null || echo 0)
  [ "$n" -ge 18 ] && break
  sleep 600
done
say "TIER1 (rigid) DONE — $n / 18"
touch _workspace/EXP72_POSE_BO_TIER1_DONE
