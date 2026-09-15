#!/usr/bin/env bash
# 72h 실험 durable 하트비트 — 1시간마다 양 arm 상태 1줄 append. 세션독립(setsid).
# 정지: touch _workspace/STOP_EXP72_HEARTBEAT
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
HB="runs/exp72_random/heartbeat.log"
while true; do
  [ -f _workspace/STOP_EXP72_HEARTBEAT ] && { echo "$(date -u +%FT%TZ) STOP" >> "$HB"; break; }
  RW=$(ps -eo args | grep -c '[r]andom_baseline_dock.py.*exp72_random')
  RN=$(grep -c '"arm": "random"' runs/exp72_random/experiment_log.jsonl 2>/dev/null || echo 0)
  SY=$(ps -eo args | grep -c '[r]un_continuous_discovery.py')
  SL=$(( $(wc -l < runs/pyrosetta_flow/experiment_log.jsonl 2>/dev/null || echo 0) - 42280 ))
  WD=$(pgrep -xf 'bash scripts/weekend_watchdog.sh' | wc -l)
  LD=$(cut -d' ' -f1 /proc/loadavg)
  CVW=$(ps -eo args | grep -c '[c]v_pipeline.py')
  CVN=$(wc -l < runs/cv_analysis/cv_master.jsonl 2>/dev/null || echo 0)
  CVPDB=$(find runs/cv_analysis/work -maxdepth 3 -name '*.pdb' 2>/dev/null | wc -l)
  echo "$(date -u +%FT%TZ) random_workers=$RW random_docks=$RN sys_procs=$SY sys_window=$SL watchdog=$WD load=$LD cv_workers=$CVW cv_done=$CVN cv_pdbs=$CVPDB" >> "$HB"
  # wet-lab 후보 aggregator 자동 갱신(CV 진행분 반영, 세션 무관 durable)
  /home/dongjukim/miniforge3/envs/bio-tools/bin/python scripts/cv_aggregate_wetlab.py >> runs/cv_analysis/aggregate.log 2>&1 || true
  sleep 3600
done
