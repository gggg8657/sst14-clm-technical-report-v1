#!/usr/bin/env bash
# 72h 실험 상태 스냅샷 → CHECK_<ts>.md (세션 무관 durable 리포트). 자기매칭 회피 [r] 패턴.
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
TS=$(date -u +%FT%TZ)
OUT="runs/exp72_random/CHECK_${TS//:/-}.md"
RW=$(ps -eo args | grep -c '[r]andom_baseline_dock.py.*exp72_random')
RN=$(grep -c '"arm": "random"' runs/exp72_random/experiment_log.jsonl 2>/dev/null || echo 0)
SY=$(ps -eo args | grep -c '[r]un_continuous_discovery.py')
SL=$(( $(wc -l < runs/pyrosetta_flow/experiment_log.jsonl 2>/dev/null || echo 0) - 42280 ))
WD=$(pgrep -xf 'bash scripts/weekend_watchdog.sh' | wc -l)
LD=$(cut -d' ' -f1-3 /proc/loadavg)
EL=$(python3 -c "import datetime as d;s=d.datetime(2026,7,8,10,44,tzinfo=d.timezone.utc);print(round((d.datetime.now(d.timezone.utc)-s).total_seconds()/3600,1))")
{
  echo "# 72h 실험 24h 체크 — $TS"
  echo ""
  echo "- 경과: **${EL}h / 72h**"
  echo "- ① 랜덤 arm 워커: **$RW / 8** 생존 | 도킹 누계: **$RN**"
  echo "- ② 시스템 arm: 프로세스 $SY개 | window-new: $SL줄"
  echo "- watchdog: $WD개 | load: $LD"
  echo ""
  if [ "$RW" -lt 8 ]; then echo "⚠️ 랜덤 워커 $((8-RW))개 사망 (나머지는 지속). 재기동: \`./scripts/launch_exp72_random.sh $((8-RW))\`"; fi
  if [ "$SY" -lt 1 ]; then echo "⚠️ 시스템 arm 정지 — watchdog가 재기동해야 정상"; fi
  echo "## 하트비트 최근 12줄"
  echo '```'
  tail -12 runs/exp72_random/heartbeat.log 2>/dev/null
  echo '```'
} > "$OUT"
echo "$OUT"
