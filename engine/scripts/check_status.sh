#!/usr/bin/env bash
# check_status.sh — 시스템 상태 한눈에 보기 (비전공자용)
#
# 사용법:  bash /home/dongjukim/Documents/workspace/tmp/SST14-M_scr/AgenticAI4SCIENCE_pyrosetta_track/repos/ai4sci-kaeri/scripts/check_status.sh
#
# 아무것도 바꾸지 않고 "읽기만" 한다. 몇 번을 실행해도 안전하다.

ROOT="/home/dongjukim/Documents/workspace/tmp/SST14-M_scr"
REPO="$ROOT/AgenticAI4SCIENCE_pyrosetta_track/repos/ai4sci-kaeri"

G=$'\033[32m'; R=$'\033[31m'; Y=$'\033[33m'; N=$'\033[0m'
ok(){ printf "  ${G}[정상]${N} %s\n" "$1"; }
bad(){ printf "  ${R}[문제]${N} %s\n" "$1"; }
warn(){ printf "  ${Y}[확인]${N} %s\n" "$1"; }

# 엔진 PID 찾기.
#   pgrep -f / pkill -f 는 "이 스크립트를 실행한 셸"의 명령줄까지 매칭해서
#   꺼져 있는 엔진을 "켜짐"으로 잘못 보고한다(자기매칭). 그래서 쓰지 않는다.
#   대신 (1) 실행 파일이 python 이고 (2) 부모가 init(PPID=1, 세션 독립 기동)인
#   프로세스만 고른다.
find_engine(){
  ps -eo pid=,ppid=,comm=,args= 2>/dev/null | awk -v pat="$1" '
    $2 == 1 && $3 ~ /^python/ && index($0, pat) > 0 { print $1; exit }'
}

echo "=============================================="
echo " SSTR2 시스템 상태 점검  ($(date '+%Y-%m-%d %H:%M'))"
echo "=============================================="

echo
echo "[1] 발굴 엔진"
pid_a=$(find_engine "run_silo_a_discovery.py")
if [ -n "$pid_a" ]; then
  secs=$(ps -o etimes= -p "$pid_a" 2>/dev/null | tr -d ' ')
  days=$(( ${secs:-0} / 86400 ))
  ok "Silo A 발굴 엔진 가동 중 (PID $pid_a, ${days}일째)"
else
  bad "Silo A 발굴 엔진이 꺼져 있음"
fi

pid_b=$(find_engine "run_continuous_discovery.py")
[ -z "$pid_b" ] && pid_b=$(find_engine "continuous.py")
if [ -n "$pid_b" ]; then
  ok "Silo B 발굴 엔진 가동 중 (PID $pid_b)"
else
  warn "Silo B 발굴 엔진 꺼짐 — 2026-07-14부터 의도적으로 정지한 상태이므로 정상"
fi

echo
echo "[2] AI 모델 서버 (LLM)"
llm_down=0
for port in 8000 8001 8002; do
  code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' "localhost:$port/v1/models" 2>/dev/null)
  if [ "$code" = "200" ]; then ok "포트 $port 응답함"
  else bad "포트 $port 응답 없음"; llm_down=$((llm_down+1)); fi
done
[ "$llm_down" -gt 0 ] && warn "AI 모델 서버가 꺼져 있으면 발굴 품질이 떨어짐 → 담당자 문의"

echo
echo "[3] 웹 화면"
if curl -s -m 3 -o /dev/null -w '' "localhost:8899/" 2>/dev/null; then
  ok "웹 서버 가동 중  →  http://localhost:8899/"
else
  bad "웹 서버 꺼짐. 켜는 법:  cd $ROOT/docs && python3 -m http.server 8899"
fi

echo
echo "[4] 최근 결과 갱신 시각"
lb="$REPO/runs/silo_a_flow/silo_a_leaderboard.json"
if [ -f "$lb" ]; then
  mtime=$(date -r "$lb" '+%Y-%m-%d %H:%M')
  hours=$(( ( $(date +%s) - $(date -r "$lb" +%s) ) / 3600 ))
  if [ "$hours" -lt 24 ]; then ok "후보 순위표 갱신됨 ($mtime, ${hours}시간 전)"
  else warn "후보 순위표가 ${hours}시간째 그대로 ($mtime) — 엔진 확인 필요"; fi
else
  bad "후보 순위표 파일 없음"
fi

echo
echo "[5] GPU 사용 현황"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | \
    while IFS=, read -r i used total; do printf "  GPU%s: %s / %s\n" "$i" "$(echo $used)" "$(echo $total)"; done
else
  warn "nvidia-smi 없음 (GPU 확인 불가)"
fi

echo
echo "=============================================="
echo " 문제가 있으면 이 화면을 그대로 복사해서 담당자에게 전달하세요."
echo " 담당자: https://github.com/gggg8657/peptide_screening_system"
echo "=============================================="
