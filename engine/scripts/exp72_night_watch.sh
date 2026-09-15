#!/usr/bin/env bash
# 야간 감시: 스코어보드+재도킹 완료 감지 → NIGHT_DONE.md 리포트(순수 읽기, 도킹 안함). setsid.
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
BIO=/home/dongjukim/miniforge3/envs/bio-tools/bin/python
OUT=runs/exp72_analysis/NIGHT_DONE.md
LOG=runs/exp72_analysis/night_watch.log
while true; do
  SB=$(pgrep -f '[b]uild_master_scoreboard.py'|wc -l)
  RD=$(ps -eo args|grep -c '[e]xp72_phase1_confirm.py')
  SBN=$(wc -l < runs/exp72_analysis/master_candidates.jsonl 2>/dev/null || echo 0)
  RDN=$(wc -l < runs/exp72_system/phase1_confirm.jsonl 2>/dev/null || echo 0)
  echo "$(date -u +%FT%TZ) scoreboard_proc=$SB rows=$SBN/33085 | redock_proc=$RD rows=$RDN/17" >> "$LOG"
  if [ "$SB" -eq 0 ] && [ "$RD" -eq 0 ]; then
    {
      echo "# 야간 검증 완료 — $(date -u +%FT%TZ)"; echo
      echo "- 마스터 스코어보드: **$SBN / 33085** 서열 (runs/exp72_analysis/master_candidates.jsonl)"
      echo "- Phase1 재도킹: **$RDN / 17** (runs/exp72_system/phase1_confirm.jsonl)"; echo
      echo "## Phase1 재도킹 결과 (nstruct20, robust+SS)"; echo '```'
      $BIO -c "
import json
for l in open('runs/exp72_system/phase1_confirm.jsonl'):
    r=json.loads(l)
    if r.get('status')=='ok':
        print(f\"  [{r.get('arm')}] {r['sequence']} med={r.get('ddg_median')} sd={r.get('ddg_sd')} n={r.get('n_converged')} SS={r.get('sg_sg_distance')}({r.get('disulfide_flag')})\")
" 2>/dev/null | sort
      echo '```'; echo
      echo "## 다음 단계 (복귀 후, 오케스트레이터 오버사이트 필요)"
      echo "- V2 구조검증 전수(shortlist) / V3 선택성(재정렬 off-target+native baseline 재산출)+MMGBSA / V4 통계·score-exploit·Pareto / V5 스코어보드 시각화"
    } > "$OUT"
    echo "$(date -u +%FT%TZ) DONE report written" >> "$LOG"
    break
  fi
  sleep 900
done
