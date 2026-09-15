#!/usr/bin/env bash
# Chrome 헤드리스 렌더 (GUI lib 경로 조립). 사용: _render.sh <html파일> <png출력> [높이]
set -e
HS=/home/dongjukim/.cache/puppeteer/chrome-headless-shell/linux-148.0.7778.97/chrome-headless-shell-linux64/chrome-headless-shell
LP="$HOME/miniforge3/envs/bio-tools/lib"
for d in "$HOME"/miniforge3/pkgs/atk-1.0*/lib "$HOME"/miniforge3/pkgs/*/lib; do
  [ -d "$d" ] && LP="$LP:$d"
done
H="${3:-2600}"
env LD_LIBRARY_PATH="$LP" HOME=/tmp "$HS" --headless --no-sandbox --disable-gpu \
  --no-first-run --disable-dev-shm-usage --user-data-dir=/tmp/cr_prof \
  --virtual-time-budget=9000 --screenshot="$2" --window-size=1440,"$H" \
  "file://$1" 2>/tmp/cr_err.log || { echo "FAIL"; tail -3 /tmp/cr_err.log; exit 1; }
