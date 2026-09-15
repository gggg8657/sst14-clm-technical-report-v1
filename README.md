# SST-14 · SSTR2 Technical Report (v1)

clm(자율 Claude Code 무한 발굴) 시스템의 파이프라인 구성요소 17페이지 기술보고 — 정적 리포트 사이트.

## 즉시 열람 (정적 서빙)
```bash
python3 -m http.server 8000
# http://localhost:8000/ 열기
```

## 파일 구조
- `index.html` + `pages/` (17 HTML) — 배포용 정적 사이트(즉시 서빙 가능)
- `assets/` — 이미지·CSS
- `SOURCES/master/` — 마스터 소스(재빌드용)
- `SOURCES/data/` — live 데이터 스냅샷(leaderboard·discovery_status·monitoring 등)
- `scripts/scrub_sensitive.py` — 민감정보 스크럽

## 재빌드
```bash
pip install -r requirements.txt
python SOURCES/master/build.py
# 주: build.py의 데이터 경로는 원본 clm 루트 가정 — SOURCES/data 심볼릭 링크 또는 경로 수정 필요.
```

## 안전
- 배포 전 `python3 scripts/scrub_sensitive.py . --dry-run` 로 재확인.
- 출처: tmp/SST14-M_scr (read-only 복사, 2026-09-15).
