# SST-14 · SSTR2 기술 보고서 (v1)

> 방사성 의약품용 SST-14 펩타이드 스크리닝 시스템의 파이프라인 17페이지 기술보고.
> **처음 오시는 분은 아래 "1분만에 보기"부터 읽어주세요.**

---

## 🌐 1분만에 보기 (설치 아무것도 필요 없음)

**아래 링크를 그냥 클릭하세요:**
👉 **https://gggg8657.github.io/sst14-clm-technical-report-v1/**

브라우저에서 웹사이트로 열립니다. 끝.
(GitHub Pages 활성화 직후엔 1-2분 정도 빌드 대기 시간이 있을 수 있어요.)

---

## 💻 내 컴퓨터에서 열어보기 (인터넷 없이 로컬)

### 준비물
- **Python 3.8 이상** — 컴퓨터에 이미 있을 확률 높음. 없으면 https://www.python.org/downloads/ 에서 설치 (설치 시 "Add Python to PATH" 체크박스 켜기).
- **Git** — https://git-scm.com/downloads

### 단계 (복붙 순서대로)

**Mac / Linux:**
```bash
git clone https://github.com/gggg8657/sst14-clm-technical-report-v1.git
cd sst14-clm-technical-report-v1
python3 -m http.server 8000
```

**Windows (PowerShell):**
```powershell
git clone https://github.com/gggg8657/sst14-clm-technical-report-v1.git
cd sst14-clm-technical-report-v1
python -m http.server 8000
```

브라우저 주소창에 **http://localhost:8000/** 입력하고 엔터. 사이트가 열립니다.

**끄고 싶을 때**: 터미널에서 `Ctrl+C`.

---

## 🛠️ 사이트를 직접 다시 만들기 (Python 지식 조금)

이 저장소는 마스터 소스를 포함하고 있어, 원본 마크다운을 편집한 뒤 HTML을 재생성할 수 있어요.

```bash
# (위 단계로 클론까지 마쳤다고 가정)
pip install -r requirements.txt   # markdown 패키지 설치
python SOURCES/master/build.py    # 재빌드
# 새 HTML이 SOURCES/master/에 생성되고, index.html이 갱신됨
```

원본 텍스트는 `SOURCES/master/_analysis/*.md`에 있습니다. 편집 후 위 명령 재실행.

---

## 🔬 엔진부터 다시 실험 재현 (전문가용, 하루+ 소요)

⚠️ **비전공자 안내**: 이 섹션은 도메인 지식·GPU·conda 환경 관리 경험이 필요합니다. **처음이시면 위 섹션까지만 하셔도 이 사이트의 모든 내용을 볼 수 있어요.**

시스템은 두 개의 발굴 루프로 데이터를 만듭니다:
- **Silo B**: 무한 탐색 루프 (PyRosetta 도킹 · Bayesian Optimization · Bandit · in-loop 선택성)
- **Silo A**: de novo 생성 (RFdiffusion + ProteinMPNN + ESMFold + FlexPepDock)

### 필요 환경
- Python 3.11+, conda(Anaconda/Miniconda)
- NVIDIA GPU (Silo A는 GPU 필수, Silo B는 CPU도 가능하나 도킹 매우 오래 걸림)
- **conda env 4개 이상** 별도 설치: `bio-tools` (PyRosetta), `pepadmet`, `esmfold`, `rfdiffusion`, `proteinmpnn` — 각 env 설치는 프로젝트별 README 참고

### 명령
```bash
conda env create -f engine/env/environment-bio-tools.yml
conda activate bio-tools

cd engine
# Silo B
python scripts/run_continuous_discovery.py \
  --input data/somatostatin_receptor/SSTR2_SST14_complex_boltz_1.pdb \
  --n-candidates 24 --max-iterations 20 --top-k 5

# Silo A (GPU)
CUDA_VISIBLE_DEVICES=0 python scripts/run_silo_a_discovery.py \
  --receptor-pdb data/somatostatin_receptor/curated/SSTR2_receptor.pdb \
  --contigs "B40-187/0 B189-327/0 12-16" --n-backbone 2 --k-seq 3 --diffusion-steps 50

# 정지 신호
touch _workspace/STOP_DISCOVERY
touch runs/silo_a_flow/STOP_SILO_A
```

---

## 이 사이트가 뭔가요? (10초 요약)

- **목적**: **SSTR2** 라는 수용체(암세포 표면에 많음)에 잘 붙는 **펩타이드 약물 후보**를 컴퓨터로 찾아내는 시스템의 보고서.
- **접근**: 자연 펩타이드(SST-14, 14개 아미노산)를 출발점으로 아미노산 하나씩 바꿔가며 도킹 점수·선택성·안정성을 평가.
- **결과 예**: F11D (11번째 위치를 Asp로 치환) — 도킹 ΔG −45.11 REU, 원본 대비 선택성 마진 +10.5 개선.
- **정직 원칙**: 모든 수치는 **상대 순위**로만 신뢰. 절대 예측은 wet-lab 검증 필수.

---

## 자주 발생하는 문제

| 증상 | 원인 | 해결 |
|---|---|---|
| `python3: command not found` | Python 미설치 | 위 "준비물" 참고해서 설치 |
| `port 8000 already in use` | 다른 프로그램이 그 포트 씀 | `python3 -m http.server 8001` 로 다른 번호 사용 |
| 페이지가 비어있거나 스타일 깨짐 | 파일 다운로드 미완 | `git clone`이 완료됐는지·`assets/` 폴더 있는지 확인 |
| `ModuleNotFoundError: markdown` | 재빌드 시 markdown 미설치 | `pip install markdown` |

---

## 파일 구조 (참고)
- `index.html` + `pages/` — 정적 사이트 (지금 보는 것)
- `assets/` — 이미지·스타일
- `SOURCES/master/` — 마스터 마크다운 소스
- `SOURCES/data/` — 데이터 스냅샷 (도킹 결과 JSON)
- `engine/` — 데이터를 만드는 엔진 코드 (Python 46MB)
- `scripts/scrub_sensitive.py` — 민감정보 마스킹 스크립트

## 정직한 한계
- 이 사이트의 수치들은 **상대 순위** 지표이며, 절대적인 약효 예측이 아닙니다.
- 실제 임상/실험 결정은 **wet-lab 검증**이 반드시 필요합니다.
- 자세한 결함/한계는 사이트 내부 "정직 원칙" 페이지 참고.

## 출처
`tmp/SST14-M_scr` (원본 read-only 복사, 2026-09-15). 민감정보(이메일·로컬경로) 스크럽 적용.
