# SST-14 · SSTR2 Technical Report (v1) — Reproducible Package

clm(자율 Claude Code 무한 발굴) 시스템의 파이프라인 구성요소 17페이지 기술보고 + **엔진 코드 포함**(데이터부터 재생성 가능).

## 3가지 사용 방식

### A. 즉시 열람 (정적 사이트, 준비된 결과)
```bash
python3 -m http.server 8000
# http://localhost:8000/
```

### B. 사이트 재빌드 (데이터 스냅샷 → HTML)
```bash
pip install -r requirements.txt
# engine/runs/pyrosetta_flow/ 에 이미 스냅샷 데이터 있음
python SOURCES/master/build.py
# 생성: SOURCES/master/index.html + pages/
```

### C. 엔진 처음부터 재실행 (데이터부터 재생성) — 시간·자원 큼
```bash
# 1) conda env 생성
conda env create -f engine/env/environment-bio-tools.yml
conda activate bio-tools

# 2) Silo B (무한 탐색 루프 — STOP_DISCOVERY 파일로 정지)
cd engine
python scripts/run_continuous_discovery.py \
  --input data/somatostatin_receptor/SSTR2_SST14_complex_boltz_1.pdb \
  --n-candidates 24 --max-iterations 20 --top-k 5

# 3) Silo A (de novo 병행) — GPU 필요
CUDA_VISIBLE_DEVICES=0 python scripts/run_silo_a_discovery.py \
  --receptor-pdb data/somatostatin_receptor/curated/SSTR2_receptor.pdb \
  --contigs "B40-187/0 B189-327/0 12-16" --n-backbone 2 --k-seq 3 --diffusion-steps 50

# 4) 결과 데이터가 engine/runs/pyrosetta_flow/ 에 생성됨 → B 로 재빌드
```

## 구조
- `index.html` + `pages/`(17) + `assets/` — 즉시 서빙(A)
- `SOURCES/master/` — 마스터 소스·build.py(B용)
- `SOURCES/data/` — live 데이터 스냅샷(B용)
- `engine/pyrosetta_flow/` (17 모듈) + `engine/AG_src/` — 엔진 코드(C용)
- `engine/scripts/` — 진입점(run_continuous_discovery.py 등)
- `engine/data/somatostatin_receptor/` — 시드 PDB(SST14 복합체·SSTR2 수용체)
- `engine/env/` — environment-bio-tools.yml, pyproject.toml, _setup_venv.sh
- `engine/runs/pyrosetta_flow/` — 데이터 스냅샷(B의 build.py가 참조하는 경로)
- `scripts/scrub_sensitive.py` — 민감정보 스크럽

## 환경·GPU 요구사항
- Python 3.11+, conda(bio-tools env: PyRosetta 포함)
- Silo A: RFdiffusion·ProteinMPNN·ESMFold GPU env(별도, 링크는 engine/env/README 참조)
- Silo B: bio-tools env로 실행

## 정직한 한계
- 엔진의 surrogate 정확도는 raw ρ≈0.13 (반감기), HC50 AUC 0.146(용혈 역전) — 상세: reports 참조.
- 모든 정량은 상대순위, wet-lab 검증 의무(H-06).
- 재현 시 GPU/env 없는 부분은 적절한 대체(예: Silo A만 CPU-only에선 생략) 필요.

## 출처
tmp/SST14-M_scr (read-only 복사, 2026-09-15). 민감정보 스크럽 적용됨.
