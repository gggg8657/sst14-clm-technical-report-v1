# 기능 분석 보고서 — PyRosetta 도킹 엔진 (실측 ΔG)

분석 루트: `AgenticAI4SCIENCE_pyrosetta_track/repos/ai4sci-kaeri`
대상 파일:
- `AG_src/scripts/flexpep_dock.py` (실측 도킹·채점 핵심)
- `pyrosetta_flow/docking_executor.py` (subprocess 실행 레이어)
- `pyrosetta_flow/scoring_pipeline.py` (대안 스코어링 체인)
- `pyrosetta_flow/runner.py` (도킹 오케스트레이션·호출부)
- `pyrosetta_flow/schema.py` (FlowConfig 기본값)

작성 원칙: 모든 주장은 `file_path:line` 인용. 미확인 항목은 "미검증" 명시. 읽기전용 분석.

---

## ① 동작 원리

### (a) 전체 호출 경로

도킹은 별도 conda 환경(`bio-tools`)의 standalone 스크립트를 subprocess로 호출하는 구조다.

- `runner.py:299` 에서 도킹 스크립트 경로를 `AG_src/scripts/flexpep_dock.py` 로 고정.
- subprocess 실행은 `docking_executor.py:_run_script` (`docking_executor.py:38-75`) 가 담당하며, `runner.py:195` 에서 re-export 하여 하위호환을 유지한다.
- conda Python 경로 해석은 `_resolve_conda_python` (`docking_executor.py:18-35`): `~/miniforge3|miniconda3|anaconda3/envs/<env>/bin/python` 직접 경로 → 없으면 `conda run -n <env>` → 그래도 없으면 `sys.executable` 폴백 (`docking_executor.py:45-51`).
- 스크립트의 stdout 마지막 줄만 JSON 으로 파싱한다 (`docking_executor.py:65-75`). 모든 PyRosetta 로그는 stderr 로 분리 (`flexpep_dock.py:18-20`, `init_pyrosetta` 의 `-mute all` `flexpep_dock.py:38-45`).

### (b) FlexPepDock refine

- PyRosetta 초기화 옵션: `-mute all -ex1 -ex2aro -ignore_unrecognized_res -flexPepDocking:pep_refine -constraints:cst_fa_weight 1.0` (`flexpep_dock.py:38-45`).
- 변이체 입력은 **MutateResidue 방식**을 선호한다 (`flexpep_dock.py:52-151`): 레퍼런스 복합체(AlphaFold3 등)의 백본을 보존하고 side-chain만 target 서열로 치환. `runner.py:740-741` 에서 `--reference-complex <template> --target-sequence <mutant>` 인자로 전달.
  - 펩타이드 체인 자동 선택: 요청 체인이 없거나 길이가 target과 크게 다르면 길이 근접 체인으로 자동 전환 (`flexpep_dock.py:96-114`).
  - 서열 차이 위치만 1→3letter 코드(`_AA1TO3` `flexpep_dock.py:75-80`)로 변이 적용 (`flexpep_dock.py:136-150`).
- 체인 재정렬: FlexPepDockingProtocol은 펩타이드가 **마지막 체인**일 것을 요구하므로, PDB 텍스트를 dump→체인 블록 재배치→reload 한다 (`reorder_peptide_last` `flexpep_dock.py:158-236`). PyRosetta 버전 호환성을 위해 객체 조작 대신 텍스트 조작을 택했다고 명시 (`flexpep_dock.py:162-165`).
- 이황화결합 처리: 펩타이드(마지막 체인)에서 Cys 잔기 탐색(`_find_peptide_cys_residues` `flexpep_dock.py:243-250`). Cys가 정확히 2개면 `conformation().detect_disulfides()` 를 우선 사용하고, 실패 시 수동 `AtomPairConstraint`(SG-SG harmonic 2.05 Å, sd 0.3) 폴백 (`flexpep_dock.py:313-333`, `_add_disulfide_constraint` `flexpep_dock.py:253-272`). refine 후 SG-SG 거리(<3.0 Å면 INTACT)를 검사해 결과에 포함 (`_check_disulfide_distance` `flexpep_dock.py:275-290`, `flexpep_dock.py:338-341`).
  - 수동 constraint를 PDB 재정렬 직후 적용하면 AtomID 손상으로 segfault가 났던 이력이 주석에 기록됨 (`flexpep_dock.py:310-312`). 이 때문에 detect_disulfides 우선.
- 실제 refine 실행: `FlexPepDockingProtocol().apply(pose)` (`flexpep_dock.py:335-336`). ab-initio 변형은 `set_lowres_preoptimize(True)` 추가 (`flexpep_dock.py:369-371`).

### (c) InterfaceAnalyzer ddG

- `compute_interface_ddg` (`flexpep_dock.py:394-406`): `InterfaceAnalyzerMover(1)` (jump_id=1로 수용체 체인 A ↔ 펩타이드 체인 B 분리), `set_pack_input(True)`, `set_pack_separated(True)`, `apply` 후 `get_interface_dG()` 반환. 더 음수일수록 강한 결합.
- 총 에너지: `compute_total_score` = `get_fa_scorefxn()(pose)` (`flexpep_dock.py:409-413`).
- 결과 JSON: `ddg`, `total_score`, `pre_score`, `score_delta`, `clash_score`, `constraint_violations`(현재 항상 0, `flexpep_dock.py:551`), 그리고 Cys 2개일 때 `disulfide_intact`/`sg_sg_distance` (`flexpep_dock.py:545-558`).

### (d) clash 계산 (peptide-only)

- `compute_clash_score` (`flexpep_dock.py:416-446`): fa_scorefxn 적용 후, **마지막 체인(펩타이드)** 잔기만 순회하며 잔기별 `fa_rep > 10.0 REU` 개수를 센다 (`flexpep_dock.py:441-446`).
- 펩타이드 체인 범위는 `chain_begin/chain_end(n_chains)` 로 한정 (`flexpep_dock.py:435-439`). 단일 체인이면 전체 폴백.
- 주석(`flexpep_dock.py:422-426`)에 버그 수정 이력 기록: 이전 구현이 전체 pose(수용체+펩타이드 ~486잔기)를 세어 native·변이체 모두 clash≈50 으로 QC 게이트(≤10) 전원 탈락했던 문제를, 펩타이드 잔기만 세도록 수정함.

### (e) 대안 스코어링 체인 (보조)

`scoring_pipeline.py:_apply_alternative_scoring` (`scoring_pipeline.py:42-281`)은 FlexPepDock ddG에 부가 점수를 적층한다. 각 단계는 독립적으로 graceful skip:
- Step 0: cheap objectives(반감기/ADMET surrogate, 서열 기반) `scoring_pipeline.py:79-95`.
- Step 0.5: pepADMET GNN 독성 추론(별도 env 배치 subprocess) → admet 페널티. 미설치/실패 시 skip, 즉 **가짜 안전 판정을 만들지 않음(fail-closed)** `scoring_pipeline.py:99-120`.
- Step 1: GNINA rescore (binary 없으면 dry-run) `scoring_pipeline.py:123-157`.
- Step 2: ECR consensus(GNINA+ddG 통합) `scoring_pipeline.py:159-189`.
- Step 3: Pareto/NSGA-II 순위(pymoo) `scoring_pipeline.py:201-243`.
- Step 4: BO 추천(부수효과 없음, 로그만) `scoring_pipeline.py:249-279`.

이 모듈은 ddG 자체를 만들지 않는다 — FlexPepDock 결과(ddG·clash)를 입력으로 받아 순위만 보강한다.

---

## ② 영향 (실측 vs mock, fail-closed 999)

### 실측 ΔG — mock 경로 없음
- `flexpep_dock.py` 전체에 mock/fake/placeholder/stub 도킹 경로는 **없다**(grep으로 미발견). PyRosetta가 import되지 않으면 스크립트 자체가 예외로 실패한다 (`init_pyrosetta` `flexpep_dock.py:35-45`, 각 함수의 지연 import).
- 따라서 stdout JSON의 `ddg`는 항상 실제 `InterfaceAnalyzerMover.get_interface_dG()` 값이다 (`flexpep_dock.py:406, 539`). 실데이터 증거: `runs/pyrosetta_flow/global_selectivity_leaderboard.json` 에 `ddg: -32.1162` 등 실측값 다수 기록.
- "mock"이라는 용어는 보조 GNINA 단계의 dry-run 폴백에만 존재(`scoring_pipeline.py:7,54,123,147-148`)하며, 이는 ddG가 아니라 부가 CNN 점수의 부재를 honest하게 표기하는 용도다.

### fail-closed 999 규약
도킹 실패 시 가짜 좋은 점수 대신 999(=명백히 탈락)로 처리하는 것이 일관된 규약이다.
- 사전 게이트(예: FWKT pharmacophore) 실패 후보: `_dock_one` 진입 시 `fail_reason`이 있으면 도킹을 건너뛰고 `ddg=999.0, total_score=999.0, clash_score=999.0` 반환 (`runner.py:721-732`).
- subprocess 예외(타임아웃·크래시·JSON 파싱 실패 등): except 블록에서 동일하게 999로 마킹 (`runner.py:757-767`). `_run_script`가 returncode≠0/타임아웃/JSON 파싱 실패를 RuntimeError로 올린다 (`docking_executor.py:57-75`).
- baseline 도킹 실패: `trial_result.get("ddg", 999.0)` (`runner.py:385`).
- validation 트라이얼 실패: `return 999.0` (`runner.py:1342-1344`), 이후 `ddg < 900` / `ddg <= 0` 필터로 catastrophic 실패 제거 (`runner.py:1362, 1371, 1379`).
- 하류 소비: PASS 판정은 `c.ddg <= ddg_threshold and c.ddg < 900` (`runner.py:282`), Pareto 입력·BO 관측도 `ddg < 900` 유효성 필터 적용(`scoring_pipeline.py:252`, `runner.py:251-253`). 즉 999는 순위/통계에서 자동 배제된다.

영향 요약: 도킹이 죽으면 결과가 사라지는 게 아니라 "확실히 나쁜 후보"로 표식되어 자연 도태된다 → 환각성 좋은 점수가 leaderboard에 침투할 수 없는 구조.

---

## ③ 관련 Action Item

- 코드 주석에 직접 기록된 결함/수정 이력(사실상의 Action Item closure):
  - clash 전체-pose 집계 버그 수정(peptide-only로 전환) — `flexpep_dock.py:422-426`. (사용자 메모리 `sstr2-clash-gate-issue [FIXED]`와 일치.)
  - 이황화 수동 constraint segfault → detect_disulfides 우선 전환 — `flexpep_dock.py:310-312`.
  - C3(무한 hang 방지) timeout 도입 — `docking_executor.py:55-58`. script_timeout 300→600 상향(refine ~4min/후보) — `schema.py:35`.
  - C4(malformed stdout 방어) JSON 파싱 try/except — `docking_executor.py:68-75`.
  - P1 분해: god-object runner에서 도킹 subprocess 레이어를 `docking_executor.py`로 추출 — `runner.py:193-195`, `docking_executor.py:3`.
- 미확인(미검증): 별도의 구조화된 Action Item 레지스트리(예: `_workspace/release/`의 R1~R7, VR-cycle 목록)는 본 분석 grep 범위에서 발견하지 못함 → 본 엔진과의 1:1 매핑은 미검증. (CLAUDE.md Stage 이력에 R1~R7/VR-cycle-10~14 언급은 있으나 본 도킹 파일들과의 직접 연결 코드는 확인 못 함.)

---

## ④ 완성도

**완성도: 약 90%**

근거:
- 실측 ΔG 파이프라인 완비: MutateResidue→체인 재정렬→이황화 처리→FlexPepDock refine→InterfaceAnalyzer ddG→peptide-only clash 전 과정이 구현·연결됨 (`flexpep_dock.py:453-558`).
- 운영 견고성 확보: timeout(`docking_executor.py:55-58`), JSON 파싱 방어(`docking_executor.py:68-75`), fail-closed 999(`runner.py:721-767`), 병렬 실행(`runner.py:769-772`), best-of-N baseline(`runner.py:371-392`), 멀티-트라이얼 검증+CV 조기종료(`runner.py:1308-1395`), baseline 캐시(`runner.py:344-408`) 등.
- 실제 산출물 존재: leaderboard에 실측 ddg/margin/hc50 539+ 후보 기록.

감점 요인:
- `constraint_violations` 가 항상 0 하드코딩 — 이황화/거리 위반을 점수에 반영하지 않음 (`flexpep_dock.py:551`). disulfide는 INTACT 플래그만 보고하고 게이트화 여부는 본 파일에서 미확인.
- `ab-initio` 경로는 detect_disulfides 우선 적용이 아니라 수동 constraint만 사용 (`flexpep_dock.py:366-367`) — refine 경로의 segfault 회피 패턴이 ab-initio엔 미적용(잠재 리스크, 미검증).
- ddG 절대값의 실험 친화도 대비 검증(calibration) 코드는 본 파일군에 없음 → 순위용 지표로만 신뢰 가능, 절대 친화도 주장 불가.

---

## ⑤ 학술 가치

**중~상 (상에 근접)**

근거:
- 상 요소: 표준 FlexPepDock refine + InterfaceAnalyzer ddG는 펩타이드-수용체 결합 평가에서 학계 통용 프로토콜이며, MutateResidue 기반 백본 보존 변이(`flexpep_dock.py:52-151`)는 변이체 ΔΔG 비교에 방법론적으로 타당. 멀티-트라이얼+CV 조기종료(`runner.py:1366-1376`)와 top-3 mean 채택(`runner.py:1382, 1395`)은 PyRosetta 확률적 출력의 재현성 문제를 정직하게 다룬다.
- 중으로 끌어내리는 요소: ddG 절대값 calibration 부재, `constraint_violations` 미반영, disulfide 게이트 강제 여부 미확인. 단일 receptor 구조(template) 의존 — ensemble/induced-fit 정도는 본 파일에서 미검증.
- 정직성(높은 학술 가치): mock 없음 + fail-closed 999 + GNINA dry-run 명시 표기 → "계산 불가능을 계산 가능한 척하지 않는다" 원칙이 코드에 구현됨. 이는 결과 신뢰성 주장에 유리.

---

## ⑥ 사용법

### conda bio-tools subprocess 직접 호출
스크립트 헤더(`flexpep_dock.py:8-17`) 기준:
```
# 직접 복합체 입력
conda run -n bio-tools python AG_src/scripts/flexpep_dock.py \
    --input complex.pdb --output refined.pdb --protocol flexpep_refine

# 레퍼런스+변이 (변이체 권장)
conda run -n bio-tools python AG_src/scripts/flexpep_dock.py \
    --input complex.pdb --output refined.pdb --protocol flexpep_refine \
    --reference-complex ref.pdb --target-sequence SGCKNFFWKTFTCA --peptide-chain 1
```
- 인자: `--input`(필수), `--output`(필수), `--protocol`(flexpep_refine|flexpep_abinitio), `--reference-complex`, `--target-sequence`, `--peptide-chain`(기본 1) — `flexpep_dock.py:457-477`.
- conda env 기본값 `bio-tools` (`schema.py:21`). `_resolve_conda_python` 가 miniforge3/miniconda3/anaconda3 envs 경로를 자동 탐색 (`docking_executor.py:26-33`).
- 출력은 stdout 마지막 줄 JSON, 로그는 stderr (`flexpep_dock.py:18-20`).

### 파이프라인을 통한 병렬 호출 (max_workers)
- iteration 내 후보 도킹은 `ThreadPoolExecutor(max_workers=...)` 로 병렬 (`runner.py:769-772`).
- `max_workers = min(n_jobs, config.max_parallel_workers, os.cpu_count() or 4)` (`runner.py:699`). 기본 `max_parallel_workers=32` (`schema.py:31`).
- validation 트라이얼은 별도 풀 `min(remaining_trials, config.validation_max_workers, cpu_count)` (`runner.py:1347`), 기본 `validation_max_workers=4` (`schema.py:43`).
- subprocess 타임아웃 `script_timeout=600s` (`schema.py:35`). 각 후보는 독립 subprocess라 GIL 영향 없음(실제 연산은 PyRosetta C++).
- baseline은 `n_baseline_trials=3` best-of-N (`schema.py:36`, `runner.py:371-392`), `reuse_baseline=True` 시 epoch 간 캐시 재사용 (`runner.py:344-408`, `schema.py:54`).

### off-target 선택성 (관련 모듈)
- 선택성 도킹은 `AG_src/scripts/offtarget_dock.py` 가 SSTR1/3/4/5에 동일 FlexPepDock+InterfaceAnalyzer 적용 (`offtarget_dock.py:5-11`). 비용 큼 → `enable_selectivity=False` 기본 (`schema.py:45-51`).

---

## ⑦ 무관 시 필요 이유

본 엔진은 프로젝트 핵심과 직접 관련되어 "무관"에 해당하지 않는다. 근거: 프로젝트 목표가 SSTR2 표적 방사성의약품 후보의 결합 강도(ΔG) 기반 스크리닝(CLAUDE.md 프로젝트 컨텍스트)이며, 본 엔진이 그 ΔG를 산출하는 유일한 실측 채점원이다. leaderboard의 `ddg`/`margin`/`delta_margin`(`runs/pyrosetta_flow/global_selectivity_leaderboard.json`)이 모두 이 엔진 출력에 의존한다.

---

## 검증 인용 목록

- `pyrosetta_flow/docking_executor.py:18-35` — conda Python 경로 해석
- `pyrosetta_flow/docking_executor.py:38-75` — subprocess 실행/timeout/JSON 파싱 방어
- `pyrosetta_flow/scoring_pipeline.py:42-281` — 대안 스코어링 체인(graceful skip, ddg<900 필터)
- `pyrosetta_flow/scoring_pipeline.py:99-120` — pepADMET fail-closed(가짜 안전판정 금지)
- `AG_src/scripts/flexpep_dock.py:35-45` — PyRosetta init 옵션
- `AG_src/scripts/flexpep_dock.py:52-151` — MutateResidue 변이체 준비
- `AG_src/scripts/flexpep_dock.py:158-236` — 펩타이드 마지막 체인 재정렬
- `AG_src/scripts/flexpep_dock.py:243-341` — 이황화 탐지/constraint/거리 검사
- `AG_src/scripts/flexpep_dock.py:394-406` — InterfaceAnalyzer ddG
- `AG_src/scripts/flexpep_dock.py:416-446` — peptide-only clash 계산
- `AG_src/scripts/flexpep_dock.py:545-558` — 결과 JSON 스키마(constraint_violations=0 하드코딩)
- `pyrosetta_flow/runner.py:299` — flexpep_dock.py 경로 고정
- `pyrosetta_flow/runner.py:371-408` — baseline best-of-N + 캐시
- `pyrosetta_flow/runner.py:716-772` — _dock_one + ThreadPool 병렬, fail-closed 999
- `pyrosetta_flow/runner.py:1308-1395` — 멀티-트라이얼 검증 + CV 조기종료
- `pyrosetta_flow/schema.py:21,31,35,36,43,54` — conda_env/max_workers/timeout/baseline 기본값
- `runs/pyrosetta_flow/global_selectivity_leaderboard.json` — 실측 ddg 값 존재 증거

미검증 항목: ① 구조화된 Action Item 레지스트리와 본 엔진의 1:1 매핑 ② disulfide INTACT 플래그의 게이트 강제 여부 ③ ab-initio 경로 segfault 회피 적용 ④ ddG 절대값 실험 calibration.
