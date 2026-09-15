# 다양성 관리 + 클러스터링 (A~E tier) — 기능 분석

> 분석 루트: `AgenticAI4SCIENCE_pyrosetta_track/repos/ai4sci-kaeri`
> 대상 파일:
> - `pyrosetta_flow/cluster_report.py` (A~E tier 분류, 결정론적)
> - `AG_src/agents/diversity_manager.py` (구조/서열 다양성 관리 에이전트)
> - `AG_src/tools/mcp/foldmason_server.py` (FoldMason CLI MCP 래퍼)
> - `AG_src/pipeline/step07_analysis.py` (FoldMason 실호출 경로)
> - 호출부: `pyrosetta_flow/runner.py`, `AG_src/pipeline/orchestrator.py`,
>   `run_pipeline_live.py`, `backend/routers/cluster.py`
> 모든 수치·동작은 소스 직독 인용. 미확인 항목은 "미검증" 명시.

**중요**: 본 분석은 서로 다른 두 시스템을 다룬다.
1. **A~E tier 분류** (`cluster_report.py`) — 약리/구조 수치 임계값 기반 결정론적 5분류.
2. **다양성 관리 클러스터링** (`diversity_manager.py`) — 유사 후보 그룹핑·중복 제거.
   이름은 둘 다 "cluster"이지만 **목적·알고리즘·tier 의미가 전혀 다르다** (③ 참조).

---

## ① 동작 원리

### 1.1 A~E tier 분류 (`cluster_report.py`)

후보 1건을 약리/구조 수치 임계값으로 **우선순위 A > B > C > D > E** 단일 배정한다
(`cluster_report.py:24-25`, `classify_cluster()` `:252-336`). 외부 의존 없는 순수 함수.

tier 정의 (`cluster_report.py:9-22` docstring + 실제 criteria 함수):

- **A — High Affinity Core** (`_criteria_a` `:159-188`):
  `ddG ≤ -8.0` AND `clash_score ≤ 5.0` AND `pLDDT ≥ 75` AND FWKT 접촉 유지.
  - pLDDT는 **옵션 처리**: None/0/NaN(ESMFold 미가동 PyRosetta-only 모드)이면
    해당 기준을 True로 스킵(`:177-179`) — 나머지 3기준으로 A 자격 부여.
  - FWKT 판정: `structural_rules.rules.fwkt_pharmacophore.pass` 불리언(`_fwkt_contact_maintained` `:70-87`).
- **B — Selectivity-Optimised** (`_criteria_b` `:191-206`):
  `selectivity_margin ≥ 3.0` AND `ddG < -5.0`("SSTR2 binding present"로 operationalize, `:201`).
- **C — Stability-Enhanced** (`_criteria_c` `:209-226`):
  `instability_index < 30` AND `BLOSUM62 total ≥ 0` AND `protease_sites ≤ 9`
  (SST-14 native baseline `_SST14_PROTEASE_BASELINE=9`, `:56`, `:220`).
- **D — Radiochemistry-Optimal** (`_criteria_d` `:229-246`):
  `GRAVY ∈ [-1.0, +0.5]` AND `|net_charge_ph74| ≤ 1.0` AND chelator site 가용.
  - chelator 판정(`_chelator_site_from_candidate` `:125-153`): **sequence 우선** —
    N-말단 non-Pro(α-NH₂) **또는** Lys 존재(ε-NH₂)이면 True. sequence 없을 때만
    `metal_coordination.n_strong ≥ 1` fallback.
  - 이는 P2-1 수정(`:133-141`): 기존 n_strong 로직이 SS-bond Cys thiol을 chelation
    부위로 과대평가하던 버그를 약리학 기준(Krenning 1992; de Jong 2002; Maecke 2005)으로 교체.
- **E — Exploratory** (`:325-336`): A~D 미충족 전부.

배치 분류 `batch_classify()` (`:339-381`): 후보별 분류 + tier별 분포 카운트/퍼센트
+ tier→식별자 그룹맵 반환. 식별자는 `name`→`sequence`→`candidate_{idx}` 순(`:360`).

각 criteria 함수는 **개별 불리언 dict**를 반환하고 `all(...)` 으로 통과 판정 — 즉
어느 항목이 왜 떨어졌는지 추적 가능(`criteria_met` 필드, `:277`,`:290` 등).

### 1.2 다양성 관리 (`diversity_manager.py` — `DiversityManagerAgent`)

QC 게이트 통과 후보가 충분히 다양한 구조 공간을 차지하는지 보장하고 유사 후보 중복을
제거한다(`:1-11`, `:72-96`). 통합 위치: QC `apply_gates()` 이후 → 최종 ranking 이전(`:10`,`:85`).

4개 공개 메서드:

- **`cluster_candidates()`** (`:122-157`): 후보를 유사도 기반 그룹핑.
  - `method="foldmason"` AND `tool_fn` 주입됨 → `_cluster_by_foldmason()` (lDDT 행렬 greedy).
  - `method="hybrid"` AND tool_fn → `_cluster_hybrid()`.
  - 그 외(또는 tool_fn 없음) → `_cluster_by_sequence()` 서열 greedy fallback,
    이때 경고 로그 "FoldMason tool 미등록 - 서열 기반 근사…"(`:151-152`).
- **`select_diverse_set(clusters, n)`** (`:159-208`): ①각 클러스터 대표 1개씩 선택 →
  ②부족하면 큰 클러스터 2번째 멤버 추가 → ③`final_score` 내림차순 반환.
- **`compute_diversity_score()`** (`:210-250`): 모든 후보에 lDDT 채워졌으면
  `diversity = 1 - mean_lDDT`(`:229-234`), 아니면 평균 서열동일성 기반
  `diversity = 1 - mean_identity`(`:236-250`). 1에 가까울수록 다양.
- **`flag_redundant()`** (`:252-291`): final_score 내림차순으로 보존, 서열동일성 ≥
  threshold(기본 0.8, `:100`)인 낮은 점수 쪽을 중복 표시.

greedy 클러스터링 공통 패턴(`_cluster_by_sequence` `:382-419`,
`_cluster_by_foldmason` `:421-490`): final_score 내림차순 정렬 → 고득점 seed가 대표 →
미배정 후보 중 유사도 ≥ threshold면 같은 클러스터에 흡수. 결과는 크기 내림차순 정렬(`:155`).

서열동일성은 **정렬 없는 위치별 단순 비교**(`_sequence_identity` `:507-521`):
짧은 쪽 기준 매칭 / 긴 쪽 길이 — 길이차 패널티 포함. (정렬 미수행은 한계, ⑤ 참조.)

### 1.3 FoldMason 구조 클러스터링 — 실제 호출 여부

**핵심 확인 결과**: FoldMason 바이너리 래퍼는 2곳에 존재하나, **`DiversityManager` 경로에서는
실호출되지 않는다**.

- `foldmason_server.py` (MCP 래퍼): `easy_msa`/`compute_lddt`/`refine_msa`/`cluster_structures`
  4툴을 `subprocess.run([foldmason, …])`로 **실제 CLI 호출**하도록 정의(`:47-69`, `:265-401`).
  단 docstring이 `_compute_lddt`/`_refine_msa`/`_cluster_structures`를
  "IMPLEMENTATION SKELETON"으로 명시(`:314`,`:342`,`:365`) — 실파싱은 미검증.
  파일 상단도 "This is an *interface definition*"(`:11`).
- `step07_analysis.py:run_foldmason_alignment()` (`:186-254`): `conda run -n <env>
  foldmason easy-msa … --alignment-type 3Di+AA`를 **실제 subprocess 호출**(`:212-223`).
  실패 시 placeholder lDDT(0.0/1.0)로 fallback(`:236-237`,`:247-254`). 이 결과는
  orchestrator에서 `Candidate.lddt`로 흘러들어감(`orchestrator.py:1096`).
- 그러나 `DiversityManagerAgent`의 `tool_fn`(=run_foldmason 콜러블)은 **어떤 호출부에서도
  주입되지 않는다**. 전 호출 지점:
  - `run_pipeline_live.py:1020` `DiversityManagerAgent(llm_provider=llm)` — tool_fn 없음.
  - `AG_src/pipeline/orchestrator.py:851` 동일 — tool_fn 없음.
  - `runs/run_live_demo.py:747` 동일 — tool_fn 없음.
  → 따라서 `method="foldmason"`을 지정해도(`run_pipeline_live.py:1021`) 항상
    `_cluster_by_sequence()` 서열 fallback으로 분기(`diversity_manager.py:145-153`).
- 추가로 환경에 `foldmason` 바이너리 자체가 **PATH에 없음**(본 분석 시점 `which foldmason` → NOT FOUND).
  `foldmason_server.py:44`는 미발견 시 문자열 `"foldmason"`로 폴백.

즉 **현재 운영 다양성 클러스터링은 100% 서열 기반**이며, 구조(3Di/lDDT) 클러스터링은
코드상 가능하나 비활성 상태다(미검증: foldmason 설치된 별도 환경 존재 여부).

### 1.4 파이프라인 통합 흐름

- **PyRosetta runner**(A~E tier): `runner.py:1163-1188` — pharma enrich 후
  `_batch_classify(cluster_input)` 호출, 결과를 candidate manifest `entry["cluster"]`에 부착.
  `_HAS_CLUSTER and _HAS_PHARMA` 동시 가용 시에만(`:1163`), 실패는 best-effort 흡수(`:1187-1188`).
- **AG 에이전트 파이프라인**(다양성): `orchestrator.py:626-644` — Step05b 선택성 이후
  `_invoke_agent("diversity_manager", {docking_results})` → `_adapt_agent_context`가
  `docking_results`→`candidates`로 변환(`:96-105`), `n_select`=config `diversity_top_n`(기본 20).
  결과 `diverse_candidates`→`selected_seq_ids` 역매핑(`:1045-1054`)으로 후보 축소.
- **백엔드 API**: `backend/routers/cluster.py:7-27` `POST /cluster/classify` —
  `batch_classify` 직접 노출, 모듈 부재 시 503.

---

## ② 영향

- **Local optimum 회피**: 다양성 관리가 유사 서열을 한 클러스터로 묶고 대표만 남겨
  (`select_diverse_set`), 실험 슬롯이 거의 동일한 후보로 채워지는 것을 방지한다.
  greedy seed가 final_score 최상위라 "좋으면서도 서로 다른" 집합을 지향.
  - 단 **구조 다양성이 아닌 서열 다양성**만 실효(③·1.3) → 동일 서열의 서로 다른
    backbone/conformer는 다양하다고 인식하지 못함. local optimum 회피력은 제한적.
- **후보 분류(triage)**: A~E tier가 후보를 약리/목적별로 묶어 다운스트림 우선순위와
  대시보드 표시(`runner.py:1190-1209` 이후 enrichment)에 사용. A=친화도, B=선택성,
  C=안정성, D=방사화학, E=탐색으로 의사결정자가 후보 성격을 즉시 파악.
- **중복 제거 효율**: `flag_redundant`가 동일/유사 후보 중 저점수 측을 제거해 도킹/정밀화
  비용 절감(`diversity_manager.py:252-291`).

---

## ③ 관련 Action Item

명시적으로 등록된 Action Item(`_workspace` 내 AI-/R번호 등)과 본 기능을 직접 연결하는
문서는 검색 결과 **발견되지 않음**(`_workspace/CONTINUOUS_DISCOVERY.md`에 divers/cluster/
foldmason 관련 Action Item grep → 결과 없음). 따라서 **직접 무관**으로 판정하되,
아래 코드 자체에서 드러난 갭(잠재 Action Item 후보)을 기록한다:

1. **(잠재) FoldMason tool_fn 미주입** — `method="foldmason"` 지정에도 항상 서열 fallback
   (1.3). 구조 클러스터링을 실제로 쓰려면 호출부에서 `tool_fn=run_foldmason_alignment`
   등을 주입해야 함. 현재 미연결.
2. **(잠재) 두 "cluster" 개념 혼동 위험** — `cluster_report`(A~E tier)와
   `diversity_manager`(구조 그룹)는 별개인데 이름·대시보드 라벨이 모두 "Cluster"
   (`backend/routers/pipelines.py:59` step07="Cluster", `agents.py:44` diversity=
   "foldmason 클러스터링"). 문서/네이밍 분리 권장.
3. **(잠재) foldmason_server skeleton** — `_compute_lddt`/`_cluster_structures` 파싱이
   skeleton 표기(1.3). 실 TSV 포맷 검증 필요.

---

## ④ 완성도 % + 근거

| 구성요소 | 완성도 | 근거 |
|---|---|---|
| A~E tier 분류 (`cluster_report.py`) | **~95%** | 5 criteria 전부 구현, P2-1 chelator 버그 수정 반영, 결정론적, runner·API·테스트(65개 test, `test_cluster_report.py`) 연결. 잔여: pLDDT 스킵·ddG operationalize 등 기준이 spec 근사. |
| 다양성 관리 — 서열 경로 | **~85%** | cluster/select/score/redundant 4메서드 + execute 완비, orchestrator·live 연결. 잔여: 서열동일성이 정렬 없는 위치 비교(`:511`)라 indel에 취약. |
| 다양성 관리 — FoldMason 구조 경로 | **~40%** | `_cluster_by_foldmason` 로직 존재하나 **tool_fn 미주입으로 비활성**, 바이너리 PATH 부재(1.3). 코드는 있으나 운영 미가동. |
| FoldMason MCP 래퍼 | **~60%** | easy_msa 실호출 구현, 그러나 lddt/cluster 핸들러 "skeleton" 표기(`foldmason_server.py:314`등), 출력 파싱 미검증. |
| step07 FoldMason 정렬 | **~80%** | 실 subprocess + 실패 fallback 구현(`step07_analysis.py:212-254`), lddt JSON 파싱은 sidecar 존재 가정. |

종합: **A~E tier는 운영급, 다양성 관리는 서열 모드 한정 운영급, 구조 클러스터링은 미가동.**

---

## ⑤ 학술 가치

- **A~E tier**: 방사성의약품 후보를 약리 목적(친화도/선택성/안정성/방사화학/탐색)으로
  분류하는 결정론적 triage는 다목적 최적화 후보 해석에 유용. chelator 판정의 약리학적
  교정(SS-bond Cys 제외, N-term/Lys 기반, `:142` 문헌 인용)은 방법론적으로 정직.
- **다양성 관리**: 구조 기반 MSA(FoldMason 3Di) 클러스터링 아이디어는
  저서열동일성 펩타이드 변이체에 적합한 선택. 단 현재 실측은 서열동일성 기반이라
  학술 주장 시 "구조 다양성" 표현은 과장 — **서열 다양성**으로 한정해야 함.
- **한계(정직성)**: 서열동일성이 정렬 없는 위치별 비교(`_sequence_identity` `:507-521`)
  → 길이 변이·삽입/결실이 있는 후보군에서 유사도 과소평가 가능. compute_diversity_score의
  lDDT 분기도 lDDT가 전 후보에 채워진 경우에만 작동(`:228-229`).

---

## ⑥ 사용법

A~E tier 분류 (라이브러리):
```python
from pyrosetta_flow.cluster_report import classify_cluster, batch_classify
res = classify_cluster({"sequence": "AGCKNFFWKTFTSC", "ddG": -9.0,
                        "clash_score": 3, "structural_rules": {...}, ...})
# res["cluster"] in {"A".."E"}, res["criteria_met"], res["note"]
batch = batch_classify([cand1, cand2, ...])  # batch["statistics"]["distribution"]
```

API:
```bash
curl -X POST .../cluster/classify -d '{"candidates": [ {...}, {...} ]}'
```

다양성 관리:
```python
from AG_src.agents.diversity_manager import DiversityManagerAgent
mgr = DiversityManagerAgent(similarity_threshold=0.8, clustering_method="foldmason",
                            tool_fn=run_foldmason)   # tool_fn 미주입 시 서열 fallback
out = mgr.execute({"candidates": cand_list, "n_select": 20, "method": "foldmason"})
# out["diverse_candidates"], out["clusters"], out["diversity_report"], out["redundant_ids"]
```

파이프라인에서는 별도 호출 불필요 — runner(A~E)·orchestrator(다양성)에 자동 통합.

---

## ⑦ 필요 이유

- **다양성 관리**: 무한 발굴 엔진은 유사 변이를 대량 생성하므로, 중복 후보가 실험·도킹
  슬롯을 잠식하면 탐색 효율이 급감한다. 대표 선택·중복 제거가 local optimum 고착을 늦추고
  제한된 wetlab 슬롯을 서로 다른 후보로 채운다.
- **A~E tier**: 단일 스칼라 점수만으로는 "왜 이 후보가 좋은가"를 설명 못 한다.
  목적별 tier가 의사결정자에게 후보 성격(고친화 vs 선택적 vs 안정 vs 방사화학)을
  즉시 제공해 다목적 trade-off 의사결정을 지원한다.

---

## 검증 인용 목록

- A~E tier 정의/우선순위: `cluster_report.py:9-25`, `:47-53`
- criteria A~D: `cluster_report.py:159-188`, `:191-206`, `:209-226`, `:229-246`
- pLDDT 옵션 스킵: `cluster_report.py:177-179`
- chelator P2-1 수정: `cluster_report.py:125-153`
- batch 통계: `cluster_report.py:339-381`
- DiversityManager 분기(foldmason/hybrid/sequence): `diversity_manager.py:145-153`
- tool_fn 미주입 시 경고: `diversity_manager.py:151-152`
- _cluster_by_foldmason / _cluster_by_sequence: `diversity_manager.py:421-490`, `:382-419`
- select_diverse_set / compute_diversity_score / flag_redundant:
  `diversity_manager.py:159-208`, `:210-250`, `:252-291`
- 서열동일성(정렬 없음): `diversity_manager.py:507-521`
- DiversityManager 생성(전부 tool_fn 없음): `run_pipeline_live.py:1020`,
  `orchestrator.py:851`, `runs/run_live_demo.py:747`
- live에서 method="foldmason" 지정: `run_pipeline_live.py:1021`
- FoldMason MCP 실 subprocess + skeleton 표기: `foldmason_server.py:47-69`, `:265-401`,
  `:11`, `:314`, `:342`, `:365`, `:44`
- step07 FoldMason 실호출/fallback: `step07_analysis.py:186-254`, `:212-223`, `:236-254`
- lddt → Candidate: `orchestrator.py:1096`
- runner A~E 통합: `runner.py:1162-1188`
- orchestrator 다양성 통합: `orchestrator.py:626-644`, `:96-105`, `:1045-1054`
- 백엔드 cluster API: `backend/routers/cluster.py:7-27`
- 환경 확인: `which foldmason` → NOT FOUND (본 분석 시점)
- 미검증: foldmason TSV/JSON 실파싱 결과, 별도 conda env의 foldmason 설치 여부,
  step07 실행 시간·실제 lDDT 값
