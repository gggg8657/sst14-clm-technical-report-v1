# 기능 분석 보고서 — QC 랭킹 · 검증 · Fail-Closed 원칙 · 약리학 가드

분석 루트: `AgenticAI4SCIENCE_pyrosetta_track/repos/ai4sci-kaeri`
원칙: 모든 주장은 `file_path:line` 인용. 미확인 항목은 "미검증"으로 명시. 본 분석은 읽기 전용.

---

## ⚠️ 핵심 사실 확인 — `pharmacology_guards.py` 실재 위치

CLAUDE.md는 `pipeline_local/scripts/pharmacology_guards.py` 를 명시하나, **분석 루트(ai4sci-kaeri repo) 안에는 존재하지 않는다.**

- 분석 루트 내 검색 결과 `pipeline_local/` 디렉토리 자체가 부재 (전체 트리 검색 0건).
- 실제 파일은 **상위 모노레포 루트**에 존재: `[LOCAL_PATH]` (1248줄).
- 분석 루트 코드는 이 모듈을 **문자열 docstring 참조로만** 언급하고, 런타임 import는 하지 않는다.
  - `backend/pharmacophore.py:13`, `:102`, `:180` — docstring 안내 문구
  - `backend/routers/status.py:75` — docstring 안내 문구
  - `attach_confidence` / `HEURISTIC_FUNCTION_DISCLAIMERS` 를 실제 import해 호출하는 런타임 코드는 분석 루트에서 **0건** (`grep` 확인). 즉 가드의 confidence 주입은 분석 루트 API 응답에 **현재 연결되어 있지 않다(미검증→사실상 미연결)**.
- 테스트 `backend/tests/test_pharmacophore.py:366` 은 `from pipeline_local.scripts.pharmacology_guards import HEURISTIC_FUNCTION_DISCLAIMERS` 를 수행하며, pytest rootdir이 상위 모노레포(`configfile: pyproject.toml`, rootdir=`SST14-M_scr`)이므로 sys.path를 통해 import가 성공한다. 4개 등록 테스트 모두 PASS 확인.

**결론**: 약리학 가드는 *모노레포 레벨 공유 자산*이며 분석 루트 repo는 이를 **테스트로 검증만** 하고 런타임 적용은 아직 미배선이다.

---

## ① 동작 원리

### (A) QC 게이트 + 랭킹 — `AG_src/agents/qc_ranker.py`

`QCRankerAgent`는 ESMFold pLDDT, FoldMason lDDT, 도킹 점수, Rosetta ΔΔG, 선택성을 하나의 랭킹 테이블로 통합한다 (`qc_ranker.py:132-149`).

게이트는 **순차 AND 조건**으로 적용된다 (`apply_gates`, `qc_ranker.py:171-287`):

- **Gate 1 — pLDDT**: `plddt_mean >= 75` AND `plddt_interface >= 70` (`qc_ranker.py:203-212`)
- **Gate 2 — Docking**: dock_score 기준 상위 `docking_top_pct%`(기본 20%)만 통과 (`qc_ranker.py:218-230`)
- **Gate 3 — Rosetta**: `ddg <= rosetta_ddg_max(-5.0)` AND `clash <= 10` AND `constraint_violations <= 0` (`qc_ranker.py:236-248`)
- **Gate 4 — Selectivity**: `selectivity_margin >= sel_margin_min`, off-target 한계 검사 (`qc_ranker.py:254-273`)

각 게이트는 `gates_enabled` 플래그로 비활성화 가능하며, 비활성 시 "DISABLED - 전체 통과" 로그를 남긴다 (`qc_ranker.py:213-215, 231-233, 249-251, 274-276`). 실패 사유는 사람이 읽을 수 있는 문자열로 누적된다 (예: `qc_ranker.py:208`, `:242`).

랭킹은 min-max 정규화 후 가중 합산 (`compute_rankings`, `qc_ranker.py:289-354`). dock_score·ddg는 "낮을수록 좋음"이므로 `invert=True`로 부호 반전 (`qc_ranker.py:332-333`). 가중치는 `DEFAULT_WEIGHTS`(`qc_ranker.py:109-115`) 또는 PyRosetta 전용 모드 `PYROSETTA_ONLY_WEIGHTS`(ddg 0.70 비중, `qc_ranker.py:117-125`). 정규화 분모가 0이면(모두 동일값) 1.0으로 처리해 0-division을 회피 (`qc_ranker.py:320-321`).

랭킹 모드는 `ddg_primary`(ddG 오름차순 top-k, `qc_ranker.py:370-387, 523-525`) 또는 `weighted`(가중합 top-k) 두 가지 (`qc_ranker.py:526-529`).

### (B) 다중 trial 검증 (통계적 재현성) — `backend/validation.py`

사용자가 명시적으로 선택한 후보만 검증하며, 전체 후보를 자동 검증하지 않는다 (`backend/validation.py:6-8`). 세 가지 통계 검증을 수행한다:

- **Rank Stability** (`validate_rank_stability`, `backend/validation.py:53-125`): 동일 서열이 여러 (run_id, iteration) 그룹에서 차지한 순위의 표준편차로 안정성 측정. `confidence = max(0, 1 - rank_stdev/top_k)`, `>= 0.7`이면 stable (`:103-105`). 출현 횟수가 `min_appearances`(기본 2) 미만이면 "Insufficient data"로 confidence 0.0 (`:87-101`).
- **Score Consistency** (`validate_score_consistency`, `:132-223`): ddG의 변동계수(CV) 기반. CV가 낮을수록 confidence 높음 (`:198`). 2σ 초과 이상치(anomalous run)를 별도 식별 (`:185-196`). 평가 2건 미만이면 confidence 0.0 (`:147-165`).
- **No-Dominance** (`validate_no_dominance`, `:230-346`): 후보의 우위가 산물(artifact)인지 점검. **median ddG**(best 아님 — 더 robust, `:243-253`)로 집계, top-k gap_ratio > 2.0이면 anomalous dominance로 판정 (`:280, 311`). 재현성(consistently_top + 평가 ≥2)이 있으면 legitimate (`:312`).

세 결과를 가중 통합 (`_WEIGHTS = {rank 0.4, score 0.4, dominance 0.2}`, `:353`) → `RELIABLE / CAUTION / UNRELIABLE` 판정 (`validate_candidate`, `:373-378`).

### (C) Fail-Closed 패턴 (실패 → 999 / NaN → 자동 탈락)

데이터 부재·실패 시 **나쁜 쪽 기본값**을 주입해 안전하게 탈락시키는 패턴이 광범위하게 사용된다 (ddG는 "낮을수록 좋음"이므로 999 = 자동 실패):

- 검증 라우터: `ddg = float(cand.get("ddG", 999.0))`, `clash = ...999.0`, 규칙 통과 실패 시 fail (`backend/routers/validation.py:53-66`).
- 성공 판정 게이트: `_is_success = status=="success" AND ddg < 900` → 999 기본값은 자동 불통과 (`backend/validation.py:32-33`, `backend/analysis.py:45`, `pyrosetta_flow/ranking.py:50`).
- Plausibility 필터(이상치 차단): ddG가 `[-60, 200]` 밖이면 ranking 전 제거 (`backend/validation.py:27-28, 36-38`) — devil's advocate 발견에 따른 hardening (`:10-13`).
- 랭킹/집계의 999 기본값: `pyrosetta_flow/ranking.py:52, 71`, `pyrosetta_flow/scoring_pipeline.py:227,234`(pareto_rank 999), `backend/status_emitter.py:220`, `backend/routers/status.py:203`.
- gnina rescoring 실패 시 `float("inf")` 주입 (`pyrosetta_flow/gnina_rescoring.py:375`).
- status 라우터 enrichment: 각 필드 계산을 `try/except`로 감싸 실패 시 해당 필드를 비워두고 다른 후보 처리를 막지 않음 (graceful degradation, `backend/routers/status.py:111-150`). 단, 이는 fail-closed가 아니라 fail-open(필드 누락)에 가까움 — 점수 자체는 999 기본값으로 보호.

### (D) 약리학 가드 (모노레포 자산) — `pipeline_local/scripts/pharmacology_guards.py`

stand-alone 선언적 가드. 외부 패키지 import 없이 문헌 정답과 대조만 수행 (`pharmacology_guards.py:18-19`). 도메인 환각 시나리오 H-01~H-06을 각각 방어 (`:9-16`):

- **`LITERATURE_VALUES`** (`:56-227`): Kyte-Doolittle, Radzicka-Wolfenden(Boman convention), Varshavsky 반감기(종 명시), Lehninger pKa, Ikai 계수, SST14-SSTR2 ref ΔG(Boltz2 −95.024 REU / FlexPepDock 553.857±4.024 REU), modification_conflict_rules(C-01~C-99, 각 화학 근거 문헌 포함)을 **문헌 직접 인용**으로 보관 (`:53`).
- **`SCALE_RANGES`** (`:234-246`): 각 척도의 생물학적 합리 범위 (GATE-C).
- **`SIGN_CONVENTIONS`** (`:613-621`): 부호 규약 (H-02, 예: rosetta_total_score NEGATIVE=favorable).
- **`HEURISTIC_FUNCTION_DISCLAIMERS`** (`:253-476`): H-06 가드. 휴리스틱 함수의 표면 단위 vs 실제 의미, 한계, 유효/무효 사용, confidence_grade="HEURISTIC"를 강제. `backend.pharmacophore.compute_fwkt_contact`(`:326-339`), `compute_chelator_site`(`:340-353`) 포함.
- **검증 함수**: `assert_literature_value`(`:640-676`), `audit_table`(`:679-720`), `assert_in_range`(`:723-742`), `check_sign_convention`(`:745-772`), `is_heuristic_function`(`:479-488`).
- **`ENDPOINT_CONFIDENCE` + `attach_confidence`** (`:790-1248`): API 엔드포인트별 신뢰 등급(A/B/C/HEURISTIC)과 경고를 응답 dict에 주입하는 헬퍼. **값 자체는 변경하지 않고 메타데이터만 추가** (`:1211-1214`). *단, 분석 루트 라우터에서 호출되지 않음 (미배선).*

---

## ② 영향 (가짜 점수 차단 = 신뢰성)

- **가짜 점수 자동 탈락**: 실패/데이터 부재가 999로 매핑되어 게이트·랭킹에서 자동 하위로 밀린다. "조용한 성공(silent pass)" 위험을 제거 (`backend/routers/validation.py:53-59`, `backend/validation.py:32-33`).
- **이상치 차단**: plausibility 필터가 비현실적 ddG(예: 산물 스파이크)를 ranking 전에 제거해 false dominance를 방지 (`backend/validation.py:36-38, 230-346`).
- **재현성 정량화**: 단일 trial 운(luck)이 아니라 다중 seed/iteration 표준편차·CV로 confidence를 매겨, 재현 불가능한 best 후보를 CAUTION/UNRELIABLE로 강등 (`backend/validation.py:373-378`).
- **문헌 환각 차단**: lookup table이 코드 현재값이 아닌 문헌 인용과 대조되므로, 무단 변경·부호 역전·척도 혼용이 회귀 테스트에서 즉시 드러난다 (`pharmacology_guards.py:53-54`).
- **휴리스틱의 정직한 노출**: 표면 단위(예: "hours")가 임상 의미가 아님을 명시해, 순위용 점수를 절대값으로 오인하는 것을 방지 (`pharmacology_guards.py:253-256`).

---

## ③ 관련 Action Item

- **Stage 5 — pharmacology guards**: CLAUDE.md Stage 적용 이력에 "2026-05-11 Stage 5 — `pipeline_local/scripts/pharmacology_guards.py` + 33 회귀 테스트(현재 39개)" 등록됨. 본 분석 시점 실제 회귀 테스트 파일 `pipeline_local/tests/test_pharmacology_guards.py` 존재 확인. 가드 자체는 구현 완료 상태.
- **모노레포-repo 경계 갭 (신규 식별)**: 분석 루트 코드는 가드를 docstring으로만 참조하고 `attach_confidence`를 런타임 배선하지 않음. CLAUDE.md "디렉토리 침범 금지" 원칙과 충돌 없이 가드를 분석 루트 API에 연결하는 작업이 미완 (미검증→미배선 확정).
- **CI Q3-1**: 분석 루트 트리(`CLAUDE.md`, `INTEGRATION_PLAN.md`) 내에서 "Q3-1" 식별자 직접 인용을 **찾지 못함 → 미검증**. (분기 회고 `_workspace/release/retro-2026-Q2.md`에 Q3 Action Items가 식별되어 있다고 CLAUDE.md가 언급하나, 그 파일 내 "Q3-1" 명칭의 CI 항목은 본 분석에서 미확인.)
- VR-cycle-09 (H-06) closure가 `HEURISTIC_FUNCTION_DISCLAIMERS`로 운영화됨 (`pharmacology_guards.py:15-16, 254`).

---

## ④ 완성도 (%) + 근거

| 구성요소 | 완성도 | 근거 |
|---|---|---|
| QC 게이트 + 랭킹 (`qc_ranker.py`) | **90%** | 4개 게이트 + 2개 랭킹 모드 + CSV 출력 + QC 보고서 모두 구현(`qc_ranker.py:171-536`). pyc 캐시 3종(311/312/313) = 다환경 실행 흔적. selectivity 게이트의 `!= 0.0` 가드(`:260`)는 0=미측정과 0=실제값을 구분 못하는 미세 한계. |
| 다중 trial 검증 (`validation.py`) | **90%** | 3개 검증 + 가중 통합 + plausibility 완비(`backend/validation.py:53-427`). 라우터 노출(`backend/routers/validation.py:132-157`) + executor 타임아웃 보호(`:101-107`). |
| Fail-closed 패턴 | **85%** | 999/inf/plausibility 광범위 적용(다수 파일). 일관성은 높으나 999/900/inf/-999가 파일별로 혼재 — 상수 미통일. |
| 약리학 가드 모듈 (`pharmacology_guards.py`) | **95%** | LITERATURE/SCALE/SIGN/HEURISTIC/ENDPOINT 5종 + 6개 검증 함수 1248줄, 39개 회귀 테스트 PASS. |
| 가드의 분석 루트 런타임 배선 | **15%** | docstring 참조만 존재, `attach_confidence` 호출 0건. 테스트 import만 성공(rootdir sys.path 의존). |

종합 추정: 분석 루트 코어 로직(QC/검증/fail-closed)은 **~88%**, 가드의 실효 통합까지 포함하면 **~70%**.

---

## ⑤ 학술 가치 (재현성 / 정직성)

- **재현성**: rank stability(seed 간 순위 표준편차)와 score consistency(ddG CV, 2σ 이상치)를 정량 지표로 보고하는 것은, in-silico 펩타이드 스크리닝에서 흔히 누락되는 "단일 run 우연성" 통제를 명시적으로 수행 (`backend/validation.py:53-223`). best가 아닌 **median 집계**(`:243-253`)는 outlier robust 통계 관행에 부합.
- **정직성(H-06)**: 휴리스틱 함수의 표면 단위와 실제 의미를 분리 기재하고, valid/invalid 사용을 명문화한 `HEURISTIC_FUNCTION_DISCLAIMERS`는 "계산 불가능을 계산 가능한 척"하는 과대 주장을 구조적으로 차단 — 방법론 보고의 투명성 측면에서 학술적으로 모범적.
- **문헌 추적성**: 모든 lookup 값에 (값, 출처, 코멘트) 3-튜플을 부착(`pharmacology_guards.py:48-54`)해 데이터 출처 감사를 가능케 함.
- **한계의 명시**: selectivity_margin이 실측 Ki와 상관 미검증(Spearman ρ≈−0.3)임을 ENDPOINT_CONFIDENCE 경고로 기록(`pharmacology_guards.py:830-834`) — 결과 과신을 방지.

---

## ⑥ 사용법

### QC 랭킹 실행
```python
from AG_src.agents.qc_ranker import QCRankerAgent
agent = QCRankerAgent()  # 또는 weights=PYROSETTA_ONLY_WEIGHTS
result = agent.execute({
    "candidates": [...],          # list[Candidate]
    "thresholds": {               # gate_thresholds.yaml 대응
        "esmfold_plddt_min": 75,
        "rosetta_ddg_max": -5.0,
        "ranking_mode": "ddg_primary",
        "top_k_by_ddg": 5,
        "gates_enabled": {"plddt": True, "docking": True, "rosetta": True, "selectivity": True},
    },
    "run_id": "run01", "iteration": 1, "output_dir": "...",
})
# result["top_candidates"], result["rank_table"], result["qc_report"]
```
(`qc_ranker.py:487-536`)

### 다중 trial 검증 (API)
```
POST /api/validation/run   {"candidate_sequences": ["AGCK..."], "top_k": 3}
POST /api/validate/selected {"candidate_ids": ["cand001"], "run_id": "..."}
GET  /api/validation/results
```
(`backend/routers/validation.py:112-157`) — 결과 `RELIABLE/CAUTION/UNRELIABLE`.

### 약리학 가드 (모노레포 루트에서)
```python
from pipeline_local.scripts.pharmacology_guards import (
    assert_literature_value, audit_table, assert_in_range,
    is_heuristic_function, attach_confidence,
)
audit_table(my_kd_table, "kyte_doolittle")        # 위반 목록 반환(빈 리스트=통과)
assert_in_range(boman, "boman_index_kcal_per_mol")
resp = attach_confidence(api_result, "/admet/{sequence}")  # confidence_grade 주입
```
(`pharmacology_guards.py:640, 679, 723, 479, 1180`) — **주의: 분석 루트 repo가 아닌 상위 모노레포 sys.path 필요.**

---

## ⑦ 필요 이유

방사성의약품 후보 스크리닝은 wet-lab 검증 전 단계의 in-silico 점수에 의존한다. 이 점수가 (a) 계산 실패를 성공으로 위장하거나, (b) 단일 run의 우연한 best를 진짜 우위로 오인하거나, (c) lookup table 환각·부호 역전으로 오염되거나, (d) 순위용 휴리스틱을 임상 절대값으로 오용되면, **틀린 후보가 합성·동물실험 단계로 진입해 막대한 자원이 낭비**된다. QC 게이트(품질 하한), 다중 trial 검증(재현성 게이트), fail-closed(실패의 안전한 탈락), 약리학 가드(문헌·부호·범위·휴리스틱 정직성)는 이 네 가지 실패 모드를 각각 차단하는 4중 방어선이다. 이는 파이프라인의 **신뢰성과 학술적 정직성**을 담보하는 핵심 인프라다.

---

## 검증 인용 목록

**QC 랭킹** — `AG_src/agents/qc_ranker.py`
- 게이트 순차 적용: `qc_ranker.py:171-287` (Gate1 `:203-212`, Gate2 `:218-230`, Gate3 `:236-248`, Gate4 `:254-273`)
- 가중치: `qc_ranker.py:109-125`; 정규화/랭킹: `:289-354`; 랭킹 모드: `:370-387, 523-529`; 실행 진입점: `:487-536`

**다중 trial 검증** — `backend/validation.py`, `backend/routers/validation.py`
- plausibility 상수: `backend/validation.py:27-29`; rank stability `:53-125`; score consistency `:132-223`; no-dominance(median) `:230-346`; 통합 판정 `:353-378`; batch `:397-427`
- 라우터: `backend/routers/validation.py:30-66`(rule-based), `:80-107`(math+timeout), `:112-157`(endpoints)

**Fail-closed (999/NaN/inf)**
- `backend/routers/validation.py:53-66`; `backend/validation.py:32-33, 36-38`; `backend/analysis.py:45,50`; `pyrosetta_flow/ranking.py:50,52,71`; `pyrosetta_flow/scoring_pipeline.py:227,234`; `pyrosetta_flow/gnina_rescoring.py:375`; `backend/status_emitter.py:220`; `backend/routers/status.py:203`
- graceful degradation enrichment: `backend/routers/status.py:60-154`

**약리학 가드** — `pipeline_local/scripts/pharmacology_guards.py` (분석 루트 외부, 모노레포 루트)
- 모듈 헤더/시나리오: `:1-37`; LITERATURE_VALUES `:56-227`; SCALE_RANGES `:234-246`; HEURISTIC_FUNCTION_DISCLAIMERS `:253-476`(fwkt `:326-339`, chelator `:340-353`); SIGN_CONVENTIONS `:613-621`; 검증 함수 `:479-488, 640-772`; ENDPOINT_CONFIDENCE/attach_confidence `:790-1248`(selectivity 경고 `:830-834`)
- 분석 루트의 docstring 참조(런타임 미배선): `backend/pharmacophore.py:9-13, 102, 180`; `backend/routers/status.py:73-75`
- 테스트(import 성공, 4 PASS): `backend/tests/test_pharmacophore.py:362-401`
- 회귀 테스트 파일 존재: `pipeline_local/tests/test_pharmacology_guards.py`

**미검증 항목 (명시)**
- "CI Q3-1": 분석 루트 트리에서 식별자 직접 인용 미확인 → 미검증.
- `attach_confidence`의 분석 루트 라우터 실런타임 적용: grep 0건 → 사실상 미배선.
- selectivity 게이트의 0.0=미측정 vs 0.0=실값 구분: 코드상 구분 없음(`qc_ranker.py:260`) — 한계로 기록.
