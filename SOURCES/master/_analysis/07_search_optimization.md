# 07. 탐색 최적화 — Thompson Bandit + 베이지안 최적화 + 수렴 감지

**대상 모듈**
- `pyrosetta_flow/bandit.py` — Thompson Sampling 위치 밴딧 (`PositionBandit`)
- `pyrosetta_flow/bayesian_optimizer.py` — GP 베이지안 최적화 (`BayesianPeptideOptimizer`)
- `pyrosetta_flow/convergence.py` — Mann-Whitney U 수렴 감지 (`ConvergenceDetector`)
- `pyrosetta_flow/adapter.py` — warm-start 래퍼 (`get_bandit_guidance` / `initialize_from_history`)

**성격**: 본 클러스터는 후보를 "더 빨리·더 똑똑하게 고르도록" 돕는 **탐색 가속(인프라) 계층**이다. 결과의 과학적 정당성(ddG·선택성·독성)은 별도 스코어링/가드 계층이 책임진다. 즉 이 모듈들은 Action Item(과학적 결론)에 **직접 무관**할 가능성이 높다 — 자세한 근거는 §3, §7.

---

## ① 동작 원리

### 1-A. Thompson Sampling 위치 밴딧 (`bandit.py`)

**실제 구현 (stub 아님).** 각 변이 가능 위치를 하나의 "암(arm)"으로 보고 Beta(α, β) 분포를 유지한다.

- 변이 가능 위치는 1-indexed로 `[1, 2, 4, 5, 6, 11, 12, 13]` 고정 (`bandit.py:24`). 즉 Cys3·Cys14(SS bond)와 Trp8·Lys9·Thr10(FWKT pharmacophore 핵심부 일부)은 암 집합에서 제외 — 도메인 제약이 위치 후보 자체에 반영됨.
- **히스토리 부트스트랩** (`bandit.py:49-100`): `record_type=="candidate"`, `status=="success"`, ddG가 `[-60, 200]` plausible 범위 내(`bandit.py:26-27, 60-61`)인 레코드만 사용. WT(`AGCKNFFWKTFTSC`) 관측이 있으면 그 평균을, 없으면 전체 ddG의 중앙값을 baseline으로 삼음(`bandit.py:74-78`). 각 후보에서 WT 대비 바뀐 위치를 식별(`bandit.py:86-90`)하고, ddG가 baseline보다 **낮으면(개선) 해당 위치들의 α를 +1**, 아니면 **β를 +1** (`bandit.py:95-100`).
- **위치 선택** (`sample_focus_positions`, `bandit.py:102-115`): 각 암의 Beta 분포에서 `random.betavariate(α, β)`로 표본을 뽑고(`bandit.py:110`) 표본값 상위 n개 위치를 반환. → 정통 Thompson Sampling (posterior sampling 기반 탐색-활용 균형). `rng` 주입 가능해 재현성 확보.
- 온라인 갱신 `update`(`bandit.py:117-124`)와 진단용 `get_arm_stats`(기댓값 α/(α+β), 관측수)(`bandit.py:126-137`) 제공.

> 주의: 개선/악화를 **이진(Bernoulli)** 으로만 본다. ddG가 1점 좋아진 것과 50점 좋아진 것을 동일하게 α+1 처리 — 개선 폭(magnitude)은 무시. 또한 다중 위치가 동시에 바뀐 후보는 **credit assignment**가 모든 변이 위치에 균등 분배되어, 어느 위치가 실제 기여했는지 구분 불가(`bandit.py:96-100`). 알고리즘 자체는 정상이나 신호 해상도는 거친 편.

### 1-B. GP 베이지안 최적화 (`bayesian_optimizer.py`)

**실제 구현 (stub 아님).** 단, BoTorch 유무에 따라 2단 폴백.

- **임베딩 계층 (pluggable)**: `PeptideEmbedder` ABC(`bayesian_optimizer.py:61-90`) 아래
  - `OneHotEmbedder` — 잔기당 20-dim one-hot 연결, `max_len` 패딩(`:96-130`). 외부 의존 없음. **실제로 runner가 사용하는 임베더**(`runner.py:479`).
  - `ESM2Embedder` — transformers/torch 필요, mean-pool last hidden state(`:136-183`). 실제 구현이나 runner에서 미사용(설치 의존).
- **GP 대리모델**:
  - BoTorch 가용 시 `SingleTaskGP` × 목적함수 개수 → `ModelListGP`, `fit_gpytorch_mll`로 marginal likelihood 최적화(`:362-372`).
  - 미가용 시 `_FallbackGP` — numpy만으로 RBF 커널 GP를 직접 구현(커널 행렬 역행렬 `np.linalg.inv`, 예측 평균/분산)(`:189-246`). 정식 GP 회귀식 그대로 — **stub이 아니라 축소판 정식 구현**.
- **획득함수(acquisition)**:
  - BoTorch 경로: qNEHVI를 import는 하나(`:33-35`), 실제 `_acquisition_botorch`는 **posterior 평균의 ref point 대비 개선량 곱(product-of-improvements)** 을 쓰는 **경량 프록시**다(`:516-531`). 주석에도 "full qNEHVI is expensive for large sets"라고 명시(`:516-517`). → 즉 진짜 qNEHVI hypervolume 계산은 **수행하지 않음**.
  - 폴백 경로: 목적별 UCB(`mean + 2.0·√var`) 합산(`:533-550`).
- **제안 생성** `suggest`(`:386-428`): reference_seq에 대해 허용 위치마다 19개 단일점 변이를 전부 enumerate(`_enumerate_mutations`, `:430-464`)하고 획득값 상위 n개 반환.
- 목적함수: runner에서 `["ddg"(최소화), "ecr_score"(최대화)]`로 설정(`runner.py:480-482`); 최소화 목적은 부호 반전으로 항상 최대화 문제로 변환(`:316-320`).

> 통합 깊이 주의: scoring_pipeline에서 BO는 `fit`→`suggest`만 호출하고 **그 제안을 다음 후보 생성에 피드백하지 않는다.** 코드 주석이 명시: "Step 4: Bayesian Optimization suggest (부수효과 없음 — 로그만)"(`scoring_pipeline.py:246`), 결과는 `BO suggest top-3 positions: ...` 출력에 그침(`scoring_pipeline.py:270-274`). 즉 BO는 현재 **관찰자/로깅 모드**로만 작동하며 탐색 루프를 실제로 조향하지 않는다.

### 1-C. Mann-Whitney U 수렴 감지 (`convergence.py`)

**실제 구현 (stub 아님), scipy 무의존.**

- `_rank_data`(`convergence.py:14-27`): 동점 평균 순위 처리 포함한 순위 부여 — 정석.
- `_mann_whitney_u`(`:30-59`): U 통계량 = R1 − n1(n1+1)/2, n_total ≥ 8일 때 연속성 보정 없는 정규근사(μ=n1·n2/2, σ=√(n1·n2·(N+1)/12)), 양측 p값 = 2·(1−Φ(z))(`:50-58`). n_total<8이면 보수적으로 1.0 반환(데이터 부족)(`:47-48`). Φ는 `math.erf` 기반 정규 CDF(`:62-64`). → 교과서 정규근사 정확. (동점 분산 보정 항은 생략 — 동점 많을 때 약간 보수적.)
- `ConvergenceDetector.is_converged`(`:79-143`): 최근 window와 직전 window의 top-k ddG를 비교. 수렴 판정 = **(p > 유의수준, 즉 더이상 유의한 개선 없음) AND (변동계수 CV < 0.15)** 동시 충족(`:126-129`). 최소 `2·window_size` iteration 필요(`:94-99`). 사람이 읽을 권고 문자열 생성.

> **중요(통합)**: runner는 매 iteration `add_iteration`→`is_converged`로 플래그를 계산하고 emitter로 보고(`runner.py:906-924`, `1287-1293`)하지만, 메인 iteration 루프(`runner.py:508`)에는 `conv_flag` 기반 `break`가 **없다**. 즉 수렴 감지는 **자문(advisory)·UI 표시용**일 뿐 루프를 조기 종료시키지 않는다. (별도의 *validation* 단계 early-stop은 CV 기반으로 존재하나 — `runner.py:1368-1376` — 이는 최종 후보 다중시행 검증용이지 본 수렴감지기와 무관.)

---

## ② 영향 (탐색 효율 / warm-start)

- **warm-start**: `get_bandit_guidance`(`adapter.py:113-128`)가 prior_records로 밴딧을 부트스트랩해 `focus_positions`를 산출. runner는 **Planner가 focus_positions를 비워둔 경우에만** 밴딧 추천으로 채운다(`runner.py:592-593`, `if bandit_guidance and not guidance.get("focus_positions")`). → LLM Planner 우선, 밴딧은 데이터 기반 **폴백**. 무한 발굴 엔진의 글로벌 리더보드 warm-start(`runner.py:498`)와 함께 "이전 실행 지식 재사용" 축을 형성.
- **탐색 효율**: 밴딧은 과거에 개선을 많이 낸 위치로 변이 예산을 편향 → 무작위 위치 대비 수렴 가속 기대. 다만 §1-A의 거친 credit assignment·이진화로 신호가 약해질 수 있음.
- **BO 영향은 현재 미미**: §1-B대로 제안이 루프에 피드백되지 않아(로깅만), 실측 탐색 효율 기여는 사실상 0에 가깝다. 향후 `suggest` 결과를 후보 풀에 주입하면 영향이 생김.
- **수렴 감지 영향**: 조기 종료가 아니라 "그만둘 만하다"는 신호 제공 → 사람/상위 오케스트레이터의 의사결정 보조. 계산 예산 절감 효과는 현재 자동화되어 있지 않음.

---

## ③ 관련 Action Item

**직접 무관 (탐색 가속은 인프라 계층)** — 근거:

1. 이 모듈들은 **어떤 후보가 SSTR2 선택적·저독성인지**를 결정하지 않는다. 그 판정은 ddG 계산(PyRosetta)·선택성 루프(`selectivity_loop`)·약리학 가드(`pharmacology_guards.py`)가 한다. 밴딧/BO/수렴은 "그 판정 대상을 **어떤 순서로 얼마나 빨리** 고를지"만 다룬다.
2. 코드 자체가 자신을 보조 역할로 선언: 밴딧은 Planner 미지정 시 폴백(`runner.py:592`), BO는 "부수효과 없음 — 로그만"(`scoring_pipeline.py:246`), 수렴은 break 없는 advisory(`runner.py:508` 루프에 미반영).
3. 따라서 과학적 결론(Action Item)을 바꾸지 않으며, 제거해도 결과의 **정확성**은 변하지 않고 **속도/예산**만 영향 받는다.

→ Action Item과의 연결고리는 **간접적**: "동일 예산으로 더 좋은 후보에 도달할 확률을 높인다"는 효율 축에서만 기여.

---

## ④ 완성도 % + 근거

| 모듈 | 완성도 | 근거 |
|------|--------|------|
| `bandit.py` (Thompson) | **90%** | 부트스트랩·표본·갱신·진단 모두 실구현, rng 주입 재현성, 도메인 위치 제약 반영. 통합도 활성(`runner.py:592`). 감점: 이진화로 개선폭 무시, 다중변이 credit assignment 미분리(`bandit.py:96-100`). 전용 테스트 다수 존재(`tests/test_bandit.py`). |
| `bayesian_optimizer.py` (GP/BO) | **65%** | GP(BoTorch+numpy 폴백)·임베더·enumerate·suggest 실구현. 단 (a) qNEHVI는 import만, 실제는 product-of-improvements **프록시**(`:516-531`); (b) suggest 결과가 루프에 **미피드백**(`scoring_pipeline.py:246`, 로깅만). 즉 "구현됐으나 미배선". |
| `convergence.py` (Mann-Whitney) | **85%** | 순위·U·정규근사·CV 게이트 정석 구현, scipy 무의존. 감점: 동점 분산보정 항 생략, 그리고 **조기종료 미배선**(advisory only, `runner.py:508` 루프에 break 없음). 전용 테스트 존재(`tests/test_convergence.py`). |
| `adapter.get_bandit_guidance` | **95%** | 얇은 래퍼, 예외 비치명 처리(`runner.py:467-468`), Planner 포맷 호환. |

**클러스터 종합 ≈ 80%** (알고리즘 구현 품질은 높음, 일부는 "구현 후 미배선"이라 운영 기여가 잠재 상태).

---

## ⑤ 학술 가치 (BO/bandit의 단백질 설계 적용 의의)

- **Thompson Sampling을 잔기 위치 선택에 적용**: 서열 공간은 조합폭발(14aa × 20AA)이라 전수 탐색 불가. 위치별 Beta-Bernoulli 밴딧은 "어느 위치를 건드릴 때 개선 확률이 높은가"를 posterior로 학습 — 위치 수준 탐색-활용 균형을 경량으로 제공하는 합리적 모델링. 단백질 변이 설계에서 밴딧 기반 위치 우선순위화는 문헌적으로 타당한 접근.
- **GP 기반 다목적 BO (qNEHVI 지향)**: ddG-선택성 등 상충 목적의 Pareto 전선을 surrogate로 탐색하는 것은 wet-lab/도킹 비용이 큰 펩타이드 설계에서 sample-efficiency를 높이는 표준 패러다임. ESM-2 임베딩 옵션은 서열 유사도를 잠재공간 거리로 환원해 GP 커널에 의미 부여 — 최신 PLM+BO 결합 흐름과 정합.
- **scipy 무의존 Mann-Whitney U + CV 이중 게이트**: 단순 best값 추적이 아니라 분포 수준(중앙 경향 변화 없음 + 저변동)에서 plateau를 통계적으로 판정 — 노이즈 큰 ddG 시계열에 대해 과적합 없는 정지 규칙. 재현성·이식성(외부 의존 0) 측면에서 실용적.
- **종합 의의**: "큰 LLM Planner + 데이터 기반 밴딧/BO 폴백 + 통계적 수렴"의 하이브리드는, 비싼 물리 시뮬레이션(PyRosetta) 호출 예산을 통계적으로 절약하려는 active-learning형 설계로서 방법론적 가치가 있다. 다만 가치 실현은 BO 피드백·수렴 조기종료의 **배선 완성**에 달려 있음.

---

## ⑥ 사용법

**밴딧 (직접)**
```python
from pyrosetta_flow.bandit import PositionBandit
bandit = PositionBandit()                 # 기본 위치 [1,2,4,5,6,11,12,13]
bandit.initialize_from_history(records)   # experiment_log 레코드 리스트
focus = bandit.sample_focus_positions(n=3)
```

**밴딧 (runner 통합 래퍼)** — `adapter.get_bandit_guidance(records, n_focus=3)` → `{"focus_positions": [...], "source": "bandit_thompson", "arm_stats": {...}}`. runner는 Planner가 focus를 비웠을 때만 사용(`runner.py:592-593`).

**베이지안 최적화**
```python
from pyrosetta_flow.bayesian_optimizer import BayesianPeptideOptimizer, OneHotEmbedder
bo = BayesianPeptideOptimizer(
    embedder=OneHotEmbedder(max_len=14),
    objectives=["ddg", "ecr_score"],
    maximize=[False, True],
)
bo.fit(obs_dicts)                          # [{"sequence","ddg","ecr_score"}, ...] (>=2건)
sugg = bo.suggest(n=3, reference_seq="AGCKNFFWKTFTSC")
```
runner는 `_HAS_BO` 시 자동 생성(`runner.py:476-486`)하고 scoring_pipeline가 `fit→suggest`(현재 로깅 전용).

**수렴 감지**
```python
from pyrosetta_flow.convergence import ConvergenceDetector
det = ConvergenceDetector(window_size=3, significance_level=0.05)
det.add_iteration(it, top_k_ddgs)
converged, details = det.is_converged()    # details: p_value, cv, recommendation
```
runner는 매 iteration 호출·emitter 보고(`runner.py:906-924`). config 키: `convergence_window_size`, `convergence_significance`, `bandit_n_focus`(`schema.py:37-39`).

---

## ⑦ 왜 필요한가 (Action Item 무관임에도)

이 클러스터는 결론을 바꾸지 않으므로 "없어도 되는 것 아니냐"는 질문이 자연스럽다. 그럼에도 필요한 이유는 **비용 구조** 때문이다.

1. **물리 시뮬레이션 예산이 병목.** PyRosetta ddG·도킹은 후보당 수초~수분. 서열 공간은 조합폭발이라 무작위/전수 탐색은 같은 예산으로 훨씬 적은 좋은 후보에 도달한다. 밴딧은 변이 예산을 "역사적으로 잘 되는 위치"로 편향해 **동일 예산당 기대 수확을 높인다.** 결론의 정확성이 아니라 **결론에 도달하는 속도·확률**을 다루므로 인프라이며, 무한 발굴 엔진처럼 장기 가동 시 그 누적 효과가 핵심 자산이 된다.
2. **warm-start = 지식의 망각 방지.** `initialize_from_history`/글로벌 리더보드는 이전 실행에서 배운 위치 선호를 다음 실행이 상속하게 한다. 이것이 없으면 매 실행이 cold-start로 동일 학습을 반복 — 무한 엔진 설계의 전제(역대 지식 재사용)와 직결.
3. **정지 규칙 = 예산 낭비 차단.** 수렴 감지는 plateau를 통계적으로 알려, 개선 없는 iteration에 시뮬레이션 예산을 더 쓰는 것을 막는 신호를 준다(현재는 advisory지만 자동 종료로 승격 시 즉시 예산 절감).
4. **Planner 환각/공백의 안전망.** 밴딧은 LLM Planner가 focus_positions를 못 줄 때의 **데이터 기반 폴백**(`runner.py:592`)으로, 탐색이 무방향으로 빠지는 것을 방지한다.

요약: 과학적 **판정**은 스코어링/가드가, 그 판정을 **언제·무엇에 대해 효율적으로 내릴지**는 본 클러스터가 책임진다. Action Item에 직접 기여하지 않지만, 한정된 시뮬레이션 예산을 가진 발굴 파이프라인의 **처리량·지속가능성**을 결정하는 보조 인프라로서 필요하다. 다만 BO 피드백 배선과 수렴 기반 자동 종료가 미완이므로, 현재 실효 기여는 "밴딧 warm-start" 축에 집중되어 있고 BO/수렴은 잠재력 상태임을 밝혀둔다.

---

## 검증 인용 목록 (file_path:line)

- `bandit.py:24` — 변이 가능 위치 고정 `[1,2,4,5,6,11,12,13]` (SS bond/pharmacophore 일부 제외)
- `bandit.py:26-27, 60-61` — ddG plausible 범위 필터 `[-60, 200]`
- `bandit.py:49-100` — `initialize_from_history` 부트스트랩 (baseline·credit·α/β 갱신)
- `bandit.py:74-78` — WT 평균 또는 중앙값 baseline
- `bandit.py:95-100` — 개선/악화 이진 → α/β +1 (magnitude 무시, 다중변이 균등 credit)
- `bandit.py:102-115` — `sample_focus_positions` (`random.betavariate` Thompson 표본)
- `bandit.py:117-137` — `update`/`get_arm_stats`
- `bayesian_optimizer.py:25-45` — BoTorch 선택적 import 가드
- `bayesian_optimizer.py:96-130` — `OneHotEmbedder` 실구현
- `bayesian_optimizer.py:136-183` — `ESM2Embedder` 실구현(미배선)
- `bayesian_optimizer.py:189-246` — `_FallbackGP` numpy RBF GP 정식 구현
- `bayesian_optimizer.py:362-372` — BoTorch `SingleTaskGP`/`ModelListGP` fit
- `bayesian_optimizer.py:430-464` — 단일점 변이 enumerate
- `bayesian_optimizer.py:516-531` — qNEHVI 대신 product-of-improvements **프록시**
- `bayesian_optimizer.py:533-550` — 폴백 UCB 획득함수
- `convergence.py:14-27` — `_rank_data` 동점 평균순위
- `convergence.py:30-59` — `_mann_whitney_u` 정규근사 p값
- `convergence.py:62-64` — `math.erf` 정규 CDF (scipy 무의존)
- `convergence.py:94-99` — 최소 `2·window_size` iteration 요구
- `convergence.py:126-129` — 수렴 = (p>유의수준) AND (CV<0.15)
- `adapter.py:113-128` — `get_bandit_guidance` warm-start 래퍼
- `runner.py:462-472` — 밴딧/수렴 detector 초기화
- `runner.py:476-486` — `BayesianPeptideOptimizer` 자동 생성(OneHotEmbedder, ddg/ecr_score)
- `runner.py:508` — 메인 iteration 루프 (conv_flag 기반 break 부재)
- `runner.py:592-593` — Planner 미지정 시에만 밴딧 focus 적용 (폴백)
- `runner.py:906-924, 1287-1293` — 수렴 계산·emitter 보고 (advisory)
- `scoring_pipeline.py:246` — "Step 4: BO suggest (부수효과 없음 — 로그만)"
- `scoring_pipeline.py:249-279` — BO `fit`→`suggest`, 결과 로깅만
- `schema.py:37-39` — config 키 `convergence_window_size`/`convergence_significance`/`bandit_n_focus`
- `tests/test_bandit.py`, `tests/test_convergence.py` — 전용 단위 테스트 존재
