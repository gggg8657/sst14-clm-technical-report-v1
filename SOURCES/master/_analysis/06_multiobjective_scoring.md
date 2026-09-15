# 06. 다목적 스코어링 (NSGA-II Pareto + scalar, GNINA, ECR) 기능 분석

> 대상 모듈
> - `pyrosetta_flow/multiobjective.py` — ObjectiveWeights, cheap_objectives, multiobjective_scalar, screen_selectivity, 독성 페널티
> - `pyrosetta_flow/pareto_ranking.py` — NSGA-II 비지배 정렬 + crowding distance
> - `pyrosetta_flow/gnina_rescoring.py` — GNINA CNN rescoring + ECR consensus
> - `pyrosetta_flow/ranking.py` — 실험 로그 기반 집계 랭킹(보조)
> - `pyrosetta_flow/scoring_pipeline.py` — 4목적 통합 체인(ECR consensus 포함)
>
> 본 보고서의 모든 사실 주장은 `file_path:line` 으로 인용한다. 미확인 항목은 **미검증**으로 명시한다.

---

## ① 동작 원리

### 1.1 4개 목적의 정의 (ΔG + 반감기 + 선택성 + ADMET)

다목적 프레임은 SST-14(`AGCKNFFWKTFTSC`) 변이체를 4축으로 평가한다 (`multiobjective.py:5-9`):

| 목적 | 방향 | 산출 출처 | 성격 |
|------|------|----------|------|
| `ddg` | ↓ (낮을수록 결합 강함) | PyRosetta FlexPepDock (실측 REU/kcal·mol) | **실측** |
| `selectivity_margin` / `delta_margin` | ↑ (양수=SSTR2 선택적) | off-target PyRosetta 도킹 | **실측** |
| `half_life_h` → `stability_norm` | ↑ | 서열 기반 앙상블 surrogate | **surrogate** |
| `admet_score` | ↑ | 물성(GRAVY/Boman/Instability/pI) surrogate | **surrogate** |

honest disclaimer (`multiobjective.py:15-18`): `half_life_h`, `admet_score` 는 **랭킹용 surrogate** 이며 임상 반감기·임상 ADMET 수치가 아니다. `ddg`·`selectivity_margin` 은 실제 FlexPepDock 결과지만 절대 친화도(Ki/Kd)가 아니다. 이는 VR-cycle-09 / H-06("계산 불가능을 계산 가능한 척 하지 않기") 원칙의 코드 반영이다.

### 1.2 비용 계층화 (cost-tiered)

실제 도킹이 비싸므로 2계층으로 나눈다 (`multiobjective.py:11-13`):

- **Layer 0 (모든 후보, 서열만, μs급)**: 반감기 + ADMET surrogate — `cheap_objectives()` (`multiobjective.py:101-155`)
- **Layer 1 (top-K, 실제 PyRosetta, 수분급)**: off-target 선택성 도킹 — `screen_selectivity()` (`multiobjective.py:361-458`)

#### Layer 0 — cheap_objectives
- 반감기: `halflife_ensemble.ensemble_halflife` (휴리스틱 A + RandomForest C 결합)을 호출, 실패 시 레거시 휴리스틱 단독으로 폴백 (`multiobjective.py:113-129`). `stability_norm` 은 log10 정규화로 0~1 변환 (`multiobjective.py:127-129`).
- ADMET: `PharmaProperties` 로 GRAVY/Boman/Instability/aliphatic/pI 계산 (`multiobjective.py:135-153`). Cys3-Cys14 SS bond 를 pI 계산에서 제외한다 (`multiobjective.py:139-140`, 0-indexed 최소·최대 Cys).
- `admet_reasonableness()` 가 4개 물성을 0~1 부분점수로 변환 후 가중 평균 (`multiobjective.py:64-90`):
  - 가중치: Instability 0.35 + GRAVY 0.30 + Boman 0.15 + pI 0.20 (`multiobjective.py:89`)
  - Instability <40 만점, 40~80 선형 감점 (`multiobjective.py:75`)
  - GRAVY ≤-1 만점(친수), ≥1 0점 (`multiobjective.py:77`)
  - Boman <1.0 약함, 1.0~4.0 선형, >4.0 만점 (`multiobjective.py:79-84`)
  - pI 6~8 만점 (`multiobjective.py:86`)

#### Layer 0.5 — pepADMET 독성 페널티
- `predict_toxicity_for_sequences()` 가 pepADMET GNN 을 별도 conda env subprocess 로 배치 추론 (`multiobjective.py:188-203`). 미설치/실패 시 빈 dict 반환(graceful).
- `apply_toxicity_to_extra()` 가 결과를 `admet_score` 에 곱셈 페널티로 반영 (`multiobjective.py:206-241`):
  - binary `is_toxic` 는 **비변별적**(oxytocin·native 까지 전부 toxic 판정)이라 게이트에 쓰지 않고 기록만 함 (`multiobjective.py:211-219`)
  - 대신 hc50 연속값을 **native SST-14 기준선 대비(home-advantage)** 로 평가 (`multiobjective.py:222-241`). native ±5.0 밴드는 "동급" 무페널티 (`_HC50_NATIVE_TOLERANCE=5.0`, `multiobjective.py:166`), 초과분에 선형 비례 페널티(하한 `_TOXIC_ADMET_PENALTY=0.4`, `multiobjective.py:163,238`).
  - 2026-06-17 (VR-A2) 추가: SMILES 파싱 실패로 선형 폴백된 경우 cyclic 구조 소실로 hc50 신뢰 불가 → 게이트 skip(fail-open), 플래그만 기록 (`multiobjective.py:225-228`).
  - `available=False`(추론 불가)면 `admet_score` 를 건드리지 않음 — fail-closed, 가짜 안전 판정 방지 (`multiobjective.py:209,217`).

#### Layer 1 — screen_selectivity
- top-K 후보의 SSTR2 정밀화 복합체를 SSTR1/3/4/5 에 off-target 도킹 (`multiobjective.py:361-458`).
- SSTR2(동일 프로토콜 baseline) + off-target 4종을 `ThreadPoolExecutor` 로 **병렬** 도킹 (순차 ~25분 → 병렬 ~6분/후보) (`multiobjective.py:401-425`).
- `selectivity_margin = min(offtarget_ddg) - baseline` (양수=SSTR2 더 강함=선택적, G-2 SSOT) (`multiobjective.py:434-435`).
- **home-advantage 보정**: native 도 동일 프로토콜에서 +margin 편향이 있으므로 native baseline 대비 `delta_margin = margin - nat_margin` 이 진짜 선택성 신호 (`multiobjective.py:436-447`). `more_selective_than_native = delta_margin > 0` (`multiobjective.py:448`).
- NaN 도킹 결과는 fail-closed 로 None 처리 (`multiobjective.py:419`), 전부 실패 시 `selectivity_margin=None` (`multiobjective.py:429-430`).
- top-K 선별: clash 게이트(`clash_max=10.0`) 통과 후 ddg 오름차순 상위 K (`select_topk_for_selectivity`, `multiobjective.py:308-323`).

### 1.3 NSGA-II Pareto front (`pareto_ranking.py`)

레거시 가중합(0.45/0.20/0.15/0.10/0.10)을 pymoo 기반 비지배 정렬 + crowding distance 로 대체 (`pareto_ranking.py:3-4`).

- **4목적 모두 최소화로 변환** (`_extract_objectives`, `pareto_ranking.py:35-53`): `[ddG, -stability, -druggability, -diversity]` — stability/druggability/diversity 는 부호 반전(`pareto_ranking.py:53`).
- **제약** (`_extract_constraints`, `pareto_ranking.py:56-77`): `hard_violations ≤ 0`, `clash_score - threshold ≤ 0` (기본 임계 10.0, `pareto_ranking.py:32`).
- `pareto_rank_candidates()` 흐름 (`pareto_ranking.py:125-190`):
  1. 목적 행렬 F(n×4) + 제약 위반 벡터 cv(양의 위반 합) 구성 (`pareto_ranking.py:158-166`)
  2. `NonDominatedSorting().do(F)` 로 front 분리 (`pareto_ranking.py:169-170`)
  3. `_penalise_infeasible()` 로 infeasible 후보를 모든 feasible front 뒤로 밀어내고, 위반량 오름차순 단일 페널티 front 로 부착 (`pareto_ranking.py:80-117,173`)
  4. front별 `calc_crowding_distance` 계산(경계점=inf), 후보당 2개 길이 ≤2 인 front 는 전부 inf (`pareto_ranking.py:180-188`)
  5. `pareto_rank`(0=최우선 front), `crowding_distance`(높을수록 고립/다양) 부여 (`pareto_ranking.py:186-188`)
- `select_from_pareto_front()`: rank 오름차순 → crowding distance 내림차순 정렬 후 상위 n (`pareto_ranking.py:193-231`).
- **주의**: `diversity` 는 현재 항상 0.0 으로 입력되어(`scoring_pipeline.py:218`) 4목적 중 1개가 사실상 비활성 상태다(미검증: 다른 호출 경로에서 채워질 가능성).

### 1.4 GNINA CNN rescoring (`gnina_rescoring.py`)

FlexPepDock 출력 PDB 를 GNINA `--score_only` 모드로 재채점 (`gnina_rescoring.py:1-9`).

- **dry-run mock 여부 — 확인됨**: `gnina` 바이너리가 PATH 에 없으면 경고 로그 + 결정론적 mock 점수 반환 (`gnina_rescoring.py:7-9,30,45-47,205-210`). mock 값은 `gnina_cnn_score/affinity/vina_score = 0.0`, `gnina_dry_run = 1.0` (`gnina_rescoring.py:32-37`).
  - **현 환경에서 실측 확인**: `which gnina` → not found. 따라서 현재 GNINA 단계는 **dry-run mock 으로 동작 중**이며 실제 CNN 점수가 아니다.
- 복합체 PDB 를 chain 기준으로 receptor(A)/peptide(B) 임시파일 분리 (`split_receptor_peptide`, `gnina_rescoring.py:97-176`).
- 단일 rescore: subprocess 호출, rc≠0/timeout 시 NaN + 에러 플래그 반환 (`gnina_rescore`, `gnina_rescoring.py:179-265`).
- 배치: `ThreadPoolExecutor` 병렬, 입력 순서 보존 (`batch_gnina_rescore`, `gnina_rescoring.py:268-324`).
- stdout 파싱: `CNNscore` 헤더 다음 줄에서 3토큰 추출, 실패 시 NaN (`_parse_gnina_output`, `gnina_rescoring.py:50-89`).

### 1.5 ECR (Exponential Rank Consensus)

여러 점수항을 순위 기반으로 통합 (`exponential_rank_consensus`, `gnina_rescoring.py:327-395`).

- 공식: `ECR_i = Σ_k exp(-rank_{i,k} / N)`, N=후보 수, 높을수록 좋음 (`gnina_rescoring.py:333-337,388`).
- 모든 지원 점수항은 **낮을수록 좋음** → 오름차순 1-based 순위 부여 (`gnina_rescoring.py:367-382`). NaN/비수치는 inf 로 밀어 최악 순위 (`gnina_rescoring.py:374-375`).
- 입력 dict 를 변형하지 않고 `ecr_score` + `ecr_ranks` 를 추가한 사본 반환, ecr_score 내림차순 정렬 (`gnina_rescoring.py:385-395`).
- 기본 score_keys: `gnina_cnn_score, gnina_cnn_affinity, gnina_vina_score` (`gnina_rescoring.py:359-360`). 단, 파이프라인에서는 **`ddg` 를 추가**해 4개 항으로 호출 (`scoring_pipeline.py:177`).

### 1.6 스칼라 가중 (multiobjective_scalar)

UI 정렬 보조용 단일 점수 (`multiobjective.py:281-305`).

- **가중치 (코드 직접 확인, `ObjectiveWeights`, `multiobjective.py:273-278`)**:
  - `ddg = 0.40` (결합, 최우선)
  - `selectivity = 0.25`
  - `stability = 0.20`
  - `admet = 0.15`
  - 합 = 1.00
- 정규화 (`multiobjective.py:293-298`): ddg 는 `(ddg_ref - ddg)/ddg_scale` (기본 ref=0, scale=50, `multiobjective.py:284-285`), selectivity_margin 은 `/20.0` 포화, stability/admet 은 이미 0~1.
- 최종 점수 = 가중 선형합 (`multiobjective.py:299-305`). runner 는 `_mo_scalar()` 로 UI 표시용으로만 사용 (`runner.py:64-73`).

### 1.7 통합 체인 배선 (scoring_pipeline.py)

`_apply_alternative_scoring()` 가 FlexPepDock 결과에 5단계를 순차 적용, 각 단계 graceful skip (`scoring_pipeline.py:42-281`):

| 단계 | 내용 | 위치 | skip 조건 |
|------|------|------|----------|
| 0 | cheap objectives (반감기+ADMET) | `scoring_pipeline.py:79-95` | 예외 시 non-fatal |
| 0.5 | pepADMET 독성 페널티 | `scoring_pipeline.py:102-120` | `SST_DISABLE_PEPADMET_TOX` env / 미설치 |
| 1 | GNINA rescore (dry-run fallback) | `scoring_pipeline.py:125-157` | `_HAS_GNINA` + PDB 존재 |
| 2 | ECR consensus (ddg+GNINA 4항) | `scoring_pipeline.py:162-189` | GNINA 결과 있을 때만 |
| 3 | Pareto ranking (NSGA-II) | `scoring_pipeline.py:201-243` | `_HAS_PARETO`(pymoo) |
| 4 | BO suggest (로그만, 부수효과 없음) | `scoring_pipeline.py:249-279` | `bo_optimizer` 전달 + 관측 ≥2 |

Pareto 입력 폴백 (`scoring_pipeline.py:204-221`): `stability_norm` 부재 시 clash 기반 proxy(`(40-clash)/40`), `admet_score` 부재 시 ECR 폴백, `diversity=0.0` 고정.

---

## ② 영향

- **선택 품질**: 가중합 단일 점수의 임의 가중치 의존을 NSGA-II Pareto front 로 대체해, 목적 간 trade-off 를 명시적으로 보존한다(front-0 후보 집합). 단일 스칼라는 UI 정렬 보조로 격하 (`runner.py:64-73`).
- **비용 효율**: cost-tiered 구조로 비싼 off-target 도킹을 top-K 에만 적용(`multiobjective.py:11-13`, `runner.py:1414-1418`), 수용체 병렬화로 후보당 ~25분→~6분 (`multiobjective.py:401-403`).
- **정직성(가드)**: surrogate/실측 구분 명시(`multiobjective.py:15-18`), fail-closed 독성 게이트(`multiobjective.py:209`), home-advantage 보정으로 native 편향 제거(`multiobjective.py:436-447`). 환각 점수 차단이 설계에 내장.
- **견고성**: 모든 옵셔널 의존(GNINA/pymoo/pepADMET)을 graceful skip 으로 처리, 부재 환경에서도 파이프라인 진행(`scoring_pipeline.py` 전반).

---

## ③ 관련 Action Item (다목적 통합)

CLAUDE.md 의 Stage 적용 이력 및 진행 메모(MEMORY.md)와 연계되는 항목:

- **다목적 통합 본체**: ΔG+반감기+선택성+ADMET 4축 통합 (`multiobjective.py:5-9`) — sstr2-selectivity-goal(2026-06-10) 의 Δmargin home-advantage 보정과 직결 (`multiobjective.py:436-447`).
- **반감기 앙상블 통합** (2026-06-09): 휴리스틱 A + RF C 결합 (`multiobjective.py:111-121`) — MEMORY 의 "ensemble" 진행 항목.
- **pepADMET 독성 통합** (B, 2026-06-09 / VR-A2 2026-06-17): binary 비변별성 발견 → hc50 home-advantage 게이트 전환 (`multiobjective.py:164-241`).
- **VR-cycle-09 / H-06 가드**: surrogate honest disclaimer 의 코드 반영 (`multiobjective.py:15-18`), CLAUDE.md Stage 8h 와 직결.
- **god-object 분리** (P1, 2026-06-09): runner.py 의 스코어링 체인을 scoring_pipeline.py 로 추출 (`scoring_pipeline.py:3`).
- **잔여 갭(권고)**: `diversity` 목적이 항상 0.0 으로 비활성 (`scoring_pipeline.py:218`) — NSGA-II 4목적 중 1개 미활용. 다양성 메트릭(서열 거리 등) 연결이 미완 Action Item 후보.

---

## ④ 완성도 및 근거

**완성도: 약 80%**

| 영역 | 완성도 | 근거 |
|------|--------|------|
| cheap_objectives / ADMET surrogate | 95% | 완전 구현 + 폴백 + SS bond 처리 (`multiobjective.py:64-155`), 테스트 보유 (`tests/test_multiobjective.py`) |
| 독성 페널티 (hc50 home-adv) | 90% | fail-closed/fail-open 분기 완비 (`multiobjective.py:206-241`), 단 pepADMET 미설치 시 skip |
| screen_selectivity (Layer 1) | 85% | 병렬 도킹+home-adv 보정 완비 (`multiobjective.py:361-458`), config-gated(`enable_selectivity`) |
| NSGA-II Pareto | 90% | 비지배 정렬+crowding+제약 처리 완비 (`pareto_ranking.py` 전체), 단 diversity 미입력 |
| GNINA rescoring | 60% | **현 환경 dry-run mock 으로만 동작**(바이너리 부재 확인), 실측 경로 미검증 (`gnina_rescoring.py:205-210`) |
| ECR consensus | 95% | 공식 구현 + NaN 처리 완비 (`gnina_rescoring.py:327-395`) |
| scalar 통합 | 90% | 가중치 정의+정규화 완비 (`multiobjective.py:273-305`), UI 보조 역할 명확 |
| diversity 목적 | 20% | 항상 0.0 입력으로 비활성 (`scoring_pipeline.py:218`) |

미완/미검증:
- GNINA 실측(live) 경로는 바이너리 부재로 본 환경에서 **미검증** — dry-run 만 확인.
- `diversity` 목적 미연결.
- pepADMET/halflife RF 의 실제 모델 정확도는 본 보고 범위 밖(surrogate 면책 명시됨).

---

## ⑤ 학술 가치 (NSGA-II 다목적 최적화의 의의)

- **Pareto-optimal trade-off 보존**: 약물 후보는 결합력↑·반감기↑·선택성↑·ADMET↑ 가 서로 상충하는 전형적 다목적 문제다. 가중합은 가중치 선택에 따라 비볼록(non-convex) front 의 일부 해를 영구히 배제하지만, NSGA-II 비지배 정렬은 trade-off surface 전체를 보존한다 (`pareto_ranking.py:3-11`). 이는 임의 가중치 가정 없이 의사결정자에게 다양한 후보 집합을 제공한다.
- **Crowding distance 로 다양성 유지**: 동일 front 내에서 고립된(다양한) 해를 우선 선택해 화학적 공간 탐색의 조기 수렴을 방지 (`pareto_ranking.py:180-188,224-228`).
- **제약 처리**: clash/hard_violation infeasible 해를 feasible front 뒤로 relegate 하는 constraint-domination 방식은 Deb 의 NSGA-II 표준 제약 처리와 정합적 (`pareto_ranking.py:80-117`).
- **ECR consensus**: 절대 점수 스케일이 이질적인 다양한 채점기(ddG REU vs GNINA CNN vs Vina kcal)를 순위 공간에서 통합하는 rank-based consensus 는 스케일 불변성을 확보, 단일 채점기 편향을 완화 (`gnina_rescoring.py:333-337`).
- **정직한 surrogate 분리**: surrogate(반감기/ADMET)와 실측(ddG/선택성)을 명시 구분하고 home-advantage 로 baseline 편향을 보정하는 방법론은 AI 후보 발굴의 재현성·신뢰성 측면에서 학술적으로 의미 있는 가드다 (`multiobjective.py:15-18,436-447`).

---

## ⑥ 사용법

### 통합 체인 (자동, runner 경유)
runner.py 가 FlexPepDock 결과에 자동 적용 (`runner.py:190,1123`, `scoring_pipeline.py:42`):
```python
from pyrosetta_flow.scoring_pipeline import _apply_alternative_scoring
candidates = _apply_alternative_scoring(candidates, iter_dir, iteration, bo_optimizer)
# 각 cand.extra_scores 에 half_life_h, admet_score, gnina_*, ecr_score,
# pareto_rank, crowding_distance 채워짐
```

### cheap objectives 단독
```python
from pyrosetta_flow.multiobjective import cheap_objectives, enrich_candidates
obj = cheap_objectives("AGCKNFFWKTFTSC")          # half_life_h, admet_score, stability_norm ...
enrich_candidates(cands)                            # cand dict 에 stability/druggability 매핑 주입
```

### NSGA-II Pareto 랭킹 단독
```python
from pyrosetta_flow.pareto_ranking import pareto_rank_candidates, select_from_pareto_front
ranked = pareto_rank_candidates(cands, clash_threshold=10.0)  # pareto_rank, crowding_distance 부여
best = select_from_pareto_front(ranked, n=5)                  # front-0 우선, crowding 내림차순
```

### GNINA + ECR 단독
```python
from pyrosetta_flow.gnina_rescoring import batch_gnina_rescore, exponential_rank_consensus
scores = batch_gnina_rescore(pdb_paths, max_workers=4)        # 바이너리 없으면 dry-run mock
consensus = exponential_rank_consensus(cands, score_keys=["ddg","gnina_cnn_score",...])
```

### 스칼라 점수 (UI 정렬)
```python
from pyrosetta_flow.multiobjective import multiobjective_scalar, ObjectiveWeights
s = multiobjective_scalar(cand, ObjectiveWeights(ddg=0.40, selectivity=0.25,
                                                  stability=0.20, admet=0.15))
```

### 선택성 (top-K, 비쌈, config-gated)
`config.enable_selectivity=True` + `config.selectivity_top_k` 설정 시 runner 가 top-K 에 자동 호출 (`runner.py:1414-1445`). pepADMET 독성은 `SST_DISABLE_PEPADMET_TOX=1` 로 비활성 가능 (`scoring_pipeline.py:104-105`).

---

## ⑦ 필요한 이유

- **단일 점수의 한계 극복**: SSTR2 방사성의약품 후보는 결합·반감기·선택성·안전성이 동시 최적일 수 없다(상충). 가중합 단일 랭킹은 가중치 선택자의 주관에 결과가 종속되므로, Pareto front 로 객관적 trade-off 집합을 확보해야 한다 (`pareto_ranking.py:3-11`).
- **계산 비용 통제**: 실제 PyRosetta off-target 도킹은 후보당 수분이 들어 전수 적용이 불가능하다. cost-tiered 구조로 저비용 surrogate 가 전수 1차 필터, 비싼 도킹은 top-K 에만 적용해야 처리량을 확보한다 (`multiobjective.py:11-13`).
- **이질적 채점기 통합**: ddG(REU)·GNINA(CNN 확률)·Vina(kcal)는 스케일이 달라 직접 합산이 불가능하다. ECR 순위 합의가 스케일 불변 통합을 제공한다 (`gnina_rescoring.py:333-337`).
- **환각 차단 / 정직성**: AI 발굴 파이프라인의 신뢰성은 "모르는 것을 아는 척하지 않음"에 달려있다. surrogate 면책 명시, fail-closed 독성 게이트, home-advantage baseline 보정이 가짜 점수의 의사결정 오염을 막는다 (`multiobjective.py:15-18,209,436-447`).
- **선택성 우선 목표**: SSTR1/3/4/5 대비 SSTR2 선택성은 off-target 부작용 회피의 핵심이다. delta_margin(home-advantage 보정) 이 native 대비 진짜 선택성 신호를 제공한다 (`multiobjective.py:447-448`).

---

## 검증 인용 목록

| 주장 | 인용 |
|------|------|
| 4목적 정의(ddg↓/half_life↑/selectivity↑/admet↑) | `multiobjective.py:5-9` |
| surrogate vs 실측 honest disclaimer | `multiobjective.py:15-18` |
| cost-tiered 2계층 구조 | `multiobjective.py:11-13` |
| admet_reasonableness 가중치(0.35/0.30/0.15/0.20) | `multiobjective.py:89` |
| cheap_objectives 반감기 앙상블+폴백 | `multiobjective.py:111-129` |
| SS bond pI 제외 처리 | `multiobjective.py:139-140` |
| 독성 페널티 상수(0.4 / ±5.0 / 200.0) | `multiobjective.py:163-168` |
| binary is_toxic 비변별 → hc50 게이트 | `multiobjective.py:211-219` |
| hc50 home-advantage 페널티 | `multiobjective.py:232-241` |
| VR-A2 선형 폴백 hc50 skip | `multiobjective.py:225-228` |
| fail-closed(available=False 무처리) | `multiobjective.py:209,217` |
| ObjectiveWeights 값(0.40/0.25/0.20/0.15) | `multiobjective.py:273-278` |
| multiobjective_scalar 정규화·선형합 | `multiobjective.py:293-305` |
| select_topk_for_selectivity clash 게이트 | `multiobjective.py:308-323` |
| screen_selectivity 병렬 도킹 | `multiobjective.py:401-425` |
| selectivity_margin 정의 | `multiobjective.py:434-435` |
| delta_margin home-advantage 보정 | `multiobjective.py:436-448` |
| NSGA-II 가중합 대체 | `pareto_ranking.py:3-4` |
| 4목적 최소화 변환 | `pareto_ranking.py:35-53` |
| 제약(hard_violations/clash) | `pareto_ranking.py:56-77` |
| infeasible relegation | `pareto_ranking.py:80-117` |
| 비지배 정렬+crowding | `pareto_ranking.py:169-188` |
| select_from_pareto_front 정렬키 | `pareto_ranking.py:224-231` |
| GNINA dry-run mock 동작 | `gnina_rescoring.py:7-9,30,45-47,205-210` |
| dry-run mock 값 | `gnina_rescoring.py:32-37` |
| GNINA 바이너리 부재(환경 실측) | `which gnina` → not found |
| ECR 공식 | `gnina_rescoring.py:333-337,388` |
| ECR NaN→inf 처리 | `gnina_rescoring.py:374-375` |
| ECR 기본 score_keys | `gnina_rescoring.py:359-360` |
| 파이프라인 5단계 체인 | `scoring_pipeline.py:42-281` |
| ECR 호출 시 ddg 추가(4항) | `scoring_pipeline.py:177` |
| Pareto diversity=0.0 고정 | `scoring_pipeline.py:218` |
| Pareto 입력 폴백(clash proxy/ECR) | `scoring_pipeline.py:204-212` |
| pepADMET env 비활성 플래그 | `scoring_pipeline.py:104-105` |
| runner scalar UI 보조 호출 | `runner.py:64-73` |
| runner 선택성 top-K config-gated | `runner.py:1414-1445` |
