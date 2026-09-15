# 04. 혈중 반감기 앙상블 (Half-life Ensemble) — 간접 검증의 대표 사례

> **한 줄 요약**: 휴리스틱(GLP-1 약동학 기반 log-multiplicative)과 RF 회귀(PEPlife2 학습)
> 두 surrogate를 log10 정규화 평균으로 결합한 앙상블. **절대 t½(임상 반감기)을 주장하지 않으며**,
> 후보 간 **상대 순위 surrogate**로서 Spearman 벤치마크로만 간접 검증된다.
> 신뢰등급: **절대값 LOW / 상대순위 MED**.

분석 대상 파일:
- `pyrosetta_flow/halflife_ensemble.py` (앙상블 결합·정규화·fallback)
- `pyrosetta_flow/halflife_model/features.py` (RF feature 추출)
- `AG_src/pipeline/step08_stability.py` (휴리스틱 log-multiplicative 모델·게이트)
- `AG_src/tests/test_halflife_benchmark.py` (Spearman 상대순위 검증)

---

## ① 동작 원리

### 1.1 두 상보적 추정기

앙상블은 두 개의 독립 surrogate를 결합한다 (`halflife_ensemble.py:3-7`).

**(A) 휴리스틱** — `AG_src.pipeline.step08_stability.predict_half_life`
GLP-1 작용제 약동학을 참고한 **log10 곱셈(log-multiplicative) 모델**이다
(`step08_stability.py:225-275`). 핵심 식:

```
log10(t½) = log10(base) + proteo_log + Σ modification_log_mult
```

- `base = 0.05h` — 짧은 선형 펩타이드 혈중 baseline (~3분) (`step08_stability.py:111`)
- `proteo_log` — 프로테아제 취약성 음수 항. 평균 취약성, dipeptide 절단부위 수,
  말단 exopeptidase를 결합 (`step08_stability.py:243-247`):
  ```
  proteo_log = -( K_PROTEO·(avg_vuln-0.8) + K_CLEAVE·cleavage_sites + K_TERM·((n_term+c_term)/2-0.8) )
  ```
  계수: `_HL_PROTEO_K=0.32`, `_HL_CLEAVE_K=0.06`, `_HL_TERM_K=0.15` (`step08_stability.py:112-114`)
- `modification_log_mult` — 곱셈(log 합) modification 항. fatty_acid가 장반감기를 주도
  (`step08_stability.py:115-121`): fatty_acid `log10(2800)`, pegylation `log10(2200)`,
  d_amino_acid `log10(35)`, cyclization `log10(1.8)`, substitution `log10(4.0)`
- 고리화는 Cys 쌍(≥4 간격) 자동 감지로 적용 (`step08_stability.py:249-254`)
- 최종 클램프: `max(0.02, min(10**log_hl, 1.0e4))` (`step08_stability.py:269`)

**(B) RF (PEPlife2)** — `halflife_model/halflife_gbr.joblib` (43MB, GBR 번들)
sklearn 회귀 모델을 joblib로 로드 (`halflife_ensemble.py:35-46`). 추론 시
`features.featurize`로 feature 벡터를 만들고 `np.expm1(model.predict)` 로 시간 단위 복원
(`halflife_ensemble.py:49-63`). feature는 길이, 20종 잔기 분율, GRAVY, net charge,
Arg/Lys·Met·Cys·Pro 카운트, cleavage_sites/density, 말단 KR, D-form/고리/화학수정/말단캡/
지질·PEG 플래그 (`features.py:11-49`). **feature 순서는 학습 시점의 `feat_cols`와
정확히 일치해야** 한다 (`features.py:1-3`, `halflife_ensemble.py:59`).

### 1.2 log10 정규화 평균 결합

두 시간값을 동일한 `[0.02h, 200h]` log10 스케일로 정규화 후 평균한다
(`halflife_ensemble.py:19, 75-78`):

```python
_HL_LO, _HL_HI = log10(0.02), log10(200.0)
norm = clip((log10(hl) - _HL_LO) / (_HL_HI - _HL_LO), 0, 1)
stability_norm = mean(가용한 정규화 값들)   # halflife_ensemble.py:90-100
```

- 두 값 모두 존재 → `source="ensemble"`
- 한쪽만 → `source="heuristic"` 또는 `"rf"`
- 둘 다 실패 → `source="none"`, `stability_norm=0.0`

### 1.3 표시값과 fallback

- **표시용 `half_life_h`** 는 절대 스케일을 더 신뢰하는 **휴리스틱**을 사용하고,
  없으면 RF로 대체 (`halflife_ensemble.py:7, 101-103`)
- **graceful fallback**: sklearn/joblib/모델 부재 시 RF는 `None`을 반환하고 휴리스틱
  단독으로 동작 (`halflife_ensemble.py:43-46, 51-52`). 휴리스틱마저 import 실패하면
  `_HAS_HEUR=False` 로 안전 처리 (`halflife_ensemble.py:22-27, 66-72`)

---

## ② 영향 (downstream)

앙상블 출력은 다목적 스코어링의 **stability objective**로 흘러간다.

- `pyrosetta_flow/multiobjective.py:115-121` 에서 `ensemble_halflife(seq)`를 호출해
  `half_life_h`, `half_life_heuristic_h`, `half_life_rf_h`, `halflife_source`,
  `stability_norm` 을 `extra_scores`에 채운다. 실패 시 휴리스틱 단독 fallback
  (`multiobjective.py:123-132`)
- `stability_norm` → pareto_ranking 의 `stability` 키로 매핑 (`multiobjective.py:263-264`)
- 가중 스칼라화에서 `stability` 가중치 **0.20** 으로 합산 (`multiobjective.py:277, 297-302`)
- `scoring_pipeline.py:206` 에서 `stability_norm` 을 소비. 동 파일 `:77` 에 정직성
  면책("half_life/admet 은 ranking surrogate, 임상 수치 아님") 명시

즉 이 기능은 **후보 랭킹의 약 20% 비중을 차지하는 안정성 축**을 담당하며, 절대 t½이
아니라 정규화 순위로만 기여한다.

---

## ③ 관련 Action Item

### A. 반감기 재보정 (완료, 2026-06-09)
기존 **additive 모델**은 cyclization +24h가 사이클릭/SS 펩타이드(SST-14)를 과대예측해
plain Spearman = **−0.5(순위 역전)** 였다 (`step08_stability.py:105-110`,
`test_halflife_benchmark.py:4-5`). **log-multiplicative 모델**로 교체하여 SST-14를
16.6h → ~0.04h 로 교정하고 문헌 8종에서 Spearman 0.86(all)/0.50(plain)을 회복
(`step08_stability.py:108-110`). 회귀 테스트로 고정 (§④).

### VR-S5-01. 프로테아제 취약성 계수 출처 (미해결, TODO)
`_PROTEASE_VULNERABILITY` 딕셔너리(`step08_stability.py:50-71`)는 트립신(R/K),
키모트립신(F/Y/W), 엘라스타제(A/V/L)의 **정성적 절단 선호 순서**를 반영한 **IN-HOUSE
휴리스틱 ranking 프록시**이며, 정량 문헌값(kcat/Km, Schechter-Berger pocket 데이터)이
아니다 (`step08_stability.py:42-48`). 절대 의미가 없고 후보 간 상대 순위로만 사용된다.
TODO: 실측 kcat/Km로 교체 또는 출처 주석 (`step08_stability.py:48`).

### A1. 게이트 정직성 정정 (완료, 2026-06-17)
`apply_stability_gate`에 신뢰등급 경고를 명시하고, 절대 임계 대신 surrogate 순위 상위
비율만 통과시키는 `relative_percentile` 모드를 추가 (`step08_stability.py:526-569`).
'144h 달성' 같은 임상 단위 해석을 금지하고 게이트를 **상대 우선순위 필터**로 재정의
(`step08_stability.py:534-543`). 회귀 테스트 `test_stability_gate_relative_percentile_mode`
로 고정 (`test_halflife_benchmark.py:79-88`).

---

## ④ 완성도 % + 근거

**완성도: 약 80%**

| 항목 | 상태 | 근거 |
|------|------|------|
| 앙상블 결합·정규화 로직 | 완료 | `halflife_ensemble.py:81-108` |
| graceful fallback (3단계) | 완료 | `halflife_ensemble.py:43-46, 66-72, 123-132` |
| 휴리스틱 재보정 (log-mult) | 완료 | `step08_stability.py:225-275` |
| RF 학습·번들 배치 | 완료 | `halflife_model/halflife_gbr.joblib` (43MB 존재) |
| 상대순위 벤치마크 | 완료 (8종, ρ≥0.7 고정) | `test_halflife_benchmark.py:21-62` |
| 게이트 정직성 정정 | 완료 | `step08_stability.py:526-569` |
| **절대값 캘리브레이션** | **미시행** | `step08_stability.py:536-538` ("절대값 캘리브레이션 미시행") |
| **프로테아제 계수 정량 출처** | **미확보 (VR-S5-01)** | `step08_stability.py:46-48` |

남은 20%는 정량 검증(in-vitro 혈청 어세이 캘리브레이션)과 계수 문헌화로, 둘 다
**절대값 신뢰등급(LOW)을 끌어올리는 작업**이다. 상대순위 기능 자체는 완비.

---

## ⑤ 학술 가치

- **정직한 surrogate 설계의 모범**: 모듈 docstring이 "둘 다 surrogate(임상 t½ 아님),
  앙상블은 강건성 향상을 노린 것이며 추가 검증 권장"임을 명시(`halflife_ensemble.py:9`).
  과대주장 없이 한계를 코드 레벨에서 노출한다.
- **이질적 추정기 앙상블**: 기전 기반 휴리스틱(해석 가능, 절대 스케일)과 데이터 기반
  RF(일반화)를 log 스케일에서 결합해 단일 모델의 편향을 상호 보완
  (`halflife_ensemble.py:3-7`).
- **순위 보존 검증 방법론**: 절대 캘리브레이션 없이도 Spearman으로 의사결정에 필요한
  "상대 우선순위"만 검증하는 접근(§특별 섹션)은 데이터가 희소한 펩타이드 t½ 도메인에서
  실용적이다.

---

## ⑥ 사용법

```python
from pyrosetta_flow.halflife_ensemble import ensemble_halflife

res = ensemble_halflife("AGCKNFFWKTFTSC", modifications=[], rec={})
# {
#   "half_life_h": 0.04,            # 표시용(휴리스틱; 절대값 신뢰 LOW)
#   "half_life_heuristic_h": 0.04,
#   "half_life_rf_h": <float|None>, # RF 미가용 시 None
#   "stability_norm": <0~1>,        # 랭킹용 (이것이 핵심 출력)
#   "halflife_source": "ensemble" | "heuristic" | "rf" | "none",
# }
```

- `modifications`: `["fatty_acid", "d_amino_acid", "cyclization", ...]` (휴리스틱에 전달)
- `rec`: RF feature용 메타(`chiral`, `lin_cyc`, `chem_mod`, `nter`, `cter`) (`features.py:38-48`)
- **올바른 사용**: `stability_norm`·`half_life_h`를 **후보 간 비교/랭킹**에만 사용
- **금지**: `half_life_h`를 임상 반감기 절대값으로 보고/해석 (`step08_stability.py:536-538`)
- 직접 휴리스틱: `from AG_src.pipeline.step08_stability import predict_half_life`
- 게이트: `apply_stability_gate(results, gate_mode="relative_percentile", top_fraction=0.5)`
  권장 (절대값 미신뢰 시 더 정직, `step08_stability.py:543-544`)

벤치마크 실행:
```bash
pytest AG_src/tests/test_halflife_benchmark.py -v
```

---

## ⑦ 필요 이유

방사성의약품 후보의 혈중 안정성은 핵심 약동학 속성이지만, **신규 펩타이드 변이체의
실측 t½ 데이터는 존재하지 않는다.** 무한 발굴 엔진은 수많은 변이체를 빠르게(μs~ms)
서열만으로 우선순위화해야 하므로 (`multiobjective.py:12, 16`):

1. 실측이 불가능한 안정성 축에 대해 **계산 가능한 surrogate**를 제공하고,
2. 단일 추정기의 편향을 앙상블로 완화하며,
3. 절대값을 주장하지 않고 **상대 순위만** 다목적 최적화(stability 가중 0.20,
   `multiobjective.py:277`)에 기여시켜 환각(임상 수치 위장)을 차단한다.

이는 프로젝트의 "계산 불가능을 계산 가능한 척하지 않는다"(VR-cycle-09/H-06) 정직성
원칙의 구체적 구현이다.

---

## ★ 특별 섹션: 간접 검증 방식 (Spearman 상대순위 벤치마크)

이 기능의 신뢰는 **절대 캘리브레이션 없이** 확보된다. 핵심 메커니즘은 다음과 같다.

### 왜 절대값이 아니라 순위인가
펩타이드 t½ 절대값을 정확히 예측하려면 in-vitro 혈청 어세이로 모델을 보정해야 하는데,
프로젝트는 이를 **미시행**으로 명시한다(`step08_stability.py:536-538`). 대신 의사결정에
실제로 필요한 것은 "A가 B보다 더 안정한가"라는 **상대 순위**뿐이다. 다목적 최적화는
순위 기반(pareto + 가중 스칼라)이므로, 절대 스케일이 틀려도 **단조 변환 하에 순위가
보존되면** 의사결정 품질은 유지된다.

### 검증 데이터셋 (문헌 t½ 8종)
`test_halflife_benchmark.py:21-30` 의 `BENCH`는 문헌 혈중 t½가 공개된 펩타이드로 구성:

| 펩타이드 | modification | 문헌 t½(h) |
|----------|-------------|-----------|
| GLP-1 | — | 0.03 |
| SST-14 | — | 0.05 |
| Exenatide | — | 2.4 |
| Octreotide | d_amino_acid | 1.7 |
| Desmopressin | d_amino_acid | 3.0 |
| Leuprolide | d_amino_acid | 3.0 |
| Liraglutide | fatty_acid | 13.0 |
| Semaglutide | fatty_acid | 168.0 |

스케일이 0.03h ~ 168h(4 orders of magnitude)에 걸쳐 있어 순위 변별이 의미를 가진다.

### Spearman 순위상관으로 검증
`_spearman`(`test_halflife_benchmark.py:33-42`)은 예측값과 문헌값을 각각 **순위로 변환**한
뒤 상관을 계산한다. 절대 오차가 아니라 **순위 일치도**만 보므로, 단조 관계만 맞으면 통과한다.
고정된 임계(회귀 테스트):

- **전체 Spearman ≥ 0.7** (`test_halflife_benchmark.py:51-55`) — 실측 ρ≈0.86 (docstring 기준)
- **plain(modification 無) Spearman ≥ 0.3** (`test_halflife_benchmark.py:58-62`) —
  재보정 전 −0.5(역전)에서 회복
- **SST-14 < 1h** (`test_halflife_benchmark.py:45-48`) — 과대예측 방지
- **fatty_acid > 10× plain** (`test_halflife_benchmark.py:65-69`) — 알부민 결합 효과 방향성
- **변이체 변별**: 말단 변이가 native보다 짧음 (`test_halflife_benchmark.py:72-76`)

### RF 모델의 별도 검증
RF는 **in-domain CV Spearman 0.78 / R²log 0.64** 로 보고됨(`halflife_ensemble.py:5`).
이 역시 절대 정확도(R²)보다 순위(Spearman)를 1차 지표로 삼는다.

### 정직성의 핵심 (절대 LOW / 상대 MED)
- **절대값 = LOW**: 캘리브레이션 미시행(`step08_stability.py:536-538`), 계수 출처 미확보
  (VR-S5-01, `step08_stability.py:46-48`). `half_life_h`를 임상 수치로 해석 금지.
- **상대순위 = MED**: 8종 문헌 벤치마크 Spearman ρ≥0.86(heuristic)/0.78(RF CV)로
  **순위 경향성만** 회귀 테스트에 고정. 충분히 검증된 절대 진실(HIGH)이 아니라,
  의사결정에 필요한 범위에서만 "순위가 맞다"는 간접 증거.

요약하면, 이 기능은 **"틀린 절대값을 정직하게 인정하면서, 맞는 상대 순위만 검증하여
필요한 신뢰만 확보"** 하는 간접 검증의 대표 사례다.

---

## 검증 인용 목록 (file_path:line)

**앙상블 결합·정규화·fallback**
- `pyrosetta_flow/halflife_ensemble.py:9` — surrogate·추가검증 권장 면책
- `pyrosetta_flow/halflife_ensemble.py:19` — log10 정규화 범위 `[0.02, 200]`
- `pyrosetta_flow/halflife_ensemble.py:35-46` — RF joblib 로드 + 로드 실패 fallback
- `pyrosetta_flow/halflife_ensemble.py:49-63` — RF 추론(`np.expm1`), feat_cols 정렬
- `pyrosetta_flow/halflife_ensemble.py:66-72` — 휴리스틱 호출 + 안전 처리
- `pyrosetta_flow/halflife_ensemble.py:75-78` — log 정규화 함수
- `pyrosetta_flow/halflife_ensemble.py:81-108` — 앙상블 결합·source 결정·표시값

**RF feature**
- `pyrosetta_flow/halflife_model/features.py:1-3` — feat_cols 학습 일치 요구
- `pyrosetta_flow/halflife_model/features.py:11-49` — featurize 전체

**휴리스틱 (log-multiplicative)**
- `AG_src/pipeline/step08_stability.py:42-48` — VR-S5-01 출처 면책
- `AG_src/pipeline/step08_stability.py:50-71` — `_PROTEASE_VULNERABILITY`
- `AG_src/pipeline/step08_stability.py:105-121` — 재보정 사유 + log-mult 계수
- `AG_src/pipeline/step08_stability.py:225-275` — `predict_half_life` log10 모델
- `AG_src/pipeline/step08_stability.py:526-569` — `apply_stability_gate` (2026-06-17 정정)
- `AG_src/pipeline/step08_stability.py:534-538` — 절대값 캘리브레이션 미시행 명시

**Spearman 상대순위 벤치마크**
- `AG_src/tests/test_halflife_benchmark.py:4-5` — 재보정 전 −0.5 역전 기록
- `AG_src/tests/test_halflife_benchmark.py:21-30` — BENCH 8종 문헌 t½
- `AG_src/tests/test_halflife_benchmark.py:33-42` — `_spearman` 구현
- `AG_src/tests/test_halflife_benchmark.py:45-76` — 5개 순위/방향성 테스트
- `AG_src/tests/test_halflife_benchmark.py:79-88` — relative_percentile 게이트 테스트

**downstream 소비**
- `pyrosetta_flow/multiobjective.py:115-132` — `ensemble_halflife` 호출 + fallback
- `pyrosetta_flow/multiobjective.py:263-264, 277, 297-302` — stability 가중 0.20
- `pyrosetta_flow/scoring_pipeline.py:77, 206` — surrogate 면책 + `stability_norm` 소비

---
*작성: 2026-06-17 · 신뢰등급 절대값 LOW / 상대순위 MED · 읽기전용 분석 (코드 미수정)*
