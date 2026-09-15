# 기능 분석 보고서 — ADMET Surrogate + pepADMET 독성 (hc50 게이트)

> 대상 모듈: `pyrosetta_flow/multiobjective.py`, `pyrosetta_flow/pepadmet_runner.py`,
> `pyrosetta_flow/pepadmet_toxicity.py`, `pyrosetta_flow/pepadmet_infer_script.py`, `backend/admet.py`
>
> **인용 규칙**: 모든 동작 주장은 `file_path:line` 으로 근거를 단다. 코드에서 직접 확인되지 않은 항목은
> "미검증"으로 명시한다. 본 문서는 읽기 전용 분석이며 코드를 수정하지 않았다.
>
> **단위 면책(가장 중요)**: pepADMET HC50 출력의 **절대 단위(μg/mL vs μM)는 원논문 대비 미검증**이다
> (`pyrosetta_flow/pepadmet_toxicity.py:24-30`, `pyrosetta_flow/pepadmet_toxicity.py:40`).
> 따라서 hc50 절대값 해석은 금지되며, **native SST-14 기준선 대비 상대 비교 전용**으로만 사용된다.

---

## ① 동작 원리

### 1-A. ADMET surrogate (물성 기반 합리성 점수)

두 갈래의 ADMET 계산이 공존한다.

**(a) `backend/admet.py` — 서열 전용 순수 파이썬 물성/약물유사성** (외부 의존 없음, `backend/admet.py:7`)
- 분자량(monoisotopic residue weight 합 + H2O): `backend/admet.py:74`
- pH 7.4 net charge (K/R = +1, D/E = −1, H = +0.1, N항 +1·C항 −1): `backend/admet.py:77-89`
- H-bond donor/acceptor (사이드체인 + backbone): `backend/admet.py:91-97`
- Kyte-Doolittle 평균 소수성 + amphipathicity(분산): `backend/admet.py:99-107`
- **druglikeness_score (0~100, 25점×4 규칙)**: MW 1200~2000 / |charge|≤3 / 소수성 [−2,+1] /
  3연속 동일 잔기 없음 — `backend/admet.py:113-136`
- 신장독성(PRRT renal retention) 위험: cationic 잔기(K/R/H) 기반 점수,
  `min(100, (n_lys+n_arg)*20 + max(0,net_charge)*15)` — `backend/admet.py:188-200`.
  근거로 DOTATATE(~25, Low)를 레퍼런스로 둠 (`backend/admet.py:155-156`).

**(b) `multiobjective.admet_reasonableness()` — 다목적 랭킹용 0~1 합리성 점수** (`pyrosetta_flow/multiobjective.py:64-90`)
- 입력: `instability_index`, `gravy`, `boman_index`, `pi` 4개 물성 (`pyrosetta_flow/multiobjective.py:69-72`).
  이 물성은 `AG_src.pipeline.pharma_properties.PharmaProperties` 가 계산 (`pyrosetta_flow/multiobjective.py:46`,
  `pyrosetta_flow/multiobjective.py:135-147`). Cys3-Cys14 SS bond 는 pI 계산 시 해당 Cys 를 제외
  (`pyrosetta_flow/multiobjective.py:138-140`, `pyrosetta_flow/multiobjective.py:146`).
- **부분점수 변환 + 가중 평균**:
  - Instability: <40 만점, 40~80 선형 감점, >80 0점 (`pyrosetta_flow/multiobjective.py:75`)
  - GRAVY: ≤−1 만점(친수), 0→0.5, ≥1 0점 (`pyrosetta_flow/multiobjective.py:77`)
  - Boman: <1 결합경향 약함(절반 스케일), 1~4 선형, >4 만점 (`pyrosetta_flow/multiobjective.py:79-84`)
  - pI: 6~8 만점, 멀어질수록 선형 감점 (`pyrosetta_flow/multiobjective.py:86`)
  - **가중**: `0.35*II + 0.30*GRAVY + 0.15*Boman + 0.20*pI` — 안정성·용해를 더 중시
    (`pyrosetta_flow/multiobjective.py:88-89`). 임계값/범위는 모두 문헌 휴리스틱
    (`pyrosetta_flow/multiobjective.py:57-62`).
- 이 점수는 `cheap_objectives()` 의 `admet_score` 가 되고 (`pyrosetta_flow/multiobjective.py:154`),
  `enrich_candidates()` 가 후보의 `druggability` 키로 매핑 (`pyrosetta_flow/multiobjective.py:264-265`).

> **honest disclaimer (VR-cycle-09 / H-06)**: `half_life_h`, `admet_score` 는 **랭킹용 surrogate** 이며
> 임상 반감기·임상 ADMET 수치가 아니다 (`pyrosetta_flow/multiobjective.py:15-18`). in-vitro assay 검증 없음.

### 1-B. pepADMET GNN 독성 추론

**모델**: `toxicity_early_stop.pth` — MLR-GAT (RGCN + MLP + Attention), JCIM 2026, 66, 936-946
(`pyrosetta_flow/pepadmet_toxicity.py:4-7`). 4개 멀티태스크 출력:
binary toxicity / 6-class type / 4-class neurotoxicity / HC50 (`pyrosetta_flow/pepadmet_infer_script.py:110-112`).

**실행 격리(conda subprocess)**: bio-tools env 의 runner 가 pepadmet env(DGL 0.4.3)를 subprocess 로 호출
(`pyrosetta_flow/pepadmet_runner.py:1-5`). `conda run --no-capture-output -n pepadmet python3 pepadmet_infer_script.py <json>`
(`pyrosetta_flow/pepadmet_runner.py:77-81`), timeout 120s (`pyrosetta_flow/pepadmet_runner.py:80`).
inline `-c` 대신 파일 직접 호출로 버퍼링 hang 방지 (`pyrosetta_flow/pepadmet_runner.py:76`,
`pyrosetta_flow/pepadmet_runner.py:32`). stdout 마지막 JSON 라인 파싱 (`pyrosetta_flow/pepadmet_runner.py:88-92`).

**그래프 빌드**: SMILES → RDKit Mol → DGL 그래프 (`pyrosetta_flow/pepadmet_infer_script.py:74-78`,
`51-71`). SMILES 파싱 실패 시 **선형 폴백** `Chem.MolFromSequence` (`pyrosetta_flow/pepadmet_infer_script.py:81-102`,
`131-135`) — 이 경우 `graph_note="linear_sequence_fallback"` 가 기록되고
(cyclic/SS bond 구조 소실), descriptor 는 2133차원 텐서(실패 시 zero) (`pyrosetta_flow/pepadmet_infer_script.py:23-48`).

**SMILES 변환**: runner 가 `smiles_converter.sequence_to_smiles()` 로 Cys3-Cys14 이황화 사이클릭 SMILES 생성
시도 (`pyrosetta_flow/pepadmet_runner.py:61-66`, `pyrosetta_flow/smiles_converter.py:1-5`,
`pyrosetta_flow/smiles_converter.py:55-60`).

**출력 행**: `binary_toxicity`(sigmoid), `is_toxic`(>0.5), `toxicity_type`(6-class argmax),
`neurotoxicity_type`(4-class argmax), `hc50`(task_3 회귀) — `pyrosetta_flow/pepadmet_infer_script.py:151-168`.
정규화 레이어가 `hc50_unit`, `hc50_reliable` 플래그 부착 (`pyrosetta_flow/pepadmet_toxicity.py:40-43`).

### 1-C. hc50 native±5 밴드 게이트 (핵심)

`apply_toxicity_to_extra()` 가 pepADMET 결과를 extra_scores 에 기록하고 admet_score 에 페널티를 반영한다
(`pyrosetta_flow/multiobjective.py:206-241`). 로직:

1. **fail-closed**: `available` 가 아니면 admet_score 를 건드리지 않음(가짜 안전판정 금지)
   (`pyrosetta_flow/multiobjective.py:217-218`).
2. **binary 는 기록만, 게이트 미사용** — `pepadmet_toxic`/`binary_toxicity` 기록되지만 게이트에 쓰이지 않음
   (`pyrosetta_flow/multiobjective.py:219-221`, 주석 `pyrosetta_flow/multiobjective.py:211-212`).
3. **신뢰도 게이트(VR-A2, 2026-06-17)**: `graph_note == "linear_sequence_fallback"` 이면
   cyclic 구조 소실로 hc50 신뢰 불가 → `hc50_reliable=False` 기록 후 **게이트 skip**(fail-open: 가짜 독성판정 방지)
   (`pyrosetta_flow/multiobjective.py:225-230`).
4. **native 대비 상대(home-advantage) 게이트**:
   - native hc50 기준선 로드 (`native_toxicity_baseline.json`, hc50 = **−55.6816**, `_native_hc50_baseline()`
     `pyrosetta_flow/multiobjective.py:171-185`).
   - `delta = hc50 − native_hc50` (`pyrosetta_flow/multiobjective.py:232`), >0 안전 / <0 독성↑.
   - **native ±5 밴드(`_HC50_NATIVE_TOLERANCE = 5.0`)는 "동급"으로 무페널티**
     (`pyrosetta_flow/multiobjective.py:166`, `pyrosetta_flow/multiobjective.py:234`).
   - 밴드 초과 독성일 때만 초과분에 선형 비례 페널티,
     `factor = max(0.4, 1 − excess/200)` (`_TOXIC_ADMET_PENALTY=0.4`,
     `_HC50_PENALTY_SCALE=200.0`) 로 admet_score 곱셈 감점
     (`pyrosetta_flow/multiobjective.py:163`, `167`, `236-241`).
   - `hc50_vs_native`, `more_toxic_than_native` 기록 (`pyrosetta_flow/multiobjective.py:233-235`).

이 게이트 산출물은 글로벌 리더보드 통과 판정에 직접 쓰인다 —
`count_passing()` 은 독성을 **`more_toxic_than_native is False`(명시적 측정)만 통과**로 인정
(`pyrosetta_flow/global_leaderboard.py:167-176`, 필드 적재 `pyrosetta_flow/global_leaderboard.py:104-106`).
파이프라인 배선은 `scoring_pipeline.py:98-117` (Step 0.5, 배치 추론 후 페널티 반영, 실패 시 graceful skip).

---

## ② 영향 — binary → hc50 교체로 "옥시토신/native 오판" 해결

**경위(정확히)**:

- **문제**: pepADMET `is_toxic`(binary)가 **비변별적** — 임상 안전 펩타이드(oxytocin),
  native SST-14, 벌독(melittin) 을 **전부 toxic=True** 로 판정 (baseline note,
  `data/somatostatin_receptor/curated/native_toxicity_baseline.json` note;
  코드 주석 `pyrosetta_flow/multiobjective.py:164`, `pyrosetta_flow/multiobjective.py:211`).
  이 binary 게이트가 in-loop 선택성 run 에서 "측정 후보 전원 toxic → admet×0.4 → full gate 탈락"
  의 원인이었다(메모리 sstr2-selectivity-goal 기록).
- **관찰된 변별력**: 동일 모델의 **hc50 연속값은 생물학적으로 정확한 순서** —
  oxytocin(−14.5, 안전) < native(−55.7) < melittin(−292.7, 독성)
  (baseline json note 의 대조군 수치).
- **교체**: 게이트를 binary 에서 **hc50 − native_hc50 상대(home-advantage)** 로 전환
  (`pyrosetta_flow/multiobjective.py:211-241`). Δmargin(선택성) 과 동일 철학 —
  "native 를 기준선으로 두고 그보다 나쁜 후보만 벌점".
- **효과**: native 동급(±5) 후보는 is_toxic=True 라도 무페널티가 되어
  (`test_apply_toxicity_native_comparable_no_penalty` `pyrosetta_flow/tests/test_multiobjective.py:118-126`),
  native 급 우수 후보가 admet×0.4 로 **부당 탈락하던 문제가 해소**되었다(에이전트 랭킹 정상화).
  재평가 시 측정 8후보 중 7개가 "독성≤native"(무페널티)로 정상화됨(메모리 기록).

> **주의(미검증)**: 위 oxytocin/native/melittin 대조 수치는 `native_toxicity_baseline.json` 의 note 와
> 메모리에 기록된 값이며, 본 분석에서 모델을 재실행해 재현하지는 않았다(읽기 전용). 따라서
> "binary 비변별성"과 "hc50 변별력"은 **기록된 측정 결과에 근거한 주장**이지 본 세션에서 재검증된 것은 아니다.

---

## ③ 관련 Action Item

| 항목 | 내용 | 상태/근거 |
|------|------|-----------|
| **B (pepADMET 통합)** | pepADMET GNN(toxicity_early_stop.pth)을 pepadmet conda env subprocess 배치 추론으로 다목적 랭킹에 연결 | 구현 완료. `scoring_pipeline.py:98-117`(Step 0.5), `pyrosetta_flow/multiobjective.py:158-203`, runner/infer 전체. 백엔드 API 도 `merge_pepadmet_into_admet_results` 로 연결 (`backend/admet.py:242-276`, `backend/routers/admet.py:17-21`) |
| **hc50 게이트 수정 (2026-06-10)** | binary→hc50 native 상대 게이트로 교체 | 완료. `pyrosetta_flow/multiobjective.py:206-241`, baseline json, 회귀 테스트 갱신/신설 (`pyrosetta_flow/tests/test_multiobjective.py:93-141`) |
| **hc50 신뢰도 게이트 (VR-A2, 2026-06-17)** | SMILES 선형 폴백(cyclic 소실) 시 hc50 신뢰불가 → 게이트 skip | 완료. `pyrosetta_flow/multiobjective.py:225-230`, `pyrosetta_flow/pepadmet_toxicity.py:24-43`, 테스트 `pyrosetta_flow/tests/test_multiobjective.py:107-115` |
| **HC50 단위 면책 (VR-A2)** | hc50 절대단위 미검증 명시, 상대비교 전용 강제 | 완료(문서화). `pyrosetta_flow/pepadmet_toxicity.py:24-30`, `pyrosetta_flow/pepadmet_toxicity.py:40` |

**남은 과제(코드 기준 미해결)**:
- hc50 **절대 단위 검증**(μg/mL vs μM, 원논문 task_3 타깃과 대조) — 현재 미검증 면책으로만 회피
  (`pyrosetta_flow/pepadmet_toxicity.py:24-30`).
- SMILES 변환 성공률 — 사이클릭 SMILES 실패 시 선형 폴백되면 hc50 게이트가 통째로 skip 되어
  독성 신호가 사라짐 (`pyrosetta_flow/multiobjective.py:225-230`). 폴백 빈도/영향은 미측정.

---

## ④ 완성도 평가

**약 75%**

근거(가점):
- 데이터 흐름 end-to-end 연결됨: cheap_objectives → enrich → pepADMET 추론 → hc50 게이트 →
  리더보드 통과 판정 (`pyrosetta_flow/multiobjective.py:101-266`, `scoring_pipeline.py:98-117`,
  `global_leaderboard.py:167-176`).
- 견고한 실패 처리: import/subprocess/timeout/JSON 파싱 모두 graceful, fail-closed(가짜 안전) vs
  fail-open(신뢰불가 hc50) 구분 명확 (`pyrosetta_flow/pepadmet_runner.py:54-99`,
  `pyrosetta_flow/multiobjective.py:217-230`).
- 회귀 테스트 5종이 게이트 분기를 모두 커버(페널티/폴백 skip/동급 무페널티/비독성/unavailable)
  (`pyrosetta_flow/tests/test_multiobjective.py:93-141`).
- 정직성 메커니즘: surrogate disclaimer, 단위 면책, home-advantage 보정.

감점:
- **hc50 절대 단위 미검증**(−), 상대비교로만 우회 (`pyrosetta_flow/pepadmet_toxicity.py:24-30`).
- **선형 폴백 시 독성 게이트 통째 skip** → cyclic 펩타이드인데 SMILES 변환이 자주 실패하면
  독성 평가가 무력화될 수 있음(빈도 미측정) (`pyrosetta_flow/multiobjective.py:225-230`).
- binary/type/neurotoxicity 출력은 **기록만 되고 게이트에 미반영** — 4개 멀티태스크 중 1개(hc50)만 의사결정에 사용
  (`pyrosetta_flow/multiobjective.py:219-221`).
- `backend/admet.py` 신장독성 점수의 계수(20/15)는 단일 레퍼런스(DOTATATE) 외 보정 근거 부재
  (`backend/admet.py:188`).

---

## ⑤ 학술적 가치

- **home-advantage 상대 게이트**: 절대값이 비변별/미검증인 모델 출력을 폐기하지 않고
  "native 기준선 대비 Δ" 로 재구성해 변별력을 회복한 설계는 일반화 가치가 있다 — 선택성 Δmargin 과
  동일 패턴으로 대칭(`pyrosetta_flow/multiobjective.py:211-214`).
- **다중 신뢰 게이트**: available(fail-closed) → graph_note(fail-open) → native-band 의 3층 게이트로
  "환각된 안전/독성 판정"을 양방향 차단 (`pyrosetta_flow/multiobjective.py:217-241`).
- **정직한 한계 노출**: surrogate disclaimer 와 단위 면책을 코드 주석으로 명문화 — 재현성·신뢰성 보고에 모범
  (`pyrosetta_flow/multiobjective.py:15-18`, `pyrosetta_flow/pepadmet_toxicity.py:24-30`).
- 한계: 단일 모델(pepADMET) 의존, 절대 단위 미검증, in-vitro/in-vivo cross-validation 부재 →
  현재로선 **랭킹 보조 신호**로서의 가치이지 임상 예측 가치는 주장 불가.

---

## ⑥ 사용법

**라이브러리(다목적 파이프라인 내부)** — 자동 호출됨:
```python
from pyrosetta_flow.multiobjective import (
    cheap_objectives, enrich_candidates,
    predict_toxicity_for_sequences, apply_toxicity_to_extra,
)
enrich_candidates(candidates)                 # admet_score/druggability 채움
tox = predict_toxicity_for_sequences([seq])   # pepADMET 배치 추론
apply_toxicity_to_extra(cand["extra_scores"], tox[seq])  # hc50 게이트 적용
```
(`pyrosetta_flow/multiobjective.py:188-241`)

**단건 독성 예측**:
```python
from pyrosetta_flow.pepadmet_toxicity import predict_toxicity
r = predict_toxicity("AGCKNFFWKTFTSC")        # smiles 생략 시 자동 변환
```
(`pyrosetta_flow/pepadmet_toxicity.py:59-80`)

**백엔드 REST API**:
- `GET /admet/{sequence}` — 물성+신장독성+pepADMET 병합 (`backend/routers/admet.py:24-26`)
- `POST /admet/batch` — 배치 (`backend/routers/admet.py:29-30`)

**환경 변수**:
- `SKIP_PEPADMET=1` — pepADMET 호출 생략(테스트/빠른 응답) (`backend/admet.py:250-251`)
- `PEPADMET_REPO` — pepADMET repo 경로 오버라이드 (`pyrosetta_flow/pepadmet_runner.py:26-30`)

**필수 전제**: pepadmet conda env + `local_models/pepadmet/repo` + `model/toxicity_early_stop.pth`
존재 (`pyrosetta_flow/pepadmet_runner.py:19-23`, `pyrosetta_flow/pepadmet_infer_script.py:105`).
미존재 시 `available=False` 로 graceful skip.

---

## ⑦ 필요 이유

1. **선택성만으로는 부족**: SSTR2 선택성이 우수해도 독성·짧은 반감기·낮은 용해 후보는 약물이 될 수 없다.
   ADMET surrogate(`admet_score`)와 pepADMET 독성이 다목적 게이트의 일부로 안전성·약물유사성을 랭킹에 주입한다
   (`pyrosetta_flow/multiobjective.py:1-9`, ObjectiveWeights.admet=0.15 `pyrosetta_flow/multiobjective.py:278`).
2. **방사성의약품(PRRT) 특이 위험**: 신장 재흡수에 의한 신장독성은 PRRT 의 용량 제한 독성 —
   cationic 함량 기반 위험 점수가 이를 조기 경고 (`backend/admet.py:150-217`).
3. **환각/오판 차단**: binary 비변별성으로 인한 native/oxytocin 오판이 우수 후보를 부당 탈락시키던 문제를
   hc50 home-advantage 게이트가 교정해, 에이전트의 자율 랭킹이 정상 작동하도록 한다
   (§② 참조, `pyrosetta_flow/multiobjective.py:206-241`).

---

## 검증 인용 목록

- 동작 확인(직접 코드):
  - `pyrosetta_flow/multiobjective.py:64-90` — admet_reasonableness 가중/임계값
  - `pyrosetta_flow/multiobjective.py:101-155` — cheap_objectives, PharmaProperties, SS bond pI 처리
  - `pyrosetta_flow/multiobjective.py:163-185` — 페널티 상수, native_hc50 기준선 로드
  - `pyrosetta_flow/multiobjective.py:206-241` — apply_toxicity_to_extra (binary 미사용·hc50 게이트·신뢰도 skip)
  - `pyrosetta_flow/pepadmet_runner.py:54-99` — subprocess/conda 호출, 실패 처리
  - `pyrosetta_flow/pepadmet_infer_script.py:105-170` — MGA 모델 로드, 4태스크 출력, 선형 폴백
  - `pyrosetta_flow/pepadmet_toxicity.py:22-56` — 정규화, hc50_unit/hc50_reliable
  - `backend/admet.py:54-228` — 물성/약물유사성/신장독성 계산
  - `backend/admet.py:242-276`, `backend/routers/admet.py:13-30` — pepADMET 병합·REST
  - `pyrosetta_flow/global_leaderboard.py:104-106`, `167-176` — 독성 통과 판정 필드/로직
  - `pyrosetta_flow/scoring_pipeline.py:98-117` — 파이프라인 Step 0.5 배선
  - `data/somatostatin_receptor/curated/native_toxicity_baseline.json` — hc50=−55.6816, 대조군 note
  - 테스트: `pyrosetta_flow/tests/test_multiobjective.py:39-141`,
    `pyrosetta_flow/tests/test_continuous_discovery.py:28-45`

- **미검증 / 측정 미재현(읽기 전용 한계)**:
  - **HC50 절대 단위(μg/mL vs μM)** — 원논문 대비 미검증, 상대비교 전용 (`pyrosetta_flow/pepadmet_toxicity.py:24-30`, `40`)
  - oxytocin(−14.5)/native(−55.7)/melittin(−292.7) 대조 수치 — baseline json note·메모리 기록 근거,
    본 세션 모델 재실행 미수행
  - "binary 전원 toxic" 비변별성 — 기록된 측정 결과 근거, 본 세션 재현 안 함
  - SMILES 사이클릭 변환 성공률 / 선형 폴백 발생 빈도 — 미측정
  - admet_reasonableness 임계값·가중(0.35/0.30/0.15/0.20), 신장독성 계수(20/15) 의 문헌 정량 보정 근거 —
    코드 주석의 휴리스틱 표기 외 외부 검증 미확인
