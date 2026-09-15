# 선택성 — in-loop 게이트 + Δmargin(home-advantage) 기능 분석

> 분석 루트: `AgenticAI4SCIENCE_pyrosetta_track/repos/ai4sci-kaeri`
> 대상 파일: `pyrosetta_flow/selectivity_loop.py`, `pyrosetta_flow/multiobjective.py`,
> `data/somatostatin_receptor/curated/native_selectivity_baseline.json`
> 모든 수치는 소스/측정 직독 인용. 미확인 항목은 "미검증"으로 명시.

---

## ① 동작 원리

### 1.1 selectivity_margin (절대 선택성)

한 후보의 SSTR2 정밀화 복합체를 off-target 수용체 4종에 도킹하여 결합 강도를 비교한다.

- off-target 수용체 정의: `multiobjective.py:333-338` `DEFAULT_OFFTARGET_RECEPTORS`
  = SSTR1 / SSTR3 / SSTR4 / SSTR5 (curated 단일체인 PDB).
- **동일 프로토콜**: 2026-06-10부터 on-target SSTR2도 off-target과 **동일한
  transplant + pre-relax 프로토콜**로 재측정한다. 근거 주석 `multiobjective.py:401-403`:
  "SSTR2 도 off-target 과 동일 transplant+pre-relax 로 재서 margin 편향 제거(이전 아티팩트 수정)."
  수용체 dict에 SSTR2를 명시적으로 추가하는 코드는 `multiobjective.py:405-407`.
- 병렬 실행: `ThreadPoolExecutor`(`multiobjective.py:404`, `multiobjective.py:424-425`),
  max_workers = min(6, 수용체 수). 주석상 순차 ~25분 → 병렬 ~6분/후보(`multiobjective.py:403`).
  ※ 실측 시간은 본 분석에서 미검증(주석 인용).
- margin 정의: `multiobjective.py:434-435`
  - `worst = min(offtarget_ddg.values())` — 가장 강한(가장 낮은 ddG) off-target.
  - `margin = worst - baseline` — 양수 = SSTR2에 더 강하게 결합 = 선택적.
  - 즉 **margin = min(offtarget ddG) − sstr2 ddG**.
- baseline 선택: 동일 프로토콜 SSTR2 값 우선, 실패 시 루프 ddG 폴백(`multiobjective.py:432-433`).
- fail-closed: NaN 도킹 결과는 None 처리(`multiobjective.py:419`), off-target이 하나도
  없으면 margin=None 반환(`multiobjective.py:428-430`).

### 1.2 Δmargin = home-advantage 보정

절대 margin은 편향되어 있다. 후보 복합체가 SSTR2 source 구조에서 유래하므로
native SST-14조차 동일 프로토콜에서 양(+)의 margin을 낸다(`multiobjective.py:436-437`).
따라서 native 기준선을 빼서 보정한다.

- `delta_margin = round(margin - nat_margin, 4)` (`multiobjective.py:438-439`).
- `more_selective_than_native = (delta_margin is not None and delta_margin > 0)` (`multiobjective.py:448`).
- native 기준선 로드: `_native_selectivity_baseline()` (`multiobjective.py:343-358`),
  `native_selectivity_baseline.json`의 `margin` 필드를 읽어 캐시.

#### native baseline 실측 (`native_selectivity_baseline.json:1`)

```json
{"sstr2": -61.3475,
 "offtarget": {"SSTR1": -40.1563, "SSTR3": -39.0837, "SSTR4": -33.984, "SSTR5": -47.9815},
 "margin": 13.37}
```

- min(offtarget) = SSTR5 = −47.9815 (가장 강한 off-target).
- 검산: −47.9815 − (−61.3475) = **13.366 ≈ 13.37** → 저장된 margin과 일치(검증 완료).
- 즉 native의 home-advantage는 **+13.37**이며, Δmargin > 0 이려면 후보의 절대 margin이 13.37을 넘어야 한다.

### 1.3 조건부 in-loop 게이트 (ddG 프록시)

off-target 도킹은 비싸서(후보×5수용체) 매 iteration 전체 후보에 못 돌린다
(`selectivity_loop.py:3-6`). 해결책은 **ddG(루프 내 실측, 강한 신호)를 유망도 프록시로**
사용하여 도킹 대상을 제한하는 것.

- 게이트 로직 `SelectivityLeaderboard.should_screen()` (`selectivity_loop.py:48-56`):
  1. 이미 도킹한 서열이면 skip (`selectivity_loop.py:50-51`).
  2. ddG가 cutoff(−10.0)보다 약하면 skip (`selectivity_loop.py:52-53`).
  3. 리더보드 미충원이면 도킹 (`selectivity_loop.py:54-55`).
  4. 충원 시 기존 top-K 최약체보다 ddG가 강하면 도킹 (`selectivity_loop.py:56`,
     `worst_ddg()`=가장 높은 ddG, `selectivity_loop.py:43-46`).
- 적격 필터 `screen_iteration_candidates()` (`selectivity_loop.py:96-118`):
  fail 없음 + clash ≤ clash_max + **이황화결합(Cys 위치) 보존**(`_disulfide_ok`,
  `selectivity_loop.py:99-100`) + PDB 존재.
- 도킹 상한: iteration당 `max_screen_per_iter`개(기본 2, `selectivity_loop.py:124`),
  ddG 강한 순으로 적용(`selectivity_loop.py:121`).
- 결과 기록: `c.extra_scores`에 selectivity_margin / delta_margin / offtarget_ddg /
  sstr2_ddg_sameprotocol 기록(`selectivity_loop.py:134-137`).

### 1.4 epoch 간 학습 (warm-start)

무한 발굴 엔진에서 run 간 학습은 글로벌 리더보드로 이어진다.

- `seed_from_global()` (`selectivity_loop.py:23-41`): 글로벌 리더보드에서
  ① screened_seqs(역대 도킹 서열) → 재도킹 회피, ② 역대 top-K → in-loop 게이트 기준선 적재.
  결과적으로 worst_ddg 임계가 끌어올려져 "역대 best보다 유망한 후보만" 도킹한다.
- 호출 위치: `runner.py:495-501`(SelectivityLeaderboard 생성 + seed),
  `runner.py:888-897`(iteration마다 screen_iteration_candidates 호출).

---

## ② 영향

### 2.1 프로토콜 편향 수정 (전/후)

- **수정 전(~2026-06-09)**: SSTR2는 루프 내 ddG(별도 프로토콜), off-target은
  transplant+pre-relax. 프로토콜이 다르면 margin이 프로토콜 차이를 선택성으로
  오인하는 아티팩트가 발생(`multiobjective.py:402` 주석 "이전 아티팩트 수정").
- **수정 후(2026-06-10)**: SSTR2도 동일 transplant+pre-relax로 재측정
  (`multiobjective.py:401-407, 427, 433`). margin이 프로토콜 차이가 아닌
  순수 수용체 친화도 차이를 반영하게 됨.

### 2.2 home-advantage 보정의 의미

source 구조 유래 편향(+13.37)을 빼므로, Δmargin>0은 "native SST-14를 초과하는
선택성"이라는 **상대적·정직한** 신호가 된다(`multiobjective.py:436-437`,
`selectivity_loop.py:7`). 절대 margin만 보면 거의 모든 후보가 양수라 변별력이 없다.

### 2.3 비용/탐색 영향

ddG 프록시 게이트로 도킹 호출을 top-K 유망 후보로 제한 → in-loop에서도 선택성을
측정 가능하게 만든 핵심 트릭(`selectivity_loop.py:3-6`). 게이트가 없으면 매
iteration 후보×5수용체 도킹이 필요해 무한 루프가 사실상 불가능.

---

## ③ 관련 Action Item (2026-06-10 선택성 GOAL)

- 출처: `_workspace/CONTINUOUS_DISCOVERY.md`(직독). 핵심 정의 인용:
  - "Δmargin = margin − native_margin(+13.37) — home-advantage 보정. >0 = native SST-14 초과 선택성."
  - "통과(passing) = Δmargin>0 & ΔG≤−15 & 독성≤native(hc50). 오늘 GOAL 의 엄격 기준."
- 즉 본 기능은 2026-06-10 "SSTR2 선택성" GOAL의 핵심 측정축이며,
  무한 발굴 엔진(`continuous.py`)의 영속 학습 대상이다(`global_selectivity_leaderboard.json`).
- 독성 게이트(hc50, home-advantage 대칭)는 `multiobjective.py:165, 172, 212`에서
  Δmargin과 동일 철학으로 구현됨(상호 보완 축).

---

## ④ 완성도 (%) + 근거

**인프라 완성도: 약 90%**
- 게이트·Δmargin·동일프로토콜·병렬·warm-start·이황화 보존·fail-closed 전부 구현되고
  runner에 통합(`runner.py:495-501, 888-897`). 회귀 테스트 존재
  (`pyrosetta_flow/tests/test_continuous_discovery.py`, seed/게이트 케이스).
- 실제 운영 데이터 축적: `n_screened_unique=539`, `n_ingested_total=564`
  (`global_selectivity_leaderboard.json:5-6`) → 파이프라인이 실제로 대량 가동됨.

**과학적 목표(임상급 선택성 입증) 완성도: 낮음~중간 (정성)**
- Δmargin은 PyRosetta 도킹 점수 차이(in-silico proxy)일 뿐, in-vitro 친화도/
  기능 assay로 검증되지 않음(`multiobjective.py:13-17` 주석: "ddG·selectivity_margin
  은 in-vitro 혈청 안정성/투과도 assay 로 검증되지 않았다").
- off-target 수용체 PDB는 SSTR2 프레임에 0.93~0.95 사전정렬된 curated 구조
  (`multiobjective.py:330-332`) → 구조 정렬 품질이 점수 신뢰도의 상한.

**인프라/과학 구분 결론**: 측정·게이트·학습 파이프라인은 사실상 완성. "선택성 후보
발견"이라는 과학 목표는 in-silico 신호 수준에서 진행 중이며 실험 검증은 미수행(미검증).

---

## ⑤ 학술 가치: **중상**

- **근거(긍정)**: SSTR2 표적 방사성의약품(예: DOTATATE 계열)에서 SSTR1/3/4/5 대비
  선택성은 off-target 흡수/방사선 독성과 직결되는 임상적 핵심 변수다. native 대비
  home-advantage 보정으로 "절대 점수 편향"을 제거하고 상대 선택성만 추출한 방법론은
  in-silico 선택성 스크리닝에서 정직성이 높은 설계다.
- **근거(한계로 인한 감점)**: Δmargin은 도킹 점수 proxy이며 실험 검증 부재
  (`multiobjective.py:13-17`). off-target 구조 정렬·단일체인 추출 가정이 결과에 영향.
  따라서 "상"이 아닌 "중상" — 방법론·엔지니어링은 발표 가치 있으나, 결과의 생물학적
  주장은 wet-lab 검증 전제 하에만 성립.

---

## ⑥ 사용법

```bash
ENV=~/miniforge3/envs/bio-tools/bin/python
cd AgenticAI4SCIENCE_pyrosetta_track/repos/ai4sci-kaeri

# 무한 발굴(선택성 게이트 포함, STOP 파일로 정지)
$ENV scripts/run_continuous_discovery.py \
    --input data/somatostatin_receptor/SSTR2_SST14_complex_boltz_1.pdb \
    --n-candidates 8 --max-iterations 4 --top-k 5 --selectivity-max-per-iter 2
```
(출처: `_workspace/CONTINUOUS_DISCOVERY.md` 실행 섹션 직독)

- 정지: `touch _workspace/STOP_DISCOVERY` (graceful), 재시작 전 `rm`.
- iteration당 도킹 개수 조절: `--selectivity-max-per-iter`(→ `max_screen_per_iter`,
  `selectivity_loop.py:124`).
- 결과 확인: `runs/pyrosetta_flow/global_selectivity_leaderboard.json`
  (`best_delta_margin`, entries의 delta_margin).
- 프로그램 직접 사용: `from pyrosetta_flow.multiobjective import screen_selectivity`
  → dict(selectivity_margin, delta_margin, more_selective_than_native, ...) 반환
  (`multiobjective.py:361, 440-450`).

---

## ⑦ 필요 이유

- SSTR2 선택성은 도킹 전에 알 수 없고 측정이 비싸다(`selectivity_loop.py:3-6`).
  ddG 프록시 게이트가 없으면 무한 루프에서 선택성을 전혀 측정할 수 없다.
- 절대 margin은 source-구조 편향으로 변별력이 없다 → home-advantage 보정(Δmargin)이
  없으면 "native 대비 실제 개선"을 구분할 수 없다(`multiobjective.py:436-437`).
- 동일 프로토콜 보정이 없으면 프로토콜 차이를 선택성으로 오인한다(이전 아티팩트,
  `multiobjective.py:402`).

---

## 검증 인용 목록

| 주장 | 인용 |
|------|------|
| off-target 5종 동일 프로토콜 도킹 | `multiobjective.py:333-338, 401-410` |
| margin = min(offtarget) − sstr2 | `multiobjective.py:434-435` |
| Δmargin = margin − native_margin | `multiobjective.py:438-439` |
| native baseline +13.37 (검산 13.366) | `native_selectivity_baseline.json:1` |
| 조건부 ddG 게이트 | `selectivity_loop.py:48-56` |
| 이황화 보존 필터 | `selectivity_loop.py:99-100` |
| iteration당 도킹 상한(2) | `selectivity_loop.py:124` |
| warm-start seed_from_global | `selectivity_loop.py:23-41` |
| runner 통합 | `runner.py:495-501, 888-897` |
| 실험 검증 부재 disclaimer | `multiobjective.py:13-17` |
| GOAL 통과 기준 | `_workspace/CONTINUOUS_DISCOVERY.md` |

---

## 현재 리더보드 실측 요약

출처: `runs/pyrosetta_flow/global_selectivity_leaderboard.json` 직독
(2026-06-17 분석 시점).

- `best_delta_margin` = **+9.1021** (서열 `ARCGKFFWKTATSC`, margin 22.4721, ddG −32.1162;
  파일 line 9-17).
- 리더보드 capacity=50, **entries 50건 전부 delta_margin > 0** (최저 +0.3777, 최고 +9.1021).
  → 즉 native(+13.37)를 초과하는 선택성 후보가 top-50 리더보드에 50건 적재됨.
- 누적 통계: `n_screened_unique` = **539** (실제 off-target 도킹한 고유 서열),
  `n_ingested_total` = **564** (line 5-6).

### 정직성 주의 (프롬프트 수치 정정)

- 프롬프트의 "현재 best Δ+0.077, native 초과 소수"는 **현 측정과 불일치**한다.
  실측 best는 **Δ+0.077이 아니라 Δ+9.1021**이며, top-50 리더보드 entries는 전부 Δ>0이다.
  "+0.077"은 2026-06-10 초기 스냅샷(당시 1건만 native 초과) 기준으로 추정되며,
  이후 무한 발굴(539 서열 도킹)로 다수 후보가 native를 초과한 것으로 보인다.
- 단, 위 Δ>0은 **PyRosetta 도킹 점수 proxy 기준**이며 wet-lab 검증이 아님을 재차 명시
  (`multiobjective.py:13-17`). "native 초과 선택성"은 in-silico 신호 한정 주장이다.
