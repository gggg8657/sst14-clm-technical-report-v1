# 변이 생성 엔진 + Scaffold 가드 — 기능 분석 보고서

> **검증 원칙**: 본 보고서의 모든 동작·정량 주장은 `file_path:line` 인용으로 검증한다. 코드에서 확인되지 않은 내용은 "미검증"으로 명시한다. 수치는 코드에 명시된 값만 인용한다.
>
> **분석 루트**: `AgenticAI4SCIENCE_pyrosetta_track/repos/ai4sci-kaeri`
> **대상 파일**: `pyrosetta_flow/runner.py`, `pyrosetta_flow/schema.py`, `pyrosetta_flow/adapter.py`(변이 생성기), `AG_src/llm/prompts.py`

---

## ① 동작 원리

### (1) 변이 위치 선택 — `_mutable_design_positions`

변이 가능 위치는 `config.design_positions` 에서 **약리단(FWKT, 7-10)** 과 **Cys 위치(이황화)** 를 제외하여 산출한다.

- `design_positions` 기본값: `[1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]` — `pyrosetta_flow/schema.py:14-18`
  (주석: 실제 변이 가능 = `[1,2,4,5,6,11,12]` — Cys3/14·FWKT 제외, `schema.py:15-16`)
- 약리단 위치 상수: `PHARMACOPHORE_POSITIONS_1IDX = (7, 8, 9, 10)` — `runner.py:102`
- Cys 위치 추출: `_disulfide_cys_positions` 가 참조 서열에서 `C` 인 위치(1-indexed)를 모두 반환 — `runner.py:114-116`
- 최종 필터: `mutable_positions = [pos for pos in config.design_positions if pos not in PHARMACOPHORE_POSITIONS_1IDX and pos not in cys]` — `runner.py:135-138`
- **안전 폴백**: 필터 결과가 비면 `config.design_positions` 전체를 그대로 반환 (`mutable_positions or list(config.design_positions)`) — `runner.py:139`
  → 미검증 우려: 이 폴백이 발동하면 FWKT/Cys 가 변이 후보에 다시 포함될 수 있으나, 후술하는 `_preserves_scaffold` 게이트가 후단에서 그런 제안을 거부하므로 실제 통과는 차단됨(아래 (3) 참조).

이 함수의 산출물은 변이 루프 진입 직전 한 번 호출되어 `mutation_positions` 로 사용된다 — `runner.py:596`.

### (2) 약리단(FWKT 7-10) · Cys3/Cys14 보존 메커니즘

세 단계로 작동한다.

1. **위치 선택 단계 제외** (사전 차단): 위 (1)에서 FWKT·Cys 를 변이 후보에서 제거 — `runner.py:132-139`.
2. **생성기 자체 제약**: 실제 변이 아미노산 풀에서 Cys 를 배제.
   - `AA_NO_CYS = list("ADEFGHIKLMNPQRSTVWY")` (19종, Cys 없음) — `adapter.py:10`
   - `generate_random_mutant` 가 이 풀에서만 치환 — `adapter.py:53-54`
   - `generate_guided_mutant` 도 추천 풀에서 `C` 를 명시 배제(`aa != "C"`) — `adapter.py:91`, 폴백 풀도 `AA_NO_CYS` — `adapter.py:96`
   → 즉 변이로 인해 **새로운 Cys 가 도입되지 않는다**. (단, 기존 Cys 가 다른 AA 로 바뀌는 것은 생성기만으로는 못 막음 — 그래서 (3) 게이트가 필요.)
3. **사후 게이트** (거부): 생성된 후보가 약리단·이황화를 보존하는지 검증 — 아래 (3).

약리단 보존 판정:
- `_pharmacophore_slice(seq) = seq[6:10]` (7-10 위치 슬라이스) — `runner.py:106-107`
- `_preserves_pharmacophore` 는 후보와 참조의 7-10 슬라이스가 **완전 동일**할 때만 True — `runner.py:110-111`

이황화 보존 판정:
- `_preserves_disulfide`: 참조 서열의 **모든 Cys 위치가 후보에서도 `C`** 여야 True — `runner.py:119-123`
- 결합 게이트 `_preserves_scaffold` = 약리단 보존 AND 이황화 보존 — `runner.py:126-129`

### (3) 거부 로직 (변이 생성 루프)

후보 1개 생성에 대해 **약리단 재시도 외층 + dedup 내층**의 이중 루프 구조 — `runner.py:597-687`.

- 약리단 재시도 한도: `PHARMACOPHORE_RETRY_LIMIT = 3` → 외층은 `range(PHARMACOPHORE_RETRY_LIMIT + 1)` = 4회 — `runner.py:103`, `runner.py:601`
- dedup 내층: `range(config.max_dedup_trials)` (기본 `max_dedup_trials = 50`) — `runner.py:603`, `schema.py:40`
- 변이 수 escalation: 내층 trial 이 진행될수록 강제 변이 수를 1→2→`max_random_mutations`→`max_random_mutations+1` 로 상향 (`max_random_mutations` 기본 3) — `runner.py:605-611`, `schema.py:41`
- 거부 분기:
  - 원본과 동일하거나 이미 본 서열 → `continue` (재시도) — `runner.py:640-641`
  - **scaffold 미보존 → `break`** (해당 dedup 내층 즉시 탈출) — `runner.py:642-643`
  - 통과 시 `mutant` 확정 + `seen_sequences` 등록 — `runner.py:644-646`
- 폴백 경로: 내층에서 mutant 미확정 시, 마지막 제안이 scaffold 위반이면 외층 `continue`; 아니면 변이 수를 `max(2, len//2)` 로 강제한 추가 random 후보를 만들어 scaffold·dedup·non-native 통과 시 채택 — `runner.py:649-672`
- **최종 실패 처리**: 4회 외층을 모두 소진해도 mutant 가 없으면 `fail_reason = "FWKT pharmacophore gate failed after 3 retries"` 로 표시하고 마지막 제안을 mutant 로 둠 — `runner.py:674-679`. 이 후보는 도킹 단계에서 `ddg=999.0` 등으로 즉시 실패 처리 — `runner.py:721-732`.

`seen_sequences` 초기값은 `{config.original_sequence}` 로, 네이티브 서열이 후보로 절대 제출되지 않도록 한다 — `runner.py:458`.

### (4) LLM 프롬프트 측 보존 명세 — `AG_src/llm/prompts.py`

LLM 직접 변이 생성 경로(`build_variant_generation_prompt`)는 동일 제약을 자연어/few-shot 으로 강제한다.

- 시스템 프롬프트 `VARIANT_DESIGN_SYSTEM_PROMPT`: "Preserve C3 and C14 (disulfide) — positions 3 and 14 FIXED", "Preserve FWKT pharmacophore — positions 7,8,9,10 FIXED" — `prompts.py:445-458`
- 거부 조건 명시: "Any mutation modifies positions 3, 7, 8, 9, 10, or 14" → output rejected — `prompts.py:456`
- `build_variant_generation_prompt` 본문도 동일 RULE 1·2 반복 + few-shot 2개 — `prompts.py:493-505`
- few-shot 예시(`_SST14_FEW_SHOT_EXAMPLES`): `AGCKNFFWKTFTSC` → `PGCKHFFWKTFISC`(pos 1/5/12 변이), `AECKNLFWKTYTSC`(pos 2/6/11 변이) — `prompts.py:417-442`. 두 예시 모두 3·7·8·9·10·14 를 보존한다(육안 검증).
- Planner 시스템 프롬프트도 "PRESERVE FWKT pharmacophore (pos 7-10) and Cys3/Cys14 disulfide — never mutate these" — `prompts.py:69-71`

> **불일치 (미검증/주의)**: `build_variant_generation_prompt` 의 docstring·예시는 가변 위치를 `[1,2,4,5,6,11,12,13]` (pos 13 포함)으로 기술 — `prompts.py:18,255,416,479`. 반면 실제 엔진 `design_positions` 기본값은 13을 포함하지 않음(`[...,11,12]`, `schema.py:17`). 또한 `format_planner_prompt` 의 fallback 표시 문자열은 `[1,2,4,5,6,7,8,9,10,11,12,14]`(14 포함) — `prompts.py:255`. 이 LLM-경로 프롬프트와 실제 실행 경로(`generate_random_mutant`/`generate_guided_mutant`, `runner.py:614,627`)의 위치 집합이 **서로 다르다**. 실행 경로의 scaffold 게이트가 최종 안전망이므로 결과 정합성은 유지되나, 프롬프트의 위치 명세는 코드 기본값과 동기화되어 있지 않음(미해소 문서 부채).

---

## ② 영향 / 효과 — 이황화 파괴 방지가 없으면?

`_preserves_disulfide` docstring 이 부재 시 버그를 직접 명시한다:

> "이전엔 가드 부재로 C14→H 변이가 통과해 SS bond 가 깨졌다." — `runner.py:120-121`

### 가드 부재 시 결함 (C14H 사례)
- SST-14 는 **Cys3-Cys14 이황화결합**으로 고리형(cyclic) 구조를 형성한다(프로젝트 컨텍스트 CLAUDE.md, `prompts.py:456`).
- 가드가 없으면 변이 생성기가 pos 14 의 `C` 를 다른 아미노산(예: His)으로 치환할 수 있고, 그 후보가 거부 없이 도킹·채점까지 진입한다.
- 결과: 이황화결합이 끊긴 **선형 펩타이드**가 "SST-14 analog" 으로 평가됨 → 구조·결합 자유에너지(ddG) 결과가 실제 cyclic scaffold 와 무관한 값이 됨 → 후보 랭킹·선택성 판정 오염.
- 이는 단순 점수 오류가 아니라 **scaffold 정체성 붕괴**(설계 대상이 아닌 분자를 평가)에 해당.

### 현재 가드의 효과
- `generate_random_mutant`/`generate_guided_mutant` 가 Cys 도입을 막고(`AA_NO_CYS`, `adapter.py:10,53,91,96`), `_mutable_design_positions` 가 Cys 위치를 변이 후보에서 제거하며(`runner.py:135-138`), `_preserves_scaffold` 가 잔존하는 위반 후보를 도킹 직전 거부(`runner.py:642-643`)한다.
- 결과적으로 도킹에 들어가는 모든 후보는 **FWKT 약리단 + Cys3/Cys14 이황화** 가 보존된 합법 scaffold 임이 보장됨(통과 후보 한정).
- 통합 스모크 테스트가 이 불변식을 간접 검증: `assert result["disulfide_intact"] is True, "Cys3-Cys14 이황화결합 유실"` — `pyrosetta_flow/tests/test_integration_smoke.py:160`.

---

## ③ 관련 Action Item

retro-Q2 (`_workspace/release/retro-2026-Q2.md:158-169`)의 8개 Action Item 은 모두 **CI 등록·harness 메타·문헌 LITERATURE_VALUES** 등 인프라/메타 작업이며, scaffold 가드 기능 자체를 직접 다루는 항목은 **없음**.

가장 근접한 간접 연결:
- **Q-2026-Q3-1 "CI에 pytest + routing test 등록"** (`retro-2026-Q2.md:162`): 본 가드(`_preserves_disulfide`/`_preserves_scaffold`/`_mutable_design_positions`)에 대한 **직접 단위 테스트가 저장소에 부재**함을 확인(검색 결과 `pyrosetta_flow/tests`, `AG_src/tests`, `tests` 에 이 함수명 직접 호출 테스트 없음 — 간접 스모크만 존재, `test_integration_smoke.py:160`). 따라서 이 가드의 회귀 보호는 Q3-1 의 사정권에 들어가나, 명시적으로 지정되진 않음.

연대 관계: 본 가드는 **2026-06-10** 에 도입(`runner.py:120,133`, `schema.py:15` 주석 날짜)되어, retro-Q2(2026-Q2 분기, CLAUDE.md Stage 이력상 2026-05-11 작성)보다 **이후** 작업이다. 따라서 **retro-Q2 Action Item 과는 직접 무관**하다.

> **결론**: 직접 연결되는 Action Item 없음. 사유 — (a) 가드 도입(2026-06-10)이 retro-Q2(2026-05-11) 이후, (b) retro-Q2 의 8건은 메타/CI/문헌 작업으로 scaffold 보존 로직과 도메인이 다름. 기능 동기는 Action Item 이 아니라 **메모리의 선택성 GOAL**(`sstr2-selectivity-goal`, 2026-06-10)에 있음 — 무한 발굴 엔진에서 다수 변이를 자동 생성할 때 scaffold 무결성을 보장하기 위함.

---

## ④ 완성도 % + 근거

**완성도: 약 85%**

근거:
- (+) 핵심 보존 로직 3중 방어(위치 제외 + 생성기 Cys 배제 + 사후 게이트) 모두 구현·연결됨 — `runner.py:135-138,642-643`, `adapter.py:10,91`.
- (+) 거부 시 재시도·escalation·폴백·실패표시까지 완비 — `runner.py:601-679`.
- (+) LLM 경로 프롬프트에도 동일 제약 명세 + few-shot — `prompts.py:445-458,493-505`.
- (+) 도킹 단계가 fail_reason 후보를 즉시 999 처리하여 오염 차단 — `runner.py:721-732`.
- (−) **직접 단위 테스트 부재**: `_preserves_disulfide`/`_preserves_scaffold`/`_mutable_design_positions` 를 직접 호출하는 회귀 테스트 미발견(간접 스모크 1건만, `test_integration_smoke.py:160`). C14H 회귀를 명시 방어하는 테스트가 없음 → 약 -10%.
- (−) **프롬프트↔코드 위치 명세 불일치**: LLM 경로 가변 위치(`[1,2,4,5,6,11,12,13]`/14 포함)와 실행 경로 `design_positions`(`[...,11,12]`)가 다름 — `prompts.py:255,479` vs `schema.py:17` → 약 -5%.
- (−) `_mutable_design_positions` 폴백(`runner.py:139`)이 FWKT/Cys 를 재포함시킬 수 있는 경로 존재(게이트가 후단에서 차단하나 방어적 설계 관점에서 미세 결함) — 미검증 영향, 감점 소.

---

## ⑤ 학술 가치 (상 / 중 / 하)

**평가: 상(高)**

근거:
- 펩타이드/단백질 설계에서 **scaffold 제약(disulfide-constrained, pharmacophore-fixed mutagenesis)** 은 표준적이고 학술적으로 중요한 설계 원리다. 무작위/유도 변이가 약물활성단(FWKT)과 구조 결정 요소(이황화결합)를 보존하도록 강제하는 것은, 설계 공간을 "생물학적으로 유효한 analog" 로 한정하는 핵심 메커니즘이다.
- SST-14 의 FWKT 는 SSTR 결합 약리단으로 알려져 있고(`prompts.py:449,497` 명세), Cys3-Cys14 이황화는 활성 형태(β-turn) 유지에 필수 — 이를 코드 레벨에서 불변식으로 박제한 것은 "유효 analog 만 평가" 라는 방법론적 정당성을 부여한다.
- 본 가드가 막는 결함(scaffold 정체성 붕괴, ②의 C14H 사례)은 in-silico 스크리닝 결과의 **타당성(validity)** 에 직결되므로, 단순 엔지니어링이 아닌 **과학적 방법론 가드**의 성격을 가진다.
- 감점 요소: 보존 판정이 "위치 동일성"이라는 단순 비교(`runner.py:111,123`)로, 보존적 치환(conservative substitution)·이황화 위상(topology) 검증 같은 정교한 구조 화학 검증은 아님. 이는 적정 단순화이나 "고급 학술 기여"라기보다 "필수 건전성 가드" 수준. 그럼에도 스크리닝 결과 신뢰성에 미치는 영향이 커 종합 **상**.

---

## ⑥ 사용법 (config / 함수)

### 관련 config (`FlowConfig`, `schema.py`)
- `original_sequence` (기본 `"AGCKNFFWKTFTSC"`, `schema.py:13`) — Cys 위치·약리단 슬라이스의 기준 서열.
- `design_positions` (기본 `[1,2,4,5,6,7,8,9,10,11,12]`, `schema.py:14-18`) — 변이 후보 풀의 원천. 여기서 FWKT·Cys 가 자동 제외됨.
- `max_dedup_trials` (기본 50, `schema.py:40`) — scaffold/dedup 재시도 내층 횟수.
- `max_random_mutations` (기본 3, `schema.py:41`) — 동시 변이 수 상한(escalation 시 +1).
- `n_candidates` (기본 8, `schema.py:19`) — iteration 당 생성 후보 수.

### 호출 흐름 (자동, `runner.py`)
1. `run_pyrosetta_agentic_mutdock_flow(config)` 진입 — `runner.py:294`.
2. 변이 단계에서 `mutation_positions = _mutable_design_positions(config)` — `runner.py:596`.
3. 후보별 루프에서 `generate_guided_mutant`(가이드 있을 때) 또는 `generate_random_mutant` 호출 — `runner.py:613-638`.
4. 제안마다 `_preserves_scaffold(proposal, config.original_sequence)` 게이트 — `runner.py:642`.
   (직접 호출형 보조: `_preserves_pharmacophore`, `_preserves_disulfide`, `_disulfide_cys_positions` — `runner.py:110,119,114`)

### LLM 직접 변이 생성 (선택적 경로)
- `build_variant_generation_prompt(reference_sequence, mutable_positions, n_mutations, variant_id)` — `prompts.py:461-532`.
- 옵션 시스템 프롬프트: `VARIANT_DESIGN_SYSTEM_PROMPT` — `prompts.py:445`.
- 사용 예: `provider.generate_json(prompt, system_prompt=VARIANT_DESIGN_SYSTEM_PROMPT)` — `prompts.py:14-21,485` (docstring 예시). 단, ④에서 지적한 위치 명세 불일치 유의.

---

## ⑦ (Action 무관 시) 왜 필요한가

본 기능은 retro-Q2 Action Item 과 직접 무관(③)하므로 필요성을 명시한다.

- **유효 analog 보장**: 무한 발굴 엔진(메모리 `sstr2-selectivity-goal`, `continuous.py`)은 매 iteration 다수 변이를 자동 생성한다. 가드가 없으면 자동 생성 과정에서 FWKT/이황화를 깨는 후보가 섞여, 결합활성을 잃은 분자가 "SST-14 analog 후보" 로 도킹·랭킹에 진입한다.
- **C14H 회귀의 영구 차단**: 실제로 가드 부재 시 C14→H 변이가 통과해 SS bond 가 깨진 전례가 있다(`runner.py:120-121`). 본 가드는 그 회귀를 코드 불변식으로 봉인한다.
- **선택성 캠페인 신뢰성**: 캠페인 목표는 SSTR2 선택성(Δmargin>0, `prompts.py:86-95`)이다. scaffold 가 깨진 후보의 ddG/선택성 수치는 무의미하므로, 가드는 **선택성 판정의 입력 건전성**을 지키는 전제 조건이다.

---

## 검증 인용 목록 (file:line)

**`pyrosetta_flow/runner.py`**
- `:102` — `PHARMACOPHORE_POSITIONS_1IDX = (7, 8, 9, 10)`
- `:103` — `PHARMACOPHORE_RETRY_LIMIT = 3`
- `:106-107` — `_pharmacophore_slice` (7-10 슬라이스)
- `:110-111` — `_preserves_pharmacophore` (슬라이스 동일성)
- `:114-116` — `_disulfide_cys_positions` (참조 Cys 위치 1-indexed)
- `:119-123` — `_preserves_disulfide` + docstring "C14→H 변이가 통과해 SS bond 가 깨졌다"
- `:126-129` — `_preserves_scaffold` = 약리단 AND 이황화
- `:132-139` — `_mutable_design_positions` (FWKT·Cys 제외 + 폴백)
- `:458` — `seen_sequences = {config.original_sequence}` (네이티브 제외)
- `:596` — `mutation_positions = _mutable_design_positions(config)`
- `:597-687` — 후보 생성 이중 루프
- `:601` — 약리단 재시도 외층 `range(PHARMACOPHORE_RETRY_LIMIT + 1)`
- `:603` — dedup 내층 `range(max_trials)`
- `:605-611` — 변이 수 escalation
- `:613-638` — guided/random 변이 호출
- `:640-643` — 거부 분기(원본/중복 continue, scaffold 위반 break)
- `:644-646` — mutant 확정 + seen 등록
- `:649-672` — 폴백 경로
- `:674-679` — 최종 실패 `fail_reason = "FWKT pharmacophore gate failed after 3 retries"`
- `:721-732` — fail_reason 후보 ddg=999 처리

**`pyrosetta_flow/schema.py`**
- `:13` — `original_sequence = "AGCKNFFWKTFTSC"`
- `:14-18` — `design_positions` 기본값 + Cys14 제거 주석
- `:19` — `n_candidates = 8`
- `:40` — `max_dedup_trials = 50`
- `:41` — `max_random_mutations = 3`

**`pyrosetta_flow/adapter.py`**
- `:10` — `AA_NO_CYS = list("ADEFGHIKLMNPQRSTVWY")`
- `:36-55` — `generate_random_mutant` (AA_NO_CYS 치환)
- `:58-99` — `generate_guided_mutant`
- `:91` — 추천 풀에서 `aa != "C"` 배제
- `:96` — 폴백 풀 `AA_NO_CYS`

**`AG_src/llm/prompts.py`**
- `:18,255,416,479` — LLM 경로 가변 위치 명세(`[1,2,4,5,6,11,12,13]` 등) — 코드 기본값과 불일치
- `:69-71` — Planner: "PRESERVE FWKT (7-10) and Cys3/Cys14 — never mutate these"
- `:86-95` — Critic: 선택성 Δmargin 목표
- `:417-442` — `_SST14_FEW_SHOT_EXAMPLES` (3/7/8/9/10/14 보존 예시)
- `:445-458` — `VARIANT_DESIGN_SYSTEM_PROMPT` (RULE 1·2, 거부 조건)
- `:461-532` — `build_variant_generation_prompt`
- `:493-505` — 프롬프트 본문 RULE 반복

**테스트 / Action Item**
- `pyrosetta_flow/tests/test_integration_smoke.py:160` — `disulfide_intact is True` 간접 검증
- `_workspace/release/retro-2026-Q2.md:158-169` — Phase E Action Item 8건(직접 무관)
- `_workspace/release/retro-2026-Q2.md:162` — Q-2026-Q3-1 "CI에 pytest 등록"(간접 연결)

**미검증 항목**
- `_preserves_disulfide`/`_preserves_scaffold`/`_mutable_design_positions` 직접 단위 테스트 부재 (저장소 검색 결과 미발견 — 직접 호출 테스트 없음, 간접 스모크만 존재)
- `_mutable_design_positions` 폴백(`runner.py:139`) 발동 시 FWKT/Cys 재포함 가능성의 실제 영향 (게이트 후단 차단으로 결과 영향은 차단 추정)
