# 12. 하네스 어댑테이션 (Harness Adaptation) 기능 분석 보고서

> 대상: 하네스 어댑테이션 메타 인프라 (Stage 0~9, VR-cycle 자가검증, 6-패턴 위임 트리)
> 루트: `[LOCAL_PATH]`
> 규칙: 모든 사실 주장에 `file_path:line` 또는 직접 인용 첨부. 미확인 항목은 §검증 필요에 기록. 읽기 전용 분석.
> 작성: 2026-06-17

---

## 0. 한 줄 정의

하네스 어댑테이션은 **발굴 정확도를 직접 올리는 기능이 아니라, 발굴 파이프라인을 굴리는 "팀 운영 방식"을 표준화·검증 가능하게 만드는 메타 인프라**다. 핵심 가치는 (1) 6개 오케스트레이션 패턴으로 작업 위임을 명시적·사후 감사 가능하게 만들고, (2) VR-cycle 자가검증 루프로 시스템이 자기 결함을 정직하게 노출하며, (3) 환각 가드(Stage 5)로 도메인 한계를 숨기지 않고 disclaimer로 드러내는 데 있다.

원본은 revfactory/harness (v1.2.0, Apache-2.0)이며, 본 프로젝트는 CLI 종속(L1)을 폐기하고 CLI-agnostic 핵심 IP(L2)만 추출했다 (`tools/harness-adaptation/ANALYSIS.md:283-292`).

---

## 1. 동작 원리

### 1.1 6개 오케스트레이션 패턴

권위 있는 명칭 출처는 원본 `plugin.json:3` 영문 열거, 상세 정의는 `reference/harness/.../agent-design-patterns.md:83-161` (분석 정리: `tools/harness-adaptation/ANALYSIS.md:109-141`).

| # | 패턴 | 적용 조건 | 본 프로젝트 매핑 | 근거 |
|---|------|----------|---------------|------|
| 1 | **Pipeline** | 순차 의존 작업, 각 단계가 직전 산출물에 강하게 의존 | 시퀀스 검증 → 변이 적용 → 도킹 → 채점 | `ANALYSIS.md:113-116` / `CLAUDE.md:42` |
| 2 | **Fan-out/Fan-in** | 동일 입력에 여러 관점·영역 분석 후 통합. "반드시 팀으로 구성" | 4 reviewer 병렬 검토 → reviewer-science 통합 | `ANALYSIS.md:118-122` / `CLAUDE.md:44,9` |
| 3 | **Expert Pool** | 입력 유형에 따라 다른 전문가 호출. 서브에이전트가 더 적합 | codex=코드, cursor-agent=분석 라우팅 | `ANALYSIS.md:123-126` / `CLAUDE.md:22` |
| 4 | **Producer-Reviewer** | 품질 보장 필요 + 객관적 검증 기준 존재. 재시도 ≤2~3 | engineer-backend 생성 ↔ reviewer 검증 루프 | `ANALYSIS.md:128-131` / `CLAUDE.md:43` |
| 5 | **Supervisor** | 작업량 가변·런타임 동적 분배 | orchestrator가 팀원에 동적 분배 | `ANALYSIS.md:133-136` / `CLAUDE.md:9` |
| 6 | **Hierarchical Delegation** | 문제가 자연스럽게 계층적 분해. 깊이 3단계 이상 금지 | 1단계 충분 시 4순위 "직접"으로 평탄화 | `ANALYSIS.md:138-141` / `CLAUDE.md:57` |

이 6패턴은 `CLAUDE.md`의 4단계 위임 트리에 1:1로 명시 매핑되어 있다 (Stage 3 작업). 즉 어느 위임 결정이 어느 패턴의 실구현인지 추적 가능하다 (`CLAUDE.md:9-10, 22-23, 41-46, 56-57`).

### 1.2 Stage 0~9 — 무엇을 했나

`CLAUDE.md`의 "Stage 적용 이력" 표(`CLAUDE.md:148-166`)와 회고(`_workspace/release/retro-2026-Q2.md:21-39`)가 출처다. 각 Stage는 독립적으로 가치를 내는 단위로 설계되었다 (`INTEGRATION_PLAN.md:4`).

| Stage | 한 일 | 사유 | 근거 |
|-------|------|------|------|
| 0 | `tools/harness-adaptation/` 디렉토리 + 원본 submodule 신설 | 어댑테이션 골격 + 원본 라이선스 보존 | `INTEGRATION_PLAN.md:8-16` |
| 1 | `_workspace/` 디렉토리 + `{NN}_{agent}_{artifact}.{ext}` 파일명 컨벤션 | 다단계 산출물 추적성 | `INTEGRATION_PLAN.md:19-33` |
| 2 | `CLAUDE.md` Harness Pointer 블록 + 트리거 키워드 | 발견 가능성 향상 | `INTEGRATION_PLAN.md:36-53` |
| 3 | 위임 트리 4분기에 6패턴 명칭 매핑 | 사후 감사 가능성 | `INTEGRATION_PLAN.md:57-73` |
| 4 | PR 템플릿 + 검증 보고서 의무화 (should-trigger/NOT-trigger/A-B) | 신규 에이전트·스킬 품질 게이트 | `INTEGRATION_PLAN.md:77-86` |
| 5 (**Critical**) | `pipeline_local/scripts/pharmacology_guards.py` + 회귀 테스트(33→39개) | 약리학 lookup table 환각 차단 | `INTEGRATION_PLAN.md:90-100` / `CLAUDE.md:152` |
| 6 | `CHANGELOG.md` + `RETROSPECTIVE_GUIDE.md` + PR 템플릿 보강 | Phase 7 진화 메커니즘 운영화 | `INTEGRATION_PLAN.md:104-113` / `CLAUDE.md:155` |
| 7 | `.claude/agents/*.md` 6개에 harness 표준 3섹션(입력/출력/에러) 보강 | 검증 가능 형식 정립 (이미 분리되어 있어 신설 불필요) | `CLAUDE.md:156` |
| 8a | `researcher` 에이전트 신설 | 외부 자료 수집 부담을 reviewer-science에서 분리 | `CLAUDE.md:158` |
| 8b | reviewer-science → pharma/biology/chemistry/math 4분리, 기존 reviewer-science는 라우터 재정의 | 도메인 경계 명확화 + 약리학 가드 직결 | `CLAUDE.md:159` |
| 8c | `scripts/auto_dispatch.sh` + orchestrator 외부 CLI 자동 dispatch | Codex/Cursor 자동 라우팅 (Expert Pool) | `CLAUDE.md:160` |
| 8d | End-to-End 사이클 dogfooding (modification_conflict checker, 71/71 tests) | 구현→검증→실험→갭→수정→완성 6단계 작동 입증 | `CLAUDE.md:161` |
| 8e/8f/8g | VR-cycle-04/07/01 closure (라우팅 회귀 + 자기 일관성 pytest + C-07 DOTA, 89/89) | §검증 8건 중 4건 자체 closure | `CLAUDE.md:162` |
| 8h | VR-cycle-09 (H-06 "계산 불가능을 계산 가능한 척") 식별·closure (93/93) | 3-layer closure, "도메인 한계의 정직한 노출" 본질 확립 | `CLAUDE.md:163` |
| 8i | VR-cycle-05/06/08 + VR-S5-01 partial closure (95/95) | GATE-F/GATE-G + disclaimer 추가. 10건 중 10건 처리 완료 | `CLAUDE.md:164` |
| 9 | Rosetta Flow End-to-End Dogfood (3 iter, 9.5분, patience early stop) | BE/FE/PyRosetta/Boltz 실 운영 + 6 critical 결함 자가 노출 + R1~R7 + VR-cycle-10~14 | `CLAUDE.md:166` / `scenario-rosetta-flow-2026-05-11.md` |

추가로 분기 회고 1차(Retro 2026-Q2)가 Stage 9와 8i 사이에 실행되어 8개 Action Item을 식별했다 (`CLAUDE.md:165`).

### 1.3 VR-cycle 자가검증 루프

VR-cycle(Verification Required cycle)은 **어댑테이션이 스스로 발견한 미검증·결함 항목을 등록하고, 후속 Stage에서 닫는** 메커니즘이다. 본질은 시스템이 자기 한계를 숨기지 않고 추적 대상으로 명시하는 것.

- **처리 현황**: 10개 항목 100% 처리 — FULL closed 5건(50%), ABSORBED 1건(VR-02→09), PARTIAL 4건(40%) (`retro-2026-Q2.md:68-75`).
- **차단된 실 회귀 결함**: `RW_TRANSFER[S]=1.15`(정답 3.40), `NEND_HALFLIFE[P]=20.0` yeast(정답 30.0 mammalian), Boman 부호 역전 — 모두 `pharmacology_guards`가 차단 (`retro-2026-Q2.md:79-86`).
- **자체 발견 1건**: `boman_index_kcal_per_mol` 범위 `[-5,+5]`가 all-K=5.55에서 너무 좁음 → `[-5,+15]`로 자가 수정. GATE-C가 자체 가설의 부정확성을 catch (`retro-2026-Q2.md:86`).
- **운영 적용 사례 (Stage 9)**: ddG=0.00이라는 비현실적 결과를 "임상 binding 값"이 아닌 "도메인 fail 신호"로 정직히 보고 — VR-cycle-09(H-06) 가드가 의도대로 작동 (`scenario-rosetta-flow-2026-05-11.md:223, 271`). Stage 9에서 신규 VR-cycle-10~14 등록 (`scenario-rosetta-flow-2026-05-11.md:208-213`).

핵심 발견(`CLAUDE.md:163`): **"하네스 본질 = 도메인 한계의 정직한 노출, 정확도 보장 X."**

---

## 2. 영향 — 시스템 품질·자가검증에 어떻게 기여하나

1. **위임의 사후 감사 가능성**: 6패턴 ↔ 4순위 트리 매핑으로 "어느 작업을 왜 그 방식으로 위임했나"를 추적 가능. Stage 3 이전엔 패턴 명칭 기록이 없어 사후 감사 불가였음 (`ANALYSIS.md:221-225` 환각 위험 지점 §Phase 2).
2. **환각 사고 예방 (Critical)**: Stage 5 환각 가드가 약리학 lookup table 4건 이상의 실 회귀를 코드 레벨에서 차단 (`retro-2026-Q2.md:79-86`). 이는 NSGA-II 순위 역전(Boman 부호) 같은 발굴 결과 왜곡을 막는다.
3. **메트릭으로 입증되는 품질 향상**: pytest 0→95/95, auto_dispatch routing 0→16/16, 에이전트 정의 6→11(+83%), CLAUDE.md 트리거 행 8→18(+125%), Conflict rules 0→10, LITERATURE_VALUES 0→5 (`retro-2026-Q2.md:58-66`).
4. **Producer-Reviewer cross-validation 신호**: Stage 9에서 6개 발견 중 5건을 2명 이상 reviewer가 독립 식별 — 단일 시각 환각을 cross-check로 걸러냄 (`scenario-rosetta-flow-2026-05-11.md:137, 224`).
5. **재현 가능한 인용 규약**: `file_path:line` 인용을 강제하여 검증자가 주장을 재현 가능 (`ANALYSIS.md:5`, `post-m0-audit` 라인 번호 인용 패턴 차용 `ANALYSIS.md:212`).

---

## 3. 관련 Action Item — Stage 이력 자체가 메타

Stage 적용 이력 표 자체가 메타 작업의 산물이다. 회고에서 **"메타 작업의 패턴 부재"**가 핵심 발견으로 기록되었다: 본 분기 작업의 절반 이상이 어댑테이션 자체의 단계적 구축(Stage 1~8 등)인데, 이들이 위임 트리 4순위 "직접 구현"으로 흡수되었을 뿐 고유 패턴 명세가 없다 (`retro-2026-Q2.md:101-107, 200`).

Q3 Action Items 8건 중 메타 인프라와 직결되는 것 (`retro-2026-Q2.md:160-169`):

| ID | 액션 | 메타 관계 |
|----|------|---------|
| Q-2026-Q3-1 (**Critical**) | CI에 pytest + routing test 등록 | 회귀 자동 차단 — 운영 단계 진입 전제 (`retro-2026-Q2.md:162, 171`) |
| Q-2026-Q3-2 | tmux team-mate 1회 실 운영 (Stage 9) | 1순위 위임 트리가 본 분기 0회 사용 — 패턴 정합성 검증 불완전 (`retro-2026-Q2.md:96, 199`) |
| Q-2026-Q3-7 | **Meta-Pipeline 패턴 정의** + 위임 트리 4순위 보강 | "메타 작업 패턴 부재" 직접 해소 (`retro-2026-Q2.md:153, 168`) |
| Q-2026-Q3-3/4 | VR-cycle-05/06 full closure (echo·토큰 비용 실측) | partial→full, 실 트래픽 필요 (`retro-2026-Q2.md:164-165`) |

Stage 9가 Q-2026-Q3-2를 사전 부분 실행하면서 다시 Q-2026-Q3-9~14를 추가 등록 (`scenario-rosetta-flow-2026-05-11.md:251-260`) — Action Item이 자기 증식하는 자가 발전 루프가 관찰된다.

---

## 4. 완성도

| 구분 | 상태 | 근거 |
|------|------|------|
| Stage 0~9 (8a~8i 포함) | **17/17 완료** | `retro-2026-Q2.md:144-145` "Stage 0/1/2/3/4/5/6/7 + 8a~8i 모두 적용" + Stage 9 추가 (`CLAUDE.md:166`) |
| §검증 필요 처리 | **10/10 (100%)** — 5 full + 1 흡수 + 4 partial | `retro-2026-Q2.md:68-75` |
| INTEGRATION_PLAN 미적용 Stage | **없음** (전부 적용 완료) | `CLAUDE.md:168-169` |

**완성도 추정: 약 90%.** Stage 골격·검증 게이트·환각 가드·진화 메커니즘은 전부 가동(17/17). 남은 10%는 **실 트래픽 검증의 공백**이다:
- 1순위 tmux team-mate 본 분기 0회 운영 (`retro-2026-Q2.md:96, 199`)
- VR-cycle-05/06이 실 트래픽 부재로 partial에 머묾 (`retro-2026-Q2.md:198`)
- CI 등록 미완료 — 회귀 테스트가 로컬에서만 작동 (`retro-2026-Q2.md:201`)
- Stage 9 dogfood에서 실제 reviewer-* Agent 호출 대신 메인 thread가 임시 수행 (`scenario-rosetta-flow-2026-05-11.md:229`)

즉 **인프라는 완성, 실 운영 데이터 누적은 진행 중**.

---

## 5. 학술 가치

- **재현성 IP의 CLI 비종속 추출**: 원본의 L1(Claude Code 종속)을 폐기하고 L2(6패턴·Phase 워크플로우·`_workspace` 컨벤션·검증 게이트)만 추출해 Codex/Cursor로 포팅 가능 (`ANALYSIS.md:145-198, 287`). 특정 벤더 도구에 묶이지 않는 멀티-에이전트 운영 방법론.
- **"정직한 한계 노출" 방법론**: 휴리스틱·degraded-mode 결과를 신뢰값처럼 보고하지 않는 가드(H-06/VR-cycle-09)는, 계산 과학 AI 파이프라인의 환각 보고 문제에 대한 일반화 가능한 패턴 (`CLAUDE.md:163`, `scenario-rosetta-flow-2026-05-11.md:271`).
- **자가검증 dogfooding 사례**: Stage 8d·9 두 차례 End-to-End dogfood가 "시스템이 자기 결함을 노출"하는 메커니즘을 입증 (Stage 9: 6 critical 결함 자가 노출 `scenario-rosetta-flow-2026-05-11.md:266-273`).
- **분기 회고의 정량 메트릭 추이**: SemVer 15 minor + 1 patch, 메트릭 before/after 표 (`retro-2026-Q2.md:43, 58-66`)가 메타 인프라 진화의 정량적 추적 사례.
- **한계 (정직성 자체의 적용)**: 원본의 "+60% 품질 향상" 주장은 n=15 author-measured 소프트웨어 태스크로 약리학 정확도에 일반화 불가하다고 본 어댑테이션이 명시 (`ANALYSIS.md:241, 272`). 즉 어댑테이션 스스로 자기 효과를 과대 주장하지 않는다.

---

## 6. 사용법 (위임 인터페이스)

`CLAUDE.md` 위임 트리(`CLAUDE.md:3-59`)와 트리거 키워드 표(`CLAUDE.md:74-90`)가 진입점.

| 위임 방식 | 호출 | 패턴 | 트리거 키워드 |
|----------|------|------|------------|
| **tmux team-mate** (`/team`) | tmux 확인 → `./scripts/launch_agent_team.sh` | Fan-out/Fan-in 또는 Supervisor | "팀", "토론", "리뷰", "검토회의" (`CLAUDE.md:15, 76`) |
| **codex** | `./scripts/agent-wrapper.sh codex <args>` | Expert Pool (코드) | "구현해/코드 작성/수정해"(단순), "리뷰해"(코드) (`CLAUDE.md:77-78`) |
| **cursor-agent** | `./scripts/agent-wrapper.sh cursor-agent <args>`; 단계별 Pipeline은 `./scripts/cursor/harness_invoke.sh` (`list`/`run`/`chain`, 기본 dry-run) | Expert Pool (분석) | "EOD/일정/상태 보고", "분석해/조사해"(구조) (`CLAUDE.md:79-80, 37`) |
| **내장 서브에이전트** (Agent tool) | Agent / `/subagent-dev` | Pipeline / Producer-Reviewer / Fan-out·Fan-in | "구현해"(복잡, 여러 파일) (`CLAUDE.md:81`) |
| **직접 구현** | (위임 없음) | Hierarchical 평탄화 | 파일 1~2개로 끝나는 작업 (`CLAUDE.md:59`) |

위임 시 필수 행동(`CLAUDE.md:63-68`): (1) 할당 시 대상·내용·예상 결과 CLI 출력, (2) 결과 수신 시 요약·변경 파일·테스트 결과 출력, (3) `logs/external_agents/`에 외부 호출 기록, (4) 실패 시 보고 + 대안. 도메인 리뷰어는 키워드로 자동 라우팅된다 (예: "약리학/Boman/GRAVY"→reviewer-pharma, "DOTA/라벨링"→reviewer-chemistry, `CLAUDE.md:87-90`).

---

## 7. 핵심 — 왜 필요한가 (발굴 정확도 메타 인프라)

**하네스 어댑테이션은 후보 펩타이드 발굴의 점수를 직접 1점도 올리지 않는다.** Δmargin 계산식도, 도킹 채점 함수도, 선택성 마진도 건드리지 않는다. 그럼에도 필요한 이유는, 발굴 결과의 **신뢰성·재현성·정직성을 떠받치는 메타 인프라**이기 때문이다.

1. **도메인 한계의 정직한 노출**: 어댑테이션의 본질은 "정확도 보장 X, 도메인 한계의 정직한 노출"로 명문화돼 있다 (`CLAUDE.md:163`). ddG=0.00 같은 degraded-mode 결과를 임상 결합 친화도인 척 보고하지 않고 fail 신호로 분류한 것이 대표 운영 사례 (`scenario-rosetta-flow-2026-05-11.md:223, 271`). 발굴이 틀릴 수 있음을 시스템이 스스로 드러낸다.

2. **환각 차단**: 약리학 lookup table 환각(Radzicka-Wolfenden S=1.15, Pro half-life=20 yeast, Boman 부호 역전)은 그 자체가 NSGA-II 순위와 후보 선정을 왜곡한다. Stage 5 가드가 이를 코드 레벨에서 차단해 **잘못된 수치로 인한 잘못된 발굴**을 예방 (`retro-2026-Q2.md:79-86`, `ANALYSIS.md:246-251` H-01~H-05).

3. **재현성**: `_workspace/` 산출물 컨벤션 + `file_path:line` 인용 강제 + SemVer CHANGELOG + 분기 회고로, 어떤 발굴 실행이 어떤 코드·에이전트·파라미터에서 나왔는지 추적 가능 (`retro-2026-Q2.md:46-54, 58-66`). 발굴 결과가 우연이 아니라 재현 가능한 절차의 산물임을 보장.

4. **자가검증의 메타 루프**: VR-cycle이 시스템 결함을 등록·closure하고, dogfooding이 결함을 자가 노출하며, 회고가 Action Item을 식별한다. 이 루프가 발굴 파이프라인 자체의 버그(Stage 9의 PyRosetta cache key 충돌, ddG silent fallback 0.0 — `scenario-rosetta-flow-2026-05-11.md:70-98`)를 잡아내, 정확도에 **간접적으로** 기여한다.

요약: 발굴 정확도를 올리는 것은 스코어링·도킹·선택성 엔진의 몫이고, 하네스 어댑테이션은 **그 엔진이 거짓말하지 않고, 추적 가능하며, 자기 결함을 노출하도록 강제하는 운영 체계**다. 정확도의 *상한*이 아니라 *신뢰의 하한*을 책임진다.

---

## 검증 인용 목록

| # | 주장 | 출처 |
|---|------|------|
| 1 | 6패턴 명칭·적용 조건 | `tools/harness-adaptation/ANALYSIS.md:109-141` (원본 `agent-design-patterns.md:83-161`) |
| 2 | 6패턴 ↔ 위임 트리 매핑 | `CLAUDE.md:9-10, 22-23, 41-46, 56-57` |
| 3 | Stage 적용 이력 표 | `CLAUDE.md:148-166` |
| 4 | Stage 0~7 통합 로드맵·사유 | `tools/harness-adaptation/INTEGRATION_PLAN.md:8-126` |
| 5 | Stage 5 Critical·환각 가드 | `INTEGRATION_PLAN.md:90-100` / `CLAUDE.md:152` |
| 6 | VR-cycle 처리 현황 10/10 | `_workspace/release/retro-2026-Q2.md:68-75` |
| 7 | 차단된 실 회귀 결함 4건 + 자체 발견 1건 | `retro-2026-Q2.md:79-86` |
| 8 | 메트릭 before/after 추이 | `retro-2026-Q2.md:58-66` |
| 9 | "메타 작업 패턴 부재" 발견 | `retro-2026-Q2.md:101-107, 200` |
| 10 | Q3 Action Items 8건 | `retro-2026-Q2.md:160-169` |
| 11 | Stage 0~8 모두 적용 완료 | `retro-2026-Q2.md:144-145` |
| 12 | Stage 9 dogfood — 6 critical 결함 자가 노출 + R1~R7 | `_workspace/release/scenario-rosetta-flow-2026-05-11.md:53-201, 266-273` |
| 13 | ddG=0.00 정직한 보고 (H-06 운영 사례) | `scenario-rosetta-flow-2026-05-11.md:223, 271` |
| 14 | "본질 = 도메인 한계의 정직한 노출, 정확도 보장 X" | `CLAUDE.md:163` |
| 15 | L1/L2/L3 레이어 분리·CLI 추상화 | `ANALYSIS.md:145-198, 283-292` |
| 16 | "+60% 일반화 불가" 자기 한계 명시 | `ANALYSIS.md:241, 272` |
| 17 | 위임 방식·트리거 키워드·필수 행동 | `CLAUDE.md:3-90` |

> **§검증 필요 (본 보고서)**: (a) 완성도 90%는 회고의 정성 평가(`retro-2026-Q2.md:196-201`)에 근거한 추정치 — 정량 산식 없음. (b) `agent-design-patterns.md` 원본 라인 번호는 `ANALYSIS.md` 인용을 신뢰한 것으로, submodule `reference/harness/` 실파일은 본 분석 시점 디렉토리에서 미확인(`ls` 결과 비어 있음 — `git submodule update --init` 필요). (c) CI 등록·tmux 실 운영은 Q3 미완료 항목으로, 본 보고서 작성 시점(2026-06-17) 추가 진척은 미확인.
