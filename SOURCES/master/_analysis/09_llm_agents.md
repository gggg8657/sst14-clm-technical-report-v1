# 09. LLM 에이전트 (Planner / Critic / Reporter, vLLM)

> 대상 모듈: `AG_src/llm/prompts.py`, `AG_src/llm/provider.py`, `AG_src/agents/{base_agent,planner,critic,reporter}.py`, `AG_src/config/pipeline_config.yaml`
> 절대 규칙: 모든 주장은 `file_path:line` 인용. 미확인 사항은 "미검증"으로 명시. 본 분석은 읽기 전용.

---

## ① 동작 원리

### 1.1 3개 LLM 에이전트의 역할 (Co-Scientist 루프)

본 파이프라인은 PyRosetta mutate→dock 루프를 LLM이 "가설 주도"로 조종하는 Co-Scientist 구조다. 세 에이전트가 각 iteration에서 순차적으로 작동한다.

- **PlannerAgent** (`AG_src/agents/planner.py:212`) — "연구 설계/실험 기획" (`planner.py:226`). 각 iteration의 `ExperimentPlan`(가설 + 파라미터 + 변이 가이드)을 생성·갱신한다 (`planner.py:241` `create_initial_plan`, `planner.py:335` `update_plan`).
- **ScientistCriticAgent** (`AG_src/agents/critic.py:266`) — "비판적 검토/원인 분석" (`critic.py:285`). QC 결과/랭크 테이블을 해석해 실패 유형을 분류하고 다음 iteration 파라미터 변경(최대 2개)을 제안한다 (`critic.py:297` `analyze_results`).
- **ReporterAgent** (`AG_src/agents/reporter.py:53`) — "최종 리포트/그림 자동화" (`reporter.py:74`). iteration 요약 Markdown·lab notebook·PyMOL 렌더 스크립트를 생성한다 (`reporter.py:197` `generate_summary_report`).

세 에이전트 모두 `BaseAgent`(`AG_src/agents/base_agent.py:56`)를 상속하며, `BaseAgent.has_llm`(`base_agent.py:197`)으로 LLM 연결 여부를 판단하고 `llm_generate_json`(`base_agent.py:221`)으로 JSON 응답을 받는다.

### 1.2 vLLM 백엔드 + Qwen3-32B no-think

LLM 호출은 `VLLMProvider`(`AG_src/llm/provider.py:251`)가 담당한다.

- 엔드포인트: OpenAI 호환 `POST /v1/chat/completions` (`provider.py:304`).
- 모델: 설정상 `qwen3-32b`, `base_url: http://localhost:8000`, `temperature: 0.3`, `max_tokens: 4096`, `timeout: 120` (`AG_src/config/pipeline_config.yaml:225-231`).
- **no-think 강제**: payload 최상위에 `chat_template_kwargs.enable_thinking=False`를 직접 삽입한다 (`provider.py:299`). 이는 Qwen3 thinking 모델이 `content=null` + reasoning만 반환하거나 보수적으로 응답하는 현상(M4 버그)을 방지하기 위함이다 (`provider.py:83-85`, `provider.py:272-274`). 설정 기본값도 `enable_thinking: false` (`pipeline_config.yaml:232`).
- thinking이 켜진 경우의 안전망: `content`가 비면 `reasoning`/`reasoning_content`로 폴백하고, 둘 다 없으면 토큰 초과 가능성을 경고하며 `None` 반환 (`provider.py:327-343`).

`create_provider`(`provider.py:351`)가 팩토리다. `config.llm.provider` 값으로 `vllm`/`ollama`/`none`을 선택하며 (`provider.py:397-420`), `config=None`이면 `NoneProvider`(`provider.py:165`)를 반환해 LLM 없이 규칙 기반 모드로 작동한다 (`provider.py:383-384`). **agent별 override**도 지원: `agent_name` 지정 시 `llm.agents.<name>` 섹션이 상위 `llm` 값을 덮어쓴다 (`provider.py:386-396`). 다만 설정 파일에서 `agents:` 블록은 현재 주석 처리되어 있어 세 에이전트가 동일 모델(qwen3-32b)을 공유한다 (`pipeline_config.yaml:241-254`).

### 1.3 JSON 스키마 강제와 프롬프트 구조

`prompts.py`는 에이전트별 시스템 프롬프트(`SYSTEM_PROMPTS`, `prompts.py:77`)와 출력 JSON 스키마(`OUTPUT_SCHEMAS`, `prompts.py:121`)를 분리 관리한다.

- JSON 모드: `VLLMProvider.generate`가 `json_mode=True`일 때 `response_format={"type":"json_object"}`를 설정한다 (`provider.py:301-302`).
- 스키마 주입: 각 user 프롬프트 끝에 해당 스키마를 `\`\`\`json ... \`\`\``로 첨부한다 (예: planner `prompts.py:320-325`, critic `prompts.py:400-405`, reporter `prompts.py:574-577`).
- planner는 두 모드를 가진다 — `default`(RFdiffusion/MPNN/ESMFold 포함, `prompts.py:32`)와 `pyrosetta_only`(`prompts.py:46`). 후자는 `mutation_guidance`(focus_positions + suggested_mutations)를 강제하고, RFdiffusion 등 외부 도구 언급을 금지한다 (`prompts.py:50-60`, 스키마 `prompts.py:122-143`).
- 파싱 견고성: 응답에서 `<think>...</think>` 블록 제거(`_strip_think_blocks`, `provider.py:427`) 후, 직접 `json.loads` → 실패 시 마크다운 코드블록/중괄호 추출(`_extract_json_block`, `provider.py:446`)로 3단계 폴백한다 (`provider.py:146-151`).

### 1.4 fail-open 폴백 (LLM 실패 시 규칙 기반)

세 에이전트는 모두 "LLM 우선, 실패 시 규칙 기반" 구조다.

- Planner: LLM 계획 생성 시도(`planner.py:265-278`) → `None`이거나 `validate_plan` 실패 시 규칙 기반 기본 파라미터(`_DEFAULT_PARAMETERS`, `planner.py:84`)로 폴백.
- Critic: `_analyze_via_llm`(`critic.py:567`)가 `None`이면 `FAILURE_ACTION_MAP`(`critic.py:150`) 기반 규칙 분석으로 폴백 (`critic.py:328-336`).
- Reporter: `_summarize_via_llm`(`reporter.py:516`)가 `None`이면 템플릿 기반 Markdown으로 폴백 (`reporter.py:222-231`).
- 전송 계층 회복력: `_http_post_json`(`provider.py:46`)이 일시적 오류(URLError/timeout/5xx)에 대해 지수 백오프 재시도(기본 1+2회, `provider.py:42-43`), HTTP 4xx는 영구 오류로 즉시 중단 (`provider.py:65-69`). 이는 vLLM 순간 장애가 조용히 규칙 기반으로 degrade되는 것을 막기 위한 D4/F18 보강이다 (`provider.py:39-41`).

**주의(미검증 위험)**: fail-open 구조이므로 vLLM이 다운되어도 파이프라인은 멈추지 않고 규칙 기반으로 계속 돈다. 로그 경고는 남지만(`planner.py:276-278` 등), "LLM이 실제로 가설을 주도했는지"는 로그를 확인해야 알 수 있다 — 코드만으로는 미검증.

---

## ② 영향 — 가설 주도 탐색

핵심 가치는 "탐색이 무작위/그리드가 아니라 LLM 가설로 유도된다"는 점이다.

- Planner가 falsifiable 가설을 명시하고(`prompts.py:42`, `prompts.py:57`), `pyrosetta_only` 모드에서 **구체적 변이 가이드**(어느 위치를 어떤 잔기로)를 생성해 다음 mutate→dock 라운드의 탐색 공간을 좁힌다 (`prompts.py:126-135`).
- Critic이 결과를 6개 실패 유형(structural/sequence/docking/stability/selectivity)으로 분류하고(`critic.py:72-79`, 시스템 프롬프트 `prompts.py:98`), 원인에 맞는 파라미터/변이 변경을 제안해 피드백 루프를 닫는다.
- **선택성 정렬(2026-06-10 보강)**: planner/critic 양쪽 프롬프트가 캠페인의 1차 목표를 "강한 SSTR2 결합"이 아니라 "SSTR2 선택성(Δmargin>0)"으로 재정의한다. planner 시스템 프롬프트는 SSTR2-고유 영역(ECL2/ECL3/TM5/TM6)과 보존 위치 보존 전략을 지시하고(`prompts.py:61-75`), critic 시스템 프롬프트는 Δmargin≤0이면 "캠페인 미성공"으로 진단하라고 지시한다 (`prompts.py:86-95`). user 프롬프트에는 in-loop 선택성 리더보드가 주입된다 (planner `prompts.py:278-291`, critic `prompts.py:372-398`). 이 신호는 `ScientistCriticAgent.execute`가 컨텍스트의 `selectivity_leaderboard`/`best_delta_margin`을 모아 전달한다 (`critic.py:550-556`).

순효과: LLM이 "Δmargin을 양수로 만드는 변이"라는 과학적 목표를 직접 추론하도록 설계되어, 단순 ddG 최적화의 함정(off-target도 강하게 결합)을 회피하는 방향으로 탐색을 편향시킨다.

---

## ③ 관련 Action Item — "큰 LLM 결정"

사용자 메모리(`sstr2-goal-decisions`)의 "큰 LLM·real PyRosetta·전면 통합" 결정이 본 모듈에 반영되어 있다.

- ollama qwen3:8b → vLLM qwen3.5-35b-a3b(MoE Active 3B) → **qwen3-32b**로 업그레이드된 이력이 설정 주석에 기록됨 (`pipeline_config.yaml:220-226`). 현재 운영 모델은 `qwen3-32b`, GPU2 서빙, "~40 tok/s, JSON+no-think 검증됨" (`pipeline_config.yaml:226`).
- "큰 LLM"으로 전환하면서 thinking 모드 비활성화가 필수 조건이 됨(보수적 응답·content null 방지) — `enable_thinking=false` 정책으로 코드·설정 양쪽에 고정 (`provider.py:299`, `pipeline_config.yaml:232`).
- agent별 다른 LLM 운영 옵션(M3, planner=CoT 강한 모델, critic/reporter=빠른 MoE)이 코드(`provider.py:386-396`)와 설정 주석(`pipeline_config.yaml:241-254`)에 준비되어 있으나 현재 비활성(주석) — **확장 여지로 식별됨, 활성화 여부 미검증**.

---

## ④ 완성도 평가

**완성도: 약 85%**

근거:

| 항목 | 상태 | 근거 |
|------|------|------|
| 3 에이전트 LLM 통합 | 완료 | planner/critic/reporter 모두 `_*_via_llm` 경로 구현 (`planner.py:532`, `critic.py:567`, `reporter.py:516`) |
| vLLM/Ollama/None 프로바이더 | 완료 | 3종 + 팩토리 (`provider.py:165,190,251,351`) |
| JSON 스키마 강제 + 파싱 폴백 | 완료 | `response_format` + 3단계 추출 (`provider.py:301`, `146-151`) |
| no-think + 전송 재시도 | 완료 | `enable_thinking` (`provider.py:299`), 백오프 (`provider.py:46`) |
| fail-open 규칙 기반 폴백 | 완료 | 세 에이전트 전부 폴백 경로 보유 |
| 선택성 in-loop 피드백 | 완료 | planner/critic 프롬프트 + execute 배선 (`prompts.py:278`, `critic.py:550`) |
| agent별 LLM override | 구현됐으나 비활성 | 코드 존재(`provider.py:386`), 설정 주석 처리(`pipeline_config.yaml:241`) |

감점 사유(미완/불일치):

1. **Critic LLM 스키마와 시스템 프롬프트의 selectivity 차원 불일치(미검증 위험)**: 시스템 프롬프트는 실패 유형에 `selectivity`를 추가하라고 지시(`prompts.py:98`)하나, `OUTPUT_SCHEMAS["critic"].failure_analysis`에는 `structural/sequence/docking/stability`만 있고 selectivity 필드가 없다 (`prompts.py:170-175`). LLM이 selectivity를 어디에 담을지 스키마상 모호 — 실제 응답 처리 영향은 미검증.
2. **planner `default` 스키마와 `_PYROSETTA_ONLY_STEPS`의 잔존 RFdiffusion 흔적**: default 스키마/스텝 템플릿에는 여전히 RFdiffusion/MPNN/ESMFold 구조가 남아 있다(`prompts.py:144-167`, `planner.py:108-155`). 운영 모드가 pyrosetta_only일 때만 정합 — 모드 설정 위치는 본 분석 범위 밖(미검증).
3. **Reporter 폴백 보고서의 모델명 하드코딩 오류**: LLM 보고서 헤더에 "생성 방식: LLM (Qwen 2.5 7B)"로 잘못 기재(`reporter.py:572`). 실제 모델은 qwen3-32b — 사소하나 명백한 표기 버그.
4. **prompts.py 모듈 docstring의 모델명 노후화**: "Qwen 2.5 7B / Qwen 3.5-35B-A3B"로 기재(`prompts.py:4`), 현 운영 모델 qwen3-32b와 불일치.

---

## ⑤ 학술 가치 — Agentic AI Scientist

- **Co-Scientist / AI Scientist 패러다임 구현체**: Planner(가설 생성)→실험(mutate-dock)→Critic(원인 분석·반증)→Reporter(기록)의 닫힌 자율 과학 루프를 LLM으로 구동한다. 이는 "AI가 가설을 세우고 검증·수정하는" agentic discovery 흐름(예: Google Co-Scientist, Sakana AI Scientist 계열)의 도메인 특화 적용이다.
- **검증 가능한 가설(falsifiability) 강제**: 프롬프트가 "falsifiable hypothesis"와 "다음 iteration의 pLDDT/ddG/도킹 분포로 검증"을 명시(`prompts.py:42`, `critic.py:521-523`)해, 단순 생성이 아니라 과학적 방법론(가설-검증)을 LLM 출력에 구조적으로 부과한다.
- **환각 방지 설계**: JSON 스키마 강제, no-think, 규칙 기반 폴백, 선택성 같은 정량 목표(Δmargin) 직결 — LLM의 비결정성을 도메인 제약으로 가두는 방식은 "신뢰 가능한 과학 에이전트" 연구에 사례 가치가 있다. 특히 critic 프롬프트가 "강한 ddG가 나쁜 선택성을 가리지 않게 하라"(`prompts.py:94-95`)고 명시한 것은 다목적 최적화에서 LLM judge의 함정을 선제 차단한 설계.
- **재현성/추적성**: 변경을 최대 2개로 제한(원인-결과 추적, `critic.py:276`)하고 `changes_from_prev`를 기록(`planner.py:570-574`)하는 것은 실험 노트북 자동화의 학술적 모범 사례.

---

## ⑥ 사용법

### 6.1 프로바이더 생성

```python
from AG_src.llm.provider import create_provider
import yaml

config = yaml.safe_load(open("AG_src/config/pipeline_config.yaml"))
provider = create_provider(config)                    # 상위 llm 설정 사용 (vllm qwen3-32b)
provider = create_provider(config, agent_name="critic")  # agent별 override 적용 (활성 시)
provider = create_provider(None)                      # NoneProvider → 규칙 기반 모드
```

(시그니처: `provider.py:351`)

### 6.2 에이전트에 LLM 주입

```python
from AG_src.agents.planner import PlannerAgent

planner = PlannerAgent(llm_provider=provider, planner_mode="pyrosetta_only")
plan = planner.create_initial_plan(receptor_config={...}, constraints={...})
# 또는 context 기반
result = planner.execute({"iteration": 1, "receptor_config": {...}, "constraints": {...}})
```

(생성자: `planner.py:223`, execute: `planner.py:501`. critic: `critic.py:281`/`critic.py:534`, reporter: `reporter.py:67`/`reporter.py:465`)

### 6.3 직접 JSON 생성 / 변이 생성 프롬프트

```python
from AG_src.llm.prompts import build_variant_generation_prompt, VARIANT_DESIGN_SYSTEM_PROMPT

prompt = build_variant_generation_prompt("AGCKNFFWKTFTSC", [1,2,4,5,6,11,12,13], n_mutations=3)
result = provider.generate_json(prompt, system_prompt=VARIANT_DESIGN_SYSTEM_PROMPT)
```

(`prompts.py:461`, `prompts.py:445`)

### 6.4 전제 조건

- vLLM 서버가 `http://localhost:8000`에서 qwen3-32b를 서빙 중이어야 함 (`pipeline_config.yaml:226-227`). 미가동 시 자동으로 규칙 기반 폴백.
- timeout 120s, max_tokens 4096 (`pipeline_config.yaml:229-231`).

---

## ⑦ 필요 이유

1. **무작위 탐색의 비효율 회피**: SST-14 가변 위치 8곳(`pipeline_config.yaml:196`)에 대한 조합 폭발을 LLM 가설로 좁혀, mutate→dock 같은 고비용(후보당 최대 1800s, `pipeline_config.yaml:209`) 평가를 가치 높은 변이에 집중시킨다.
2. **선택성이라는 다목적 함정 해결**: ddG만 최적화하면 native SST-14처럼 pan-agonist가 되어버리는 함정을, "Δmargin을 양수로"라는 명시적 목표를 LLM에 부과해 회피한다 (`prompts.py:61-75`, `prompts.py:86-95`).
3. **자율 실험 루프의 두뇌**: 코드 기반 에이전트(qc_ranker, diversity_manager, builder, `pipeline_config.yaml:234`)는 "실행"을 담당하고, LLM 에이전트는 "왜/무엇을 다음에"라는 의사결정을 담당한다 — Co-Scientist 분업의 핵심 두뇌 역할.
4. **운영 안정성**: vLLM 장애·토큰 초과·JSON 깨짐에도 파이프라인이 멈추지 않도록 fail-open + 재시도 + 다단계 파싱을 갖춰, 무한 발굴 엔진(메모리 `sstr2-selectivity-goal`) 같은 장기 무인 운영을 가능케 한다.

---

## 검증 인용 목록

- `AG_src/llm/provider.py:46-82` — `_http_post_json` 백오프 재시도, 4xx 영구 오류 처리
- `AG_src/llm/provider.py:122-151` — `generate_json`, think 블록 제거 + JSON 폴백
- `AG_src/llm/provider.py:165-183` — `NoneProvider` (규칙 기반 모드)
- `AG_src/llm/provider.py:251-344` — `VLLMProvider`, `enable_thinking`, content/reasoning 폴백
- `AG_src/llm/provider.py:299` — `chat_template_kwargs.enable_thinking` payload 주입
- `AG_src/llm/provider.py:301-302` — `response_format` JSON 모드
- `AG_src/llm/provider.py:351-420` — `create_provider` 팩토리 + agent override
- `AG_src/llm/provider.py:427-470` — `_strip_think_blocks`, `_extract_json_block`
- `AG_src/llm/prompts.py:32-75` — planner default / pyrosetta_only 시스템 프롬프트 (선택성 가이드)
- `AG_src/llm/prompts.py:80-113` — critic / reporter 시스템 프롬프트 (선택성 정렬)
- `AG_src/llm/prompts.py:121-205` — `OUTPUT_SCHEMAS` (4종 JSON 스키마)
- `AG_src/llm/prompts.py:228-327` — `format_planner_prompt` (+ 선택성 리더보드 주입)
- `AG_src/llm/prompts.py:330-407` — `format_critic_prompt` (+ 선택성 섹션)
- `AG_src/llm/prompts.py:445-532` — `VARIANT_DESIGN_SYSTEM_PROMPT`, `build_variant_generation_prompt`
- `AG_src/llm/prompts.py:535-579` — `format_reporter_prompt`
- `AG_src/agents/base_agent.py:56-239` — `BaseAgent`, `has_llm`, `llm_generate_json`
- `AG_src/agents/planner.py:212-613` — `PlannerAgent`, LLM 분기 + 규칙 폴백 + sanitize
- `AG_src/agents/critic.py:150-259` — `FAILURE_ACTION_MAP` (규칙 기반 폴백)
- `AG_src/agents/critic.py:266-634` — `ScientistCriticAgent`, `_analyze_via_llm`, 선택성 배선
- `AG_src/agents/reporter.py:53-586` — `ReporterAgent`, `_summarize_via_llm`, 폴백 템플릿
- `AG_src/agents/reporter.py:572` — 모델명 표기 오류 ("Qwen 2.5 7B")
- `AG_src/config/pipeline_config.yaml:219-254` — LLM 설정 (vllm qwen3-32b, enable_thinking false, agent override 주석)
