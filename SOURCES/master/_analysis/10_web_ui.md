# 10. 웹 UI 기능 분석 보고서 (FastAPI + React + Mol*)

> 루트: `AgenticAI4SCIENCE_pyrosetta_track/repos/ai4sci-kaeri`
> 대상: `backend/` (FastAPI), `frontend/src/` (React 19 + Vite + Mol*)
> 규칙: 모든 사실은 `file_path:line` 인용. 미확인 항목은 "미검증" 명시. 읽기전용 분석.

---

## ① 동작 원리

### A. 백엔드 아키텍처 (FastAPI)

- **앱 팩토리**: `create_app()` 이 모든 라우터를 `/api` 프리픽스로 등록 (`backend/main.py:45-135`). 모듈 레벨 `app = create_app()` 으로 `uvicorn backend.main:app` 기동 (`backend/main.py:139`, `backend/main.py:7`).
- **CORS**: `http://localhost:5173`(Vite dev), `http://localhost:8787`(API) 만 허용, 메소드 `GET/POST/PUT/DELETE/OPTIONS` (`backend/main.py:65-70`).
- **표준 에러 응답**: `{"error","detail","status_code"}` 형식으로 전역 예외(500)·HTTPException·ValidationError(422) 핸들러 정의 (`backend/main.py:73-108`).
- **포트**: 기본 `8787`, `API_PORT` env 로 override (`backend/state.py:56`).

### B. StatusEmitter → JSON 파일 → 폴링 (핵심 데이터 흐름)

이 시스템의 라이브 모니터링은 **SSE/WebSocket 푸시가 아니라 "파일 + 폴링"** 으로 동작한다.

1. **쓰기(Writer)**: 파이프라인(`scripts/run_pyrosetta_flow.py`)이 `StatusEmitter` 인스턴스를 통해 상태를 갱신.
   - 상태 파일 기본 경로 `/tmp/pipeline_local_status.json` (`backend/status_emitter.py:30-34`, env `PIPELINE_STATUS_FILE`).
   - 상태 dict 스키마: `steps`(13개 파이프라인 단계), `agents`(5개), `rosetta_substeps`(7개), `timeline`, `candidates`, `qc_gates`, `convergence`, `best_candidate`, `molecules`, `visualization_images`, `completed` 등 (`backend/status_emitter.py:91-114`).
   - 모든 갱신 메소드(`update_step`, `update_agent`, `set_candidates` 등)가 매번 `flush()` 호출 (`backend/status_emitter.py:117-134`).
   - **flush() 는 원자적 쓰기**: `fcntl.flock` 배타 락 + temp write + `rename` 으로 동시 워커의 torn-write 방지 (`backend/status_emitter.py:117-133`, C5 fix 주석).

2. **읽기(Reader)**: `GET /api/status` (`backend/routers/status.py:157-164`).
   - `read_status()` 가 파일 mtime 캐시로 I/O 절약하되 `server_time` 은 캐시 hit 시에도 매번 갱신 (`backend/state.py:127-153`).
   - `is_active_run` 파생 필드 주입: `(not completed) AND (steps 또는 candidates 존재)` (`backend/state.py:101-124`).
   - 파일 부재 시 `{"error":"no_status_file","connected":False}` 반환 (`backend/state.py:136-137`).
   - **on-the-fly enrichment**: candidates 에 6개 약리/구조 필드(instability_index, gravy, net_charge_ph74, selectivity_margin, fwkt_contact, chelator_site_available)를 응답 시점에 머지 (`backend/routers/status.py:60-164`). ⑤⑥은 sequence-only 휴리스틱이라 HEURISTIC 으로 명시 (`backend/routers/status.py:73-76`, `138-150`).

3. **폴링(Frontend)**: `usePipelineStatus(2000)` 훅이 2초 간격으로 `fetch('/api/status')` (`frontend/src/hooks/usePipelineStatus.ts:262-296`, `329-343`).
   - `AbortController` 로 이전 요청 취소 (`usePipelineStatus.ts:263-265`).
   - `no_status_file`/`connected:false` 감지 시 `INITIAL_STATE` 로 리셋 — Live 대시보드로 오인 방지 (`usePipelineStatus.ts:274-283`).
   - snake_case BE → camelCase FE 매핑은 `parseStatusData()` (`usePipelineStatus.ts:170-236`).
   - 동시에 `/api/runs` 폴링으로 아카이브 목록 갱신, `switchRun()` 으로 과거 run 조회(`/api/runs/{run_id}`) (`usePipelineStatus.ts:252-260`, `298-326`).

> **폴링 중복(설계상 의도)**: `usePipelineStatus` 가 `/api/status`(2s), `useExperiment` 가 `/api/experiment/status`(3s) 를 별도 폴링. 전자는 대시보드 시각화, 후자는 start/stop 제어용 (`usePipelineStatus.ts:238-245`, `frontend/src/hooks/useExperiment.ts:74-93`).

### C. 실험 제어 (Experiment Control)

- `POST /api/experiment/run` 이 `subprocess.Popen` 으로 `scripts/run_pyrosetta_flow.py` 를 `bio-tools` conda python 으로 기동 (`backend/routers/experiment.py:205-293`, `145-155`).
  - 템플릿 PDB 기본값 `SSTR2_SST14_complex_boltz_1.pdb` (`experiment.py:220-222`).
  - 시작 즉시 `phase:"initializing"` 최소 status 파일 기록 (`experiment.py:84-98`, `273`).
  - **watchdog 스레드**: 기본 1시간(`EXPERIMENT_MAX_RUNTIME`) 초과 시 `SIGTERM→SIGKILL` (`experiment.py:30-31`, `112-142`).
  - 정상/중지/타임아웃 종료 시 `_auto_archive()` 로 `archives/{run_id}_dashboard.json` 저장 (`experiment.py:34-70`, `296-317`).
- `GET /api/experiment/models`: vLLM(`port 8000` `/v1/models`) 우선 조회, 실패 시 ollama 폴백, 최종 폴백 `["qwen3-32b"]` (`experiment.py:163-189`).
- FE `useExperiment` 가 config/models 로드 + start/stop + 상태 폴링 담당 (`useExperiment.ts:53-141`).

### D. 에이전트 로그 — REST + SSE

여기에만 진짜 **SSE(Server-Sent Events)** 가 존재한다.

- `GET /api/agents/{run_id}/log`: `runs_local/{run_id}/experiment_log.jsonl` 등에서 누적 로그 1회 반환 (`backend/routers/agents.py:176-181`, `60-75`).
- `GET /api/agents/{run_id}/stream`: `text/event-stream` 응답, `_tail_log_file` 이 jsonl 파일을 0.25s 간격 tail 하며 새 라인을 큐로 푸시, 10s 무이벤트 시 `: ping` keepalive (`agents.py:193-225`, `133-173`).
- path traversal 방어: `..`, `/`, `\` 토큰 거부 (`agents.py:68-74`, `111-114`).
- 에이전트 6종 고정 정의(planner/builder/qcranker/diversity/critic/reporter) (`agents.py:40-47`). 단, FE `AgentMonitor` 가 소비하는 에이전트는 `status` JSON 의 5종(`status_emitter.py:58-64`)이며 이 SSE 로그 6종과 별개 — **두 에이전트 표현이 분리**되어 있음(미검증: FE 가 `/api/agents/.../stream` 을 실제 구독하는 컴포넌트는 본 분석 파일 범위에서 확인 안 함).

### E. Mol* 3D 구조 뷰어

- **PDB 서빙**: `GET /api/structures/{rel_path}` 가 `runs/`, `runs/pyrosetta_flow/`, `runs_local/`, `data/` 4개 베이스에서 `.pdb` 만 서빙 (`backend/routers/static.py:20-39`). 보안: `relative_to` 경로 탈출 차단 + symlink 거부 (`static.py:24-31`). 이미지는 `/api/images/{rel_path}` (`static.py:42-60`).
- **Mol* 로딩**: `MoleculeViewer` 컴포넌트가 `PluginContext` 초기화 → `mountAsync` → `builders.data.download(Asset.Url(pdbUrl))` → `parseTrajectory(..., PdbProvider)` → preset 적용 (`frontend/src/components/MoleculeViewer.tsx:50-91`).
- 4가지 view mode(Complex/Cartoon/Ball&Stick/Surface)를 Mol* preset 으로 매핑 (`MoleculeViewer.tsx:240-265`).
- dark theme 배경(`0x0f172a`), 카메라 리셋, fullscreen, ESC 닫기, focus trap 접근성 (`MoleculeViewer.tsx:57`, `112-129`, `41`).
- `pdbUrl` 은 candidate `pdb_path`(=`source`) 에서 `/api/structures/...` 로 변환 (`SelectivityExplorerPage.tsx:423-447`, `CandidatePage.tsx:81`). 셀렉티비티 receptor 구조는 별도 `GET /api/selectivity/structure/{name}` 로 CIF/PDB 서빙 (`backend/routers/selectivity.py:96-115`).

### F. 셀렉티비티 라우터

- `GET /api/selectivity/receptors`: SSTR1/2/3/4/5 CIF 파일 존재 여부·크기 반환, 0개 로드 시 error 로그 (`selectivity.py:39-68`).
- `POST /api/selectivity/upload`: receptor 구조 파일 업로드 (`selectivity.py:71-93`).
- 셀렉티비티 분석 job 은 in-memory `_JOBS` dict 에 보관, `status.py` 가 완료 job 의 margin 을 candidate 에 머지 (`selectivity.py:120`, `status.py:39-57`).

---

## ② 영향 (Impact)

- **단일 SSOT 패턴의 장점**: 컴퓨트 레이어(파이프라인 워커)와 표현 레이어(웹)가 JSON 파일 1개로만 결합 → 백엔드가 죽어도 파일이 남고, 파이프라인이 죽어도 마지막 상태가 보존됨. flock 원자 쓰기로 동시성 안전 (`status_emitter.py:117-133`, `state.py:71-90`).
- **폴링 비용**: `/api/status`(2s) + `/api/runs`(2s) + `/api/experiment/status`(3s) 가 클라이언트마다 상시 발생. mtime 캐시로 파일 파싱은 절약되나 HTTP 왕복·enrichment 계산(`_enrich_candidates`)은 매 폴링마다 수행 (`status.py:60-164`) — 후보 수 증가 시 응답 지연 가능(미검증: 부하 측정 없음).
- **아카이브 기반 회고**: 완료 run 이 자동으로 `archives/{run_id}_dashboard.json` + PDB 사본으로 보존되어 과거 run 재현 가능 (`status_emitter.py:397-425`, `experiment.py:34-70`).
- **무한 발굴 엔진과의 단절(중대)**: 아래 ④ 참조 — 본 UI 는 `/api/experiment/*`(유한 run) 만 제어하며, 무한 발굴 엔진(`pyrosetta_flow/continuous.py`)의 출력을 전혀 표시·제어하지 못함.

---

## ③ 관련 Action Item

> 근거: `MEMORY.md` — `sstr2-ui-goal` 메모리("**차기 GOAL(미착수)**: 무한엔진용 풀 웹 모니터링·제어 UI"), UI goal 프롬프트 `_workspace/GOAL_UI_2026-06-10.md`.

1. **[UI 연동, 최우선] 무한엔진 ↔ 웹 연결**: 백엔드에 `/api/discovery/*` 라우터를 신설하여 `runs/pyrosetta_flow/discovery_status.json` / `global_selectivity_leaderboard.json` 을 서빙하고, FE 폴링 훅(`useDiscovery` 등)을 추가. (현재 0개 — ④ 참조)
2. **[UI 연동] 무한엔진 제어**: `_workspace/discovery_control.json` 을 통한 시작/정지/파라미터 조정 엔드포인트. (현재 파일만 존재, 웹 제어 경로 없음)
3. **[UI goal Phase] 전면 재디자인**: UI goal(L3)이 "무한엔진 연결 + 전면 재디자인"을 함께 요구 — 현재 13단계 유한 파이프라인 시각화(`status_emitter.py:42-56`)가 무한 epoch 모델과 정보구조가 맞지 않음.
4. **[리팩토링 후보] 에이전트 표현 이원화 해소**: `status` JSON 5-에이전트(`status_emitter.py:58-64`) vs `/api/agents` SSE 6-에이전트(`agents.py:40-47`) 통합 검토.
5. **[성능] 폴링 → SSE 통합 검토**: 기존 SSE 인프라(`agents.py:193-225`)를 status 전반으로 확장하면 3중 폴링 제거 가능(`usePipelineStatus.ts:238-245` 의 TODO 주석과 일치).

---

## ④ 완성도 추정: **약 70%** (유한 파이프라인 UI 기준 풍부, 무한엔진 연동 0%)

### 근거 — 풍부한 부분 (완성)
- 백엔드 라우터 **20+개** 등록(`main.py:113-133`): status/experiment/agents/selectivity/static 외 admet/analysis/validation/cluster/stability/benchmark/wetlab/flexpepdock/binding_pocket/strategies/silo_a 등.
- 프론트 **15개 페이지 라우트** + **40+ 컴포넌트** (`frontend/src/App.tsx:277-364`, `frontend/src/components/` 다수).
- 핵심 모니터링 기능 작동: StatusEmitter 원자 쓰기, mtime 캐시 폴링, candidate 6-field enrichment, 아카이브 자동 보존, Mol* 4-mode 3D 뷰어, 에이전트 SSE 스트림, 셀렉티비티 receptor 관리, 실험 start/stop/watchdog.
- 테스트 자산 존재: `backend/tests/` 18개, `frontend/src/**/__tests__/` 10+개.

### 근거 — 미완성 / 단절 (핵심 감점)
- **무한엔진 미연결 (검증 완료)**:
  - `grep -rn "discovery" backend/` → **0건**.
  - `grep -rn "discovery" frontend/src/` → **0건**.
  - `grep -rn "api/discovery"` (py/ts/tsx) → **0건**.
  - 즉 `backend/main.py` 라우터 목록(`main.py:113-133`)에 `/api/discovery/*` **없음**.
  - 반면 무한엔진 출력 파일은 실재: `runs/pyrosetta_flow/discovery_status.json`(21KB, 키: `running, epochs_done, elapsed_seconds, global_best_delta_margin, passing_count, diversity_level, top, last_epoch, history`), `runs/pyrosetta_flow/global_selectivity_leaderboard.json`(25KB), 제어 파일 `_workspace/discovery_control.json` — **모두 어떤 라우터도 읽지 않음**.
  - 엔진 본체 `pyrosetta_flow/continuous.py` 는 존재(웹과 독립 실행).
  - 결론: **무한 발굴 엔진은 웹 UI 와 완전히 분리되어 있으며, 현재 웹에서 무한엔진 상태를 보거나 제어할 방법이 없다.**
- `MEMORY.md` 가 UI goal 을 "차기 GOAL(미착수)"로 명시.

---

## ⑤ 학술적 가치 (과학 모니터링 UX)

- **계산과학 재현성 UX**: run → 자동 아카이브(JSON+PDB 사본) → `switchRun` 으로 과거 run 재조회(`usePipelineStatus.ts:298-326`, `status_emitter.py:397-425`)는 *in-silico* 실험의 재현성·감사 추적을 UI 레벨에서 구현한 사례.
- **휴리스틱 정직성**: enrichment 6필드 중 sequence-only 추정치(fwkt_contact, chelator_site)를 HEURISTIC 으로 라벨링(`status.py:73-76`)하고, FE 가 fail reason 을 과학적 한국어 설명으로 변환(`CandidateTable.tsx:13-54`) — "계산 불가능을 가능한 척하지 않는다"는 프로젝트 원칙(CLAUDE.md VR-cycle-09)의 UX 구현.
- **다중 표현 분자 시각화**: Mol* 4-mode(complex/cartoon/ball&stick/surface) preset(`MoleculeViewer.tsx:240-265`)으로 SSTR2-펩타이드 복합체의 결합 인터페이스 검토 지원.
- **에이전트 의사결정 투명성**: planner hypothesis / critic proposed_changes(old→new+rationale) / reporter summary 를 패널로 노출(`AgentMonitor.tsx:119-191`) — agentic 과학 파이프라인의 의사결정 과정 가시화.

---

## ⑥ 사용법 (기동)

> 미검증: 아래 명령은 `backend/main.py:7`, `frontend/package.json:7`, `vite.config.ts:14-21` 기반 추론. 실제 실행 로그는 본 분석 범위 밖.

**백엔드 (포트 8787)**:
```
# 레포 루트(ai4sci-kaeri)에서
uvicorn backend.main:app --host 0.0.0.0 --port 8787 --reload   # backend/main.py:7
# 또는
python -m backend.main                                          # backend/main.py:142-144
```
- 상태 파일 경로 override: `PIPELINE_STATUS_FILE=/tmp/pipeline_local_status.json` (`state.py:40-45`).
- 모델 목록은 vLLM `localhost:8000` 가 떠 있어야 정상(없으면 ollama→default 폴백) (`experiment.py:163-189`).

**프론트엔드 (포트 5173, Vite dev)**:
```
cd frontend
npm install
npm run dev          # vite — package.json:7
```
- Vite dev 서버가 `/api` 요청을 `http://127.0.0.1:8787` 로 프록시 (`vite.config.ts:14-21`).
- 프로덕션: `npm run build` → `npm run preview` (`package.json:8,10`).

**브라우저**: `http://localhost:5173` 접속 → `/` 가 `/console` 로 리다이렉트 (`App.tsx:278`).

---

## ⑦ 필요 이유 (Why)

- agentic *in-silico* 스크리닝은 수십~수백 epoch/iteration·다수 후보를 생성 → CLI 로그만으로는 수렴 추세·후보 랭킹·게이트 통과율·3D 결합 구조를 실시간 파악 불가. 웹 UI 는 이를 **단일 화면 모니터링**으로 압축.
- 파이프라인(컴퓨트, `bio-tools` conda)과 표현(웹)을 **JSON 파일로 디커플링**(`status_emitter.py:30-34`)함으로써, GPU 워커 환경과 무관하게 어디서든 브라우저로 관찰·제어(start/stop) 가능.
- 그러나 ④에서 확인했듯 **무한 발굴 엔진(현 프로젝트의 핵심 산출물)은 아직 이 모니터링 체계 밖에 있어**, 무한엔진을 웹에 연결하는 것이 UI goal 의 존재 이유이자 다음 우선순위다.

---

## 검증 인용 목록 (file_path:line)

**백엔드**
- `backend/main.py:7` — uvicorn 기동 커맨드 (docstring)
- `backend/main.py:45-135` — create_app, 라우터 등록 20+개
- `backend/main.py:65-70` — CORS (5173/8787)
- `backend/main.py:73-108` — 표준 에러 핸들러
- `backend/main.py:113-133` — 라우터 목록(discovery 부재 근거)
- `backend/main.py:139,142-144` — module-level app, __main__
- `backend/status_emitter.py:30-34` — STATUS_FILE 경로(/tmp/pipeline_local_status.json)
- `backend/status_emitter.py:42-56` — DEFAULT_STEPS 13개(유한 파이프라인)
- `backend/status_emitter.py:58-64` — DEFAULT_AGENTS 5개
- `backend/status_emitter.py:91-114` — 상태 dict 스키마
- `backend/status_emitter.py:117-133` — flush() flock 원자 쓰기
- `backend/status_emitter.py:199-224` — set_candidates 병합·ddG 재랭크
- `backend/status_emitter.py:397-425` — _save_archive(PDB 사본 포함)
- `backend/state.py:40-49` — STATUS_FILE/ARCHIVE_DIR env
- `backend/state.py:56` — PORT 8787
- `backend/state.py:71-90` — atomic_write_json
- `backend/state.py:101-124` — _with_runtime_fields(is_active_run/server_time)
- `backend/state.py:127-153` — read_status mtime 캐시
- `backend/routers/status.py:39-57` — selectivity margin lookup
- `backend/routers/status.py:60-164` — _enrich_candidates 6필드(휴리스틱 라벨 73-76)
- `backend/routers/status.py:157-164` — GET /api/status
- `backend/routers/status.py:187-225` — /api/runs, /api/runs/{run_id}
- `backend/routers/experiment.py:30-31,112-142` — watchdog
- `backend/routers/experiment.py:34-70,296-317` — _auto_archive
- `backend/routers/experiment.py:84-98,273` — initializing status
- `backend/routers/experiment.py:163-189` — /experiment/models(vLLM→ollama→default)
- `backend/routers/experiment.py:205-293` — POST /experiment/run(subprocess)
- `backend/routers/agents.py:40-47` — SSE 에이전트 6종
- `backend/routers/agents.py:60-75,111-114` — 로그 경로·path traversal 방어
- `backend/routers/agents.py:133-173` — _tail_log_file
- `backend/routers/agents.py:176-181` — GET /agents/{run_id}/log
- `backend/routers/agents.py:193-225` — GET /agents/{run_id}/stream (SSE)
- `backend/routers/static.py:20-39` — GET /api/structures/{rel_path} (.pdb)
- `backend/routers/static.py:42-60` — GET /api/images/{rel_path}
- `backend/routers/selectivity.py:39-68` — /selectivity/receptors
- `backend/routers/selectivity.py:71-93` — /selectivity/upload
- `backend/routers/selectivity.py:96-115` — /selectivity/structure/{name} (CIF/PDB)
- `backend/routers/selectivity.py:120` — _JOBS in-memory

**프론트엔드**
- `frontend/src/hooks/usePipelineStatus.ts:170-236` — parseStatusData(snake→camel)
- `frontend/src/hooks/usePipelineStatus.ts:238-245` — 폴링 중복 NOTE
- `frontend/src/hooks/usePipelineStatus.ts:262-296` — fetchLiveStatus(/api/status)
- `frontend/src/hooks/usePipelineStatus.ts:274-283` — no_status_file 리셋
- `frontend/src/hooks/usePipelineStatus.ts:298-326` — switchRun(/api/runs/{id})
- `frontend/src/hooks/usePipelineStatus.ts:329-343` — 2s setInterval 폴링
- `frontend/src/hooks/useExperiment.ts:53-141` — config/models/start/stop/3s 폴링
- `frontend/src/components/MoleculeViewer.tsx:50-91` — Mol* init/PDB 로드
- `frontend/src/components/MoleculeViewer.tsx:240-265` — view mode preset 매핑
- `frontend/src/components/AgentMonitor.tsx:119-191` — ReportPanel(plan/critic/reporter)
- `frontend/src/components/CandidateTable.tsx:13-54` — humanizeFailReason(과학적 설명)
- `frontend/src/components/ConvergenceGraph.tsx:56-60` — 빈 데이터 가드
- `frontend/src/pages/SelectivityExplorerPage.tsx:423-447` — pdb_path→/api/structures URL
- `frontend/src/App.tsx:277-364` — 15개 라우트
- `frontend/vite.config.ts:14-21` — /api → 127.0.0.1:8787 프록시
- `frontend/package.json:7,8,10` — dev/build/preview 스크립트
- `frontend/package.json:20` — molstar ^5.6.1

**무한엔진 미연결 근거 (grep/ls 검증)**
- `grep -rn "discovery" backend/` → 0건
- `grep -rn "discovery" frontend/src/` → 0건
- `grep -rn "api/discovery"` (py/ts/tsx) → 0건
- `runs/pyrosetta_flow/discovery_status.json` (21KB, 실재, 미소비)
- `runs/pyrosetta_flow/global_selectivity_leaderboard.json` (25KB, 실재, 미소비)
- `_workspace/discovery_control.json` (실재, 웹 제어 경로 없음)
- `pyrosetta_flow/continuous.py` (엔진 본체, 웹 독립 실행)
- `MEMORY.md` sstr2-ui-goal — UI goal "미착수" 명시
