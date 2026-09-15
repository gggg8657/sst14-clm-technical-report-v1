# 08. 무한 발굴 엔진 + 영속성 — 기능 분석 보고서

> 분석 루트: `AgenticAI4SCIENCE_pyrosetta_track/repos/ai4sci-kaeri`
> 대상: `pyrosetta_flow/continuous.py`, `pyrosetta_flow/global_leaderboard.py`, `runner.py`(baseline_cache·warm-start·ingest), `scripts/run_continuous_discovery.py`, `scripts/autopush_results.sh`(루트), `experiment_log.jsonl` dedup
> 원칙: 모든 사실은 `file_path:line` 인용. 코드로 미확인된 항목은 "미검증"으로 명시. 읽기 전용 분석.

---

## ① 동작 원리

### 1.1 epoch 무한 루프 (while + STOP)

엔진의 심장은 `run_continuous_discovery()`의 `while True:` 루프다 (`pyrosetta_flow/continuous.py:125`). 각 epoch 시작 시 두 종료 조건을 먼저 검사한다.

- **STOP 파일 감지**: `stop_path.exists()` 이면 `stop_reason="stop_file"` 로 graceful 종료 (`continuous.py:126-129`).
- **max_epochs 도달**: `max_epochs` 가 정수면 그만큼만, `None` 이면 무한 (`continuous.py:130-132`).

종료 조건 통과 후 `epoch += 1` 하고 (`continuous.py:134`) 본 작업을 수행한다. 즉 STOP 파일은 "현재 epoch 까지 마치고" 멈추는 게 아니라 **다음 epoch 진입 직전**에 검사되므로, 진행 중인 epoch 은 끝까지 완주한다.

세 번째 종료 경로는 **목표 도달 자동 정지**다: control 의 `target_pass_count` 와 `stop_on_target=true` 가 동시에 설정되면, 글로벌 리더보드의 통과 건수가 목표 이상일 때 `stop_reason="target_reached"` 로 종료한다 (`continuous.py:145-152`).

epoch 실패는 루프를 죽이지 않는다 — `run_pyrosetta_agentic_mutdock_flow(cfg)` 를 try/except 로 감싸 예외를 `err` 에 기록하고 continue 한다 (`continuous.py:159-166`).

### 1.2 다양성 탈출 (patience 정체 → 변이 수↑ + seed 이동)

`DiversityPolicy` 클래스 (`continuous.py:57-84`)가 local optimum 탈출을 담당한다.

- 글로벌 best Δmargin 이 개선되면 (`global_best > self._best + 1e-9`) `level=0`, `_stale=0` 으로 리셋 — 탐색 집중 모드 복귀 (`continuous.py:72-76`).
- 개선이 없으면 `_stale += 1`, `_stale >= patience` 이면 `level += 1` (다양성 단계 상승) 후 stale 리셋 (`continuous.py:77-81`).
- 다음 epoch 변이 수 = `min(max_mutations_cap, base_mutations + level)` (`continuous.py:82`). 즉 정체가 길수록 변이 수가 cap(기본 6)까지 단조 증가.

epoch 마다 **seed 도 이동**시켜 탐색 영역을 옮긴다: `seed_base = base_config.seed_base + epoch * 1000` (`continuous.py:139`). 직전 epoch 이 결정한 `next_max_mutations` 를 다음 epoch 의 `max_random_mutations` 로 주입한다 (`continuous.py:141-142`).

실측 상태 파일 기준 현재 `diversity_level: 29`, `epochs_done: 96` (`runs/pyrosetta_flow/discovery_status.json` 헤더) — 장기간 best 정체 시 레벨이 계속 누적되나, 실제 변이 수는 cap(6)에서 포화된다 (`continuous.py:82` 의 `min`).

### 1.3 control 파일 핫 리로드 (재시작 불필요)

매 epoch 시작 시 control JSON 을 다시 읽는다: `control = _read_json(control_path)` (`continuous.py:135`). 안전 화이트리스트 `_CONTROL_FIELDS` (`continuous.py:28-32`)에 속한 필드만 `replace(base, **overrides)` 로 base config 에 덮어쓴다 (`_apply_control`, `continuous.py:51-54`). `template_pdb` 같은 구조적 필드는 화이트리스트에 없어 무시된다 — 실행 중 사람이 knobs 만 안전하게 조절 가능.

화이트리스트 필드: `n_candidates, max_iterations, top_k, selectivity_max_per_iter, max_random_mutations, rosetta_ddg_max, objective_mode, design_positions, validation_n_trials, selectivity_top_k` (`continuous.py:28-32`).

### 1.4 warm-start (run 간 누적 학습의 적용 지점)

엔진은 base config 에 `inloop_selectivity=True, reuse_baseline=True` 를 강제한다 (`continuous.py:110`). warm-start 는 runner 내부에서 두 층으로 작동한다.

1. **서열 dedup (experiment_log.jsonl)**: runner 가 prior records 를 로드해 (`runner.py:328`) `extract_historical_sequences()` 로 과거 시도 서열 집합을 만들고 (`runner.py:331`), dedup 에 사용. n_prior 를 stderr 로 보고 (`runner.py:334-335`).
2. **선택성 warm-start (global leaderboard)**: in-loop 선택성이 켜지면 `GlobalSelectivityLeaderboard.load()` 로 영속 리더보드를 읽고 (`runner.py:499`), 엔트리·screened 서열이 있으면 `_sel_leaderboard.seed_from_global(...)` 로 in-loop 리더보드를 warm-start 한다 (`runner.py:500-503`). 이로써 ① 역대 도킹 서열 재도킹 회피 ② 게이트 기준선(worst_ddg) 상향이 일어난다 (`selectivity_loop.py:23-41`).

### 1.5 baseline 1회 측정 후 캐시

native SST-14 baseline(비교 기준 ddG)은 매 epoch 재도킹하면 낭비이므로 첫 epoch 만 측정하고 캐시한다.

- 재사용 조건: `reuse_baseline=True` 이고 `baseline_cache.json` 이 존재하며, **같은 template_pdb + 같은 native 서열**일 때만 (`runner.py:349-354`). 안전 가드로 캐시 오염을 방지.
- 캐시 적중 시 재도킹을 생략하고 비교용 PDB 만 현재 epoch 경로로 복사, ddG/total/clash 를 캐시에서 복원 (`runner.py:358-366`).
- 캐시 미스 시 best-of-N trial 로 측정 후(`runner.py:368-392`), `reuse_baseline` 이면 `baseline_cache.json` + `baseline_cached.pdb` 로 저장 (`runner.py:394-408`). 저장 실패는 non-fatal (`runner.py:407-408`).

### 1.6 epoch 종료 시 영속 (ingest)

run 종료 시 이번 run 의 선택성 측정을 글로벌 리더보드에 누적한다: `_global_lb.ingest_artifacts(artifacts.to_dict())` → `_global_lb.save(_global_lb_path)` (`runner.py:1543-1548`). 영속 실패도 non-fatal (`runner.py:1549-1550`).

---

## ② 영향: run 간 누적 학습

엔진의 핵심 가치는 "단발 run 의 일회성 결과" 를 "디스크에 영속되는 누적 지식" 으로 바꾸는 데 있다. 두 영속 파일이 상보적으로 작동한다.

| 축 | 영속 파일 | 역할 | 인용 |
|----|----------|------|------|
| 서열 dedup + bandit | `experiment_log.jsonl` | 시도 서열 재생성 회피, 위치 탐색 warm-start | `runner.py:328-335`, `ranking.py:35-41` |
| 선택성(Δmargin) | `global_selectivity_leaderboard.json` | 역대 Δmargin best, 도킹 서열 dedup, in-loop warm-start | `global_leaderboard.py:42-67`, `runner.py:499-503` |
| 진행 상황 | `discovery_status.json` | epoch·역대 best·통과 수·다양성 레벨 | `continuous.py:190-200` |

**Δmargin = selectivity_margin − native_margin** (home-advantage 보정; >0 = native 초과 선택성) (`global_leaderboard.py:8`). Δ 미계산(post-loop 경로 등)이면 native baseline 으로 backfill 한다 (`global_leaderboard.py:94-98`): `_native_margin()` 이 `data/somatostatin_receptor/curated/native_selectivity_baseline.json` 의 `margin` 을 캐시 로드 (`global_leaderboard.py:28-39`).

**dedup 동작 형태**: dedup 은 write 시점이 아니라 **read 시점의 set 기반**이다. `experiment_log.jsonl` 은 append-only (`ranking.py:8-15`)이고, 로드 시 `record_type=="candidate"` 인 서열을 set 으로 추출해 중복을 무효화한다 (`ranking.py:35-41`). 글로벌 리더보드도 동일 서열은 더 높은 Δmargin 측정으로만 갱신하고(`global_leaderboard.py:111-118`), `screened_seqs` set 으로 재도킹을 막는다(`global_leaderboard.py:59-64`).

**누적 학습의 실증** (`runs/pyrosetta_flow/global_selectivity_leaderboard.json` 헤더): `n_screened_unique: 539`, `n_ingested_total: 564`, `n_unique: 50`(capacity), `best_delta_margin: 9.1021`. 즉 96 epoch 누적으로 539 개 고유 서열을 도킹했고, 중복 측정(564−539=25건)은 더 나은 Δ 만 유지 정책으로 흡수됐다.

**fail-open / fail-closed 균형**:
- 리더보드 로드 손상 시 빈 리더보드로 시작 (fail-open: 발굴은 계속) (`global_leaderboard.py:65-66`).
- 통과 판정은 독성 미측정(None)을 보수적으로 불통과 처리 (fail-closed: 가짜 통과 방지) (`global_leaderboard.py:167-179`).

---

## ③ 관련 Action Item: 무한 엔진 구축

메모리/진행 기록상 무한 발굴 엔진 구축은 SSTR2 선택성 GOAL(2026-06-10)의 산출물로 명시돼 있다. 코드 관점의 구성 요소는 다음 신규 모듈로 완결된다.

- `continuous.py` — epoch 루프 + DiversityPolicy + control/STOP/status 제어 (`continuous.py` 전체)
- `global_leaderboard.py` — 영속 Δmargin 리더보드 + ingest + backfill + warm-start payload (`global_leaderboard.py` 전체)
- `runner.py` 의 baseline_cache·global warm-start·ingest 통합 (`runner.py:344-408, 491-504, 1542-1550`)
- `scripts/run_continuous_discovery.py` — CLI 진입점 (`scripts/run_continuous_discovery.py` 전체)
- `scripts/autopush_results.sh` (루트) — 6h 결과 push (`scripts/autopush_results.sh` 전체)

프로젝트 진행 보고서가 이 Action Item 을 "선택성 방법론·인프라(무한 발굴 엔진) 100%" 로 기록한다 (`_workspace/reports/01_PROGRESS_REPORT_2026-06-17.md:15, :128`).

---

## ④ 완성도 평가

**완성도: 약 90%** (방법론·인프라 골격은 완성, 운영 자동화·문서의 일부가 외부 의존).

| 항목 | 상태 | 근거 |
|------|------|------|
| epoch 무한 루프 + 3종 종료(STOP/max/target) | 완성 | `continuous.py:125-152` |
| DiversityPolicy(정체 탈출·seed 이동·변이 cap) | 완성 | `continuous.py:57-84, 139-142` |
| control 핫 리로드(화이트리스트) | 완성 | `continuous.py:28-32, 135-136` |
| baseline 캐시(안전 가드 포함) | 완성 | `runner.py:344-408` |
| 글로벌 리더보드 영속(load/save/ingest/backfill/dedup) | 완성 | `global_leaderboard.py` 전체 |
| warm-start(experiment_log + global LB 2층) | 완성 | `runner.py:328-335, 499-503` |
| 회귀 테스트 | 완성(9개) | `pyrosetta_flow/tests/test_continuous_discovery.py:11-113` |
| atomic write(status/leaderboard) | 완성 | `continuous.py:44-48`, `global_leaderboard.py:80-82` |
| 실 운영 가동 | 가동 중 | `discovery_status.json`: `running:true, epochs_done:96, elapsed≈579,448s(≈6.7일)` |
| 6h autopush **스케줄러** | 부분 | 스크립트는 완성(`autopush_results.sh`)·커밋 cadence 는 6h 간격 실증(아래 ⑥)이나, cron/timer 등록 정의 파일이 리포에 미확인 → **스케줄링 주체 미검증** |

**테스트 커버리지** (`test_continuous_discovery.py`): dedup·best 갱신, save/load roundtrip, count_passing, ingest_artifacts, warm-start dedup·gate, capacity 채움 전 탐색, DiversityPolicy 정체-탈출·리셋, 변이 cap, control 화이트리스트 — 9개 (`test_continuous_discovery.py:11, 24, 36, 49, 70, 83, 94, 104, 113`).

**감점 사유 (≈10%)**:
1. autopush 의 6h 트리거(cron/systemd timer/tmux watch loop)가 리포 내 정의 파일로 확인되지 않음 — 스케줄링은 환경 외부에 존재할 가능성. 스크립트 자체는 멱등·fail-safe (`autopush_results.sh:25-35`). **미검증**.
2. `discovery_status.json` 의 `top[].ts` 가 빈 문자열 — 타임스탬프가 ingest 경로에서 누락(`global_leaderboard.py:99-109` 의 `ts=c.get("ts","")`, candidate 에 ts 부재). 기능 영향은 경미하나 추적성 손실.

---

## ⑤ 학술적 가치: 자율 발굴 루프

본 엔진은 **닫힌 자율 설계-측정-학습 루프(closed-loop autonomous discovery)** 를 펩타이드 방사성의약품 선택성 최적화에 구현한 사례다.

- **자율성**: 사람 개입 없이 STOP 까지 무한히 변이→도킹→선택성 측정→피드백을 반복. 수렴 시 종료가 아니라 다양성 주입으로 계속 탐색하는 것이 설계 의도 (`continuous.py:10-11`, `CONTINUOUS_DISCOVERY.md:70`).
- **explore/exploit 균형의 명시적 정책화**: DiversityPolicy 가 best 정체를 신호로 변이 수를 단조 증가시키고 개선 시 base 로 리셋 — 이는 simulated-annealing 류 재가열(reheating)/restart 휴리스틱을 epoch 단위로 구현한 것 (`continuous.py:69-84`).
- **영속 메모리에 의한 누적 학습**: run 을 거듭할수록 dedup 으로 탐색 공간 중복을 제거하고, 글로벌 리더보드가 게이트 기준선을 끌어올려 점점 "역대 best 보다 유망한 후보만" 비싼 off-target 도킹에 투입(비용-편익 최적화) (`selectivity_loop.py:43-56`).
- **정직성 가드**: home-advantage(Δmargin) 보정으로 native 대비 진짜 초과 선택성만 평가하고 (`global_leaderboard.py:8`), 독성 미측정을 통과로 인정하지 않는 fail-closed 통과 판정 (`global_leaderboard.py:167-179`). 환각·가짜 성공 방지를 방법론에 내장.

실측상 96 epoch / ≈6.7일 연속 가동으로 539 고유 서열을 스크리닝하고 best Δ=+9.10, 통과 47건을 누적 — 자율 루프가 실제로 점증적 발견을 산출함을 보인다 (`discovery_status.json` 헤더, `global_selectivity_leaderboard.json` 헤더).

---

## ⑥ 사용법

### 가동 (무한; STOP 파일로 정지)
```bash
ENV=~/miniforge3/envs/bio-tools/bin/python
cd AgenticAI4SCIENCE_pyrosetta_track/repos/ai4sci-kaeri
$ENV scripts/run_continuous_discovery.py \
    --input data/somatostatin_receptor/SSTR2_SST14_complex_boltz_1.pdb \
    --n-candidates 8 --max-iterations 4 --top-k 5 --selectivity-max-per-iter 2
```
(`CONTINUOUS_DISCOVERY.md:19-31`, CLI 인자: `scripts/run_continuous_discovery.py:26-55`)

tmux 등 장기 세션에서 백그라운드 가동 권장 — 엔진은 STOP 까지 무한 루프이므로 분리된 세션 필요(운영 관행; tmux 구체 명령은 리포 내 미확인 → **미검증**).

### 정지 (STOP 파일)
```bash
touch _workspace/STOP_DISCOVERY      # 현재 epoch 마치고 graceful 종료
rm   _workspace/STOP_DISCOVERY       # 다음 실행 전 반드시 삭제
```
(`CONTINUOUS_DISCOVERY.md:35-39`; STOP 검사 `continuous.py:126-129`, 기본 경로 `scripts/run_continuous_discovery.py:50-51`)

### 실시간 조절 (control 파일 — 재시작 불필요)
`_workspace/discovery_control.json` 편집 → 다음 epoch 부터 반영 (`continuous.py:135`). 실제 파일 내용 예 (`_workspace/discovery_control.json`):
```json
{ "n_candidates": 8, "max_iterations": 20, "top_k": 5,
  "selectivity_max_per_iter": 2, "rosetta_ddg_max": -15.0, "objective_mode": "auto",
  "patience": 3, "base_mutations": 3, "max_mutations_cap": 6,
  "target_pass_count": 3, "stop_on_target": false }
```
knobs 표는 `CONTINUOUS_DISCOVERY.md:44-52`.

### 모니터링
```bash
cat runs/pyrosetta_flow/discovery_status.json | python -m json.tool
# global_best_delta_margin, passing_count, diversity_level, top[], history[]
```
(`continuous.py:190-200`, `CONTINUOUS_DISCOVERY.md:73-76`)

### 6h autopush (외부 모니터링용 GitHub 스냅샷)
`scripts/autopush_results.sh` (루트 `SST14-M_scr/scripts/`)가 결과 경로만 `git add -f` 로 강제 스테이징(.gitignore 무시), 변경 없으면 skip, 실패해도 죽지 않음(다음 주기 재시도) (`autopush_results.sh:14-35`). push 대상: experiment_log, discovery_status, global leaderboard, baseline_cache, run 로그/산출물 (`autopush_results.sh:14-23`).
- **6h 주기 실증**: 커밋 cadence 가 00:22 → 06:21 → 12:21 → 18:22 → 00:22 로 약 6시간 간격 (git log autopush 커밋). 단 트리거 정의(cron/timer) 파일은 리포 내 미확인 → 스케줄러 주체 **미검증**.

---

## ⑦ 필요 이유

1. **단발 run 의 한계 극복**: 변이 공간(14aa × 20 AA)은 단발 run 으로 탐색 불가. epoch 루프 + seed 이동으로 공간을 점진 스윕한다 (`continuous.py:139`).
2. **비싼 측정의 재사용**: 선택성 도킹(후보 × off-target 수용체)은 고비용이라 매 iteration 전체 후보에 못 돌린다 (`selectivity_loop.py:3-6`). baseline 캐시(`runner.py:344-408`) + screened dedup(`global_leaderboard.py:59-64`)으로 중복 도킹을 제거해 GPU·시간 예산을 절약한다.
3. **수렴 정체 탈출**: 단순 best 추종은 local optimum 에 갇힌다. DiversityPolicy 가 정체를 감지해 변이 다양성을 끌어올려 탈출시킨다 (`continuous.py:69-84`).
4. **무중단 사람 조절**: 장기 가동 중 파라미터를 바꾸려 재시작하면 누적 상태가 끊긴다. control 핫 리로드로 재시작 없이 조절 (`continuous.py:135`).
5. **외부 가시성·내구성**: 6h autopush 로 결과를 GitHub 에 스냅샷해 원격 모니터링·복구 지점을 제공 (`autopush_results.sh:1-3`).
6. **정직한 평가의 영속화**: home-advantage Δmargin + fail-closed 통과 판정을 디스크에 누적해, run 간 비교가 환각 없이 일관되게 유지된다 (`global_leaderboard.py:8, 167-179`).

---

## 검증 인용 목록

- epoch 무한 while + STOP/max_epochs: `pyrosetta_flow/continuous.py:125-132`
- 목표 도달 자동 정지: `pyrosetta_flow/continuous.py:145-152`
- epoch 실패 격리(continue): `pyrosetta_flow/continuous.py:159-166`
- DiversityPolicy(정체→level↑, 개선→reset, 변이 cap): `pyrosetta_flow/continuous.py:57-84`
- seed 이동 + 직전 변이 수 주입: `pyrosetta_flow/continuous.py:139-142`
- control 화이트리스트 + 핫 리로드: `pyrosetta_flow/continuous.py:28-32, 135-136`
- atomic write(status): `pyrosetta_flow/continuous.py:44-48, 190-200`
- 엔진 강제 플래그(inloop_selectivity·reuse_baseline): `pyrosetta_flow/continuous.py:110`
- Δmargin 정의 + native backfill: `pyrosetta_flow/global_leaderboard.py:8, 28-39, 94-98`
- load fail-open / save atomic: `pyrosetta_flow/global_leaderboard.py:52-67, 69-82`
- add_measurement(동일 서열 더 나은 Δ만 유지): `pyrosetta_flow/global_leaderboard.py:85-121`
- ingest_artifacts: `pyrosetta_flow/global_leaderboard.py:123-150`
- count_passing(fail-closed 독성): `pyrosetta_flow/global_leaderboard.py:167-179`
- warm_start_payload: `pyrosetta_flow/global_leaderboard.py:181-183`
- baseline 캐시(안전 가드·복원·저장): `pyrosetta_flow/runner.py:344-408`
- experiment_log dedup 로드: `pyrosetta_flow/runner.py:328-335`, `pyrosetta_flow/ranking.py:8-15, 18-41`
- global LB warm-start 연결: `pyrosetta_flow/runner.py:491-504`
- epoch 종료 ingest·save: `pyrosetta_flow/runner.py:1542-1550`
- seed_from_global(dedup + 게이트 상향): `pyrosetta_flow/selectivity_loop.py:23-56`
- CLI 인자·기본 경로: `scripts/run_continuous_discovery.py:26-55, 61-90`
- autopush(강제 스테이징·skip·fail-safe): `scripts/autopush_results.sh:14-35`
- 회귀 테스트 9개: `pyrosetta_flow/tests/test_continuous_discovery.py:11, 24, 36, 49, 70, 83, 94, 104, 113`
- 실 운영 상태(96 epoch, best Δ9.10, 통과 47): `runs/pyrosetta_flow/discovery_status.json`(헤더), `runs/pyrosetta_flow/global_selectivity_leaderboard.json`(헤더)
- 6h 커밋 cadence: git log autopush 커밋(00:22/06:21/12:21/18:22)
- 사용법 출처: `_workspace/CONTINUOUS_DISCOVERY.md:17-81`

### 미검증 항목
- autopush 6h 트리거(cron/systemd/tmux watch)의 정의 파일 — 리포 내 미확인. 스크립트 자체와 커밋 간격 실증만 확인됨.
- tmux 백그라운드 가동의 구체 명령 — 리포 내 미확인(운영 관행 추정).
- `discovery_status.json` `top[].ts` 빈 문자열 — candidate ts 부재에 기인(기능 영향 경미).
