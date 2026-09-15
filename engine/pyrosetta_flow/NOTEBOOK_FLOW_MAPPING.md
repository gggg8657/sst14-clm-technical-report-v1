# PyRosetta Notebook-to-Pipeline Mapping

Reference notebook: `PRST_N_FM/notebooks/SSTR2_SST14_demo.ipynb` (read-only)
Architecture: [ARCHITECTURE.md](../ARCHITECTURE.md) (Silo B = pyrosetta_flow)

## Mapping Table

| Notebook Section | Notebook Intent | Fitted Pipeline Module |
|---|---|---|
| Template/Complex 준비 | SSTR2-SST14 기준 복합체를 출발점으로 고정 | `pyrosetta_flow.adapter.validate_config()` |
| Mutation 생성 | 고정 위치를 제외한 변이 후보 생성 | `pyrosetta_flow.adapter.generate_random_mutant()` |
| Mutate -> Dock -> ddG | 후보별 FlexPepDock + Rosetta ddG 계산 | `AG_src/scripts/flexpep_dock.py` 호출 (`pyrosetta_flow.runner`) |
| Iterative optimization | 결과 기반 다음 실험 가설 갱신 | `PlannerAgent -> ScientistCriticAgent` 루프 |
| Report/Notebook | iteration 요약과 rank/report 저장 | `ReporterAgent` + `FlowArtifacts` |

## Scope Notes

- 비교 시각화 플롯은 초기 범위에서 제외하고 JSON/Markdown artifact 중심으로 기록한다.
- 기존 `silo_a`, `silo_b`, `global_orchestrator`는 수정하지 않고 `ai4sci-kaeri` 내부 adapter/runner 경로로만 확장한다.

## Current Status (2026-03-04)

- Pipeline 테스트: 118 tests, 93% coverage (`pyrosetta_flow/tests/`)
- 리팩토링 22/22 완료 (보안 수정 C1-C5 포함)
- 전체 시스템 아키텍처: [ARCHITECTURE.md](../ARCHITECTURE.md)
