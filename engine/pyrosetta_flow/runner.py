from __future__ import annotations

import json
import os
import random
import shutil
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

from AG_src.agents.critic import ScientistCriticAgent
from AG_src.agents.planner import PlannerAgent
from AG_src.agents.qc_ranker import Candidate, PYROSETTA_ONLY_WEIGHTS, QCRankerAgent
from AG_src.agents.reporter import ReporterAgent
from AG_src.llm import create_provider
from backend.status_emitter import StatusEmitter

from .adapter import (
    candidate_to_dict,
    choose_objective_mode,
    get_bandit_guidance,
    generate_dedup_fallback,
    generate_guided_mutant,
    generate_intensification_mutant,
    generate_random_mutant,
    get_high_confidence_seeds,
    notebook_mapping,
    validate_config,
)
from AG_src.pipeline.step07_analysis import generate_pymol_renders
from .ranking import (
    append_experiment_records,
    build_historical_candidates,
    extract_historical_sequences,
    load_experiment_records,
    load_experiment_records_guarded,
    summarize_top_hits,
)
from .convergence import ConvergenceDetector
from .schema import CandidateResult, FlowArtifacts, FlowConfig, IterationSummary

try:
    from .rcsb_sequence_search import search_similar_peptides, SequenceSearchResult
    _HAS_RCSB = True
except ImportError:  # pragma: no cover
    _HAS_RCSB = False

# 2026-06-09 P1: GNINA/Pareto 옵셔널 의존은 scoring_pipeline.py 로 이동 (거기서만 사용).

# 다목적 통합 (반감기 + ADMET + 선택성) cheap-objective 스칼라
_HAS_MO = False
try:
    from .multiobjective import multiobjective_scalar as _multiobjective_scalar
    _HAS_MO = True
except ImportError:  # pragma: no cover
    _multiobjective_scalar = None  # type: ignore[assignment]


def _mo_scalar(extra_scores: Dict[str, Any], ddg: float):
    """extra_scores + ddg 로 다목적 스칼라 점수 계산 (UI 표시용)."""
    if not _HAS_MO:
        return None
    cand = dict(extra_scores or {})
    cand["ddg"] = ddg
    try:
        return _multiobjective_scalar(cand)
    except Exception:
        return None

BayesianPeptideOptimizer = None  # type: ignore[assignment]
OneHotEmbedder = None  # type: ignore[assignment]
_HAS_BO = False
try:
    from .bayesian_optimizer import BayesianPeptideOptimizer, OneHotEmbedder  # type: ignore[assignment]
    _HAS_BO = True
except ImportError:  # pragma: no cover
    pass


_HAS_PHARMA = False
_PharmaProperties = None  # type: ignore[assignment]
try:
    from AG_src.pipeline.pharma_properties import PharmaProperties as _PharmaProperties  # type: ignore[assignment]
    _HAS_PHARMA = True
except ImportError:  # pragma: no cover
    pass

_HAS_CLUSTER = False
_batch_classify = None  # type: ignore[assignment]
try:
    from .cluster_report import batch_classify as _batch_classify  # type: ignore[assignment]
    _HAS_CLUSTER = True
except ImportError:  # pragma: no cover
    pass

_HAS_CONTACT_FP = False
_analyze_contact_fingerprint_warning = None  # type: ignore[assignment]
try:
    from .contact_fingerprint import analyze_contact_fingerprint_warning as _analyze_contact_fingerprint_warning  # type: ignore[assignment]
    _HAS_CONTACT_FP = True
except ImportError:  # pragma: no cover
    pass


PHARMACOPHORE_POSITIONS_1IDX = (7, 8, 9, 10)
PHARMACOPHORE_RETRY_LIMIT = 3


def _pharmacophore_slice(sequence: str) -> str:
    return sequence[PHARMACOPHORE_POSITIONS_1IDX[0] - 1 : PHARMACOPHORE_POSITIONS_1IDX[-1]]


def _preserves_pharmacophore(sequence: str, reference_sequence: str) -> bool:
    return _pharmacophore_slice(sequence) == _pharmacophore_slice(reference_sequence)


def _disulfide_cys_positions(sequence: str) -> tuple:
    """참조 서열의 Cys 위치(1-indexed) — SST-14 의 Cys3-Cys14 이황화결합 보존용."""
    return tuple(i + 1 for i, a in enumerate(sequence.upper()) if a == "C")


def _preserves_disulfide(sequence: str, reference_sequence: str) -> bool:
    """2026-06-10: 참조의 모든 Cys 위치가 sequence 에서도 Cys 여야 한다 (이황화결합 보존).
    이전엔 가드 부재로 C14→H 변이가 통과해 SS bond 가 깨졌다."""
    cys = _disulfide_cys_positions(reference_sequence)
    return all(pos <= len(sequence) and sequence[pos - 1] == "C" for pos in cys)


def _preserves_scaffold(sequence: str, reference_sequence: str) -> bool:
    """FWKT pharmacophore + Cys 이황화 둘 다 보존."""
    return (_preserves_pharmacophore(sequence, reference_sequence)
            and _preserves_disulfide(sequence, reference_sequence))


def _mutable_design_positions(config: "FlowConfig") -> List[int]:
    # 2026-06-10: FWKT(7-10) 뿐 아니라 Cys 위치(이황화결합)도 변이 대상에서 제외.
    cys = set(_disulfide_cys_positions(config.original_sequence))
    mutable_positions = [
        pos for pos in config.design_positions
        if pos not in PHARMACOPHORE_POSITIONS_1IDX and pos not in cys
    ]
    return mutable_positions or list(config.design_positions)


def _append_discussion_log(
    discussion_log_path: Path,
    iteration: int,
    hypothesis: str,
    round_idx: int,
    expert_verdicts: Dict[str, Any],
    fanin: Dict[str, Any],
    final_focus: List[int],
    discussion_turns: Optional[List[Dict[str, Any]]] = None,
    discussion_rounds: int = 1,
    scientific_verdict: Optional[str] = None,
    docking_decision: Optional[bool] = None,
    reason_for_docking: Optional[str] = None,
    risk_level: Optional[str] = None,
    qc_status: Optional[str] = None,
    position_map_consistent: Optional[bool] = None,
    selectivity_evidence_present: Optional[bool] = None,
    minority_dissent: Optional[Dict[str, Any]] = None,
    stall_detected: Optional[bool] = None,
    stall_streak: Optional[int] = None,
) -> None:
    """5-전문가 토론 기록을 discussion_log.jsonl 에 append (iteration별 1줄).

    스키마::

        {
            "iteration": int,
            "hypothesis": str,
            "round": int,
            "expert_verdicts": [
                {"domain": str, "severity": str, "concerns": [str], "llm_backend": str}, ...
            ],
            "fanin": {
                "approve": bool,
                "merged_concerns": [str],
                "scientific_verdict": str,
                "docking_decision": bool,
                "reason_for_docking": str,
                "risk_level": str,
                "qc_status": str,
                "position_map_consistent": bool,
                "selectivity_evidence_present": bool,
                "minority_dissent": {"domain": str, "severity": str, "reason": str} | None,
                "stall_detected": bool,
                "stall_streak": int,
            },
            "final_focus": [int],
            "discussion_rounds": int,
            "discussion_turns": [...],
        }

    Args:
        discussion_log_path: discussion_log.jsonl 경로
        iteration: 현재 iteration 번호
        hypothesis: Planner 가설 문자열
        round_idx: pre-review 라운드 번호
        expert_verdicts: {domain: {severity, concerns}} dict (expert_panel 반환값)
        fanin: {approve, merged_concerns} dict
        final_focus: 최종 확정 focus_positions 리스트
        discussion_turns: 라운드별·전문가별 turn 기록 (expert_panel 반환값의 discussion_turns, 선택)
        discussion_rounds: 실제 수행된 토론 라운드 수 (선택, 기본 1)
        scientific_verdict: 과학적 판정 ("approve"|"conditional"|"reject"|"invalid")
        docking_decision: 실제 도킹 진행 여부
        reason_for_docking: 도킹 결정 사유
        risk_level: 위험 수준
        qc_status: QC 상태 ("pass"|"fail")
        position_map_consistent: position map 일관성 여부
        selectivity_evidence_present: 선택성 근거 존재 여부
        minority_dissent: 소수 반대의견 (C) — {domain, severity, reason} 또는 None
        stall_detected: 정체(stall) 감지 여부 (E)
        stall_streak: 동일 불일치 연속 라운드 수 (E)
    """
    # expert_verdicts dict → list 형식으로 정규화
    # llm_backend(F: 모델 이질성 추적, 예: "hetero:mistral-7b-instruct@..." vs "default:qwen3-32b")
    # 구버전 verdict에는 없을 수 있음 → "unknown" 폴백으로 하위호환 유지.
    verdicts_list = [
        {
            "domain": domain,
            "severity": v.get("severity", "low"),
            "concerns": v.get("concerns", []),
            "llm_backend": v.get("llm_backend", "unknown"),
        }
        for domain, v in (expert_verdicts or {}).items()
    ]
    _fanin_entry: Dict[str, Any] = {
        "approve": fanin.get("approve", True),
        "merged_concerns": fanin.get("merged_concerns", []),
    }
    # 신규 필드 전파 (None이 아닐 때만 기록)
    if scientific_verdict is not None:
        _fanin_entry["scientific_verdict"] = scientific_verdict
    if docking_decision is not None:
        _fanin_entry["docking_decision"] = docking_decision
    if reason_for_docking is not None:
        _fanin_entry["reason_for_docking"] = reason_for_docking
    if risk_level is not None:
        _fanin_entry["risk_level"] = risk_level
    if qc_status is not None:
        _fanin_entry["qc_status"] = qc_status
    if position_map_consistent is not None:
        _fanin_entry["position_map_consistent"] = position_map_consistent
    if selectivity_evidence_present is not None:
        _fanin_entry["selectivity_evidence_present"] = selectivity_evidence_present
    # C: 소수 반대의견 — 존재하면(None이 아니면) 기록. 값 자체가 None이어도 명시적으로 남긴다.
    if minority_dissent is not None:
        _fanin_entry["minority_dissent"] = minority_dissent
    if stall_detected is not None:
        _fanin_entry["stall_detected"] = stall_detected
    if stall_streak is not None:
        _fanin_entry["stall_streak"] = stall_streak

    entry = {
        "iteration": iteration,
        "ts": __import__("datetime").datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hypothesis": hypothesis,
        "round": round_idx,
        "expert_verdicts": verdicts_list,
        "fanin": _fanin_entry,
        "final_focus": list(final_focus or []),
        "discussion_rounds": discussion_rounds,
        "discussion_turns": discussion_turns or [],
    }
    try:
        discussion_log_path.parent.mkdir(parents=True, exist_ok=True)
        with discussion_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as _exc:
        print(
            f"  [discussion_log] append 실패 (무시): {_exc}",
            file=sys.stderr,
        )


def _rcsb_check_candidates(
    sequences: Dict[str, str],
    identity_cutoff: float = 0.4,
    max_results: int = 5,
) -> Dict[str, list]:
    """RCSB PDB에서 후보 서열의 유사 구조를 검색합니다 (best-effort).

    네트워크 미연결 또는 rcsb_sequence_search 미설치 시 빈 결과 반환.

    Args:
        sequences: {candidate_id: amino_acid_sequence} 매핑
        identity_cutoff: 최소 서열 동일성 (0.0~1.0)
        max_results: 후보당 최대 히트 수

    Returns:
        {candidate_id: [{"pdb_id", "identity", "evalue"}, ...]}
    """
    if not _HAS_RCSB or not sequences:
        return {}

    results: Dict[str, list] = {}
    for cand_id, seq in sequences.items():
        if not seq or len(seq) < 5:
            continue
        try:
            search_result = search_similar_peptides(
                sequence=seq,
                identity_cutoff=identity_cutoff,
                max_results=max_results,
            )
            hits = []
            for hit in search_result.hits:
                hits.append({
                    "pdb_id": hit.pdb_id,
                    "identifier": hit.identifier,
                    "identity": hit.sequence_identity,
                    "evalue": hit.evalue,
                    "bitscore": hit.bitscore,
                })
            if hits:
                results[cand_id] = hits
        except Exception as exc:
            print(f"  [rcsb] {cand_id} search failed: {exc}", file=sys.stderr)
    return results


# 2026-06-09 P1 분해: 대안 스코어링 체인(GNINA/ECR/Pareto/BO + cheap objectives)을
# scoring_pipeline.py 로 추출. 하위호환: 기존 import 경로 보존을 위해 re-export.
from .scoring_pipeline import _apply_alternative_scoring  # noqa: E402,F401


# 2026-06-09 P1 분해: 도킹 subprocess 실행 레이어를 docking_executor.py 로 추출.
# 하위호환: 기존 `from pyrosetta_flow.runner import _run_script` 등을 위해 re-export.
from .docking_executor import _resolve_conda_python, _run_script  # noqa: E402,F401


def _read_pipeline_config(repo_root: Path) -> Dict[str, Any]:
    cfg_path = repo_root / "AG_src" / "config" / "pipeline_config.yaml"
    if not cfg_path.exists() or yaml is None:
        return {}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def _resolve_llm_model(cli_model: str | None) -> str | None:
    """Resolve model name from CLI arg > LLM_MODEL env var > None (use config file).

    Priority: --llm-model CLI argument > LLM_MODEL env var > config file default.
    """
    if cli_model:
        return cli_model
    env_model = os.environ.get("LLM_MODEL")
    if env_model:
        return env_model
    return None


def _resolve_planner_llm(base_llm_cfg: Dict[str, Any], shared_llm: Any) -> Any:
    """Planner 전용 LLM provider를 결정한다.

    env PLANNER_MODEL/PLANNER_BASE_URL이 하나라도 지정되면 planner만 별도
    provider(교체 가능한 강한 모델 등)를 쓰도록 새 provider를 만든다.
    둘 다 미지정이면 기존 동작대로 critic/reporter와 동일한 shared_llm을 반환한다
    (하위호환 — 기존 "전 에이전트 동일 모델 공유" 동작 완전 보존).

    base_llm_cfg의 provider/api_key/timeout 등은 상속하고, model/base_url만
    PLANNER_MODEL/PLANNER_BASE_URL로 override한다. base_url만 지정되고
    model이 없으면 base_llm_cfg의 기존 model을 그대로 쓰며, 그 반대도 동일하다.

    Args:
        base_llm_cfg: critic/reporter가 사용하는 llm 설정 dict (pipeline_config.yaml
            구조 — {"llm": {"provider":..., "model":..., "base_url":..., ...}}).
        shared_llm: critic/reporter/패널이 공유하는 이미 생성된 LLMProvider 인스턴스.

    Returns:
        PLANNER_MODEL/PLANNER_BASE_URL 미지정 시 shared_llm 그대로,
        지정 시 별도로 생성한 LLMProvider 인스턴스.
    """
    planner_model = os.environ.get("PLANNER_MODEL")
    planner_base_url = os.environ.get("PLANNER_BASE_URL")
    if not planner_model and not planner_base_url:
        return shared_llm

    _base = (base_llm_cfg or {}).get("llm", {}) or {}
    _planner_llm_section: Dict[str, Any] = dict(_base)
    if planner_model:
        _planner_llm_section["model"] = planner_model
    if planner_base_url:
        _planner_llm_section["base_url"] = planner_base_url
    # provider 미지정 시 base_llm_cfg의 기존 provider(기본 vllm) 상속.
    _planner_llm_section.setdefault("provider", "vllm")
    # Mistral 계열은 chat_template_kwargs 미지원(HTTP 400) → 자동으로 끈다.
    # env PLANNER_SEND_CHAT_TEMPLATE_KWARGS로 명시 override 가능(0/1).
    _sctk_env = os.environ.get("PLANNER_SEND_CHAT_TEMPLATE_KWARGS")
    if _sctk_env is not None:
        _planner_llm_section["send_chat_template_kwargs"] = _sctk_env not in ("0", "false", "False", "")
    elif "mistral" in (planner_model or "").lower():
        _planner_llm_section["send_chat_template_kwargs"] = False
    print(
        f"  [planner-model] PLANNER_MODEL/PLANNER_BASE_URL 감지 — planner 전용 provider 생성: "
        f"provider={_planner_llm_section.get('provider')}, model={_planner_llm_section.get('model')}, "
        f"base_url={_planner_llm_section.get('base_url')}",
        file=sys.stderr,
    )
    return create_provider({"llm": _planner_llm_section})


def _candidate_to_qc(candidate: CandidateResult, seq_id: int, objective_mode: str) -> Candidate:
    return Candidate(
        candidate_id=candidate.candidate_id,
        backbone_id=0,
        seq_id=seq_id,
        sequence=candidate.sequence,
        plddt_mean=0.0,     # Not available in PyRosetta-only mode
        plddt_interface=0.0, # Not available in PyRosetta-only mode
        dock_score=0.0,      # Not available in PyRosetta-only mode
        ddg=candidate.ddg,
        clash_count=int(candidate.clash_score),
        constraint_violations=0,
        lddt=0.0,            # Not available in PyRosetta-only mode
    )


def _summarize_iteration(
    iteration: int,
    run_id: str,
    hypothesis: str,
    objective_mode: str,
    selected: List[CandidateResult],
    report_paths: Dict[str, str],
    critic_hypothesis: str,
) -> IterationSummary:
    if selected:
        ddgs = [c.ddg for c in selected]
        best_ddg = min(ddgs)
        mean_ddg = statistics.mean(ddgs)
    else:
        best_ddg = 0.0
        mean_ddg = 0.0
    return IterationSummary(
        iteration=iteration,
        run_id=run_id,
        hypothesis=hypothesis,
        objective_mode=objective_mode,
        n_candidates=len(selected),
        best_ddg=round(best_ddg, 4),
        mean_ddg=round(mean_ddg, 4),
        selected_ids=[c.candidate_id for c in selected],
        critic_hypothesis=critic_hypothesis,
        report_paths=report_paths,
    )


def _emit_candidates(
    emitter: StatusEmitter,
    candidates: List[CandidateResult],
    ddg_threshold: float = -5.0,
    flow_dir: Path | None = None,
) -> None:
    emitter.set_candidates(
        [
            {
                "rank": idx + 1,
                "id": c.candidate_id,
                "sequence": c.sequence,
                "ddG": round(c.ddg, 3),
                "totalScore": round(c.total_score, 3),
                "clashScore": round(c.clash_score, 1),
                "finalScore": round(-c.ddg, 3),
                "result": (
                    "PASS" if c.selected else
                    "PASS" if (c.ddg <= ddg_threshold and c.ddg < 900) else
                    "FAIL"
                ),
                "failReason": c.fail_reason if c.fail_reason else "",
                **({"pdb_path": str(flow_dir / f"iter_{c.iteration:02d}" / f"cand_{int(c.candidate_id.split('cand')[1]):03d}.pdb")}
                   if flow_dir and "cand" in c.candidate_id else {}),
            }
            for idx, c in enumerate(sorted(candidates, key=lambda x: x.ddg))
        ]
    )


def run_pyrosetta_agentic_mutdock_flow(config: FlowConfig) -> FlowArtifacts:
    validate_config(config)
    repo_root = Path(__file__).resolve().parent.parent
    flow_dir = repo_root / config.output_dir / "sst14_agentic_mutdock"
    flow_dir.mkdir(parents=True, exist_ok=True)
    flexpep_script = repo_root / "AG_src" / "scripts" / "flexpep_dock.py"
    run_id = f"sst14_mutdock_{config.seed_base}"

    # E1: seed_base → np/torch 전역 시드 설정 (재현성; None이면 기존 비결정적 동작)
    if config.seed_base is not None:
        import numpy as _np_seed
        _np_seed.random.seed(config.seed_base)
        try:
            import torch as _torch_seed
            _torch_seed.manual_seed(config.seed_base)
        except ImportError:
            pass
        print(f"  [runner] 전역 seed 설정: {config.seed_base}", file=sys.stderr)
    exp_log_path = repo_root / config.output_dir / "experiment_log.jsonl"
    run_status = "success"
    run_failure_stage = ""
    run_error_summary = ""
    run_records: List[Dict[str, Any]] = []
    # [작업1] iteration partial flush 추적: 이미 experiment_log에 쓴 인덱스 상한
    _exp_flush_idx: int = 0

    model_override = _resolve_llm_model(config.llm_model_override)
    # FlowConfig에 llm_provider/llm_base_url가 지정되면 pipeline_config 대신 사용
    if getattr(config, 'llm_provider', None) and getattr(config, 'llm_base_url', None):
        _llm_cfg = {
            "llm": {
                "provider": config.llm_provider,
                "model": model_override or config.llm_model_override or "qwen3:8b",
                "base_url": config.llm_base_url,
            }
        }
        llm = create_provider(_llm_cfg, model_override=model_override)
    else:
        _llm_cfg = _read_pipeline_config(repo_root)
        llm = create_provider(_llm_cfg, model_override=model_override)
    # Planner 전용 모델 교체: PLANNER_MODEL/PLANNER_BASE_URL env가 지정되면
    # critic/reporter/패널(P1/P2)은 기존 llm(공유 Qwen)을 그대로 쓰고,
    # planner만 별도 provider(예: 인프라팀이 준비하는 강한 모델, 포트 8002 등)로 교체한다.
    # 미지정 시 기존 동작(전 에이전트 동일 llm 공유) 완전히 보존.
    planner_llm = _resolve_planner_llm(base_llm_cfg=_llm_cfg, shared_llm=llm)
    emitter = StatusEmitter(run_id=run_id, total_iterations=config.max_iterations, llm_model=str(llm))
    # PyRosetta-only flow는 step01~05/05b를 건너뜀 — "pending" 채로 방치하면 UI 혼란
    for _skip_id in ["step01", "step02", "step03", "step03b", "step04", "step05", "step05b"]:
        emitter.update_step(_skip_id, "skipped")
    planner = PlannerAgent(llm_provider=planner_llm, planner_mode=config.planner_mode)
    critic = ScientistCriticAgent(llm_provider=llm)
    reporter = ReporterAgent(runs_base_dir=str(repo_root / config.output_dir), llm_provider=llm)
    qcranker = QCRankerAgent(weights=PYROSETTA_ONLY_WEIGHTS, llm_provider="none")
    # 2026-07-01: 데이터 무결성 가드 — warm-start 시점에 experiment_log.jsonl 이
    # 이전 high-water-mark보다 줄어들었으면(외부 요인에 의한 유실) 강하게 경고하고
    # 인시던트 마커를 남긴다. (파이프라인은 계속 진행 — fail-open으로 무한정지시키지 않음)
    prior_records = load_experiment_records_guarded(exp_log_path)
    emitter.set_historical_candidates(build_historical_candidates(prior_records))
    # Cross-run dedup: seed seen_sequences with all historically tried sequences
    historical_sequences = extract_historical_sequences(prior_records)
    historical_top_hits = summarize_top_hits(prior_records, top_n=10)
    n_prior = len(historical_sequences)
    if n_prior:
        print(f"  [history] Loaded {n_prior} unique sequences from prior runs (dedup enabled)", file=sys.stderr)
        print(f"  [history] Top prior hit: {historical_top_hits[0]['sequence']} ddG={historical_top_hits[0]['ddg']}" if historical_top_hits else "  [history] No successful prior candidates", file=sys.stderr)

    t_baseline = emitter.start_step("step06_baseline")
    t_prepare0 = emitter.start_rosetta_substep("step06_prepare")
    n_baseline_trials = config.n_baseline_trials
    emitter.append_timeline_event(0, "rosetta.prepare", "running", f"Preparing baseline ({n_baseline_trials} trials, best-of)")
    baseline_out = flow_dir / "baseline_refined.pdb"
    baseline: Dict[str, Any] = {}
    # 무한 엔진: native SST-14 baseline 을 epoch 간 캐시 (첫 도킹만 측정, 이후 재사용).
    # 변이는 template_pdb 에서 시작 → baseline 은 비교 기준값일 뿐이라 재도킹 불필요.
    _baseline_cache = repo_root / config.output_dir / "baseline_cache.json"
    _baseline_cache_pdb = repo_root / config.output_dir / "baseline_cached.pdb"
    _cached = None
    if getattr(config, "reuse_baseline", False) and _baseline_cache.exists():
        try:
            _c = json.loads(_baseline_cache.read_text(encoding="utf-8"))
            # 같은 template + 같은 native 서열일 때만 재사용 (안전)
            if _c.get("template_pdb") == config.template_pdb and _c.get("sequence") == config.original_sequence:
                _cached = _c
        except Exception as exc:
            print(f"  [baseline] 캐시 로드 실패(재도킹): {exc}", file=sys.stderr)
    try:
        if _cached is not None:
            # 캐시 재사용 — 재도킹 생략, 비교용 PDB 만 현재 epoch 경로로 복사
            if _baseline_cache_pdb.exists():
                shutil.copy2(str(_baseline_cache_pdb), str(baseline_out))
            best_baseline_ddg = float(_cached.get("ddg", 0.0))
            baseline = {"ddg": best_baseline_ddg,
                        "total_score": float(_cached.get("total_score", 0.0)),
                        "clash_score": float(_cached.get("clash_score", 0.0))}
            print(f"  [baseline] 캐시 재사용 (native SST-14 재도킹 생략): ddG={best_baseline_ddg:.3f}", file=sys.stderr)
        else:
            # Run multiple baseline refinements and pick best by ddG
            best_baseline: Dict[str, Any] = {}
            best_baseline_ddg = float("inf")
            for trial_idx in range(n_baseline_trials):
                trial_out = flow_dir / f"baseline_trial_{trial_idx}.pdb"
                trial_result = _run_script(
                    flexpep_script,
                    [
                        "--input", config.template_pdb,
                        "--output", str(trial_out),
                        "--protocol", "flexpep_refine",
                        "--peptide-chain", str(config.peptide_chain),
                    ],
                    config.conda_env,
                    repo_root,
                    timeout=config.script_timeout,
                )
                trial_ddg = float(trial_result.get("ddg", 999.0))
                print(f"  [baseline] trial {trial_idx + 1}/{n_baseline_trials}: ddG={trial_ddg:.3f} total={float(trial_result.get('total_score', 0)):.3f}", file=sys.stderr)
                if trial_ddg < best_baseline_ddg:
                    best_baseline_ddg = trial_ddg
                    best_baseline = trial_result
                    # Copy best trial PDB to canonical baseline path
                    shutil.copy2(str(trial_out), str(baseline_out))
            baseline = best_baseline
            # 캐시 저장 (다음 epoch 재사용용) — reuse_baseline 일 때만
            if getattr(config, "reuse_baseline", False) and best_baseline:
                try:
                    _baseline_cache.parent.mkdir(parents=True, exist_ok=True)
                    _baseline_cache.write_text(json.dumps({
                        "template_pdb": config.template_pdb,
                        "sequence": config.original_sequence,
                        "ddg": best_baseline_ddg,
                        "total_score": float(baseline.get("total_score", 0.0)),
                        "clash_score": float(baseline.get("clash_score", 0.0)),
                    }, indent=2, ensure_ascii=False), encoding="utf-8")
                    if baseline_out.exists():
                        shutil.copy2(str(baseline_out), str(_baseline_cache_pdb))
                    print(f"  [baseline] 캐시 저장 (이후 epoch 재사용): {_baseline_cache.name}", file=sys.stderr)
                except Exception as exc:
                    print(f"  [baseline] 캐시 저장 실패(non-fatal): {exc}", file=sys.stderr)
        emitter.complete_rosetta_substep("step06_prepare", t_prepare0)
        emitter.append_timeline_event(0, "rosetta.prepare", "completed", f"Baseline ready (best of {n_baseline_trials}: ddG={best_baseline_ddg:.1f})")
        baseline_ddg = float(baseline.get("ddg", 0.0))
        baseline_total = float(baseline.get("total_score", 0.0))
        baseline_clash = float(baseline.get("clash_score", 0.0))

        # Adaptive gate: set initial thresholds relative to baseline
        if getattr(config, "gate_mode", "static") == "adaptive" and baseline_ddg < 0:
            # Initial gate = 10% of baseline ddG (e.g., baseline=-48 → gate=-4.8)
            initial_ddg_gate = round(baseline_ddg * 0.1, 1)
            initial_clash_gate = max(int(baseline_clash * 2), 10)
            config.rosetta_ddg_max = initial_ddg_gate
            config.rosetta_clash_max = initial_clash_gate
            print(f"  [adaptive] Initial gates from baseline: ddG≤{initial_ddg_gate} clash≤{initial_clash_gate}", file=sys.stderr)

        emitter.set_baseline({
            "sequence": config.original_sequence,
            "pdb": str(baseline_out),
            "ddg": baseline_ddg,
            "total_score": baseline_total,
            "clash_score": baseline_clash,
        })
        # Add baseline as reference candidate in the ranking table
        emitter.set_candidates([{
            "rank": 0,
            "id": "baseline_SST14",
            "sequence": config.original_sequence,
            "ddG": round(baseline_ddg, 3),
            "totalScore": round(baseline_total, 3),
            "clashScore": round(baseline_clash, 1),
            "finalScore": round(-baseline_ddg, 3),
            "result": "REF",
            "failReason": "",
        }])
        emitter.complete_step("step06_baseline", t_baseline)
    except Exception as exc:
        emitter.fail_rosetta_substep("step06_prepare", t_prepare0)
        emitter.append_timeline_event(0, "rosetta.prepare", "failed", f"Baseline preparation failed: {exc}")
        emitter.fail_step("step06_baseline", t_baseline)
        run_status = "completed_with_warnings"
        run_failure_stage = "baseline_prepare"
        run_error_summary = str(exc)
        # Fail-open: keep loop alive to produce candidates/ranking from iterations.
        emitter.append_timeline_event(0, "runner", "running", "Fail-open enabled: continue without baseline pose")

    iterations_out: List[Dict[str, Any]] = []
    final_selected: List[CandidateResult] = []
    # Cross-run dedup 강화: historical 서열 전체를 seen_sequences seed로 사용.
    # 이로써 13,990건 중 중복 3,178건→재평가 방지.
    # (과거: current run만 dedup, 중복률 77% 원인)
    seen_sequences: set = {config.original_sequence} | set(historical_sequences)
    if historical_sequences:
        print(
            f"  [dedup] seen_sequences seed: native + {len(historical_sequences)} historical "
            f"→ cross-run 중복 완전 차단",
            file=sys.stderr,
        )
    critic_feedback: Dict[str, Any] = {}

    # ---------------------------------------------------------------------------
    # Intensification / Random 비율 환경변수
    # ---------------------------------------------------------------------------
    # INTENSIFY_HIGH_CONF=1: high_confidence seed 주변 집중 탐색 활성화
    # INTENSIFY_RATIO=0.3: 전체 n_candidates 중 intensification 할당 비율 (기본 0.0=비활성)
    # RANDOM_RATIO=0.25: 전체 n_candidates 중 순수 random 변이 비율 (기본 0.11→0.25 상향)
    # DEDUP_RETRY_MAX=2: guided/intensification 슬롯 중복 시 dedup_fallback 재시도 횟수
    _intensify_enabled = os.environ.get("INTENSIFY_HIGH_CONF", "0").strip() not in ("0", "false", "False")
    try:
        _intensify_ratio = float(os.environ.get("INTENSIFY_RATIO", "0.30"))
    except ValueError:
        _intensify_ratio = 0.30
    try:
        _random_ratio = float(os.environ.get("RANDOM_RATIO", "0.25"))
    except ValueError:
        _random_ratio = 0.25
    try:
        _dedup_retry_max = int(os.environ.get("DEDUP_RETRY_MAX", "2"))
    except ValueError:
        _dedup_retry_max = 2
    # intensification guidance 캐시 (epoch 경계마다 mmgbsa 파일에서 갱신)
    _intensify_seeds: List[Dict[str, Any]] = []  # [{sequence, dg_bind, consensus_flag}, ...]
    _intensify_guidance_cache: Dict[str, Any] = {}  # seed_seq → guidance dict
    if _intensify_enabled:
        print(
            f"  [intensify] INTENSIFY_HIGH_CONF=1, ratio={_intensify_ratio:.0%}, "
            f"RANDOM_RATIO={_random_ratio:.0%}",
            file=sys.stderr,
        )

    # Multi-Armed Bandit: data-driven fallback for focus_positions
    bandit_guidance: Dict[str, Any] = {}
    _bandit_instance = None  # PositionBandit 인스턴스 (직접 재샘플용)
    if prior_records:
        try:
            bandit_guidance = get_bandit_guidance(prior_records, n_focus=config.bandit_n_focus)
            print(f"  [bandit] Thompson sampling focus: {bandit_guidance.get('focus_positions', [])}", file=sys.stderr)
            # bandit 인스턴스 직접 생성 (exploration boost 재샘플 필요 시)
            try:
                from .bandit import PositionBandit
                _bandit_instance = PositionBandit()
                _bandit_instance.initialize_from_history(prior_records)
            except Exception:
                _bandit_instance = None
        except Exception as exc:
            print(f"  [bandit] Initialization failed (non-fatal): {exc}", file=sys.stderr)

    # 작업 A: tried-focus 블랙리스트 — 최근 K iteration 의 focus frozenset 누적
    # 환경변수: FOCUS_BLACKLIST_ENABLED=1(기본), FOCUS_BLACKLIST_K=8(기본)
    _focus_blacklist_enabled = os.environ.get("FOCUS_BLACKLIST_ENABLED", "1").strip() not in ("0", "false", "False")
    _focus_blacklist_k = int(os.environ.get("FOCUS_BLACKLIST_K", "8"))
    _tried_focus: List[Any] = []  # frozenset[int] 목록, 최대 K개 유지
    convergence_detector = ConvergenceDetector(
        window_size=config.convergence_window_size,
        significance_level=config.convergence_significance,
    )

    # BO optimizer: iteration 간 공유 (fit은 매 iteration 내에서 수행)
    _bo_optimizer: Optional[Any] = None
    if _HAS_BO:
        try:
            _bo_optimizer = BayesianPeptideOptimizer(
                embedder=OneHotEmbedder(max_len=len(config.original_sequence)),
                objectives=["ddg", "ecr_score"],
                maximize=[False, True],  # ddg 최소화, ecr_score 최대화
            )
            print("  [bo] BayesianPeptideOptimizer initialized", file=sys.stderr)
        except Exception as exc:
            print(f"  [bo] Initialization failed (non-fatal): {exc}", file=sys.stderr)
            _bo_optimizer = None

    # B2: BO 제안 포지션 iteration 간 전달 버퍼 (None=미실행, []=실행했으나 제안 없음)
    _bo_suggested: List[int] = []

    # 작업 B: MM-GBSA 데몬 consensus 신호 — epoch 경계마다 비동기 소비
    # 환경변수:
    #   MMGBSA_CONSENSUS_ENABLED=1 (기본 활성화)
    #   MMGBSA_BANDIT_BOOST=1.5 (high_confidence 위치 alpha 가중)
    #   MMGBSA_LEADERBOARD_BOOST=3.0 (리더보드 랭킹 부스트 — global_leaderboard 가 직접 사용)
    # caveat: MM-GBSA 는 45분 주기 단일 스냅샷 비동기 신호.
    #         도킹 대체 아님, 직교 보조 신호로만 사용.
    _mmgbsa_enabled = os.environ.get("MMGBSA_CONSENSUS_ENABLED", "1").strip() not in ("0", "false", "False")
    _mmgbsa_bandit_boost = float(os.environ.get("MMGBSA_BANDIT_BOOST", "1.5"))
    # mmgbsa_consensus.json 기본 경로: output_dir(runs/pyrosetta_flow) 직하
    _mmgbsa_path = repo_root / config.output_dir / "mmgbsa_consensus.json"
    _mmgbsa_cache: Optional[Dict[str, Any]] = None  # 마지막으로 읽은 payload (stale 감지용)

    # 2026-06-10: in-loop 선택성 리더보드 (조건부 게이트). config.inloop_selectivity 시 매 iteration
    # 유망(ddG 강한) 후보만 off-target 도킹 → Δmargin(native 보정) → Planner/Critic 피드백.
    _sel_leaderboard = None
    _global_lb = None
    _global_lb_path = repo_root / config.output_dir / "global_selectivity_leaderboard.json"
    if getattr(config, "inloop_selectivity", False):
        try:
            from .selectivity_loop import SelectivityLeaderboard
            from .global_leaderboard import GlobalSelectivityLeaderboard
            _sel_leaderboard = SelectivityLeaderboard(capacity=config.top_k)
            # 무한 엔진: 글로벌 리더보드로 warm-start (역대 도킹 서열 dedup + 게이트 기준선)
            _global_lb = GlobalSelectivityLeaderboard.load(_global_lb_path)
            if _global_lb.entries or _global_lb.screened_seqs:
                _sel_leaderboard.seed_from_global(_global_lb.warm_start_payload())
                print(f"  [sel-loop] 글로벌 warm-start: {len(_global_lb.screened_seqs)} 서열 기측정, "
                      f"역대 best Δ={_global_lb.best_delta()}", file=sys.stderr)
            print("  [sel-loop] in-loop selectivity 활성화 (조건부 게이트)", file=sys.stderr)
        except Exception as exc:
            print(f"  [sel-loop] init 실패(non-fatal): {exc}", file=sys.stderr)

    for iteration in range(1, config.max_iterations + 1):
        emitter.set_iteration(iteration)
        emitter.reset_rosetta_substeps()
        objective_mode = choose_objective_mode(config.objective_mode, iteration)

        # 작업 B: epoch 경계에서 mmgbsa_consensus.json 읽어 bandit 에 직교 신호 반영
        # caveat: 비동기 45분 주기 데몬 산출 — 파일 없으면 graceful skip
        if _mmgbsa_enabled and _bandit_instance is not None and _mmgbsa_path.exists():
            try:
                _mmgbsa_raw = json.loads(_mmgbsa_path.read_text(encoding="utf-8"))
                _mmgbsa_ts = _mmgbsa_raw.get("generated_at", "")
                # stale 감지: 이전에 읽은 것과 같은 타임스탬프면 skip (중복 적립 방지)
                _prev_ts = (_mmgbsa_cache or {}).get("generated_at", "")
                if _mmgbsa_ts != _prev_ts:
                    _mmgbsa_results_dict = _mmgbsa_raw.get("results", {})
                    _native_dg = float(_mmgbsa_raw.get("native_dg", 0.0))
                    _n_updated = _bandit_instance.update_from_mmgbsa(
                        mmgbsa_results=_mmgbsa_results_dict,
                        native_dg=_native_dg,
                        boost_weight=_mmgbsa_bandit_boost,
                    )
                    _mmgbsa_cache = _mmgbsa_raw
                    _n_hc = sum(
                        1 for v in _mmgbsa_results_dict.values()
                        if v.get("consensus_flag") == "high_confidence"
                    )
                    print(
                        f"  [mmgbsa] 새 파일 소비(ts={_mmgbsa_ts}): "
                        f"high_confidence={_n_hc}건, bandit arms 업데이트={_n_updated}. "
                        f"caveat: 45분 주기 단일 스냅샷 비동기 신호 — 보조 전용",
                        file=sys.stderr,
                    )
                else:
                    pass  # 동일 타임스탬프: 중복 skip (로그 불필요)
            except Exception as _mmgbsa_exc:
                print(
                    f"  [mmgbsa] 읽기 실패(graceful skip): {_mmgbsa_exc}",
                    file=sys.stderr,
                )

        # Intensification seed 갱신: mmgbsa_consensus.json에서 high_confidence 서열 로드
        # (INTENSIFY_HIGH_CONF=1 이고 파일이 존재할 때마다 epoch 경계에서 갱신)
        if _intensify_enabled and _mmgbsa_path.exists():
            try:
                _new_seeds = get_high_confidence_seeds(_mmgbsa_path)
                if _new_seeds:
                    _intensify_seeds = _new_seeds
                    # guidance 캐시 재구성 (seed 서열별로 사전 계산)
                    from .bandit import get_intensification_guidance
                    _mutation_positions_for_intens = [
                        pos for pos in config.design_positions
                        if pos not in (7, 8, 9, 10) and pos not in (3, 14)
                    ]
                    _intensify_guidance_cache = {}
                    for _seed_entry in _intensify_seeds:
                        _sseq = _seed_entry["sequence"]
                        _ig = get_intensification_guidance(
                            seed_sequence=_sseq,
                            reference_sequence=config.original_sequence,
                            mutable_positions=_mutation_positions_for_intens,
                            rng=random.Random(config.seed_base + iteration * 7),
                        )
                        _intensify_guidance_cache[_sseq] = _ig
                    print(
                        f"  [intensify] seed 갱신: {len(_intensify_seeds)}건 high_confidence. "
                        f"best={_intensify_seeds[0]['sequence']} dg={_intensify_seeds[0]['dg_bind']:.1f}",
                        file=sys.stderr,
                    )
            except Exception as _intens_exc:
                print(
                    f"  [intensify] seed 로드 실패(non-fatal): {_intens_exc}",
                    file=sys.stderr,
                )


        emitter.append_timeline_event(iteration, "planner", "running", "Planner generating hypothesis")
        emitter.update_agent("planner", status="active", message=f"Iteration {iteration} planning")
        prev_results: Dict[str, Any] = {"objective_mode": objective_mode}
        if final_selected:
            prev_results["top_candidates"] = [
                {
                    "sequence": c.sequence,
                    "ddg": c.ddg,
                    "id": c.candidate_id,
                    # surrogate 값 전달 (계산 X — extra_scores에서 읽기만)
                    "half_life_h": c.extra_scores.get("half_life_h"),
                    "admet_score": c.extra_scores.get("admet_score"),
                    "hc50": c.extra_scores.get("hc50"),
                }
                for c in sorted(final_selected, key=lambda x: x.ddg)[:5]
            ]
            prev_results["best_ddg"] = min(c.ddg for c in final_selected)
        # 2026-06-10: in-loop 선택성 리더보드를 Planner 에 피드백 (Δmargin>0 = native 초과 선택성)
        if _sel_leaderboard is not None and _sel_leaderboard.entries:
            prev_results["selectivity_leaderboard"] = _sel_leaderboard.summary()
            prev_results["best_delta_margin"] = _sel_leaderboard.best_delta()
        if historical_top_hits:
            prev_results["historical_top_hits"] = historical_top_hits
            prev_results["n_historical_sequences"] = n_prior
        # 중복 억제: 최근 평가 서열 N개를 LLM 프롬프트에 주입 ("이미 평가했으니 제안 금지")
        # DEDUP_INJECT_N env로 주입 수 제어 (기본 50, 0=비활성)
        try:
            _dedup_inject_n = int(os.environ.get("DEDUP_INJECT_N", "50"))
        except ValueError:
            _dedup_inject_n = 50
        if _dedup_inject_n > 0:
            # seen_sequences에서 native 제외, 최근 N개만 샘플
            _eval_pool = [s for s in seen_sequences if s != config.original_sequence]
            # 리스트 잘라내기 (순서 비결정적이므로 sorted해서 slice)
            _eval_sample = sorted(_eval_pool)[-_dedup_inject_n:]
            if _eval_sample:
                prev_results["evaluated_seqs"] = _eval_sample
        plan = planner.execute(
            {
                "iteration": iteration,
                "receptor_config": {"name": "SSTR2", "chain": "A"},
                "constraints": {
                    "max_iterations": config.max_iterations,
                    "reference_sequence": config.original_sequence,
                    "design_positions": config.design_positions,
                },
                "critic_feedback": critic_feedback,
                "previous_results": prev_results,
                # Silo B experiment_log 경로 — 궤적 주입용 (pyrosetta_flow 전용)
                "silo_b_log_path": exp_log_path,
            }
        ).get("plan")
        hypothesis = getattr(plan, "hypothesis", f"Iteration {iteration} mutate->dock optimization")
        emitter.update_agent(
            "planner",
            status="idle",
            message=hypothesis[:80],
            task_count_delta=1,
            report={
                "type": "plan",
                "iteration": iteration,
                "run_id": run_id,
                "hypothesis": hypothesis,
                "strategy": objective_mode,
            },
        )
        emitter.append_timeline_event(iteration, "planner", "completed", hypothesis[:120])

        iter_dir = flow_dir / f"iter_{iteration:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        reports_dir = iter_dir / "08_reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # -- persist planner report per iteration --
        planner_report = {
            "type": "planner",
            "iteration": iteration,
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hypothesis": hypothesis,
            "strategy": objective_mode,
            "critic_feedback_used": critic_feedback or None,
            "previous_best_ddg": prev_results.get("best_ddg"),
        }
        (reports_dir / "planner_report.json").write_text(
            json.dumps(planner_report, indent=2, ensure_ascii=False), encoding="utf-8",
        )

        # ── Track1-P1/P2: Planner↔Critic 사전검토(pre-review) 루프 ──────────
        # config.preview_rounds > 0 이면 도킹 전에 mini-loop 실행.
        # 0이면 기존 동작(하위호환). 라운드당 LLM 호출:
        #   P1(expert_panel=False): 2회(critic 검토 + planner 재고)
        #   P2(expert_panel=True) : 5+1회(4 전문가+통합 + planner 재고)
        _n_preview = getattr(config, "preview_rounds", 1)
        _use_expert_panel = getattr(config, "expert_panel", True)
        if _n_preview > 0 and plan is not None:
            # Track1-P2: ExpertPanelAgent lazy-import (의존성 분리)
            _expert_panel_agent = None
            if _use_expert_panel:
                try:
                    from AG_src.agents.expert_panel import ExpertPanelAgent
                    _expert_panel_agent = ExpertPanelAgent(critic)
                except Exception as _ep_exc:
                    print(
                        f"  [pre-review] ExpertPanelAgent 로드 실패: {_ep_exc} — P1 단일 모드로 폴백",
                        file=sys.stderr,
                    )

            _prereview_log: List[Dict[str, Any]] = []
            for _pr_round in range(1, _n_preview + 1):
                _pr_guidance = getattr(plan, "parameters", {}).get("mutation_guidance", {})
                _pr_hypothesis = getattr(plan, "hypothesis", "")
                _mode_label = (
                    "expert_panel(P2)" if _expert_panel_agent is not None else "single_critic(P1)"
                )
                emitter.append_timeline_event(
                    iteration, "critic.prereview", "running",
                    f"Pre-review round {_pr_round}/{_n_preview} [{_mode_label}]: "
                    f"evaluating hypothesis before docking"
                )
                # --- 공통 컨텍스트 수집 ---
                _pr_sel_lb = (_sel_leaderboard.summary() if _sel_leaderboard else [])
                _pr_best_delta = (_sel_leaderboard.best_delta() if _sel_leaderboard else None)
                _pr_traj: Optional[str] = None
                try:
                    from AG_src.llm.prompts import format_trajectory_summary
                    _pr_traj = format_trajectory_summary(
                        records_or_prev_results=None,
                        k=4,
                        silo_b_log_path=exp_log_path,
                    )
                except Exception:
                    pass

                # --- P2 패널 or P1 단일 Critic 분기 ---
                if _expert_panel_agent is not None:
                    # Track1-P2: 4 전문가 fan-out + 통합 fan-in (5 LLM calls)
                    # max_discussion_rounds: env PANEL_MAX_DISCUSSION_ROUNDS (기본 5) 반영
                    import os as _runner_os
                    _max_disc_rounds: Optional[int] = None
                    _env_max = _runner_os.environ.get("PANEL_MAX_DISCUSSION_ROUNDS")
                    if _env_max is not None:
                        try:
                            _max_disc_rounds = int(_env_max)
                        except ValueError:
                            pass
                    _panel_result = _expert_panel_agent.prereview_panel(
                        iteration=iteration,
                        hypothesis=_pr_hypothesis,
                        mutation_guidance=_pr_guidance,
                        trajectory_summary=_pr_traj,
                        selectivity_leaderboard=_pr_sel_lb,
                        best_delta_margin=_pr_best_delta,
                        round_idx=_pr_round,
                        max_discussion_rounds=_max_disc_rounds,
                    )
                    # P1 단일 형식으로 정규화 (Planner 재고 API 호환)
                    prereview_result: Dict[str, Any] = {
                        "concerns": _panel_result.get("merged_concerns", []),
                        "suggested_revisions": _panel_result.get("suggested_revisions", {}),
                        "approve": _panel_result.get("approve", True),
                        "expert_verdicts": _panel_result.get("expert_verdicts", {}),
                        "llm_calls": _panel_result.get("llm_calls", 0),
                        "mode": "expert_panel",
                        # turn-taking 토론 기록 — discussion_log/웹뷰로 전달 (누락 시 turns=0 버그)
                        "discussion_turns": _panel_result.get("discussion_turns", []),
                        "discussion_rounds": _panel_result.get("discussion_rounds", 1),
                        # 신규 필드 (verdict/docking_decision 분리)
                        "scientific_verdict": _panel_result.get("scientific_verdict", "approve"),
                        "docking_decision": _panel_result.get("docking_decision", True),
                        "reason_for_docking": _panel_result.get("reason_for_docking", "proceed"),
                        "risk_level": _panel_result.get("risk_level", "low"),
                        "qc_status": _panel_result.get("qc_status", "pass"),
                        "position_map_consistent": _panel_result.get("position_map_consistent", True),
                        "selectivity_evidence_present": _panel_result.get("selectivity_evidence_present", False),
                        "consensus_status": _panel_result.get("consensus_status", "approve"),
                        # Planner revise 프롬프트가 소수 의견을 명시적으로 검토하도록
                        # 전달 — 누락 시 revise_guidance_from_prereview가 이를 볼 수 없었음.
                        "minority_dissent": _panel_result.get("minority_dissent"),
                    }
                else:
                    # Track1-P1: 단일 Critic (하위호환)
                    prereview_result = critic.prereview_hypothesis(
                        iteration=iteration,
                        hypothesis=_pr_hypothesis,
                        mutation_guidance=_pr_guidance,
                        trajectory_summary=_pr_traj,
                        selectivity_leaderboard=_pr_sel_lb,
                        best_delta_margin=_pr_best_delta,
                        round_idx=_pr_round,
                    )
                    prereview_result["mode"] = "single_critic"
                    prereview_result["llm_calls"] = 1

                # expert_verdicts: P2(expert_panel)일 때 4-전문가 verdict dict
                # approve=True/False 모두 항상 기록 (빈 dict {} 포함)
                _raw_expert_verdicts: Dict[str, Any] = (
                    prereview_result.get("expert_verdicts") or {}
                )
                _pr_entry: Dict[str, Any] = {
                    "round": _pr_round,
                    "mode": prereview_result.get("mode", "single_critic"),
                    "llm_calls": prereview_result.get("llm_calls", 1),
                    "original_hypothesis": _pr_hypothesis,
                    "original_focus": list(_pr_guidance.get("focus_positions") or []),
                    "approve": prereview_result.get("approve", True),
                    "concerns": prereview_result.get("concerns", []),
                    "suggested_positions": (
                        prereview_result.get("suggested_revisions", {}).get("focus_positions") or []
                    ),
                    # expert_verdicts: P2에서는 4 domain dict, P1에서는 {} — 항상 기록
                    "expert_verdicts": {
                        domain: {
                            "severity": v.get("severity", "?"),
                            "concerns": v.get("concerns", []),
                        }
                        for domain, v in _raw_expert_verdicts.items()
                    },
                }
                # --- approve=True → 도킹 진행 ---
                _sci_verdict = prereview_result.get("scientific_verdict", "approve")
                _docking_dec = prereview_result.get("docking_decision", True)
                _reason_dock = prereview_result.get("reason_for_docking", "proceed")
                if prereview_result.get("approve", True):
                    _display_verdict = _sci_verdict
                    if _sci_verdict not in ("approve",) or prereview_result.get("consensus_status") == "forced_pass":
                        _display_verdict = (
                            f"{_sci_verdict}+forced_pass(exploratory)"
                            if prereview_result.get("consensus_status") == "forced_pass"
                            else _sci_verdict
                        )
                    emitter.append_timeline_event(
                        iteration, "critic.prereview", "completed",
                        f"Round {_pr_round}: scientific_verdict={_display_verdict} [{_mode_label}] — docking_decision={_docking_dec}"
                    )
                    _pr_entry["action"] = "approved"
                    _pr_entry["scientific_verdict"] = _sci_verdict
                    _pr_entry["docking_decision"] = _docking_dec
                    _pr_entry["reason_for_docking"] = _reason_dock
                    _pr_entry["qc_status"] = prereview_result.get("qc_status", "pass")
                    _pr_entry["position_map_consistent"] = prereview_result.get("position_map_consistent", True)
                    _prereview_log.append(_pr_entry)
                    # discussion_log: approve=True (통과)도 토론 전문 기록
                    _append_discussion_log(
                        discussion_log_path=flow_dir / "discussion_log.jsonl",
                        iteration=iteration,
                        hypothesis=_pr_hypothesis,
                        round_idx=_pr_round,
                        expert_verdicts=_raw_expert_verdicts,
                        fanin={
                            "approve": prereview_result.get("approve", True),
                            "merged_concerns": prereview_result.get("concerns", []),
                        },
                        final_focus=list(_pr_guidance.get("focus_positions") or []),
                        discussion_turns=prereview_result.get("discussion_turns"),
                        discussion_rounds=prereview_result.get("discussion_rounds", 1),
                        scientific_verdict=_sci_verdict,
                        docking_decision=_docking_dec,
                        reason_for_docking=_reason_dock,
                        risk_level=prereview_result.get("risk_level"),
                        qc_status=prereview_result.get("qc_status"),
                        position_map_consistent=prereview_result.get("position_map_consistent"),
                        selectivity_evidence_present=prereview_result.get("selectivity_evidence_present"),
                        minority_dissent=prereview_result.get("minority_dissent"),
                        stall_detected=prereview_result.get("stall_detected"),
                        stall_streak=prereview_result.get("stall_streak"),
                    )
                    break
                # --- approve=False → Planner 재고 ---
                emitter.append_timeline_event(
                    iteration, "critic.prereview", "running",
                    f"Round {_pr_round}: rejected [{_mode_label}] — Planner revising guidance"
                )
                plan = planner.revise_guidance_from_prereview(
                    plan=plan,
                    critic_prereview=prereview_result,
                    iteration=iteration,
                    round_idx=_pr_round,
                )
                _revised_focus = (
                    plan.parameters.get("mutation_guidance", {}).get("focus_positions") or []
                )
                # 작업 A: expert 거부 실효화 — 거부된 focus 를 블랙리스트에 추가하고
                # planner 재고가 같은 focus 로 회귀했으면 bandit 재샘플로 강제 다양화
                if _focus_blacklist_enabled and _revised_focus:
                    _rejected_frozenset = frozenset(_pr_guidance.get("focus_positions") or [])
                    _revised_frozenset = frozenset(_revised_focus)
                    # 거부된 focus 를 블랙리스트에 추가 (이번 iteration 전용)
                    if _rejected_frozenset and _rejected_frozenset not in _tried_focus:
                        _tried_focus.append(_rejected_frozenset)
                        _tried_focus = _tried_focus[-_focus_blacklist_k:]
                    # planner 재고가 여전히 같거나 블랙리스트에 있으면 bandit 강제 재샘플
                    if _revised_frozenset in _tried_focus or _revised_frozenset == _rejected_frozenset:
                        _resample_rng = random.Random(config.seed_base + iteration * 37 + _pr_round)
                        if _bandit_instance is not None:
                            try:
                                from .bandit import MUTABLE_POSITIONS_1IDX as _MPOS
                                _resampled_focus = _bandit_instance.sample_focus_with_exploration(
                                    n=len(_revised_focus) or config.bandit_n_focus,
                                    tried_focus=list(_tried_focus),
                                    rng=_resample_rng,
                                )
                                print(
                                    f"  [focus-blacklist] expert 거부 후 LLM 재고={list(_revised_frozenset)} "
                                    f"→ bandit 재샘플={_resampled_focus} (블랙리스트 size={len(_tried_focus)})",
                                    file=sys.stderr,
                                )
                                _revised_focus = _resampled_focus
                                # plan 의 mutation_guidance 도 갱신
                                if plan is not None and hasattr(plan, "parameters"):
                                    _mg = plan.parameters.get("mutation_guidance", {})
                                    _mg["focus_positions"] = list(_resampled_focus)
                                    plan.parameters["mutation_guidance"] = _mg
                            except Exception as _re_exc:
                                print(
                                    f"  [focus-blacklist] bandit 재샘플 실패(non-fatal): {_re_exc}",
                                    file=sys.stderr,
                                )
                        else:
                            # bandit 없으면 random_fallback 으로 다양화
                            _mutable = _mutable_design_positions(config)
                            _resample_rng.shuffle(_mutable)
                            _n_need = len(_revised_focus) or config.bandit_n_focus
                            _revised_focus = _mutable[:min(_n_need, len(_mutable))]
                            print(
                                f"  [focus-blacklist] bandit 없음 → random 재샘플={_revised_focus}",
                                file=sys.stderr,
                            )
                            if plan is not None and hasattr(plan, "parameters"):
                                _mg = plan.parameters.get("mutation_guidance", {})
                                _mg["focus_positions"] = list(_revised_focus)
                                plan.parameters["mutation_guidance"] = _mg
                _pr_entry["revised_focus"] = list(_revised_focus)
                _pr_entry["action"] = "revised"
                _pr_entry["scientific_verdict"] = prereview_result.get("scientific_verdict", "conditional")
                _pr_entry["docking_decision"] = prereview_result.get("docking_decision", False)
                _pr_entry["reason_for_docking"] = prereview_result.get("reason_for_docking", "pending revision")
                _pr_entry["qc_status"] = prereview_result.get("qc_status", "pass")
                _pr_entry["position_map_consistent"] = prereview_result.get("position_map_consistent", True)
                _prereview_log.append(_pr_entry)
                # discussion_log: approve=False (거부 → revision)도 기록
                _append_discussion_log(
                    discussion_log_path=flow_dir / "discussion_log.jsonl",
                    iteration=iteration,
                    hypothesis=_pr_hypothesis,
                    round_idx=_pr_round,
                    expert_verdicts=_raw_expert_verdicts,
                    fanin={
                        "approve": False,
                        "merged_concerns": prereview_result.get("concerns", []),
                    },
                    final_focus=list(_revised_focus),
                    discussion_turns=prereview_result.get("discussion_turns"),
                    discussion_rounds=prereview_result.get("discussion_rounds", 1),
                    scientific_verdict=prereview_result.get("scientific_verdict", "conditional"),
                    docking_decision=prereview_result.get("docking_decision", False),
                    reason_for_docking=prereview_result.get("reason_for_docking", "pending revision"),
                    risk_level=prereview_result.get("risk_level"),
                    qc_status=prereview_result.get("qc_status"),
                    position_map_consistent=prereview_result.get("position_map_consistent"),
                    selectivity_evidence_present=prereview_result.get("selectivity_evidence_present"),
                    minority_dissent=prereview_result.get("minority_dissent"),
                    stall_detected=prereview_result.get("stall_detected"),
                    stall_streak=prereview_result.get("stall_streak"),
                )
                # hypothesis 갱신 반영
                hypothesis = getattr(plan, "hypothesis", hypothesis)
                emitter.append_timeline_event(
                    iteration, "critic.prereview", "completed",
                    f"Round {_pr_round}: revised focus={_revised_focus}"
                )
            # planner_report에 사전검토 이력 기록 (provenance)
            planner_report["pre_review_rounds"] = _prereview_log
            planner_report["pre_review_mode"] = (
                "expert_panel" if _expert_panel_agent is not None else "single_critic"
            )
            _total_llm_calls = sum(e.get("llm_calls", 1) for e in _prereview_log)
            planner_report["pre_review_llm_calls"] = _total_llm_calls
            planner_report["final_focus_positions"] = (
                plan.parameters.get("mutation_guidance", {}).get("focus_positions") or []
            )
            (reports_dir / "planner_report.json").write_text(
                json.dumps(planner_report, indent=2, ensure_ascii=False), encoding="utf-8",
            )
            print(
                f"  [pre-review] iter={iteration} 완료: "
                f"mode={planner_report['pre_review_mode']}, "
                f"{len(_prereview_log)} 라운드, "
                f"llm_calls={_total_llm_calls}, "
                f"final_focus={planner_report['final_focus_positions']}",
                file=sys.stderr,
            )
        # ── 사전검토 루프 끝 ──────────────────────────────────────────────────

        # 작업 A: final_focus 를 tried-focus 블랙리스트에 등록 (iteration 경계)
        if _focus_blacklist_enabled:
            _final_fp = (
                plan.parameters.get("mutation_guidance", {}).get("focus_positions") or []
                if plan is not None and hasattr(plan, "parameters") else []
            )
            if _final_fp:
                _final_fs = frozenset(_final_fp)
                _tried_focus.append(_final_fs)
                _tried_focus = _tried_focus[-_focus_blacklist_k:]

        # reject_hard: expert panel이 high≥2(치명 다수)로 hard-reject → 도킹 skip, 다음 가설
        # docking_decision=False가 설정된 경우도 skip.
        # (PANEL_HARD_REJECT_HIGH=1. high 1/medium은 forced_pass로 통과 — 여기 안 걸림)
        _prereview_docking_decision = prereview_result.get("docking_decision", True)
        _prereview_sci_verdict = prereview_result.get("scientific_verdict", "approve")
        _is_reject_hard = (
            _n_preview > 0
            and prereview_result.get("consensus_status") == "reject_hard"
        )
        _is_docking_blocked = _n_preview > 0 and not _prereview_docking_decision
        if _is_reject_hard or _is_docking_blocked:
            print(
                f"  [reject_hard] iter={iteration} "
                f"scientific_verdict={_prereview_sci_verdict} "
                f"docking_decision={_prereview_docking_decision} — 도킹 skip(다음 가설)",
                file=sys.stderr,
            )
            emitter.append_timeline_event(
                iteration, "critic.prereview", "failed",
                f"reject_hard: scientific_verdict={_prereview_sci_verdict}, docking_decision=False — skip",
            )
            continue

        candidates: List[CandidateResult] = []
        t_prepare = emitter.start_rosetta_substep("step06_prepare")
        emitter.append_timeline_event(iteration, "rosetta.prepare", "running", "Iteration directory and run context prepared")
        emitter.complete_rosetta_substep("step06_prepare", t_prepare)
        emitter.append_timeline_event(iteration, "rosetta.prepare", "completed", "Ready for mutation and docking")

        t_step06 = emitter.start_step("step06")
        t_mutate = emitter.start_rosetta_substep("step06_mutate")
        emitter.append_timeline_event(iteration, "rosetta.mutate", "running", "Generating peptide mutants")
        try:
            candidate_jobs: List[Dict[str, Any]] = []
            guidance = getattr(plan, "parameters", {}).get("mutation_guidance", {})
            # provenance: LLM planner가 focus_positions를 직접 줬는지(bandit 병합 前 포착)
            _llm_had_focus = bool(guidance.get("focus_positions"))

            # 작업 A: LLM focus 가 블랙리스트와 겹치면 bandit exploration 재샘플로 교체
            if _focus_blacklist_enabled and _llm_had_focus and _tried_focus and _bandit_instance is not None:
                _llm_focus_fs = frozenset(guidance.get("focus_positions", []))
                _recent_blacklist = _tried_focus[-_focus_blacklist_k:]
                # 최근 K개 중 과반(>= K/2+1) 이 동일 focus 이면 고착으로 판정
                _repeat_count = sum(1 for f in _recent_blacklist if f == _llm_focus_fs)
                _majority_threshold = max(2, _focus_blacklist_k // 2 + 1)
                if _repeat_count >= _majority_threshold:
                    try:
                        _rng_expl = random.Random(config.seed_base + iteration * 71)
                        _bandit_focus = _bandit_instance.sample_focus_with_exploration(
                            n=len(guidance["focus_positions"]),
                            tried_focus=_recent_blacklist,
                            rng=_rng_expl,
                        )
                        print(
                            f"  [focus-blacklist] LLM focus 고착 감지(반복 {_repeat_count}/{_focus_blacklist_k}) "
                            f"{list(_llm_focus_fs)} → bandit 탐색 교체={_bandit_focus}",
                            file=sys.stderr,
                        )
                        guidance["focus_positions"] = _bandit_focus
                        _llm_had_focus = False  # provenance: bandit 교체
                    except Exception as _bl_exc:
                        print(
                            f"  [focus-blacklist] 교체 실패(non-fatal): {_bl_exc}",
                            file=sys.stderr,
                        )

            # Merge bandit guidance: use bandit focus_positions as fallback
            # when the LLM planner did not provide focus_positions
            if bandit_guidance and not guidance.get("focus_positions"):
                # 작업 A: bandit fallback 도 블랙리스트 적용 (exploration boost)
                if _focus_blacklist_enabled and _tried_focus and _bandit_instance is not None:
                    try:
                        _rng_bd = random.Random(config.seed_base + iteration * 53)
                        _bandit_expl_focus = _bandit_instance.sample_focus_with_exploration(
                            n=config.bandit_n_focus,
                            tried_focus=_tried_focus[-_focus_blacklist_k:],
                            rng=_rng_bd,
                        )
                        guidance.setdefault("focus_positions", _bandit_expl_focus)
                        print(
                            f"  [bandit+blacklist] exploration focus: {_bandit_expl_focus}",
                            file=sys.stderr,
                        )
                    except Exception:
                        guidance.setdefault("focus_positions", bandit_guidance.get("focus_positions", []))
                else:
                    guidance.setdefault("focus_positions", bandit_guidance.get("focus_positions", []))
            # B2: BO 제안 포지션을 bandit보다 우선 적용 (LLM guidance 없고 bandit도 없을 때)
            # _bo_suggested는 이전 iteration의 suggest() 결과 (루프 하단에서 갱신)
            if not guidance.get("focus_positions") and _bo_suggested:
                guidance.setdefault("focus_positions", list(_bo_suggested))
                print(
                    f"  [bo→guidance] BO suggested focus_positions 적용: {_bo_suggested}",
                    file=sys.stderr,
                )
            n_guided = min(guidance.get("n_guided", config.n_candidates), config.n_candidates)
            # 변이 출처 라벨: LLM이 focus 제공=llm_guided / bandit fallback=bandit / bo=bo_guided / 없음=random
            _has_bandit_focus = bandit_guidance and bandit_guidance.get("focus_positions")
            _guided_origin = (
                "llm_guided" if _llm_had_focus
                else ("bandit" if _has_bandit_focus and guidance.get("focus_positions")
                      else ("bo_guided" if _bo_suggested and guidance.get("focus_positions")
                            else "random"))
            )

            # ---------------------------------------------------------------------------
            # 후보 슬롯 배분 (n_candidates 기준):
            #   intensification 슬롯: INTENSIFY_HIGH_CONF=1 + seed 있을 때 _intensify_ratio
            #   random 슬롯: _random_ratio (기본 25%)
            #   guided 슬롯: 나머지 (LLM/bandit/BO guidance)
            # ---------------------------------------------------------------------------
            mutation_positions = _mutable_design_positions(config)
            _n_total = config.n_candidates
            _has_intens_seeds = _intensify_enabled and bool(_intensify_seeds) and bool(_intensify_guidance_cache)
            if _has_intens_seeds:
                _n_intens = max(1, round(_n_total * _intensify_ratio))
            else:
                _n_intens = 0
            _n_random = max(1, round(_n_total * _random_ratio))
            _n_guided_slots = max(0, _n_total - _n_intens - _n_random)
            # n_guided 덮어쓰기 (기존 n_guided는 슬롯 상한으로만 사용)
            n_guided = min(n_guided, _n_guided_slots)
            if iteration == 1 or (_intensify_enabled and _has_intens_seeds):
                print(
                    f"  [slots] iter={iteration} n_total={_n_total}: "
                    f"guided={_n_guided_slots}, intensify={_n_intens}, random={_n_random}",
                    file=sys.stderr,
                )

            for idx in range(1, config.n_candidates + 1):
                mutant = None
                mutant_source = ""  # provenance: 이 후보 변이가 어디서 왔나
                fail_reason = ""
                last_proposal = config.original_sequence
                # 슬롯 구분: 1~_n_guided_slots=guided, 다음 _n_intens=intensification, 나머지=random
                _slot_type: str
                if idx <= _n_guided_slots:
                    _slot_type = "guided"
                elif idx <= _n_guided_slots + _n_intens:
                    _slot_type = "intensification"
                else:
                    _slot_type = "random"
                # intensification guidance 선택 (seed 로테이션 — idx 기준 round-robin)
                _intens_guidance: Dict[str, Any] = {}
                if _slot_type == "intensification" and _has_intens_seeds:
                    _seed_idx = (idx - _n_guided_slots - 1) % len(_intensify_seeds)
                    _seed_seq = _intensify_seeds[_seed_idx]["sequence"]
                    _intens_guidance = _intensify_guidance_cache.get(_seed_seq, {})

                for pharmacophore_attempt in range(PHARMACOPHORE_RETRY_LIMIT + 1):
                    max_trials = config.max_dedup_trials
                    for trial in range(max_trials):
                        # Escalate mutation count on repeated dedup failures
                        force_n_mutations = None
                        if trial >= max_trials * 3 // 5:
                            force_n_mutations = min(len(mutation_positions), config.max_random_mutations + 1)
                        elif trial >= max_trials * 2 // 5:
                            force_n_mutations = config.max_random_mutations
                        elif trial >= max_trials // 5:
                            force_n_mutations = 2

                        if _slot_type == "intensification" and _intens_guidance and trial < max_trials * 2 // 5:
                            # Intensification: high_confidence seed 주변 부분조합 탐색
                            proposal = generate_intensification_mutant(
                                config.original_sequence,
                                _intens_guidance,
                                mutation_positions,
                                rng=random.Random(
                                    config.seed_base
                                    + iteration * 1000
                                    + idx * 100
                                    + pharmacophore_attempt * max_trials
                                    + trial
                                ),
                                partial_combination=True,
                            )
                        elif _slot_type == "guided" and idx <= n_guided and guidance.get("focus_positions") and trial < max_trials * 2 // 5:
                            proposal = generate_guided_mutant(
                                config.original_sequence,
                                mutation_positions,
                                guidance,
                                rng=random.Random(
                                    config.seed_base
                                    + iteration * 1000
                                    + idx * 100
                                    + pharmacophore_attempt * max_trials
                                    + trial
                                ),
                            )
                        else:
                            proposal = generate_random_mutant(
                                config.original_sequence,
                                mutation_positions,
                                rng=random.Random(
                                    config.seed_base
                                    + iteration * 1000
                                    + idx * 100
                                    + pharmacophore_attempt * max_trials
                                    + trial
                                ),
                                n_mutations=force_n_mutations,
                            )
                        # 이 trial의 proposal 출처 (슬롯 타입 기반)
                        if _slot_type == "intensification" and _intens_guidance and trial < max_trials * 2 // 5:
                            proposal_source = "intensification"
                        elif _slot_type == "guided" and idx <= n_guided and guidance.get("focus_positions") and trial < max_trials * 2 // 5:
                            proposal_source = _guided_origin
                        else:
                            proposal_source = "random"
                        last_proposal = proposal
                        if proposal == config.original_sequence or proposal in seen_sequences:
                            continue
                        if not _preserves_scaffold(proposal, config.original_sequence):
                            break
                        mutant = proposal
                        mutant_source = proposal_source
                        seen_sequences.add(proposal)
                        break
                    if mutant is not None:
                        break
                    if not _preserves_scaffold(last_proposal, config.original_sequence):
                        continue
                    # --- 2단계 fall-back: generate_dedup_fallback ---
                    # pharmacophore_attempt 루프 내에서 guided/intensification이
                    # max_trials 회 전부 중복이었을 때 호출.
                    # 단일 시도가 아닌 최대 200회 재시도로 슬롯 낭비 방지.
                    _fb_rng = random.Random(
                        config.seed_base
                        + iteration * 1000
                        + idx * 100
                        + pharmacophore_attempt * max_trials
                        + 99
                    )
                    fallback = generate_dedup_fallback(
                        original_seq=config.original_sequence,
                        design_positions=mutation_positions,
                        seen_sequences=seen_sequences,
                        rng=_fb_rng,
                        max_attempts=200,
                        min_mutations=2,
                        max_mutations=max(3, len(mutation_positions) // 2),
                    )
                    if fallback is not None and _preserves_scaffold(fallback, config.original_sequence):
                        mutant = fallback
                        mutant_source = "dedup_fallback"
                        seen_sequences.add(fallback)
                        last_proposal = fallback
                        break
                    elif fallback is not None:
                        last_proposal = fallback

                # --- 최종 안전망: dedup_fallback도 실패(탐색 공간 포화) ---
                # DEDUP_RETRY_MAX 번 추가 시도 후에도 없으면 last_proposal 제출
                # (중복이어도 도킹해서 cache hit 처리됨 — 슬롯 완전 낭비보다 낫다)
                if mutant is None:
                    for _extra in range(_dedup_retry_max):
                        _extra_rng = random.Random(
                            config.seed_base
                            + iteration * 2000
                            + idx * 200
                            + _extra * 17
                        )
                        _extra_fb = generate_dedup_fallback(
                            original_seq=config.original_sequence,
                            design_positions=mutation_positions,
                            seen_sequences=seen_sequences,
                            rng=_extra_rng,
                            max_attempts=100,
                            min_mutations=3,
                            max_mutations=len(mutation_positions),
                        )
                        if _extra_fb is not None and _preserves_scaffold(_extra_fb, config.original_sequence):
                            mutant = _extra_fb
                            mutant_source = "dedup_fallback"
                            seen_sequences.add(_extra_fb)
                            last_proposal = _extra_fb
                            break

                if mutant is None:
                    # 탐색 공간 완전 포화 또는 scaffold 제약 불가피 실패
                    fail_reason = (
                        "dedup_exhausted: all candidates in seen_sequences or "
                        "FWKT pharmacophore gate failed after "
                        f"{PHARMACOPHORE_RETRY_LIMIT} retries + {_dedup_retry_max} extra"
                    )
                    if last_proposal != config.original_sequence:
                        mutant = last_proposal
                        mutant_source = "dedup_exhausted"
                    else:
                        # 최후 수단: 강제 랜덤 (native 이외 아무 것이나)
                        mutant = generate_random_mutant(
                            config.original_sequence,
                            mutation_positions,
                            rng=random.Random(config.seed_base + iteration * 9999 + idx),
                            n_mutations=3,
                        )
                        mutant_source = "dedup_exhausted"
                    print(
                        f"  [dedup] iter={iteration} cand={idx}: 탐색 공간 포화 "
                        f"(seen={len(seen_sequences)}) → dedup_exhausted 제출",
                        file=sys.stderr,
                    )
                candidate_jobs.append(
                    {
                        "idx": idx,
                        "mutant": mutant,
                        "out_pdb": iter_dir / f"cand_{idx:03d}.pdb",
                        "fail_reason": fail_reason,
                        "mutation_source": mutant_source or "unknown",
                    }
                )

            emitter.complete_rosetta_substep("step06_mutate", t_mutate)
            emitter.append_timeline_event(
                iteration,
                "rosetta.mutate",
                "completed",
                "Mutants generated" if candidate_jobs else "No candidates to mutate",
            )

            t_refine = emitter.start_rosetta_substep("step06_refine")
            n_jobs = len(candidate_jobs)

            # ── nstruct 평탄화 병렬 도킹 (접근 A) ─────────────────────────────────
            # 환경변수 FLEXPEP_NSTRUCT 로 trial 수 조절 (기본 5, flexpep_dock.py 와 동일 기본값).
            # runner 에서 nstruct=1 subprocess 를 (후보 × nstruct) 개 평탄화해 병렬 제출 →
            # 완료 후 후보별 그룹화하여 robust 통계(median/sd/n_converged) 집계.
            # LLM 호출은 iteration 단위(Planner/Critic)이므로 n_candidates 증가 없이
            # nstruct 평탄화만으로 병렬도를 확보 → vLLM 8000 부하 무증가.
            try:
                _flat_nstruct = max(1, int(os.environ.get("FLEXPEP_NSTRUCT", "5")))
            except ValueError:
                _flat_nstruct = 5

            # DOCK_MAX_WORKERS env 로 상한 조절.
            # 기본: cpu_count - 16 (vLLM·SiloA·autopush·MM-GBSA 데몬용 여유 코어 보존).
            # 최소 보장: max(n_jobs, 4) — 후보 수보다는 항상 크게.
            _cpu = os.cpu_count() or 4
            try:
                _env_max = int(os.environ.get("DOCK_MAX_WORKERS", str(max(_cpu - 16, 8))))
            except ValueError:
                _env_max = max(_cpu - 16, 8)
            _n_trials_total = n_jobs * _flat_nstruct
            max_workers = min(_n_trials_total, config.max_parallel_workers, _env_max)
            max_workers = max(max_workers, min(n_jobs, 4))  # 최소 후보 수 or 4

            emitter.append_timeline_event(
                iteration, "rosetta.refine", "running",
                f"Running FlexPepDock refinement ({n_jobs} candidates × nstruct={_flat_nstruct} "
                f"= {_n_trials_total} trials, {max_workers} parallel workers "
                f"[cpu={_cpu}, DOCK_MAX_WORKERS={_env_max}])",
            )
            print(
                f"  [dock-parallel] 후보={n_jobs} × nstruct={_flat_nstruct} = {_n_trials_total} trials, "
                f"max_workers={max_workers} (cpu={_cpu}, DOCK_MAX_WORKERS={_env_max}, "
                f"config.max_parallel_workers={config.max_parallel_workers})",
                file=sys.stderr,
            )

            # Emit per-candidate "running" events
            for job in candidate_jobs:
                cid = f"iter{iteration:02d}_cand{int(job['idx']):03d}"
                emitter.append_timeline_event(
                    iteration, f"rosetta.refine.{cid}", "running",
                    f"{cid}: {job['mutant']}",
                )

            iteration_failures: List[str] = []
            completed_count = 0

            # ── Trial 단위 함수 (nstruct=1 subprocess 1회) ────────────────────────
            def _dock_one_trial(
                job: Dict[str, Any],
                trial_idx: int,
                trial_out_pdb: Path,
            ) -> Dict[str, Any]:
                """후보 1개의 단일 trial subprocess를 실행하고 raw 결과 dict 반환.

                실패 시 {"trial_failed": True, "fail_reason": str} 반환.
                성공 시 {"ddg": float, "total_score": float, "clash_score": float, ...} 반환.
                """
                fail_reason_pre = str(job.get("fail_reason", ""))
                if fail_reason_pre:
                    return {
                        "trial_failed": True,
                        "fail_reason": fail_reason_pre,
                        "trial_idx": trial_idx,
                        "candidate_idx": int(job["idx"]),
                    }
                try:
                    result = _run_script(
                        flexpep_script,
                        [
                            "--input", config.template_pdb,
                            "--output", str(trial_out_pdb),
                            "--protocol", "flexpep_refine",
                            "--reference-complex", config.template_pdb,
                            "--target-sequence", str(job["mutant"]),
                            "--peptide-chain", str(config.peptide_chain),
                            "--nstruct", "1",
                        ],
                        config.conda_env,
                        repo_root,
                        timeout=config.script_timeout,
                    )
                    result["trial_failed"] = False
                    result["trial_idx"] = trial_idx
                    result["candidate_idx"] = int(job["idx"])
                    return result
                except Exception as exc:
                    return {
                        "trial_failed": True,
                        "fail_reason": str(exc),
                        "trial_idx": trial_idx,
                        "candidate_idx": int(job["idx"]),
                    }

            # ── 평탄화: (후보 × nstruct) 태스크 리스트 구성 ─────────────────────
            trial_tasks: List[tuple] = []  # (job, trial_idx, trial_out_pdb)
            for job in candidate_jobs:
                if str(job.get("fail_reason", "")):
                    # fail 후보는 trial 1개만 (즉시 실패 반환)
                    trial_tasks.append((job, 0, Path(job["out_pdb"])))
                else:
                    base_pdb = Path(job["out_pdb"])
                    for t in range(_flat_nstruct):
                        trial_pdb = base_pdb.with_name(
                            f"{base_pdb.stem}_trial{t:02d}{base_pdb.suffix}"
                        )
                        trial_tasks.append((job, t, trial_pdb))

            # ── 병렬 제출 + 결과 수집 ────────────────────────────────────────────
            # trial_results: 후보 idx → [raw trial result dict] 맵
            trial_results: Dict[int, List[Dict[str, Any]]] = {
                int(job["idx"]): [] for job in candidate_jobs
            }

            # [작업2] chunked 제출: max_workers*2 단위로 분할 → 동시 subprocess 수 안정화.
            # 후보별 trial_results 집계 정확성은 future 완료 순서와 무관(cand_idx 기반 분류).
            _chunk_size = max_workers * 2
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                pending_futures: Dict[Any, tuple] = {}
                for _chunk_start in range(0, len(trial_tasks), _chunk_size):
                    _chunk = trial_tasks[_chunk_start: _chunk_start + _chunk_size]
                    for job, t_idx, t_pdb in _chunk:
                        fut = executor.submit(_dock_one_trial, job, t_idx, t_pdb)
                        pending_futures[fut] = (job, t_idx)
                    # chunk 내 완료를 즉시 수집 (배압 제어)
                    _done_futs = [f for f in list(pending_futures) if f.done()]
                    for f in _done_futs:
                        tr = f.result()
                        cand_idx = tr["candidate_idx"]
                        trial_results[cand_idx].append(tr)
                        del pending_futures[f]
                # 나머지 완료 대기
                for future in as_completed(pending_futures):
                    tr = future.result()
                    cand_idx = tr["candidate_idx"]
                    trial_results[cand_idx].append(tr)

            # ── 후보별 robust 통계 집계 (접근 A 핵심) ──────────────────────────
            # flexpep_dock.run_flexpep_refine_pose 와 동일 로직:
            #   수렴 기준: ddg < 0 (ddg ≥ 0 제외)
            #   통계: median/mean/sd/min over converged_ddgs
            #   대표값(ddg): ddg_median (없으면 전체 min)
            for job in candidate_jobs:
                idx = int(job["idx"])
                mutant = str(job["mutant"])
                candidate_id = f"iter{iteration:02d}_cand{idx:03d}"
                _src = str(job.get("mutation_source", "unknown"))
                fail_reason_pre = str(job.get("fail_reason", ""))

                if fail_reason_pre:
                    cand_result = CandidateResult(
                        iteration=iteration,
                        candidate_id=candidate_id,
                        sequence=mutant,
                        ddg=999.0,
                        total_score=999.0,
                        clash_score=999.0,
                        objective_mode=objective_mode,
                        fail_reason=fail_reason_pre,
                        extra_scores={"mutation_source": _src},
                    )
                    candidates.append(cand_result)
                    completed_count += 1
                    iteration_failures.append(f"{candidate_id} refine failed: {fail_reason_pre}")
                    emitter.append_timeline_event(
                        iteration, f"rosetta.refine.{candidate_id}", "failed",
                        f"{candidate_id}: FAILED ({fail_reason_pre[:120]})",
                    )
                    _emit_candidates(emitter, candidates, ddg_threshold=config.rosetta_ddg_max, flow_dir=flow_dir)
                    continue

                trials = trial_results.get(idx, [])
                # ddg is None = 비물리(unphysical) pose로 대표값 산출 불가 → 실패로 분류
                # (physical floor patch가 unphysical pose의 ddg를 None으로 출력 → float() 예외 방지)
                successful = [
                    tr for tr in trials
                    if not tr.get("trial_failed", True) and tr.get("ddg") is not None
                ]
                failed_trials = [
                    tr for tr in trials
                    if tr.get("trial_failed", True) or tr.get("ddg") is None
                ]

                if not successful:
                    # 모든 trial 실패 → 첫 번째 실패 이유 사용
                    first_fail = failed_trials[0].get("fail_reason", "all trials failed") if failed_trials else "no trials"
                    cand_result = CandidateResult(
                        iteration=iteration,
                        candidate_id=candidate_id,
                        sequence=mutant,
                        ddg=999.0,
                        total_score=999.0,
                        clash_score=999.0,
                        objective_mode=objective_mode,
                        fail_reason=first_fail,
                        extra_scores={"mutation_source": _src, "n_trials": len(trials), "n_failed": len(failed_trials)},
                    )
                    candidates.append(cand_result)
                    completed_count += 1
                    iteration_failures.append(f"{candidate_id} refine failed: {first_fail}")
                    emitter.append_timeline_event(
                        iteration, f"rosetta.refine.{candidate_id}", "failed",
                        f"{candidate_id}: FAILED (all {len(trials)} trials failed)",
                    )
                    _emit_candidates(emitter, candidates, ddg_threshold=config.rosetta_ddg_max, flow_dir=flow_dir)
                    continue

                # 수렴 집계: ddg < 0 인 trial 만
                all_ddgs = [float(tr.get("ddg", 0.0)) for tr in successful]
                converged_ddgs = [d for d in all_ddgs if d < 0]
                n_converged = len(converged_ddgs)
                n_total = len(successful)

                if converged_ddgs:
                    ddg_median = statistics.median(converged_ddgs)
                    ddg_mean = statistics.mean(converged_ddgs)
                    ddg_sd = statistics.stdev(converged_ddgs) if n_converged > 1 else 0.0
                    ddg_min = min(converged_ddgs)
                    # 대표 trial: median 에 가장 가까운 수렴 trial
                    rep_tr = min(
                        (tr for tr in successful if float(tr.get("ddg", 0.0)) < 0),
                        key=lambda tr: abs(float(tr.get("ddg", 0.0)) - ddg_median),
                    )
                else:
                    # 수렴 없음 → 전체 중 최솟값 trial
                    ddg_median = min(all_ddgs)
                    ddg_mean = statistics.mean(all_ddgs)
                    ddg_sd = statistics.stdev(all_ddgs) if len(all_ddgs) > 1 else 0.0
                    ddg_min = ddg_median
                    rep_tr = min(successful, key=lambda tr: float(tr.get("ddg", 0.0)))

                rep_total_score = float(rep_tr.get("total_score", 0.0))
                rep_clash = float(rep_tr.get("clash_score", 0.0))

                extra: Dict[str, Any] = {
                    "mutation_source": _src,
                    "ddg_median": round(ddg_median, 4),
                    "ddg_mean": round(ddg_mean, 4),
                    "ddg_sd": round(ddg_sd, 4),
                    "ddg_min": round(ddg_min, 4),
                    "n_converged": n_converged,
                    "n_total": n_total,
                    "n_failed_trials": len(failed_trials),
                    "nstruct_flat": _flat_nstruct,
                }
                # disulfide 정보가 대표 trial에 있으면 전파
                for _dkey in ("disulfide_intact", "sg_sg_distance"):
                    if _dkey in rep_tr:
                        extra[_dkey] = rep_tr[_dkey]

                # ── contact fingerprint (1차 대표 PDB) ────────────────────
                # 대표 trial PDB 가 존재할 때 contact fingerprint 분석 수행.
                # extra_scores 에 contact_fingerprint/pharmacophore_contact_intact 기록.
                # bio-tools 환경 불필요(순수 PDB 텍스트 파싱).
                if _HAS_CONTACT_FP:
                    try:
                        _rep_trial_idx = rep_tr.get("trial_idx", 0)
                        _rep_pdb_candidate = Path(job["out_pdb"])
                        _rep_pdb_path = _rep_pdb_candidate.with_name(
                            f"{_rep_pdb_candidate.stem}_trial{_rep_trial_idx:02d}{_rep_pdb_candidate.suffix}"
                        )
                        # 대표 trial PDB 가 존재하면 분석, 없으면 candidate pdb 시도
                        _fp_pdb = str(_rep_pdb_path) if _rep_pdb_path.exists() else str(_rep_pdb_candidate)
                        if Path(_fp_pdb).exists():
                            _fp_result = _analyze_contact_fingerprint_warning(
                                _fp_pdb,
                                ddg=ddg_median,
                                peptide_seq=mutant,
                            )
                            extra["contact_fingerprint"] = _fp_result.get("contact_fingerprint", {})
                            extra["pharmacophore_contact_intact"] = _fp_result.get("pharmacophore_contact_intact", None)
                            extra["binding_mechanism_warning"] = _fp_result.get("binding_mechanism_warning", False)
                            _fp_intact = extra["pharmacophore_contact_intact"]
                            print(
                                f"  [contact_fp] {candidate_id}: "
                                f"pharmacophore_intact={_fp_intact} "
                                f"k9_d122={extra['contact_fingerprint'].get('k9_d122_salt_bridge_ang')} Å "
                                f"w8_q126={extra['contact_fingerprint'].get('w8_q126_hbond_ang')} Å",
                                file=sys.stderr,
                            )
                    except Exception as _fp_exc:
                        print(
                            f"  [contact_fp] {candidate_id} 분석 실패(non-fatal): {_fp_exc}",
                            file=sys.stderr,
                        )

                cand_result = CandidateResult(
                    iteration=iteration,
                    candidate_id=candidate_id,
                    sequence=mutant,
                    ddg=round(ddg_median, 4),
                    total_score=rep_total_score,
                    clash_score=rep_clash,
                    objective_mode=objective_mode,
                    extra_scores=extra,
                )
                candidates.append(cand_result)
                completed_count += 1

                print(
                    f"  [dock-flat] {candidate_id}: ddG_median={ddg_median:.2f} "
                    f"n_converged={n_converged}/{n_total} nstruct_flat={_flat_nstruct}",
                    file=sys.stderr,
                )
                emitter.append_timeline_event(
                    iteration, f"rosetta.refine.{candidate_id}", "completed",
                    f"{candidate_id}: ddG={ddg_median:.2f} (median, n={n_converged}/{n_total}) seq={mutant}",
                )
                # Stream completed candidate to UI immediately
                _emit_candidates(emitter, candidates, ddg_threshold=config.rosetta_ddg_max, flow_dir=flow_dir)

            emitter.complete_rosetta_substep("step06_refine", t_refine)
            if iteration_failures:
                run_status = "completed_with_warnings"
                if not run_failure_stage:
                    run_failure_stage = f"iteration_{iteration}_rosetta_partial"
                run_error_summary = f"{len(iteration_failures)} candidate refine failures (latest iteration={iteration})"
                emitter.append_timeline_event(
                    iteration,
                    "rosetta.refine",
                    "completed",
                    f"Docking refinement completed with {len(iteration_failures)} failures",
                )
                if iteration < config.max_iterations:
                    emitter.append_timeline_event(
                        iteration,
                        "continue_to_next_iteration",
                        "running",
                        "Partial failures recorded; continuing to next iteration",
                    )
            else:
                emitter.append_timeline_event(iteration, "rosetta.refine", "completed", "Docking refinement completed")
            t_score = emitter.start_rosetta_substep("step06_score")
            emitter.append_timeline_event(iteration, "rosetta.score", "running", "Aggregating ddG/score/clash metrics")
            _ = {
                "mean_ddg": statistics.mean([c.ddg for c in candidates]) if candidates else 0.0,
                "best_ddg": min([c.ddg for c in candidates]) if candidates else 0.0,
            }
            emitter.complete_rosetta_substep("step06_score", t_score)
            emitter.append_timeline_event(iteration, "rosetta.score", "completed", "Rosetta scoring metrics ready")

            # ── Stage-2 정밀 재도킹 블록 ────────────────────────────────────
            # 환경 변수 STAGE2_NSTRUCT / FlowConfig.stage2_* 로 제어.
            # 기본 활성: stage2_enabled=True, trigger_ddg=None→native baseline -20.28.
            # 1차에서 유망한 후보(top-k + trigger_ddg 조건)만 nstruct=20 재도킹.
            # 결과를 후보 ddg/extra_scores 에 승격, docking_stage=2 표시.
            # 엔진 throughput 보호: stage2_top_k(기본 3)으로 iteration당 소수 재도킹.
            _stage2_enabled = getattr(config, "stage2_enabled", True)
            if _stage2_enabled and candidates:
                try:
                    # nstruct 결정 (env STAGE2_NSTRUCT 우선)
                    try:
                        _s2_nstruct = max(1, int(os.environ.get("STAGE2_NSTRUCT", str(getattr(config, "stage2_nstruct", 20)))))
                    except ValueError:
                        _s2_nstruct = 20

                    # trigger ddg 기준값 결정
                    # env STAGE2_TRIGGER_DDG 우선, 없으면 config, 없으면 native baseline
                    _s2_trigger_default = getattr(config, "stage2_trigger_ddg", None)
                    try:
                        _s2_trigger_env = os.environ.get("STAGE2_TRIGGER_DDG")
                        _s2_trigger_ddg: float = (
                            float(_s2_trigger_env) if _s2_trigger_env
                            else (_s2_trigger_default if _s2_trigger_default is not None
                                  else baseline_ddg if baseline_ddg < 0
                                  else -20.28)
                        )
                    except (ValueError, NameError):
                        _s2_trigger_ddg = -20.28

                    _s2_top_k = max(1, getattr(config, "stage2_top_k", 3))

                    # 1차 유망 후보 선정: fail_reason 없고 ddg < trigger_ddg
                    _s2_candidates = [
                        c for c in candidates
                        if not c.fail_reason
                        and c.ddg < _s2_trigger_ddg
                    ]
                    # top-k 제한 (ddg 오름차순)
                    _s2_candidates = sorted(_s2_candidates, key=lambda c: c.ddg)[:_s2_top_k]

                    if _s2_candidates:
                        emitter.append_timeline_event(
                            iteration, "rosetta.stage2", "running",
                            f"Stage-2 정밀 재도킹: {len(_s2_candidates)}건 × nstruct={_s2_nstruct} "
                            f"(trigger_ddg={_s2_trigger_ddg:.2f})",
                        )
                        print(
                            f"  [stage2] iter={iteration} 유망 후보={len(_s2_candidates)}건 "
                            f"nstruct={_s2_nstruct} trigger={_s2_trigger_ddg:.2f}",
                            file=sys.stderr,
                        )

                        # Stage-2 trial 목록 구성
                        _s2_trial_tasks: List[tuple] = []
                        for _s2c in _s2_candidates:
                            _s2_cnum = _s2c.candidate_id.split("cand")[-1]
                            _s2_base_pdb = iter_dir / f"cand_{int(_s2_cnum):03d}.pdb"
                            for _t2 in range(_s2_nstruct):
                                _s2_trial_pdb = _s2_base_pdb.with_name(
                                    f"{_s2_base_pdb.stem}_s2trial{_t2:02d}{_s2_base_pdb.suffix}"
                                )
                                _s2_trial_tasks.append((_s2c, _t2, _s2_trial_pdb))

                        # 병렬 제출
                        _s2_trial_results: Dict[str, List[Dict[str, Any]]] = {
                            c.candidate_id: [] for c in _s2_candidates
                        }
                        _s2_max_workers = min(len(_s2_trial_tasks), max_workers)
                        _s2_max_workers = max(_s2_max_workers, 1)

                        def _dock_one_s2_trial(
                            cand: CandidateResult,
                            trial_idx: int,
                            trial_out_pdb: Path,
                        ) -> Dict[str, Any]:
                            try:
                                result_s2 = _run_script(
                                    flexpep_script,
                                    [
                                        "--input", config.template_pdb,
                                        "--output", str(trial_out_pdb),
                                        "--protocol", "flexpep_refine",
                                        "--reference-complex", config.template_pdb,
                                        "--target-sequence", cand.sequence,
                                        "--peptide-chain", str(config.peptide_chain),
                                        "--nstruct", "1",
                                    ],
                                    config.conda_env,
                                    repo_root,
                                    timeout=config.script_timeout,
                                )
                                result_s2["trial_failed"] = False
                                result_s2["trial_idx"] = trial_idx
                                result_s2["candidate_id"] = cand.candidate_id
                                return result_s2
                            except Exception as _s2_exc:
                                return {
                                    "trial_failed": True,
                                    "fail_reason": str(_s2_exc),
                                    "trial_idx": trial_idx,
                                    "candidate_id": cand.candidate_id,
                                }

                        with ThreadPoolExecutor(max_workers=_s2_max_workers) as _s2_executor:
                            _s2_futs = {
                                _s2_executor.submit(_dock_one_s2_trial, _s2c, _t2, _s2_pdb): _s2c.candidate_id
                                for _s2c, _t2, _s2_pdb in _s2_trial_tasks
                            }
                            for _s2_fut in as_completed(_s2_futs):
                                _s2_tr = _s2_fut.result()
                                _s2_cid = _s2_tr.get("candidate_id", "")
                                if _s2_cid in _s2_trial_results:
                                    _s2_trial_results[_s2_cid].append(_s2_tr)

                        # stage-2 집계 및 후보 ddg 승격
                        for _s2c in _s2_candidates:
                            _s2_trials = _s2_trial_results.get(_s2c.candidate_id, [])
                            _s2_ok = [
                                tr for tr in _s2_trials
                                if not tr.get("trial_failed", True) and tr.get("ddg") is not None
                            ]
                            if not _s2_ok:
                                print(
                                    f"  [stage2] {_s2c.candidate_id}: 모든 trial 실패, 1차 결과 유지",
                                    file=sys.stderr,
                                )
                                continue

                            _s2_ddgs_all = [float(tr.get("ddg", 0.0)) for tr in _s2_ok]
                            _s2_conv = [d for d in _s2_ddgs_all if d < 0]
                            if _s2_conv:
                                _s2_median = statistics.median(_s2_conv)
                                _s2_mean = statistics.mean(_s2_conv)
                                _s2_sd = statistics.stdev(_s2_conv) if len(_s2_conv) > 1 else 0.0
                                _s2_min = min(_s2_conv)
                                _s2_n_conv = len(_s2_conv)
                            else:
                                _s2_median = min(_s2_ddgs_all)
                                _s2_mean = statistics.mean(_s2_ddgs_all)
                                _s2_sd = statistics.stdev(_s2_ddgs_all) if len(_s2_ddgs_all) > 1 else 0.0
                                _s2_min = _s2_median
                                _s2_n_conv = 0

                            # 대표 trial (median 근접)
                            _s2_conv_trials = [tr for tr in _s2_ok if float(tr.get("ddg", 0.0)) < 0]
                            if _s2_conv_trials:
                                _s2_rep = min(_s2_conv_trials, key=lambda tr: abs(float(tr.get("ddg", 0.0)) - _s2_median))
                            else:
                                _s2_rep = min(_s2_ok, key=lambda tr: float(tr.get("ddg", 0.0)))

                            # contact fingerprint (stage-2 대표 PDB)
                            _s2_cnum = _s2c.candidate_id.split("cand")[-1]
                            _s2_base_pdb = iter_dir / f"cand_{int(_s2_cnum):03d}.pdb"
                            _s2_rep_trial_idx = _s2_rep.get("trial_idx", 0)
                            _s2_rep_pdb = _s2_base_pdb.with_name(
                                f"{_s2_base_pdb.stem}_s2trial{_s2_rep_trial_idx:02d}{_s2_base_pdb.suffix}"
                            )
                            _s2_fp: Dict[str, Any] = {}
                            if _HAS_CONTACT_FP and _s2_rep_pdb.exists():
                                try:
                                    _s2_fp_result = _analyze_contact_fingerprint_warning(
                                        str(_s2_rep_pdb),
                                        ddg=_s2_median,
                                        peptide_seq=_s2c.sequence,
                                    )
                                    _s2_fp = {
                                        "contact_fingerprint": _s2_fp_result.get("contact_fingerprint", {}),
                                        "pharmacophore_contact_intact": _s2_fp_result.get("pharmacophore_contact_intact"),
                                        "binding_mechanism_warning": _s2_fp_result.get("binding_mechanism_warning", False),
                                    }
                                except Exception as _s2_fp_exc:
                                    print(
                                        f"  [stage2][contact_fp] {_s2c.candidate_id} 실패(non-fatal): {_s2_fp_exc}",
                                        file=sys.stderr,
                                    )

                            # extra_scores 갱신 (1차 통계 보존, 2차 통계 추가)
                            _s2c.extra_scores.update({
                                "docking_stage": 2,
                                "stage2_ddg_median": round(_s2_median, 4),
                                "stage2_ddg_mean": round(_s2_mean, 4),
                                "stage2_ddg_sd": round(_s2_sd, 4),
                                "stage2_ddg_min": round(_s2_min, 4),
                                "stage2_n_converged": _s2_n_conv,
                                "stage2_n_total": len(_s2_ok),
                                "stage2_nstruct": _s2_nstruct,
                            })
                            _s2c.extra_scores.update(_s2_fp)

                            # ddg 승격 (2차 대표값으로 대체)
                            _prev_ddg = _s2c.ddg
                            _s2c.ddg = round(_s2_median, 4)
                            _s2c.total_score = float(_s2_rep.get("total_score", _s2c.total_score))
                            _s2c.clash_score = float(_s2_rep.get("clash_score", _s2c.clash_score))

                            print(
                                f"  [stage2] {_s2c.candidate_id}: "
                                f"ddG 1차={_prev_ddg:.2f} → 2차={_s2_median:.2f} "
                                f"n_conv={_s2_n_conv}/{len(_s2_ok)} nstruct={_s2_nstruct} "
                                f"contact_intact={_s2_fp.get('pharmacophore_contact_intact')}",
                                file=sys.stderr,
                            )

                        _emit_candidates(emitter, candidates, ddg_threshold=config.rosetta_ddg_max, flow_dir=flow_dir)
                        emitter.append_timeline_event(
                            iteration, "rosetta.stage2", "completed",
                            f"Stage-2 완료: {len(_s2_candidates)}건 정밀 재도킹",
                        )
                    else:
                        print(
                            f"  [stage2] iter={iteration} 유망 후보 없음 (trigger_ddg={_s2_trigger_ddg:.2f}). 1차 결과 사용.",
                            file=sys.stderr,
                        )
                except Exception as _s2_block_exc:
                    print(
                        f"  [stage2] 블록 예외(non-fatal, 1차 결과 유지): {_s2_block_exc}",
                        file=sys.stderr,
                    )
            # ── Stage-2 블록 끝 ────────────────────────────────────────────

        except Exception as exc:
            # Keep sub-step status truthful in failure scenarios.
            try:
                emitter.fail_rosetta_substep("step06_refine", t_refine)  # type: ignore[name-defined]
                emitter.append_timeline_event(iteration, "rosetta.refine", "failed", f"Refinement failed: {exc}")
            except Exception:
                emitter.fail_rosetta_substep("step06_mutate", t_mutate)
                emitter.append_timeline_event(iteration, "rosetta.mutate", "failed", f"Mutation stage failed: {exc}")
            emitter.fail_step("step06", t_step06)
            run_status = "completed_with_warnings"
            run_failure_stage = f"iteration_{iteration}_rosetta"
            run_error_summary = str(exc)
            # Keep historical ranking update path reachable for later iterations.
            if iteration < config.max_iterations:
                emitter.append_timeline_event(
                    iteration,
                    "continue_to_next_iteration",
                    "running",
                    "Iteration-level Rosetta failure captured; continuing",
                )
                continue

        qc_candidates = [_candidate_to_qc(c, i + 1, objective_mode) for i, c in enumerate(candidates)]
        thresholds = {
            "gates_enabled": {
                "plddt": False,       # ESMFold 필요 → OFF
                "docking": False,     # DiffDock/Boltz2 필요 → OFF
                "rosetta": True,      # 로컬 PyRosetta → ON
                "selectivity": False, # off-target 구조 필요 → OFF
            },
            "rosetta_ddg_max": config.rosetta_ddg_max,
            "rosetta_clash_max": config.rosetta_clash_max,
            "rosetta_constraint_violations_max": 0,
            "ranking_mode": "ddg_primary",
            "top_k_by_ddg": config.top_k,
        }
        t_qc = emitter.start_rosetta_substep("step06_qc")
        emitter.append_timeline_event(iteration, "qc", "running", "Applying QC gates and ranking")
        emitter.update_agent("qc-ranker", status="active", message="Ranking candidates")
        qc_result = qcranker.execute(
            {
                "candidates": qc_candidates,
                "thresholds": thresholds,
                "run_id": run_id,
                "iteration": iteration,
                "top_k": config.top_k,
            }
        )
        emitter.update_agent("qc-ranker", status="idle", message="Ranking done", task_count_delta=1)
        emitter.complete_rosetta_substep("step06_qc", t_qc)
        emitter.append_timeline_event(iteration, "qc", "completed", "QC ranking completed")

        selected_ids = {c.candidate_id for c in qc_result["top_candidates"]}
        selected: List[CandidateResult] = []
        for c in candidates:
            if c.candidate_id in selected_ids:
                c.selected = True
                selected.append(c)

        _emit_candidates(emitter, candidates, ddg_threshold=config.rosetta_ddg_max, flow_dir=flow_dir)

        # -- In-loop 선택성 (조건부 게이트): 유망(ddG 강한) 후보만 off-target 도킹 → Δmargin --
        if _sel_leaderboard is not None:
            try:
                from .selectivity_loop import screen_iteration_candidates
                emitter.append_timeline_event(iteration, "selectivity", "running", "In-loop selectivity 게이트 평가")
                screened = screen_iteration_candidates(
                    candidates, iter_dir, _sel_leaderboard,
                    original_sequence=config.original_sequence,
                    conda_env=config.conda_env,
                    max_screen_per_iter=getattr(config, "selectivity_max_per_iter", 2),
                    clash_max=float(config.rosetta_clash_max),
                    timeout=config.script_timeout,
                )
                emitter.append_timeline_event(
                    iteration, "selectivity", "completed",
                    f"In-loop selectivity: {len(screened)}건 도킹, leaderboard best Δ={_sel_leaderboard.best_delta()}",
                )
            except Exception as exc:
                print(f"  [sel-loop] iteration screening 실패(non-fatal): {exc}", file=sys.stderr)

        # -- Convergence detection --
        if selected:
            convergence_detector.add_iteration(iteration, [c.ddg for c in selected])
            conv_flag, conv_details = convergence_detector.is_converged()
            emitter.set_convergence({
                "converged": conv_flag,
                "p_value": conv_details.get("p_value"),
                "cv": conv_details.get("cv"),
                "recommendation": conv_details.get("recommendation", ""),
            })
            if conv_flag:
                emitter.append_timeline_event(
                    iteration, "convergence", "completed",
                    f"Converged: p={conv_details['p_value']:.4f}, CV={conv_details['cv']:.4f}",
                )
                print(
                    f"  [convergence] iter {iteration}: CONVERGED "
                    f"(p={conv_details['p_value']:.4f}, CV={conv_details['cv']:.4f})",
                    file=sys.stderr,
                )
            else:
                print(
                    f"  [convergence] iter {iteration}: not converged "
                    f"({conv_details.get('recommendation', '')})",
                    file=sys.stderr,
                )

        now_iso = datetime.now(timezone.utc).isoformat()
        run_records.extend(
            [
                {
                    "record_type": "candidate",
                    "status": "failed" if c.fail_reason else "success",
                    "run_id": run_id,
                    "iteration": iteration,
                    "candidate_id": c.candidate_id,
                    "sequence": c.sequence,
                    "ddg": c.ddg,
                    "total_score": c.total_score,
                    "clash": c.clash_score,
                    "selected": c.selected,
                    "final_score": round(-c.ddg, 3),
                    "mutation_source": c.extra_scores.get("mutation_source", "unknown"),
                    "error_summary": c.fail_reason,
                    "ts": now_iso,
                    # 2단계 도킹 표시 (1차=1, 2차=2, 미실행=1)
                    "docking_stage": c.extra_scores.get("docking_stage", 1),
                    # contact fingerprint 요약
                    "pharmacophore_contact_intact": c.extra_scores.get("pharmacophore_contact_intact"),
                    "binding_mechanism_warning": c.extra_scores.get("binding_mechanism_warning", False),
                }
                for c in candidates
            ]
        )
        # [작업1] iteration partial flush: 신규 레코드만 추가 (중복 방지)
        # append_experiment_records 는 append 전용이므로 _exp_flush_idx 이후 슬라이스만 전달.
        _new_records = run_records[_exp_flush_idx:]
        if _new_records:
            append_experiment_records(exp_log_path, _new_records)
            _exp_flush_idx = len(run_records)

        emitter.set_qc_gates(
            [
                {
                    "name": "RosettaGate",
                    "criterion": f"ddG <= {config.rosetta_ddg_max}",
                    "passed": len(selected),
                    "failed": max(0, len(candidates) - len(selected)),
                    "total": len(candidates),
                }
            ]
        )

        t_critic = emitter.start_rosetta_substep("step06_critic")
        emitter.append_timeline_event(iteration, "critic", "running", "Critic analyzing candidate outcomes")
        emitter.update_agent("critic", status="active", message="Analyzing results")
        # surrogate_map: {candidate_id → {half_life_h, admet_score, hc50}} (계산 X, 전달만)
        _surrogate_map = {
            c.candidate_id: {
                "half_life_h": c.extra_scores.get("half_life_h"),
                "admet_score": c.extra_scores.get("admet_score"),
                "hc50": c.extra_scores.get("hc50"),
            }
            for c in candidates
            if c.extra_scores
        }
        critic_analysis = critic.execute(
            {
                "rank_table": qc_result["rank_table"],
                "qc_report": qc_result["qc_report"],
                "iteration": iteration,
                "current_params": {
                    "n_candidates": config.n_candidates,
                    "objective_mode": objective_mode,
                    "rosetta_ddg_max": config.rosetta_ddg_max,
                    "rosetta_clash_max": config.rosetta_clash_max,
                },
                # 2026-06-10: 선택성 리더보드를 Critic 에 제공 (Δmargin>0 = native 초과 선택성)
                "selectivity_leaderboard": (_sel_leaderboard.summary() if _sel_leaderboard else []),
                "best_delta_margin": (_sel_leaderboard.best_delta() if _sel_leaderboard else None),
                # Silo B experiment_log 경로 — 궤적 주입용 (pyrosetta_flow 전용)
                "silo_b_log_path": exp_log_path,
                # surrogate 전달 (계산 X — extra_scores에서 읽기만)
                "surrogate_map": _surrogate_map,
            }
        ).get("critic_analysis")
        emitter.update_agent(
            "critic",
            status="idle",
            message=(getattr(critic_analysis, "hypothesis", "") or "" or "")[:80],
            task_count_delta=1,
            report={
                "type": "critic",
                "iteration": iteration,
                "hypothesis": getattr(critic_analysis, "hypothesis", "") or "",
                "proposed_changes": [
                    {
                        "parameter": c.parameter_name,
                        "old": str(c.old_value),
                        "new": str(c.new_value),
                        "rationale": c.rationale,
                    }
                    for c in getattr(critic_analysis, "proposed_changes", [])
                ],
            },
        )
        emitter.complete_rosetta_substep("step06_critic", t_critic)
        emitter.append_timeline_event(
            iteration,
            "critic",
            "completed",
            (getattr(critic_analysis, "hypothesis", "") or "" or "")[:120],
        )
        critic_feedback = {
            "hypothesis": getattr(critic_analysis, "hypothesis", "") or "",
            "proposed_changes": [
                {
                    "parameter_name": c.parameter_name,
                    "old_value": c.old_value,
                    "new_value": c.new_value,
                    "rationale": c.rationale,
                }
                for c in getattr(critic_analysis, "proposed_changes", [])
            ],
        }

        # -- Adaptive gate: apply Critic's proposed threshold changes --
        if getattr(config, "gate_mode", "static") == "adaptive":
            for change in critic_feedback["proposed_changes"]:
                pname = change["parameter_name"]
                new_val = change["new_value"]
                if pname == "rosetta_ddg_max" and isinstance(new_val, (int, float)):
                    old = config.rosetta_ddg_max
                    config.rosetta_ddg_max = float(new_val)
                    print(f"  [adaptive] rosetta_ddg_max: {old} → {config.rosetta_ddg_max} (Critic)", file=sys.stderr)
                elif pname == "rosetta_clash_max" and isinstance(new_val, (int, float)):
                    old = config.rosetta_clash_max
                    config.rosetta_clash_max = int(new_val)
                    print(f"  [adaptive] rosetta_clash_max: {old} → {config.rosetta_clash_max} (Critic)", file=sys.stderr)

        # -- persist critic report per iteration --
        # collect PDB paths for this iteration's candidates
        iter_pdb_paths = sorted(str(p) for p in iter_dir.glob("cand_*.pdb"))
        critic_report = {
            "type": "critic",
            "iteration": iteration,
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hypothesis": critic_feedback["hypothesis"],
            "proposed_changes": critic_feedback["proposed_changes"],
            "current_params": {
                "n_candidates": config.n_candidates,
                "objective_mode": objective_mode,
                "rosetta_ddg_max": config.rosetta_ddg_max,
                "rosetta_clash_max": config.rosetta_clash_max,
            },
            "candidate_pdbs": iter_pdb_paths,
            "receptor_pdb": config.template_pdb,
        }
        (reports_dir / "critic_report.json").write_text(
            json.dumps(critic_report, indent=2, ensure_ascii=False), encoding="utf-8",
        )

        t_reporter = emitter.start_rosetta_substep("step06_reporter")
        emitter.append_timeline_event(iteration, "reporter", "running", "Reporter writing iteration artifacts")
        emitter.update_agent("reporter", status="active", message="Writing reports")
        report_paths = reporter.execute(
            {
                "run_id": run_id,
                "iteration": iteration,
                "rank_table": qc_result["rank_table"],
                "top_candidates": qc_result["top_candidates"],
                "receptor_pdb": config.template_pdb,
                "output_dir": str(iter_dir),
                "critic_analysis": critic_analysis,
            }
        ).get("report_paths", {})
        emitter.update_agent(
            "reporter",
            status="idle",
            message="Reports saved",
            task_count_delta=1,
            report={
                "type": "reporter",
                "iteration": iteration,
                "summary": str(report_paths.get("summary_md", "")),
            },
        )
        # -- persist iteration manifest (reporter meta + PDB paths) --
        rank_table_obj = qc_result.get("rank_table")
        # RankTable object → extract ranked_candidates list
        if hasattr(rank_table_obj, "ranked_candidates"):
            rank_rows = [
                {
                    "candidate_id": c.candidate_id,
                    "sequence": c.sequence,
                    "ddg": c.ddg,
                    "clash_count": c.clash_count,
                    "pass_gates": c.pass_gates,
                    "fail_reasons": c.fail_reasons,
                    "final_score": c.final_score,
                    "pdb_path": c.pdb_path,
                }
                for c in rank_table_obj.ranked_candidates
            ]
        elif isinstance(rank_table_obj, list):
            rank_rows = rank_table_obj
        else:
            rank_rows = []
        candidate_manifest = []
        for row in rank_rows:
            cid = row.get("candidate_id", "")
            # derive PDB path: iter_XX/cand_NNN.pdb
            cand_num = cid.split("cand")[-1] if "cand" in cid else ""
            pdb_file = iter_dir / f"cand_{int(cand_num):03d}.pdb" if cand_num.isdigit() else None
            candidate_manifest.append({
                "candidate_id": cid,
                "sequence": row.get("sequence", ""),
                "ddg": row.get("ddg"),
                "clash_count": row.get("clash_count"),
                "pass_gates": row.get("pass_gates"),
                "fail_reasons": row.get("fail_reasons", ""),
                "pdb_path": str(pdb_file) if pdb_file and pdb_file.exists() else None,
            })
        # -- Alternative scoring chain: GNINA → ECR → Pareto 단발 비지배 정렬 → BO --
        # (Pareto: 세대 루프 없는 단발 NonDominatedSorting, NSGA-II 정렬 단계 활용)
        # B2: Tuple 반환 — (candidates, bo_suggested_positions)
        candidates, _bo_suggested = _apply_alternative_scoring(
            candidates,
            iter_dir=iter_dir,
            iteration=iteration,
            bo_optimizer=_bo_optimizer,
        )
        # B2: BO 포지션 로그 (다음 iteration guidance로 전달됨)
        if _bo_suggested:
            print(
                f"  [bo] suggested positions buffered for next iter: {_bo_suggested}",
                file=sys.stderr,
            )

        # -- RCSB PDB sequence similarity check (best-effort) --
        rcsb_matches: Dict[str, list] = {}
        selected_seqs = {c.candidate_id: c.sequence for c in selected if c.sequence}
        if selected_seqs and _HAS_RCSB:
            print(f"  [rcsb] Checking {len(selected_seqs)} selected candidates against RCSB PDB...", file=sys.stderr)
            rcsb_matches = _rcsb_check_candidates(selected_seqs, identity_cutoff=0.4, max_results=5)
            if rcsb_matches:
                print(f"  [rcsb] Found PDB matches for {len(rcsb_matches)} candidates", file=sys.stderr)
            else:
                print("  [rcsb] No PDB matches found (or network unavailable)", file=sys.stderr)
        # Enrich candidate_manifest with RCSB hits
        for entry in candidate_manifest:
            cid = entry["candidate_id"]
            if cid in rcsb_matches:
                entry["rcsb_hits"] = rcsb_matches[cid]

        # -- Pharma properties enrichment (best-effort) --
        if _HAS_PHARMA:
            try:
                pp = _PharmaProperties(reference_seq=config.original_sequence)
                for entry in candidate_manifest:
                    seq = entry.get("sequence", "")
                    if seq and len(seq) >= 5:
                        try:
                            pharma = pp.calculate_all(seq)
                            entry["pharma"] = pharma
                        except Exception:
                            pass
                print(f"  [pharma] Enriched {sum(1 for e in candidate_manifest if 'pharma' in e)} candidates", file=sys.stderr)

                # D2: structural_rules → candidate_manifest 개별 Boolean 필드 write
                # parse_structural_rules_to_columns()은 서열 기반 판정(잔기 종류 체크)이며
                # PDB 3D 좌표 기반 실측 거리 계산이 아님을 주의.
                try:
                    from .pdb_store import parse_structural_rules_to_columns as _parse_sr
                    _sr_written = 0
                    for entry in candidate_manifest:
                        pharma_d = entry.get("pharma", {})
                        sr = pharma_d.get("structural_rules", {})
                        if sr:
                            sr_cols = _parse_sr(sr)
                            for col, val in sr_cols.items():
                                if val is not None:
                                    entry[col] = val
                            _sr_written += 1
                    print(f"  [pharma] key-contact Boolean fields written: {_sr_written}", file=sys.stderr)
                except Exception as _sr_exc:
                    print(f"  [pharma] structural_rules parse skipped: {_sr_exc}", file=sys.stderr)

            except Exception as exc:
                print(f"  [pharma] Failed: {exc}", file=sys.stderr)

        # -- Cluster A~E classification (best-effort) --
        if _HAS_CLUSTER and _HAS_PHARMA:
            try:
                cluster_input = []
                for entry in candidate_manifest:
                    pharma = entry.get("pharma", {})
                    cluster_input.append({
                        "sequence": entry.get("sequence", ""),  # chelator_site sequence-based 판정용
                        "ddG": entry.get("ddG", 0),
                        "clash_score": entry.get("clashScore", entry.get("clash_score", 99)),
                        "pLDDT": entry.get("pLDDT"),
                        "structural_rules": pharma.get("structural_rules", {}),
                        "instability_index": pharma.get("instability_index", 99),
                        "blosum62": pharma.get("blosum62", {}),
                        "protease_sites": pharma.get("protease_sites", {}),
                        "gravy": pharma.get("gravy", 0),
                        "net_charge_ph74": pharma.get("net_charge_ph74", 0),
                        "metal_coordination": pharma.get("metal_coordination", {}),
                        "selectivity_margin": entry.get("selectivity_margin"),
                    })
                cluster_result = _batch_classify(cluster_input)
                for i, entry in enumerate(candidate_manifest):
                    if i < len(cluster_result.get("results", [])):
                        entry["cluster"] = cluster_result["results"][i].get("classification", {})
                print(f"  [cluster] Classified {len(candidate_manifest)} candidates into A~E", file=sys.stderr)
            except Exception as exc:
                print(f"  [cluster] Failed: {exc}", file=sys.stderr)

        # -- Dashboard enrichment: pharma/cluster 값 포함한 최종 candidate 상태 push --
        if _HAS_PHARMA or _HAS_CLUSTER:
            try:
                manifest_by_cid = {e["candidate_id"]: e for e in candidate_manifest}
                enriched_entries = []
                for idx, c in enumerate(sorted(candidates, key=lambda x: x.ddg)):
                    entry = manifest_by_cid.get(c.candidate_id, {})
                    pharma = entry.get("pharma", {})
                    cluster_info = entry.get("cluster", {})
                    candidate_entry: Dict[str, Any] = {
                        "rank": idx + 1,
                        "id": c.candidate_id,
                        "sequence": c.sequence,
                        "ddG": round(c.ddg, 3),
                        "totalScore": round(c.total_score, 3),
                        "clashScore": round(c.clash_score, 1),
                        "finalScore": round(-c.ddg, 3),
                        "result": (
                            "PASS" if c.selected else
                            "PASS" if (c.ddg <= config.rosetta_ddg_max and c.ddg < 900) else
                            "FAIL"
                        ),
                        "failReason": c.fail_reason if c.fail_reason else "",
                        "mw": pharma.get("molecular_weight", {}).get("mw_average"),
                        "instability_index": pharma.get("instability_index"),
                        "radiolysis_score": pharma.get("radiolysis_susceptibility", {}).get("total_score"),
                        "cluster": cluster_info.get("cluster"),
                        # reviewer-pharma 2026-05-14 — tier1-cluster-data 머지 sprint
                        # gravy / net_charge_ph74 : pharma_properties.calculate_all() 직접 키
                        "gravy": pharma.get("gravy"),
                        "net_charge_ph74": pharma.get("net_charge_ph74"),
                        # selectivity_margin: step05b 결과가 manifest entry에 있으면 포함
                        "selectivity_margin": entry.get("selectivity_margin", c.extra_scores.get("selectivity_margin")),
                        # 다목적 cheap-objectives (Step 0 enrichment): 반감기 + ADMET surrogate + 통합 점수
                        "half_life_h": c.extra_scores.get("half_life_h"),
                        "admet_score": c.extra_scores.get("admet_score"),
                        "boman_index": c.extra_scores.get("boman_index"),
                        "pi": c.extra_scores.get("pi"),
                        "mo_score": _mo_scalar(c.extra_scores, c.ddg) if _HAS_MO else None,
                        # fwkt_contact: _criteria_a()가 항상 계산 → 모든 cluster에서 존재
                        "fwkt_contact": cluster_info.get("criteria_met", {}).get("A", {}).get("fwkt_contact"),
                        # chelator_site_available: _criteria_d()는 cluster D/E에서만 포함
                        # cluster A/B/C 후보는 status._enrich_candidates() fallback으로 채워짐
                        "chelator_site_available": cluster_info.get("criteria_met", {}).get("D", {}).get("chelator_site_available"),
                    }
                    if "cand" in c.candidate_id:
                        candidate_entry["pdb_path"] = str(
                            flow_dir / f"iter_{c.iteration:02d}" / f"cand_{int(c.candidate_id.split('cand')[1]):03d}.pdb"
                        )
                    enriched_entries.append(candidate_entry)
                emitter.set_candidates(enriched_entries)
                print(f"  [dashboard-enrich] Pushed {len(enriched_entries)} enriched candidates", file=sys.stderr)
            except Exception as exc:
                print(f"  [dashboard-enrich] Failed: {exc}", file=sys.stderr)

        iteration_manifest = {
            "type": "iteration_manifest",
            "iteration": iteration,
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "receptor_pdb": config.template_pdb,
            "baseline_pdb": str(flow_dir / "baseline_refined.pdb"),
            "candidates": candidate_manifest,
            "rcsb_match_summary": {
                "checked": len(selected_seqs),
                "matched": len(rcsb_matches),
                "identity_cutoff": 0.4,
            },
            "report_paths": {k: str(v) for k, v in report_paths.items()},
            "planner_report": str(reports_dir / "planner_report.json"),
            "critic_report": str(reports_dir / "critic_report.json"),
        }
        (reports_dir / "iteration_manifest.json").write_text(
            json.dumps(iteration_manifest, indent=2, ensure_ascii=False), encoding="utf-8",
        )

        emitter.complete_rosetta_substep("step06_reporter", t_reporter)
        emitter.append_timeline_event(
            iteration,
            "reporter",
            "completed",
            str(report_paths.get("summary_md", "")),
        )
        emitter.complete_step("step06", t_step06)

        iter_summary = _summarize_iteration(
            iteration,
            run_id,
            hypothesis,
            objective_mode,
            selected,
            report_paths,
            getattr(critic_analysis, "hypothesis", "") or "" or "",
        )
        iterations_out.append(
            {"summary": iter_summary.__dict__, "candidates": [candidate_to_dict(c) for c in candidates]}
        )
        _conv_flag, _ = convergence_detector.is_converged() if selected else (False, {})
        emitter.add_convergence_point(
            iteration=iteration,
            best_ddg=min((c.ddg for c in selected), default=0.0),
            top_candidates=len(selected),
            converged=_conv_flag,
        )
        final_selected = selected

        # Update running best candidate after each iteration
        if final_selected:
            iter_best = min(final_selected, key=lambda x: x.ddg)
            emitter.set_best_candidate(
                {
                    "id": iter_best.candidate_id,
                    "sequence": iter_best.sequence,
                    "ddG": round(iter_best.ddg, 3),
                    "totalScore": round(iter_best.total_score, 3),
                }
            )

    # -- Optional: multi-trial validation for final candidates --
    validation_stats: Dict[str, Any] = {}
    if config.validation_n_trials > 1 and final_selected:
        print(
            f"\n{'='*60}\n"
            f"  Multi-trial validation: {config.validation_n_trials} trials × "
            f"{len(final_selected)} candidates\n"
            f"{'='*60}",
            file=sys.stderr,
        )
        emitter.append_timeline_event(
            config.max_iterations, "validation", "running",
            f"Running {config.validation_n_trials}-trial validation for {len(final_selected)} candidates",
        )
        for cand in final_selected:
            trial_ddgs: List[float] = [cand.ddg]  # trial 0 = existing result
            cand_pdb = flow_dir / f"iter_{cand.iteration:02d}" / f"cand_{int(cand.candidate_id.split('cand')[1]):03d}.pdb"

            def _run_validation_trial(trial_idx: int) -> float:
                try:
                    result = _run_script(
                        flexpep_script,
                        [
                            "--input", config.template_pdb,
                            "--output", str(cand_pdb.with_suffix(f".val{trial_idx}.pdb")),
                            "--protocol", "flexpep_refine",
                            "--reference-complex", config.template_pdb,
                            "--target-sequence", cand.sequence,
                            "--peptide-chain", str(config.peptide_chain),
                        ],
                        config.conda_env,
                        repo_root,
                        timeout=config.script_timeout,
                    )
                    return float(result.get("ddg", 999.0))
                except Exception:
                    return 999.0

            remaining_trials = config.validation_n_trials - 1
            workers = min(remaining_trials, config.validation_max_workers, os.cpu_count() or 4)
            early_stopped = False

            with ThreadPoolExecutor(max_workers=workers) as pool:
                # Submit trials in batches to allow early stopping
                batch_size = max(workers, 4)
                trial_idx = 1
                while trial_idx <= remaining_trials:
                    batch_end = min(trial_idx + batch_size, remaining_trials + 1)
                    futures = {
                        pool.submit(_run_validation_trial, t): t
                        for t in range(trial_idx, batch_end)
                    }
                    for fut in as_completed(futures):
                        ddg = fut.result()
                        if ddg < 900:  # filter catastrophic failures
                            trial_ddgs.append(ddg)
                    trial_idx = batch_end

                    # Early stopping: check CV after ≥5 valid trials
                    if (
                        config.validation_early_stop_cv > 0
                        and len(trial_ddgs) >= 5
                    ):
                        sane = [d for d in trial_ddgs if d <= 0]
                        if len(sane) >= 4:
                            cv = abs(statistics.stdev(sane) / statistics.mean(sane)) if statistics.mean(sane) != 0 else 999
                            if cv < config.validation_early_stop_cv:
                                early_stopped = True
                                break

            # Compute stats from sane trials (ddG ≤ 0)
            sane_ddgs = sorted([d for d in trial_ddgs if d <= 0])
            if not sane_ddgs:
                sane_ddgs = sorted(trial_ddgs)  # fallback
            top3_mean = statistics.mean(sane_ddgs[:3]) if len(sane_ddgs) >= 3 else statistics.mean(sane_ddgs)
            cand_stats = {
                "n_trials": len(trial_ddgs),
                "n_sane": len(sane_ddgs),
                "top3_mean": round(top3_mean, 4),
                "median": round(statistics.median(sane_ddgs), 4),
                "mean": round(statistics.mean(sane_ddgs), 4),
                "stdev": round(statistics.stdev(sane_ddgs), 4) if len(sane_ddgs) > 1 else 0.0,
                "best": round(min(sane_ddgs), 4),
                "early_stopped": early_stopped,
            }
            validation_stats[cand.candidate_id] = cand_stats
            # Update candidate ddG to top-3 mean for downstream ranking
            cand.ddg = top3_mean
            es_tag = " [early-stop]" if early_stopped else ""
            print(
                f"  {cand.candidate_id}: top3_mean={top3_mean:.2f} "
                f"stdev={cand_stats['stdev']:.2f} "
                f"({cand_stats['n_sane']}/{cand_stats['n_trials']} ok){es_tag}",
                file=sys.stderr,
            )

        emitter.append_timeline_event(
            config.max_iterations, "validation", "completed",
            f"Validation complete: {len(validation_stats)} candidates validated",
        )

    # ------------------------------------------------------------------
    # 최종 단계: 선택성(off-target SSTR1/3/4/5 실제 도킹) — config-gated, 비쌈
    #   top-K 후보를 SSTR2 정렬 큐레이션 수용체에 transplant+relax+dock 하여
    #   selectivity_margin = min(offtarget_ddg) - sstr2_ddg 산출 (양수=SSTR2 선택적).
    # ------------------------------------------------------------------
    if getattr(config, "enable_selectivity", False) and final_selected:
        try:
            from .multiobjective import screen_selectivity
            sel_k = getattr(config, "selectivity_top_k", 3)
            sel_targets = sorted(final_selected, key=lambda c: c.ddg)[:sel_k]
            print(f"  [selectivity] off-target screening for top-{len(sel_targets)} candidates",
                  file=sys.stderr)
            emitter.append_timeline_event(
                config.max_iterations, "selectivity", "running",
                f"Off-target docking (SSTR1/3/4/5) for top-{len(sel_targets)}",
            )
            for cand in sel_targets:
                try:
                    cid = cand.candidate_id
                    pdb = str(flow_dir / f"iter_{cand.iteration:02d}" /
                              f"cand_{int(cid.split('cand')[1]):03d}.pdb") if "cand" in cid else None
                    if not pdb or not Path(pdb).exists():
                        continue
                    sel = screen_selectivity(
                        sstr2_complex_pdb=pdb,
                        on_target_ddg=cand.ddg,
                        conda_env=config.conda_env,
                        timeout=config.script_timeout,
                    )
                    if sel.get("selectivity_margin") is not None:
                        cand.extra_scores["selectivity_margin"] = sel["selectivity_margin"]
                        cand.extra_scores["offtarget_ddg"] = sel.get("offtarget_ddg")
                        print(f"  [selectivity] {cand.sequence}: margin={sel['selectivity_margin']:.2f} "
                              f"(off-target {sel.get('offtarget_ddg')})", file=sys.stderr)
                except Exception as _se:
                    print(f"  [selectivity] {cand.candidate_id} failed (non-fatal): {_se}",
                          file=sys.stderr)
            emitter.append_timeline_event(
                config.max_iterations, "selectivity", "completed",
                "Off-target selectivity screening complete",
            )
        except Exception as exc:
            print(f"  [selectivity] stage failed (non-fatal): {exc}", file=sys.stderr)

    summary = {
        "mode": "agentic_mutate_then_dock",
        "objective_mode_requested": config.objective_mode,
        "iterations": config.max_iterations,
        "best_final_ddg": min((c.ddg for c in final_selected), default=0.0),
        "run_status": run_status,
        "failure_stage": run_failure_stage,
        "error_summary": run_error_summary,
        "validation_stats": validation_stats if validation_stats else None,
    }
    if run_status == "failed":
        run_records.append(
            {
                "record_type": "candidate",
                "status": "failed",
                "run_id": run_id,
                "iteration": int(summary["iterations"]),
                "candidate_id": f"{run_id}_failed",
                "sequence": config.original_sequence,
                "ddg": 999.0,
                "clash": 999.0,
                "selected": False,
                "plddt": 0.0,
                "dock_score": 0.0,
                "lddt": 0.0,
                "selectivity": 0.0,
                "final_score": -999.0,
                "error_summary": run_error_summary or run_failure_stage or "run failed",
                "failure_stage": run_failure_stage or "unknown",
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )
    # [작업1] 말단 flush: run_failed 레코드 등 아직 미기록 신규 항목만 쓴다 (중복 방지).
    _final_new = run_records[_exp_flush_idx:]
    if _final_new:
        append_experiment_records(exp_log_path, _final_new)
        _exp_flush_idx = len(run_records)  # noqa: F841 (추적용)
    historical = build_historical_candidates(load_experiment_records(exp_log_path))
    emitter.set_historical_candidates(historical)
    if final_selected:
        best = min(final_selected, key=lambda x: x.ddg)
        emitter.set_best_candidate(
            {
                "id": best.candidate_id,
                "sequence": best.sequence,
                "ddG": round(best.ddg, 3),
                "totalScore": round(best.total_score, 3),
            }
        )
    # -- Optional: generate PyMOL visualization renders --
    try:
        if final_selected:
            viz_dir = flow_dir / "renders"
            top_pdbs = [
                str(flow_dir / f"iter_{c.iteration:02d}" / f"cand_{int(c.candidate_id.split('cand')[1]):03d}.pdb")
                for c in sorted(final_selected, key=lambda x: x.ddg)[:3]
            ]
            render_paths = generate_pymol_renders(
                top_candidates=top_pdbs,
                receptor_pdb=config.template_pdb,
                output_dir=viz_dir,
            )
            if render_paths:
                view_labels = {
                    "overview": "Overview",
                    "closeup": "Close-up",
                    "interface": "Interface",
                    "electrostatics": "Electrostatics",
                }
                viz_images = [
                    {
                        "type": view,
                        "label": view_labels.get(view, view),
                        "url": f"/api/images/{Path(png_path).relative_to(repo_root / 'runs')}",
                    }
                    for view, png_path in render_paths.items()
                ]
                emitter.set_visualization_images(viz_images)
    except Exception as viz_exc:
        print(f"  [viz] PyMOL render skipped: {viz_exc}", file=sys.stderr)

    emitter.set_completed()

    artifacts = FlowArtifacts.from_parts(
        run_id=run_id,
        config=config,
        notebook_mapping=notebook_mapping(),
        baseline=baseline,
        iterations=iterations_out,
        final_candidates=[candidate_to_dict(c) for c in final_selected],
        summary=summary,
    )

    # 무한 엔진: 이번 run 의 선택성 측정을 글로벌 리더보드에 누적·영속 (다음 epoch warm-start 용)
    if _global_lb is not None:
        try:
            ing = _global_lb.ingest_artifacts(artifacts.to_dict())
            # 작업 B: 글로벌 리더보드 저장 전 MM-GBSA consensus 반영
            if _mmgbsa_enabled and _mmgbsa_path.exists():
                try:
                    _enrich_result = _global_lb.enrich_from_mmgbsa_consensus(_mmgbsa_path)
                    print(
                        f"  [mmgbsa→lb] 리더보드 MM-GBSA 반영: "
                        f"enriched={_enrich_result.get('enriched', 0)}, "
                        f"skipped={_enrich_result.get('skipped', 0)}, "
                        f"ts={_enrich_result.get('generated_at', '')}. "
                        f"caveat: 비동기 단일 스냅샷, 도킹 대체 아님",
                        file=sys.stderr,
                    )
                except Exception as _enr_exc:
                    print(f"  [mmgbsa→lb] enrich 실패(non-fatal): {_enr_exc}", file=sys.stderr)
            _global_lb.save(_global_lb_path)
            print(f"  [global-lb] +{ing['n_measurements']} 측정, 역대 best Δ={ing['best_delta_margin']}, "
                  f"unique={ing['n_unique']}, improved={ing['improved_best']}", file=sys.stderr)
        except Exception as exc:
            print(f"  [global-lb] 영속 실패(non-fatal): {exc}", file=sys.stderr)

    return artifacts


def run_pyrosetta_notebook_flow(config: FlowConfig) -> FlowArtifacts:
    """Backward compatibility wrapper."""
    return run_pyrosetta_agentic_mutdock_flow(config)
