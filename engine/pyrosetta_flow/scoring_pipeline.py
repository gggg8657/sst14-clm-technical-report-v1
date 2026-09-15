"""scoring_pipeline.py
===================
대안 스코어링 체인 (god-object runner.py 에서 분리, 2026-06-09 P1).

FlexPepDock ddG 결과에 부가 스코어링을 적용한다 (각 단계 graceful skip):
  0. cheap objectives — 반감기/ADMET surrogate (multiobjective)
  0b. OOD gate — D-AA/DOTA/cyclic 후보 surrogate 신뢰도 격리 (B3 패치 2026-06-23)
  1. GNINA rescore — CNN 도킹 스코어 (binary 없으면 dry-run)
  2. ECR consensus — GNINA+ddG 통합 순위
  3. Pareto ranking — 단발 비지배 정렬 (NSGA-II 정렬 단계 활용, pymoo; 세대 루프 없음)
  4. BO suggest — 다음 iteration 추천 위치 + guidance 반환 (B2 패치 2026-06-23)

runner.py 는 `_apply_alternative_scoring` 을 re-export 하여 하위호환을 유지한다.
GNINA/Pareto 옵셔널 의존은 본 모듈이 자체 보유. BO 는 bo_optimizer 인자(None 여부)로만 분기.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .schema import CandidateResult

# ── 이슈6: GNINA dry-run enforce 환경변수 (기본 1 = enforce) ──
# GNINA 바이너리 부재 시 dry-run 0점이 ECR/Pareto/리더보드 순위에 기여하지 않도록
# ECR 계산에서 dry-run 후보를 제외한다.
# GNINA_DRY_RUN_ENFORCE=0 으로 비활성화 가능 (하위호환, 이전 동작 복원).
_GNINA_DRY_RUN_ENFORCE: bool = (
    os.environ.get("GNINA_DRY_RUN_ENFORCE", "1").strip().lower() not in ("0", "false", "no")
)

# ── B3: OOD 정책 상수 (pharmacology_guards.py ENDPOINT_CONFIDENCE 최소 이식) ──
# 출처: pharmacology_guards.py:L1110 layer1_halflife_ensemble, L:974 /pharmacology/batch
_SCORING_OOD_POLICY: Dict[str, Any] = {
    "d_amino_acid": {
        "recommended_for_decision": False,
        "surrogate_reliability": "damped",   # stability_norm, admet_score × _OOD_DAMPING
        "pareto_treatment": "hard_violation",
        "source": "pharmacology_guards.py:L1110 layer1_halflife_ensemble",
        "note": "D-AA는 L-AA 학습 모델 OOD. 반감기 surrogate 신뢰 불가.",
    },
    "dota_chelator": {
        "recommended_for_decision": False,
        "surrogate_reliability": "damped",
        "pareto_treatment": "hard_violation",
        "source": "pharmacology_guards.py:L443,L477 DOTA OOD",
        "note": "금속 킬레이터 결합 후 ADMET 소분자/펩타이드 모델 완전 OOD.",
    },
    "cyclic_ss_bond": {
        "recommended_for_decision": True,
        "surrogate_reliability": "normal",
        "pareto_treatment": "normal",
        "source": "pharmacology_guards.py:L706-713 SS-bond OOD guard",
        "note": "우리 타겟(SST-14)은 cyclic이 정상. OOD 아님 — _detect_ood_flags에서 제외됨.",
    },
}

# OOD 감쇠 배율: 임의값(정확도 향상 아닌 오염 신호 격리 목적)
_OOD_DAMPING: float = 0.5

# ── 옵셔널 스코어링 의존 (graceful) ──────────────────────────────────────
batch_gnina_rescore = None  # type: ignore[assignment]
exponential_rank_consensus = None  # type: ignore[assignment]
_HAS_GNINA = False
try:
    from .gnina_rescoring import batch_gnina_rescore, exponential_rank_consensus  # type: ignore[assignment]
    _HAS_GNINA = True
except ImportError:  # pragma: no cover
    pass

pareto_rank_candidates = None  # type: ignore[assignment]
_HAS_PARETO = False
try:
    from .pareto_ranking import pareto_rank_candidates  # type: ignore[assignment]
    _HAS_PARETO = True
except ImportError:  # pragma: no cover
    pass


def _compute_mean_pairwise_hamming(sequences: List[str]) -> float:
    """서열 목록의 평균 pairwise Hamming 거리를 계산한다."""
    if len(sequences) <= 1:
        return 0.0

    total_distance = 0
    pair_count = 0
    for idx, seq_a in enumerate(sequences):
        for seq_b in sequences[idx + 1:]:
            min_len = min(len(seq_a), len(seq_b))
            distance = sum(1 for pos in range(min_len) if seq_a[pos] != seq_b[pos])
            distance += abs(len(seq_a) - len(seq_b))
            total_distance += distance
            pair_count += 1

    return float(total_distance / pair_count) if pair_count else 0.0


def _detect_ood_flags(
    sequence: str,
    extra_scores: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """후보 서열에 대한 OOD(out-of-distribution) 플래그를 감지한다.

    pharmacology_guards.check_pepadmet_applicability() 판정 기준 이식.
    D-AA 소문자 토큰, DOTA 태그, SS-bond(Cys 패턴) 감지.

    한계(정직성):
    - D-AA 감지는 소문자 토큰 관례에 의존 — B1 패치 미적용 시 표준 L-AA 소문자 오기입도 OOD로 오판 가능.
    - SS-bond는 SMILES 없으면 Cys 위치 기반 heuristic — SST-14 무관 Cys 서열도 OOD 판정 위험.
    - OOD 감쇠(×0.5)는 임의 배율. 실제 D-AA 친화도가 높아도 false-negative 발생 가능.

    Args:
        sequence: 후보 아미노산 서열
        extra_scores: 기존 extra_scores (has_dota, smiles 키 참조용)

    Returns:
        dict with keys:
            is_ood (bool): OOD 조건 하나 이상 충족 시 True
            ood_reasons (List[str]): OOD 사유 목록
            recommended_for_decision (bool): False = surrogate 신뢰 전 격리 권장
    """
    reasons: List[str] = []
    es = extra_scores or {}

    # 1. D-AA 토큰 감지: 소문자 아미노산 = D-AA 관례 (표준 20종 소문자 subset)
    _D_AA_TOKENS = set("acdefghiklmnpqrstvwy")
    d_aa_detected = any(ch in _D_AA_TOKENS for ch in sequence)
    if d_aa_detected:
        reasons.append(
            "D-AA 토큰 감지(소문자): surrogate(halflife/ADMET)는 L-AA 기준 학습 — D-AA 효과 미반영"
        )

    # 2. DOTA 태그 감지: extra_scores 내 has_dota / dota_chelator 키
    dota_detected = bool(es.get("has_dota") or es.get("dota_chelator"))
    if dota_detected:
        reasons.append("DOTA 킬레이터 결합: 소분자/펩타이드 ADMET 모델 모두 OOD")

    # 3. [도메인 보정 2026-06-23] cyclic SS-bond는 OOD 트리거에서 제외.
    # 우리 타겟이 본질적으로 cyclic SST-14(Cys3-Cys14)라 cyclic이 "정상"이며,
    # cyclic을 OOD로 판정하면 모든 핵심 후보가 Pareto front-0 배제·surrogate 감쇠되어
    # 발굴이 망가진다. OOD는 D-AA(소문자)·DOTA(비펩타이드 화학) 등 surrogate 분포
    # 밖 변형에만 적용. (hum B3 제안의 cyclic OOD 논리는 우리 도메인과 불일치.)
    # cyclic 펩타이드의 hc50 등 절대값 해석 한계는 별도(용혈게이트 NA·상대순위)로 처리.

    is_ood = bool(reasons)
    return {
        "is_ood": is_ood,
        "ood_reasons": reasons,
        "recommended_for_decision": not is_ood,
    }


def _apply_alternative_scoring(
    candidates: List["CandidateResult"],
    iter_dir: "Path",
    iteration: int,
    bo_optimizer: Optional[Any] = None,
) -> Tuple[List["CandidateResult"], List[int]]:
    """FlexPepDock 결과에 대안 스코어링 체인을 적용합니다 (optional).

    각 단계는 독립적으로 graceful skip됩니다.

    파이프라인:
        0. cheap objectives — 반감기 + ADMET surrogate (모든 후보)
        0b. OOD gate       — D-AA/DOTA/cyclic 후보 surrogate 신뢰도 soft 격리 (B3)
        1. GNINA rescore  — PDB 파일이 있으면 CNN 스코어 추가 (dry-run fallback)
        2. ECR consensus  — GNINA + ddG 통합 순위 (gnina 결과 있을 때만)
        3. Pareto ranking — 단발 비지배 정렬 (NSGA-II 정렬 단계 활용, pymoo 필요; 세대 루프 없음)
        4. BO suggest     — 다음 iteration용 추천 위치/잔기 반환 (B2 패치: guidance 전달용 포지션 반환)

    Args:
        candidates: FlexPepDock 결과 CandidateResult 리스트
        iter_dir: 현재 iteration PDB 파일 디렉토리
        iteration: 현재 iteration 번호 (로그용)
        bo_optimizer: 이미 생성된 BayesianPeptideOptimizer 인스턴스 (None이면 BO 단계 skip)

    Returns:
        Tuple of:
            candidates: 리스트 (in-place 수정 + 반환). 각 CandidateResult.extra_scores 갱신.
            bo_suggested_positions: BO suggest() 결과 포지션 리스트. BO 미실행 시 빈 리스트.
    """
    if not candidates:
        return candidates, []

    prefix = f"  [alt-score iter{iteration:02d}]"

    # ------------------------------------------------------------------
    # Step 0: 다목적 cheap objectives — 반감기(half-life) + ADMET surrogate
    #   서열만으로 계산(저비용). 모든 후보에 적용. selectivity(off-target 실제
    #   도킹)는 비싸므로 top-K 에서만 별도 단계로 수행한다.
    #   honest disclaimer: half_life/admet 은 ranking surrogate (임상 수치 아님).
    # ------------------------------------------------------------------
    try:
        from .multiobjective import cheap_objectives
        for cand in candidates:
            obj = cheap_objectives(cand.sequence)
            cand.extra_scores.update({
                k: v for k, v in obj.items() if k != "sequence"
            })
        hl_vals = [c.extra_scores.get("half_life_h") for c in candidates
                   if c.extra_scores.get("half_life_h") == c.extra_scores.get("half_life_h")]
        if hl_vals:
            print(
                f"{prefix} cheap-objectives: half-life {min(hl_vals):.1f}~{max(hl_vals):.1f}h, "
                f"ADMET surrogate computed for {len(candidates)} candidates",
                file=sys.stderr,
            )
    except Exception as exc:
        print(f"{prefix} cheap-objectives failed (non-fatal): {exc}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 0b: OOD 게이트 — D-AA/DOTA/cyclic 후보 surrogate 신뢰도 soft 격리 (B3 패치)
    #   hард reject 아닌 soft: surrogate 값 _OOD_DAMPING 배율 감쇠 + 플래그 기록.
    #   Pareto hard_violations=1 처리로 front-0 배제 (D-AA 전용 레이어 분리는 미래 작업).
    #   honest: 감쇠 배율(_OOD_DAMPING=0.5)은 임의값 — 정확도 향상 보장 X, 오염 격리 목적.
    # ------------------------------------------------------------------
    n_ood = 0
    for cand in candidates:
        ood_info = _detect_ood_flags(cand.sequence, cand.extra_scores)
        cand.extra_scores["is_ood"] = ood_info["is_ood"]
        cand.extra_scores["ood_reasons"] = ood_info["ood_reasons"]
        cand.extra_scores["recommended_for_decision"] = ood_info["recommended_for_decision"]
        if ood_info["is_ood"]:
            n_ood += 1
            # surrogate 값 감쇠: stability_norm, admet_score 신뢰도 하락 표시
            for key in ("stability_norm", "admet_score"):
                if key in cand.extra_scores and cand.extra_scores[key] is not None:
                    cand.extra_scores[key] = float(cand.extra_scores[key]) * _OOD_DAMPING
                    cand.extra_scores[f"{key}_ood_damped"] = True
    if n_ood:
        print(
            f"{prefix} OOD gate: {n_ood}/{len(candidates)} 후보 OOD 감지 "
            f"(surrogate ×{_OOD_DAMPING} 감쇠, Pareto hard_violation 처리)",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # Step 0.5: pepADMET 실제 독성 ML 추론 (배치 subprocess) → admet_score 페널티
    #   2026-06-09 B: pepADMET GNN(toxicity_early_stop.pth)을 pepadmet env 로 배치 추론.
    #   독성 후보는 admet_score 에 페널티. 미설치/실패 시 graceful skip (fail-closed: 가짜 안전판정 X).
    # ------------------------------------------------------------------
    try:
        import os as _os
        if _os.environ.get("SST_DISABLE_PEPADMET_TOX", "").lower() in ("1", "true", "yes"):
            raise RuntimeError("pepADMET toxicity disabled via SST_DISABLE_PEPADMET_TOX")
        from .multiobjective import predict_toxicity_for_sequences, apply_toxicity_to_extra
        tox_seqs = [c.sequence for c in candidates if not c.fail_reason and c.sequence]
        tox_map = predict_toxicity_for_sequences(tox_seqs)
        if tox_map:
            n_toxic = 0
            for cand in candidates:
                tox = tox_map.get(cand.sequence)
                if tox:
                    apply_toxicity_to_extra(cand.extra_scores, tox)
                    if tox.get("is_toxic"):
                        n_toxic += 1
            print(f"{prefix} pepADMET toxicity: {n_toxic}/{len(tox_map)} toxic (admet 페널티 반영)",
                  file=sys.stderr)
    except Exception as exc:
        print(f"{prefix} pepADMET toxicity skipped (non-fatal): {exc}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 1: GNINA rescore (optional, dry-run when binary missing)
    # ------------------------------------------------------------------
    gnina_scores_by_id: Dict[str, Dict[str, float]] = {}
    if _HAS_GNINA:
        pdb_paths: List[str] = []
        cand_ids: List[str] = []
        for cand in candidates:
            if cand.fail_reason:
                continue
            # cand_{NNN}.pdb 형태로 PDB 위치 추론
            try:
                num_str = cand.candidate_id.split("cand")[-1]
                pdb_path = iter_dir / f"cand_{int(num_str):03d}.pdb"
            except (ValueError, IndexError):
                continue
            if pdb_path.exists():
                pdb_paths.append(str(pdb_path))
                cand_ids.append(cand.candidate_id)

        if pdb_paths:
            try:
                gnina_results = batch_gnina_rescore(pdb_paths, max_workers=2)
                for cand_id, scores in zip(cand_ids, gnina_results):
                    gnina_scores_by_id[cand_id] = scores
                dry = any(s.get("gnina_dry_run") for s in gnina_results)
                mode_tag = "dry-run" if dry else "live"
                print(
                    f"{prefix} GNINA rescore {len(pdb_paths)} PDBs [{mode_tag}]",
                    file=sys.stderr,
                )
            except Exception as exc:
                print(
                    f"{prefix} GNINA rescore failed (non-fatal): {exc}",
                    file=sys.stderr,
                )

    # ------------------------------------------------------------------
    # Step 2: ECR consensus (GNINA scores 있으면 ddG와 통합)
    #
    # 이슈6 enforce: GNINA_DRY_RUN_ENFORCE=1(기본)일 때, dry-run 점수(gnina_dry_run==1.0)
    # 후보는 ECR 계산에서 제외한다. dry-run은 "측정 안 됨" — 0점(중립값)으로 순위에
    # 기여하는 것 자체가 부정확한 결과를 낳는다.
    # 제외된 후보의 ecr_score는 설정되지 않으며, Pareto 단계에서 ECR 폴백 없이
    # admet_score만 사용된다 (Pareto Step 3의 drug_val 폴백 로직 유지).
    # ------------------------------------------------------------------
    ecr_by_id: Dict[str, float] = {}
    if _HAS_GNINA and gnina_scores_by_id:
        # dry-run enforce: gnina_dry_run==1.0 후보를 ECR 입력에서 제외
        n_dry_run_excluded = 0
        ecr_input: List[Dict] = []
        for cand in candidates:
            g = gnina_scores_by_id.get(cand.candidate_id, {})
            is_dry_run = float(g.get("gnina_dry_run", 0.0)) == 1.0
            if _GNINA_DRY_RUN_ENFORCE and is_dry_run:
                # dry-run 후보: ECR에 포함하지 않음 (tag 기록만)
                n_dry_run_excluded += 1
                continue
            ecr_input.append({
                "candidate_id": cand.candidate_id,
                "ddg": cand.ddg,
                "gnina_cnn_score": g.get("gnina_cnn_score", float("nan")),
                "gnina_cnn_affinity": g.get("gnina_cnn_affinity", float("nan")),
                "gnina_vina_score": g.get("gnina_vina_score", float("nan")),
            })
        if n_dry_run_excluded:
            print(
                f"{prefix} ECR dry-run enforce: {n_dry_run_excluded} dry-run 후보 ECR 제외 "
                f"(gnina_dry_run==1.0, GNINA 미측정 → 순위 기여 금지)",
                file=sys.stderr,
            )
        if not ecr_input:
            # 모든 후보가 dry-run이거나 gnina 스코어 없음 → ECR skip
            print(
                f"{prefix} ECR consensus skipped (no live GNINA scores)",
                file=sys.stderr,
            )
        else:
            try:
                ecr_results = exponential_rank_consensus(
                    ecr_input,
                    score_keys=["ddg", "gnina_cnn_score", "gnina_cnn_affinity", "gnina_vina_score"],
                )
                for row in ecr_results:
                    ecr_by_id[row["candidate_id"]] = float(row.get("ecr_score", 0.0))
                print(
                    f"{prefix} ECR consensus computed for {len(ecr_results)} candidates "
                    f"(live GNINA only)",
                    file=sys.stderr,
                )
            except Exception as exc:
                print(
                    f"{prefix} ECR consensus failed (non-fatal): {exc}",
                    file=sys.stderr,
                )

    # GNINA 및 ECR 스코어를 extra_scores에 저장 (Pareto 단계 이전)
    for cand in candidates:
        if cand.candidate_id in gnina_scores_by_id:
            cand.extra_scores.update(gnina_scores_by_id[cand.candidate_id])
        if cand.candidate_id in ecr_by_id:
            cand.extra_scores["ecr_score"] = ecr_by_id[cand.candidate_id]

    # ------------------------------------------------------------------
    # Step 3: Pareto ranking — 단발 비지배 정렬 (pymoo NonDominatedSorting 활용; 세대 루프 없음)
    # ------------------------------------------------------------------
    if _HAS_PARETO:
        pareto_input: List[Dict] = []
        valid_sequences = [
            (cand.candidate_id, cand.sequence)
            for cand in candidates
            if not cand.fail_reason
        ]
        for cand in candidates:
            # stability: Step 0 에서 계산한 반감기 기반 surrogate(0~1, 높을수록 안정).
            #   부재 시 clash 기반 proxy 로 폴백.
            #
            # 이슈9 enforce: halflife_source='none'(휴리스틱·RF 둘 다 실패) 이면
            # stability_norm 값이 있어도 "측정 안 됨"으로 간주 → Pareto 기여 0.
            # clash proxy를 사용하지 않는 이유: clash는 구조적 충돌 지표이며
            # 반감기와 다른 차원. 미측정 반감기를 clash로 대체하면 "반감기 unknown"을
            # 낮은 clash(= 좋은 점수)로 보상하는 fail-open이 된다.
            # halflife_source='heuristic': 단일 모델만 작동, 앙상블보다 낮은 신뢰.
            # 하지만 heuristic은 여전히 측정된 값이므로 그대로 사용.
            halflife_source = cand.extra_scores.get("halflife_source")
            stability_val = cand.extra_scores.get("stability_norm")
            if halflife_source == "none":
                # 반감기 측정 실패 → Pareto stability 차원 기여 0 (미측정)
                stability_val = 0.0
                cand.extra_scores["stability_norm_enforced"] = "halflife_source=none → 0.0"
            elif stability_val is None:
                # halflife_source 자체가 없으면 (Step 0 skip) → clash proxy
                stability_val = max(0.0, 40.0 - cand.clash_score) / 40.0
            # druggability: ADMET surrogate(0~1). 부재 시 ECR 폴백.
            # 이슈10 추가 확인: admet_score가 있어도 available=False로 인해 변경이
            # 없었다면 그 값은 PharmaProperties에서 온 physicochemical 점수.
            # apply_toxicity_to_extra에서 available=False는 이미 early return(no-op).
            # pepADMET 전체 skip 시 admet_score는 physicochemical만 반영 — 이미 정직.
            drug_val = cand.extra_scores.get("admet_score")
            if drug_val is None:
                drug_val = ecr_by_id.get(cand.candidate_id, 0.0)
            # OOD 후보는 Pareto에 포함하되 hard_violations=1 처리 → front-0 배제
            is_ood = cand.extra_scores.get("is_ood", False)
            other_sequences = [
                seq for candidate_id, seq in valid_sequences
                if candidate_id != cand.candidate_id
            ]
            if other_sequences:
                diversity = sum(
                    _compute_mean_pairwise_hamming([cand.sequence, other_seq])
                    for other_seq in other_sequences
                ) / len(other_sequences)
            else:
                diversity = 0.0
            hard_violations = 1 if (cand.fail_reason or is_ood) else 0
            # pocket_contacts=0: soft penalty (hard reject 아님). 키 없으면 graceful skip.
            if (
                "pocket_contacts" in cand.extra_scores
                and type(cand.extra_scores["pocket_contacts"]) is int
                and cand.extra_scores["pocket_contacts"] == 0
                and hard_violations == 0
            ):
                hard_violations = 0.5
            pareto_input.append({
                "candidate_id": cand.candidate_id,
                "ddG": cand.ddg,
                "stability": float(stability_val),      # 반감기 기반(높을수록 좋음)
                "druggability": float(drug_val),         # ADMET 합리성(높을수록 좋음)
                "diversity": float(diversity),            # Hamming pairwise 평균 (crowding distance 실효화용)
                "hard_violations": hard_violations,
                "clash_score": cand.clash_score,
                # OOD 메타: 리더보드/리포트 추적용
                "is_ood": is_ood,
                "ood_reasons": cand.extra_scores.get("ood_reasons", []),
            })
        try:
            ranked = pareto_rank_candidates(pareto_input, clash_threshold=10.0)
            # pareto_rank, crowding_distance를 candidate extra_scores에 반영
            rank_map: Dict[str, Dict[str, Any]] = {
                r["candidate_id"]: {
                    "pareto_rank": r.get("pareto_rank", 999),
                    "crowding_distance": r.get("crowding_distance", 0.0),
                }
                for r in ranked
            }
            for cand in candidates:
                cand.extra_scores.update(rank_map.get(cand.candidate_id, {}))
            front0_count = sum(1 for r in ranked if r.get("pareto_rank", 999) == 0)
            print(
                f"{prefix} Pareto ranking done — front-0: {front0_count}/{len(ranked)} candidates",
                file=sys.stderr,
            )
        except Exception as exc:
            print(
                f"{prefix} Pareto ranking failed (non-fatal): {exc}",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # Step 4: Bayesian Optimization suggest (B2 패치 2026-06-23)
    #   bo_optimizer 가 전달되면 BO 가용한 것 (runner 가 _HAS_BO 시에만 생성·전달).
    #   변경: suggest() 결과 포지션을 bo_suggested_positions에 수집해 반환 → runner에서
    #   다음 iteration bandit guidance로 전달 (탐색-활용 피드백 루프).
    #   하위호환: BO 미실행/실패 시 빈 리스트 반환, 기존 동작 보존.
    # ------------------------------------------------------------------
    bo_suggested_positions: List[int] = []   # BO 포지션 수집 버퍼 (반환용)

    if bo_optimizer is not None:
        valid_obs = [
            c for c in candidates
            if not c.fail_reason and c.ddg < 900
        ]
        if len(valid_obs) >= 2:
            try:
                obs_dicts = [
                    {
                        "sequence": c.sequence,
                        "ddg": c.ddg,
                        "ecr_score": ecr_by_id.get(c.candidate_id, 0.0),
                    }
                    for c in valid_obs
                ]
                bo_optimizer.fit(obs_dicts)
                suggestions = bo_optimizer.suggest(
                    n=3,
                    reference_seq=valid_obs[0].sequence,
                )
                if suggestions:
                    top_pos = [s.get("position") for s in suggestions[:3]]
                    # B2: 포지션 수집 (None 필터링)
                    bo_suggested_positions = [p for p in top_pos if p is not None]
                    print(
                        f"{prefix} BO suggest top-3 positions: {top_pos} "
                        f"→ guidance 전달 예정",
                        file=sys.stderr,
                    )
                    # 감사 추적: 첫 유효 관측 후보에 bo_acquisition_value 기록
                    if suggestions and valid_obs:
                        valid_obs[0].extra_scores["bo_acquisition_value"] = float(
                            suggestions[0].get("acquisition_value", float("nan"))
                        )
                    # bo_top_positions: Pareto front-0 후보에도 기록 (모니터링용)
                    for sugg in suggestions:
                        pos = sugg.get("position")
                        if pos is None:
                            continue
                        for cand in candidates:
                            if cand.extra_scores.get("pareto_rank") == 0:
                                top_list = cand.extra_scores.setdefault("bo_top_positions", [])
                                if pos not in top_list:
                                    top_list.append(pos)
            except Exception as exc:
                print(
                    f"{prefix} BO suggest failed (non-fatal): {exc}",
                    file=sys.stderr,
                )

    return candidates, bo_suggested_positions
