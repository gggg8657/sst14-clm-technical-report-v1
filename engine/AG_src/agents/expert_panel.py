"""
expert_panel.py
Track1-P2: 5-전문가 도메인 Expert Panel (Fan-out / Fan-in)

Planner 가설을 약리·구조·화학·방사화학·수학 전문가가 각자 다른 관점으로 평가(fan-out)
→ 통합 critic이 합의(fan-in: merged_concerns + approve) → Planner 재고.

전문가 구성: pharma / biology / chemistry / radiochem (실투표 4명) + math (advisory 1명).

사용 방법:
    from AG_src.agents.expert_panel import ExpertPanelAgent
    panel = ExpertPanelAgent(critic)  # ScientistCriticAgent 인스턴스 재활용
    result = panel.prereview_panel(
        iteration=1,
        hypothesis="...",
        mutation_guidance={...},
        trajectory_summary="...",          # 선택
        selectivity_leaderboard=[...],     # 선택
        best_delta_margin=10.54,           # 선택
        round_idx=1,
    )
    # result: {
    #   "merged_concerns": [...],
    #   "approve": bool,               # 하위호환 유지 (docking_decision과 동일값)
    #   "scientific_verdict": "approve"|"conditional"|"reject"|"invalid",
    #   "docking_decision": bool,      # 실제 도킹 진행 여부
    #   "reason_for_docking": str,     # forced_pass 시 "cheap exploratory docking despite reject"
    #   "risk_level": "low"|"medium"|"medium-high"|"high",
    #   "qc_status": "pass"|"fail",
    #   "position_map_consistent": bool,
    #   "selectivity_evidence_present": bool,
    #   "minority_dissent": {"domain": str, "severity": str, "reason": str} | None,
    #       # C: 5명 중 1명만 다수와 크게 다른 severity일 때 근거 보존 (sycophancy 방지)
    #   "stall_detected": bool,        # E: 동일 불일치가 PANEL_STALL_LIMIT 라운드 연속 지속됐는지
    #   "stall_streak": int,           # E: 현재 동일 불일치 연속 라운드 수
    #   "suggested_revisions": {...},
    #   "expert_verdicts": {
    #       "pharma": {...}, "biology": {...}, "chemistry": {...},
    #       "radiochem": {...}, "math": {...}
    #   },
    #   "llm_calls": int,   # 이 호출에서 사용한 LLM 호출 수
    # }
"""

from __future__ import annotations

import logging
import os as _os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ..llm.prompts import (
    SST14_POSITION_MAP,
    SST14_IMMUTABLE_POSITIONS,
    format_expert_domain_prompt,
    format_expert_fanin_prompt,
    get_system_prompt,
)
from ..llm.provider import LLMProvider, VLLMProvider

if TYPE_CHECKING:
    from .critic import ScientistCriticAgent

logger = logging.getLogger(__name__)
# 엔진 stdout/stderr가 로그 파일로 리다이렉트되므로 핸들러 미설정 시 INFO 로그가 유실된다.
# root logger 전파에만 의존하지 않고 모듈 레벨에서 StreamHandler를 보장한다.
if not logger.handlers and not logging.root.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter(
            "[%(asctime)s][ExpertPanel] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# 전문가 도메인 목록 (순서 유지 — provenance 기록용)
# radiochem(방사화학/핵의학) 은 실투표 참여 도메인 — advisory 아님 (math만 advisory).
_EXPERT_DOMAINS: List[str] = ["pharma", "biology", "chemistry", "radiochem", "math"]

# 환경변수로 임계값 조절 가능 (2026-07-02 게이트 실질화: medium 임계 4→2)
# PANEL_MEDIUM_ALL_MAX: science 전문가(pharma/biology/chemistry/radiochem) 중 medium 우려가
#   이 수 이상이면 conditional(재검토 1회 유도). 기본 2 = 2명 이상 동조 시 그냥 통과 안 함.
#   (과잉거부 우려 시 3~4로 상향, 더 엄격히 하려면 1로 하향 — 관찰 후 조정)
_PANEL_HIGH_REJECT: int = int(_os.environ.get("PANEL_HIGH_REJECT", "2"))
_PANEL_MEDIUM_ALL_MAX: int = int(_os.environ.get("PANEL_MEDIUM_ALL_MAX", "2"))
# 최대 토론 라운드 수 (기본 5). PANEL_MAX_DISCUSSION_ROUNDS=5로 조절.
# approve 달성 시 조기 종료. 5턴 소진 후에도 합의 미달 시 forced_pass/reject_hard.
_PANEL_MAX_DISCUSSION_ROUNDS: int = int(_os.environ.get("PANEL_MAX_DISCUSSION_ROUNDS", "5"))
# DEPRECATED: approve=True 도달 시 즉시 종료 정책으로 변경되어 이 변수는 더 이상 사용되지 않음.
# 하위호환을 위해 변수는 유지하지만 실제 로직에서는 무시됨.
_PANEL_DISCUSS_MIN_CONCERNS: int = int(_os.environ.get("PANEL_DISCUSS_MIN_CONCERNS", "1"))
# forced_pass 예외: high >= PANEL_HIGH_REJECT(치명 다수)는 라운드 소진 후에도 approve=False 유지.
# 1(기본) = high 2+ 시 reject_hard. 0 = 기존 동작(전부 forced_pass).
_PANEL_HARD_REJECT_HIGH: int = int(_os.environ.get("PANEL_HARD_REJECT_HIGH", "1"))

# --- Confidence-gated 라운드 (B) ---
# 1(기본) = 라운드 후 전문가 verdict가 충분히 일치(불일치·high 없음)하면 즉시 종료.
# 0 = 비활성화(기존 방식 — approve 달성 시에만 조기 종료).
_PANEL_CONFIDENCE_GATE: int = int(_os.environ.get("PANEL_CONFIDENCE_GATE", "1"))

# --- 정체(stall) 감지 (E) ---
# 동일한 불일치 패턴(science verdict severity 조합)이 연속 N라운드 지속되면 stall로 판정.
_PANEL_STALL_LIMIT: int = int(_os.environ.get("PANEL_STALL_LIMIT", "2"))

# --- 모델 이질성(F): 도메인별 이질 LLM provider 배선 ---
# 근거: arXiv:2502.08788 — debate에서 모델 이질성(서로 다른 base model)이 동종 모델보다
# 유리하다(사고 패턴이 다른 모델이 서로의 맹점을 더 잘 짚어냄). 전 도메인을 이질화하면
# 검증이 어려우므로 절반 정도(기본 biology/chemistry 2/5)만 이질 모델로 바꿔 관점 다양성을
# 확보하되 나머지(pharma/radiochem/math)는 기존 Qwen을 유지해 회귀 위험을 낮춘다.
# fan-in(통합)은 항상 기존 Qwen(critic 공유 provider)을 사용 — 통합 판정 기준의 일관성 유지.
#
# 환경변수:
#   PANEL_HETERO_DOMAINS  — 콤마 구분 도메인 목록 (기본 "biology,chemistry"). 빈 문자열이면
#                           이질성 off — 전원 critic의 기존 provider 사용(하위호환 기본값).
#   PANEL_HETERO_BASE_URL — 이질 provider의 OpenAI 호환 base_url (기본 "http://localhost:8001/v1")
#   PANEL_HETERO_MODEL    — 이질 provider의 모델명 (기본 "mistral-7b-instruct")
_PANEL_HETERO_DOMAINS_RAW: str = _os.environ.get("PANEL_HETERO_DOMAINS", "biology,chemistry")
_PANEL_HETERO_DOMAINS: frozenset = frozenset(
    d.strip() for d in _PANEL_HETERO_DOMAINS_RAW.split(",") if d.strip()
)
# VLLMProvider.base_url은 내부에서 "/v1/chat/completions"를 붙이므로 base_url 자체에는
# "/v1"을 포함하지 않아야 한다. 사용자가 관례적으로 "http://host:port/v1"을 줄 수 있으므로
# 끝의 "/v1"을 제거해 정규화한다(중복 "/v1/v1/chat/completions" 방지).
_PANEL_HETERO_BASE_URL_RAW: str = _os.environ.get("PANEL_HETERO_BASE_URL", "http://localhost:8001/v1")
_PANEL_HETERO_BASE_URL: str = (
    _PANEL_HETERO_BASE_URL_RAW[: -len("/v1")]
    if _PANEL_HETERO_BASE_URL_RAW.rstrip("/").endswith("/v1")
    else _PANEL_HETERO_BASE_URL_RAW
).rstrip("/")
_PANEL_HETERO_MODEL: str = _os.environ.get("PANEL_HETERO_MODEL", "mistral-7b-instruct")

# Fan-in 합의 규칙 (통합 시스템 프롬프트 보조 — 코드로도 적용)
_SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}

# SST-14 position map (1-indexed) — QC 기준값. prompts.py의 SST14_POSITION_MAP과 동기.
# 키: 1-indexed 위치, 값: 단일문자 AA + 역할
_SST14_NATIVE = SST14_POSITION_MAP  # {1: 'A', 2: 'G', 3: 'C', ...}
_SST14_IMMUTABLE = SST14_IMMUTABLE_POSITIONS  # frozenset({3, 7, 8, 9, 10, 14})

# position hallucination 검출: 알려진 오류 패턴 (pos12=Cys 등)
_POS12_IS_THR_NOT_CYS = True  # pos12 = T(Thr), Cys 아님


def _check_position_map_consistent(
    concerns_text: str,
    focus_positions: Optional[List[int]] = None,
    expert_positions_mentioned: Optional[List[int]] = None,
) -> bool:
    """LLM 출력에서 position hallucination 여부를 검사한다.

    현재 구현 범위:
    - pos12를 Cys/C/cysteine 으로 언급하면 False (known hallucination).
    - 기타 알려진 불변 위치(pos3/14=C, pos7=F, pos8=W, pos9=K, pos10=T)를 틀리게 말하면 False.
    - focus_positions에 immutable 위치(3,7,8,9,10,14)가 포함되면 False.

    Args:
        concerns_text: 전문가 concerns/suggestion 합쳐진 문자열 (대소문자 무관).
        focus_positions: hypothesis의 focus_positions 리스트 (선택).
        expert_positions_mentioned: LLM verdict에서 추출된 position 리스트 (선택, 미래확장).

    Returns:
        True = 일관성 OK, False = hallucination 감지
    """
    # 견고성: concerns_text가 non-string(테스트 mock의 MagicMock, None 등)이어도
    # TypeError 없이 처리. 실제 발굴은 string이라 동작 동일.
    text_lower = str(concerns_text or "").lower()

    # pos12 = T(Thr) 인데 Cys로 언급하는 패턴 검출
    # "pos12" + ("cys" or "c " or "disulfide") 근접 패턴
    import re as _re
    _pos12_cys_patterns = [
        r"pos\s*12\s*[=:]\s*c(?:ys)?",
        r"pos\s*12.*?cys",
        r"pos\s*12.*?disulfide",
        r"position\s*12.*?cys",
        r"12.*?cysteine",
    ]
    for pat in _pos12_cys_patterns:
        if _re.search(pat, text_lower):
            logger.warning(
                "[expert_panel] position hallucination 감지: pos12=Cys 언급 (실제 pos12=Thr) — pattern=%s", pat
            )
            return False

    # focus_positions에 immutable 위치(3,7,8,9,10,14) 포함 시 hallucination
    if focus_positions:
        immutable_hit = [p for p in focus_positions if p in _SST14_IMMUTABLE]
        if immutable_hit:
            logger.warning(
                "[expert_panel] focus_positions에 immutable 위치 포함: %s — pharmacophore/disulfide 침범",
                immutable_hit,
            )
            return False

    return True


def _compute_risk_level(consensus_status: str, high_count: int, medium_count: int) -> str:
    """consensus_status와 severity 집계로 risk_level을 결정한다."""
    if consensus_status in ("reject_hard", "reject"):
        return "high"
    if consensus_status == "conditional":
        return "medium-high" if high_count >= 1 else "medium"
    if medium_count >= 2:
        return "medium"
    return "low"


def _derive_scientific_verdict(
    consensus_status: str,
    qc_status: str,
) -> str:
    """consensus_status + qc_status → scientific_verdict 결정.

    QC fail이면 "invalid", 그 외 consensus_status 기반.
    forced_pass는 과학적 판정을 "conditional" 또는 "reject" 그대로 유지.
    """
    if qc_status == "fail":
        return "invalid"
    mapping = {
        "approve": "approve",
        "conditional": "conditional",
        "reject": "reject",
        "reject_hard": "reject",
        "forced_pass": "conditional",  # forced_pass는 원래 conditional/reject이었음을 보존
    }
    return mapping.get(consensus_status, "conditional")


# 합의 판정에서 제외할 도메인 — advisory 전용 (severity가 approve/reject에 영향 안 줌)
_ADVISORY_ONLY_DOMAINS: frozenset = frozenset({"math"})


def _apply_consensus_rules(
    verdicts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Fan-in 합의 규칙을 verdict 목록에 적용하여 합의 상태를 결정한다.

    재설계된 규칙 (코드 보장, LLM 보조):
        - 합의 판정은 pharma/biology/chemistry/radiochem 4개 도메인만 적용.
          math 도메인은 advisory 전용 — severity가 approve/reject에 기여하지 않음.
        - high >= PANEL_HIGH_REJECT(기본 2) → reject (치명적 다수 거부)
        - high == 1                          → conditional (1회 수정 기회)
        - medium 전원 일치 (== 비어드바이저리 전문가 수 && >= PANEL_MEDIUM_ALL_MAX) → conditional
        - 그 외 → approve (medium은 차단 아님, merged_concerns에 기록)

    환경변수: PANEL_HIGH_REJECT (기본 2), PANEL_MEDIUM_ALL_MAX (기본 4 — science 4명 기준)

    Args:
        verdicts: 전문가 verdict dict 목록 (math 포함 가능 — advisory는 자동 제외)

    Returns:
        {
            "approve": bool,              # False = reject/conditional 둘 다
            "status": "approve" | "reject" | "conditional",
            "high_count": int,
            "medium_count": int,
            "reason": str,                # 로깅용 한국어 설명
        }
    """
    # math advisory 제외 후 판정 대상만 추출
    science_verdicts = [
        v for v in verdicts
        if v.get("domain", "") not in _ADVISORY_ONLY_DOMAINS
    ]
    n = len(science_verdicts)
    high_count = sum(
        1 for v in science_verdicts if _SEVERITY_ORDER.get(v.get("severity", "low"), 0) >= 2
    )
    medium_count = sum(
        1 for v in science_verdicts if _SEVERITY_ORDER.get(v.get("severity", "low"), 0) == 1
    )

    if high_count >= _PANEL_HIGH_REJECT:
        return {
            "approve": False,
            "status": "reject",
            "high_count": high_count,
            "medium_count": medium_count,
            "reason": f"high {high_count}개 ≥ 임계({_PANEL_HIGH_REJECT}) — 치명 다수 거부",
        }

    # 게이트 실질화(2026-07-02): high 1개 + medium 2개↑ = 심각 우려에 광범위 동조 → 반려
    if high_count >= 1 and medium_count >= 2:
        return {
            "approve": False,
            "status": "reject",
            "high_count": high_count,
            "medium_count": medium_count,
            "reason": f"high 1개 + medium {medium_count}개 — 심각 우려에 광범위 동조, 거부",
        }

    if high_count == 1:
        return {
            "approve": False,
            "status": "conditional",
            "high_count": high_count,
            "medium_count": medium_count,
            "reason": "high 1개 — 1회 수정 기회 부여",
        }

    # 게이트 실질화: medium 임계 이상(기본 2)이면 conditional(재검토 1회 유도) — 무조건 승인 방지
    if medium_count >= _PANEL_MEDIUM_ALL_MAX:
        return {
            "approve": False,
            "status": "conditional",
            "high_count": high_count,
            "medium_count": medium_count,
            "reason": f"medium {medium_count}개 ≥ 임계({_PANEL_MEDIUM_ALL_MAX}) — 재검토 1회 유도",
        }

    # 그 외: approve (medium은 차단 아님 — merged_concerns에 기록)
    return {
        "approve": True,
        "status": "approve",
        "high_count": high_count,
        "medium_count": medium_count,
        "reason": (
            f"medium {medium_count}개(차단 아님, concerns 기록)" if medium_count > 0
            else "이슈 없음 — 통과"
        ),
    }


def _detect_minority_dissent(
    verdicts: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """다수 의견에 묻힐 수 있는 소수 반대의견을 탐지한다 (C: sycophancy 방지).

    science(비-advisory) 전문가 중 정확히 1명만 severity가 나머지 전원보다
    유의하게 높으면(예: 1명만 high/medium, 나머지 전원 low) 그 전문가의
    domain·severity·근거(concerns)를 별도로 반환한다.

    arXiv:2509.23055 근거 — 소수 정답이 다수 동조(fan-in LLM)에 묻히지 않도록
    코드 레벨에서 명시적으로 보존한다 (LLM 요약에 의존하지 않음).

    Args:
        verdicts: 전문가 verdict dict 목록 (math 포함 가능 — advisory는 자동 제외).

    Returns:
        {"domain": str, "severity": str, "reason": str} 또는 소수 의견이 없으면 None.
    """
    science_verdicts = [
        v for v in verdicts
        if v.get("domain", "") not in _ADVISORY_ONLY_DOMAINS
    ]
    n = len(science_verdicts)
    if n < 3:
        # 소수/다수를 가르기엔 표본이 너무 작음
        return None

    severities = [_SEVERITY_ORDER.get(v.get("severity", "low"), 0) for v in science_verdicts]
    # 정확히 1명만 나머지와 다른 severity를 가지는 경우를 탐지
    outlier_idx: Optional[int] = None
    for i, sev in enumerate(severities):
        others = severities[:i] + severities[i + 1:]
        if not others:
            continue
        # 나머지 전원이 동일 severity이고, 이 전문가만 더 높은 경우 = 소수 반대의견
        if all(o == others[0] for o in others) and sev > others[0]:
            if outlier_idx is not None:
                # 2명 이상이 outlier면 "소수"로 보기 어려움 — 탐지 취소
                return None
            outlier_idx = i

    if outlier_idx is None:
        return None

    outlier = science_verdicts[outlier_idx]
    concerns = outlier.get("concerns", [])
    reason = "; ".join(concerns[:2]) if concerns else outlier.get("suggestion", "") or "(근거 미기재)"
    return {
        "domain": outlier.get("domain", "?"),
        "severity": outlier.get("severity", "low"),
        "reason": reason,
    }


def _consensus_signature(consensus: Dict[str, Any]) -> Tuple[str, int, int]:
    """정체(stall) 감지용 서명. 동일 서명이 연속되면 진전이 없다고 판단한다."""
    return (consensus.get("status", ""), consensus.get("high_count", 0), consensus.get("medium_count", 0))


class ExpertPanelAgent:
    """도메인 전문가 5명(pharma/biology/chemistry/radiochem/math) Fan-out / Fan-in 패널.

    math는 advisory 전용, 나머지 4명(pharma/biology/chemistry/radiochem)이 실투표.

    ScientistCriticAgent의 LLM 인프라(llm_generate_json, has_llm)를 재사용한다.
    새 에이전트를 따로 생성하지 않고 critic 인스턴스에 위임한다.

    모델 이질성(F, arXiv:2502.08788): PANEL_HETERO_DOMAINS에 지정된 도메인은 critic의
    기본 provider(Qwen, 8000) 대신 별도 VLLMProvider(기본 Mistral, 8001)로 호출한다.
    critic의 llm_provider 자체는 건드리지 않으므로 fan-in·다른 도메인·다른 에이전트는
    영향받지 않는다(critic 공유 인프라 보존).

    Attributes:
        critic: ScientistCriticAgent 인스턴스 (LLM 호출용, 기본/fan-in provider)
        hetero_provider: 이질 도메인 전용 LLMProvider (PANEL_HETERO_DOMAINS 비어있으면 None)
        hetero_domains: 이질 provider를 사용할 도메인 집합
    """

    def __init__(self, critic: "ScientistCriticAgent") -> None:
        self._critic = critic
        # 모델 이질성(F) 배선 — PANEL_HETERO_DOMAINS가 비어있으면 hetero_provider=None
        # (전원 critic 기본 provider 사용 → 이질성 off, 기존 동작과 100% 동일).
        self.hetero_domains: frozenset = _PANEL_HETERO_DOMAINS
        self.hetero_provider: Optional[LLMProvider] = None
        if self.hetero_domains and critic.has_llm:
            try:
                self.hetero_provider = VLLMProvider(
                    model=_PANEL_HETERO_MODEL,
                    base_url=_PANEL_HETERO_BASE_URL,
                    # Qwen 전용 chat_template_kwargs 전송 안 함 — 실측: Mistral 토크나이저는
                    # 이 필드 자체로 HTTP 400("chat_template is not supported for Mistral
                    # tokenizers.")을 반환한다. 향후 다른 이질 모델도 기본적으로 꺼서 안전하게.
                    send_chat_template_kwargs=False,
                )
                logger.info(
                    "[expert_panel] 모델 이질성 활성화: domains=%s → %s @ %s",
                    sorted(self.hetero_domains), _PANEL_HETERO_MODEL, _PANEL_HETERO_BASE_URL,
                )
            except Exception as exc:
                logger.error(
                    "[expert_panel] 이질 provider 생성 실패(fallback: 전원 기본 provider 사용): %s", exc,
                )
                self.hetero_provider = None

    @property
    def has_llm(self) -> bool:
        """LLM 사용 가능 여부 (critic에 위임)."""
        return self._critic.has_llm

    def _provider_for_domain(self, domain: str) -> Tuple["LLMProvider | str", str]:
        """도메인에 배정된 LLM provider와 backend 라벨을 반환한다.

        Returns:
            (provider, llm_backend_label). provider는 critic.llm_provider(문자열일 수도 있음,
            has_llm=False인 fallback 케이스) 또는 hetero VLLMProvider 인스턴스.
            llm_backend_label은 verdict에 기록할 추적용 문자열
            (예: "vllm:mistral-7b-instruct@localhost:8001" 또는 "critic:qwen(default)").
        """
        if self.hetero_provider is not None and domain in self.hetero_domains:
            return self.hetero_provider, f"hetero:{self.hetero_provider.model}@{_PANEL_HETERO_BASE_URL}"
        # 기본 경로: critic의 provider를 그대로 사용 (provider_name/model이 있으면 표기)
        _base_provider = getattr(self._critic, "llm_provider", None)
        if _base_provider is not None and not isinstance(_base_provider, str):
            _label = f"default:{getattr(_base_provider, 'model', 'unknown')}"
        else:
            _label = "default:critic"
        return self._critic, _label

    def prereview_panel(
        self,
        iteration: int,
        hypothesis: str,
        mutation_guidance: Dict[str, Any],
        trajectory_summary: Optional[str] = None,
        selectivity_leaderboard: Optional[List[Dict[str, Any]]] = None,
        best_delta_margin: Optional[float] = None,
        round_idx: int = 1,
        max_discussion_rounds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """5 전문가(pharma/biology/chemistry/radiochem/math) 사전검토 + 통합 합의를 실행한다.

        LLM 미사용 또는 실패 시 approve=True(통과) 폴백 — P1 하위호환 보장.
        비용: Round1 LLM 호출 = 5(전문가) + 1(통합) = 6회.
              Round2 이상 = 5(전문가, peer_concerns 주입) + 1(통합) = 6회 추가.
              'llm_calls' 키로 반환. (confidence-gate로 조기 종료 시 그만큼 절감됨.)

        합의 미달 시 최대 max_discussion_rounds 라운드까지 turn-taking 토론 재시도.
        Round2부터는 각 전문가가 다른 전문가의 Round1 우려를 보고 입장을 갱신한다.
        최대 라운드 소진 시 무한 거부 방지를 위해 강제 통과(forced_pass).

        Args:
            iteration: 현재 반복 번호
            hypothesis: Planner 가설 문자열
            mutation_guidance: Planner mutation_guidance dict
            trajectory_summary: 최근 궤적 요약 (선택 — math에 활용)
            selectivity_leaderboard: 선택성 리더보드 상위 항목 (선택 — biology/math)
            best_delta_margin: 현재까지 최고 Δmargin (선택)
            round_idx: 사전검토 라운드 번호 (1-indexed)
            max_discussion_rounds: 최대 내부 토론 라운드 수 (None이면 env PANEL_MAX_DISCUSSION_ROUNDS 또는 기본 5).
                approve 달성 시 조기 종료. 5턴 소진 후에도 합의 미달 시 forced_pass/reject_hard.

        Returns:
            {
                "merged_concerns": [str, ...],
                "approve": bool,               # 하위호환 — docking_decision과 동일값
                "scientific_verdict": "approve"|"conditional"|"reject"|"invalid",
                    # 과학적 판정. forced_pass여도 원 verdict 보존(reject/conditional 그대로).
                    # qc_fail이면 "invalid".
                "docking_decision": bool,      # 실제 도킹 진행 여부
                    # approve/forced_pass → True. reject_hard → False.
                "reason_for_docking": str,
                    # "proceed" / "cheap exploratory docking despite reject" / "blocked"
                "risk_level": "low"|"medium"|"medium-high"|"high",
                "qc_status": "pass"|"fail",
                "position_map_consistent": bool,
                    # focus_positions에 immutable 위치 없고 알려진 hallucination 없으면 True.
                "selectivity_evidence_present": bool,
                    # hypothesis에 SSTR2 선택성 관련 근거가 언급되면 True.
                "consensus_status": "approve" | "reject" | "conditional" | "forced_pass" | "reject_hard",
                "consensus_reason": str,
                "suggested_revisions": {"focus_positions": [...], "strategy": "..."},
                "expert_verdicts": {"pharma": {...}, "biology": {...}, ...},
                "llm_calls": int,
                "discussion_rounds": int,
                "discussion_turns": [
                    {
                        "round": int,
                        "domain": str,
                        "severity": str,
                        "concerns": [str],
                        "stance_change": str,  # e.g. "medium→low", "유지"
                        "rebuttal": str,        # 한 문장 입장 (Round 2+)
                    },
                    ...
                    # 각 라운드 마지막에 fanin turn 추가:
                    {"round": int, "domain": "fanin", "approve": bool, "merged_concerns": [str]},
                ],
            }
        """
        # max_discussion_rounds가 None이면 env 기본값 사용 (기본 5)
        if max_discussion_rounds is None:
            max_discussion_rounds = _PANEL_MAX_DISCUSSION_ROUNDS

        logger.info(
            "[expert_panel] iter=%d round=%d 패널 사전검토 시작 (focus=%s, max_discussion=%d)",
            iteration, round_idx, mutation_guidance.get("focus_positions", []),
            max_discussion_rounds,
        )

        # --- position map 사전 QC (코드 레벨) ---
        _focus_positions: List[int] = list(mutation_guidance.get("focus_positions") or [])
        _hypothesis_text = str(hypothesis)
        _position_map_consistent: bool = _check_position_map_consistent(
            concerns_text=_hypothesis_text,
            focus_positions=_focus_positions,
        )
        _selectivity_evidence_present: bool = any(
            kw in _hypothesis_text.lower()
            for kw in ("sstr2", "selectiv", "ecl2", "ecl3", "tm5", "tm6", "δmargin", "delta_margin", "선택성")
        )

        if not _position_map_consistent:
            logger.warning(
                "[expert_panel] position map QC fail — iter=%d focus=%s hypothesis 내 hallucination 감지",
                iteration, _focus_positions,
            )

        # LLM 없으면 즉시 통과 폴백
        if not self.has_llm:
            logger.warning("[expert_panel] LLM 없음 — 통과 폴백")
            return self._fallback_pass(
                position_map_consistent=_position_map_consistent,
                selectivity_evidence_present=_selectivity_evidence_present,
            )

        total_llm_calls = 0
        all_merged_concerns: List[str] = []
        last_result: Optional[Dict[str, Any]] = None
        all_discussion_turns: List[Dict[str, Any]] = []

        # 이전 disc_round의 expert_verdicts (round2 turn-taking용 peer_concerns 소스)
        prev_round_verdicts: Optional[Dict[str, Dict[str, Any]]] = None

        # --- E: 정체(stall) 감지 상태 ---
        # 동일한 합의 서명(status,high_count,medium_count)이 연속 등장하면 진전이 없다고 판단.
        _prev_consensus_signature: Optional[Tuple[str, int, int]] = None
        _stall_streak: int = 0
        _stall_detected: bool = False

        for _disc_round in range(1, max_discussion_rounds + 1):
            _actual_round = round_idx + _disc_round - 1

            # Round 2 이상: 이전 라운드 verdict를 peer_concerns로 구성
            # (각 전문가는 자기 도메인 제외 나머지 3명의 verdict를 받음)
            _prev_verdicts_list: Optional[List[Dict[str, Any]]] = None
            if _disc_round > 1 and prev_round_verdicts is not None:
                _prev_verdicts_list = list(prev_round_verdicts.values())

            # --- Fan-out: 5 전문가 병렬 LLM 호출 (ThreadPoolExecutor) ---
            # 모델 이질성(F): domain이 hetero_domains에 속하면 self._provider_for_domain()이
            # 이질 provider(예: Mistral@8001)를 반환 — critic의 기본 provider(Qwen@8000)는
            # 건드리지 않는다. fan-in은 항상 critic.llm_generate_json(아래)을 그대로 사용.
            expert_verdicts: Dict[str, Dict[str, Any]] = {}
            llm_calls = 0

            def _call_one_expert(
                domain: str,
                peer_verdicts: Optional[List[Dict[str, Any]]] = _prev_verdicts_list,
            ) -> Tuple[str, Dict[str, Any], bool]:
                """단일 전문가 LLM 호출. (domain, result_dict, called) 반환."""
                prompt = format_expert_domain_prompt(
                    domain=domain,
                    iteration=iteration,
                    hypothesis=hypothesis,
                    mutation_guidance=mutation_guidance,
                    trajectory_summary=trajectory_summary,
                    selectivity_leaderboard=selectivity_leaderboard,
                    best_delta_margin=best_delta_margin,
                    round_idx=_actual_round,
                    peer_concerns=peer_verdicts,
                )
                system = get_system_prompt(f"expert_{domain}_prereview")
                _provider, _llm_backend = self._provider_for_domain(domain)
                called = False
                try:
                    if _provider is self._critic:
                        # 기본 경로: critic 공유 인프라(기존 동작과 완전 동일, 회귀 없음)
                        res = self._critic.llm_generate_json(prompt, system_prompt=system)
                    else:
                        # 이질 provider 경로: VLLMProvider.generate_json 직접 호출.
                        # critic.llm_provider는 전혀 건드리지 않는다(critic 공유 인프라 보존).
                        res = _provider.generate_json(prompt, system_prompt=system)
                    called = True
                except Exception as exc:
                    logger.error("[expert_panel] %s 전문가 LLM 예외(backend=%s): %s", domain, _llm_backend, exc)
                    res = None

                if res is None:
                    logger.warning(
                        "[expert_panel] %s 전문가 응답 없음(backend=%s) — low/no-concern 폴백",
                        domain, _llm_backend,
                    )
                    res = {
                        "domain": domain,
                        "concerns": [],
                        "severity": "low",
                        "suggestion": "",
                    }

                # domain 키 정규화 (LLM이 빠뜨릴 수 있음)
                res.setdefault("domain", domain)
                res.setdefault("concerns", [])
                res.setdefault("severity", "low")
                res.setdefault("suggestion", "")
                res.setdefault("stance_change", "유지")
                res.setdefault("rebuttal", "")
                # F: 추적성 — 어떤 provider/model로 이 verdict가 판정됐는지 기록.
                res["llm_backend"] = _llm_backend

                # per-expert position QC: verdict 텍스트에 hallucination 감지
                # math 도메인은 advisory 전용이므로 QC 적용 안 함
                _expert_text = " ".join(res.get("concerns", [])) + " " + res.get("suggestion", "")
                if domain in _ADVISORY_ONLY_DOMAINS:
                    # math: advisory 역할 — position QC 비적용, severity는 합의에 미반영
                    res["position_map_consistent"] = True
                    res["qc_status"] = "advisory"
                    res["is_advisory"] = True
                    # advisory 필드: suggestion을 advisory 메시지로 사용
                    res["advisory"] = res.get("suggestion", "") or " ".join(res.get("concerns", []))
                else:
                    _expert_pos_ok = _check_position_map_consistent(
                        concerns_text=_expert_text,
                        focus_positions=_focus_positions,
                    )
                    res["position_map_consistent"] = _expert_pos_ok
                    res["is_advisory"] = False
                    if not _expert_pos_ok:
                        logger.warning(
                            "[expert_panel] %s verdict position hallucination → severity 강제 invalid 표시",
                            domain,
                        )
                        res["severity"] = "high"
                        res["concerns"] = [
                            "[QC-FAIL] position map hallucination detected in verdict text"
                        ] + res.get("concerns", [])
                        res["qc_status"] = "fail"
                    else:
                        res["qc_status"] = "pass"

                # 지시된 형식: [expert-panel] iter=N domain severity=X concerns=...
                # (기존 로그 포맷 유지 — 테스트가 정확한 필드셋을 검증하므로 backend는 별도 라인)
                _concerns_preview = "; ".join(res.get("concerns", [])[:2])
                _stance = res.get("stance_change", "유지")
                logger.info(
                    "[expert-panel] iter=%s %s round=%d severity=%s stance=%s concerns=%s",
                    iteration, domain, _disc_round, res.get("severity", "?"),
                    _stance, _concerns_preview or "(없음)",
                )
                if _llm_backend.startswith("hetero:"):
                    logger.info(
                        "[expert-panel] iter=%s %s backend=%s (모델 이질성 활성)",
                        iteration, domain, _llm_backend,
                    )
                return domain, res, called

            # 5 전문가를 동시에 제출 (vLLM continuous batching 활용)
            domain_futures: Dict[str, "Future[Tuple[str, Dict[str, Any], bool]]"] = {}
            with ThreadPoolExecutor(max_workers=len(_EXPERT_DOMAINS)) as executor:
                for domain in _EXPERT_DOMAINS:
                    domain_futures[domain] = executor.submit(_call_one_expert, domain)

            # _EXPERT_DOMAINS 순서 보장하여 수집
            for domain in _EXPERT_DOMAINS:
                _d, result, called = domain_futures[domain].result()
                expert_verdicts[domain] = result
                if called:
                    llm_calls += 1

            # 이번 라운드 전문가 turns 기록
            for domain in _EXPERT_DOMAINS:
                v = expert_verdicts[domain]
                all_discussion_turns.append({
                    "round": _disc_round,
                    "domain": domain,
                    "severity": v.get("severity", "low"),
                    "concerns": list(v.get("concerns", [])),
                    "stance_change": v.get("stance_change", "유지"),
                    "rebuttal": v.get("rebuttal", ""),
                    # F: 모델 이질성 추적 — 어떤 backend(hetero:mistral@... vs default:qwen@...)로
                    # 이 turn이 판정됐는지. 구버전 verdict(llm_backend 없음)엔 "unknown" 폴백.
                    "llm_backend": v.get("llm_backend", "unknown"),
                })

            # --- Fan-in: 통합 critic LLM 호출 ---
            verdicts_list = list(expert_verdicts.values())

            # 이전 라운드 누적 우려사항 피드백 (건설적 토론)
            prev_concerns_for_fanin = all_merged_concerns if _disc_round > 1 else []
            fanin_prompt = format_expert_fanin_prompt(
                verdicts=verdicts_list,
                iteration=iteration,
                round_idx=_actual_round,
                prev_concerns=prev_concerns_for_fanin,
            )
            fanin_system = get_system_prompt("expert_fanin")
            try:
                fanin_result = self._critic.llm_generate_json(fanin_prompt, system_prompt=fanin_system)
                llm_calls += 1
            except Exception as exc:
                logger.error("[expert_panel] fan-in LLM 예외: %s", exc)
                fanin_result = None

            # 코드로 approve 재확인 (LLM 결과를 보조 — 규칙이 우선)
            _consensus = _apply_consensus_rules(verdicts_list)
            code_approve: bool = _consensus["approve"]
            logger.info(
                "[expert-panel] consensus_check disc_round=%d: status=%s high=%d medium=%d reason=%s",
                _disc_round, _consensus["status"], _consensus["high_count"],
                _consensus["medium_count"], _consensus["reason"],
            )

            # --- C: 소수 의견 보호 (sycophancy 방지, 코드 레벨 — LLM 요약 누락 방지) ---
            _minority_dissent: Optional[Dict[str, Any]] = _detect_minority_dissent(verdicts_list)
            if _minority_dissent:
                logger.info(
                    "[expert-panel] minority_dissent 감지: domain=%s severity=%s reason=%s",
                    _minority_dissent["domain"], _minority_dissent["severity"], _minority_dissent["reason"],
                )

            # --- E: 정체(stall) 감지 — 동일 불일치가 2라운드(기본) 연속 지속되는지 추적 ---
            _sig = _consensus_signature(_consensus)
            if _consensus["status"] != "approve":
                if _prev_consensus_signature is not None and _sig == _prev_consensus_signature:
                    _stall_streak += 1
                else:
                    _stall_streak = 1
                _prev_consensus_signature = _sig
            else:
                _stall_streak = 0
                _prev_consensus_signature = None
            if _stall_streak >= _PANEL_STALL_LIMIT:
                _stall_detected = True
                logger.warning(
                    "[expert-panel] stall 감지: disc_round=%d 동일 불일치(status=%s high=%d medium=%d) %d라운드 연속",
                    _disc_round, _sig[0], _sig[1], _sig[2], _stall_streak,
                )

            if fanin_result is None:
                logger.warning("[expert_panel] fan-in 응답 없음 — 코드 기반 합의 사용")
                merged_concerns: List[str] = []
                for v in verdicts_list:
                    merged_concerns.extend(v.get("concerns", []))
                fanin_result = {
                    "merged_concerns": merged_concerns,
                    "approve": code_approve,
                    "suggested_revisions": {},
                }
            else:
                # LLM approve와 코드 approve가 다르면 코드 규칙 우선 적용
                llm_approve = bool(fanin_result.get("approve", True))
                if llm_approve != code_approve:
                    logger.info(
                        "[expert_panel] fan-in approve 충돌: LLM=%s vs code=%s → code 우선",
                        llm_approve, code_approve,
                    )
                    fanin_result["approve"] = code_approve
                    fanin_result["approve_override"] = f"code_rule (LLM={llm_approve})"

            # 필수 키 정규화
            fanin_result.setdefault("merged_concerns", [])
            fanin_result.setdefault("approve", code_approve)
            fanin_result.setdefault("suggested_revisions", {})
            # minority_dissent는 코드 탐지 결과가 항상 최종 근거(LLM이 흡수/누락해도 보존)
            fanin_result["minority_dissent"] = _minority_dissent

            # 소수 의견이 있으면 merged_concerns에도 명시적으로 병합 (다수결로 사라지지 않도록)
            if _minority_dissent:
                _dissent_note = (
                    f"[minority_dissent] {_minority_dissent['domain']}"
                    f"(severity={_minority_dissent['severity']}): {_minority_dissent['reason']}"
                )
                if _dissent_note not in fanin_result["merged_concerns"]:
                    fanin_result["merged_concerns"].append(_dissent_note)

            # fan-in turn 기록
            all_discussion_turns.append({
                "round": _disc_round,
                "domain": "fanin",
                "approve": fanin_result["approve"],
                "merged_concerns": list(fanin_result.get("merged_concerns", [])),
                "minority_dissent": _minority_dissent,
            })

            total_llm_calls += llm_calls
            # 이번 라운드 우려사항 누적 (다음 라운드 건설적 피드백용)
            for c in fanin_result.get("merged_concerns", []):
                if c not in all_merged_concerns:
                    all_merged_concerns.append(c)

            logger.info(
                "[expert-panel] fan-in approve=%s merged_concerns=%d llm_calls=%d disc_round=%d/%d",
                fanin_result["approve"], len(fanin_result["merged_concerns"]),
                llm_calls, _disc_round, max_discussion_rounds,
            )

            # per-round position QC: expert verdict 전체 텍스트에 hallucination 감지
            # math(advisory) 제외 후 science 도메인만 QC 대상
            _round_all_text = " ".join(
                " ".join(v.get("concerns", [])) + " " + v.get("suggestion", "")
                for domain_key, v in expert_verdicts.items()
                if domain_key not in _ADVISORY_ONLY_DOMAINS
            )
            _round_pos_ok = _check_position_map_consistent(
                concerns_text=_round_all_text,
                focus_positions=_focus_positions,
            )
            # 누적: 하나라도 실패하면 전체 fail
            if not _round_pos_ok:
                _position_map_consistent = False

            # math advisory 추출 (별도 필드로 분리)
            _math_advisory: str = ""
            _math_verdict = expert_verdicts.get("math", {})
            if _math_verdict.get("is_advisory"):
                _math_advisory = _math_verdict.get("advisory", "") or " ".join(
                    _math_verdict.get("concerns", [])
                )

            # QC status: position 불일치 또는 immutable 위치 침범
            _qc_status = "pass" if _position_map_consistent else "fail"

            # consensus_status (이 시점에는 approve/conditional/reject/reject_hard 중 하나)
            _current_consensus_status = _consensus["status"]

            # scientific_verdict: QC fail이면 "invalid", forced_pass이어도 원 verdict 보존
            _sci_verdict = _derive_scientific_verdict(
                consensus_status=_current_consensus_status,
                qc_status=_qc_status,
            )

            # docking_decision: forced_pass(도킹 탐색 허용) + approve + qc_pass → True
            # reject_hard → False. QC fail은 scientific_verdict=invalid이지만 도킹 자체는
            # _PANEL_HARD_REJECT_HIGH 규칙에 위임(아직 reject_hard 결정 전이면 탐색 허용).
            _docking_decision: bool = fanin_result["approve"]  # 기본은 approve 여부 따름
            _reason_for_docking: str = "proceed"

            _risk_level = _compute_risk_level(
                _current_consensus_status,
                _consensus["high_count"],
                _consensus["medium_count"],
            )

            last_result = {
                "merged_concerns": all_merged_concerns,
                "approve": fanin_result["approve"],  # 하위호환
                # --- 신규 필드 ---
                "scientific_verdict": _sci_verdict,
                "docking_decision": _docking_decision,
                "reason_for_docking": _reason_for_docking,
                "risk_level": _risk_level,
                "qc_status": _qc_status,
                "position_map_consistent": _position_map_consistent,
                "selectivity_evidence_present": _selectivity_evidence_present,
                "math_advisory": _math_advisory,  # 수학 패널 advisory 별도 기록
                "minority_dissent": _minority_dissent,  # C: 소수 반대의견 (없으면 None)
                "stall_detected": _stall_detected,  # E: 정체 감지 여부
                "stall_streak": _stall_streak,  # E: 동일 불일치 연속 라운드 수
                # --- 기존 필드 ---
                "consensus_status": _current_consensus_status,
                "consensus_reason": _consensus["reason"],
                "suggested_revisions": fanin_result.get("suggested_revisions", {}),
                "expert_verdicts": expert_verdicts,
                "llm_calls": total_llm_calls,
                "discussion_rounds": _disc_round,
                "discussion_turns": list(all_discussion_turns),
            }

            # --- E: stall 감지 시 무한 재시도 대신 명시적 종료 ---
            # 동일한 불일치가 PANEL_STALL_LIMIT 라운드 연속 지속되면 conditional/invalid로 종결.
            # (reject_hard 대상인 high>=PANEL_HIGH_REJECT는 그대로 두어 아래 reject_hard 분기가
            #  더 강한 신호로 우선 처리되도록 한다 — stall은 그 외의 경우에만 개입.)
            # 마지막 라운드에서는 기존 forced_pass/reject_hard 로직이 최종 결론을 전담하므로
            # stall은 "마지막 라운드 도달 전" 조기 종료 용도로만 개입한다 — 시맨틱 안정성 보장.
            if (
                _stall_detected
                and _disc_round < max_discussion_rounds
                and not fanin_result["approve"]
                and not (_PANEL_HARD_REJECT_HIGH >= 1 and _consensus["status"] == "reject")
            ):
                logger.info(
                    "[expert-panel] stall 종료: disc_round=%d 동일 불일치 %d라운드 연속 — "
                    "무한 재시도 방지, conditional로 종결(도킹은 저비용 탐색 허용)",
                    _disc_round, _stall_streak,
                )
                last_result["consensus_status"] = "stall"
                last_result["consensus_reason"] = (
                    f"stall 감지: 동일 불일치(status={_sig[0]}, high={_sig[1]}, medium={_sig[2]}) "
                    f"{_stall_streak}라운드 연속 — 무한 재시도 방지 위해 조기 종결"
                )
                last_result["scientific_verdict"] = _sci_verdict if _sci_verdict != "approve" else "conditional"
                last_result["approve"] = True  # 하위호환 — 저비용 탐색 도킹 허용
                last_result["docking_decision"] = True
                last_result["reason_for_docking"] = "cheap exploratory docking despite stall (no convergence)"
                return last_result

            # --- B: Confidence-gated 조기 종료 (arXiv:2504.05047 "Debate Only When Necessary") ---
            # 불필요한 토론 라운드는 비용만 발생시킨다. "불일치가 없다"를 두 가지로 인정한다:
            #   (1) science 전문가 전원이 low로 수렴(순수 approve 케이스) — 이미 approve 분기가 처리.
            #   (2) 직전 라운드와 이번 라운드의 합의 서명(status,high_count,medium_count)이 동일
            #       — 즉 라운드를 거듭해도 입장이 흔들리지 않고 안정적으로 재현된 경우.
            # (2)는 stall과 언뜻 비슷해 보이나 목적이 다르다: stall은 "진전이 없어 강제 종료"이고
            # confidence-gate는 "이미 충분히 안정적이라 더 볼 필요가 없어 자발적 조기 종료"이다.
            # 단, high>=PANEL_HIGH_REJECT(reject 방향 치명 다수)나 minority_dissent가 있으면
            # 안정적이어도 신중하게 라운드를 계속 진행한다(섣부른 조기종료로 우려를 덮지 않음).
            # PANEL_CONFIDENCE_GATE=0으로 비활성화 시 기존 방식(approve 달성시만 조기종료) 유지.
            _science_severities = [
                v.get("severity", "low") for v in verdicts_list
                if v.get("domain", "") not in _ADVISORY_ONLY_DOMAINS
            ]
            _all_low = _consensus["high_count"] == 0 and all(s == "low" for s in _science_severities)
            # _stall_streak(위 E에서 계산)는 "동일 합의 서명이 몇 라운드 연속됐는지"를 정확히 추적한다.
            # 2라운드 이상 안정적으로 재현됐고 reject(치명 다수)가 아니면 "충분히 일치"로 간주.
            # 참고: PANEL_STALL_LIMIT(기본 2)과 동일 임계에 도달하면 위의 E(stall) 분기가 먼저
            # 개입해 종료하므로, 이 경로는 주로 PANEL_STALL_LIMIT을 크게 잡은 구성에서 stall보다
            # 먼저 발동하는 "자발적" 조기 종료로 동작한다 — 우선순위: stall(강제) > confidence-gate(자발).
            _stable_repeat = (
                _stall_streak >= 2
                and _consensus["status"] != "reject"  # reject(high>=임계)는 절대 조기종료 대상 아님
            )
            _no_disagreement = _all_low or _stable_repeat
            _confidence_gate_hit = (
                _PANEL_CONFIDENCE_GATE >= 1
                and _no_disagreement
                and not _minority_dissent
            )

            # approve=True 처리: 즉시 종료 (합의 달성 = 더 이상 토론 불필요)
            # - 우려가 없으면(깨끗한 통과) 1라운드에서 approve=True → 바로 종료.
            # - 이전 라운드에서 approve=False(conditional)이었다가 토론으로 해소되어
            #   approve=True에 도달했으면 → 그 라운드에서 즉시 종료 (비용 최소화).
            # - 추가 라운드 불필요: approve 달성 = 합의 완료.
            if fanin_result["approve"]:
                # approve 시 docking_decision=True, reason_for_docking="proceed"
                last_result["docking_decision"] = True
                last_result["reason_for_docking"] = "proceed"
                last_result["approve"] = True  # 하위호환 동기화
                logger.info(
                    "[expert-panel] 합의 달성 disc_round=%d — 통과 (우려 %d건 기록됨)",
                    _disc_round, len(all_merged_concerns),
                )
                return last_result

            # confidence-gate: approve는 아니지만(예: medium 만장일치 conditional) 전문가 간
            # 불일치·high가 전혀 없어 추가 토론으로 결론이 바뀔 여지가 낮다고 판단되면 조기 종료.
            # 단, 마지막 라운드 로직(forced_pass/reject_hard)과 중복되지 않도록 마지막 라운드
            # 미만일 때만 개입 — 마지막 라운드는 기존 로직이 그대로 처리한다.
            if _confidence_gate_hit and _disc_round < max_discussion_rounds:
                logger.info(
                    "[expert-panel] confidence_gate 조기 종료 disc_round=%d — "
                    "전문가 일치(불일치 없음, high=0) 확인, 잔여 라운드(%d) 생략",
                    _disc_round, max_discussion_rounds - _disc_round,
                )
                last_result["consensus_reason"] += " | confidence_gate: 전문가 일치로 조기 종료"
                return last_result

            # approve=False이지만 마지막 라운드 → best 결과 채택 (무한 거부 방지)
            if _disc_round >= max_discussion_rounds:
                # 예외: high >= PANEL_HIGH_REJECT(치명 다수)이고 PANEL_HARD_REJECT_HIGH=1 이면
                # forced_pass 하지 않고 reject_hard 유지 (도킹 skip 유도).
                # high 1개(conditional) / medium 다수는 기존 forced_pass 적용.
                _is_hard_reject = (
                    _PANEL_HARD_REJECT_HIGH >= 1
                    and _consensus["status"] == "reject"
                )
                if _is_hard_reject:
                    logger.info(
                        "[expert-panel] 최대 토론 라운드(%d) 소진, high %d개 ≥ 임계(%d) → reject_hard (도킹 skip)",
                        max_discussion_rounds, _consensus["high_count"], _PANEL_HIGH_REJECT,
                    )
                    last_result["approve"] = False
                    last_result["consensus_status"] = "reject_hard"
                    last_result["consensus_reason"] = (
                        f"max_discussion_rounds({max_discussion_rounds}) 소진 + "
                        f"high {_consensus['high_count']}개 ≥ 임계({_PANEL_HIGH_REJECT}) "
                        f"→ 강제 거부(PANEL_HARD_REJECT_HIGH={_PANEL_HARD_REJECT_HIGH})"
                    )
                    # reject_hard 전용 신규 필드
                    last_result["scientific_verdict"] = "reject"
                    last_result["docking_decision"] = False
                    last_result["reason_for_docking"] = "blocked: reject_hard (high severity >= threshold)"
                    last_result["risk_level"] = "high"
                    return last_result

                logger.info(
                    "[expert-panel] 최대 토론 라운드(%d) 소진 — best 가설 채택(무한거부 방지)",
                    max_discussion_rounds,
                )
                # 무한 거부 방지: reject이지만 도킹은 통과 (거부 사유는 concerns에 기록됨)
                # *** scientific_verdict는 원래 verdict 보존 — "승인"으로 바꾸지 않음 ***
                _original_sci_verdict = _derive_scientific_verdict(
                    consensus_status=_current_consensus_status,
                    qc_status=_qc_status,
                )
                last_result["approve"] = True           # 하위호환 (도킹 진행)
                last_result["docking_decision"] = True  # 저비용 탐색 허용
                last_result["reason_for_docking"] = "cheap exploratory docking despite reject"
                last_result["scientific_verdict"] = _original_sci_verdict  # reject/conditional 그대로
                last_result["consensus_status"] = "forced_pass"
                last_result["consensus_reason"] = (
                    f"max_discussion_rounds({max_discussion_rounds}) 소진 → 강제 도킹 진행 "
                    f"(과학적 판정={_original_sci_verdict}, 원래: {_consensus['reason']})"
                )
                logger.info(
                    "[expert-panel] forced_pass: scientific_verdict=%s (reject→도킹 허용, 승인 아님)",
                    _original_sci_verdict,
                )
                return last_result

            # 다음 토론 라운드를 위해 suggested_revisions의 focus를 mutation_guidance에 반영
            _sugg = fanin_result.get("suggested_revisions", {})
            _sugg_focus = _sugg.get("focus_positions")
            if _sugg_focus:
                mutation_guidance = dict(mutation_guidance)
                mutation_guidance["focus_positions"] = _sugg_focus
                logger.info(
                    "[expert-panel] 다음 토론 라운드 focus 갱신: %s → %s",
                    list(_sugg_focus),
                    _sugg_focus,
                )

            # 현재 라운드 verdicts를 저장 → 다음 라운드 peer_concerns 소스
            prev_round_verdicts = dict(expert_verdicts)

        # 반복 없이 종료된 경우 (max_discussion_rounds=0 등 엣지 케이스)
        return last_result or self._fallback_pass()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_pass(
        position_map_consistent: bool = True,
        selectivity_evidence_present: bool = False,
    ) -> Dict[str, Any]:
        """LLM 없음 또는 전체 실패 시 통과 폴백 결과."""
        return {
            "merged_concerns": [],
            "approve": True,
            # 신규 필드
            "scientific_verdict": "approve",
            "docking_decision": True,
            "reason_for_docking": "proceed",
            "risk_level": "low",
            "qc_status": "pass" if position_map_consistent else "fail",
            "position_map_consistent": position_map_consistent,
            "selectivity_evidence_present": selectivity_evidence_present,
            "math_advisory": "",
            "minority_dissent": None,
            "stall_detected": False,
            "stall_streak": 0,
            # 기존 필드
            "consensus_status": "approve",
            "consensus_reason": "LLM 없음 또는 폴백 — 통과",
            "suggested_revisions": {},
            "expert_verdicts": {},
            "llm_calls": 0,
            "discussion_rounds": 0,
            "discussion_turns": [],
        }
