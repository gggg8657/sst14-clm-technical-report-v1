"""
agent_output_validator.py
=========================
에이전트 출력 JSON 스키마 검증 레이어

각 에이전트(planner, qc_ranker, diversity_manager, critic, reporter)의
execute() 반환값이 기대 형식을 충족하는지 검사한다.

외부 의존성 없음 - isinstance()와 dict 기반 검증만 사용.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# 에이전트별 예상 스키마 정의
# ---------------------------------------------------------------------------

# 스키마 형식:
#   required_keys: 반드시 존재해야 하는 최상위 키 목록
#   type_checks:   키 -> 기대 타입 (tuple 이면 isinstance() 다중 허용)
#   nested:        중첩 객체에 대한 추가 검증 규칙
#                  각 항목은 (부모_키, 필수_속성_리스트) 튜플

_AGENT_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "planner": {
        # planner 결과 필수 키
        "required_keys": ["status", "plan", "run_id"],
        "type_checks": {
            "status": str,
            "run_id": str,
            # plan 은 ExperimentPlan 객체이므로 object(Any) 허용
            "plan": object,
        },
        # plan 객체가 갖춰야 할 속성 목록
        "plan_attrs": ["run_id", "iteration", "parameters", "hypothesis"],
    },
    "qc_ranker": {
        # qc_ranker 결과 필수 키
        "required_keys": ["status", "rank_table", "qc_report", "top_candidates"],
        "type_checks": {
            "status": str,
            "top_candidates": list,
            # rank_table, qc_report 는 dataclass 객체
            "rank_table": object,
            "qc_report": object,
        },
    },
    "diversity_manager": {
        # diversity_manager 결과 필수 키
        "required_keys": ["status", "diverse_candidates", "clusters", "redundant_ids"],
        "type_checks": {
            "status": str,
            "diverse_candidates": list,
            "clusters": list,
            "redundant_ids": list,
        },
    },
    "critic": {
        # critic 결과 필수 키
        "required_keys": ["status", "critic_analysis"],
        "type_checks": {
            "status": str,
            "critic_analysis": object,
        },
        # critic_analysis 객체가 갖춰야 할 속성 목록
        "analysis_attrs": ["iteration", "failure_summary", "proposed_changes", "hypothesis"],
    },
    "reporter": {
        # reporter 결과 필수 키
        "required_keys": ["status", "report_paths"],
        "type_checks": {
            "status": str,
            "report_paths": dict,
        },
    },
}


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------


def validate_agent_output(
    agent_name: str,
    result: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """에이전트 출력 result 를 스키마에 맞게 검증한다.

    Args:
        agent_name: 에이전트 논리 이름 (``planner``, ``critic`` 등).
        result:     agent.execute() 가 반환한 dict.

    Returns:
        ``(True, [])`` - 검증 통과
        ``(False, [에러_메시지, ...])`` - 검증 실패 (예외 미발생)
    """
    # 알 수 없는 에이전트는 경고 없이 통과
    schema = _AGENT_SCHEMAS.get(agent_name)
    if schema is None:
        return True, []

    errors: List[str] = []

    # result 가 dict 가 아니면 즉시 실패
    if not isinstance(result, dict):
        return False, [
            f"[{agent_name}] result 는 dict 여야 하지만 {type(result).__name__} 가 반환됐습니다."
        ]

    # 1) 필수 키 존재 여부 검사
    errors.extend(_check_required_keys(agent_name, result, schema))

    # 2) 타입 검사 (키가 존재하는 경우에만)
    errors.extend(_check_types(agent_name, result, schema))

    # 3) 중첩 객체 속성 검사 (planner.plan, critic.critic_analysis)
    errors.extend(_check_nested_attrs(agent_name, result, schema))

    ok = len(errors) == 0
    return ok, errors


# ---------------------------------------------------------------------------
# 내부 검증 헬퍼
# ---------------------------------------------------------------------------


def _check_required_keys(
    agent_name: str,
    result: Dict[str, Any],
    schema: Dict[str, Any],
) -> List[str]:
    """필수 키가 result 에 모두 존재하는지 확인한다."""
    errors: List[str] = []
    for key in schema.get("required_keys", []):
        if key not in result:
            errors.append(
                f"[{agent_name}] 필수 키 '{key}' 가 출력에 없습니다."
            )
    return errors


def _check_types(
    agent_name: str,
    result: Dict[str, Any],
    schema: Dict[str, Any],
) -> List[str]:
    """result 의 각 키가 기대 타입과 일치하는지 확인한다."""
    errors: List[str] = []
    type_checks: Dict[str, Any] = schema.get("type_checks", {})

    for key, expected_type in type_checks.items():
        if key not in result:
            # 필수 키 검사에서 이미 보고했으므로 중복 방지
            continue
        value = result[key]
        if value is None:
            # None 은 "없음" 상태로 간주 - 별도 경고 없이 통과
            continue
        if expected_type is object:
            # object 는 "임의 타입 허용" 표시이므로 통과
            continue
        if not isinstance(value, expected_type):
            errors.append(
                f"[{agent_name}] 키 '{key}' 타입 불일치: "
                f"기대={getattr(expected_type, '__name__', str(expected_type))}, "
                f"실제={type(value).__name__}"
            )
    return errors


def _check_nested_attrs(
    agent_name: str,
    result: Dict[str, Any],
    schema: Dict[str, Any],
) -> List[str]:
    """중첩 객체(plan, critic_analysis)의 필수 속성을 확인한다."""
    errors: List[str] = []

    # planner.plan 속성 검증
    if agent_name == "planner" and "plan_attrs" in schema:
        plan = result.get("plan")
        if plan is not None:
            errors.extend(
                _check_object_attrs("planner", "plan", plan, schema["plan_attrs"])
            )

    # critic.critic_analysis 속성 검증
    if agent_name == "critic" and "analysis_attrs" in schema:
        analysis = result.get("critic_analysis")
        if analysis is not None:
            errors.extend(
                _check_object_attrs(
                    "critic", "critic_analysis", analysis, schema["analysis_attrs"]
                )
            )

    return errors


def _check_object_attrs(
    agent_name: str,
    field_name: str,
    obj: Any,
    required_attrs: List[str],
) -> List[str]:
    """obj 가 required_attrs 에 나열된 속성을 모두 갖고 있는지 확인한다."""
    errors: List[str] = []
    for attr in required_attrs:
        # dict 와 dataclass 모두 지원
        if isinstance(obj, dict):
            if attr not in obj:
                errors.append(
                    f"[{agent_name}] '{field_name}.{attr}' 속성이 없습니다."
                )
        elif not hasattr(obj, attr):
            errors.append(
                f"[{agent_name}] '{field_name}.{attr}' 속성이 없습니다."
            )
    return errors
