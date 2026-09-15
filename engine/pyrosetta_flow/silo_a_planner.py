"""silo_a_planner.py — Silo A 전용 피드백 루프 플래너.

Silo A de novo 발굴 엔진의 epoch 간 적응 로직.
직전까지의 리더보드 요약을 분석해 다음 epoch 생성 파라미터를 조정한다.

설계 원칙:
- Silo B(runner.py·continuous.py·pyrosetta_flow/ 타 모듈)와 완전 분리.
- LLM 경로 (vLLM GPU2 호출): 리더보드 요약 → JSON 파라미터 제안.
- 규칙 기반 fallback: LLM 불가 시 patience 정체 탈출 규칙 적용.
- 환각 0 원칙: 적용된 파라미터만 기록. 미적용 제안은 fallback_reason으로 명시.
- provenance 기록: runs/silo_a_flow/silo_a_planner_e{NN}.json (epoch별).

분리 경계 (MUST):
- pyrosetta_flow/runner.py·continuous.py 미import.
- runs/pyrosetta_flow/ 경로 미접근.
- 신규/수정: silo_a_flow.py·run_silo_a_discovery.py·silo_a_planner.py에 한정.
"""
from __future__ import annotations

import json
import logging
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 타입 / 상수
# ---------------------------------------------------------------------------

_DEFAULT_VLLM_URL = "http://localhost:8000"
# L3-a 수정: vLLM 8000 서빙명과 일치시킴 (qwen3-32b → 72B로 라우팅됨).
# "Qwen/Qwen3-32B"는 HuggingFace 경로 형식 → 8000에서 404. "qwen3-32b"는 서빙명 → 200.
# 검증: curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/v1/chat/completions
#        -H "Authorization: Bearer EMPTY" -d '{"model":"qwen3-32b","messages":[...]}'  → 200
_DEFAULT_LLM_MODEL = "qwen3-32b"
_LLM_TIMEOUT = 60        # 초
_LLM_MAX_RETRIES = 2
_LLM_RETRY_BACKOFF = 1.5

# 바인더 길이 범위 (contig 내 바인더 길이 부분)
_BINDER_LEN_MIN = 10
_BINDER_LEN_MAX = 20

# diffusion_steps 범위
_DIFFUSION_STEPS_MIN = 30
_DIFFUSION_STEPS_MAX = 200

# 정체 단계별 diffusion_steps 사다리 (stagnation_count 임계 → 강제 steps 값)
# 누적 정체가 깊을수록 더 큰 탐색 반경을 강제 적용한다.
# 임계는 patience_epochs 배수로 동작: 1×→50, 2×→100, 4×→150, 8×→200
_STAGNATION_STEPS_LADDER: List[Tuple[int, int]] = [
    # (patience 배수 임계, 강제 diffusion_steps)
    (8, 200),
    (4, 150),
    (2, 100),
    (1, 50),
]

# L3-b 수정: hotspot 풀을 SSTR2-고유 결합 포켓 잔기로 한정.
# 임의 잔기(B120/B220/B300 등) 셔플 → 일관 탐색 불가 문제 해결.
# 출처: SSTR2 결정구조 기반 포켓 잔기 (chain B 기준):
#   ECL2: B192, B193, B195, B197
#   TM5:  B205, B208, B209, B212
#   TM6:  B272, B273, B276, B279
#   ECL3: B284, B286
# 이 풀 내에서만 hotspot 선택/교체 → de novo 바인더가 SSTR2-고유 영역만 타겟
_HOTSPOT_POOL = [
    # ECL2 (extracellular loop 2) — SSTR2 선택성 핵심 영역
    "B192", "B193", "B195", "B197",
    # TM5 (transmembrane helix 5) — 펩타이드 주요 접촉 잔기
    "B205", "B208", "B209", "B212",
    # TM6 (transmembrane helix 6) — 활성화 관련 포켓
    "B272", "B273", "B276", "B279",
    # ECL3 (extracellular loop 3) — SST14 결합 확장 영역
    "B284", "B286",
]

# 기본 핫스팟: ECL2+TM5 대표값 (--hotspot-res CLI로 오버라이드 가능)
_DEFAULT_HOTSPOT = ["B192", "B197", "B205", "B209", "B272", "B284"]


# ---------------------------------------------------------------------------
# 플래너 결정 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class SiloAPlannerDecision:
    """한 epoch에 대한 플래너 결정."""
    epoch: int
    hypothesis: str                          # 조정 근거 (자연어)
    adjustment_type: str                     # "llm_guided" | "rule_stagnation" | "rule_initial" | "rule_noop"
    applied_params: Dict[str, Any]           # 실제 적용된 파라미터 변경사항
    llm_raw_response: Optional[str] = None   # LLM 원문 (None = 규칙 fallback)
    fallback_reason: Optional[str] = None    # LLM 불가 시 fallback 사유
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    leaderboard_summary: Optional[Dict[str, Any]] = None  # 결정 근거 리더보드 요약

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "hypothesis": self.hypothesis,
            "adjustment_type": self.adjustment_type,
            "applied_params": self.applied_params,
            "llm_raw_response": self.llm_raw_response,
            "fallback_reason": self.fallback_reason,
            "created_at": self.created_at,
            "leaderboard_summary": self.leaderboard_summary,
        }


# ---------------------------------------------------------------------------
# 리더보드 요약 헬퍼
# ---------------------------------------------------------------------------

def _summarize_leaderboard(
    leaderboard_path: Path,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Silo A 리더보드를 읽어 플래너 입력용 요약 dict 반환.

    환각 0: 실측값만 포함. 파일 없거나 파싱 실패 시 빈 dict.
    """
    if not leaderboard_path.exists():
        return {"status": "empty", "n_total": 0, "entries": []}

    try:
        data = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("[SiloAPlanner] 리더보드 파싱 실패: %s", exc)
        return {"status": "parse_error", "n_total": 0, "entries": []}

    entries = data.get("entries", [])
    top = entries[:top_k]

    # 요약 통계
    ddgs = [e.get("ddg") for e in entries if e.get("ddg") is not None]
    sel_margins = [e.get("selectivity_margin") for e in entries if e.get("selectivity_margin") is not None]

    # 최근 사용된 파라미터 추출 (extra_scores.iter_dir 등에서 역추적 가능한 것만)
    hotspot_usage: Dict[str, int] = {}
    for e in entries:
        extra = e.get("extra_scores", {}) or {}
        # backbone_pdb에서 epoch 추출 시도
        bb_pdb = extra.get("backbone_pdb", "")
        # 현재 hotspot 직접 저장 안됨 — 통계만

    summary: Dict[str, Any] = {
        "status": "ok",
        "n_total": int(data.get("n_total", 0)),
        "n_unique": len(entries),
        "best_ddg": min(ddgs) if ddgs else None,
        "mean_ddg_top5": (sum(ddgs[:5]) / len(ddgs[:5])) if len(ddgs) >= 1 else None,
        "best_selectivity_margin": max(sel_margins) if sel_margins else None,
        "n_with_ddg": len(ddgs),
        "n_with_selectivity": len(sel_margins),
        "top_entries": [
            {
                "sequence": e.get("sequence", "")[:20],
                "ddg": e.get("ddg"),
                "selectivity_margin": e.get("selectivity_margin"),
                "plddt": e.get("plddt"),
                "plddt_pass": e.get("plddt_pass"),
                "fail_reason": e.get("fail_reason", "")[:80] if e.get("fail_reason") else "",
                # surrogate 필드 (MED/LOW 신뢰 등급 — 참고만)
                "half_life_h": e.get("half_life_h"),      # MED(상대순위) / LOW(절대)
                "admet_score": e.get("admet_score"),       # MED(상대순위)
                "hc50": e.get("hc50"),                    # LOW / L-aa 역변별 AUC=0.146
            }
            for e in top
        ],
    }
    return summary


# ---------------------------------------------------------------------------
# 궤적 요약 헬퍼 (Trajectory Summary) — Silo A 전용
# ---------------------------------------------------------------------------

def _summarize_trajectory(
    experiment_log_path: Path,
    k: int = 6,
) -> str:
    """Silo A experiment_log.jsonl에서 최근 K epoch의 궤적 요약 반환.

    분리 경계: Silo A experiment_log 경로만 허용. runs/pyrosetta_flow/ 접근 금지.
    환각 0 원칙: 실측값만 포함. 없으면 "데이터 없음" 반환.

    Args:
        experiment_log_path: Silo A experiment_log.jsonl 경로.
            반드시 silo_a_flow 경로여야 한다. pyrosetta_flow 접근 금지.
        k: 최근 K epoch 포함 (기본 6, 최대 8).

    Returns:
        "## Silo A 최근 궤적" 섹션 문자열 (압축 개조식, 5~8줄).
    """
    k = min(max(k, 1), 8)

    # 분리 경계 강제: pyrosetta_flow 경로 침범 금지
    path_str = str(experiment_log_path)
    if "pyrosetta_flow" in path_str and "silo_a" not in path_str:
        logger.error("[SiloAPlanner._summarize_trajectory] Silo B 경로 접근 시도 차단: %s", path_str)
        return "## Silo A 최근 궤적\n- [오류] Silo B 경로 접근 금지"

    if not experiment_log_path.exists():
        return "## Silo A 최근 궤적\n- 궤적 데이터 없음 (로그 미존재)"

    # 레코드 로드
    raw_records: List[Dict[str, Any]] = []
    try:
        with experiment_log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except Exception as exc:
        logger.warning("[SiloAPlanner._summarize_trajectory] 로그 읽기 실패: %s", exc)
        return f"## Silo A 최근 궤적\n- 로그 읽기 실패: {exc}"

    if not raw_records:
        return "## Silo A 최근 궤적\n- 궤적 데이터 없음 (빈 로그)"

    # epoch 기준 그룹화
    epoch_map: Dict[int, List[Dict[str, Any]]] = {}
    for rec in raw_records:
        ep = rec.get("epoch")
        if ep is None:
            # epoch 필드 없으면 candidate_id에서 추출 시도
            cid = rec.get("candidate_id", "")
            try:
                # silo_a_e{NN}_... 형식
                parts = cid.split("_")
                ep_part = next((p for p in parts if p.startswith("e") and p[1:].isdigit()), None)
                ep = int(ep_part[1:]) if ep_part else 0
            except Exception:
                ep = 0
        epoch_map.setdefault(int(ep), []).append(rec)

    if not epoch_map:
        return "## Silo A 최근 궤적\n- 궤적 데이터 없음 (epoch 필드 없음)"

    recent_epochs = sorted(epoch_map.keys())[-k:]

    lines: List[str] = ["## Silo A 최근 궤적 (recent trajectory)"]

    prev_best_ddg: Optional[float] = None
    prev_best_margin: Optional[float] = None

    for ep in recent_epochs:
        recs = epoch_map[ep]

        # 성공 레코드 (fail_reason 없고 ddg 유효)
        success_recs = [
            r for r in recs
            if not r.get("fail_reason")
            and r.get("ddg") is not None
            and float(r.get("ddg", 999)) < 500
        ]

        n_total = len(recs)
        n_success = len(success_recs)

        # best ddG
        best_ddg: Optional[float] = None
        best_seq: Optional[str] = None
        if success_recs:
            best_rec = min(success_recs, key=lambda r: float(r.get("ddg", 999)))
            best_ddg = float(best_rec.get("ddg", 999))
            best_seq = best_rec.get("sequence", "")

        # selectivity_margin / delta_margin
        sel_vals = [
            float(r.get("selectivity_margin"))
            for r in recs
            if r.get("selectivity_margin") is not None
        ]
        delta_vals = [
            float(r.get("delta_margin"))
            for r in recs
            if r.get("delta_margin") is not None
        ]
        best_margin: Optional[float] = max(sel_vals) if sel_vals else None
        best_delta: Optional[float] = max(delta_vals) if delta_vals else None

        # pLDDT 통계
        plddt_vals = [
            float(r.get("plddt"))
            for r in recs
            if r.get("plddt") is not None
        ]
        mean_plddt: Optional[float] = (
            sum(plddt_vals) / len(plddt_vals) if plddt_vals else None
        )

        # 개선 여부
        ddg_trend = ""
        if best_ddg is not None and prev_best_ddg is not None:
            delta = best_ddg - prev_best_ddg
            if delta < -0.5:
                ddg_trend = f"↑개선({delta:+.1f})"
            elif delta > 0.5:
                ddg_trend = f"↓악화({delta:+.1f})"
            else:
                ddg_trend = "→정체"
        elif best_ddg is not None:
            ddg_trend = "초기"

        margin_trend = ""
        if best_margin is not None and prev_best_margin is not None:
            dm = best_margin - prev_best_margin
            if dm > 0.1:
                margin_trend = f"sel↑({dm:+.1f})"
            elif dm < -0.1:
                margin_trend = f"sel↓({dm:+.1f})"
            else:
                margin_trend = "sel→"
        elif best_margin is not None:
            margin_trend = "sel초기"

        # extra_scores에서 hotspot, binder_length 추출
        hotspot_hint = ""
        binder_hint = ""
        for rec in recs[:1]:
            extra = rec.get("extra_scores", {}) or {}
            hs = extra.get("hotspot_res")
            if hs and isinstance(hs, list):
                hotspot_hint = f"hs={','.join(str(h) for h in hs[:3])}"
            bpdb = extra.get("backbone_pdb", "")
            # contig 길이: backbone_pdb 경로에서 직접 파악 불가 — 시퀀스 길이로 대체
            if best_seq:
                binder_hint = f"len={len(best_seq)}"

        # 요약 라인
        ddg_str = f"ddG={best_ddg:.1f}" if best_ddg is not None else "ddG=N/A"
        margin_str = (
            f"Δmargin={best_delta:.2f}" if best_delta is not None
            else (f"margin={best_margin:.2f}" if best_margin is not None else "sel=N/A")
        )
        plddt_str = f"pLDDT={mean_plddt:.0f}" if mean_plddt is not None else ""
        trend_str = " ".join(filter(None, [ddg_trend, margin_trend]))
        seq_str = f"best={best_seq[:15]}" if best_seq else ""
        hints = " ".join(filter(None, [hotspot_hint, binder_hint, plddt_str]))

        line = (
            f"- epoch {ep}: {ddg_str} {margin_str} [{trend_str}] "
            f"n={n_success}/{n_total}"
        )
        if seq_str:
            line += f" {seq_str}"
        if hints:
            line += f" ({hints})"
        lines.append(line)

        if best_ddg is not None:
            prev_best_ddg = best_ddg
        if best_margin is not None:
            prev_best_margin = best_margin

    # 패턴 감지
    stagnation_epochs = 0
    prev_ddg_check: Optional[float] = None
    for ep in recent_epochs:
        recs = epoch_map[ep]
        success_recs = [
            r for r in recs
            if not r.get("fail_reason") and r.get("ddg") is not None and float(r.get("ddg", 999)) < 500
        ]
        if success_recs:
            ep_best = min(float(r.get("ddg", 999)) for r in success_recs)
            if prev_ddg_check is not None and abs(ep_best - prev_ddg_check) < 0.5:
                stagnation_epochs += 1
            prev_ddg_check = ep_best

    if stagnation_epochs >= 3:
        lines.append(
            f"- [패턴] 최근 {stagnation_epochs}epoch ddG 정체 — "
            "핫스팟/바인더 길이/diffusion_steps 조정이나 arm 전환 필요."
        )

    sel_missing = sum(
        1 for ep in recent_epochs
        if all(
            r.get("selectivity_margin") is None
            for r in epoch_map.get(ep, [])
        )
    )
    if sel_missing >= max(2, len(recent_epochs) - 1):
        lines.append(
            f"- [패턴] 최근 {sel_missing}epoch 선택성 미측정 — "
            "selectivity_enabled=False 또는 선택성 게이트 미통과."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM 호출 헬퍼 (AG_src/llm/provider.py 스타일 재사용)
# ---------------------------------------------------------------------------

def _http_post_json(
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout: int,
    max_retries: int = _LLM_MAX_RETRIES,
) -> Optional[Dict[str, Any]]:
    """POST JSON 후 응답 dict 반환. 일시적 오류 backoff 재시도."""
    data = json.dumps(payload).encode("utf-8")
    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:
                logger.error("[SiloAPlanner] LLM HTTP %s (영구 오류): %s", e.code, url)
                return None
            last_err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
        except Exception as e:
            logger.error("[SiloAPlanner] LLM 예기치 않은 오류: %s", e)
            return None
        if attempt < max_retries:
            sleep_s = _LLM_RETRY_BACKOFF * (2 ** attempt)
            logger.warning("[SiloAPlanner] LLM 일시적 오류(시도 %d/%d): %s — %.1f초 후 재시도",
                           attempt + 1, max_retries + 1, last_err, sleep_s)
            time.sleep(sleep_s)
    logger.error("[SiloAPlanner] LLM %d회 시도 모두 실패: %s", max_retries + 1, last_err)
    return None


def _call_llm_for_plan(
    summary: Dict[str, Any],
    current_params: Dict[str, Any],
    vllm_url: str,
    model: str,
    timeout: int,
    trajectory_text: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """vLLM에 리더보드 요약 + 현재 파라미터 + 궤적을 주고 다음 epoch 조정 JSON을 요청.

    반환: {
        "hypothesis": str,
        "pocket_rationale": str,       # 어느 포켓을 왜 선택하는지 근거
        "diversity_rationale": str,    # 탐색 다양성 전략 근거
        "hotspot_res": List[str] | null,
        "binder_length_range": [int, int] | null,   # contig 바인더 부분
        "diffusion_steps": int | null,
        "n_backbone": int | null,
        "k_seq_per_backbone": int | null,
        "arm_preference": "rfdiffusion" | "diffpepbuilder" | null,
    }
    LLM 호출 실패 또는 파싱 실패 시 None.
    """
    # L3-b: LLM 프롬프트에 SSTR2 포켓 잔기 명시 — 포켓 외 잔기 제안 억제
    _pocket_residues_str = (
        "ECL2(B192/193/195/197), TM5(B205/208/209/212), "
        "TM6(B272/273/276/279), ECL3(B284/286)"
    )
    system_prompt = (
        "당신은 SSTR2 표적 de novo 펩타이드 바인더 설계 전문 전략가입니다.\n"
        "임무: 리더보드 요약·현재 파라미터·최근 궤적을 분석하고, "
        "다음 epoch 생성 파라미터를 JSON으로 제안하십시오.\n\n"
        "## 추론 구조 (4단계 — 반드시 이 순서로 내부 추론 후 JSON 출력)\n"
        "1. 포켓-선택성 분석: ECL2·TM5·TM6·ECL3 중 어느 포켓 조합이 SSTR2-고유 선택성에 "
        "기여하는지 추론. SSTR1/SSTR3/SSTR5와 구조적으로 가장 차별화된 포켓을 우선하라.\n"
        "   - ECL2(B192/193/195/197): SSTR2 선택성 핵심 — 서브타입 간 서열 다양성 최대.\n"
        "   - TM5(B205/208/209/212): 펩타이드 주요 접촉면, 친화도 기여 높음.\n"
        "   - TM6(B272/273/276/279): 활성화 게이트, 선택적 결합에 기여.\n"
        "   - ECL3(B284/286): SST14 결합 확장 영역, 길이 의존적 접촉.\n"
        "2. 궤적 기반 전략 점검: 최근 궤적에서 ddG 정체·반복 전략을 식별하고, "
        "직전과 다른 포켓 조합·바인더 길이·diffusion_steps를 탐색하라.\n"
        "   정체(3+ epoch 무개선)면 반드시 다른 포켓 조합 또는 길이 범위를 제안하라.\n"
        "3. 탐색 다양성 전략: 바인더 길이 범위와 diffusion_steps의 탐색 의미 — "
        "길이 확장(>14aa)은 ECL3 접촉 증가, steps 증가(>100)는 구조 다양성 확대.\n"
        "4. 다목적 균형: ddG·Δmargin(선택성)·반감기·독성 surrogate를 균형 있게 고려. "
        "surrogate(half_life_h·admet_score·hc50)는 참고 신호(신뢰 등급: 반감기/ADMET MED, "
        "hc50 L-aa 역변별 AUC=0.146) — surrogate만 좇아 ddG·Δmargin을 희생하지 말 것.\n\n"
        "## 허용 hotspot 잔기 (이 풀 외 잔기 사용 금지)\n"
        f"{_pocket_residues_str}\n"
        "B150·B200·B220 등 임의 잔기 사용 금지.\n\n"
        "## 출력 규칙\n"
        "- hypothesis: 한국어, 위 4단계 추론 결과를 **구조화된 근거** 형식으로 서술 "
        "(포켓 선택 이유·기대 효과·궤적 기반 판단 포함, 3-6문장 허용).\n"
        "- pocket_rationale: 한국어, 이번에 선택한 포켓 조합의 SSTR2-선택성 논리 (1-2문장).\n"
        "- diversity_rationale: 한국어, 탐색 다양성 관점에서의 길이·steps 선택 근거 (1-2문장).\n"
        "- 변경하지 않아도 될 파라미터는 null로 반환.\n"
        "- JSON 외의 텍스트 출력 금지."
    )

    # 궤적 섹션 (있으면 포함)
    trajectory_section = ""
    if trajectory_text:
        trajectory_section = f"\n\n{trajectory_text}\n"

    user_prompt = (
        f"## 현재 리더보드 요약\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n\n"
        f"## 현재 생성 파라미터\n{json.dumps(current_params, ensure_ascii=False, indent=2)}"
        f"{trajectory_section}\n\n"
        "다음 JSON 형식으로만 응답하십시오:\n"
        "{\n"
        '  "hypothesis": "구조화된 조정 근거 (한국어, 포켓 선택·궤적 판단·기대효과 포함, 3-6문장)",\n'
        '  "pocket_rationale": "이번 포켓 조합의 SSTR2-선택성 논리 (한국어, 1-2문장)",\n'
        '  "diversity_rationale": "탐색 다양성 전략 근거 (한국어, 1-2문장)",\n'
        '  "hotspot_res": ["B192", "B197", ...] or null,\n'
        '  "binder_length_range": [min_int, max_int] or null,\n'
        '  "diffusion_steps": int_or_null,\n'
        '  "n_backbone": int_or_null,\n'
        '  "k_seq_per_backbone": int_or_null,\n'
        '  "arm_preference": "rfdiffusion" or "diffpepbuilder" or null\n'
        "}\n"
        "JSON 외의 텍스트는 출력하지 마십시오."
    )

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.4,
        "max_tokens": 2048,  # 풍부화 hypothesis·pocket_rationale·diversity_rationale 포함
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }

    url = f"{vllm_url.rstrip('/')}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer EMPTY",
    }

    body = _http_post_json(url, payload, headers, timeout)
    if body is None:
        return None

    choices = body.get("choices", [])
    if not choices:
        return None

    content = choices[0].get("message", {}).get("content", "")
    if not content:
        return None

    # <think> 블록 제거 (Qwen3 thinking 모드 혼입 방지)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    if "<think>" in content and "</think>" not in content:
        content = content.split("<think>", 1)[0].strip()

    # JSON 파싱
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # 중괄호 추출 시도
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                pass
    logger.warning("[SiloAPlanner] LLM 응답 JSON 파싱 실패: %r", content[:200])
    return None


# ---------------------------------------------------------------------------
# de novo 맞춤 경량 사전검토 (Critic Pre-review)
# ---------------------------------------------------------------------------

@dataclass
class SiloACriticResult:
    """de novo 전략 사전검토 결과 (2-관점: 구조 + 다양성)."""
    verdict: str                          # "approve" | "concerns"
    structure_ok: bool                    # 포켓-선택성 관점 승인 여부
    diversity_ok: bool                    # 다양성·탐색 정체 관점 승인 여부
    concerns: List[str]                   # 지적 사항 목록 (빈 리스트=없음)
    suggested_revisions: Optional[Dict[str, Any]]  # 수정 제안 (None=제안 없음)
    raw_response: Optional[str] = None    # LLM 원문
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "structure_ok": self.structure_ok,
            "diversity_ok": self.diversity_ok,
            "concerns": self.concerns,
            "suggested_revisions": self.suggested_revisions,
            "raw_response": self.raw_response,
            "created_at": self.created_at,
        }


def _call_critic_prereview(
    planner_proposal: Dict[str, Any],
    current_params: Dict[str, Any],
    summary: Dict[str, Any],
    trajectory_text: Optional[str],
    vllm_url: str,
    model: str,
    timeout: int,
) -> SiloACriticResult:
    """de novo 맞춤 경량 사전검토 — 구조·다양성 2-관점.

    Silo B 4-전문가 패널 복사 아님.
    2-관점만 평가:
      1) 구조 관점: 선택된 hotspot이 SSTR2-고유 포켓(ECL2/TM5/TM6/ECL3)인가,
         선택성 기여가 논리적인가.
      2) 다양성 관점: 직전 전략과 동일하거나 탐색 정체를 반복하는가.

    반환: SiloACriticResult (LLM 실패 시 자동 approve — fail-open).
    """
    _pocket_residues_str = (
        "ECL2(B192/193/195/197), TM5(B205/208/209/212), "
        "TM6(B272/273/276/279), ECL3(B284/286)"
    )

    system_prompt = (
        "당신은 SSTR2 de novo 펩타이드 바인더 설계 사전검토 전문가입니다.\n"
        "플래너가 제안한 de novo 생성 전략을 2-관점으로 사전검토하십시오.\n\n"
        "## 관점 1: 구조·선택성\n"
        "- 선택된 hotspot_res가 SSTR2-고유 포켓(ECL2/TM5/TM6/ECL3) 잔기인지 확인.\n"
        f"  허용 풀: {_pocket_residues_str}\n"
        "- 이 포켓 조합이 SSTR1/SSTR3/SSTR5와 구조적으로 차별화된 영역을 타겟하는지 판단.\n"
        "- 특히 ECL2(선택성 핵심)가 포함됐는지, 또는 제외된 경우 그 이유가 있는지 평가.\n\n"
        "## 관점 2: 다양성·탐색 정체\n"
        "- 직전 파라미터와 비교해 hotspot·길이·steps가 실질적으로 달라졌는지 확인.\n"
        "- 궤적에서 이미 실패한 전략(같은 포켓·같은 길이)을 반복하는지 판단.\n"
        "- diffusion_steps가 다양성 확대에 충분한지(정체 시 steps↑ 권장) 평가.\n\n"
        "## 출력 규칙\n"
        "verdict: 'approve' (2관점 모두 OK) 또는 'concerns' (하나 이상 문제).\n"
        "structure_ok: bool — 구조·선택성 관점 OK 여부.\n"
        "diversity_ok: bool — 다양성·탐색 관점 OK 여부.\n"
        "concerns: 문제 사항 목록 (없으면 빈 배열).\n"
        "suggested_revisions: 수정 제안 파라미터 dict 또는 null.\n"
        "JSON 외 텍스트 출력 금지."
    )

    user_prompt = (
        f"## 플래너 제안\n{json.dumps(planner_proposal, ensure_ascii=False, indent=2)}\n\n"
        f"## 직전 파라미터\n{json.dumps(current_params, ensure_ascii=False, indent=2)}\n\n"
        f"## 리더보드 요약\n{json.dumps(summary, ensure_ascii=False, indent=2)}"
    )
    if trajectory_text:
        user_prompt += f"\n\n{trajectory_text}"
    user_prompt += (
        "\n\n다음 JSON 형식으로만 응답하십시오:\n"
        "{\n"
        '  "verdict": "approve" or "concerns",\n'
        '  "structure_ok": true or false,\n'
        '  "diversity_ok": true or false,\n'
        '  "concerns": ["concern1", "concern2"] or [],\n'
        '  "suggested_revisions": {"hotspot_res": [...], ...} or null\n'
        "}\n"
        "JSON 외의 텍스트는 출력하지 마십시오."
    )

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }

    url = f"{vllm_url.rstrip('/')}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer EMPTY",
    }

    body = _http_post_json(url, payload, headers, timeout)
    if body is None:
        logger.warning("[SiloACritic] LLM 호출 실패 → fail-open (자동 approve)")
        return SiloACriticResult(
            verdict="approve",
            structure_ok=True,
            diversity_ok=True,
            concerns=[],
            suggested_revisions=None,
            raw_response=None,
        )

    choices = body.get("choices", [])
    if not choices:
        logger.warning("[SiloACritic] LLM 응답 choices 없음 → fail-open")
        return SiloACriticResult(
            verdict="approve",
            structure_ok=True,
            diversity_ok=True,
            concerns=[],
            suggested_revisions=None,
            raw_response=None,
        )

    content = choices[0].get("message", {}).get("content", "")
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    raw_response = content[:500]  # 로그용 트림

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end > start:
            try:
                parsed = json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                parsed = None
        else:
            parsed = None

    if parsed is None:
        logger.warning("[SiloACritic] JSON 파싱 실패 → fail-open")
        return SiloACriticResult(
            verdict="approve",
            structure_ok=True,
            diversity_ok=True,
            concerns=[],
            suggested_revisions=None,
            raw_response=raw_response,
        )

    verdict = parsed.get("verdict", "approve")
    structure_ok = bool(parsed.get("structure_ok", True))
    diversity_ok = bool(parsed.get("diversity_ok", True))
    concerns_raw = parsed.get("concerns", [])
    concerns = [str(c) for c in concerns_raw] if isinstance(concerns_raw, list) else []
    suggested_revisions = parsed.get("suggested_revisions")
    if not isinstance(suggested_revisions, dict):
        suggested_revisions = None

    return SiloACriticResult(
        verdict=verdict,
        structure_ok=structure_ok,
        diversity_ok=diversity_ok,
        concerns=concerns,
        suggested_revisions=suggested_revisions,
        raw_response=raw_response,
    )


def _append_discussion_log(
    discussion_log_path: Path,
    epoch: int,
    planner_proposal: Dict[str, Any],
    critic_result: SiloACriticResult,
    final_params: Optional[Dict[str, Any]] = None,
) -> None:
    """사전검토 토론 결과를 runs/silo_a_flow/silo_a_discussion_log.jsonl에 기록.

    원자적 append: 파일 잠금 없이도 JSONL 라인 단위 append는 안전.
    환각 0: 실제 결정 결과만 기록.
    """
    discussion_log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "epoch": epoch,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "planner_proposal": {
            "hypothesis": planner_proposal.get("hypothesis", ""),
            "pocket_rationale": planner_proposal.get("pocket_rationale", ""),
            "diversity_rationale": planner_proposal.get("diversity_rationale", ""),
            "hotspot_res": planner_proposal.get("hotspot_res"),
            "binder_length_range": planner_proposal.get("binder_length_range"),
            "diffusion_steps": planner_proposal.get("diffusion_steps"),
            "arm_preference": planner_proposal.get("arm_preference"),
        },
        "critic_result": critic_result.to_dict(),
        "final_applied": final_params,
    }
    line = json.dumps(record, ensure_ascii=False) + "\n"
    try:
        with discussion_log_path.open("a", encoding="utf-8") as fh:
            fh.write(line)
        logger.debug("[SiloACritic] discussion_log append: epoch=%d verdict=%s",
                     epoch, critic_result.verdict)
    except Exception as exc:
        logger.warning("[SiloACritic] discussion_log 기록 실패: %s", exc)


# ---------------------------------------------------------------------------
# 파라미터 적용 유효성 검증
# ---------------------------------------------------------------------------

def _parse_contig_binder_length(contigs: str) -> Tuple[int, int]:
    """contig 문자열에서 바인더 길이 범위를 파싱.

    예) "B40-187/0 B189-327/0 12-16" → (12, 16)
        "B1-369/0 14"    → (14, 14)
    """
    # 공백 이후 마지막 토큰이 바인더 부분
    parts = contigs.strip().split()
    binder_part = parts[-1] if parts else "12-16"
    if "-" in binder_part:
        lo, hi = binder_part.split("-", 1)
        try:
            return int(lo), int(hi)
        except ValueError:
            return 12, 16
    try:
        v = int(binder_part)
        return v, v
    except ValueError:
        return 12, 16


def _build_contig_with_binder_length(receptor_part: str, lo: int, hi: int) -> str:
    """receptor contig 부분 + 바인더 길이 범위로 contig 문자열 재구성.

    예) receptor_part="B40-187/0 B189-327/0", lo=12, hi=16 → "B40-187/0 B189-327/0 12-16"
    """
    if lo == hi:
        return f"{receptor_part} {lo}"
    return f"{receptor_part} {lo}-{hi}"


def _validate_and_clamp_params(proposed: Dict[str, Any]) -> Dict[str, Any]:
    """LLM 제안 파라미터를 안전 범위로 clamp하고 유효성 검증.

    반환: 유효한 파라미터만 포함한 dict (null/None 제외).
    """
    cleaned: Dict[str, Any] = {}

    # diffusion_steps
    if proposed.get("diffusion_steps") is not None:
        try:
            steps = int(proposed["diffusion_steps"])
            cleaned["diffusion_steps"] = max(_DIFFUSION_STEPS_MIN, min(_DIFFUSION_STEPS_MAX, steps))
        except (TypeError, ValueError):
            logger.warning("[SiloAPlanner] diffusion_steps 값 무효: %r", proposed.get("diffusion_steps"))

    # n_backbone
    if proposed.get("n_backbone") is not None:
        try:
            nb = int(proposed["n_backbone"])
            cleaned["n_backbone"] = max(1, min(10, nb))
        except (TypeError, ValueError):
            pass

    # k_seq_per_backbone
    if proposed.get("k_seq_per_backbone") is not None:
        try:
            ks = int(proposed["k_seq_per_backbone"])
            cleaned["k_seq_per_backbone"] = max(1, min(8, ks))
        except (TypeError, ValueError):
            pass

    # hotspot_res: List[str], 각 항목이 "B\d+" 형식인지 검증
    if proposed.get("hotspot_res") is not None:
        hs = proposed["hotspot_res"]
        if isinstance(hs, list) and len(hs) >= 1:
            valid_hs = [h for h in hs if isinstance(h, str) and re.match(r"^B\d+$", h)]
            if valid_hs:
                cleaned["hotspot_res"] = valid_hs
            else:
                logger.warning("[SiloAPlanner] hotspot_res 유효 항목 없음: %r", hs)

    # binder_length_range: [min, max]
    if proposed.get("binder_length_range") is not None:
        blr = proposed["binder_length_range"]
        if isinstance(blr, (list, tuple)) and len(blr) == 2:
            try:
                lo, hi = int(blr[0]), int(blr[1])
                lo = max(_BINDER_LEN_MIN, min(_BINDER_LEN_MAX, lo))
                hi = max(lo, min(_BINDER_LEN_MAX, hi))
                cleaned["binder_length_range"] = [lo, hi]
            except (TypeError, ValueError):
                logger.warning("[SiloAPlanner] binder_length_range 값 무효: %r", blr)

    # arm_preference (현재는 기록만, 실제 arm 전환 미구현)
    if proposed.get("arm_preference") in ("rfdiffusion", "diffpepbuilder"):
        cleaned["arm_preference"] = proposed["arm_preference"]

    return cleaned


# ---------------------------------------------------------------------------
# 규칙 기반 fallback (정체 탈출 / 다양성 확대)
# ---------------------------------------------------------------------------

def _get_ladder_steps(stagnation_count: int, patience_epochs: int) -> Optional[int]:
    """stagnation_count와 patience_epochs 배수에 따른 사다리 diffusion_steps 결정.

    _STAGNATION_STEPS_LADDER를 내림차순으로 순회해 첫 매칭 임계를 반환한다.
    stagnation_count < patience_epochs 이면 None 반환 (정체 미발생).

    예) patience=50, stagnation=197:
        197/50 ≈ 3.9 → 2× 임계(100) 적용, 4× 임계(150)는 넘지 않음
        → 결과: 100
    """
    if stagnation_count < patience_epochs or patience_epochs <= 0:
        return None
    ratio = stagnation_count // patience_epochs  # 배수 (정수 나눗셈)
    for threshold_mult, forced_steps in _STAGNATION_STEPS_LADDER:
        if ratio >= threshold_mult:
            return forced_steps
    return None


def _rule_based_adjustment(
    summary: Dict[str, Any],
    current_params: Dict[str, Any],
    patience_epochs: int,
    stagnation_count: int,
) -> Tuple[Dict[str, Any], str]:
    """정체 상황에서 규칙 기반으로 파라미터를 조정.

    Silo B DiversityPolicy 패턴 참고 (hotspot 셔플, 바인더 길이 확장, steps 상향).
    197 epoch 정체와 같은 장기 정체에 대응하기 위해 stagnation_count 배수에 따라
    diffusion_steps를 사다리 방식으로 단계적 강제 증가한다.

    반환: (adjusted_params dict, hypothesis str)
    """
    if stagnation_count < patience_epochs:
        # 아직 정체 아님 — 변경 없음
        return {}, f"정체 미발생(stagnation_count={stagnation_count}/{patience_epochs}) — 현재 파라미터 유지"

    # 정체 탈출 전략
    rng = random.Random()  # 재현 가능한 랜덤 (seed 미설정 = 시각 기반)
    adj: Dict[str, Any] = {}
    reasons: List[str] = []

    # 1) hotspot 셔플: 기존 핫스팟 중 무작위 2개 교체
    curr_hs = current_params.get("hotspot_res", list(_DEFAULT_HOTSPOT))
    pool_not_current = [h for h in _HOTSPOT_POOL if h not in curr_hs]
    if pool_not_current and len(pool_not_current) >= 2:
        new_hs = list(curr_hs)
        n_replace = min(2, len(pool_not_current), len(new_hs))
        replace_idx = rng.sample(range(len(new_hs)), n_replace)
        new_picks = rng.sample(pool_not_current, n_replace)
        for idx, pick in zip(replace_idx, new_picks):
            new_hs[idx] = pick
        adj["hotspot_res"] = new_hs
        reasons.append(f"핫스팟 {n_replace}개 교체({', '.join(new_picks)})")

    # 2) 바인더 길이 확장 (contig에서 현재 범위 +2 확대)
    curr_contig = current_params.get("contigs", "B40-187/0 B189-327/0 12-16")
    contig_parts = curr_contig.strip().split()
    receptor_part = " ".join(contig_parts[:-1]) if len(contig_parts) > 1 else "B40-187/0 B189-327/0"
    curr_lo, curr_hi = _parse_contig_binder_length(curr_contig)
    new_lo = max(_BINDER_LEN_MIN, curr_lo - 1)
    new_hi = min(_BINDER_LEN_MAX, curr_hi + 2)
    if (new_lo, new_hi) != (curr_lo, curr_hi):
        adj["binder_length_range"] = [new_lo, new_hi]
        adj["contigs"] = _build_contig_with_binder_length(receptor_part, new_lo, new_hi)
        reasons.append(f"바인더 길이 확장({curr_lo}-{curr_hi} → {new_lo}-{new_hi})")

    # 3) diffusion_steps 사다리 강제 상향 (장기 정체 탈출)
    #    stagnation_count // patience_epochs 배수로 사다리 단계 결정:
    #      1× → 50, 2× → 100, 4× → 150, 8× → 200
    curr_steps = current_params.get("diffusion_steps", 50)
    ladder_steps = _get_ladder_steps(stagnation_count, patience_epochs)
    if ladder_steps is not None and ladder_steps > curr_steps:
        # 사다리 강제 적용 (단순 1.3배보다 우선)
        new_steps = ladder_steps
        reasons.append(
            f"diffusion_steps 사다리 강제({curr_steps} → {new_steps}, "
            f"stagnation={stagnation_count}, patience={patience_epochs}, "
            f"ratio={stagnation_count // patience_epochs if patience_epochs > 0 else 'inf'}×)"
        )
    else:
        # 단기 정체: 기존 1.3배 점진 상향
        new_steps = min(_DIFFUSION_STEPS_MAX, int(curr_steps * 1.3))
        if new_steps != curr_steps:
            reasons.append(f"diffusion_steps 상향({curr_steps} → {new_steps})")
    if new_steps != curr_steps:
        adj["diffusion_steps"] = new_steps

    hypothesis = (
        f"정체 탈출({stagnation_count}epoch 무개선): "
        + ("; ".join(reasons) if reasons else "파라미터 변경 없음 (이미 최대 범위)")
    )
    return adj, hypothesis


# ---------------------------------------------------------------------------
# 핵심 퍼블릭 함수: 다음 epoch 파라미터 결정
# ---------------------------------------------------------------------------

def plan_next_epoch(
    epoch: int,
    leaderboard_path: Path,
    current_params: Dict[str, Any],
    stagnation_count: int,
    patience_epochs: int = 3,
    vllm_url: str = _DEFAULT_VLLM_URL,
    vllm_model: str = _DEFAULT_LLM_MODEL,
    vllm_timeout: int = _LLM_TIMEOUT,
    output_dir: Optional[Path] = None,
    experiment_log_path: Optional[Path] = None,
    trajectory_k: int = 6,
    prereview_enabled: bool = True,
    discussion_log_path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], SiloAPlannerDecision]:
    """다음 epoch 파라미터를 결정하고 provenance를 기록.

    Args:
        epoch: 현재 epoch 번호.
        leaderboard_path: Silo A 리더보드 JSON 경로.
        current_params: 현재 epoch 생성 파라미터 dict.
            {contigs, hotspot_res, diffusion_steps, n_backbone, k_seq_per_backbone, ...}
        stagnation_count: 리더보드 best_ddg가 개선되지 않은 연속 epoch 수.
        patience_epochs: 이 값 이상 정체 시 파라미터 조정 트리거.
        vllm_url: vLLM 서버 URL.
        vllm_model: 사용할 모델 이름.
        vllm_timeout: LLM 호출 타임아웃(초).
        output_dir: provenance 파일 저장 디렉토리 (None=저장 생략).
        experiment_log_path: Silo A experiment_log.jsonl 경로.
            None이면 leaderboard_path 부모 디렉토리에서 탐색.
            **반드시 Silo A 경로여야 한다.** pyrosetta_flow 경로 금지.
        trajectory_k: 궤적 요약에 포함할 최근 epoch 수 (기본 6).
        prereview_enabled: True이면 플래너 제안 후 구조·다양성 2-관점 사전검토 1라운드 실행.
            False이면 기존 단독 플래너 동작 (하위호환). 기본값=True.
        discussion_log_path: 사전검토 토론 결과 기록 경로
            (None이면 leaderboard_path 부모 / silo_a_discussion_log.jsonl).

    Returns:
        (updated_params, SiloAPlannerDecision)
        updated_params: current_params 에 조정사항을 병합한 새 dict.
        SiloAPlannerDecision: 결정 내역 (기록용).
    """
    # 1. 리더보드 요약
    summary = _summarize_leaderboard(leaderboard_path)

    # 1b. 궤적 요약 생성 (Silo A 경로만)
    trajectory_text: Optional[str] = None
    if epoch > 1:
        if experiment_log_path is None:
            # leaderboard_path 부모 디렉토리에서 experiment_log.jsonl 탐색
            candidate_log = leaderboard_path.parent / "experiment_log.jsonl"
            if candidate_log.exists():
                experiment_log_path = candidate_log
        if experiment_log_path is not None:
            trajectory_text = _summarize_trajectory(experiment_log_path, k=trajectory_k)
            logger.debug("[SiloAPlanner] 궤적 요약 생성: %d 줄", len(trajectory_text.splitlines()))

    # 2. LLM 경로 시도
    llm_proposed: Optional[Dict[str, Any]] = None
    llm_raw: Optional[str] = None
    fallback_reason: Optional[str] = None

    if stagnation_count >= patience_epochs or epoch == 1:
        # 정체 시 또는 첫 epoch 이후 LLM 호출 (첫 epoch는 초기 탐색 방향 설정)
        try:
            llm_result = _call_llm_for_plan(
                summary=summary,
                current_params=current_params,
                vllm_url=vllm_url,
                model=vllm_model,
                timeout=vllm_timeout,
                trajectory_text=trajectory_text,
            )
            if llm_result is not None:
                llm_raw = json.dumps(llm_result, ensure_ascii=False)
                validated = _validate_and_clamp_params(llm_result)
                if validated:
                    llm_proposed = validated
                    hypothesis_text = llm_result.get("hypothesis", "LLM 제안 (근거 없음)")
                    logger.info("[SiloAPlanner] LLM 경로 적용: %s", hypothesis_text)
                else:
                    fallback_reason = "LLM 제안 파라미터가 유효성 검증 실패 — 규칙 fallback"
            else:
                fallback_reason = "LLM 응답 None (연결 실패 또는 파싱 실패)"
        except Exception as exc:
            fallback_reason = f"LLM 호출 예외: {exc!r}"
            logger.warning("[SiloAPlanner] LLM 예외 → 규칙 fallback: %s", exc)
    else:
        fallback_reason = (
            f"정체 미발생(stagnation_count={stagnation_count}/{patience_epochs}) "
            "— LLM 호출 생략"
        )

    # 3. 규칙 fallback (LLM 실패 또는 stagnation_count < patience_epochs)
    if llm_proposed is not None:
        # LLM 성공 경로
        adj_params = llm_proposed
        hypothesis = llm_result.get("hypothesis", "LLM 파라미터 조정") if isinstance(llm_result, dict) else "LLM 파라미터 조정"  # type: ignore[possibly-undefined]
        adj_type = "llm_guided"

        # 3a. de novo 맞춤 사전검토 (prereview_enabled=True이고 LLM 성공 경로일 때만)
        if prereview_enabled and isinstance(llm_result, dict):  # type: ignore[possibly-undefined]
            try:
                critic = _call_critic_prereview(
                    planner_proposal=llm_result,
                    current_params=current_params,
                    summary=summary,
                    trajectory_text=trajectory_text,
                    vllm_url=vllm_url,
                    model=vllm_model,
                    timeout=vllm_timeout,
                )
                logger.info(
                    "[SiloACritic] epoch=%d verdict=%s struct_ok=%s div_ok=%s concerns=%s",
                    epoch, critic.verdict, critic.structure_ok, critic.diversity_ok,
                    critic.concerns,
                )

                # critic이 concerns를 제시하고 수정 제안이 있으면 병합 (단 1회 재고 — 재호출 없음)
                if critic.verdict == "concerns" and critic.suggested_revisions:
                    rev_validated = _validate_and_clamp_params(critic.suggested_revisions)
                    if rev_validated:
                        adj_params = dict(adj_params)
                        adj_params.update(rev_validated)
                        hypothesis = (
                            hypothesis
                            + f" [Critic 재고: {'; '.join(critic.concerns[:2])}]"
                        )
                        adj_type = "llm_guided_with_prereview"
                        logger.info("[SiloACritic] 재고 적용: %s", rev_validated)

                # discussion_log 기록
                _disc_log_path = discussion_log_path
                if _disc_log_path is None:
                    _disc_log_path = leaderboard_path.parent / "silo_a_discussion_log.jsonl"
                _append_discussion_log(
                    _disc_log_path, epoch, llm_result, critic,
                    final_params=adj_params,
                )

            except Exception as exc:
                logger.warning("[SiloACritic] 사전검토 예외 — 플래너 제안 그대로 사용: %s", exc)

        # contigs 재구성 (binder_length_range 적용)
        if "binder_length_range" in adj_params:
            lo, hi = adj_params.pop("binder_length_range")
            contig_parts = current_params.get("contigs", "B40-187/0 B189-327/0 12-16").strip().split()
            receptor_part = " ".join(contig_parts[:-1]) if len(contig_parts) > 1 else "B40-187/0 B189-327/0"
            adj_params["contigs"] = _build_contig_with_binder_length(receptor_part, lo, hi)
    else:
        # 규칙 fallback
        adj_params, hypothesis = _rule_based_adjustment(
            summary=summary,
            current_params=current_params,
            patience_epochs=patience_epochs,
            stagnation_count=stagnation_count,
        )
        adj_type = "rule_stagnation" if stagnation_count >= patience_epochs else "rule_noop"

    # 4. 현재 파라미터에 조정사항 병합
    updated_params = dict(current_params)
    updated_params.update(adj_params)

    # 5. 결정 기록
    decision = SiloAPlannerDecision(
        epoch=epoch,
        hypothesis=hypothesis,
        adjustment_type=adj_type,
        applied_params=adj_params,
        llm_raw_response=llm_raw,
        fallback_reason=fallback_reason,
        leaderboard_summary=summary,
    )

    # 6. Provenance 파일 저장
    if output_dir is not None:
        _save_planner_record(output_dir, epoch, decision)

    return updated_params, decision


# ---------------------------------------------------------------------------
# Provenance 저장
# ---------------------------------------------------------------------------

def _save_planner_record(output_dir: Path, epoch: int, decision: SiloAPlannerDecision) -> None:
    """epoch별 플래너 결정을 runs/silo_a_flow/silo_a_planner_e{NN}.json에 기록."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"silo_a_planner_e{epoch:04d}.json"
    payload = decision.to_dict()
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    logger.info("[SiloAPlanner] Provenance 저장: %s", path)


# ---------------------------------------------------------------------------
# 정체 카운터 헬퍼
# ---------------------------------------------------------------------------

def update_stagnation_count(
    prev_best_ddg: Optional[float],
    current_best_ddg: Optional[float],
    prev_stagnation_count: int,
) -> Tuple[int, bool]:
    """best_ddg 개선 여부에 따라 stagnation_count 업데이트.

    Returns:
        (new_stagnation_count, improved_flag)
        improved_flag=True 이면 개선 발생.
    """
    if current_best_ddg is None:
        # 아직 유효 도킹 없음 — 정체로 간주
        return prev_stagnation_count + 1, False

    if prev_best_ddg is None:
        # 최초 유효 결과 — 개선 발생
        return 0, True

    # ddG가 낮을수록 좋음 (음수 방향)
    if current_best_ddg < prev_best_ddg - 0.1:  # 0.1 kcal/mol 유의미한 개선
        return 0, True

    return prev_stagnation_count + 1, False
