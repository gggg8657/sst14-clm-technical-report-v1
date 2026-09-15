#!/usr/bin/env python3
"""
build_discussion_view.py
에이전트 토론 로그(discussion_log.jsonl + discovery_run_provenance.log)를
docs/discussion_data.json 으로 변환한다.

실행: python scripts/build_discussion_view.py  (repo 루트 기준)
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

# ── 경로 설정 ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DISCUSSION_LOG = REPO_ROOT / "runs" / "pyrosetta_flow" / "sst14_agentic_mutdock" / "discussion_log.jsonl"
PROVENANCE_LOG = REPO_ROOT / "runs" / "pyrosetta_flow" / "discovery_run_provenance.log"
# 출력은 프로젝트 루트 docs/ 에 — 웹뷰(docs/discussion/index.html → ../discussion_data.json)가 읽는 위치.
# (이전엔 REPO_ROOT/docs 에 써서 웹뷰가 stale 데이터를 읽는 경로 버그가 있었음.
#  build_structure_view.py 등 다른 빌더와 동일하게 프로젝트 루트 docs/ 로 통일.)
PROJECT_ROOT = REPO_ROOT.parent.parent.parent  # ai4sci-kaeri → repos → AgenticAI4SCIENCE… → SST14-M_scr
OUT_DIR = PROJECT_ROOT / "docs"
OUT_FILE = OUT_DIR / "discussion_data.json"

# ── 유틸 ─────────────────────────────────────────────────────────────────────

def parse_provenance(path: Path) -> tuple[dict[int, str], dict[int, int]]:
    """provenance 로그에서 iteration별 첫 타임스탬프 + llm_calls 추출."""
    timestamps: dict[int, str] = {}
    llm_calls_map: dict[int, int] = {}

    # 패턴: [2026-06-23 03:06:35][ScientistCritic] INFO: [pre-review] Iteration 1 Round 1 사전검토 시작
    ts_pattern = re.compile(
        r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*\[pre-review\] Iteration (\d+) Round \d+ 사전검토 시작"
    )
    # 패턴: [pre-review] iter=1 완료: mode=expert_panel, 1 라운드, llm_calls=5, final_focus=...
    llm_pattern = re.compile(
        r"\[pre-review\] iter=(\d+) 완료:.*mode=expert_panel.*llm_calls=(\d+)"
    )

    if not path.exists():
        return timestamps, llm_calls_map

    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = ts_pattern.search(line)
            if m:
                iteration = int(m.group(2))
                if iteration not in timestamps:
                    # ISO 8601 형식으로 변환
                    timestamps[iteration] = m.group(1).replace(" ", "T")
                continue

            m2 = llm_pattern.search(line)
            if m2:
                iteration = int(m2.group(1))
                if iteration not in llm_calls_map:
                    llm_calls_map[iteration] = int(m2.group(2))

    return timestamps, llm_calls_map


def load_discussion_groups(path: Path) -> list[dict]:
    """discussion_log.jsonl → 로그 순서 기반 '토론 그룹' 리스트.

    iteration 번호는 run마다 1~20으로 리셋되어 충돌·혼동을 유발하므로,
    로그 순서대로 처리해 (a) 연속 동일 iteration 레코드를 한 토론으로 묶고,
    (b) iteration이 이전보다 작아지면(리셋) 새 run으로 판정,
    (c) 전역 순차 고유번호(uid=1,2,3,...)를 부여한다.

    반환: [{"uid": int, "run": int, "iteration": int, "records": [...]}, ...] (로그 순서)
    """
    groups: list[dict] = []
    uid = 0
    run = 1
    prev_iter: int | None = None
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            it = rec["iteration"]
            # run 경계: iteration이 이전보다 작아지면(리셋) 새 run
            if prev_iter is not None and it < prev_iter:
                run += 1
            # 새 토론 시작: 직전 그룹과 iteration이 다르면(연속 동일은 같은 토론의 여러 라운드)
            if groups and groups[-1]["iteration"] == it and groups[-1]["run"] == run:
                groups[-1]["records"].append(rec)
            else:
                uid += 1
                groups.append({"uid": uid, "run": run, "iteration": it, "records": [rec]})
            prev_iter = it
    return groups


# 패널 설계상 최대 토론 라운드 수 (AG_src/agents/expert_panel.py::_PANEL_MAX_DISCUSSION_ROUNDS
# 기본값과 동일 — env override는 반영하지 못하지만 표시용 상수이므로 하드코딩 허용).
_MAX_DISCUSSION_ROUNDS_DEFAULT = 5


def _normalize_expert_verdict(v: dict) -> dict:
    """expert_verdicts 항목 하나를 정규화 (하위호환: llm_backend 없는 과거 레코드는 'unknown').

    5-전문가 패널(pharma/biology/chemistry/radiochem/math) 도입 이전 레코드는
    domain 4개(pharma/biology/chemistry/math)만 있을 수 있음 — 그대로 통과(하드코딩 없음).
    """
    return {
        "domain": v.get("domain", ""),
        "severity": v.get("severity", "low"),
        "concerns": v.get("concerns", []),
        # F: 모델 이질성 배지용. 없으면 "unknown"(구버전 레코드) — "n/a" 대신 unknown으로
        # 실제 backend 미상임을 명시.
        "llm_backend": v.get("llm_backend") or "unknown",
    }


_KST = timezone(timedelta(hours=9))


def _to_kst(ts_iso: str | None) -> str | None:
    """UTC ISO 문자열/naive 로그 타임스탬프를 KST 표기로 변환 (없으면 None)."""
    if not ts_iso:
        return None
    s = ts_iso.replace("Z", "").replace("T", " ").strip()
    try:
        dt = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return dt.astimezone(_KST).strftime("%Y-%m-%d %H:%M:%S KST")
    except ValueError:
        return ts_iso


def build_iteration_entry(
    group: dict,
    timestamps: dict[int, str],
    llm_calls_map: dict[int, int],
) -> dict:
    """단일 토론 그룹(uid/run/iteration/records) 정규화 엔트리 생성."""
    uid = group["uid"]
    run = group["run"]
    iteration = group["iteration"]
    records = group["records"]
    # 마지막 레코드가 가장 최종 상태
    rep = records[-1]

    # discussion_turns: 여러 레코드에서 모든 turn을 수집한다.
    # 단일 레코드(iteration=1 approve)의 경우 마지막 레코드의 turns를 사용.
    # 여러 레코드가 있을 경우(approve=False → revision → approve) 최종 레코드의 turns를 사용.
    raw_turns = rep.get("discussion_turns") or []
    discussion_turns = [
        {**t, "llm_backend": t.get("llm_backend") or "unknown"}
        for t in raw_turns
    ]
    discussion_rounds = rep.get("discussion_rounds", 1)

    raw_fanin = rep.get("fanin") or {"approve": False, "merged_concerns": []}
    # minority_dissent / stall_detected / stall_streak — 새 패널(커밋 96f299f1) 필드.
    # 과거 레코드에는 없으므로 graceful하게 None/False 처리(하위호환).
    fanin = {
        **raw_fanin,
        "minority_dissent": raw_fanin.get("minority_dissent"),
        "stall_detected": raw_fanin.get("stall_detected", False),
        "stall_streak": raw_fanin.get("stall_streak", 0),
    }

    raw_verdicts = rep.get("expert_verdicts") or []
    expert_verdicts = [_normalize_expert_verdict(v) for v in raw_verdicts]

    return {
        "uid": uid,                 # 전역 고유번호 (run 무관 순차 — 화면 표시용)
        "run": run,                 # 발굴 run 번호 (iteration 리셋마다 +1)
        "iteration": iteration,     # run 내 iteration (1~20, 참고용)
        # 각 토론 레코드의 자체 ts(신규 필드) 우선 — run 충돌 없이 정확. 구버전은 provenance 폴백.
        "timestamp": _to_kst(rep.get("ts") or timestamps.get(iteration)),
        "hypothesis": rep.get("hypothesis", ""),
        "expert_verdicts": expert_verdicts,
        "fanin": fanin,
        "final_focus": rep.get("final_focus", []),
        "llm_calls": llm_calls_map.get(iteration),
        "records_count": len(records),
        "discussion_rounds": discussion_rounds,
        "max_discussion_rounds": _MAX_DISCUSSION_ROUNDS_DEFAULT,
        "discussion_turns": discussion_turns,
    }


def main() -> None:
    if not DISCUSSION_LOG.exists():
        print(f"[오류] discussion_log.jsonl 없음: {DISCUSSION_LOG}", file=sys.stderr)
        sys.exit(1)

    t0 = datetime.now()

    # 데이터 로드 (로그 순서 기반 그룹 — 전역 고유번호 uid 부여)
    groups = load_discussion_groups(DISCUSSION_LOG)
    timestamps, llm_calls_map = parse_provenance(PROVENANCE_LOG)

    # 엔트리 생성 후 uid 최신순(내림차순) 정렬 — 최근 토론이 위로
    all_entries = [
        build_iteration_entry(g, timestamps, llm_calls_map)
        for g in groups
    ]
    all_entries.sort(key=lambda e: e["uid"], reverse=True)

    # 통계는 전체 기준, 표시는 최근 60개까지(옛 iter가 20창 독점하는 문제 완화)
    approve_count = sum(1 for e in all_entries if e["fanin"].get("approve"))
    reject_count = len(all_entries) - approve_count
    entries = all_entries[:60]

    _kst_now = datetime.now(timezone.utc).astimezone(_KST)
    output = {
        "generated_at": _kst_now.strftime("%Y-%m-%d %H:%M:%S KST"),
        "total_iterations": len(all_entries),
        "approve_count": approve_count,
        "reject_count": reject_count,
        "iterations": entries,
    }

    # 출력
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"[OK] discussion_data.json 생성 완료")
    print(f"  총 iteration: {len(entries)}")
    print(f"  approve: {approve_count}  reject: {reject_count}")
    print(f"  출력: {OUT_FILE}")
    print(f"  소요: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
