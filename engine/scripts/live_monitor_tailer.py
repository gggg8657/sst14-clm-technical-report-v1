#!/usr/bin/env python3
"""
live_monitor_tailer.py
발굴 엔진의 provenance 로그를 ~2초 주기로 tail·파싱하여
docs/live/live_status.json 을 생성한다. 라이브 모니터 페이지(docs/live/index.html)가
이 JSON을 폴링해 실시간 진행상황(패널 토론·도킹·랭킹)을 렌더한다.

- 출력 JSON은 매우 자주(2초) 재작성되므로 .gitignore 대상(autopush 커밋 제외).
- 별도 서버 불필요: 기존 정적 http.server(8899)가 그대로 서빙.

실행: setsid nohup python3 scripts/live_monitor_tailer.py &   (PPID=1 세션 독립)
"""
from __future__ import annotations

import json
import os
import re
import statistics
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROV_LOG = REPO_ROOT / "runs" / "pyrosetta_flow" / "discovery_run_provenance.log"
PROJECT_ROOT = REPO_ROOT.parent.parent.parent  # ai4sci-kaeri → repos → AgenticAI4SCIENCE… → SST14-M_scr
OUT_DIR = PROJECT_ROOT / "docs" / "live"
OUT_FILE = OUT_DIR / "live_status.json"
TREND_FILE = OUT_DIR / "affinity_trend.json"
TREND_EVERY = 15        # 몇 tick마다 전체-로그 추세 재계산 (2s×15=30s)
TREND_WINDOW = 150      # 그래프에 보낼 최근 epoch 수
# ── Silo A (de novo) ──
SILO_A_LOG = REPO_ROOT / "runs" / "silo_a_flow" / "discovery_run_provenance.log"
SILO_A_STATUS = REPO_ROOT / "runs" / "silo_a_flow" / "discovery_status.json"
SILO_A_LB = REPO_ROOT / "runs" / "silo_a_flow" / "silo_a_leaderboard.json"
SILO_A_OUT = OUT_DIR / "silo_a_status.json"

TAIL_BYTES = 900_000       # 최근 ~900KB 만 읽어 여러 iteration 커버 (로그는 수십 MB)
POLL_SEC = 2.0
_KST = timezone(timedelta(hours=9))

# ── 정규식 ──────────────────────────────────────────────────────────────────
RE_TS = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\[([A-Za-z]+)\]")
RE_PLANNER_START = re.compile(r"\[Planner\].*Iteration (\d+) 계획 갱신 시작")
RE_PLANNER_DONE = re.compile(r"\[Planner\].*계획 갱신 완료: \S*iter(\d+)")
RE_PANEL_START = re.compile(r"\[expert_panel\] iter=(\d+) round=\d+ 패널 사전검토 시작 \(focus=\[([^\]]*)\]")
RE_PANEL_VERDICT = re.compile(
    r"\[expert-panel\] iter=(\d+) (\w+) round=(\d+) severity=(\w+) stance=(\S+) concerns=(.*)"
)
RE_PANEL_BACKEND = re.compile(r"\[expert-panel\] iter=(\d+) (\w+) backend=(\S+)")
RE_CONSENSUS = re.compile(
    r"\[expert-panel\] consensus_check disc_round=(\d+): status=(\w+) high=(\d+) medium=(\d+) reason=(.*)"
)
RE_FANIN = re.compile(
    r"\[expert-panel\] fan-in approve=(\w+) merged_concerns=(\d+) llm_calls=(\d+) disc_round=(\d+)/(\d+)"
)
RE_CONSENSUS_OK = re.compile(r"\[expert-panel\] 합의 달성 disc_round=(\d+)")
RE_STALL = re.compile(r"\[expert-panel\] stall 종료: disc_round=(\d+)")
RE_MINORITY = re.compile(r"\[expert-panel\] minority_dissent 감지: domain=(\w+) severity=(\w+) reason=(.*)")
RE_HETERO = re.compile(r"\[expert_panel\] 모델 이질성 활성화: domains=\[([^\]]*)\] → (\S+) @ (\S+)")
RE_PREREVIEW_DONE = re.compile(r"\[pre-review\] iter=(\d+) 완료.*final_focus=\[([^\]]*)\]")
RE_SLOTS = re.compile(r"\[slots\] iter=(\d+) n_total=(\d+): guided=(\d+), intensify=(\d+), random=(\d+)")
RE_DOCK_PARALLEL = re.compile(
    r"\[dock-parallel\] 후보=(\d+) × nstruct=(\d+) = (\d+) trials, max_workers=(\d+)"
)
RE_DOCK_FLAT = re.compile(r"\[dock-flat\] iter(\d+)_cand(\d+): ddG_median=([-\d.]+) n_converged=(\S+)")
RE_STAGE2_HDR = re.compile(r"\[stage2\] iter=(\d+) 유망 후보=(\d+)건 nstruct=(\d+) trigger=([-\d.]+)")
RE_STAGE2 = re.compile(
    r"\[stage2\] iter(\d+)_cand(\d+): ddG 1차=([-\d.]+) → 2차=([-\d.]+) n_conv=(\S+) nstruct=(\d+)"
)
RE_GATE_START = re.compile(r"게이트 적용 시작: (\d+)개 후보")
RE_GATE_LINE = re.compile(r"Gate(\d) \(([^)]+)\): (.*)")
RE_QC_REPORT = re.compile(r"QC 보고서: (\d+)/(\d+) 통과 \(([\d.]+)%\)")
RE_CONVERGENCE = re.compile(r"\[convergence\] iter (\d+): (\w+) \(p=([\d.]+), CV=([\d.]+)\)")
RE_RCSB = re.compile(r"\[rcsb\] Found PDB matches for (\d+)")
RE_PHARMA = re.compile(r"\[pharma\] Enriched (\d+)")
RE_CLUSTER = re.compile(r"\[cluster\] Classified (\d+) candidates into (\S+)")
RE_BO = re.compile(r"\[bo\] suggested positions buffered for next iter: \[([^\]]*)\]")
RE_DASH = re.compile(r"\[dashboard-enrich\] Pushed (\d+)")
RE_TOP5 = re.compile(r"Top-5 by ddG: \[(.*)\]")
RE_TOP5_ITEM = re.compile(r"\('(iter\d+_cand\d+)', '([-\d.]+)'\)")
RE_GATE = re.compile(r"게이트 완료: (\d+) 통과 / (\d+) 실패")
RE_BEST = re.compile(r"\[intensify\] seed 갱신: (\d+)건 high_confidence. best=(\w+) dg=([-\d.]+)")
RE_MMGBSA = re.compile(r"\[mmgbsa\] 새 파일 소비.*high_confidence=(\d+)건")

# 별칭 → 실제 모델명 (디스커션 뷰와 동일 매핑)
ALIAS_MODEL = {
    "qwen3-32b": "Qwen3.5-122B (Alibaba)",
    "mistral-7b-instruct": "GLM-Z1-32B (Zhipu)",
    "hetero-8b": "GLM-Z1-32B (Zhipu)",
    "mistral-small-22b": "Qwen3-32B (Alibaba)",
    "planner-model": "Qwen3-32B (Alibaba)",
}

DOMAIN_KO = {
    "pharma": "약리", "biology": "생물", "chemistry": "화학",
    "radiochem": "방사화학", "math": "수학(자문)",
}

PHASE_LABEL = {
    "planning": "🧭 계획 갱신 중 (Planner)",
    "panel": "💬 전문가 패널 토론 중",
    "slots": "🎯 후보 선정 중",
    "docking": "🧪 도킹 실행 중 (FlexPepDock)",
    "stage2": "🔬 정밀 도킹 중 (Stage-2 nstruct=20)",
    "ranking": "📊 채점·랭킹 중",
    "reporting": "📝 보고서 생성 중",
    "feedback": "🔁 피드백 반영 중 (BO·bandit·intensify)",
    "idle": "⏸ 대기",
}


def _to_kst(ts: str | None) -> str | None:
    if not ts:
        return None
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return dt.astimezone(_KST).strftime("%H:%M:%S")
    except ValueError:
        return None


def real_model(alias: str) -> str:
    a = (alias or "").split("@")[0].replace("hetero:", "").strip()
    return ALIAS_MODEL.get(a, a or "Qwen3.5-122B (Alibaba)")


def read_tail(path: Path) -> list[str]:
    if not path.exists():
        return []
    size = path.stat().st_size
    with path.open("rb") as fh:
        if size > TAIL_BYTES:
            fh.seek(size - TAIL_BYTES)
        data = fh.read()
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    return lines[1:] if size > TAIL_BYTES and lines else lines  # 잘린 첫 줄 폐기


def find_engine_pid() -> int | None:
    """continuous 발굴 러너 PID 탐색 (없으면 None)."""
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                cmd = fh.read().decode("utf-8", errors="replace").replace("\x00", " ")
        except (OSError, IOError):
            continue
        if "run_continuous_discovery" in cmd or ("continuous" in cmd and "discovery" in cmd):
            return int(pid)
    return None


def proc_uptime(pid: int | None) -> str | None:
    if not pid:
        return None
    try:
        with open(f"/proc/{pid}/stat") as fh:
            starttime = int(fh.read().split()[21])
        with open("/proc/uptime") as fh:
            up = float(fh.read().split()[0])
        hz = os.sysconf("SC_CLK_TCK")
        sec = up - (starttime / hz)
        h, m = int(sec // 3600), int((sec % 3600) // 60)
        return f"{h}시간 {m}분" if h else f"{m}분"
    except (OSError, IOError, IndexError, ValueError):
        return None


def parse(lines: list[str]) -> dict:
    cur_iter = 0
    focus: list[int] = []
    phase = "idle"
    last_phase_idx = -1
    hetero_info: dict | None = None

    panel_events: list[dict] = []       # 현재 패널 iter의 verdict 이벤트
    panel_consensus: list[dict] = []
    panel_iter = 0
    panel_final: dict | None = None
    minority: dict | None = None
    backends: dict[tuple, str] = {}     # (iter, domain) → backend alias

    dock_total = 0
    dock_trials = 0
    dock_workers = 0
    dock_iter = 0
    dock_batch_open = False              # 마지막 dock-parallel 이후에만 결과 수집(warm-start 이전 run 오염 방지)
    dock_nstruct = 0
    dock_results: dict[str, dict] = {}  # cand → {ddg, nconv}
    stage2: dict[str, dict] = {}
    stage2_hdr: dict | None = None
    # 평가(evaluation) 단계 상태 — dock-parallel(새 배치)마다 리셋
    gates: dict[int, dict] = {}
    gate_total = 0
    qc_report: dict | None = None
    convergence: dict | None = None
    eval_steps: dict[str, dict] = {}
    top5: list[dict] = []
    gate = None
    best = None
    mmgbsa_hc = None

    feed: list[dict] = []               # 활동 로그(사람용) — (kst_time, text)
    last_ts_kst = None

    def push_feed(txt: str):
        feed.append({"t": last_ts_kst, "text": txt})

    for idx, raw in enumerate(lines):
        line = raw.rstrip("\n")
        m = RE_TS.match(line)
        if m:
            last_ts_kst = _to_kst(m.group(1))

        # ── phase 마커(최신 우선) ──
        if RE_PLANNER_START.search(line):
            mm = RE_PLANNER_START.search(line)
            cur_iter = int(mm.group(1))     # 라인 순서 최신값(warm-start run 리셋 대응 — max 금지)
            phase, last_phase_idx = "planning", idx
            push_feed(f"Iteration {mm.group(1)} 계획 갱신 시작")
        if RE_PANEL_START.search(line):
            mm = RE_PANEL_START.search(line)
            it = int(mm.group(1))
            cur_iter = it
            phase, last_phase_idx = "panel", idx
            if it != panel_iter:
                panel_iter, panel_events, panel_consensus = it, [], []
                panel_final, minority = None, None
            focus = [int(x) for x in mm.group(2).split(",") if x.strip().isdigit()]
        m = RE_HETERO.search(line)
        if m:
            hetero_info = {
                "domains": [d.strip().strip("'") for d in m.group(1).split(",")],
                "model": real_model(m.group(2)),
                "url": m.group(3),
            }
        m = RE_PANEL_BACKEND.search(line)
        if m:
            backends[(int(m.group(1)), m.group(2))] = m.group(3)
        m = RE_PANEL_VERDICT.search(line)
        if m:
            it = int(m.group(1))
            phase, last_phase_idx = "panel", idx
            if it != panel_iter:
                panel_iter, panel_events, panel_consensus = it, [], []
                panel_final, minority = None, None
            cur_iter = it
            domain = m.group(2)
            bk = backends.get((it, domain), "")
            panel_events.append({
                "domain": domain, "domain_ko": DOMAIN_KO.get(domain, domain),
                "round": int(m.group(3)), "severity": m.group(4),
                "stance": m.group(5), "concern": m.group(6).strip()[:400],
                "model": real_model(bk) if bk else None,
                "t": last_ts_kst,
            })
        m = RE_CONSENSUS.search(line)
        if m:
            panel_consensus.append({
                "round": int(m.group(1)), "status": m.group(2),
                "high": int(m.group(3)), "medium": int(m.group(4)),
                "reason": m.group(5).strip()[:200],
            })
        m = RE_FANIN.search(line)
        if m:
            panel_final = {
                "approve": m.group(1) == "True", "merged": int(m.group(2)),
                "llm_calls": int(m.group(3)), "round": int(m.group(4)),
                "max_rounds": int(m.group(5)), "kind": "fanin",
            }
        if RE_CONSENSUS_OK.search(line) and panel_final:
            panel_final["result"] = "approve"
            push_feed(f"iter {panel_iter} 패널 합의: 승인 (round {panel_final['round']})")
        if RE_STALL.search(line) and panel_final:
            panel_final["result"] = "stall"
            push_feed(f"iter {panel_iter} 정체(stall) 종료 → 탐색 도킹 허용")
        m = RE_MINORITY.search(line)
        if m:
            minority = {"domain": m.group(1), "domain_ko": DOMAIN_KO.get(m.group(1), m.group(1)),
                        "severity": m.group(2), "reason": m.group(3).strip()[:300]}
        m = RE_PREREVIEW_DONE.search(line)
        if m:
            focus = [int(x) for x in m.group(2).split(",") if x.strip().isdigit()]
        m = RE_SLOTS.search(line)
        if m:
            phase, last_phase_idx = "slots", idx
            push_feed(f"iter {m.group(1)} 후보 {m.group(2)}개 (guided {m.group(3)}/intensify {m.group(4)}/random {m.group(5)})")

        # ── 도킹 ──
        m = RE_DOCK_PARALLEL.search(line)
        if m:
            # 새 도킹 배치 시작 → 이전(및 warm-start 이전 run) 결과 리셋, 이후 dock-flat만 수집
            phase, last_phase_idx = "docking", idx
            dock_total = int(m.group(1))
            dock_trials = int(m.group(3))
            dock_workers = int(m.group(4))
            dock_iter = cur_iter
            dock_nstruct = int(m.group(2))
            dock_results, stage2 = {}, {}
            stage2_hdr, qc_report, convergence = None, None, None
            gates, gate_total, eval_steps = {}, 0, {}
            dock_batch_open = True
            push_feed(f"iter {cur_iter} 도킹 시작: {dock_total}후보 × nstruct={m.group(2)} = {dock_trials} trials")
        m = RE_DOCK_FLAT.search(line)
        if m and dock_batch_open:
            dock_iter = int(m.group(1))
            dock_results[m.group(2)] = {"ddg": float(m.group(3)), "nconv": m.group(4)}
            phase, last_phase_idx = "docking", idx
        m = RE_STAGE2_HDR.search(line)
        if m and dock_batch_open:
            stage2_hdr = {"n": int(m.group(2)), "nstruct": int(m.group(3)), "trigger": float(m.group(4))}
            phase, last_phase_idx = "stage2", idx
        m = RE_STAGE2.search(line)
        if m and dock_batch_open:
            stage2[m.group(2)] = {
                "dg1": float(m.group(3)), "dg2": float(m.group(4)),
                "nconv": m.group(5), "nstruct": int(m.group(6)),
            }
            phase, last_phase_idx = "stage2", idx
        # ── 평가(evaluation) 단계 ──
        if dock_batch_open:
            m = RE_GATE_START.search(line)
            if m:
                gate_total = int(m.group(1))
                phase, last_phase_idx = "ranking", idx
            m = RE_GATE_LINE.search(line)
            if m:
                gates[int(m.group(1))] = {"name": m.group(2), "detail": m.group(3).strip()}
            m = RE_QC_REPORT.search(line)
            if m:
                qc_report = {"pass": int(m.group(1)), "total": int(m.group(2)), "pct": float(m.group(3))}
            m = RE_CONVERGENCE.search(line)
            if m:
                convergence = {"status": m.group(2), "p": float(m.group(3)), "cv": float(m.group(4))}
                eval_steps["convergence"] = {"detail": f"{m.group(2)} (p={m.group(3)}, CV={m.group(4)})"}
            m = RE_RCSB.search(line)
            if m:
                eval_steps["rcsb"] = {"detail": f"PDB 매칭 {m.group(1)}건"}
            m = RE_PHARMA.search(line)
            if m:
                eval_steps["pharma"] = {"detail": f"약리 enrich {m.group(1)}건"}
            m = RE_CLUSTER.search(line)
            if m:
                eval_steps["cluster"] = {"detail": f"{m.group(1)}개 → 클러스터 {m.group(2)}"}
            m = RE_BO.search(line)
            if m:
                eval_steps["bo"] = {"detail": f"다음 iter 제안 위치 [{m.group(1)}]"}
            m = RE_DASH.search(line)
            if m:
                eval_steps["dashboard"] = {"detail": f"대시보드 반영 {m.group(1)}건"}
        m = RE_TOP5.search(line)
        if m:
            phase, last_phase_idx = "ranking", idx
            top5 = [{"cand": mm.group(1), "ddg": float(mm.group(2))}
                    for mm in RE_TOP5_ITEM.finditer(m.group(1))]
        m = RE_GATE.search(line)
        if m:
            gate = {"pass": int(m.group(1)), "fail": int(m.group(2))}
        if "[Reporter]" in line and "리포트 생성 완료" in line:
            phase, last_phase_idx = "reporting", idx
        m = RE_BEST.search(line)
        if m:
            phase, last_phase_idx = "feedback", idx
            best = {"hc": int(m.group(1)), "seq": m.group(2), "dg": float(m.group(3))}
            push_feed(f"best 갱신: {m.group(2)} dg={m.group(3)}")
        m = RE_MMGBSA.search(line)
        if m:
            mmgbsa_hc = int(m.group(1))

    # 최근 활동만
    feed_tail = feed[-14:][::-1]

    dock_done = len(dock_results)
    dock_pct = round(100 * dock_done / dock_total) if dock_total else 0
    dock_list = sorted(
        ({"cand": c, **v, "stage2": stage2.get(c)} for c, v in dock_results.items()),
        key=lambda x: int(re.sub(r"\D", "", x["cand"]) or 0),
    )
    # stage2 진행: 헤더가 있으면 n건 중 done
    stage2_list = sorted(
        ({"cand": c, **v} for c, v in stage2.items()),
        key=lambda x: int(re.sub(r"\D", "", x["cand"]) or 0),
    )
    # QC 게이트 4종 리스트(번호순)
    gate_list = [{"num": k, **gates[k]} for k in sorted(gates)]
    # 평가 후속 단계(정의된 순서로)
    _STEP_ORDER = [
        ("convergence", "🔁 수렴 검정"), ("rcsb", "🗄 RCSB 대조"),
        ("pharma", "💊 약리 enrich"), ("cluster", "🧬 클러스터링"),
        ("bo", "📈 BO 제안"), ("dashboard", "📊 대시보드 반영"),
    ]
    eval_step_list = [
        {"key": k, "label": lbl, "done": k in eval_steps,
         "detail": eval_steps.get(k, {}).get("detail", "")}
        for k, lbl in _STEP_ORDER
    ]

    return {
        "iteration": cur_iter,
        "focus": focus,
        "phase": phase,
        "phase_label": PHASE_LABEL.get(phase, phase),
        "hetero": hetero_info,
        "panel": {
            "iter": panel_iter,
            "events": panel_events[-30:],
            "consensus": panel_consensus,
            "final": panel_final,
            "minority": minority,
        },
        "docking": {
            "iter": dock_iter, "total": dock_total, "done": dock_done,
            "pct": dock_pct, "trials": dock_trials, "workers": dock_workers,
            "nstruct": dock_nstruct, "results": dock_list, "gate": gate,
        },
        "stage2": {"header": stage2_hdr, "results": stage2_list},
        "eval": {
            "gate_total": gate_total,
            "gates": gate_list,
            "qc_report": qc_report,
            "convergence": convergence,
            "steps": eval_step_list,
        },
        "ranking": {"top5": top5},
        "best": best,
        "mmgbsa_hc": mmgbsa_hc,
        "feed": feed_tail,
    }


# ── 전체-로그 affinity 추세 (epoch별 학습 궤적) ────────────────────────────
_T_PLAN = re.compile(r"\[Planner\].*Iteration (\d+) 계획 갱신 시작")
_T_FLAT = re.compile(r"\[dock-flat\] iter(\d+)_cand\d+: ddG_median=([-\d.]+)")
_T_TOP = re.compile(r"Top-5 by ddG: \[(.*)\]")
_T_TOPITEM = re.compile(r"'iter\d+_cand\d+', '([-\d.]+)'")
_T_GATE = re.compile(r"게이트 완료: (\d+) 통과 / (\d+) 실패")
_T_FOCUS = re.compile(r"\[pre-review\] iter=(\d+) 완료.*final_focus=\[([^\]]*)\]")
_T_REJECT = re.compile(r"\[reject_hard\] iter=(\d+)")


def parse_full_trend() -> dict:
    """전체 provenance 로그를 훑어 epoch별 best/median ddG·통과율·focus·reject를 추출.

    iteration 번호는 run마다 리셋되므로 로그 순서 전역 인덱스(i)를 x축으로 쓴다.
    'best'는 그 epoch의 raw 도킹 최고치(min dock-flat 또는 Top-1)로, 게이트 통과와 무관하게
    "그 회차 도킹이 얼마나 강한 결합을 찾았나"를 보여준다(학습 신호). 통과율은 별도 계열.
    """
    epochs: list[dict] = []
    cur: dict | None = None
    last_ts: str | None = None

    def flush():
        nonlocal cur
        if cur and (cur["flats"] or cur["top1"] is not None):
            cur["best"] = round(min(cur["flats"]), 2) if cur["flats"] else cur["top1"]
            cur["median"] = round(statistics.median(cur["flats"]), 2) if cur["flats"] else None
            del cur["flats"]
            epochs.append(cur)
        cur = None

    if not PROV_LOG.exists():
        return {"epochs": [], "n_epochs": 0}
    with PROV_LOG.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = RE_TS.match(line)
            if m:
                last_ts = m.group(1)
            m = _T_PLAN.search(line)
            if m:
                flush()
                cur = {"iter": int(m.group(1)), "ts": last_ts, "flats": [],
                       "top1": None, "pass": None, "total": None, "focus": None, "reject": False}
                continue
            if cur is None:
                continue
            m = _T_FLAT.search(line)
            if m:
                cur["flats"].append(float(m.group(2)))
            m = _T_TOP.search(line)
            if m:
                items = _T_TOPITEM.findall(m.group(1))
                cur["top1"] = float(items[0]) if items else None
            m = _T_GATE.search(line)
            if m:
                cur["pass"], cur["total"] = int(m.group(1)), int(m.group(1)) + int(m.group(2))
            m = _T_FOCUS.search(line)
            if m:
                cur["focus"] = [int(x) for x in m.group(2).split(",") if x.strip().isdigit()]
            if _T_REJECT.search(line):
                cur["reject"] = True
        flush()

    # 전역 인덱스 + 누적 best(단조) + 롤링 평균(window 5)
    all_best = [e["best"] for e in epochs if e["best"] is not None]
    all_time_best = min(all_best) if all_best else None
    cum = 1e9
    for i, e in enumerate(epochs):
        e["i"] = i + 1
        if e["best"] is not None:
            cum = min(cum, e["best"])
        e["cum_best"] = round(cum, 2) if cum < 1e9 else None

    win = epochs[-TREND_WINDOW:]
    # 롤링 평균은 표시 윈도우 내에서 계산
    bests = [e["best"] for e in win]
    for idx, e in enumerate(win):
        lo = max(0, idx - 4)
        seg = [b for b in bests[lo:idx + 1] if b is not None]
        e["roll"] = round(sum(seg) / len(seg), 2) if seg else None

    return {
        "epochs": win,
        "n_epochs": len(epochs),
        "all_time_best": all_time_best,
        "window": len(win),
    }


# ── Silo A (de novo RFdiffusion/DiffPepBuilder) 파서 ────────────────────────
_SA_EPOCH = re.compile(r"=== Epoch (\d+) 시작(?: \(contigs='([^']*)', steps=(\d+)\))?")
_SA_ARM = re.compile(r"arm=(\w+): RFdiffusion=(\S+) \| DiffPepBuilder=(\S+)")
_SA_CUM = re.compile(r"이전 누적: (\d+)건 리더보드, (\d+)건 총 처리")
_SA_SUMMARY = re.compile(
    r"Epoch (\d+): (\d+)건 시도, 도킹=(\d+), 선택성=(\d+) \| 글로벌 best ddG=([-\d.]+), sel_margin=([-\d.]+)"
)
_SA_FEEDBACK = re.compile(r"피드백 플래너\(epoch=(\d+), stagnation=(\d+)\): \[(\w+)\] (.*)")
_SA_FBSTATUS = re.compile(r"피드백: (\S+) \((\d+)/(\d+)\) \[([^\]]*)\]")

_SA_STAGES = [
    ("rfdiff", "🧬 RFdiffusion 백본"), ("seq", "✍️ 서열 설계"),
    ("fold", "🔬 ESMFold 폴딩"), ("dock", "🧪 FlexPepDock 도킹"),
    ("select", "🎯 선택성 교차검증"), ("feedback", "🔁 피드백 플래너"),
]


def _count_flexpep_procs() -> int:
    n = 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                cmd = fh.read().decode("utf-8", errors="replace")
        except (OSError, IOError):
            continue
        if "flexpep_dock.py" in cmd:
            n += 1
    return n


def parse_silo_a() -> dict:
    """Silo A(de novo) 상태 — provenance 로그 tail + status.json + 리더보드."""
    out: dict = {"available": SILO_A_LOG.exists()}
    if not SILO_A_LOG.exists():
        return out

    # status.json (구조화 요약)
    status = {}
    if SILO_A_STATUS.exists():
        try:
            status = json.loads(SILO_A_STATUS.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            status = {}

    # provenance 로그 tail 파싱
    lines = SILO_A_LOG.read_bytes()[-120_000:].decode("utf-8", errors="replace").splitlines()
    epoch = contigs = steps = None
    arm = None
    cum_lb = cum_total = None
    summary = None
    feedback = None
    fb_status = None
    for line in lines:
        m = _SA_EPOCH.search(line)
        if m:
            epoch = int(m.group(1))
            if m.group(2) is not None:
                contigs, steps = m.group(2), int(m.group(3))
        m = _SA_ARM.search(line)
        if m:
            arm = {"mode": m.group(1), "rfdiff": m.group(2), "diffpep": m.group(3)}
        m = _SA_CUM.search(line)
        if m:
            cum_lb, cum_total = int(m.group(1)), int(m.group(2))
        m = _SA_SUMMARY.search(line)
        if m:
            summary = {"epoch": int(m.group(1)), "attempted": int(m.group(2)),
                       "docked": int(m.group(3)), "selected": int(m.group(4)),
                       "best_ddg": float(m.group(5)), "sel_margin": float(m.group(6))}
        m = _SA_FEEDBACK.search(line)
        if m:
            feedback = {"epoch": int(m.group(1)), "stagnation": int(m.group(2)),
                        "mode": m.group(3), "text": m.group(4).strip()[:400]}
        m = _SA_FBSTATUS.search(line)
        if m:
            fb_status = {"state": m.group(1), "n": int(m.group(2)),
                         "patience": int(m.group(3)), "ladder": m.group(4)}

    # 실행 단계 감지: flexpep_dock 프로세스 수 → 도킹 중
    n_dock = _count_flexpep_procs()
    stage = "dock" if n_dock > 0 else ("feedback" if feedback else "seq")

    # 리더보드 top
    lb_top = []
    if SILO_A_LB.exists():
        try:
            lb = json.loads(SILO_A_LB.read_text(encoding="utf-8"))
            entries = lb if isinstance(lb, list) else lb.get("entries") or lb.get("leaderboard") or []
            for e in entries[:10]:
                if isinstance(e, dict):
                    lb_top.append({
                        "sequence": e.get("sequence", ""),
                        "ddg": e.get("ddg"),
                        "sel_margin": e.get("selectivity_margin"),
                        "delta_margin": e.get("delta_margin"),
                        "cls": e.get("candidate_class", ""),
                        "src": e.get("mutation_source", ""),
                        "fail": e.get("fail_reason") or "",
                    })
        except Exception:  # noqa: BLE001
            pass

    mtime = SILO_A_STATUS.stat().st_mtime if SILO_A_STATUS.exists() else SILO_A_LOG.stat().st_mtime
    out.update({
        "running": status.get("stop_reason") == "running" or n_dock > 0,
        "total_candidates": status.get("total_candidates"),
        "best_ddg": status.get("best_ddg"),
        "best_sel_margin": status.get("best_selectivity_margin"),
        "elapsed_sec": status.get("elapsed_sec"),
        "gpu": status.get("cuda_visible_devices"),
        "updated_at": status.get("updated_at"),
        "log_age_sec": round(time.time() - mtime),
        "epoch": epoch, "contigs": contigs, "steps": steps, "arm": arm,
        "cum_lb": cum_lb, "cum_total": cum_total,
        "summary": summary, "feedback": feedback, "fb_status": fb_status,
        "n_dock_procs": n_dock, "stage": stage,
        "stages": [{"key": k, "label": lbl} for k, lbl in _SA_STAGES],
        "leaderboard": lb_top,
    })
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[live-tailer] 시작 — {PROV_LOG} → {OUT_FILE} (매 {POLL_SEC}s)")
    _tick = 0
    while True:
        try:
            lines = read_tail(PROV_LOG)
            state = parse(lines)
            # 30초마다 전체-로그 affinity 추세 재계산 (별도 파일)
            if _tick % TREND_EVERY == 0:
                try:
                    trend = parse_full_trend()
                    trend["generated_at"] = datetime.now(timezone.utc).astimezone(_KST).strftime("%Y-%m-%d %H:%M:%S KST")
                    ttmp = TREND_FILE.with_suffix(".json.tmp")
                    with ttmp.open("w", encoding="utf-8") as fh:
                        json.dump(trend, fh, ensure_ascii=False)
                    os.replace(ttmp, TREND_FILE)
                except Exception as te:  # noqa: BLE001
                    print(f"[live-tailer] 추세 경고: {te}")
            _tick += 1
            pid = find_engine_pid()
            mtime = PROV_LOG.stat().st_mtime if PROV_LOG.exists() else 0
            age = time.time() - mtime
            now_kst = datetime.now(timezone.utc).astimezone(_KST)
            state["engine"] = {
                "pid": pid,
                # PID 존재 = 활성(도킹 중엔 dock-flat 간격이 길어 로그 나이만으론 오탐).
                # PID 못 찾을 때만 로그 나이(5분)로 보조 판정.
                "alive": (pid is not None) or (age < 300),
                "log_age_sec": round(age),
                "uptime": proc_uptime(pid),
            }
            state["generated_at"] = now_kst.strftime("%Y-%m-%d %H:%M:%S KST")
            tmp = OUT_FILE.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(state, fh, ensure_ascii=False)
            os.replace(tmp, OUT_FILE)      # 원자적 교체 (폴링 중 부분읽기 방지)

            # Silo A 상태 (매 tick)
            try:
                sa = parse_silo_a()
                sa["generated_at"] = now_kst.strftime("%Y-%m-%d %H:%M:%S KST")
                satmp = SILO_A_OUT.with_suffix(".json.tmp")
                with satmp.open("w", encoding="utf-8") as fh:
                    json.dump(sa, fh, ensure_ascii=False)
                os.replace(satmp, SILO_A_OUT)
            except Exception as sae:  # noqa: BLE001
                print(f"[live-tailer] Silo A 경고: {sae}")
        except Exception as e:  # noqa: BLE001 — 데몬은 죽지 않아야 함
            print(f"[live-tailer] 경고: {e}")
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
