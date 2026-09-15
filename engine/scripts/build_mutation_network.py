#!/usr/bin/env python3
"""build_mutation_network.py — SST14 변이 관계 네트워크 JSON 생성.

읽기 전용: runs/pyrosetta_flow/global_selectivity_leaderboard.json
           runs/pyrosetta_flow/experiment_log.jsonl (선택, mutation_source)
출력: docs/mutation_network/network_data.json

노드:
  - native (AGCKNFFWKTFTSC) 1개
  - leaderboard top50 (완전 데이터: ddg, delta_margin, hc50, rank)
  - 합계 51개

엣지:
  - Hamming distance ≤ 2 (51개 노드 사이)
  - native→모든 엔트리도 포함 (Hamming 무관)

통계:
  - screened_seqs 973개 기반 위치별 변이 빈도 (별도 집계)

한계:
  - 명시적 parent→child lineage 없음 → Hamming 유사도로 재구성
  - delta_margin/hc50은 leaderboard top50에만 있음
  - screened_seqs 923개(top50 외)는 ddg만 보유
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

NATIVE = "AGCKNFFWKTFTSC"
REPO_ROOT = Path(__file__).parent.parent
LEADERBOARD_PATH = REPO_ROOT / "runs" / "pyrosetta_flow" / "global_selectivity_leaderboard.json"
EXPERIMENT_LOG_PATH = REPO_ROOT / "runs" / "pyrosetta_flow" / "experiment_log.jsonl"
DOCS_DIR = REPO_ROOT.parent.parent.parent.parent / "docs" / "mutation_network"

# docs 경로 절대 지정 (상대 경로 대신)
DOCS_DIR_ABS = Path("[LOCAL_PATH]")
OUTPUT_PATH = DOCS_DIR_ABS / "network_data.json"


def get_mutations(seq: str, native: str = NATIVE) -> list[dict]:
    """서열과 native의 변이 위치 목록 반환."""
    muts = []
    for i, (s, n) in enumerate(zip(seq, native)):
        if s != n:
            muts.append({"pos": i + 1, "from": n, "to": s, "label": f"{n}{i+1}{s}"})
    return muts


def hamming(a: str, b: str) -> int:
    """Hamming distance 계산 (동일 길이 가정)."""
    return sum(x != y for x, y in zip(a, b))


def load_leaderboard(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_mutation_sources(path: Path, screened_set: set[str]) -> dict[str, str]:
    """experiment_log에서 screened 서열의 mutation_source(최근) 수집."""
    sources: dict[str, str] = {}
    if not path.exists():
        return sources
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            seq = d.get("sequence", "")
            src = d.get("mutation_source", "")
            if seq in screened_set and src:
                sources[seq] = src  # 마지막 기록 우선
    return sources


def build_network(lb: dict, mut_sources: dict[str, str]) -> dict:
    """네트워크 JSON 구조 생성."""
    entries = lb["entries"]  # top50
    screened_seqs: list[str] = lb.get("screened_seqs", [])
    screened_set = set(screened_seqs)

    # ── 노드 생성 ──
    nodes: list[dict] = []

    # native 노드
    nodes.append({
        "id": "native",
        "sequence": NATIVE,
        "label": "Native",
        "is_native": True,
        "ddg": 0.0,
        "delta_margin": 0.0,
        "hc50": None,
        "hc50_vs_native": None,
        "more_toxic_than_native": False,
        "mutations": [],
        "mutation_labels": [],
        "n_mutations": 0,
        "rank": 0,
        "run_id": "native",
        "mutation_source": None,
        "group": "native",
    })

    # leaderboard top50 노드
    for rank, e in enumerate(entries, start=1):
        seq = e["sequence"]
        muts = get_mutations(seq)
        nodes.append({
            "id": seq,
            "sequence": seq,
            "label": seq,
            "is_native": False,
            "ddg": e.get("ddg", 0.0),
            "delta_margin": e.get("delta_margin", 0.0),
            "hc50": e.get("hc50"),
            "hc50_vs_native": e.get("hc50_vs_native"),
            "more_toxic_than_native": e.get("more_toxic_than_native", False),
            "mutations": muts,
            "mutation_labels": [m["label"] for m in muts],
            "n_mutations": len(muts),
            "rank": rank,
            "run_id": e.get("run_id", ""),
            "mutation_source": mut_sources.get(seq),
            "group": "top50",
        })

    # ── 엣지 생성 ──
    # 규칙 A: native → 모든 top50 (native 중심 방사형)
    # 규칙 B: top50 사이 Hamming ≤ 2
    edges: list[dict] = []
    edge_set: set[str] = set()

    node_ids = [n["id"] for n in nodes]
    node_map = {n["id"]: n for n in nodes}

    def add_edge(src: str, tgt: str, h: int, edge_type: str) -> None:
        eid = f"{min(src, tgt)}--{max(src, tgt)}"
        if eid not in edge_set:
            edge_set.add(eid)
            edges.append({
                "source": src,
                "target": tgt,
                "hamming": h,
                "weight": max(1, 3 - h),
                "type": edge_type,
            })

    for n in nodes:
        if n["is_native"]:
            continue
        h = hamming(n["sequence"], NATIVE)
        add_edge("native", n["id"], h, "native_to_entry")

    non_native = [n for n in nodes if not n["is_native"]]
    for i in range(len(non_native)):
        for j in range(i + 1, len(non_native)):
            si = non_native[i]["sequence"]
            sj = non_native[j]["sequence"]
            h = hamming(si, sj)
            if h <= 2:
                add_edge(non_native[i]["id"], non_native[j]["id"], h, "similarity")

    # ── 위치별 변이 빈도 (screened_seqs 973개 전체 기준) ──
    pos_freq: Counter[int] = Counter()
    mut_freq: Counter[str] = Counter()
    aa_at_pos: dict[int, Counter] = {}

    for seq in screened_seqs:
        if len(seq) != len(NATIVE):
            continue
        for i, (s, n) in enumerate(zip(seq, NATIVE)):
            if s != n:
                pos_freq[i + 1] += 1
                mut_freq[f"{n}{i+1}{s}"] += 1
                aa_at_pos.setdefault(i + 1, Counter())[s] += 1

    # pos_aa_top: 위치별 상위 AA
    pos_aa_top = {
        str(pos): [{"aa": aa, "count": cnt} for aa, cnt in ctr.most_common(5)]
        for pos, ctr in aa_at_pos.items()
    }

    return {
        "meta": {
            "native": NATIVE,
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "n_leaderboard": len(entries),
            "n_screened_total": len(screened_seqs),
            "edge_rules": [
                "native→entry: native 중심 방사형 (Hamming 무관)",
                "entry→entry: Hamming distance ≤ 2",
            ],
            "lineage_note": "명시적 parent→child 없음. Hamming distance 기반 유사도 재구성.",
            "data_completeness": {
                "ddg": "top50 전체",
                "delta_margin": "top50 전체",
                "hc50": "top50 전체",
                "mutation_source": "일부(mutation_source 기록 이후 8159건)",
            },
            "drop_note": (
                f"screened_seqs {len(screened_seqs)}개 중 top50 외 "
                f"{len(screened_seqs)-len(entries)}개는 "
                "위치별 빈도 통계에만 사용 (ddg만 보유, delta_margin/hc50 없어 노드 제외)."
            ),
        },
        "nodes": nodes,
        "edges": edges,
        "pos_freq": {str(k): v for k, v in sorted(pos_freq.items())},
        "pos_aa_top": pos_aa_top,
        "mut_freq": dict(mut_freq.most_common(30)),
    }


def main() -> None:
    print(f"[build_mutation_network] 로딩: {LEADERBOARD_PATH}")
    if not LEADERBOARD_PATH.exists():
        print(f"ERROR: {LEADERBOARD_PATH} 없음", file=sys.stderr)
        sys.exit(1)

    lb = load_leaderboard(LEADERBOARD_PATH)
    screened_set = set(lb.get("screened_seqs", []))

    print(f"[build_mutation_network] mutation_source 로딩: {EXPERIMENT_LOG_PATH}")
    mut_sources = load_mutation_sources(EXPERIMENT_LOG_PATH, screened_set)
    print(f"  → mutation_source 보유 서열: {len(mut_sources)}개")

    net = build_network(lb, mut_sources)

    DOCS_DIR_ABS.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(net, f, ensure_ascii=False, indent=2)

    print(f"[build_mutation_network] 완료:")
    print(f"  노드: {net['meta']['n_nodes']}")
    print(f"  엣지: {net['meta']['n_edges']}")
    print(f"  출력: {OUTPUT_PATH}")
    print(f"  lineage 한계: {net['meta']['lineage_note']}")
    print(f"  드롭 명시: {net['meta']['drop_note']}")


if __name__ == "__main__":
    main()
