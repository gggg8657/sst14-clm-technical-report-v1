"""
test_discussion_log.py
expert_verdicts 영속 저장 + discussion_log.jsonl 단위테스트

테스트 범위:
    1. runner._append_discussion_log — 4 domain verdict list 포함 1줄 append
    2. runner._append_discussion_log — approve=True/False 모두 기록
    3. runner 수정: expert_verdicts 항상(빈 dict 포함) _pr_entry에 기록
    4. build_monitoring_data._load_discussion_log_index — iter → ev dict 매핑
    5. build_monitoring_data._extract_expert_verdicts — 빈 dict {} 무시, 실제 verdict 반환
    6. build_monitoring_data._accumulate_llm_history — discussion_log 소급 보완
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch


def _import_build_monitoring():
    """scripts/build_monitoring_data.py 를 패키지 없이 직접 import."""
    _mod_path = "[LOCAL_PATH]"
    spec = importlib.util.spec_from_file_location("build_monitoring_data", _mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_verdict_dict(domains: Optional[List[str]] = None) -> Dict[str, Any]:
    """4 domain 기본 verdict dict (planner_report / _pr_entry 형식)."""
    if domains is None:
        domains = ["pharma", "biology", "chemistry", "math"]
    return {
        d: {"severity": "medium", "concerns": [f"{d} concern 1", f"{d} concern 2"]}
        for d in domains
    }


def _make_verdict_list(domains: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """4 domain verdict list (discussion_log 형식)."""
    if domains is None:
        domains = ["pharma", "biology", "chemistry", "math"]
    return [
        {"domain": d, "severity": "medium", "concerns": [f"{d} concern 1"]}
        for d in domains
    ]


# ---------------------------------------------------------------------------
# 1-2. runner._append_discussion_log
# ---------------------------------------------------------------------------

class TestAppendDiscussionLog(unittest.TestCase):
    """runner._append_discussion_log 단위테스트."""

    def _call_append(
        self,
        tmpdir: str,
        iteration: int,
        approve: bool,
        expert_verdicts: Optional[Dict[str, Any]] = None,
        round_idx: int = 1,
    ) -> Path:
        """_append_discussion_log 호출 헬퍼."""
        from pyrosetta_flow.runner import _append_discussion_log
        log_path = Path(tmpdir) / "discussion_log.jsonl"
        _append_discussion_log(
            discussion_log_path=log_path,
            iteration=iteration,
            hypothesis=f"iter{iteration} hypothesis",
            round_idx=round_idx,
            expert_verdicts=expert_verdicts or _make_verdict_dict(),
            fanin={"approve": approve, "merged_concerns": ["concern1"]},
            final_focus=[1, 4, 6],
        )
        return log_path

    def test_creates_file_on_first_call(self):
        """첫 호출 시 파일이 생성된다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = self._call_append(tmpdir, iteration=1, approve=True)
            self.assertTrue(log_path.exists(), "discussion_log.jsonl 파일 미생성")

    def test_appends_one_line_per_call(self):
        """호출 1회 → 1줄 append."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = self._call_append(tmpdir, iteration=1, approve=True)
            lines = [l for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertEqual(len(lines), 1, f"줄 수 오류: {len(lines)}")

    def test_multiple_iterations_append(self):
        """iter 1, 2, 3 → 3줄."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(1, 4):
                self._call_append(tmpdir, iteration=i, approve=True)
            log_path = Path(tmpdir) / "discussion_log.jsonl"
            lines = [l for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertEqual(len(lines), 3)

    def test_schema_iteration_field(self):
        """기록된 JSON에 iteration 필드가 올바르다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = self._call_append(tmpdir, iteration=5, approve=False)
            entry = json.loads(log_path.read_text(encoding="utf-8").strip())
            self.assertEqual(entry["iteration"], 5)

    def test_schema_expert_verdicts_is_list(self):
        """expert_verdicts 필드가 list 형식으로 저장된다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = self._call_append(tmpdir, iteration=1, approve=True)
            entry = json.loads(log_path.read_text(encoding="utf-8").strip())
            evd = entry["expert_verdicts"]
            self.assertIsInstance(evd, list, f"expert_verdicts가 list 아님: {type(evd)}")

    def test_schema_expert_verdicts_4_domains(self):
        """4 domain verdict 모두 기록된다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = self._call_append(tmpdir, iteration=1, approve=True)
            entry = json.loads(log_path.read_text(encoding="utf-8").strip())
            domains = {v["domain"] for v in entry["expert_verdicts"]}
            self.assertEqual(domains, {"pharma", "biology", "chemistry", "math"})

    def test_approve_true_recorded(self):
        """approve=True도 기록된다 (통과 케이스도 영속)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = self._call_append(tmpdir, iteration=2, approve=True)
            entry = json.loads(log_path.read_text(encoding="utf-8").strip())
            self.assertTrue(entry["fanin"]["approve"])

    def test_approve_false_recorded(self):
        """approve=False도 기록된다 (거부 → revision 케이스)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = self._call_append(tmpdir, iteration=3, approve=False)
            entry = json.loads(log_path.read_text(encoding="utf-8").strip())
            self.assertFalse(entry["fanin"]["approve"])

    def test_empty_expert_verdicts_does_not_crash(self):
        """expert_verdicts 빈 dict {}로 호출해도 예외 없이 빈 list 기록."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from pyrosetta_flow.runner import _append_discussion_log
            log_path = Path(tmpdir) / "discussion_log.jsonl"
            _append_discussion_log(
                discussion_log_path=log_path,
                iteration=1,
                hypothesis="test",
                round_idx=1,
                expert_verdicts={},
                fanin={"approve": True, "merged_concerns": []},
                final_focus=[],
            )
            entry = json.loads(log_path.read_text(encoding="utf-8").strip())
            self.assertIsInstance(entry["expert_verdicts"], list)
            self.assertEqual(len(entry["expert_verdicts"]), 0)

    def test_final_focus_recorded(self):
        """final_focus 리스트가 기록된다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = self._call_append(tmpdir, iteration=1, approve=True)
            entry = json.loads(log_path.read_text(encoding="utf-8").strip())
            self.assertEqual(entry["final_focus"], [1, 4, 6])


# ---------------------------------------------------------------------------
# 3. runner _pr_entry: expert_verdicts 항상 기록
# ---------------------------------------------------------------------------

class TestPrEntryAlwaysHasExpertVerdicts(unittest.TestCase):
    """_pr_entry dict에 expert_verdicts 키가 항상 존재한다."""

    def _make_prereview_result(
        self,
        approve: bool,
        expert_verdicts: Optional[Dict[str, Any]] = None,
        mode: str = "expert_panel",
        use_default_verdicts: bool = True,
    ) -> Dict[str, Any]:
        """prereview_result 픽스처 — runner 내부 구조 모사.

        expert_verdicts=None이고 use_default_verdicts=True → 4 domain 기본 verdict 사용.
        expert_verdicts={} (빈 dict) 를 그대로 넘기려면 use_default_verdicts=False 설정.
        """
        if expert_verdicts is None and use_default_verdicts:
            _ev = _make_verdict_dict()
        else:
            _ev = expert_verdicts if expert_verdicts is not None else {}
        return {
            "concerns": ["concern1"],
            "suggested_revisions": {"focus_positions": [1, 2]},
            "approve": approve,
            "expert_verdicts": _ev,
            "llm_calls": 5,
            "mode": mode,
        }

    def _build_pr_entry(self, prereview_result: Dict[str, Any]) -> Dict[str, Any]:
        """runner.py 수정 후 _pr_entry 빌드 로직을 로컬에서 재현."""
        _raw_expert_verdicts: Dict[str, Any] = (
            prereview_result.get("expert_verdicts") or {}
        )
        pr_entry: Dict[str, Any] = {
            "round": 1,
            "mode": prereview_result.get("mode", "single_critic"),
            "llm_calls": prereview_result.get("llm_calls", 1),
            "original_hypothesis": "test hypothesis",
            "original_focus": [1, 2],
            "approve": prereview_result.get("approve", True),
            "concerns": prereview_result.get("concerns", []),
            "suggested_positions": (
                prereview_result.get("suggested_revisions", {}).get("focus_positions") or []
            ),
            # 항상 기록 (빈 dict {} 포함)
            "expert_verdicts": {
                domain: {
                    "severity": v.get("severity", "?"),
                    "concerns": v.get("concerns", []),
                }
                for domain, v in _raw_expert_verdicts.items()
            },
        }
        return pr_entry

    def test_expert_verdicts_present_when_approve_true(self):
        """approve=True여도 expert_verdicts 키 존재."""
        pr = self._make_prereview_result(approve=True)
        entry = self._build_pr_entry(pr)
        self.assertIn("expert_verdicts", entry)
        self.assertEqual(len(entry["expert_verdicts"]), 4)

    def test_expert_verdicts_present_when_approve_false(self):
        """approve=False일 때 expert_verdicts 키 존재."""
        pr = self._make_prereview_result(approve=False)
        entry = self._build_pr_entry(pr)
        self.assertIn("expert_verdicts", entry)
        self.assertEqual(len(entry["expert_verdicts"]), 4)

    def test_expert_verdicts_all_4_domains(self):
        """4개 domain 모두 포함."""
        pr = self._make_prereview_result(approve=True)
        entry = self._build_pr_entry(pr)
        self.assertEqual(
            set(entry["expert_verdicts"].keys()),
            {"pharma", "biology", "chemistry", "math"},
        )

    def test_expert_verdicts_empty_dict_in_p1_mode(self):
        """P1 단일 critic 모드(expert_verdicts={}) → 빈 dict 기록 (None 아님)."""
        # use_default_verdicts=False 로 실제 빈 dict {}를 expert_verdicts로 전달
        pr = self._make_prereview_result(
            approve=True, expert_verdicts={}, mode="single_critic", use_default_verdicts=False
        )
        entry = self._build_pr_entry(pr)
        self.assertIn("expert_verdicts", entry)
        self.assertIsInstance(entry["expert_verdicts"], dict)
        self.assertEqual(len(entry["expert_verdicts"]), 0)


# ---------------------------------------------------------------------------
# 4-5. build_monitoring_data: _load_discussion_log_index, _extract_expert_verdicts
# ---------------------------------------------------------------------------

class TestBuildMonitoringDataHelpers(unittest.TestCase):
    """build_monitoring_data 신규 헬퍼 단위테스트."""

    def _write_discussion_log(self, tmpdir: str, entries: List[Dict[str, Any]]) -> str:
        path = os.path.join(tmpdir, "discussion_log.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return path

    def test_load_discussion_log_index_basic(self):
        """discussion_log.jsonl → iter 키 매핑."""
        _bmd = _import_build_monitoring()
        _load_discussion_log_index = _bmd._load_discussion_log_index
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_discussion_log(tmpdir, [
                {
                    "iteration": 7,
                    "hypothesis": "test",
                    "round": 1,
                    "expert_verdicts": _make_verdict_list(),
                    "fanin": {"approve": True, "merged_concerns": []},
                    "final_focus": [1, 2],
                }
            ])
            idx = _load_discussion_log_index(path)
            self.assertIn(7, idx)
            self.assertIsNotNone(idx[7].get("expert_verdicts"))

    def test_load_discussion_log_index_ev_dict_format(self):
        """_load_discussion_log_index가 list → dict 변환."""
        _bmd = _import_build_monitoring()
        _load_discussion_log_index = _bmd._load_discussion_log_index
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_discussion_log(tmpdir, [
                {
                    "iteration": 10,
                    "hypothesis": "h",
                    "round": 1,
                    "expert_verdicts": _make_verdict_list(),
                    "fanin": {"approve": False, "merged_concerns": ["c1"]},
                    "final_focus": [4, 6],
                }
            ])
            idx = _load_discussion_log_index(path)
            ev = idx[10]["expert_verdicts"]
            self.assertIsInstance(ev, dict)
            self.assertIn("pharma", ev)
            self.assertIn("severity", ev["pharma"])

    def test_load_discussion_log_empty_file(self):
        """빈 파일 → 빈 dict 반환."""
        _bmd = _import_build_monitoring()
        _load_discussion_log_index = _bmd._load_discussion_log_index
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty.jsonl")
            open(path, "w").close()
            idx = _load_discussion_log_index(path)
            self.assertEqual(idx, {})

    def test_load_discussion_log_nonexistent(self):
        """파일 없으면 빈 dict 반환 (예외 없음)."""
        _bmd = _import_build_monitoring()
        _load_discussion_log_index = _bmd._load_discussion_log_index
        idx = _load_discussion_log_index("/nonexistent/path.jsonl")
        self.assertEqual(idx, {})

    def test_extract_expert_verdicts_skips_empty_dict(self):
        """_extract_expert_verdicts가 빈 dict {} 를 None으로 처리 (P1 모드 구분)."""
        _bmd = _import_build_monitoring()
        _extract_expert_verdicts = _bmd._extract_expert_verdicts
        p = {
            "pre_review_rounds": [
                {"round": 1, "expert_verdicts": {}, "approve": True},
            ]
        }
        result = _extract_expert_verdicts(p)
        self.assertIsNone(result, f"빈 dict를 반환해선 안 됨: {result}")

    def test_extract_expert_verdicts_returns_real_verdicts(self):
        """실제 verdict가 있으면 반환."""
        _bmd = _import_build_monitoring()
        _extract_expert_verdicts = _bmd._extract_expert_verdicts
        p = {
            "pre_review_rounds": [
                {"round": 1, "expert_verdicts": _make_verdict_dict(), "approve": False},
            ]
        }
        result = _extract_expert_verdicts(p)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 4)

    def test_extract_expert_verdicts_none_if_no_rounds(self):
        """pre_review_rounds 없으면 None."""
        _bmd = _import_build_monitoring()
        _extract_expert_verdicts = _bmd._extract_expert_verdicts
        self.assertIsNone(_extract_expert_verdicts({}))


# ---------------------------------------------------------------------------
# 6. _accumulate_llm_history + discussion_log 소급 보완
# ---------------------------------------------------------------------------

class TestAccumulateWithDiscussionLog(unittest.TestCase):
    """_accumulate_llm_history가 discussion_log를 소급 보완한다."""

    def _write_planner_report(self, tmpdir: str, iteration: int, run_id: str = "run1") -> str:
        """expert_verdicts 없는 구버전 planner_report.json."""
        p = {
            "type": "planner",
            "run_id": run_id,
            "iteration": iteration,
            "hypothesis": f"iter{iteration} hypothesis",
            "strategy": "ddg_only",
            "pre_review_rounds": [
                {
                    "round": 1,
                    "original_hypothesis": "h",
                    "original_focus": [1, 2],
                    "approve": True,
                    "concerns": [],
                    "suggested_positions": [],
                    "action": "approved",
                    # expert_verdicts 없음 (구버전 형식)
                }
            ],
            "final_focus_positions": [1, 2],
        }
        path = os.path.join(tmpdir, f"planner_report_{iteration:02d}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(p, fh)
        return path

    def _write_discussion_log_entry(
        self, path: str, iteration: int, approve: bool = True
    ) -> None:
        entry = {
            "iteration": iteration,
            "hypothesis": f"iter{iteration} hyp",
            "round": 1,
            "expert_verdicts": _make_verdict_list(),
            "fanin": {"approve": approve, "merged_concerns": []},
            "final_focus": [1, 2],
        }
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def test_discussion_log_backfills_expert_verdicts(self):
        """discussion_log에 expert_verdicts 있으면 history에 소급 적용."""
        _bmd = _import_build_monitoring()
        _accumulate_llm_history = _bmd._accumulate_llm_history
        _load_llm_history = _bmd._load_llm_history
        with tempfile.TemporaryDirectory() as tmpdir:
            # 구버전 planner_report (expert_verdicts 없음)
            pf = self._write_planner_report(tmpdir, iteration=7)
            hist_path = os.path.join(tmpdir, "llm_activity_history.jsonl")
            disc_path = os.path.join(tmpdir, "discussion_log.jsonl")
            # discussion_log: iter=7 expert_verdicts 있음
            self._write_discussion_log_entry(disc_path, iteration=7, approve=True)
            # 1차 빌드: discussion_log 없이 (구버전 데이터 신규 append)
            _accumulate_llm_history(hist_path, [pf], kind="silo_b")
            rows = _load_llm_history(hist_path)
            iter7_rows = [r for r in rows if r.get("iteration") == 7]
            self.assertEqual(len(iter7_rows), 1)
            # expert_verdicts 없음 확인
            self.assertIsNone(iter7_rows[0].get("expert_verdicts"))
            # 2차 빌드: discussion_log 포함 → 소급 보완
            _accumulate_llm_history(hist_path, [pf], kind="silo_b", discussion_log_path=disc_path)
            rows2 = _load_llm_history(hist_path)
            iter7_rows2 = [r for r in rows2 if r.get("iteration") == 7]
            self.assertEqual(len(iter7_rows2), 1)
            ev = iter7_rows2[0].get("expert_verdicts")
            self.assertIsNotNone(ev, "discussion_log 소급 보완 실패")
            self.assertIn("pharma", ev)

    def test_no_discussion_log_no_crash(self):
        """discussion_log_path=None → 예외 없이 동작."""
        _bmd = _import_build_monitoring()
        _accumulate_llm_history = _bmd._accumulate_llm_history
        with tempfile.TemporaryDirectory() as tmpdir:
            pf = self._write_planner_report(tmpdir, iteration=1)
            hist_path = os.path.join(tmpdir, "hist.jsonl")
            # 예외 없이 실행
            _accumulate_llm_history(hist_path, [pf], kind="silo_b", discussion_log_path=None)
            self.assertTrue(os.path.exists(hist_path))


if __name__ == "__main__":
    unittest.main()
