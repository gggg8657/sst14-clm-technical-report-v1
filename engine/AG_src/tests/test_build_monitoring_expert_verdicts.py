"""
test_build_monitoring_expert_verdicts.py
P2 관측성 보완: build_monitoring_data.py가 expert_verdicts/mode/pre_review_rounds를
llm_activity 누적 이력 row에 올바르게 기록하는지 검증.

커버 항목:
  1. expert_verdicts 있는 planner_report → row에 expert_verdicts 포함
  2. pre_review_mode/pre_review_rounds/pre_review_llm_calls 기록
  3. expert_verdicts 없는 경우 → None으로 안전 처리
  4. Silo A 행은 expert_verdicts=None
  5. py_compile / import 안전
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
_ROOT = "[LOCAL_PATH]"
_REPO = f"{_ROOT}/AgenticAI4SCIENCE_pyrosetta_track/repos/ai4sci-kaeri"
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Helper: planner_report fixture
# ---------------------------------------------------------------------------

def _make_planner_report(
    iteration: int = 1,
    run_id: str = "test_run",
    with_expert_verdicts: bool = True,
    approve: bool = False,
    mode: str = "expert_panel",
    pre_review_llm_calls: int = 5,
) -> Dict[str, Any]:
    """테스트용 planner_report dict 생성."""
    ev: Optional[Dict[str, Any]] = None
    if with_expert_verdicts:
        ev = {
            "pharma": {"severity": "medium", "concerns": ["PK 우려"]},
            "biology": {"severity": "medium", "concerns": ["SSTR2 선택성 불충분"]},
            "chemistry": {"severity": "low", "concerns": []},
            "math": {"severity": "high", "concerns": ["국소최적 위험"]},
        }
    pre_review_rounds = []
    if ev is not None or mode == "single_critic":
        round_entry: Dict[str, Any] = {
            "round": 1,
            "mode": mode,
            "llm_calls": pre_review_llm_calls,
            "approve": approve,
            "concerns": ["test_concern"],
            "action": "revised" if not approve else "approved",
        }
        if ev:
            round_entry["expert_verdicts"] = ev
        pre_review_rounds.append(round_entry)

    report: Dict[str, Any] = {
        "type": "planner",
        "iteration": iteration,
        "run_id": run_id,
        "timestamp": "2026-06-23T06:00:00+00:00",
        "hypothesis": f"Test hypothesis iter {iteration}",
        "strategy": "ddg_plus_constraints",
        "previous_best_ddg": -25.0,
        "pre_review_mode": mode,
        "pre_review_llm_calls": pre_review_llm_calls,
        "final_focus_positions": [1, 2, 5],
    }
    if pre_review_rounds:
        report["pre_review_rounds"] = pre_review_rounds
    return report


# ---------------------------------------------------------------------------
# _accumulate_llm_history 임포트 (scripts/build_monitoring_data.py)
# ---------------------------------------------------------------------------

def _import_accumulate() -> Any:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_monitoring_data",
        f"{_ROOT}/scripts/build_monitoring_data.py",
    )
    mod = importlib.util.load_from_spec(spec)  # type: ignore[attr-defined]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _load_module():
    """build_monitoring_data 모듈 동적 로드."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bmd",
        f"{_ROOT}/scripts/build_monitoring_data.py",
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# 테스트
# ---------------------------------------------------------------------------

class TestBuildMonitoringExpertVerdicts(unittest.TestCase):
    """build_monitoring_data._accumulate_llm_history의 P2 필드 기록 검증."""

    def setUp(self):
        self.mod = _load_module()

    def _run_accumulate(
        self,
        planner_reports: List[Dict[str, Any]],
        kind: str = "silo_b",
    ) -> List[Dict[str, Any]]:
        """임시 디렉토리에 planner_report 파일을 쓰고 _accumulate_llm_history 실행 후 row 반환."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pf_list = []
            for idx, report in enumerate(planner_reports):
                pf = os.path.join(tmpdir, f"planner_report_{idx:02d}.json")
                with open(pf, "w", encoding="utf-8") as f:
                    json.dump(report, f, ensure_ascii=False)
                pf_list.append(pf)
            hist_path = os.path.join(tmpdir, "llm_activity_history.jsonl")
            self.mod._accumulate_llm_history(hist_path, pf_list, kind=kind)
            return self.mod._load_llm_history(hist_path)

    def test_expert_verdicts_written_to_row(self):
        """expert_verdicts가 있는 planner_report → row에 expert_verdicts 포함."""
        report = _make_planner_report(iteration=1, with_expert_verdicts=True, approve=False)
        rows = self._run_accumulate([report])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertIn("expert_verdicts", row, "expert_verdicts 키 누락")
        self.assertIsNotNone(row["expert_verdicts"])
        ev = row["expert_verdicts"]
        for domain in ("pharma", "biology", "chemistry", "math"):
            self.assertIn(domain, ev, f"도메인 '{domain}' 누락")

    def test_pre_review_mode_written(self):
        """pre_review_mode가 row에 기록된다."""
        report = _make_planner_report(mode="expert_panel")
        rows = self._run_accumulate([report])
        self.assertEqual(rows[0].get("pre_review_mode"), "expert_panel")

    def test_pre_review_rounds_count_written(self):
        """pre_review_rounds(라운드 수)가 row에 기록된다."""
        report = _make_planner_report(iteration=2, with_expert_verdicts=True, approve=False)
        rows = self._run_accumulate([report])
        self.assertEqual(rows[0].get("pre_review_rounds"), 1)

    def test_pre_review_llm_calls_written(self):
        """pre_review_llm_calls가 row에 기록된다."""
        report = _make_planner_report(pre_review_llm_calls=5)
        rows = self._run_accumulate([report])
        self.assertEqual(rows[0].get("pre_review_llm_calls"), 5)

    def test_final_focus_positions_written(self):
        """final_focus_positions가 row에 기록된다."""
        report = _make_planner_report(iteration=3)
        rows = self._run_accumulate([report])
        self.assertEqual(rows[0].get("final_focus_positions"), [1, 2, 5])

    def test_no_expert_verdicts_none_safe(self):
        """expert_verdicts 없는 planner_report → row에 expert_verdicts=None."""
        report = _make_planner_report(with_expert_verdicts=False, mode="single_critic")
        rows = self._run_accumulate([report])
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0].get("expert_verdicts"), "expert_verdicts 없으면 None 기대")

    def test_no_pre_review_rounds_none_safe(self):
        """pre_review_rounds 없는 planner_report → pre_review_rounds=None."""
        report: Dict[str, Any] = {
            "type": "planner", "iteration": 99, "run_id": "no_prereview",
            "hypothesis": "bare hypothesis", "strategy": "ddg_only",
            "previous_best_ddg": -10.0,
        }
        rows = self._run_accumulate([report])
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0].get("pre_review_rounds"))
        self.assertIsNone(rows[0].get("expert_verdicts"))

    def test_silo_a_row_expert_verdicts_none(self):
        """Silo A 행은 expert_verdicts=None."""
        # Silo A planner에는 epoch 기반 키
        report = {"epoch": 5, "hypothesis": "silo-a test", "adjustment_type": "warmup"}
        rows = self._run_accumulate([report], kind="silo_a")
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0].get("expert_verdicts"))

    def test_multiple_rounds_uses_last_expert_verdicts(self):
        """여러 pre_review 라운드 중 마지막 round의 expert_verdicts 사용."""
        ev_round1 = {
            "pharma": {"severity": "high", "concerns": ["round1 concern"]},
            "biology": {"severity": "medium", "concerns": []},
            "chemistry": {"severity": "low", "concerns": []},
            "math": {"severity": "low", "concerns": []},
        }
        ev_round2 = {
            "pharma": {"severity": "low", "concerns": []},
            "biology": {"severity": "low", "concerns": []},
            "chemistry": {"severity": "low", "concerns": []},
            "math": {"severity": "low", "concerns": ["round2 concern"]},
        }
        report: Dict[str, Any] = {
            "type": "planner", "iteration": 5, "run_id": "multi_round",
            "hypothesis": "two-round test", "strategy": "selectivity",
            "previous_best_ddg": -30.0,
            "pre_review_mode": "expert_panel",
            "pre_review_llm_calls": 10,
            "final_focus_positions": [1, 5],
            "pre_review_rounds": [
                {"round": 1, "mode": "expert_panel", "approve": False,
                 "expert_verdicts": ev_round1, "action": "revised"},
                {"round": 2, "mode": "expert_panel", "approve": True,
                 "expert_verdicts": ev_round2, "action": "approved"},
            ],
        }
        rows = self._run_accumulate([report])
        self.assertEqual(len(rows), 1)
        ev = rows[0]["expert_verdicts"]
        self.assertIsNotNone(ev)
        # 마지막 라운드(round2)에는 math에 'round2 concern'
        self.assertIn("round2 concern", ev.get("math", {}).get("concerns", []))

    def test_dedup_prevents_duplicate_rows(self):
        """동일 (run_id, iteration) 조합은 중복 추가되지 않는다."""
        report = _make_planner_report(iteration=1, run_id="same_run")
        rows = self._run_accumulate([report, report])
        self.assertEqual(len(rows), 1, "중복 row 허용됨 — dedup 미작동")

    def test_upsert_updates_existing_row_without_expert_verdicts(self):
        """기존 항목에 expert_verdicts=None이고 새 파일에 있으면 소급 업데이트된다."""
        # 1단계: expert_verdicts 없는 항목 먼저 추가
        report_no_ev = _make_planner_report(iteration=5, run_id="upsert_run", with_expert_verdicts=False)
        rows = self._run_accumulate([report_no_ev])
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0].get("expert_verdicts"))

        # 2단계: 같은 (run_id, iteration)에 expert_verdicts 있는 항목으로 업데이트
        report_with_ev = _make_planner_report(iteration=5, run_id="upsert_run", with_expert_verdicts=True)
        # 같은 hist_path에 다시 실행 — tempdir 대신 직접 구현
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            hist_path = os.path.join(tmpdir, "llm_activity_history.jsonl")
            # 1단계: no_ev 먼저
            pf1 = os.path.join(tmpdir, "report_01.json")
            with open(pf1, "w") as f:
                json.dump(report_no_ev, f, ensure_ascii=False)
            self.mod._accumulate_llm_history(hist_path, [pf1], kind="silo_b")
            rows1 = self.mod._load_llm_history(hist_path)
            self.assertIsNone(rows1[0].get("expert_verdicts"), "1단계: ev 없어야 함")

            # 2단계: with_ev로 재실행 (다른 파일 경로, 동일 run_id/iter)
            pf2 = os.path.join(tmpdir, "report_02.json")
            with open(pf2, "w") as f:
                json.dump(report_with_ev, f, ensure_ascii=False)
            self.mod._accumulate_llm_history(hist_path, [pf2], kind="silo_b")
            rows2 = self.mod._load_llm_history(hist_path)
            self.assertEqual(len(rows2), 1, "upsert 후 행 수는 1이어야 함")
            self.assertIsNotNone(rows2[0].get("expert_verdicts"), "upsert 후 ev 있어야 함")


# ---------------------------------------------------------------------------
# import/compile 안전 테스트
# ---------------------------------------------------------------------------

class TestBuildMonitoringImportSafety(unittest.TestCase):
    def test_py_compile(self):
        """build_monitoring_data.py py_compile 통과."""
        import py_compile
        py_compile.compile(f"{_ROOT}/scripts/build_monitoring_data.py", doraise=True)

    def test_module_importable(self):
        """build_monitoring_data 모듈 동적 로드 성공."""
        mod = _load_module()
        self.assertTrue(hasattr(mod, "_accumulate_llm_history"))
        self.assertTrue(hasattr(mod, "_load_llm_history"))
        self.assertTrue(hasattr(mod, "build_silo_b"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
