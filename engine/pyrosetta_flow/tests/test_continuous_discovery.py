"""무한 발굴 엔진 단위 테스트 (도킹 없이 로직만) — 2026-06-10."""
import json
from pathlib import Path

from pyrosetta_flow.global_leaderboard import GlobalSelectivityLeaderboard, NATIVE_SST14
from pyrosetta_flow.selectivity_loop import SelectivityLeaderboard
from pyrosetta_flow.continuous import DiversityPolicy, _apply_control
from pyrosetta_flow.schema import FlowConfig


# ---------------------------------------------------------------- global LB
def test_global_lb_dedup_and_best():
    lb = GlobalSelectivityLeaderboard(capacity=5)
    assert lb.add_measurement("AAA", -20, 14.0, 0.63) is True   # 신규 best
    assert lb.add_measurement("BBB", -18, 12.0, -1.37) is False  # best 갱신 아님
    # 같은 서열 더 나쁜 Δ → 무시(best 불변)
    assert lb.add_measurement("AAA", -19, 10.0, -3.0) is False
    assert lb.best_delta() == 0.63
    # 같은 서열 더 좋은 Δ → 갱신, 신규 best
    assert lb.add_measurement("AAA", -21, 15.0, 1.63) is True
    assert lb.best_delta() == 1.63
    assert len([e for e in lb.entries if e["sequence"] == "AAA"]) == 1, "서열 dedup"


def test_global_lb_save_load_roundtrip(tmp_path):
    p = tmp_path / "glb.json"
    lb = GlobalSelectivityLeaderboard(capacity=3)
    lb.add_measurement("AAA", -20, 14.0, 0.63,
                       extra={"pepadmet_hc50": -55.4, "hc50_vs_native": 0.28, "more_toxic_than_native": False})
    lb.save(p)
    lb2 = GlobalSelectivityLeaderboard.load(p)
    assert lb2.best_delta() == 0.63
    assert "AAA" in lb2.screened_seqs
    assert lb2.entries[0]["hc50"] == -55.4


def test_global_lb_count_passing():
    lb = GlobalSelectivityLeaderboard(capacity=10)
    # PASS: L-aa는 hc50 역변별 때문에 more_toxic_than_native 판정 skip
    lb.add_measurement("AGCKNFFWKTFTSC", -20, 14.0, 0.63, extra={"more_toxic_than_native": True})
    # PASS: D-aa marker도 명시적 비독성이면 통과
    lb.add_measurement("AGCkNFFWKTFTSC", -20, 14.2, 0.8, extra={"more_toxic_than_native": False})
    # FAIL: D-aa marker는 기존 독성↑ gate 유지
    lb.add_measurement("AGCdNFFWKTFTSC", -20, 14.5, 1.1, extra={"more_toxic_than_native": True})
    # FAIL: Δ<=0
    lb.add_measurement("AGCANFFWKTFTSC", -30, 12.0, -1.37, extra={"more_toxic_than_native": False})
    # FAIL: ddG 약함
    lb.add_measurement("AGCRNFFWKTFTSC", -10, 14.0, 0.63, extra={"more_toxic_than_native": False})
    assert lb.count_passing(ddg_max=-15.0) == 2


def test_global_lb_robust_ddg_stats_from_experiment_log(tmp_path):
    log = tmp_path / "experiment_log.jsonl"
    rows = [
        {"record_type": "candidate", "status": "success", "sequence": "OUTLIER", "ddg": -100.0},
        {"record_type": "candidate", "status": "success", "sequence": "OUTLIER", "ddg": 10.0},
        {"record_type": "candidate", "status": "success", "sequence": "OUTLIER", "ddg": -5.0},
        {"record_type": "candidate", "status": "success", "sequence": "OUTLIER", "ddg": -4.0},
        {"record_type": "candidate", "status": "success", "sequence": "STEADY", "ddg": -30.0},
        {"record_type": "candidate", "status": "success", "sequence": "STEADY", "ddg": -32.0},
        {"record_type": "candidate", "status": "success", "sequence": NATIVE_SST14, "ddg": -42.0},
        {"record_type": "candidate", "status": "success", "sequence": NATIVE_SST14, "ddg": -40.0},
    ]
    log.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    p = tmp_path / "global_selectivity_leaderboard.json"
    p.write_text(json.dumps({
        "entries": [
            {"sequence": "OUTLIER", "ddg": -100.0, "delta_margin": 10.0},
            {"sequence": "STEADY", "ddg": -30.0, "delta_margin": 1.0},
        ],
        "screened_seqs": [],
    }), encoding="utf-8")

    lb = GlobalSelectivityLeaderboard.load(p, experiment_log_path=log)
    assert lb.entries[0]["sequence"] == "STEADY"
    assert lb.entries[1]["sequence"] == "OUTLIER"
    assert lb.entries[1]["ddg_median"] == -5.0
    assert lb.entries[1]["ddg_n_converged"] == 3
    assert lb.native_ddg_stats["ddg_median"] == -41.0


def test_global_lb_legacy_sort_flag():
    lb = GlobalSelectivityLeaderboard(capacity=5, use_robust_ddg=False)
    lb.add_measurement("WEAK_SELECTIVE", -10, 20.0, 9.0)
    lb.add_measurement("STRONG_LESS_SELECTIVE", -50, 12.0, 1.0)
    assert lb.entries[0]["sequence"] == "WEAK_SELECTIVE"


def test_global_lb_ingest_artifacts():
    lb = GlobalSelectivityLeaderboard(capacity=10)
    artifacts = {
        "run_id": "r1",
        "iterations": [
            {"candidates": [
                {"sequence": "AAA", "ddg": -20, "extra_scores": {"selectivity_margin": 14.0, "delta_margin": 0.63}},
                {"sequence": "NOSEL", "ddg": -25, "extra_scores": {}},  # 선택성 미측정 → skip
            ]}
        ],
        "final_candidates": [
            {"sequence": "BBB", "ddg": -30, "extra_scores": {"selectivity_margin": 11.0, "delta_margin": -2.37}},
        ],
    }
    out = lb.ingest_artifacts(artifacts)
    assert out["n_measurements"] == 2          # AAA, BBB (NOSEL 제외)
    assert out["best_delta_margin"] == 0.63
    assert lb.count_passing() >= 0


# -------------------------------------------------------------- warm-start
def test_inloop_seed_from_global_dedup_and_gate():
    # capacity=1: 글로벌 1건 seed 로 리더보드가 가득 차 게이트가 즉시 무는 상황
    g = GlobalSelectivityLeaderboard(capacity=5)
    g.add_measurement("STRONG", -40, 14.0, 0.63)
    inloop = SelectivityLeaderboard(capacity=1)
    inloop.seed_from_global(g.warm_start_payload())
    # 역대 도킹 서열은 재도킹 안 함 (dedup)
    assert inloop.should_screen("STRONG", -50) is False
    # 게이트 기준선이 역대 best ddG(-40)로 올라감 → 그보다 약하면 skip, 강하면 screen
    assert inloop.should_screen("NEW_WEAK", -30) is False
    assert inloop.should_screen("NEW_STRONG", -45) is True


def test_inloop_seed_explores_until_capacity_filled():
    # 글로벌 hit < capacity 이면 (초기) 게이트가 안 물고 자유 탐색
    g = GlobalSelectivityLeaderboard(capacity=5)
    g.add_measurement("STRONG", -40, 14.0, 0.63)
    inloop = SelectivityLeaderboard(capacity=3)   # 1 seed < capacity 3 → 미충원
    inloop.seed_from_global(g.warm_start_payload())
    assert inloop.should_screen("STRONG", -50) is False      # dedup 은 유지
    assert inloop.should_screen("NEW_WEAK", -30) is True      # 미충원 → 탐색 허용


# ------------------------------------------------------------ diversity pol
def test_diversity_policy_escalates_on_plateau_and_resets():
    pol = DiversityPolicy(patience=2, base_mutations=3, max_mutations_cap=6)
    d = pol.update(0.5); assert d["improved"] and d["level"] == 0 and d["max_random_mutations"] == 3
    d = pol.update(0.5); assert not d["improved"] and d["level"] == 0   # stale=1
    d = pol.update(0.5); assert d["level"] == 1 and d["max_random_mutations"] == 4  # patience 도달 → 상승
    d = pol.update(0.5); assert d["level"] == 1                          # stale=1
    d = pol.update(0.5); assert d["level"] == 2 and d["max_random_mutations"] == 5
    d = pol.update(0.9); assert d["improved"] and d["level"] == 0 and d["max_random_mutations"] == 3  # 개선 → 리셋


def test_diversity_policy_caps_mutations():
    pol = DiversityPolicy(patience=1, base_mutations=5, max_mutations_cap=6)
    pol.update(0.1)
    for _ in range(10):
        d = pol.update(0.1)
    assert d["max_random_mutations"] == 6, "cap 초과 금지"


# ---------------------------------------------------------------- control
def test_apply_control_whitelist_only():
    base = FlowConfig(template_pdb="x", n_candidates=8, max_iterations=4)
    cfg = _apply_control(base, {
        "n_candidates": 12,             # 허용
        "max_iterations": 6,            # 허용
        "template_pdb": "HACK",         # 비허용 → 무시
        "patience": 99,                 # control 전용 → 무시
    })
    assert cfg.n_candidates == 12 and cfg.max_iterations == 6
    assert cfg.template_pdb == "x", "화이트리스트 외 필드는 변경 금지"
