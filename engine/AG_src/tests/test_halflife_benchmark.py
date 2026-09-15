"""반감기 surrogate 재보정 회귀 테스트 (2026-06-09 A).

문헌 혈중 t½ 가 공개된 펩타이드들에 대해 predict_half_life 의 **상대 순위**가
보존되는지 고정. 재보정(log-multiplicative) 이전엔 cyclization +24h additive 때문에
SST-14 가 16.6h 로 과대예측되고 plain Spearman 이 -0.5(역전)였다.

검증 기준:
  - SST-14 (사이클릭, 실제 ~0.05h) 가 과대예측되지 않음 (< 1h)
  - 전체 Spearman >= 0.7, plain(modification 無) Spearman >= 0.3
  - fatty-acid 펩타이드(semaglutide/liraglutide) 가 plain 펩타이드보다 김
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from AG_src.pipeline.step08_stability import predict_half_life

# (name, sequence, modifications, 문헌 혈중 t½ hours)
BENCH = [
    ("GLP-1",        "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR", [],               0.03),
    ("SST-14",       "AGCKNFFWKTFTSC",                 [],               0.05),
    ("Exenatide",    "HGEGTFTSDLSKQMEEEAVRLFIEWLKNGGPSSGAPPPS", [],      2.4),
    ("Octreotide",   "FCFWKTCT",                       ["d_amino_acid"], 1.7),
    ("Desmopressin", "CYFQNCPRG",                      ["d_amino_acid"], 3.0),
    ("Leuprolide",   "HWSYLLRP",                       ["d_amino_acid"], 3.0),
    ("Liraglutide",  "HAEGTFTSDVSSYLEGQAAKEFIAWLVRGR", ["fatty_acid"],   13.0),
    ("Semaglutide",  "HAEGTFTSDVSSYLEGQAAKEFIAWLVRGR", ["fatty_acid"],   168.0),
]


def _spearman(xs, ys):
    import numpy as np
    def rank(a):
        a = np.asarray(a, float); r = np.empty(len(a)); o = np.argsort(a); r[o] = np.arange(len(a))
        for v in np.unique(a):
            idx = np.where(a == v)[0]; r[idx] = r[idx].mean()
        return r
    rx, ry = rank(xs), rank(ys); rx -= rx.mean(); ry -= ry.mean()
    d = (np.sqrt((rx ** 2).sum()) * np.sqrt((ry ** 2).sum()))
    return float((rx * ry).sum() / d) if d else 0.0


def test_sst14_not_overpredicted():
    """SST-14 (실제 ~0.05h) 가 과대예측되지 않아야 한다 (재보정 전 16.6h 버그)."""
    hl = predict_half_life("AGCKNFFWKTFTSC", [])
    assert hl < 1.0, f"SST-14 t½ 과대예측: {hl}h (실제 ~0.05h, <1h 기대)"


def test_overall_rank_correlation():
    preds = [predict_half_life(s, m) for _, s, m, _ in BENCH]
    lits = [b[3] for b in BENCH]
    rho = _spearman(lits, preds)
    assert rho >= 0.7, f"전체 Spearman {rho:.3f} < 0.7"


def test_plain_peptide_rank_not_inverted():
    """modification 無 펩타이드의 순위가 역전되지 않음 (재보정 전 -0.5)."""
    plain = [(b[3], predict_half_life(b[1], b[2])) for b in BENCH if not b[2]]
    rho = _spearman([p[0] for p in plain], [p[1] for p in plain])
    assert rho >= 0.3, f"plain Spearman {rho:.3f} < 0.3 (순위 역전 위험)"


def test_fatty_acid_extends_half_life():
    """지방산 아실화가 plain 대비 반감기를 크게 늘려야 한다 (알부민 결합)."""
    plain_glp1 = predict_half_life("HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR", [])
    fa_glp1 = predict_half_life("HAEGTFTSDVSSYLEGQAAKEFIAWLVRGR", ["fatty_acid"])
    assert fa_glp1 > 10 * plain_glp1, f"fatty_acid 효과 부족: {fa_glp1} vs {plain_glp1}"


def test_variant_discrimination_within_sst14():
    """스크리닝 실사용: SST-14 변이체 간 변별력(말단 변이가 더 짧음)."""
    native = predict_half_life("AGCKNFFWKTFTSC", [])
    terminal_mut = predict_half_life("YGCKNFFWKTFTST", [])  # N/C말단 변이 → exopeptidase 취약
    assert terminal_mut < native, f"말단 변이 변별 실패: {terminal_mut} vs {native}"


def test_stability_gate_relative_percentile_mode():
    """A1(2026-06-17): 절대 임계 대신 surrogate 순위 상위 비율로 게이트 (임상 t½ 미신뢰 대응)."""
    from types import SimpleNamespace
    from AG_src.pipeline.step08_stability import apply_stability_gate
    rs = [SimpleNamespace(modified_half_life_hours=h) for h in [10.0, 5.0, 200.0, 50.0]]
    passed, failed = apply_stability_gate(rs, gate_mode="relative_percentile", top_fraction=0.5)
    assert len(passed) == 2 and len(failed) == 2
    assert passed[0].modified_half_life_hours == 200.0  # 순위 상위
    assert all(p.modified_half_life_hours >= f.modified_half_life_hours
               for p in passed for f in failed)
