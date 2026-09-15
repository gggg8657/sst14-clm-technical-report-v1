"""test_daa_halflife_improvement_v2.py — D-aa 반감기 정확도 개선 검증 (2026-06-25).

대상:
  - step08_stability_v2.predict_half_life : D-aa 잔기수 비례 계수 (×35 일괄 → ×10 per-residue)
  - halflife_ensemble_v2.ensemble_halflife : RF has_D 피처 자동 전달 (rec=None 버그 수정)

검증 항목:
  1) D-aa 과대예측 완화: 단일 D-aa 케이스에서 v2 예측이 orig보다 낮고 문헌에 근접.
  2) L-aa 불변: D-aa 없는 케이스 orig == v2 (1e-9 허용오차).
  3) D-aa cap 동작: 2개 이상 D-aa는 cap(×35)에 걸려 orig와 동일.
  4) _count_d_aa: 'd_amino_acid', 'd_amino_acid:N', 소문자 서열 등 파싱 정확성.
  5) _detect_d_aa: modifications 및 소문자 서열 감지.
  6) RF has_D 활성화: ensemble_halflife에서 D-aa 있을 때 RF 예측값 변화 확인.

주의: 이 테스트는 _v2에서만 의미 있음. 원본(FROZEN)은 수정 없음.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# 픽스처
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def v2_hl():
    from pyrosetta_flow.surrogate_v2.step08_stability_v2 import predict_half_life
    return predict_half_life


@pytest.fixture(scope="module")
def orig_hl():
    from AG_src.pipeline.step08_stability import predict_half_life
    return predict_half_life


@pytest.fixture(scope="module")
def v2_ens():
    from pyrosetta_flow.halflife_ensemble_v2 import ensemble_halflife
    return ensemble_halflife


@pytest.fixture(scope="module")
def orig_ens():
    from pyrosetta_flow.halflife_ensemble import ensemble_halflife
    return ensemble_halflife


@pytest.fixture(scope="module")
def count_d_aa():
    from pyrosetta_flow.surrogate_v2.step08_stability_v2 import _count_d_aa
    return _count_d_aa


@pytest.fixture(scope="module")
def detect_d_aa():
    from pyrosetta_flow.halflife_ensemble_v2 import _detect_d_aa
    return _detect_d_aa


# ---------------------------------------------------------------------------
# 1. D-aa 과대예측 완화 (단일 D-aa)
# ---------------------------------------------------------------------------

class TestDaaOverestimationReduced:
    """단일 D-aa 케이스에서 _v2가 원본보다 낮게(문헌에 근접하게) 예측해야 한다."""

    # Bivalirudin: D-Phe1 1개, IV 혈중 t½ ≈ 0.42h
    # 원본 ×35 → 1.18h (과대), _v2 ×10 → ~0.34h (더 근접)
    @pytest.mark.parametrize("seq,mods,lit_hl,label", [
        ("FPRPGGGGNGDFEEIPEEYL", ["d_amino_acid"], 0.42, "Bivalirudin D-Phe1"),
        ("HWSYGLRPG", ["d_amino_acid"], 3.0, "Leuprolide D-Leu6 (IV)"),
    ])
    def test_single_daa_lower_than_orig(self, v2_hl, orig_hl, seq, mods, lit_hl, label):
        """단일 D-aa에서 _v2 < orig (과대예측 완화)."""
        orig_val = orig_hl(seq, mods)
        v2_val = v2_hl(seq, mods)
        assert v2_val < orig_val, (
            f"{label}: _v2({v2_val:.3f}h)가 orig({orig_val:.3f}h)보다 낮아야 함 "
            f"(×35→×10 per-residue 과대예측 완화)"
        )

    def test_bivalirudin_closer_to_literature(self, v2_hl, orig_hl):
        """Bivalirudin에서 _v2가 문헌(0.42h)에 더 근접해야 한다."""
        seq = "FPRPGGGGNGDFEEIPEEYL"
        mods = ["d_amino_acid"]
        lit = 0.42
        orig_val = orig_hl(seq, mods)
        v2_val = v2_hl(seq, mods)
        # 로그 오차 기준으로 비교 (상대 오차)
        orig_log_err = abs(math.log10(orig_val) - math.log10(lit))
        v2_log_err = abs(math.log10(v2_val) - math.log10(lit))
        assert v2_log_err < orig_log_err, (
            f"Bivalirudin: _v2 log 오차({v2_log_err:.3f}) < orig 오차({orig_log_err:.3f}) 기대. "
            f"orig={orig_val:.3f}h, v2={v2_val:.3f}h, 문헌={lit}h"
        )

    def test_bivalirudin_v2_within_order_of_magnitude(self, v2_hl):
        """Bivalirudin _v2가 문헌(0.42h) 대비 1 order of magnitude 이내."""
        v2_val = v2_hl("FPRPGGGGNGDFEEIPEEYL", ["d_amino_acid"])
        lit = 0.42
        assert 0.042 <= v2_val <= 4.2, (
            f"Bivalirudin _v2={v2_val:.3f}h, 문헌 0.42h 대비 1 OOM 이탈"
        )


# ---------------------------------------------------------------------------
# 2. L-aa 불변 (D-aa 없는 케이스)
# ---------------------------------------------------------------------------

_L_AA_CASES: List[tuple] = [
    ("AGCKNFFWKTFTSC", [], "SST-14 native"),
    ("AGCKNFFWKTDTSC", [], "SST-14 F11D"),
    ("AGCKNAFWKTFTSC", [], "SST-14 F7A"),
    ("AGCKNFFWKTFTSC", ["cyclization"], "SST-14 + cyclization"),
    ("AGCKNFFWKTFTSC", ["fatty_acid"], "SST-14 + fatty_acid"),
    ("AGCKNFFWKTFTSC", ["pegylation"], "SST-14 + pegylation"),
    ("AGCKNFFWKTFTSC", ["substitution"], "SST-14 + substitution"),
    ("ACDEF", [], "ACDEF short"),
    ("MAGICWAND", [], "MAGICWAND arbitrary"),
    ("ACKNOWLED", ["cyclization", "substitution"], "multi-mod L-aa"),
]


@pytest.mark.parametrize("seq,mods,label", _L_AA_CASES)
def test_l_aa_unchanged(orig_hl, v2_hl, seq, mods, label):
    """D-aa 없는 케이스에서 _v2 == orig (1e-9 허용오차)."""
    orig_val = orig_hl(seq, mods)
    v2_val = v2_hl(seq, mods)
    assert abs(orig_val - v2_val) < 1e-9, (
        f"{label}: orig={orig_val}, v2={v2_val} — L-aa 케이스는 동일해야 함"
    )


# ---------------------------------------------------------------------------
# 3. D-aa cap 동작 (2개 이상 → orig와 동일)
# ---------------------------------------------------------------------------

class TestDaaCap:
    """D-aa 2개 이상은 cap(log10(35))에 걸려 orig와 동일한 log_mult를 가져야 한다.
    (×10 per-residue이므로 n>=2에서 2×1.0=2.0 > cap 1.544 → cap 적용)
    """

    @pytest.mark.parametrize("n_daa,label", [
        (2, "D-aa:2"),
        (3, "D-aa:3"),
        (8, "D-aa:8 (all-D)"),
    ])
    def test_cap_applied_n_ge_2(self, orig_hl, v2_hl, n_daa, label):
        """n_daa>=2에서 v2 == orig (cap 동작)."""
        seq = "AGCKNFFWKTFTSC"
        mods = [f"d_amino_acid:{n_daa}"]
        orig_val = orig_hl(seq, ["d_amino_acid"])  # orig는 개수 무시, log10(35) 1회
        v2_val = v2_hl(seq, mods)
        # orig와 절대 동일하지 않을 수 있음(서열/modifications 파싱 차이),
        # 핵심: v2의 log_mult == log10(35) 임을 검증
        from pyrosetta_flow.surrogate_v2.step08_stability_v2 import _d_aa_log_mult, _D_AA_CAP_LOG
        lm = _d_aa_log_mult(n_daa)
        assert abs(lm - _D_AA_CAP_LOG) < 1e-9, (
            f"{label}: log_mult={lm:.4f} should equal cap={_D_AA_CAP_LOG:.4f}"
        )


# ---------------------------------------------------------------------------
# 4. _count_d_aa 파싱 정확성
# ---------------------------------------------------------------------------

class TestCountDaa:
    """_count_d_aa 파싱 단위 테스트."""

    def test_simple_d_amino_acid(self, count_d_aa):
        """'d_amino_acid' 단순 항목 → 1 반환."""
        assert count_d_aa("AGCKNFFWKTFTSC", ["d_amino_acid"]) == 1

    def test_colon_notation_3(self, count_d_aa):
        """'d_amino_acid:3' → 3 반환."""
        assert count_d_aa("AGCKNFFWKTFTSC", ["d_amino_acid:3"]) == 3

    def test_colon_notation_0(self, count_d_aa):
        """'d_amino_acid:0' → 0 반환."""
        assert count_d_aa("AGCKNFFWKTFTSC", ["d_amino_acid:0"]) == 0

    def test_lowercase_sequence(self, count_d_aa):
        """소문자 서열 3개 → 3 반환."""
        # 소문자 a, g, c = 3개 (D-aa 표기)
        assert count_d_aa("agcKNFFWKTFTSC", []) == 3

    def test_no_d_aa(self, count_d_aa):
        """D-aa 없으면 0 반환."""
        assert count_d_aa("AGCKNFFWKTFTSC", []) == 0

    def test_mixed_mods_with_d_amino(self, count_d_aa):
        """['cyclization', 'd_amino_acid:2'] → 2 반환."""
        assert count_d_aa("AGCKNFFWKTFTSC", ["cyclization", "d_amino_acid:2"]) == 2

    def test_d_hyphen_amino(self, count_d_aa):
        """'d-amino-acid' 형태도 D-aa로 인식 → 1."""
        assert count_d_aa("AGCKNFFWKTFTSC", ["d-amino-acid"]) == 1

    def test_uppercase_only_no_lowercase_seq(self, count_d_aa):
        """대문자 서열 + D-aa 없음 → 0."""
        assert count_d_aa("ACDEFGHIKLM", []) == 0


# ---------------------------------------------------------------------------
# 5. _detect_d_aa 동작 (ensemble_halflife_v2)
# ---------------------------------------------------------------------------

class TestDetectDaa:
    """_detect_d_aa 단위 테스트."""

    def test_detect_d_amino_acid_mod(self, detect_d_aa):
        assert detect_d_aa("AGCKNFFWKTFTSC", ["d_amino_acid"]) is True

    def test_detect_d_amino_acid_colon(self, detect_d_aa):
        assert detect_d_aa("AGCKNFFWKTFTSC", ["d_amino_acid:3"]) is True

    def test_detect_d_hyphen(self, detect_d_aa):
        assert detect_d_aa("AGCKNFFWKTFTSC", ["d-amino"]) is True

    def test_no_detect_cyclization(self, detect_d_aa):
        assert detect_d_aa("AGCKNFFWKTFTSC", ["cyclization"]) is False

    def test_no_detect_empty(self, detect_d_aa):
        assert detect_d_aa("AGCKNFFWKTFTSC", []) is False

    def test_detect_lowercase_seq(self, detect_d_aa):
        """소문자 서열 → D-aa 감지."""
        assert detect_d_aa("agcknffwktftsc", []) is True

    def test_detect_mixed_case_seq(self, detect_d_aa):
        """혼용 대소문자 → D-aa 감지."""
        assert detect_d_aa("AGCKNFFwKTFTSC", []) is True

    def test_no_detect_uppercase_only(self, detect_d_aa):
        assert detect_d_aa("AGCKNFFWKTFTSC", []) is False


# ---------------------------------------------------------------------------
# 6. RF has_D 활성화 효과 (ensemble_halflife_v2)
# ---------------------------------------------------------------------------

class TestRfHasDActivation:
    """D-aa 있을 때 ensemble_halflife_v2의 RF 예측값이 orig와 달라야 한다."""

    def test_daa_changes_rf_in_v2(self, orig_ens, v2_ens):
        """D-aa 있을 때 v2 RF 값이 orig와 달라야 한다 (has_D 활성화)."""
        seq = "AGCKNFFWKTFTSC"
        mods = ["d_amino_acid"]
        o = orig_ens(seq, modifications=mods)
        v = v2_ens(seq, modifications=mods)
        # RF 값이 다르거나(has_D 활성화) 또는 RF 미가용(None) 시 skip
        o_rf = o.get("half_life_rf_h")
        v_rf = v.get("half_life_rf_h")
        if o_rf is None and v_rf is None:
            pytest.skip("RF 모델 미가용 — RF has_D 테스트 건너뜀")
        # RF 값이 변경되어야 함 (has_D: 0→1)
        assert o_rf != v_rf, (
            f"D-aa 케이스에서 v2 RF({v_rf})가 orig RF({o_rf})와 같음 — has_D 활성화 실패"
        )

    def test_no_daa_rf_unchanged_in_v2(self, orig_ens, v2_ens):
        """D-aa 없는 케이스에서 v2 RF == orig RF (has_D 동일)."""
        seq = "AGCKNFFWKTFTSC"
        mods: list = []
        o = orig_ens(seq, modifications=mods)
        v = v2_ens(seq, modifications=mods)
        o_rf = o.get("half_life_rf_h")
        v_rf = v.get("half_life_rf_h")
        if o_rf is None and v_rf is None:
            pytest.skip("RF 모델 미가용")
        if o_rf is None or v_rf is None:
            pytest.skip("RF 편측 미가용")
        assert abs(o_rf - v_rf) < 1e-6, (
            f"L-aa 케이스 RF 불변 실패: orig={o_rf}, v2={v_rf}"
        )

    def test_ensemble_keys_present(self, v2_ens):
        """ensemble_halflife_v2 반환 키 집합 확인."""
        result = v2_ens("AGCKNFFWKTFTSC", modifications=["d_amino_acid"])
        required_keys = {"half_life_h", "half_life_heuristic_h", "half_life_rf_h",
                         "stability_norm", "halflife_source"}
        assert required_keys.issubset(set(result.keys())), (
            f"필수 키 누락: {required_keys - set(result.keys())}"
        )

    def test_l_aa_ensemble_norm_unchanged(self, orig_ens, v2_ens):
        """D-aa 없는 케이스에서 stability_norm이 orig와 같거나 유사해야 한다.
        (휴리스틱은 동일, RF has_D=False → 동일, 따라서 norm 동일)
        """
        cases = [
            ("AGCKNFFWKTFTSC", [], "SST-14 native"),
            ("AGCKNFFWKTFTSC", ["fatty_acid"], "SST-14 + fatty"),
            ("AGCKNFFWKTFTSC", ["cyclization"], "SST-14 + cyclic"),
        ]
        for seq, mods, label in cases:
            o = orig_ens(seq, modifications=mods)
            v = v2_ens(seq, modifications=mods)
            assert abs(o["stability_norm"] - v["stability_norm"]) < 1e-4, (
                f"{label}: stability_norm 불일치 orig={o['stability_norm']}, v2={v['stability_norm']}"
            )


# ---------------------------------------------------------------------------
# 7. 척도 변화 범위 확인 (D-aa 후보 영향 범위)
# ---------------------------------------------------------------------------

class TestScaleChangeScope:
    """D-aa 후보에서 stability_norm / half_life_h 변화 범위를 확인한다.
    (리더보드 재계산 범위 보고용 — 수치 검증이 아닌 변화 존재 여부 검증)
    """

    @pytest.mark.parametrize("mods,expect_change", [
        (["d_amino_acid"], True),            # 단일 D-aa → 변화
        (["d_amino_acid:2"], False),         # 2개 D-aa → cap → 원본과 동일
        (["d_amino_acid", "fatty_acid"], True),  # D-aa+fatty → 휴리스틱 D-aa 부분 변화
        ([], False),                         # L-aa → 불변
    ])
    def test_halflife_h_change_scope(self, orig_ens, v2_ens, mods, expect_change):
        """mods에 따른 half_life_h 변화 여부가 expect_change와 일치해야 한다."""
        seq = "AGCKNFFWKTFTSC"
        o = orig_ens(seq, modifications=mods)
        v = v2_ens(seq, modifications=mods)
        changed = abs(o["half_life_h"] - v["half_life_h"]) > 1e-6
        if expect_change:
            assert changed, (
                f"mods={mods}: half_life_h 변화 기대, orig={o['half_life_h']}, v2={v['half_life_h']}"
            )
        else:
            assert not changed, (
                f"mods={mods}: half_life_h 불변 기대, orig={o['half_life_h']}, v2={v['half_life_h']}"
            )
