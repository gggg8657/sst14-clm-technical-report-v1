"""저복잡도 아티팩트 필터 단위 테스트."""
import sys
from pathlib import Path

# silo_a_flow 임포트 경로 설정
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyrosetta_flow.silo_a_flow import _is_low_complexity


def test_poly_g_is_artifact():
    """GGGGGGGGGGGGGGG → 아티팩트 (G 100%)"""
    flag, reason = _is_low_complexity("GGGGGGGGGGGGGGG")
    assert flag is True
    assert "단일잔기과다" in reason


def test_poly_l_high_frac_is_artifact():
    """LLLLLLLALLLLLAR → 아티팩트 (L ~80%)"""
    flag, reason = _is_low_complexity("LLLLLLLALLLLLAR")
    assert flag is True
    assert "단일잔기과다" in reason


def test_diverse_seq_passes():
    """SSLVLAQLVYERRAR → 통과 (다양한 잔기)"""
    flag, reason = _is_low_complexity("SSLVLAQLVYERRAR")
    assert flag is False
    assert reason == ""


def test_sst14_passes():
    """AGCKNFFWKTFTSC (SST-14) → 통과"""
    flag, reason = _is_low_complexity("AGCKNFFWKTFTSC")
    assert flag is False


def test_few_unique_is_artifact():
    """GAGAGAGAGAGAG → 고유 잔기 2개 → 아티팩트"""
    flag, reason = _is_low_complexity("GAGAGAGAGAGAG")
    assert flag is True
    assert "고유잔기부족" in reason


def test_low_entropy_is_artifact():
    """AAAAAAAAABCD → 엔트로피 낮음 → 아티팩트"""
    flag, reason = _is_low_complexity("AAAAAAAAABCD")
    # A가 ~75% → 단일잔기과다로 걸릴 수도 있음
    assert flag is True


def test_empty_seq_is_artifact():
    """빈 서열 → 아티팩트"""
    flag, reason = _is_low_complexity("")
    assert flag is True


def test_mutation_candidate_passes():
    """AGCKNFFWKTDTSC (F11D 변이체) → 통과"""
    flag, reason = _is_low_complexity("AGCKNFFWKTDTSC")
    assert flag is False
