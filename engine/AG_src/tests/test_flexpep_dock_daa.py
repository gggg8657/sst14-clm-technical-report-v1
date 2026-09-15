"""D-amino acid 최소 지원 회귀 테스트 (AG_src/scripts/flexpep_dock.py).

목적:
  1. L-aa 전용 동작이 D-aa 지원 추가 이후에도 완전히 보존되는지 확인
     (native no-op, 기존 L 변이 케이스).
  2. 소문자 1-letter 코드가 올바른 D-ResidueType(3-letter, "D"+표준명)으로
     치환되는지 확인.
  3. unknown amino acid code가 이제 silent skip이 아니라 명시적 ValueError로
     reject되는지 확인 (2026-07-01, P0 위험 — 오귀속 차단).

PyRosetta는 `bio-tools` conda env에서만 사용 가능하므로, 미설치 환경에서는
importorskip으로 전체 모듈을 건너뛴다.

실행:
  conda run -n bio-tools python -m pytest AG_src/tests/test_flexpep_dock_daa.py -q
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("pyrosetta")
pytest.importorskip(
    "pyrosetta.rosetta.protocols.simple_moves",
    reason="D-aa FlexPepDock tests require real PyRosetta MutateResidue support",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
AG_SRC_SCRIPTS = REPO_ROOT / "AG_src" / "scripts"
if str(AG_SRC_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(AG_SRC_SCRIPTS))

NATIVE_PDB = (
    REPO_ROOT / "data" / "somatostatin_receptor" / "SSTR2_SST14_complex_boltz_1.pdb"
)
NATIVE_SEQ = "AGCKNFFWKTFTSC"

pytestmark = pytest.mark.skipif(
    not NATIVE_PDB.exists(),
    reason=f"native reference complex not found: {NATIVE_PDB}",
)


@pytest.fixture(scope="module")
def flexpep_dock_module():
    module_path = AG_SRC_SCRIPTS / "flexpep_dock.py"
    spec = importlib.util.spec_from_file_location("ag_src_flexpep_dock", module_path)
    assert spec is not None and spec.loader is not None
    fd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fd)

    fd.init_pyrosetta()
    return fd


def _peptide_sequence_and_names(pose, chain: int):
    begin = pose.chain_begin(chain)
    end = pose.chain_end(chain)
    seq = "".join(pose.residue(i).name1() for i in range(begin, end + 1))
    names = [pose.residue(i).name() for i in range(begin, end + 1)]
    return seq, names


class TestLAminoAcidRegression:
    """D-aa 지원 추가 이후에도 L-aa 경로가 완전히 보존되는지 확인."""

    def test_native_sequence_noop(self, flexpep_dock_module):
        fd = flexpep_dock_module
        pose, chain = fd.prepare_complex_by_mutation(
            str(NATIVE_PDB), NATIVE_SEQ, peptide_chain=1
        )
        seq, _names = _peptide_sequence_and_names(pose, chain)
        assert seq == NATIVE_SEQ

    def test_l_aa_mutation_applied_correctly(self, flexpep_dock_module):
        """기존 알려진 변이 케이스 (F7I, T13N) — 전부 L-form이어야 함."""
        fd = flexpep_dock_module
        target = "AGCKNIFWKTFNSC"
        pose, chain = fd.prepare_complex_by_mutation(
            str(NATIVE_PDB), target, peptide_chain=1
        )
        seq, names = _peptide_sequence_and_names(pose, chain)
        assert seq == target
        # L-form residue 이름은 "D" 접두 3-letter 코드가 아니어야 함
        for name in names:
            base = name.split(":")[0]
            assert not base.startswith("D") or base == "ASP", (
                f"Expected L-form residue, got {name}"
            )

    def test_unknown_amino_acid_raises_value_error(self, flexpep_dock_module):
        """silent skip 금지 — unknown AA는 명시적 예외로 reject."""
        fd = flexpep_dock_module
        target = "AGCKNXFWKTFTSC"  # 'X' = unknown
        with pytest.raises(ValueError, match="Unknown amino acid code"):
            fd.prepare_complex_by_mutation(str(NATIVE_PDB), target, peptide_chain=1)


class TestDAminoAcidMinimalSupport:
    """소문자 1-letter → D-ResidueType 최소 지원."""

    @pytest.mark.parametrize(
        "letter,expected_d_name",
        [
            ("a", "DALA"),
            ("c", "DCYS"),
            ("f", "DPHE"),
            ("k", "DLYS"),
            ("t", "DTHR"),
            ("w", "DTRP"),
        ],
    )
    def test_lowercase_maps_to_d_residue(
        self, flexpep_dock_module, letter, expected_d_name
    ):
        fd = flexpep_dock_module
        # pos12 (0-idx 11, native='T') 에 D-aa 삽입
        target = list(NATIVE_SEQ)
        target[11] = letter
        target = "".join(target)
        pose, chain = fd.prepare_complex_by_mutation(
            str(NATIVE_PDB), target, peptide_chain=1
        )
        _seq, names = _peptide_sequence_and_names(pose, chain)
        mutated_name = names[11]
        assert mutated_name.startswith(expected_d_name), (
            f"Expected {expected_d_name}, got {mutated_name}"
        )

    def test_d_residue_type_is_d_aa(self, flexpep_dock_module):
        """MutateResidue 적용 후 residue.type().is_d_aa() == True 확인."""
        fd = flexpep_dock_module
        target = list(NATIVE_SEQ)
        target[11] = "t"  # D-Thr
        target = "".join(target)
        pose, chain = fd.prepare_complex_by_mutation(
            str(NATIVE_PDB), target, peptide_chain=1
        )
        begin = pose.chain_begin(chain)
        mutated_residue = pose.residue(begin + 11)
        assert mutated_residue.type().is_d_aa() is True

    def test_glycine_achiral_no_d_variant(self, flexpep_dock_module):
        """Gly는 achiral이므로 소문자 'g'도 L과 동일한 GLY로 처리."""
        fd = flexpep_dock_module
        target = list(NATIVE_SEQ)
        target[1] = "g"  # native pos2도 이미 G이므로 다른 위치로 변경해 diff 유발
        target[3] = "g"  # K4 -> Gly(d/l 무관)
        target = "".join(target)
        pose, chain = fd.prepare_complex_by_mutation(
            str(NATIVE_PDB), target, peptide_chain=1
        )
        _seq, names = _peptide_sequence_and_names(pose, chain)
        assert names[3].split(":")[0] == "GLY"

    def test_d_cys_detected_by_disulfide_finder(self, flexpep_dock_module):
        """D-Cys도 _find_peptide_cys_residues (name1()=='C' 기반)가 검출해야 함."""
        fd = flexpep_dock_module
        target = list(NATIVE_SEQ)
        target[2] = "c"  # pos3 native Cys -> D-Cys (SS bond 파트너 pos14는 L 유지)
        target = "".join(target)
        pose, chain = fd.prepare_complex_by_mutation(
            str(NATIVE_PDB), target, peptide_chain=1
        )
        pose = fd.reorder_peptide_last(pose, chain)
        cys_residues = fd._find_peptide_cys_residues(pose)
        assert len(cys_residues) == 2, (
            f"Expected 2 Cys residues (1 D + 1 L) detected, got {cys_residues}"
        )
