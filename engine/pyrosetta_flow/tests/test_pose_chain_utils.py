import logging

import pytest

from pyrosetta_flow.pose_chain_utils import detect_peptide_chain


class _MockPdbInfo:
    def __init__(self, chains):
        self._chains = chains

    def chain(self, index):
        return self._chains[index - 1]


class _MockPose:
    def __init__(self, chains):
        self._chains = chains
        self._pdb_info = _MockPdbInfo(chains)

    def total_residue(self):
        return len(self._chains)

    def pdb_info(self):
        return self._pdb_info


@pytest.mark.parametrize(
    ("chains", "expected"),
    [
        (["A"] * 14 + ["B"] * 287, "A"),
        (["A"] * 287 + ["B"] * 14, "B"),
    ],
)
def test_detects_exact_length_independent_of_chain_order(chains, expected):
    assert detect_peptide_chain(_MockPose(chains), 14) == expected


def test_mismatch_warns_and_selects_closest_chain(caplog):
    pose = _MockPose(["A"] * 287 + ["B"] * 13)
    with caplog.at_level(logging.WARNING):
        assert detect_peptide_chain(pose, 14) == "B"
    assert "No chain matches peptide target_len=14" in caplog.text


def test_empty_pose_is_rejected():
    with pytest.raises(ValueError, match="no PDB chains"):
        detect_peptide_chain(_MockPose([]), 14)
