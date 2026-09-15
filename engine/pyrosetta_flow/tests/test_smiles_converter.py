import pytest

from pyrosetta_flow import smiles_converter


def test_three_letter_sequence_marks_d_lysine():
    names = smiles_converter._three_letter_sequence("AGCkN")

    assert names == ["Ala", "Gly", "Cys", "D-Lys", "Asn"]


@pytest.mark.skipif(not smiles_converter._HAS_RDKIT, reason="RDKit is not installed")
def test_sequence_to_smiles_preserves_d_lysine_stereochemistry():
    l_smiles = smiles_converter.sequence_to_smiles("AGCKNFFWKTFTSC")
    d_smiles = smiles_converter.sequence_to_smiles("AGCkNFFWKTFTSC")

    assert l_smiles
    assert d_smiles
    assert l_smiles != d_smiles
    assert "[C@H]" in d_smiles or "[C@@H]" in d_smiles

