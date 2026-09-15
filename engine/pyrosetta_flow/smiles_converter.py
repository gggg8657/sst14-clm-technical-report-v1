"""
SST-14 유사체 서열 → cyclic peptide SMILES 변환.

pepADMET은 SMILES + 서열 이중 입력이 필요.
Cys3-Cys14 이황화결합을 포함한 사이클릭 SMILES를 생성.

Usage:
    from pyrosetta_flow.smiles_converter import sequence_to_smiles
    smiles = sequence_to_smiles("AGCKNFFWKTFTSC")
"""
from __future__ import annotations

_HAS_RDKIT = False
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    _HAS_RDKIT = True
except ImportError:
    pass

from .multiobjective import _has_d_aa_marker


_AA_THREE_LETTER = {
    "A": "Ala", "C": "Cys", "D": "Asp", "E": "Glu", "F": "Phe",
    "G": "Gly", "H": "His", "I": "Ile", "K": "Lys", "L": "Leu",
    "M": "Met", "N": "Asn", "P": "Pro", "Q": "Gln", "R": "Arg",
    "S": "Ser", "T": "Thr", "V": "Val", "W": "Trp", "Y": "Tyr",
}


def _three_letter_sequence(sequence: str) -> list[str]:
    """Return residue names, explicitly marking lowercase residues as D-form."""
    return [
        f"D-{_AA_THREE_LETTER[aa.upper()]}" if aa.islower() else _AA_THREE_LETTER[aa]
        for aa in sequence
    ]


def _invert_d_residue_chirality(mol, sequence: str) -> None:
    """Invert each lowercase residue's alpha-carbon stereocentre in-place."""
    d_positions = {i + 1 for i, aa in enumerate(sequence) if aa.islower()}
    for atom in mol.GetAtoms():
        residue = atom.GetPDBResidueInfo()
        if (
            residue is not None
            and residue.GetResidueNumber() in d_positions
            and residue.GetName().strip() == "CA"
        ):
            tag = atom.GetChiralTag()
            if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
                atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
            elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
                atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)


def sequence_to_smiles(
    sequence: str,
    ss_bond_positions: tuple[int, int] | None = None,
) -> str | None:
    """Convert peptide sequence to SMILES with optional SS bond.

    Parameters
    ----------
    sequence : str
        One-letter amino acid sequence (e.g., "AGCKNFFWKTFTSC").
    ss_bond_positions : tuple[int, int] or None
        1-indexed positions of Cys residues forming SS bond.
        Default: auto-detect first and last Cys.

    Returns
    -------
    str or None
        Canonical SMILES string, or None if RDKit unavailable or conversion fails.
    """
    if not _HAS_RDKIT:
        return None

    if not sequence or not isinstance(sequence, str):
        return None

    raw_seq = sequence.strip()
    seq = raw_seq if _has_d_aa_marker(raw_seq) else raw_seq.upper()

    try:
        # MolFromSequence accepts L-amino-acid one-letter codes only. Build that
        # backbone, then invert the marked alpha carbons to obtain mixed D/L
        # stereochemistry (e.g. lowercase k is D-Lys).
        _three_letter_sequence(seq)  # validates residues and preserves D-* naming
        rdkit_seq = "".join(aa.upper() for aa in seq)

        # Build linear peptide
        mol = Chem.MolFromSequence(rdkit_seq)
        if mol is None:
            return None
        _invert_d_residue_chirality(mol, seq)

        # If SS bond positions specified or auto-detect Cys pairs
        if ss_bond_positions is None:
            cys_positions = [i for i, aa in enumerate(seq) if aa.upper() == "C"]
            if len(cys_positions) >= 2:
                ss_bond_positions = (cys_positions[0] + 1, cys_positions[-1] + 1)

        if ss_bond_positions:
            # Form disulfide bond between specified Cys residues
            # Find sulfur atoms of the two Cys residues
            pos1_0idx = ss_bond_positions[0] - 1
            pos2_0idx = ss_bond_positions[1] - 1

            # Get residue info to find SG atoms
            sg_atoms = []
            for atom in mol.GetAtoms():
                ri = atom.GetPDBResidueInfo()
                if ri is not None:
                    res_idx = ri.GetResidueNumber() - 1
                    atom_name = ri.GetName().strip()
                    if res_idx in (pos1_0idx, pos2_0idx) and atom_name == "SG":
                        sg_atoms.append(atom.GetIdx())

            if len(sg_atoms) == 2:
                # Remove H from SG atoms and form S-S bond
                emol = Chem.RWMol(mol)
                emol.AddBond(sg_atoms[0], sg_atoms[1], Chem.BondType.SINGLE)

                # Remove one H from each SG (disulfide formation loses 2H)
                try:
                    mol = emol.GetMol()
                    mol = Chem.RemoveHs(mol)
                    Chem.SanitizeMol(mol)
                except Exception:
                    # If SS bond formation fails, return linear SMILES
                    mol = Chem.MolFromSequence(rdkit_seq)
                    if mol is not None:
                        _invert_d_residue_chirality(mol, seq)

        smiles = Chem.MolToSmiles(mol)
        return smiles

    except Exception:
        return None


def smiles_to_mw(smiles: str) -> float | None:
    """Calculate molecular weight from SMILES."""
    if not _HAS_RDKIT or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Descriptors.ExactMolWt(mol)
    except Exception:
        return None


def batch_convert(sequences: list[str]) -> list[dict]:
    """Convert multiple sequences to SMILES.

    Returns list of {"sequence": str, "smiles": str|None, "mw": float|None}.
    """
    results = []
    for seq in sequences:
        smiles = sequence_to_smiles(seq)
        mw = smiles_to_mw(smiles) if smiles else None
        results.append({
            "sequence": seq,
            "smiles": smiles,
            "mw": mw,
        })
    return results
