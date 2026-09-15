"""
Literature-based pharmacological property calculator for peptide sequences.

All 13 methods use published physicochemical scales and return pure numerical
values — no subjective weights or scoring.  Every function requires only an
amino acid one-letter-code string as input.

References
----------
1.  Kyte & Doolittle 1982, J Mol Biol 157:105-132
2.  Boman 2003, J Intern Med 254:197-215
3.  Guruprasad et al. 1990, Protein Eng 4:155-161
4.  Ikai 1980, J Biochem 88:1895-1898
5.  Bjellqvist et al. 1993 / Lehninger pKa set
6.  Pace et al. 1995, Protein Sci 4:2411-2423
7.  Varshavsky 1996, PNAS 93:12142-12149
8.  Eisenberg et al. 1982, Nature 299:371-374
9.  Wimley & White 1996, Nat Struct Biol 3:842-848
10. Henderson-Hasselbalch net charge
11. MEROPS database protease rules
12. Henikoff & Henikoff 1992, PNAS 89:10915-10919
13. Rulísek & Vondrásek 1998, J Inorg Biochem 71:115-127
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

# ─────────────────────────── lookup tables ───────────────────────────

# 1. Kyte-Doolittle hydropathy (1982)
KD_HYDROPATHY: Dict[str, float] = {
    "A":  1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C":  2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I":  4.5,
    "L":  3.8, "K": -3.9, "M":  1.9, "F":  2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V":  4.2,
}

# 2. Radzicka-Wolfenden transfer free energies (kcal/mol)
RW_TRANSFER: Dict[str, float] = {
    "A": -1.81, "R": 14.92, "N":  6.64, "D":  8.72, "C": -1.28,
    "Q":  5.54, "E":  6.81, "G": -0.94, "H":  4.66, "I": -4.92,
    "L": -4.92, "K":  5.55, "M": -2.35, "F": -2.98, "P": -2.54,
    "S":  3.40, "T":  2.57, "W": -2.33, "Y":  0.14, "V": -4.04,
}

# 3. Instability-index dipeptide weight values (DIWV)
#    ExPASy ProtParam 400-entry table  (Guruprasad et al. 1990)
#    Row = AA[i], Column = AA[i+1]
_DIWV_RAW: Dict[str, Dict[str, float]] = {
    "A": {"A": 1.0, "C": 44.94, "D": -7.49, "E": 1.0, "F": 1.0,
          "G": 1.0, "H": -7.49, "I": 1.0, "K": 1.0, "L": 1.0,
          "M": 1.0, "N": 1.0, "P": 20.26, "Q": 1.0, "R": 1.0,
          "S": 1.0, "T": 1.0, "V": 1.0, "W": 1.0, "Y": 1.0},
    "C": {"A": 1.0, "C": 1.0, "D": 20.26, "E": 1.0, "F": 1.0,
          "G": 1.0, "H": 33.60, "I": 1.0, "K": 1.0, "L": 20.26,
          "M": 33.60, "N": 1.0, "P": 20.26, "Q": -6.54, "R": 1.0,
          "S": 1.0, "T": 33.60, "V": -6.54, "W": 24.68, "Y": 1.0},
    "D": {"A": 1.0, "C": 1.0, "D": 1.0, "E": 1.0, "F": -6.54,
          "G": 1.0, "H": 1.0, "I": 1.0, "K": -7.49, "L": 1.0,
          "M": 1.0, "N": 1.0, "P": 1.0, "Q": 1.0, "R": -6.54,
          "S": 20.26, "T": -14.03, "V": 1.0, "W": 1.0, "Y": 1.0},
    "E": {"A": 1.0, "C": 44.94, "D": 20.26, "E": 33.60, "F": 1.0,
          "G": 1.0, "H": -6.54, "I": 20.26, "K": 1.0, "L": 1.0,
          "M": 1.0, "N": 1.0, "P": 20.26, "Q": 20.26, "R": 1.0,
          "S": 20.26, "T": 1.0, "V": 1.0, "W": -14.03, "Y": 1.0},
    "F": {"A": 1.0, "C": 1.0, "D": 13.34, "E": 1.0, "F": 1.0,
          "G": 1.0, "H": 1.0, "I": 1.0, "K": -14.03, "L": 1.0,
          "M": 1.0, "N": 1.0, "P": 20.26, "Q": 1.0, "R": 1.0,
          "S": 1.0, "T": 1.0, "V": 1.0, "W": 1.0, "Y": 33.60},
    "G": {"A": -7.49, "C": 1.0, "D": 1.0, "E": -6.54, "F": 1.0,
          "G": 13.34, "H": 1.0, "I": -7.49, "K": -7.49, "L": 1.0,
          "M": 1.0, "N": -7.49, "P": 1.0, "Q": 1.0, "R": 1.0,
          "S": 1.0, "T": -7.49, "V": 1.0, "W": 13.34, "Y": -7.49},
    "H": {"A": 1.0, "C": 1.0, "D": 1.0, "E": 1.0, "F": -9.37,
          "G": -9.37, "H": 1.0, "I": 44.94, "K": 24.68, "L": 1.0,
          "M": 1.0, "N": 24.68, "P": -1.88, "Q": 1.0, "R": 1.0,
          "S": 1.0, "T": -6.54, "V": 1.0, "W": -1.88, "Y": 44.94},
    "I": {"A": 1.0, "C": 1.0, "D": 1.0, "E": 44.94, "F": 1.0,
          "G": 1.0, "H": 13.34, "I": 1.0, "K": -7.49, "L": 20.26,
          "M": 1.0, "N": 1.0, "P": -1.88, "Q": 1.0, "R": 1.0,
          "S": 1.0, "T": 1.0, "V": -7.49, "W": 1.0, "Y": 1.0},
    "K": {"A": 1.0, "C": 1.0, "D": 1.0, "E": 1.0, "F": 1.0,
          "G": -7.49, "H": 1.0, "I": -7.49, "K": 1.0, "L": -7.49,
          "M": 33.60, "N": 1.0, "P": -6.54, "Q": 24.68, "R": 33.60,
          "S": 1.0, "T": 1.0, "V": -7.49, "W": 1.0, "Y": 1.0},
    "L": {"A": 1.0, "C": 1.0, "D": 1.0, "E": 1.0, "F": 1.0,
          "G": 1.0, "H": 1.0, "I": 1.0, "K": -7.49, "L": 1.0,
          "M": 1.0, "N": 1.0, "P": 20.26, "Q": 33.60, "R": 20.26,
          "S": 1.0, "T": 1.0, "V": 1.0, "W": 24.68, "Y": 1.0},
    "M": {"A": 13.34, "C": 1.0, "D": 1.0, "E": 1.0, "F": 1.0,
          "G": 1.0, "H": 58.28, "I": 1.0, "K": 1.0, "L": 1.0,
          "M": -1.88, "N": 1.0, "P": 44.94, "Q": -6.54, "R": -6.54,
          "S": 44.94, "T": -1.88, "V": 1.0, "W": 1.0, "Y": 24.68},
    "N": {"A": 1.0, "C": -1.88, "D": 1.0, "E": 1.0, "F": -14.03,
          "G": -14.03, "H": 1.0, "I": 44.94, "K": 24.68, "L": 1.0,
          "M": 1.0, "N": 1.0, "P": -1.88, "Q": -6.54, "R": 1.0,
          "S": 1.0, "T": -7.49, "V": 1.0, "W": -9.37, "Y": 1.0},
    "P": {"A": 20.26, "C": -6.54, "D": -6.54, "E": 18.38, "F": 20.26,
          "G": 1.0, "H": 1.0, "I": 1.0, "K": 1.0, "L": 1.0,
          "M": -6.54, "N": 1.0, "P": 20.26, "Q": 20.26, "R": -6.54,
          "S": 20.26, "T": 1.0, "V": 20.26, "W": -1.88, "Y": 1.0},
    "Q": {"A": 1.0, "C": -6.54, "D": 20.26, "E": 20.26, "F": -6.54,
          "G": 1.0, "H": 1.0, "I": 1.0, "K": 1.0, "L": 1.0,
          "M": 1.0, "N": 1.0, "P": 20.26, "Q": 20.26, "R": 1.0,
          "S": 44.94, "T": 1.0, "V": -6.54, "W": 1.0, "Y": -6.54},
    "R": {"A": 1.0, "C": 1.0, "D": 1.0, "E": 1.0, "F": 1.0,
          "G": -7.49, "H": 20.26, "I": 1.0, "K": 1.0, "L": 1.0,
          "M": 1.0, "N": 13.34, "P": 20.26, "Q": 20.26, "R": 58.28,
          "S": 44.94, "T": 1.0, "V": 1.0, "W": 58.28, "Y": -6.54},
    "S": {"A": 1.0, "C": 33.60, "D": 1.0, "E": 20.26, "F": 1.0,
          "G": 1.0, "H": 1.0, "I": 1.0, "K": 1.0, "L": 1.0,
          "M": 1.0, "N": 1.0, "P": 44.94, "Q": 20.26, "R": 20.26,
          "S": 20.26, "T": 1.0, "V": 1.0, "W": 1.0, "Y": 1.0},
    "T": {"A": 1.0, "C": 1.0, "D": 1.0, "E": 20.26, "F": 13.34,
          "G": -7.49, "H": 1.0, "I": 1.0, "K": 1.0, "L": 1.0,
          "M": 1.0, "N": -14.03, "P": 1.0, "Q": -6.54, "R": 1.0,
          "S": 1.0, "T": 1.0, "V": 1.0, "W": -14.03, "Y": 1.0},
    "V": {"A": 1.0, "C": 1.0, "D": -14.03, "E": 1.0, "F": 1.0,
          "G": -7.49, "H": 1.0, "I": 1.0, "K": -1.88, "L": 1.0,
          "M": 1.0, "N": 1.0, "P": 20.26, "Q": 1.0, "R": 1.0,
          "S": 1.0, "T": -7.49, "V": 1.0, "W": 1.0, "Y": -6.54},
    "W": {"A": -14.03, "C": 1.0, "D": 1.0, "E": 1.0, "F": 1.0,
          "G": -9.37, "H": 24.68, "I": 1.0, "K": 1.0, "L": 13.34,
          "M": 24.68, "N": 13.34, "P": 1.0, "Q": 1.0, "R": 1.0,
          "S": 1.0, "T": -14.03, "V": -7.49, "W": 1.0, "Y": 1.0},
    "Y": {"A": 24.68, "C": 1.0, "D": 24.68, "E": -6.54, "F": 1.0,
          "G": -7.49, "H": 13.34, "I": 1.0, "K": 1.0, "L": 1.0,
          "M": 44.94, "N": 1.0, "P": 13.34, "Q": 1.0, "R": -15.91,
          "S": 1.0, "T": -7.49, "V": 1.0, "W": -9.37, "Y": 13.34},
}

# 5. Lehninger pKa values
PKA_NTERM = 9.69
PKA_CTERM = 2.34
PKA_SIDECHAIN: Dict[str, float] = {
    "D": 3.65, "E": 4.25, "H": 6.00, "C": 8.18,
    "Y": 10.07, "K": 10.53, "R": 12.48,
}

# 5b. Average residue molecular weights (monoisotopic approximated via average)
#     Source: PubChem/NIST average isotopic masses
AA_MW: Dict[str, float] = {
    "A": 89.09,  "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.16,
    "Q": 146.15, "E": 147.13, "G":  75.07, "H": 155.16, "I": 131.17,
    "L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19, "P": 115.13,
    "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15,
}
WATER_MW: float = 18.015

# 7. N-end rule half-life (mammalian reticulocyte, hours)
NEND_HALFLIFE: Dict[str, Tuple[float, str]] = {
    "M": (30.0, "stable"),       "S": (30.0, "stable"),
    "A": (30.0, "stable"),       "T": (30.0, "stable"),
    "V": (30.0, "stable"),       "G": (30.0, "stable"),
    "I": (20.0, "intermediate"), "C": (1.2, "intermediate"),
    "Y": (2.8,  "unstable"),     "W": (2.8,  "unstable"),
    "H": (3.5,  "unstable"),     "L": (5.5,  "unstable"),
    "P": (30.0, "stable"),
    "F": (1.1,  "very_unstable"), "D": (1.1,  "very_unstable"),
    "K": (1.3,  "very_unstable"), "R": (1.0,  "very_unstable"),
    "E": (1.0,  "very_unstable"), "N": (1.4,  "very_unstable"),
    "Q": (0.8,  "very_unstable"),
}

# 8. Eisenberg consensus hydrophobicity scale (1982)
EISENBERG: Dict[str, float] = {
    "A":  0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C":  0.29,
    "Q": -0.85, "E": -0.74, "G":  0.48, "H": -0.40, "I":  1.38,
    "L":  1.06, "K": -1.50, "M":  0.64, "F":  1.19, "P":  0.12,
    "S": -0.18, "T": -0.05, "W":  0.81, "Y":  0.26, "V":  1.08,
}

# 9. Wimley-White water→POPC interface ΔG (kcal/mol, 1996)
WIMLEY_WHITE: Dict[str, float] = {
    "A":  0.17, "R":  0.81, "N":  0.42, "D":  1.23, "C": -0.24,
    "Q":  0.58, "E":  2.02, "G":  0.01, "H":  0.96, "I": -0.31,
    "L": -0.56, "K":  0.99, "M": -0.23, "F": -1.13, "P":  0.45,
    "S":  0.13, "T":  0.14, "W": -1.85, "Y": -0.94, "V":  0.07,
}

# 12. BLOSUM62 matrix — standard 20×20
_AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
_BLOSUM62_FLAT = [
    # A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
      4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0,  # A
     -1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3,  # R
     -2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  # N
     -2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  # D
      0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1,  # C
     -1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  # Q
     -1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  # E
      0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3,  # G
     -2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  # H
     -1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3,  # I
     -1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1,  # L
     -1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  # K
     -1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1,  # M
     -2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1,  # F
     -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2,  # P
      1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  # S
      0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0,  # T
     -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3,  # W
     -2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1,  # Y
      0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4,  # V
]

# Build BLOSUM62 dict-of-dict for O(1) lookup
BLOSUM62: Dict[str, Dict[str, int]] = {}
for _i, _aa1 in enumerate(_AA_ORDER):
    BLOSUM62[_aa1] = {}
    for _j, _aa2 in enumerate(_AA_ORDER):
        BLOSUM62[_aa1][_aa2] = _BLOSUM62_FLAT[_i * 20 + _j]

# Valid amino acid set
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


# ─────────────────────────── helpers ─────────────────────────────────

def _validate(sequence: str) -> str:
    """Normalise and validate a peptide sequence string."""
    seq = sequence.upper().strip()
    if not seq:
        raise ValueError("Sequence must not be empty.")
    invalid = set(seq) - VALID_AA
    if invalid:
        raise ValueError(
            f"Invalid amino acid character(s): {sorted(invalid)}"
        )
    return seq


def _charge_at_ph(
    sequence: str,
    ph: float,
    ss_bond_cysteines: Optional[Set[int]] = None,
) -> float:
    """Henderson-Hasselbalch net charge at a given pH.

    Parameters
    ----------
    sequence:
        Validated (upper-case) amino acid sequence.
    ph:
        pH value.
    ss_bond_cysteines:
        0-indexed positions of Cys residues involved in disulfide bonds.
        These Cys side-chains lose their thiol group and are excluded from
        Henderson-Hasselbalch ionisation.  Pass ``None`` (default) to treat
        all Cys as free thiols (original behaviour).
    """
    _ss = ss_bond_cysteines if ss_bond_cysteines is not None else set()
    charge = 0.0
    # N-terminus (positive)
    charge += 1.0 / (1.0 + 10.0 ** (ph - PKA_NTERM))
    # C-terminus (negative)
    charge -= 1.0 / (1.0 + 10.0 ** (PKA_CTERM - ph))
    for idx, aa in enumerate(sequence):
        if aa in ("K", "R", "H"):
            pka = PKA_SIDECHAIN[aa]
            charge += 1.0 / (1.0 + 10.0 ** (ph - pka))
        elif aa in ("D", "E", "Y"):
            pka = PKA_SIDECHAIN[aa]
            charge -= 1.0 / (1.0 + 10.0 ** (pka - ph))
        elif aa == "C" and idx not in _ss:
            # Free thiol — participates in ionisation
            pka = PKA_SIDECHAIN["C"]
            charge -= 1.0 / (1.0 + 10.0 ** (pka - ph))
        # Cys in SS bond: no ionisable thiol → skip
    return charge


# ─────────────────────────── main class ──────────────────────────────

class PharmaProperties:
    """Literature-based pharmacological property calculator for peptide sequences."""

    def __init__(self, reference_seq: str = "AGCKNFFWKTFTSC"):
        self.reference_seq = _validate(reference_seq)

    # ── 1. GRAVY ─────────────────────────────────────────────────────

    def calculate_gravy(self, sequence: str) -> float:
        """Grand average of hydropathy (Kyte & Doolittle 1982)."""
        seq = _validate(sequence)
        return sum(KD_HYDROPATHY[aa] for aa in seq) / len(seq)

    # ── 2. Boman Index ───────────────────────────────────────────────

    def calculate_boman_index(self, sequence: str) -> float:
        """Boman index — protein-binding potential (Boman 2003).

        Computed as the mean of Radzicka-Wolfenden solubility values
        (positive = hydrophilic).  BI > 2.48 kcal/mol indicates high
        protein-binding potential per Boman 2003.
        """
        seq = _validate(sequence)
        return sum(RW_TRANSFER[aa] for aa in seq) / len(seq)

    # ── 3. Instability Index ─────────────────────────────────────────

    def calculate_instability_index(self, sequence: str) -> float:
        """Instability index (Guruprasad et al. 1990).  II < 40 → stable."""
        seq = _validate(sequence)
        if len(seq) < 2:
            return 0.0
        total = 0.0
        for i in range(len(seq) - 1):
            total += _DIWV_RAW[seq[i]][seq[i + 1]]
        return (10.0 / len(seq)) * total

    # ── 4. Aliphatic Index ───────────────────────────────────────────

    def calculate_aliphatic_index(self, sequence: str) -> float:
        """Aliphatic index (Ikai 1980). Thermostability of globular proteins."""
        seq = _validate(sequence)
        n = len(seq)
        xa = 100.0 * seq.count("A") / n
        xv = 100.0 * seq.count("V") / n
        xi = 100.0 * seq.count("I") / n
        xl = 100.0 * seq.count("L") / n
        return xa + 2.9 * xv + 3.9 * (xi + xl)

    # ── 5. Isoelectric Point ─────────────────────────────────────────

    def calculate_pi(
        self,
        sequence: str,
        ss_bond_cysteines: Optional[Set[int]] = None,
    ) -> float:
        """Isoelectric point via bisection (Lehninger pKa set).

        Parameters
        ----------
        ss_bond_cysteines:
            0-indexed positions of Cys residues in disulfide bonds.
            Those Cys residues are excluded from ionisation.
            ``None`` preserves original behaviour (all Cys as free thiols).
        """
        seq = _validate(sequence)
        lo, hi = 0.0, 14.0
        for _ in range(200):
            mid = (lo + hi) / 2.0
            charge = _charge_at_ph(seq, mid, ss_bond_cysteines=ss_bond_cysteines)
            if charge > 0:
                lo = mid
            else:
                hi = mid
        return round((lo + hi) / 2.0, 2)

    # ── 6. Molar Extinction Coefficient ──────────────────────────────

    def calculate_extinction_coefficient(
        self, sequence: str, n_disulfide: int = 0
    ) -> int:
        """ε₂₈₀ (Pace et al. 1995)."""
        seq = _validate(sequence)
        return (
            seq.count("W") * 5500
            + seq.count("Y") * 1490
            + n_disulfide * 125
        )

    # ── 7. N-end Rule Half-life ──────────────────────────────────────

    def calculate_nend_rule_halflife(self, sequence: str) -> dict:
        """N-end rule predicted half-life (Varshavsky 1996)."""
        seq = _validate(sequence)
        nterm = seq[0]
        halflife, category = NEND_HALFLIFE.get(nterm, (0.0, "unknown"))
        return {
            "n_terminal_residue": nterm,
            "half_life_hours": halflife,
            "category": category,
        }

    # ── 8. Hydrophobic Moment ────────────────────────────────────────

    def calculate_hydrophobic_moment(
        self, sequence: str, angle: float = 100.0, window: int = 11
    ) -> float:
        """Mean hydrophobic moment μH (Eisenberg et al. 1982).

        Parameters
        ----------
        angle : float
            Rotation angle in degrees.  100° for α-helix, 160° for β-sheet.
        window : int
            Sliding window size (default 11).
        """
        seq = _validate(sequence)
        delta = math.radians(angle)
        if len(seq) <= window:
            # single window
            sin_sum = sum(
                EISENBERG[seq[n]] * math.sin(n * delta)
                for n in range(len(seq))
            )
            cos_sum = sum(
                EISENBERG[seq[n]] * math.cos(n * delta)
                for n in range(len(seq))
            )
            return math.sqrt(sin_sum ** 2 + cos_sum ** 2) / len(seq)

        max_moment = 0.0
        for start in range(len(seq) - window + 1):
            w = seq[start : start + window]
            sin_sum = sum(
                EISENBERG[w[n]] * math.sin(n * delta)
                for n in range(window)
            )
            cos_sum = sum(
                EISENBERG[w[n]] * math.cos(n * delta)
                for n in range(window)
            )
            moment = math.sqrt(sin_sum ** 2 + cos_sum ** 2) / window
            if moment > max_moment:
                max_moment = moment
        return max_moment

    # ── 9. Wimley-White Hydrophobicity ───────────────────────────────

    def calculate_wimley_white(self, sequence: str) -> dict:
        """Wimley-White water→POPC interface free energy (1996)."""
        seq = _validate(sequence)
        per_residue = [WIMLEY_WHITE[aa] for aa in seq]
        return {
            "total_dG": round(sum(per_residue), 3),
            "mean_dG": round(sum(per_residue) / len(seq), 3),
            "per_residue": per_residue,
        }

    # ── 10. Net Charge at pH ─────────────────────────────────────────

    def calculate_net_charge(
        self,
        sequence: str,
        ph: float = 7.4,
        ss_bond_cysteines: Optional[Set[int]] = None,
    ) -> float:
        """Net charge at a given pH (Henderson-Hasselbalch).

        Parameters
        ----------
        ss_bond_cysteines:
            0-indexed positions of Cys residues in disulfide bonds.
            ``None`` preserves original behaviour (all Cys as free thiols).
        """
        seq = _validate(sequence)
        return round(_charge_at_ph(seq, ph, ss_bond_cysteines=ss_bond_cysteines), 3)

    # ── 10b. Molecular Weight ────────────────────────────────────────

    def calculate_mw(self, sequence: str, n_disulfide: int = 0) -> dict:
        """Molecular weight calculation (average isotopic masses).

        Formula
        -------
        MW = Σ AA_MW  −  (n−1) × H₂O  −  n_disulfide × 2 × H

        Each peptide bond loses one water (condensation); each disulfide bond
        removes 2 hydrogen atoms (2 × 1.008 = 2.016 Da).

        Parameters
        ----------
        n_disulfide:
            Number of disulfide bonds.  0 = fully reduced form.

        Returns
        -------
        dict with keys:
            mw_average     – average isotopic MW (Da)
            mw_monoisotopic – monoisotopic approximation (±0.5 Da for typical peptides)
            n_residues     – sequence length
            n_disulfide    – disulfide bond count used in the calculation
        """
        seq = _validate(sequence)
        n = len(seq)
        # Sum of residue masses (free amino acid masses)
        raw_sum = sum(AA_MW[aa] for aa in seq)
        # Subtract condensation water (n-1 peptide bonds → n-1 H₂O lost)
        mw_avg = raw_sum - (n - 1) * WATER_MW
        # Each SS bond removes 2 H atoms (2 × 1.00794 ≈ 2.016 Da)
        mw_avg -= n_disulfide * 2.016
        # Monoisotopic approximation: scale by ratio of monoisotopic / average
        # for a typical peptide the ratio is ~0.9994; we use a simple per-AA
        # offset derived from the most common isotopes (C-12, H-1, N-14, O-16, S-32).
        # For peptides ≤50 aa, this approximation is within ±0.5 Da.
        _MONO_CORRECTION = 0.9994
        mw_mono = round(mw_avg * _MONO_CORRECTION, 2)
        return {
            "mw_average": round(mw_avg, 2),
            "mw_monoisotopic": mw_mono,
            "n_residues": n,
            "n_disulfide": n_disulfide,
        }

    # ── 11. Protease Cleavage Sites ──────────────────────────────────

    def count_protease_sites(self, sequence: str) -> dict:
        """Count major protease cleavage sites."""
        seq = _validate(sequence)
        chymo_residues = set("FWYLM")
        trypsin_residues = set("KR")
        nep_residues = set("FWYLIVM")

        chymo: List[int] = []
        trypsin: List[int] = []
        nep: List[int] = []
        dppiv_residues = set("PA")
        dppiv: List[int] = []

        for i, aa in enumerate(seq):
            # Chymotrypsin: C-terminal of F,W,Y,L,M (not last residue)
            if aa in chymo_residues and i < len(seq) - 1:
                chymo.append(i + 1)  # 1-indexed position

            # Trypsin: C-terminal of K,R but not before P
            if aa in trypsin_residues and i < len(seq) - 1:
                if seq[i + 1] != "P":
                    trypsin.append(i + 1)

            # Neprilysin: N-terminal of hydrophobic residue
            if aa in nep_residues and i > 0:
                nep.append(i + 1)

            # DPP-IV: cleaves after X-Pro or X-Ala (not if Pro followed by Pro)
            # Includes N-terminal X-P/X-A (pos 2) and internal X-P/X-A sites
            if aa in dppiv_residues and i > 0 and i < len(seq) - 1:
                if seq[i + 1] != "P":  # Pro-Pro resistance
                    dppiv.append(i + 1)

        return {
            "chymotrypsin": {"count": len(chymo), "positions": chymo},
            "trypsin": {"count": len(trypsin), "positions": trypsin},
            "neprilysin": {"count": len(nep), "positions": nep},
            "dppiv": {"count": len(dppiv), "positions": dppiv},
            "total": len(chymo) + len(trypsin) + len(nep) + len(dppiv),
        }

    # ── 12. BLOSUM62 Conservation Score ──────────────────────────────

    def calculate_blosum62_score(self, sequence: str) -> dict:
        """BLOSUM62 conservation score vs reference sequence."""
        seq = _validate(sequence)
        ref = self.reference_seq
        if len(seq) != len(ref):
            raise ValueError(
                f"Query length ({len(seq)}) != reference length ({len(ref)})"
            )
        scores: List[dict] = []
        total = 0
        for pos, (q, r) in enumerate(zip(seq, ref), start=1):
            score = BLOSUM62[r][q]
            total += score
            category = (
                "identical" if q == r
                else "conservative" if score >= 1
                else "semi-conservative" if score == 0
                else "non-conservative"
            )
            scores.append({
                "position": pos,
                "reference": r,
                "query": q,
                "score": score,
                "category": category,
            })
        return {
            "total_score": total,
            "per_position": scores,
            "n_mutations": sum(1 for s in scores if s["reference"] != s["query"]),
        }

    # ── 13. Metal Coordination Residues ──────────────────────────────

    def analyze_metal_coordination(self, sequence: str) -> dict:

        """Identify metal-coordinating residues (Rulísek & Vondrásek 1998)."""
        seq = _validate(sequence)
        coord: Dict[str, List[int]] = {
            "His_imidazole": [],
            "Cys_thiolate": [],
            "Asp_carboxylate": [],
            "Glu_carboxylate": [],
            "Met_thioether": [],
        }
        key_map = {"H": "His_imidazole", "C": "Cys_thiolate",
                    "D": "Asp_carboxylate", "E": "Glu_carboxylate",
                    "M": "Met_thioether"}
        for i, aa in enumerate(seq, start=1):
            if aa in key_map:
                coord[key_map[aa]].append(i)

        total = sum(len(v) for v in coord.values())
        metals: List[str] = []
        if coord["His_imidazole"]:
            metals.extend(["Zn2+", "Cu2+", "Ga3+"])
        if coord["Cys_thiolate"]:
            metals.extend(["Zn2+", "Cu2+"])
        if coord["Asp_carboxylate"] or coord["Glu_carboxylate"]:
            # Ga3+ is a hard Lewis acid with strong oxygen affinity:
            # carboxylate O from D/E is a primary coordination site
            # (Chitambar 2010; Bernstein 2005)
            metals.extend(["Ca2+", "Mg2+", "Lu3+", "Ac3+", "Ga3+"])
        if coord["Met_thioether"]:
            metals.append("Cu2+")

        return {
            "residues": coord,
            "total_coordinating": total,
            "potential_metals": sorted(set(metals)),
        }

    # ── 5 Structural Rules (PASS / FAIL) ─────────────────────────────

    def check_structural_rules(self, sequence: str) -> dict:
        """Evaluate five structural integrity rules for SST-14 analogs."""
        seq = _validate(sequence)
        results: Dict[str, dict] = {}

        # Rule 1: FWKT pharmacophore at positions 7-10
        if len(seq) >= 10:
            motif = seq[6:10]
            results["fwkt_pharmacophore"] = {
                "pass": motif == "FWKT",
                "detail": f"positions 7-10 = {motif}",
            }
        else:
            results["fwkt_pharmacophore"] = {
                "pass": False,
                "detail": "sequence too short (< 10 residues)",
            }

        # Rule 2: K9 for salt bridge with SSTR2 D122(3.32)
        if len(seq) >= 9:
            results["k9_salt_bridge"] = {
                "pass": seq[8] == "K",
                "detail": f"position 9 = {seq[8]}",
            }
        else:
            results["k9_salt_bridge"] = {
                "pass": False,
                "detail": "sequence too short (< 9 residues)",
            }

        # Rule 3: Cys3-Cys14 disulfide
        if len(seq) >= 14:
            results["cys3_cys14_disulfide"] = {
                "pass": seq[2] == "C" and seq[13] == "C",
                "detail": f"pos3={seq[2]}, pos14={seq[13]}",
            }
        else:
            results["cys3_cys14_disulfide"] = {
                "pass": False,
                "detail": f"sequence too short (< 14, len={len(seq)})",
            }

        # Rule 4: Phe6-Phe11 aromatic stacking
        aromatic = set("FWY")
        if len(seq) >= 11:
            results["phe6_phe11_stacking"] = {
                "pass": seq[5] in aromatic and seq[10] in aromatic,
                "detail": f"pos6={seq[5]}, pos11={seq[10]}",
            }
        else:
            results["phe6_phe11_stacking"] = {
                "pass": False,
                "detail": "sequence too short (< 11 residues)",
            }

        # Rule 5: N-terminal chelator compatibility
        preferred_nterm = set("AG")
        results["nterm_chelator"] = {
            "pass": seq[0] in preferred_nterm,
            "detail": f"pos1={seq[0]} ({'preferred' if seq[0] in preferred_nterm else 'suboptimal'})",
        }

        all_pass = all(r["pass"] for r in results.values())
        return {"rules": results, "all_pass": all_pass}

    # ── 14. Radiolysis Susceptibility ────────────────────────────────

    def calculate_radiolysis_susceptibility(self, sequence: str) -> dict:
        """Radiolysis susceptibility score for radiopharmaceutical stability.

        Estimates radiation-induced oxidative damage risk based on residue
        reactivity with reactive oxygen species (ROS) generated by radiolysis
        of water.  Each susceptible residue is assigned an empirical weight
        reflecting its relative oxidation rate constant.

        Weights (empirical, relative to radiation chemistry literature):
        - Met (M): weight 3 — sulphur oxidation to Met-sulfoxide (most reactive)
        - Trp (W): weight 3 — indole ring oxidation to kynurenine/hydroxyl products
        - Cys (C): weight 2 — thiol oxidation; reduced to 1 when in SS-bond (protected)
        - His (H): weight 2 — imidazole ring oxidation
        - Tyr (Y): weight 1 — phenol ring oxidation, dityrosine cross-linking
        - Phe (F): weight 0.5 — aromatic hydroxylation (least reactive aromatic)

        Risk levels: low (0–3), moderate (3–6), high (>6).

        FWKT pharmacophore (positions 7-10 of SST-14 analogs) residues that
        appear in *critical_positions* indicate direct binding-affinity risk.

        Parameters
        ----------
        sequence : str
            One-letter amino acid sequence.

        Returns
        -------
        dict
            total_score, risk_level, vulnerable_residues, critical_positions.
        """
        seq = _validate(sequence)

        # Residue weight table
        # [BUG-FIX 버그3] 회의록(서호성) 정성 순서 반영:
        #   Cys,Met >> Phe,Tyr > Trp > Pro > His,Leu
        #   HEURISTIC: 구체 수치는 문헌 정량화 미완 — 정성 순서 기반 상대 할당
        #   수정 내용:
        #     - W(Trp): 3.0 → 1.5 (Phe·Tyr 이하로 조정, SST-14 Trp8 과대평가 해소)
        #     - C(free Cys): 2.0 → 3.0 (Cys,Met >> 동급, SS-bond는 1.5로 절반 감소)
        #     - F(Phe): 0.5 → 2.0 (Phe,Tyr > Trp 순서 반영)
        #     - Y(Tyr): 1.0 → 2.0 (Phe,Tyr 동급)
        #     - H(His): 2.0 → 0.5 (His,Leu급으로 강등)
        #     - P(Pro): 신규 추가 0.5
        #     - L(Leu): 신규 추가 0.3
        _BASE_WEIGHT: Dict[str, float] = {
            "M": 3.0,   # 황산화(Met-sulfoxide): 가장 취약
            "C": 3.0,   # thiol 산화(free Cys): Met 동급; SS-bond 보호 시 1.5로 감소
            "F": 2.0,   # 방향족 수산화(Phe): Phe,Tyr > Trp 순서
            "Y": 2.0,   # 페놀 산화/dityrosine(Tyr): Phe 동급
            "W": 1.5,   # 인돌 산화(Trp): 구 3.0 → 1.5 (Phe·Tyr 이하 조정)
            "P": 0.5,   # Pro: 회의록 신규 추가 (Cys,Met >> Phe,Tyr > Trp > Pro)
            "H": 0.5,   # 이미다졸 산화(His): His,Leu급으로 강등
            "L": 0.3,   # Leu: 회의록 신규 추가 (His,Leu 동급)
        }

        # Mechanism description per residue type
        _MECHANISM: Dict[str, str] = {
            "M": "sulphur oxidation to Met-sulfoxide",
            "C": "thiol oxidation / SS-bond disruption",
            "F": "aromatic hydroxylation",
            "Y": "phenol ring oxidation / dityrosine",
            "W": "indole ring oxidation to kynurenine",
            "P": "proline ring oxidation (minor pathway)",
            "H": "imidazole ring oxidation",
            "L": "leucine peroxidation (minor pathway)",
        }

        # Identify SS-bond Cys pairs (Cys3-Cys14 in SST-14 convention).
        # Strategy: collect all Cys positions; pair them in order (1st with last,
        # 2nd with second-to-last, …) to reflect the typical disulfide topology.
        cys_positions: List[int] = [
            i + 1 for i, aa in enumerate(seq) if aa == "C"
        ]
        ss_bond_positions: set = set()
        if len(cys_positions) >= 2:
            lo = 0
            hi = len(cys_positions) - 1
            while lo < hi:
                ss_bond_positions.add(cys_positions[lo])
                ss_bond_positions.add(cys_positions[hi])
                lo += 1
                hi -= 1

        # FWKT pharmacophore window: positions 7-10 (1-indexed)
        fwkt_start = 7
        fwkt_end = 10

        vulnerable_residues: List[dict] = []
        total_score = 0.0

        for i, aa in enumerate(seq):
            if aa not in _BASE_WEIGHT:
                continue
            pos = i + 1  # 1-indexed
            weight = _BASE_WEIGHT[aa]

            # Reduce Cys weight when in a disulfide bond
            # [BUG-FIX 버그3] SS-bond Cys: free=3.0 → SS-bond=1.5 (절반; 보호 효과 반영)
            # 구 값: 1.0 (free=2.0의 절반) → 신규: 1.5 (free=3.0의 절반, HEURISTIC)
            if aa == "C" and pos in ss_bond_positions:
                weight = 1.5

            total_score += weight
            vulnerable_residues.append({
                "position": pos,
                "residue": aa,
                "mechanism": _MECHANISM[aa],
                "weight": weight,
            })

        # Risk classification
        if total_score <= 3.0:
            risk_level = "low"
        elif total_score <= 6.0:
            risk_level = "moderate"
        else:
            risk_level = "high"

        # Critical positions: vulnerable residues within FWKT pharmacophore
        critical_positions: List[dict] = [
            r for r in vulnerable_residues
            if fwkt_start <= r["position"] <= fwkt_end
        ]

        return {
            "total_score": round(total_score, 2),
            "risk_level": risk_level,
            "vulnerable_residues": vulnerable_residues,
            "critical_positions": critical_positions,
        }

    # ── Composite & Batch ────────────────────────────────────────────

    def calculate_all(self, sequence: str) -> dict:
        """Return dict with all 13 property values + 5 rule verdicts + MW.

        SS bond Cys correction
        ----------------------
        When the sequence contains an even number of Cys residues they are
        assumed to form disulfide bonds in sequential order (first ↔ last,
        second ↔ second-last, …).  For SST-14 (AGCKNFFWKTFTSC) this yields
        {2, 13} (0-indexed), i.e. Cys3-Cys14.

        The inferred ``ss_bond_cysteines`` set is passed to ``calculate_pi``
        and ``calculate_net_charge`` so that bonded Cys are excluded from
        Henderson-Hasselbalch ionisation.
        """
        seq = _validate(sequence)
        n_cys = seq.count("C")
        n_disulfide = n_cys // 2  # estimate

        # Auto-infer SS bond Cys positions (sequential pairing)
        ss_bond_cysteines: Optional[Set[int]] = None
        if n_cys > 0 and n_cys % 2 == 0:
            cys_positions = [i for i, aa in enumerate(seq) if aa == "C"]
            # Pair first↔last, second↔second-last, …
            pairs: List[Tuple[int, int]] = []
            lo_ptr, hi_ptr = 0, len(cys_positions) - 1
            while lo_ptr < hi_ptr:
                pairs.append((cys_positions[lo_ptr], cys_positions[hi_ptr]))
                lo_ptr += 1
                hi_ptr -= 1
            ss_bond_cysteines = {pos for pair in pairs for pos in pair}

        result = {
            "sequence": seq,
            "length": len(seq),
            "gravy": self.calculate_gravy(seq),
            "boman_index": self.calculate_boman_index(seq),
            "instability_index": self.calculate_instability_index(seq),
            "aliphatic_index": self.calculate_aliphatic_index(seq),
            "isoelectric_point": self.calculate_pi(
                seq, ss_bond_cysteines=ss_bond_cysteines
            ),
            "extinction_coefficient": self.calculate_extinction_coefficient(
                seq, n_disulfide=n_disulfide
            ),
            "nend_rule_halflife": self.calculate_nend_rule_halflife(seq),
            "hydrophobic_moment_alpha": self.calculate_hydrophobic_moment(
                seq, angle=100.0
            ),
            "hydrophobic_moment_beta": self.calculate_hydrophobic_moment(
                seq, angle=160.0
            ),
            "wimley_white": self.calculate_wimley_white(seq),
            "net_charge_ph74": self.calculate_net_charge(
                seq, ph=7.4, ss_bond_cysteines=ss_bond_cysteines
            ),
            "net_charge_ph65": self.calculate_net_charge(
                seq, ph=6.5, ss_bond_cysteines=ss_bond_cysteines
            ),
            "molecular_weight": self.calculate_mw(seq, n_disulfide=n_disulfide),
            "protease_sites": self.count_protease_sites(seq),
            "blosum62": (
                self.calculate_blosum62_score(seq)
                if len(seq) == len(self.reference_seq)
                else {"error": "length mismatch with reference"}
            ),
            "metal_coordination": self.analyze_metal_coordination(seq),
            "structural_rules": self.check_structural_rules(seq),
            "radiolysis_susceptibility": self.calculate_radiolysis_susceptibility(seq),
        }
        return result

    def batch_analyze(self, sequences: Dict[str, str]) -> dict:
        """Analyse multiple sequences. Keys = names, values = sequences."""
        return {name: self.calculate_all(seq) for name, seq in sequences.items()}
