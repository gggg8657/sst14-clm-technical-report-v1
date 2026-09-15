"""
Tests for SS bond Cys pI/charge correction and MW calculation.

Covers:
- _charge_at_ph / calculate_pi / calculate_net_charge with ss_bond_cysteines
- calculate_mw
- calculate_all integration (auto-inferred SS bond + MW key)
- Backward-compatibility: None default preserves original behaviour
"""

from __future__ import annotations

import sys
import os
import unittest

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    ),
)

from AG_src.pipeline.pharma_properties import (
    PharmaProperties,
    _charge_at_ph,
)

SST14 = "AGCKNFFWKTFTSC"
# Cys3 = index 2,  Cys14 = index 13  (0-indexed)
SST14_SS_CYSTEINES = {2, 13}


# ─── SS bond pI / charge ─────────────────────────────────────────────────────


class TestSSBondChargeAtPh(unittest.TestCase):
    """Low-level _charge_at_ph with ss_bond_cysteines."""

    def test_default_none_same_as_no_arg(self):
        """Passing ss_bond_cysteines=None must give the same result as calling
        without the argument (backward compatibility)."""
        c_default = _charge_at_ph(SST14, 7.4)
        c_none = _charge_at_ph(SST14, 7.4, ss_bond_cysteines=None)
        self.assertAlmostEqual(c_default, c_none, places=10)

    def test_ss_bond_increases_charge_at_neutral_ph(self):
        """Removing thiol ionisation from 2 Cys raises net charge at pH 7.4."""
        c_free = _charge_at_ph(SST14, 7.4)
        c_ss = _charge_at_ph(SST14, 7.4, ss_bond_cysteines=SST14_SS_CYSTEINES)
        self.assertGreater(c_ss, c_free)

    def test_empty_ss_set_same_as_free(self):
        """Empty set {} means no Cys are bonded — identical to free-thiol."""
        c_free = _charge_at_ph(SST14, 7.4)
        c_empty = _charge_at_ph(SST14, 7.4, ss_bond_cysteines=set())
        self.assertAlmostEqual(c_free, c_empty, places=10)

    def test_partial_ss_bond(self):
        """Bonding only one Cys gives intermediate charge."""
        c_free = _charge_at_ph(SST14, 7.4)
        c_ss_full = _charge_at_ph(SST14, 7.4, ss_bond_cysteines=SST14_SS_CYSTEINES)
        c_ss_one = _charge_at_ph(SST14, 7.4, ss_bond_cysteines={2})
        # one bonded Cys → charge between free-thiol and fully bonded
        self.assertGreater(c_ss_one, c_free)
        self.assertLess(c_ss_one, c_ss_full)


class TestSSBondCalculatePi(unittest.TestCase):
    """calculate_pi with ss_bond_cysteines parameter."""

    def setUp(self):
        self.pp = PharmaProperties()

    def test_sst14_pi_increases_with_ss_bond(self):
        """Cys3-Cys14 SS bond removes negative charge → pI must increase."""
        pi_free = self.pp.calculate_pi(SST14)
        pi_ss = self.pp.calculate_pi(SST14, ss_bond_cysteines=SST14_SS_CYSTEINES)
        self.assertGreater(pi_ss, pi_free)

    def test_sst14_pi_old_value(self):
        """Without SS bond correction, SST-14 pI ≈ 9.04 (historical baseline)."""
        pi = self.pp.calculate_pi(SST14)
        self.assertAlmostEqual(pi, 9.04, delta=0.05)

    def test_sst14_pi_corrected_value(self):
        """With SS bond correction, SST-14 pI ≈ 10.62."""
        pi = self.pp.calculate_pi(SST14, ss_bond_cysteines=SST14_SS_CYSTEINES)
        self.assertAlmostEqual(pi, 10.62, delta=0.10)

    def test_backward_compat_none(self):
        """Explicit None gives same result as calling without the argument."""
        pi_default = self.pp.calculate_pi(SST14)
        pi_none = self.pp.calculate_pi(SST14, ss_bond_cysteines=None)
        self.assertAlmostEqual(pi_default, pi_none, places=10)

    def test_all_cys_bonded_sequence(self):
        """Sequence with 4 Cys — all bonded → higher pI than all-free."""
        seq = "ACACACAC"
        ss = {1, 3, 5, 7}
        pi_free = self.pp.calculate_pi(seq)
        pi_ss = self.pp.calculate_pi(seq, ss_bond_cysteines=ss)
        self.assertGreater(pi_ss, pi_free)

    def test_no_cys_sequence_unaffected(self):
        """Sequence without Cys — SS bond set has no effect."""
        seq = "AGKNFFWKTFTS"
        pi_none = self.pp.calculate_pi(seq)
        pi_with = self.pp.calculate_pi(seq, ss_bond_cysteines={0, 5})
        self.assertAlmostEqual(pi_none, pi_with, places=10)


class TestSSBondCalculateNetCharge(unittest.TestCase):
    """calculate_net_charge with ss_bond_cysteines parameter."""

    def setUp(self):
        self.pp = PharmaProperties()

    def test_backward_compat_default(self):
        """Default (no ss_bond_cysteines) behaves identically to original."""
        c1 = self.pp.calculate_net_charge(SST14, ph=7.4)
        c2 = self.pp.calculate_net_charge(SST14, ph=7.4, ss_bond_cysteines=None)
        self.assertAlmostEqual(c1, c2, places=10)

    def test_ss_bond_increases_charge(self):
        """Bonded Cys excluded → net charge at pH 7.4 should increase."""
        c_free = self.pp.calculate_net_charge(SST14, ph=7.4)
        c_ss = self.pp.calculate_net_charge(
            SST14, ph=7.4, ss_bond_cysteines=SST14_SS_CYSTEINES
        )
        self.assertGreater(c_ss, c_free)

    def test_sst14_positive_charge_corrected(self):
        """Corrected SST-14 charge at pH 7.4 should be positive and > uncorrected."""
        c = self.pp.calculate_net_charge(
            SST14, ph=7.4, ss_bond_cysteines=SST14_SS_CYSTEINES
        )
        self.assertGreater(c, 0.0)

    def test_different_ph_values(self):
        """SS bond correction works at multiple pH values."""
        for ph in [2.0, 6.5, 7.4, 10.0]:
            c_free = self.pp.calculate_net_charge(SST14, ph=ph)
            c_ss = self.pp.calculate_net_charge(
                SST14, ph=ph, ss_bond_cysteines=SST14_SS_CYSTEINES
            )
            # SS bonded Cys removes a negative contribution at all pH
            # At very low pH Cys is protonated anyway → negligible difference
            self.assertGreaterEqual(c_ss, c_free - 1e-9)


class TestCalculateAllSSBondIntegration(unittest.TestCase):
    """calculate_all must auto-infer SS bond and apply correction."""

    def setUp(self):
        self.pp = PharmaProperties()

    def test_sst14_pi_uses_ss_correction(self):
        """SST-14 has 2 Cys (even) → auto-corrected pI > uncorrected."""
        result = self.pp.calculate_all(SST14)
        pi_corrected = result["isoelectric_point"]
        pi_uncorrected = self.pp.calculate_pi(SST14)  # no ss_bond_cysteines
        self.assertGreater(pi_corrected, pi_uncorrected)

    def test_sst14_net_charge_uses_ss_correction(self):
        """calculate_all net charges must be corrected."""
        result = self.pp.calculate_all(SST14)
        c_corrected = result["net_charge_ph74"]
        c_uncorrected = self.pp.calculate_net_charge(SST14, ph=7.4)
        self.assertGreater(c_corrected, c_uncorrected)

    def test_odd_cys_no_auto_correction(self):
        """Odd Cys count → no auto-correction (ss_bond_cysteines stays None)."""
        seq = "AGCKNFFWKTFTCC"  # 3 Cys → odd
        result = self.pp.calculate_all(seq)
        pi_all = result["isoelectric_point"]
        pi_manual = self.pp.calculate_pi(seq)  # free thiol (no correction)
        self.assertAlmostEqual(pi_all, pi_manual, places=10)

    def test_no_cys_no_correction(self):
        """No Cys → identical to uncorrected."""
        seq = "AGKNFFWKTFTS"
        result = self.pp.calculate_all(seq)
        pi_all = result["isoelectric_point"]
        pi_manual = self.pp.calculate_pi(seq)
        self.assertAlmostEqual(pi_all, pi_manual, places=10)

    def test_molecular_weight_key_present(self):
        """calculate_all must include 'molecular_weight' key."""
        result = self.pp.calculate_all(SST14)
        self.assertIn("molecular_weight", result)
        mw = result["molecular_weight"]
        self.assertIn("mw_average", mw)
        self.assertIn("mw_monoisotopic", mw)
        self.assertIn("n_residues", mw)
        self.assertIn("n_disulfide", mw)


# ─── MW calculation ──────────────────────────────────────────────────────────


class TestCalculateMW(unittest.TestCase):
    """calculate_mw correctness."""

    def setUp(self):
        self.pp = PharmaProperties()

    def test_sst14_reduced_matches_reference(self):
        """SST-14 (no SS bond) should match peptides-package result 1639.91 ±1 Da."""
        mw = self.pp.calculate_mw(SST14, n_disulfide=0)
        self.assertAlmostEqual(mw["mw_average"], 1639.91, delta=1.0)

    def test_sst14_ss_bond_reduces_mw(self):
        """1 SS bond removes 2.016 Da."""
        mw_red = self.pp.calculate_mw(SST14, n_disulfide=0)
        mw_ss = self.pp.calculate_mw(SST14, n_disulfide=1)
        self.assertAlmostEqual(
            mw_red["mw_average"] - mw_ss["mw_average"], 2.016, delta=0.01
        )

    def test_returns_dict_keys(self):
        """Return value must have all required keys."""
        mw = self.pp.calculate_mw(SST14)
        for key in ("mw_average", "mw_monoisotopic", "n_residues", "n_disulfide"):
            self.assertIn(key, mw)

    def test_n_residues_correct(self):
        self.assertEqual(self.pp.calculate_mw(SST14)["n_residues"], 14)

    def test_polyalanine_10mer(self):
        """10xAla: sum(89.09*10) - 9*18.015 = 890.9 - 162.135 = 728.765"""
        seq = "AAAAAAAAAA"  # 10 residues
        mw = self.pp.calculate_mw(seq, n_disulfide=0)
        expected = 89.09 * 10 - 9 * 18.015
        self.assertAlmostEqual(mw["mw_average"], expected, delta=0.1)

    def test_single_glycine(self):
        """Single Gly: 75.07 Da (no peptide bond condensation)."""
        mw = self.pp.calculate_mw("G", n_disulfide=0)
        self.assertAlmostEqual(mw["mw_average"], 75.07, delta=0.1)

    def test_monoisotopic_less_than_average(self):
        """Monoisotopic MW is always ≤ average MW for standard peptides."""
        mw = self.pp.calculate_mw(SST14, n_disulfide=1)
        self.assertLessEqual(mw["mw_monoisotopic"], mw["mw_average"])

    def test_all_aa_single_residue(self):
        """Each single-letter AA should produce a positive MW."""
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            mw = self.pp.calculate_mw(aa, n_disulfide=0)
            self.assertGreater(mw["mw_average"], 0.0, msg=f"Failed for {aa}")

    def test_n_disulfide_stored(self):
        """n_disulfide in return dict must reflect the argument passed."""
        mw = self.pp.calculate_mw(SST14, n_disulfide=2)
        self.assertEqual(mw["n_disulfide"], 2)


# ─── Backward-compat: existing 35 tests must still see original pI ───────────


class TestBackwardCompatibilityPi(unittest.TestCase):
    """Ensure calculate_pi() without ss_bond_cysteines matches historical value."""

    def setUp(self):
        self.pp = PharmaProperties()

    def test_sst14_pi_old_value_unchanged(self):
        """calculate_pi(SST14) with no extra args must still return ~9.04."""
        pi = self.pp.calculate_pi(SST14)
        self.assertAlmostEqual(pi, 9.04, delta=0.05)

    def test_polylys_pi_high(self):
        """Polylysine pI should still be > 10."""
        pi = self.pp.calculate_pi("KKKK")
        self.assertGreater(pi, 10.0)

    def test_polyasp_pi_low(self):
        """Polyaspartate pI should still be < 4."""
        pi = self.pp.calculate_pi("DDDD")
        self.assertLess(pi, 4.0)


if __name__ == "__main__":
    unittest.main()
