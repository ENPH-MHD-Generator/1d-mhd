"""
Tests for magnetohydrodynamics.stability.friedberg_criterion.

There's no `main_ref` golden-oracle equivalent for this physics (tests/reference_main.py
predates the paper's incorporation into this repo), so correctness is established three
ways instead: (1) hand-derived formula values computed independently in this file, not by
calling the implementation's own helpers; (2) a loose-tolerance sanity check against
Friedberg's own Sec. 7.3 worked numeric example; (3) a self-consistency check between the
exact (5.13) and asymptotic (6.23) criteria, which were transcribed from different
equations in the paper and should agree in the appropriate joint limit.
"""
import numpy as np
import pytest
from scipy import constants

from magnetohydrodynamics.stability.friedberg_criterion import FriedbergAsymptoticCriterion, FriedbergCriterion


# --- Hand-derived formula values -------------------------------------------------------
# Independently computed (not by calling `alpha`/`critical_hall_parameter`) for an
# arbitrary point: Te=6000 K, Tp=2000 K, E_I=3.894 eV (Caesium), f_I=0.5.

def _hand_alpha(Te, EI_eV, fI):
    """A differently-structured but algebraically identical rearrangement of Friedberg
    (4.3)/(7.1): alpha = (1/2)*(2kTe/(3kTe+2E_I))*(2-f_I)/(1-f_I)."""
    EI_J = EI_eV * constants.e
    return 0.5 * (2.0 * constants.k * Te / (3.0 * constants.k * Te + 2.0 * EI_J)) * (2.0 - fI) / (1.0 - fI)


def _hand_beta_crit(Te, Tp, EI_eV, fI):
    a = _hand_alpha(Te, EI_eV, fI)
    dT = Te / Tp - 1.0
    return np.sqrt(4.0 * a * (2.0 + 1.0 / dT) * (1.0 + a * (1.0 + 1.0 / dT)))


@pytest.fixture
def exact() -> FriedbergCriterion:
    return FriedbergCriterion()


@pytest.fixture
def asymptotic() -> FriedbergAsymptoticCriterion:
    return FriedbergAsymptoticCriterion()


class TestAlpha:
    def test_matches_hand_derivation(self, exact):
        expected = _hand_alpha(6000.0, 3.894, 0.5)
        assert exact.alpha(6000.0, 3.894, 0.5) == pytest.approx(expected, rel=1e-12)

    def test_finite_as_fI_approaches_zero(self, exact):
        # f_I=0: (2-f_I)/(1-f_I) = 2, so alpha is finite and nonzero (weakly ionised seed).
        result = exact.alpha(6000.0, 3.894, 0.0)
        assert np.isfinite(result)
        assert result > 0.0

    def test_diverges_as_fI_approaches_one(self, exact):
        with np.errstate(divide="ignore"):
            result = exact.alpha(6000.0, 3.894, 1.0)
        assert np.isinf(result)
        assert not np.isnan(result)


class TestCriticalHallParameterExact:
    def test_matches_hand_derivation(self, exact):
        expected = _hand_beta_crit(6000.0, 2000.0, 3.894, 0.5)
        assert exact.critical_hall_parameter(6000.0, 2000.0, 3.894, 0.5) == pytest.approx(expected, rel=1e-12)

    def test_freidberg_section_7_3_worked_example(self, exact):
        """Sec. 7.3's reference case: M=1.8, S_C=100 MW/m^3, B0=8T gives
        Te=4223 K, Te/Tp=8.783, 1-f_I=0.01679, beta=8.249 (the actual solved beta, which
        at marginal-stability design equals beta_crit).

        Tolerance is deliberately loose (10%): the published intermediates are only given
        to 4 significant figures, and that rounding gets amplified through the alpha^2
        term. This was verified during development -- reproducing Friedberg's own Sec. 7.1
        numeric procedure exactly (not just this formula) still lands ~8.6% off the stated
        8.249. Do not tighten this to chase bit-for-bit reproduction of the paper; the
        point is "right ballpark, no silent regression," not exact replication.
        """
        Te = 4223.0
        Tp = Te / 8.783
        f_I = 1.0 - 0.01679
        result = exact.critical_hall_parameter(Te, Tp, 3.894, f_I)
        assert result == pytest.approx(8.249, rel=0.10)

    def test_diverges_as_temperatures_converge(self, exact):
        """Delta_T -> 0 (Te == Tp): no electron/gas temperature difference to drive the
        mode -- unconditionally stable, beta_crit -> +inf."""
        with np.errstate(divide="ignore", invalid="ignore"):
            result = exact.critical_hall_parameter(2000.0, 2000.0, 3.894, 0.5)
        assert np.isinf(result)

    def test_diverges_as_fI_approaches_one(self, exact):
        """f_I -> 1: nearly-full seed ionisation -- unconditionally stable, per the
        paper's central thesis -- beta_crit -> +inf."""
        with np.errstate(divide="ignore", invalid="ignore"):
            result = exact.critical_hall_parameter(6000.0, 2000.0, 3.894, 1.0)
        assert np.isinf(result)

    def test_no_unhandled_warnings_at_divergent_inputs(self, exact, recwarn):
        exact.critical_hall_parameter(2000.0, 2000.0, 3.894, 1.0)
        assert len(recwarn) == 0

    def test_array_broadcast_matches_scalar_loop(self, exact):
        Te = np.linspace(3000.0, 8000.0, 5)
        Tp = 2000.0
        f_I = np.linspace(0.1, 0.9, 5)
        result = exact.critical_hall_parameter(Te, Tp, 3.894, f_I)
        expected = np.array([
            exact.critical_hall_parameter(float(te), Tp, 3.894, float(fi)) for te, fi in zip(Te, f_I)
        ])
        assert result.shape == (5,)
        np.testing.assert_allclose(result, expected, rtol=1e-12)


class TestCriticalHallParameterAsymptotic:
    def test_matches_hand_derivation(self, asymptotic):
        EI_J = 3.894 * constants.e
        expected = np.sqrt(2.0) * (constants.k * 6000.0 / EI_J) / (1.0 - 0.9)
        assert asymptotic.critical_hall_parameter(6000.0, np.nan, 3.894, 0.9) == pytest.approx(expected, rel=1e-12)

    def test_ignores_gas_temperature(self, asymptotic):
        """gas_temperature is only accepted for StabilityModel interface compatibility --
        the asymptotic limit has no Tp dependence."""
        a = asymptotic.critical_hall_parameter(6000.0, 500.0, 3.894, 0.9)
        b = asymptotic.critical_hall_parameter(6000.0, 5000.0, 3.894, 0.9)
        assert a == pytest.approx(b, rel=1e-12)

    def test_diverges_as_fI_approaches_one(self, asymptotic):
        with np.errstate(divide="ignore"):
            result = asymptotic.critical_hall_parameter(6000.0, np.nan, 3.894, 1.0)
        assert np.isinf(result)

    def test_array_broadcast_matches_scalar_loop(self, asymptotic):
        f_I = np.linspace(0.9, 0.999, 4)
        result = asymptotic.critical_hall_parameter(6000.0, np.nan, 3.894, f_I)
        expected = np.array([asymptotic.critical_hall_parameter(6000.0, np.nan, 3.894, float(fi)) for fi in f_I])
        np.testing.assert_allclose(result, expected, rtol=1e-12)


class TestExactVsAsymptoticSelfConsistency:
    """The exact (5.13) and asymptotic (6.23) criteria were transcribed from different
    equations in the paper; (6.23) is only valid in the joint limit f_I -> 1 *and*
    Delta_T -> infinity (Friedberg's own ordering scheme, eq. 6.24) *and* kTe/E_I << 1.
    Holding Tp fixed while sweeping f_I -> 1 alone does NOT converge the two formulas
    (verified during development -- the ratio settles near a constant offset, not 1),
    because Delta_T stays finite. This test instead scales Delta_T up alongside f_I (one
    valid choice consistent with eq. 6.24's ordering) and uses an artificially large E_I
    to keep kTe/E_I comfortably small, isolating the f_I/Delta_T ordering from that
    separate assumption."""

    def test_ratio_converges_to_one_in_the_joint_limit(self, exact, asymptotic):
        Te, E_I = 2000.0, 100.0  # kTe/E_I ~ 1.7e-3, comfortably << 1
        f_I_values = [0.999, 0.9999, 0.99999, 0.999999]
        deviations = []
        for f_I in f_I_values:
            delta_T = 1.0 / np.sqrt(1.0 - f_I)  # grows as f_I -> 1
            Tp = Te / (1.0 + delta_T)
            exact_beta_crit = exact.critical_hall_parameter(Te, Tp, E_I, f_I)
            asymptotic_beta_crit = asymptotic.critical_hall_parameter(Te, np.nan, E_I, f_I)
            deviations.append(abs(exact_beta_crit / asymptotic_beta_crit - 1.0))

        # Each step toward f_I -> 1 (with Delta_T scaled up alongside it) should bring the
        # two formulas closer together.
        for earlier, later in zip(deviations, deviations[1:]):
            assert later < earlier
        assert deviations[-1] < 0.01
