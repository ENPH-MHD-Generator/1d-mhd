"""
Tests for magnetohydrodynamics.stability.design_point.MarginalDesignPointSolver --
newly extracted from examples/design_curves.py, no prior test coverage (only checked
via a printed, manually-eyeballed validation in that example script). Reproduces the
same Sec. 7.3 loose-tolerance check test_friedberg_criterion.py already applies to the
underlying criterion, plus internal-consistency checks specific to the iterative design
procedure itself.
"""
import numpy as np
import pytest
from scipy import constants

from magnetohydrodynamics.stability.design_point import MarginalDesignPointSolver

REFERENCE_GAS_TEMPERATURE = 481.0
REFERENCE_GAS_PRESSURE = 0.801e6
REFERENCE_MACH_NUMBER = 1.8
REFERENCE_FLOW_SPEED = 735.0
REFERENCE_GAS_NUMBER_DENSITY = REFERENCE_GAS_PRESSURE / (constants.k * REFERENCE_GAS_TEMPERATURE)


@pytest.fixture
def solver(gas_type, seed_type) -> MarginalDesignPointSolver:
    return MarginalDesignPointSolver(
        gas_type, seed_type,
        gas_temperature=REFERENCE_GAS_TEMPERATURE, gas_number_density=REFERENCE_GAS_NUMBER_DENSITY,
        flow_speed=REFERENCE_FLOW_SPEED, mach_number=REFERENCE_MACH_NUMBER,
    )


class TestSolve:
    def test_converged_beta_equals_beta_crit(self, solver: MarginalDesignPointSolver):
        """By construction, the root-find seeks beta(Te) == beta_crit(Te) -- verify the
        returned point actually satisfies that (not just that some Te was returned)."""
        result = solver.solve(magnetic_field=8.0, target_power_density=100e6)
        assert result["beta"] == pytest.approx(result["beta_crit"], rel=1e-6)

    def test_section_7_3_worked_example(self, solver: MarginalDesignPointSolver):
        """Friedberg's own Sec. 7.3 worked example: B0=8T, S_C=100 MW/m^3, Caesium seed.
        A loose (10-15%) tolerance is deliberate -- see test_friedberg_criterion.py's
        identical note on test_freidberg_section_7_3_worked_example: the published
        intermediates are only 4 significant figures, and that rounding is amplified
        through the alpha^2 term and further downstream quantities. Verified during
        development that every value here lands within 15%, several within 10% -- this
        is "right ballpark, no silent regression," not exact replication."""
        result = solver.solve(magnetic_field=8.0, target_power_density=100e6)
        assert result["electron_temperature"] == pytest.approx(4223.0, rel=0.10)
        assert result["beta"] == pytest.approx(8.249, rel=0.10)
        assert result["ohmic_to_load_ratio"] == pytest.approx(0.3720, rel=0.15)
        assert result["one_minus_f_I"] == pytest.approx(0.01679, rel=0.15)
        assert result["n_e"] == pytest.approx(7.478e19, rel=0.15)

    def test_raises_with_a_clear_message_when_te_scan_range_has_no_valid_bracket(self, solver: MarginalDesignPointSolver):
        with pytest.raises(ValueError, match="No valid sign change"):
            solver.solve(magnetic_field=8.0, target_power_density=100e6, te_scan_range=(490.0, 495.0), te_scan_points=5)


class TestSweep:
    def test_returns_one_array_per_quantity_matching_b0_values_length(self, solver: MarginalDesignPointSolver):
        b0_values = np.linspace(5.0, 10.0, 4)
        curve = solver.sweep(b0_values, target_power_density=100e6)
        assert set(curve) >= {"electron_temperature", "beta", "n_e", "ohmic_to_load_ratio", "conductivity"}
        for values in curve.values():
            assert values.shape == (4,)

    def test_matches_calling_solve_directly(self, solver: MarginalDesignPointSolver):
        b0_values = np.array([6.0, 8.0])
        curve = solver.sweep(b0_values, target_power_density=100e6)
        for i, b0 in enumerate(b0_values):
            expected = solver.solve(magnetic_field=float(b0), target_power_density=100e6)
            assert curve["electron_temperature"][i] == pytest.approx(expected["electron_temperature"])
