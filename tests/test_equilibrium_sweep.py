"""
Tests for magnetohydrodynamics.stability.equilibrium_sweep.EquilibriumSweep --
newly extracted from examples/stability_boundary.py, no prior test coverage. `grid`
and `volume_grid` are vectorized batch evaluations built on top of
HallSolver.solve_equilibrium_batch (already covered by test_hall_solver.py's
TestSolveEquilibriumBatch) and FriedbergCriterion (covered by
test_friedberg_criterion.py) -- these tests focus on the glue: shapes, and that a
single-point "grid" agrees with calling those two pieces directly.
"""
import numpy as np
import pytest

from magnetohydrodynamics.operating_point import OperatingPoint
from magnetohydrodynamics.stability.equilibrium_sweep import EquilibriumSweep
from magnetohydrodynamics.stability.friedberg_criterion import FriedbergCriterion


@pytest.fixture
def base() -> OperatingPoint:
    return OperatingPoint(B0=0.5, Tp=2000.0, p0=10.0e3, v0=150.0, seed_fraction=6.18e-3, load_resistivity=0.5)


@pytest.fixture
def sweep(hall_solver, base, seed_type) -> EquilibriumSweep:
    return EquilibriumSweep(hall_solver, base, seed_type.ionization_potential)


class TestGrid:
    def test_shapes_match_axis_lengths(self, sweep: EquilibriumSweep):
        x_values = np.linspace(0.1, 2.0, 4)
        y_values = np.linspace(500.0, 3000.0, 3)
        grid = sweep.grid("B0", x_values, "Tp", y_values)
        for key in ("beta", "beta_crit", "beta_crit_asymptotic", "margin", "margin_asymptotic", "Te", "ionization_fraction", "stable"):
            assert getattr(grid, key).shape == (3, 4)

    def test_single_point_matches_direct_solve_and_criterion(self, hall_solver, base, seed_type, sweep: EquilibriumSweep):
        grid = sweep.grid("B0", np.array([0.5]), "Tp", np.array([2000.0]))

        point = base.resolve()
        expected_result = hall_solver.solve_equilibrium_batch(**point.as_kwargs())
        expected_fI = expected_result.electron_number_density / expected_result.seed_number_density
        expected_beta_crit = FriedbergCriterion().critical_hall_parameter(
            expected_result.electron_temperature, base.Tp, seed_type.ionization_potential, expected_fI,
        )

        assert grid.Te[0, 0] == pytest.approx(float(expected_result.electron_temperature))
        assert grid.beta_crit[0, 0] == pytest.approx(float(expected_beta_crit))
        assert grid.stable[0, 0] == (grid.margin[0, 0] >= 1.0)

    def test_margin_ge_one_matches_stable_flag(self, sweep: EquilibriumSweep):
        grid = sweep.grid("B0", np.linspace(0.1, 5.0, 6), "seed_fraction", np.logspace(-5, -1, 6))
        np.testing.assert_array_equal(grid.stable, grid.margin >= 1.0)


class TestMatchedLoad:
    def test_converges_to_the_matched_load_condition(self, sweep: EquilibriumSweep):
        """At convergence, load_resistivity should equal sqrt(1+beta^2)*resistivity
        (eq. 6.10) self-consistently -- verify by checking that the bisection has
        actually settled (one more iteration barely moves beta/resistivity)."""
        seed_fraction, magnetic_field, gas_temperature = 2e-4, 1.7, 18.0
        result = sweep.matched_load(seed_fraction, magnetic_field, gas_temperature, iters=49)
        one_more = sweep.matched_load(seed_fraction, magnetic_field, gas_temperature, iters=50)
        assert float(result.hall_parameter) == pytest.approx(float(one_more.hall_parameter), rel=1e-6)

    def test_converges_at_a_point_that_used_to_settle_into_a_period_2_cycle(self, sweep: EquilibriumSweep):
        """This exact point used to be a counterexample: the old Picard-iteration
        version of matched_load never converged here (49 vs. 50 iterations disagreed
        by >50%, indefinitely) -- see matched_load's docstring for the diagnosis
        (a single root sitting on an ionisation-avalanche cliff, not bistability) and
        the bisection fix. Kept as a regression test for that specific point."""
        seed_fraction, magnetic_field, gas_temperature = 6.18e-3, 0.5, 2000.0
        result = sweep.matched_load(seed_fraction, magnetic_field, gas_temperature, iters=49)
        one_more = sweep.matched_load(seed_fraction, magnetic_field, gas_temperature, iters=50)
        assert float(result.hall_parameter) == pytest.approx(float(one_more.hall_parameter), rel=1e-4)

    def test_converges_broadly_across_a_grid_that_used_to_be_half_non_convergent(self, sweep: EquilibriumSweep):
        """The old Picard-iteration version failed to converge at ~50% of points on a
        grid spanning volume_grid's typical ranges (checked during the investigation
        that led to the bisection fix). Verify none do now: 49 vs. 50 iterations
        should agree everywhere, not just at hand-picked points."""
        seed_fraction_values = np.logspace(-5, -1, 6)
        b0_values = np.linspace(0.1, 5.0, 6)
        tp_values = np.logspace(0.0, np.log10(6000.0), 6)
        SF, B0, TP = np.meshgrid(seed_fraction_values, b0_values, tp_values, indexing="ij")

        result = sweep.matched_load(SF, B0, TP, iters=49)
        one_more = sweep.matched_load(SF, B0, TP, iters=50)
        relative_difference = np.abs(result.hall_parameter - one_more.hall_parameter) / np.abs(one_more.hall_parameter)
        assert np.max(relative_difference) < 1e-3

    def test_broadcasts_over_array_inputs(self, sweep: EquilibriumSweep):
        seed_fraction = np.array([1e-4, 1e-3, 1e-2])
        result = sweep.matched_load(seed_fraction=seed_fraction, magnetic_field=0.5, gas_temperature=2000.0)
        assert np.asarray(result.hall_parameter).shape == (3,)


class TestVolumeGrid:
    def test_shape_matches_all_three_axes(self, sweep: EquilibriumSweep):
        seed_fraction_values = np.logspace(-5, -1, 3)
        b0_values = np.linspace(0.1, 5.0, 4)
        tp_values = np.logspace(2.0, 3.5, 5)
        grid = sweep.volume_grid(seed_fraction_values, b0_values, tp_values)
        for key in ("margin", "stable", "Te", "ionization_fraction", "load_power_density"):
            assert getattr(grid, key).shape == (3, 4, 5)

    def test_stable_flag_matches_margin(self, sweep: EquilibriumSweep):
        grid = sweep.volume_grid(np.logspace(-5, -1, 3), np.linspace(0.1, 5.0, 3), np.logspace(2.0, 3.5, 3))
        np.testing.assert_array_equal(grid.stable, grid.margin >= 1.0)


class TestMarginMinusLevel:
    def test_matches_direct_criterion_evaluation(self, hall_solver, base, seed_type, sweep: EquilibriumSweep):
        value = sweep.margin_minus_level(1.0, B0=0.5)

        point = base.resolve(B0=0.5)
        result = hall_solver.solve_equilibrium_batch(**point.as_kwargs())
        fI = result.electron_number_density / result.seed_number_density
        expected_margin = FriedbergCriterion().stability_margin(
            result.hall_parameter, result.electron_temperature, point.gas_temperature, seed_type.ionization_potential, fI,
        )
        assert float(value) == pytest.approx(float(expected_margin) - 1.0)

    def test_level_shifts_the_zero_crossing(self, sweep: EquilibriumSweep):
        """margin_minus_level(level, ...) == margin_minus_level(1.0, ...) - (level - 1.0),
        so a higher level should read lower (further from a stable crossing) at a
        fixed point, by exactly the difference in levels."""
        default_level = sweep.margin_minus_level(1.0, B0=0.5)
        higher_level = sweep.margin_minus_level(1.5, B0=0.5)
        assert float(higher_level) == pytest.approx(float(default_level) - 0.5)

    def test_broadcasts_over_arrays(self, sweep: EquilibriumSweep):
        b0 = np.array([0.2, 0.5, 1.0])
        values = np.asarray(sweep.margin_minus_level(1.0, B0=b0))
        assert values.shape == (3,)


class TestCriticalLoadResistivitySurface:
    def test_returns_arrays_shaped_by_the_two_sweep_axes(self, sweep: EquilibriumSweep):
        seed_fraction_values = np.logspace(-4, -2, 3)
        b0_values = np.linspace(0.5, 2.0, 3)
        lower, upper, lower_power, upper_power = sweep.critical_load_resistivity_surface(
            seed_fraction_values, b0_values, scan_points=25,
        )
        for array in (lower, upper, lower_power, upper_power):
            assert array.shape == (3, 3)

    def test_lower_boundary_is_finite_where_found_and_load_power_is_positive_there(self, sweep: EquilibriumSweep):
        seed_fraction_values = np.logspace(-4, -2, 4)
        b0_values = np.linspace(0.5, 3.0, 4)
        lower, _upper, lower_power, _upper_power = sweep.critical_load_resistivity_surface(
            seed_fraction_values, b0_values, scan_points=30,
        )
        found = np.isfinite(lower)
        assert found.any()  # this operating range should find at least some boundary points
        assert np.all(lower_power[found] > 0.0)

    def test_higher_level_raises_the_lower_boundary(self, sweep: EquilibriumSweep):
        """Raising eta_L only ever pushes the system toward stability (see the
        docstring's note on no re-destabilisation being observed) -- so demanding a
        stability margin higher than 1.0 (level=1.5, a safety buffer) should require
        at least as much load resistivity as demanding exactly marginal stability
        (level=1.0), at every point where both are found."""
        seed_fraction_values = np.logspace(-4, -2, 4)
        b0_values = np.linspace(0.5, 3.0, 4)
        lower_default, _u, _lp, _up = sweep.critical_load_resistivity_surface(seed_fraction_values, b0_values, scan_points=30)
        lower_buffered, _u, _lp, _up = sweep.critical_load_resistivity_surface(seed_fraction_values, b0_values, level=1.5, scan_points=30)
        both_found = np.isfinite(lower_default) & np.isfinite(lower_buffered)
        assert both_found.any()
        assert np.all(lower_buffered[both_found] >= lower_default[both_found])

    def test_detects_a_genuine_stability_window(self, sweep: EquilibriumSweep, monkeypatch):
        """A real unstable-stable-unstable window (`upper` distinct from `lower`) isn't
        observed anywhere in this codebase's actual physics (see the method's own
        docstring) -- so the "find the first AND last sign change, not just the first"
        logic is exercised here against a synthetic margin_minus_level standing in for
        the real one, the same way the deleted find_crossings unit tests did before
        this method's vectorized-bisection rewrite folded that logic in directly."""
        log_lo, log_hi = 1.0, 2.0  # roots at load_resistivity = 10, 100

        def fake_margin_minus_level(self, level, **overrides):
            log_lr = np.log10(np.asarray(overrides["load_resistivity"], dtype=float))
            return -(log_lr - log_lo) * (log_lr - log_hi)  # negative outside [log_lo, log_hi], positive inside

        monkeypatch.setattr(EquilibriumSweep, "margin_minus_level", fake_margin_minus_level)
        lower, upper, _lower_power, _upper_power = sweep.critical_load_resistivity_surface(
            np.array([1e-3]), np.array([1.0]), load_resistivity_bracket=(1.0, 1000.0), scan_points=30,
        )
        assert lower[0, 0] == pytest.approx(10.0, rel=1e-3)
        assert upper[0, 0] == pytest.approx(100.0, rel=1e-3)
