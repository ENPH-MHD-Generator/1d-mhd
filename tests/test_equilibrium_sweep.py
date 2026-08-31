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

from magnetohydrodynamics.stability.equilibrium_sweep import EquilibriumSweep
from magnetohydrodynamics.stability.friedberg_criterion import FriedbergCriterion
from magnetohydrodynamics.stability.operating_point import OperatingPoint


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
            assert grid[key].shape == (3, 4)

    def test_single_point_matches_direct_solve_and_criterion(self, hall_solver, base, seed_type, sweep: EquilibriumSweep):
        grid = sweep.grid("B0", np.array([0.5]), "Tp", np.array([2000.0]))

        point = base.resolve()
        expected_result = hall_solver.solve_equilibrium_batch(**point)
        expected_fI = expected_result["electron_number_density"] / expected_result["seed_number_density"]
        expected_beta_crit = FriedbergCriterion().critical_hall_parameter(
            expected_result["electron_temperature"], base.Tp, seed_type.ionization_potential, expected_fI,
        )

        assert grid["Te"][0, 0] == pytest.approx(float(expected_result["electron_temperature"]))
        assert grid["beta_crit"][0, 0] == pytest.approx(float(expected_beta_crit))
        assert grid["stable"][0, 0] == (grid["margin"][0, 0] >= 1.0)

    def test_margin_ge_one_matches_stable_flag(self, sweep: EquilibriumSweep):
        grid = sweep.grid("B0", np.linspace(0.1, 5.0, 6), "seed_fraction", np.logspace(-5, -1, 6))
        np.testing.assert_array_equal(grid["stable"], grid["margin"] >= 1.0)


class TestMatchedLoad:
    def test_converges_to_the_matched_load_condition_at_a_known_convergent_point(self, sweep: EquilibriumSweep):
        """At convergence, load_resistivity should equal sqrt(1+beta^2)*resistivity
        (eq. 6.10) self-consistently -- verify by checking one more iteration barely
        moves beta/resistivity. NOT every point in parameter space converges this way
        -- see the next test, and matched_load's docstring -- this point was picked
        because it's known (checked directly) to converge cleanly."""
        point = dict(seed_fraction=2e-4, magnetic_field=1.7, gas_temperature=18.0)
        result = sweep.matched_load(iters=15, **point)
        one_more = sweep.matched_load(iters=16, **point)
        assert float(result["hall_parameter"]) == pytest.approx(float(one_more["hall_parameter"]), rel=1e-6)

    def test_known_limitation_many_points_settle_into_a_period_2_cycle_instead(self, sweep: EquilibriumSweep):
        """Documents matched_load's KNOWN LIMITATION rather than hiding it: at many
        (seed_fraction, B0, Tp) points, this undamped fixed-point iteration never
        converges -- consecutive iteration counts keep disagreeing by a large relative
        amount no matter how many iterations are run. This test exists so that, if a
        future fix changes this behavior, it fails here first (a prompt to update this
        test and matched_load's docstring together) rather than the regression going
        unnoticed."""
        point = dict(seed_fraction=6.18e-3, magnetic_field=0.5, gas_temperature=2000.0)
        result = sweep.matched_load(iters=15, **point)
        one_more = sweep.matched_load(iters=16, **point)
        relative_difference = abs(float(result["hall_parameter"]) - float(one_more["hall_parameter"])) / float(one_more["hall_parameter"])
        assert relative_difference > 0.5  # nowhere near converged -- a real period-2 cycle, not slow convergence

    def test_broadcasts_over_array_inputs(self, sweep: EquilibriumSweep):
        seed_fraction = np.array([1e-4, 1e-3, 1e-2])
        result = sweep.matched_load(seed_fraction=seed_fraction, magnetic_field=0.5, gas_temperature=2000.0)
        assert result["hall_parameter"].shape == (3,)


class TestVolumeGrid:
    def test_shape_matches_all_three_axes(self, sweep: EquilibriumSweep):
        seed_fraction_values = np.logspace(-5, -1, 3)
        b0_values = np.linspace(0.1, 5.0, 4)
        tp_values = np.logspace(2.0, 3.5, 5)
        grid = sweep.volume_grid(seed_fraction_values, b0_values, tp_values)
        for key in ("margin", "stable", "Te", "ionization_fraction", "load_power_density"):
            assert grid[key].shape == (3, 4, 5)

    def test_stable_flag_matches_margin(self, sweep: EquilibriumSweep):
        grid = sweep.volume_grid(np.logspace(-5, -1, 3), np.linspace(0.1, 5.0, 3), np.logspace(2.0, 3.5, 3))
        np.testing.assert_array_equal(grid["stable"], grid["margin"] >= 1.0)


class TestMarginMinusOne:
    def test_matches_direct_criterion_evaluation(self, hall_solver, base, seed_type, sweep: EquilibriumSweep):
        value = sweep.margin_minus_one(B0=0.5)

        point = base.resolve(B0=0.5)
        result = hall_solver.solve_equilibrium_batch(**point)
        fI = result["electron_number_density"] / result["seed_number_density"]
        expected_margin = FriedbergCriterion().stability_margin(
            result["hall_parameter"], result["electron_temperature"], point["gas_temperature"], seed_type.ionization_potential, fI,
        )
        assert float(value) == pytest.approx(float(expected_margin) - 1.0)

    def test_broadcasts_over_arrays(self, sweep: EquilibriumSweep):
        b0 = np.array([0.2, 0.5, 1.0])
        values = sweep.margin_minus_one(B0=b0)
        assert values.shape == (3,)


class TestFindCrossings:
    def test_finds_a_single_crossing(self):
        # Root at 5.3, deliberately not on a grid point -- landing exactly on one would
        # make np.sign(0) register as a sign change on both neighboring intervals.
        axis = np.linspace(0.0, 10.0, 11)
        values = axis - 5.3
        roots = EquilibriumSweep.find_crossings(values, axis, objective=lambda x: x - 5.3)
        assert len(roots) == 1
        assert roots[0] == pytest.approx(5.3, abs=1e-4)

    def test_finds_a_stability_window_with_two_crossings(self):
        """unstable-stable-unstable: two crossings, not zero -- checking only the
        endpoints (both negative here) would have missed both. Roots at 3.3/7.3,
        deliberately off-grid -- see test_finds_a_single_crossing's comment."""
        axis = np.linspace(0.0, 10.0, 11)

        def objective(x):
            return -(x - 3.3) * (x - 7.3)

        values = objective(axis)
        roots = EquilibriumSweep.find_crossings(values, axis, objective)
        assert len(roots) == 2
        assert roots[0] == pytest.approx(3.3, abs=1e-4)
        assert roots[1] == pytest.approx(7.3, abs=1e-4)

    def test_no_crossings_returns_empty_list(self):
        axis = np.linspace(0.0, 10.0, 11)
        values = axis + 1.0  # always positive, never crosses zero
        roots = EquilibriumSweep.find_crossings(values, axis, objective=lambda x: x + 1.0)
        assert roots == []


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
