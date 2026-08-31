"""
Tests for magnetohydrodynamics.stability.seed_density_bounds.SeedDensityBounds --
newly extracted from examples/stability_boundary.py, no prior test coverage.
"""
import numpy as np
import pytest

from magnetohydrodynamics.stability.seed_density_bounds import SeedDensityBounds


@pytest.fixture
def bounds(gas_type, seed_type) -> SeedDensityBounds:
    return SeedDensityBounds(gas_type, seed_type)


class TestCeiling:
    def test_returns_shape_matching_the_two_axes(self, bounds: SeedDensityBounds):
        Te_values = np.array([5000.0, 15000.0, 40000.0])
        beta_values = np.logspace(0.0, 2.0, 5)
        ceiling = bounds.ceiling(Te_values, beta_values, reference_gas_number_density=1e24)
        assert ceiling.shape == (3, 5)

    def test_ceiling_decreases_as_beta_target_increases(self, bounds: SeedDensityBounds):
        """A higher design-target beta demands a stability margin closer to the edge,
        which (per the docstring) means less room for seed dilution -- the ceiling
        should be monotonically non-increasing in beta at fixed Te."""
        Te_values = np.array([15000.0])
        beta_values = np.logspace(0.0, 2.0, 8)
        ceiling = bounds.ceiling(Te_values, beta_values, reference_gas_number_density=1e24)
        row = ceiling[0]
        finite = row[np.isfinite(row)]
        assert len(finite) > 1
        assert np.all(np.diff(finite) <= 1e-12)

    def test_nan_when_no_ceiling_found_in_bracket(self, bounds: SeedDensityBounds):
        """An absurdly high beta target with a narrow ns_bracket should fail to find a
        sign change -- NaN, not an exception."""
        result = bounds.ceiling(
            np.array([5000.0]), np.array([1e6]), reference_gas_number_density=1e24,
            ns_bracket=(1e20, 1e21),
        )
        assert np.isnan(result[0, 0])


class TestMinMaxWindow:
    def test_returns_shapes_matching_the_two_axes(self, bounds: SeedDensityBounds):
        Te_values = np.linspace(500.0, 5000.0, 4)
        beta_values = np.linspace(1.0, 30.0, 5)
        max_ns, min_ne = bounds.min_max_window(
            Te_values, beta_values, reference_gas_number_density=1.21e26, reference_flow_speed=735.0,
        )
        assert max_ns.shape == (4, 5)
        assert min_ne.shape == (4, 5)

    def test_min_ne_is_always_finite_and_positive(self, bounds: SeedDensityBounds):
        """min_ne is closed-form algebra (no root-find), so it should never be NaN --
        unlike max_ns, which depends on brentq finding a sign change."""
        Te_values = np.linspace(500.0, 5000.0, 4)
        beta_values = np.linspace(1.0, 30.0, 5)
        _max_ns, min_ne = bounds.min_max_window(
            Te_values, beta_values, reference_gas_number_density=1.21e26, reference_flow_speed=735.0,
        )
        assert np.all(np.isfinite(min_ne))
        assert np.all(min_ne > 0.0)

    def test_min_ne_scales_with_target_power_density(self, bounds: SeedDensityBounds):
        """min_ne is directly proportional to target_power_density (see the closed-form
        S_L=... expression in the docstring) -- doubling S_C should double min_ne."""
        Te_values = np.array([2000.0])
        beta_values = np.array([10.0])
        _max_ns, min_ne_1x = bounds.min_max_window(
            Te_values, beta_values, reference_gas_number_density=1.21e26, reference_flow_speed=735.0, target_power_density=100e6,
        )
        _max_ns, min_ne_2x = bounds.min_max_window(
            Te_values, beta_values, reference_gas_number_density=1.21e26, reference_flow_speed=735.0, target_power_density=200e6,
        )
        assert min_ne_2x[0, 0] == pytest.approx(2.0 * min_ne_1x[0, 0], rel=1e-9)
