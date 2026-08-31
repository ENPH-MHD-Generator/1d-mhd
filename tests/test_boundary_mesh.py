"""
Tests for magnetohydrodynamics.stability.boundary_mesh.StabilityBoundaryMesh -- newly
extracted from examples/stability_boundary.py, no prior test coverage. Uses a small
synthetic margin field with a KNOWN boundary location/orientation (not a real
HallSolver-derived field) so the marching-cubes coordinate remapping and the
stable-direction geometry can be checked against values computed independently of the
implementation.
"""
import numpy as np
import pytest

from magnetohydrodynamics.stability.boundary_mesh import StabilityBoundaryMesh


@pytest.fixture
def axes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seed_fraction_values = np.logspace(-4, -2, 5)
    b0_values = np.linspace(0.0, 5.0, 11)
    tp_values = np.logspace(2.0, 3.0, 5)
    return seed_fraction_values, b0_values, tp_values


@pytest.fixture
def mesh(axes) -> StabilityBoundaryMesh:
    """margin depends only on the B0 axis, crossing 1.0 exactly at B0=2.5 (the grid's
    midpoint) and increasing with B0 -- i.e. "more field is more stable" -- so the
    boundary is a flat plane at B0=2.5 and the stable direction is always +B0."""
    seed_fraction_values, b0_values, tp_values = axes
    _SF, B0, _TP = np.meshgrid(seed_fraction_values, b0_values, tp_values, indexing="ij")
    margin = 0.4 * B0  # == 1.0 exactly at B0 = 2.5
    load_power_density = np.full_like(margin, 1e6)
    grid = dict(margin=margin, load_power_density=load_power_density)
    return StabilityBoundaryMesh(grid, seed_fraction_values, b0_values, tp_values)


class TestExtract:
    def test_vertices_land_on_the_known_boundary_plane(self, mesh: StabilityBoundaryMesh):
        vertices, faces, _vertex_power = mesh.extract()
        assert len(vertices) > 0
        assert faces.shape[1] == 3
        np.testing.assert_allclose(vertices[:, 1], 2.5, atol=1e-6)  # the B0 coordinate, exactly at the known boundary

    def test_vertices_are_within_the_axis_ranges(self, mesh: StabilityBoundaryMesh, axes):
        seed_fraction_values, _b0_values, tp_values = axes
        vertices, _faces, _vertex_power = mesh.extract()
        log_sf = np.log10(seed_fraction_values)
        assert vertices[:, 0].min() >= log_sf.min() - 1e-9
        assert vertices[:, 0].max() <= log_sf.max() + 1e-9
        assert vertices[:, 2].min() >= tp_values.min() - 1e-6
        assert vertices[:, 2].max() <= tp_values.max() + 1e-6

    def test_vertex_power_matches_the_constant_field(self, mesh: StabilityBoundaryMesh):
        _vertices, _faces, vertex_power = mesh.extract()
        np.testing.assert_allclose(vertex_power, 1e6, rtol=1e-6)


class TestStableDirectionSegments:
    def test_segments_point_toward_higher_B0(self, mesh: StabilityBoundaryMesh):
        """margin increases with B0 here, so every stable-direction arrow should point
        toward larger B0 (the segments' second, i.e. B0, coordinate should increase
        from start to end)."""
        vertices, faces, _vertex_power = mesh.extract()
        starts, ends = mesh.stable_direction_segments(vertices, faces, num_arrows=10)
        assert len(starts) > 0
        assert len(starts) == len(ends)
        assert np.all(ends[:, 1] > starts[:, 1])

    def test_segment_lengths_are_consistent_fractions_of_the_display_range(self, mesh: StabilityBoundaryMesh, axes):
        """The whole point of the Jacobian correction (see the module docstring) is
        that every arrow is the same small fraction of the plot box, regardless of
        where on the (highly nonlinear, log-displayed) Tp axis it sits."""
        seed_fraction_values, b0_values, tp_values = axes
        vertices, faces, _vertex_power = mesh.extract()
        starts, ends = mesh.stable_direction_segments(vertices, faces, num_arrows=15, arrow_length_frac=0.05)

        log_sf_values = np.log10(seed_fraction_values)
        display_ranges = np.array([
            log_sf_values[-1] - log_sf_values[0],
            b0_values[-1] - b0_values[0],
            tp_values[-1] - tp_values[0],
        ])

        fractional_lengths = np.linalg.norm((ends - starts) / display_ranges, axis=1)
        np.testing.assert_allclose(fractional_lengths, 0.05, rtol=1e-6)
