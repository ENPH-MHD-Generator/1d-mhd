from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates
from skimage.measure import marching_cubes

from magnetohydrodynamics.stability.stability_grid import VolumeGrid


class StabilityBoundaryMesh:
    """Isosurface extraction and stable-direction geometry for a precomputed 3-D
    margin field (see `EquilibriumSweep.volume_grid`) over (seed_fraction, B0, Tp).
    Bundles the grid together with its three axes, since every operation here needs
    both: the axes are what turn marching_cubes' fractional grid-index vertices into
    real (log10(seed_fraction), B0, Tp) coordinates."""

    def __init__(self, grid: VolumeGrid, seed_fraction_values: np.ndarray, b0_values: np.ndarray, tp_values: np.ndarray):
        self._grid = grid
        self._seed_fraction_values = seed_fraction_values
        self._b0_values = b0_values
        self._tp_values = tp_values
        self._margin_capped = np.clip(np.nan_to_num(grid.margin, nan=0.0, posinf=1e6, neginf=0.0), 0.0, 1e6)

    def extract(self, level: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Marching-cubes isosurface (skimage.measure.marching_cubes) of the margin
        field at margin=`level`=1 (the stability boundary), rather than extracting a
        scatter of boundary-adjacent grid cells.

        A scatter built from `scipy.interpolate.RegularGridInterpolator` upsampling
        shows visible "terracing": trilinear interpolation connects coarse samples
        with straight-line ramps -- wherever the true field is a near-cliff (this
        system's ionisation-avalanche transitions are exactly that), the interpolated
        crossing location within each coarse cell is very sensitive to which cell
        you're in, and stitching cells together traces out the coarse grid's own cell
        boundaries. Marching cubes doesn't have this failure mode: it solves for the
        exact interpolated crossing position along each grid edge from the real
        corner values directly, giving sub-cell accuracy instead of "which cell", and
        returns a proper triangulated surface (a few thousand vertices) rather than a
        ~100,000-point scatter -- both smoother-looking and far lighter to render.

        Returns (vertices, faces, vertex_power): vertices in REAL coordinates
        (log10(seed_fraction), B0, Tp [K, linear]) -- not the fractional grid-index
        coordinates marching_cubes itself works in -- faces as vertex-index triples,
        and the load power density interpolated at each vertex (for coloring)."""
        index_vertices, faces, _normals, _values = marching_cubes(self._margin_capped, level=level)

        log_sf = np.log10(self._seed_fraction_values)
        log_tp = np.log10(self._tp_values)
        real_log_sf = np.interp(index_vertices[:, 0], np.arange(len(self._seed_fraction_values)), log_sf)
        real_b0 = np.interp(index_vertices[:, 1], np.arange(len(self._b0_values)), self._b0_values)
        real_tp = 10.0 ** np.interp(index_vertices[:, 2], np.arange(len(self._tp_values)), log_tp)
        vertices = np.stack([real_log_sf, real_b0, real_tp], axis=-1)

        vertex_power = map_coordinates(self._grid.load_power_density, index_vertices.T, order=1, mode="nearest")
        return vertices, faces, vertex_power

    def stable_direction_segments(
            self, vertices: np.ndarray, faces: np.ndarray,
            num_arrows: int = 30, arrow_length_frac: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        """For `num_arrows` faces sampled evenly across the mesh, a short line segment
        from that face's centroid along its own local surface normal, oriented to
        point toward increasing margin (the stable side) and normalized to a
        consistent length *as displayed* -- this mesh can face different directions on
        different sheets, so each arrow needs its own locally-estimated direction.

        Two earlier approaches both had the same underlying problem: the visible
        arrow length ends up depending on Tp itself, because the grid (and the mesh
        geometry) is uniform in log10(Tp) while the z-axis displays plain Tp. A
        per-face finite-difference *gradient* has a magnitude that varies a lot; a
        FIXED-size step in a rescaled log10(Tp) cube fixes that, but converting a
        large *finite* step through the nonlinear 10^x back-transform still blows up
        in absolute Tp terms at high Tp (checked: roughly 60x longer near Tp~6000 K
        than near Tp~100 K).

        This gets its *direction* from the mesh's own triangle geometry (computed in
        the rescaled cube, so no axis dominates just because of its raw units), but
        converts that direction to real (log10(seed_fraction), B0, Tp) space using the
        LOCAL LINEAR (Jacobian) approximation of the log10(Tp) -> Tp transform --
        d(Tp) = Tp * ln(10) * d(log10 Tp), evaluated at each arrow's own centroid --
        rather than applying the full nonlinear transform to a large step. The
        resulting real-space tangent is then normalized as a fraction of each axis's
        own *displayed* range (log10(seed_fraction) range, B0 range, linear Tp range),
        so every arrow's length is a consistent, small fraction of the plot box in
        every direction, regardless of where on the Tp axis it sits.

        marching_cubes' own per-vertex `normals` were checked directly against the
        margin field first (confirmed, consistently, to point toward the unstable
        side) but aren't used here -- they're in raw grid-index units, and the mesh's
        own triangle geometry gives an equally-valid normal directly in whatever space
        it's computed in.

        Returns (starts, ends), both (N, 3) arrays in the same real coordinates as
        `vertices`."""
        log_sf = np.log10(self._seed_fraction_values)
        log_tp = np.log10(self._tp_values)
        cube_lo = np.array([log_sf[0], self._b0_values[0], log_tp[0]])
        cube_ranges = np.array([log_sf[-1] - log_sf[0], self._b0_values[-1] - self._b0_values[0], log_tp[-1] - log_tp[0]])
        display_ranges = np.array([log_sf[-1] - log_sf[0], self._b0_values[-1] - self._b0_values[0], self._tp_values[-1] - self._tp_values[0]])

        def to_cube(point_real: np.ndarray) -> np.ndarray:
            return (np.array([point_real[0], point_real[1], np.log10(point_real[2])]) - cube_lo) / cube_ranges

        def cube_to_query(point_cube: np.ndarray) -> np.ndarray:
            """cube coords -> (log10(seed_fraction), B0, log10(Tp)), i.e. what the
            interpolator (built over the grid's own uniform-in-log10(Tp) axes)
            expects."""
            return point_cube * cube_ranges + cube_lo

        interpolator = RegularGridInterpolator((log_sf, self._b0_values, log_tp), self._margin_capped, bounds_error=False, fill_value=None)

        stride = max(1, len(faces) // num_arrows)
        starts, ends = [], []
        for face in faces[::stride]:
            triangle_real = vertices[face]
            centroid_real = triangle_real.mean(axis=0)
            triangle_cube = np.array([to_cube(v) for v in triangle_real])
            centroid_cube = triangle_cube.mean(axis=0)

            normal_cube = np.cross(triangle_cube[1] - triangle_cube[0], triangle_cube[2] - triangle_cube[0])
            norm = np.linalg.norm(normal_cube)
            if norm < 1e-12:
                continue
            normal_cube_unit = normal_cube / norm

            probe = 0.02
            plus = interpolator(cube_to_query(centroid_cube + probe * normal_cube_unit))[0]
            minus = interpolator(cube_to_query(centroid_cube - probe * normal_cube_unit))[0]
            if not (np.isfinite(plus) and np.isfinite(minus)):
                continue
            if plus < minus:
                normal_cube_unit = -normal_cube_unit

            # local Jacobian d(real)/d(cube), evaluated at the centroid: log_sf and B0
            # are already linear in their cube coordinate, but Tp = 10^(log_tp) means
            # d(Tp)/d(cube_z) = Tp * ln(10) * cube_ranges[2].
            jacobian_diag = np.array([cube_ranges[0], cube_ranges[1], cube_ranges[2] * centroid_real[2] * np.log(10.0)])
            tangent_real = normal_cube_unit * jacobian_diag

            fractional_tangent = tangent_real / display_ranges
            fractional_norm = np.linalg.norm(fractional_tangent)
            if fractional_norm < 1e-12:
                continue

            starts.append(centroid_real)
            ends.append(centroid_real + (fractional_tangent / fractional_norm) * arrow_length_frac * display_ranges)
        return np.array(starts), np.array(ends)
