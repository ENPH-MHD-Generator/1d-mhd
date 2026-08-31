"""
Visualize the Velikhov-ionisation instability's stability boundary (Friedberg 2025, see
magnetohydrodynamics/stability/velikhov_instability.py) as a function of two physical
inputs at a time, holding the rest fixed at the default operating point.

A boundary plot needs a self-consistent local equilibrium (Te, ne, f_I, beta) at every
grid point -- exactly what HallSolver.solve_equilibrium computes for a single slice, with
no axial integration needed (unlike HallSolver.march, which isn't used here).

Of the seven candidate inputs (cross section, seed fraction, T_p, p_p, B, v_p, load
resistivity), the load-resistivity-normalised parameter Z from the paper is *derived*
(it falls out of load_resistivity and the self-consistently solved local resistivity
eta(Te)), not a free input -- so it is never a direct sweep axis here, only ever a
byproduct you could read off a solved grid if you wanted it.

The plotted quantity throughout is the stability margin beta_crit/beta: >= 1 is stable,
< 1 is unstable (Friedberg's criterion is beta <= beta_crit for stability).

Run with:
    uv run python examples/stability_boundary.py            # saves PNGs, no window
    uv run python examples/stability_boundary.py --show      # also opens windows
    uv run python examples/stability_boundary.py --no-save   # windows only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import constants
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates
from scipy.optimize import brentq
from skimage.measure import marching_cubes

from magnetohydrodynamics.ionization.saha_ionization import LocalThermodynamicEquilibrium
from magnetohydrodynamics.presets import build_default_hall_solver, default_gas_type, default_operating_point, default_seed_type
from magnetohydrodynamics.solver.hall_solver import HallSolver
from magnetohydrodynamics.stability import (
    critical_hall_parameter,
    critical_hall_parameter_asymptotic,
    stability_margin,
)
from magnetohydrodynamics.transport.mhd_transport_model import MHDTransportModel

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Appended to every plot title as a brief reminder of which way the margin ratio reads.
STABILITY_NOTE = r"(stable: $\beta_{crit}/\beta \geq 1$)"

# Display metadata for each physical knob: (symbol, unit, format spec), used both for
# axis labels the demo sweeps don't already set explicitly and for the "fixed at..."
# annotation on every plot.
PHYSICAL_META = {
    "B0": (r"$B_0$", "T", "{:.3g}"),
    "Tp": (r"$T_p$", "K", "{:.0f}"),
    "p0": (r"$p_0$", "Pa", "{:.3g}"),
    "v0": (r"$v_0$", "m/s", "{:.3g}"),
    "seed_fraction": ("seed fraction", "", "{:.2e}"),
    "load_resistivity": (r"$\eta_L$", r"$\Omega\cdot$m", "{:.3g}"),
}

# Friedberg's own Sec. 7.1 worked-example inlet conditions (HTGR + Laval nozzle, M=1.8):
# Tp=481 K, pp=0.801 MPa, vp=735 m/s. Used ONLY as the reference n_p/v_p for
# seed_fraction_min_max_surface's absolute S_C=100 MW/m^3 power-density constraint --
# NOT this script's own base_operating_point() (Tp=2000K, p0=10kPa, v0=150 m/s, this
# repo's own default, much more dilute and slower). Checked numerically first: with this
# repo's default as the reference, the region satisfying BOTH the stability ceiling and
# the power floor covers only ~4% of the tested (Te, beta) grid -- a sliver invisible at
# plot resolution -- whereas Friedberg's own S_C=100 MW/m^3 figure was derived to pair
# with HIS reference conditions, under which the valid region covers ~75% of the same
# grid. The "surfaces never usefully overlap" finding wasn't wrong physics, just a
# mismatched reference point for an absolute (not ratio) power target.
PAPER_REFERENCE_GAS_NUMBER_DENSITY = 0.801e6 / (constants.k * 481.0)  # n_p = pp/(k Tp) [m^-3]
PAPER_REFERENCE_FLOW_SPEED = 735.0  # v_p [m/s]

# stability_volume_grid's axes -- shared between demo_sweeps (the static matplotlib
# point cloud) and main()'s optional interactive Plotly export, so both describe the
# same grid without repeating the resolution/range numbers in two places.
VOLUME_SEED_FRACTION_VALUES = np.logspace(-5, -1, 45)
VOLUME_B0_VALUES = np.linspace(0.1, 5.0, 45)
VOLUME_TP_VALUES = np.logspace(0.0, np.log10(6000.0), 45)


def base_operating_point() -> dict:
    """The default operating point (presets.default_operating_point()), expressed as
    the *physical* knobs a sweep can vary: B0, Tp, p0, v0, seed_fraction, load_resistivity.
    (Not solve_equilibrium's raw args directly -- gas_number_density/seed_number_density
    are derived from these at each grid point, see `_resolve_point`, since sweeping e.g.
    Tp at fixed p0 must recompute n_p = p0/(k Tp) rather than holding n_p fixed.)"""
    params = default_operating_point()
    return dict(
        B0=params["magnetic_field"],
        Tp=params["inlet_gas_temperature"],
        p0=params["inlet_pressure"],
        v0=params["inlet_speed"],
        seed_fraction=params["inlet_seed_fraction"],
        load_resistivity=params["load_resistance"] * params["area"] / params["length"],
    )


def _resolve_point(base: dict, **overrides) -> dict:
    """Physical knobs (B0, Tp, p0, v0, seed_fraction, load_resistivity) -> the six raw
    args HallSolver.solve_equilibrium needs, recomputing gas_number_density/
    seed_number_density from whichever of Tp/p0/seed_fraction ended up varied."""
    point = {**base, **overrides}
    gas_number_density = point["p0"] / (constants.k * point["Tp"])
    seed_number_density = point["seed_fraction"] * gas_number_density
    return dict(
        flow_speed=point["v0"],
        gas_temperature=point["Tp"],
        gas_number_density=gas_number_density,
        seed_number_density=seed_number_density,
        magnetic_field=point["B0"],
        load_resistivity=point["load_resistivity"],
    )


def _fixed_params_text(base: dict, *varied_keys: str) -> str:
    """'fixed: ...' string listing every physical knob NOT in `varied_keys`, for
    annotating a plot with what's being held constant."""
    parts = []
    for key, value in base.items():
        if key in varied_keys:
            continue
        symbol, unit, fmt = PHYSICAL_META[key]
        formatted = fmt.format(value) + (f" {unit}" if unit else "")
        parts.append(f"{symbol} = {formatted}")
    return "fixed:  " + ",   ".join(parts)


def sweep_grid(
        hall_solver: HallSolver,
        base: dict,
        x_key: str, x_values: np.ndarray,
        y_key: str, y_values: np.ndarray,
) -> dict[str, np.ndarray]:
    """Solve the local equilibrium at every (x, y) combination and evaluate both
    stability criteria there, via a single vectorized `solve_equilibrium_batch` call
    instead of one `solve_equilibrium` (and one Plasma object) per grid point. Measured
    speedup on a 250x250 grid: ~5.5s -> ~0.02s -- the physics is identical either way
    (`solve_equilibrium_batch` runs the exact same `_iterate_equilibrium` fixed-point
    loop HallSolver.solve_equilibrium does), the win is purely from replacing 62,500
    Python-level calls with one call operating on 62,500-element arrays.

    Returns 2-D arrays of shape (len(y_values), len(x_values))."""
    X, Y = np.meshgrid(x_values, y_values)
    point = _resolve_point(base, **{x_key: X, y_key: Y})
    result = hall_solver.solve_equilibrium_batch(**point)

    ionization_potential = default_seed_type().ionization_potential
    ionization_fraction = result["electron_number_density"] / result["seed_number_density"]
    beta = result["hall_parameter"]
    beta_crit = critical_hall_parameter(result["electron_temperature"], point["gas_temperature"], ionization_potential, ionization_fraction)
    beta_crit_asymptotic = critical_hall_parameter_asymptotic(result["electron_temperature"], ionization_potential, ionization_fraction)

    margin = stability_margin(beta, beta_crit)
    return dict(
        beta=beta,
        beta_crit=beta_crit,
        beta_crit_asymptotic=beta_crit_asymptotic,
        margin=margin,
        margin_asymptotic=stability_margin(beta, beta_crit_asymptotic),
        Te=result["electron_temperature"],
        ionization_fraction=ionization_fraction,
        stable=margin >= 1.0,
    )


def _draw_stable_direction_markers(ax, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, length: float, color: str = "steelblue") -> None:
    """Short vertical line segments (with a triangle marker on top) from each finite
    (x, y, z) sample point straight up by `length`, indicating the stable side on a 3-D
    surface plot. Plotted one at a time with plain `ax.plot`/`ax.scatter` rather than
    `Axes3D.quiver` -- quiver's arrows came out wildly mis-scaled (stretching across the
    whole plot) whenever the height axis's raw numeric range was very different from the
    other two axes' (e.g. Tp in the hundreds/thousands vs. log10-scaled axes of order 1-10);
    this avoids relying on quiver's internal scaling entirely."""
    xs, ys, zs = np.ravel(xs), np.ravel(ys), np.ravel(zs)
    finite = np.isfinite(zs)
    for x, y, z in zip(xs[finite], ys[finite], zs[finite]):
        ax.plot([x, x], [y, y], [z, z + length], color=color, linewidth=1.5)
    ax.scatter(xs[finite], ys[finite], zs[finite] + length, marker="^", color=color, s=12, depthshade=False)


def _draw_boundary_contour(ax, x_values, y_values, margin, color="black", halo_color="white", halo_extra=1.0, linewidths=2.0, **contour_kwargs):
    """A contour line at margin == 1 (marginal stability), with a thin halo so it stays
    visible against the colormap. Kept thin deliberately: the exact and asymptotic
    boundaries often sit almost on top of each other, and a thick halo around one (an
    earlier version used +2.0) paints over the other's dashes in its gaps, making the
    dashed line disappear wherever the two nearly coincide. Distinct colors (not just
    dash pattern) are what actually keeps both readable when they overlap -- see
    plot_stability_boundary's two calls to this function."""
    cs = ax.contour(x_values, y_values, margin, levels=[1.0], colors=color, linewidths=linewidths, **contour_kwargs)
    cs.set(path_effects=[pe.withStroke(linewidth=linewidths + halo_extra, foreground=halo_color)])
    return cs


def plot_stability_boundary(
        grid: dict, base: dict, x_key: str, x_values: np.ndarray, y_key: str, y_values: np.ndarray,
        title: str, x_label: str, y_label: str, y_log: bool = False,
        margin_cap: float = 2.0,
) -> plt.Figure:
    """Continuously-shaded stability margin ratio beta_crit/beta: red (<1) unstable,
    white (=1) marginal, blue (>1) stable, with the exact (5.13) boundary as a solid
    black contour and the high-ionisation asymptotic (6.23) boundary overlaid as a dashed
    lime one -- a distinct color, not just dash pattern, since the two often sit almost on
    top of each other and a same-colored halo would otherwise paint over the dashes in
    their gaps wherever that happens (see _draw_boundary_contour's docstring). A caption
    lists every physical input held fixed for this sweep.

    The ratio is unbounded above (beta_crit -> +inf deep in the fully-ionised, stable
    regime), which would otherwise wash out all the interesting structure near the actual
    boundary (ratio ~ 1) if plotted on a linear 0..max scale. Instead the colormap is
    saturated at `margin_cap`: anything at least that stable renders as the same
    "over"-colored blue, shown with an arrow on the colorbar, so the color resolution is
    spent on the marginal region instead."""
    margin = grid["margin"]
    margin_asymptotic = grid["margin_asymptotic"]

    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=margin_cap)

    fig, ax = plt.subplots(figsize=(7, 6))
    # contourf with many (200) levels: a genuine fine banding, dense enough to read as
    # continuous, but computed from real marching-squares interpolation of the data --
    # unlike pcolormesh(shading="gouraud") (used in an earlier version of this plot),
    # which linearly blends colors *within* each grid cell regardless of how sharply the
    # underlying field actually changes there. This system's stability transition can be
    # very steep (a near step-function in places -- see e.g. the seed-fraction sweeps'
    # ionisation-avalanche cliff), and gouraud's per-cell linear blend showed up as
    # visible diagonal streaking/banding artifacts across that cliff even at high grid
    # resolution; contourf's level-boundary interpolation doesn't have that failure mode.
    margin_filled = np.nan_to_num(margin, nan=0.0, posinf=margin_cap * 10.0, neginf=0.0)
    levels = np.linspace(0.0, margin_cap, 200)
    mesh = ax.contourf(x_values, y_values, margin_filled, levels=levels, cmap="RdBu", norm=norm, extend="max")
    cbar = fig.colorbar(mesh, ax=ax, label=r"stability margin  $\beta_{crit}/\beta$")
    cbar.ax.axhline(1.0, color="black", linewidth=1.0)  # marks the marginal-stability level on the bar itself

    _draw_boundary_contour(ax, x_values, y_values, margin, linewidths=2.0)
    _draw_boundary_contour(
        ax, x_values, y_values, margin_asymptotic,
        color="lime", halo_color="black", halo_extra=0.8, linewidths=1.5, linestyles="dashed",
    )
    ax.plot([], [], color="black", linewidth=2.0, label="exact boundary (5.13)")
    ax.plot([], [], color="lime", linewidth=1.5, linestyle="dashed", label="asymptotic boundary (6.23)")

    if y_log:
        ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{title}  {STABILITY_NOTE}", fontsize=11.5)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.text(
        0.5, 0.04, _fixed_params_text(base, x_key, y_key), ha="center", va="bottom", fontsize=8.5,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.85),
    )
    return fig


def _margin_batch(hall_solver: HallSolver, base: dict, **overrides) -> np.ndarray:
    """beta_crit/beta - 1, evaluated via solve_equilibrium_batch for whatever
    combination of scalar/array physical-knob `overrides` is given (merged with `base`).
    Used both for the big vectorized scans below and, with scalar args, as the objective
    handed to brentq for individual root refinement -- this script never needs to
    construct a Plasma object directly."""
    point = _resolve_point(base, **overrides)
    result = hall_solver.solve_equilibrium_batch(**point)
    ionization_potential = default_seed_type().ionization_potential
    ionization_fraction = result["electron_number_density"] / result["seed_number_density"]
    beta_crit = critical_hall_parameter(result["electron_temperature"], point["gas_temperature"], ionization_potential, ionization_fraction)
    return beta_crit / result["hall_parameter"] - 1.0


def _find_crossings_1d(margin_minus_one_values: np.ndarray, axis_values: np.ndarray, objective) -> list[float]:
    """Every value of `axis_values` (refined by bisection) where a precomputed 1-D
    array of (beta_crit/beta - 1) changes sign -- the vectorized-scan counterpart of
    scanning-then-bisecting one point at a time. `objective(value) -> beta_crit/beta - 1`
    is only called a handful of times per detected crossing (brentq's own refinement),
    not once per scan point.

    Scanning for *all* sign changes -- not just checking the two endpoints -- matters
    because this system can have a genuine stability *window*
    (unstable-stable-unstable) rather than a single threshold: checking only the
    endpoints can't tell "never crosses" apart from "crosses an even number of times,
    same sign at both ends"."""
    sign_changes = np.where(np.diff(np.sign(margin_minus_one_values)) != 0)[0]
    roots = []
    for k in sign_changes:
        try:
            roots.append(brentq(objective, axis_values[k], axis_values[k + 1], xtol=1e-4 * axis_values[k] + 1e-8))
        except (ValueError, RuntimeError):
            pass
    return sorted(roots)


def critical_load_resistivity_surface(
        hall_solver: HallSolver, base: dict,
        seed_fraction_values: np.ndarray, b0_values: np.ndarray,
        load_resistivity_bracket: tuple[float, float] = (1e-4, 20.0), scan_points: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For each (seed_fraction, B0), finds the critical load resistivity -- the stability
    boundary surface over the three inputs that matter most directly: seed fraction and
    B0 together set beta and (via Saha/energy balance) f_I, and load resistivity is the
    next most consequential *free dial* (as opposed to Tp or the flow conditions, which
    are usually fixed by the working gas/environment rather than tuned) -- it sets
    Z = eta_L/eta, which Delta_T depends on through the whole current/heating loop
    (eq. 3.14).

    Returns (lower, upper, lower_power, upper_power): the boundary height(s) in Ohm*m --
    `upper` is all-NaN unless the margin actually re-crosses 1 at higher resistivity
    (checked numerically during development: not observed in the range tested here, but
    handled the same way as the B0 stability window for robustness) -- plus the load
    power density S_L *at* each boundary point, so the surface's color can carry a
    genuinely independent piece of information instead of re-deriving the height.

    The scan itself (the expensive part -- every (B0, seed_fraction, load_resistivity)
    combination) is one vectorized `_margin_batch` call; only the brentq refinement of
    each detected crossing falls back to a handful of individual (still batch-based,
    just scalar-shaped) evaluations."""
    scan_values = np.logspace(np.log10(load_resistivity_bracket[0]), np.log10(load_resistivity_bracket[1]), scan_points)
    B0, SF, LR = np.meshgrid(b0_values, seed_fraction_values, scan_values, indexing="ij")
    values = _margin_batch(hall_solver, base, B0=B0, seed_fraction=SF, load_resistivity=LR)  # shape (len(b0), len(sf), scan_points)

    shape = (len(b0_values), len(seed_fraction_values))
    lower = np.full(shape, np.nan)
    upper = np.full(shape, np.nan)
    lower_power = np.full(shape, np.nan)
    upper_power = np.full(shape, np.nan)

    for i, b0 in enumerate(b0_values):
        for j, seed_fraction in enumerate(seed_fraction_values):
            def objective(lr, b0=b0, seed_fraction=seed_fraction) -> float:
                return float(_margin_batch(hall_solver, base, B0=b0, seed_fraction=seed_fraction, load_resistivity=lr))

            roots = _find_crossings_1d(values[i, j, :], scan_values, objective)
            if not roots:
                continue
            lower[i, j] = roots[0]
            point = _resolve_point(base, B0=b0, seed_fraction=seed_fraction, load_resistivity=roots[0])
            lower_power[i, j] = float(hall_solver.solve_equilibrium_batch(**point)["load_power_density"])
            if len(roots) > 1:
                upper[i, j] = roots[-1]
                point = _resolve_point(base, B0=b0, seed_fraction=seed_fraction, load_resistivity=roots[-1])
                upper_power[i, j] = float(hall_solver.solve_equilibrium_batch(**point)["load_power_density"])
    return lower, upper, lower_power, upper_power


def plot_critical_load_resistivity_surface(
        lower: np.ndarray, upper: np.ndarray, lower_power: np.ndarray, upper_power: np.ndarray,
        seed_fraction_values: np.ndarray, b0_values: np.ndarray, base: dict,
) -> plt.Figure:
    """3-D surface of the critical (marginal-stability) load resistivity as a function of
    seed fraction and B0 -- the "stability boundary surface" over the three inputs that
    matter most directly to the criterion. Height is log10(critical eta_L) (it spans
    orders of magnitude); color is log10(load power density S_L) delivered *at* that
    boundary point (also spans orders of magnitude) -- an independent quantity from the
    height, rather than just re-plotting the z-axis as color.

    Increasing eta_L only ever pushes the system *toward* stability here (raising Z drives
    Delta_T -> 0, which drives beta_crit -> infinity -- verified numerically: no
    re-destabilisation was found anywhere in the tested range), so the stable side is
    unambiguous: everything ABOVE the surface (higher load resistivity than critical).
    That's marked two ways: sparse upward arrows on the surface itself (readable from any
    viewing angle) and a caption spelling it out."""
    log_seed_fraction = np.log10(seed_fraction_values)
    X, Y = np.meshgrid(log_seed_fraction, b0_values)  # shape (len(b0_values), len(seed_fraction_values))

    finite_power = np.concatenate([lower_power[np.isfinite(lower_power)], upper_power[np.isfinite(upper_power)]])
    finite_power = finite_power[finite_power > 0]
    norm = mcolors.LogNorm(vmin=np.min(finite_power), vmax=np.max(finite_power)) if finite_power.size else mcolors.LogNorm(1.0, 10.0)
    cmap = plt.get_cmap("plasma")

    fig = plt.figure(figsize=(8, 7.5))
    ax = fig.add_subplot(projection="3d")

    def add_surface(height: np.ndarray, power: np.ndarray, **kwargs):
        safe_power = np.where(np.isfinite(power) & (power > 0), power, norm.vmin)
        facecolors = cmap(norm(safe_power))
        return ax.plot_surface(X, Y, np.ma.masked_invalid(np.log10(height)), facecolors=facecolors, **kwargs)

    add_surface(lower, lower_power, edgecolor="none", alpha=0.95)
    has_upper = np.any(np.isfinite(upper))
    if has_upper:
        add_surface(upper, upper_power, edgecolor="none", alpha=0.7)

    # Sparse "stable this way" markers straight up off the lower surface, in log10(eta_L)
    # units so a fixed length reads sensibly regardless of the surface's actual magnitude
    # at that point.
    stride_x, stride_y = max(1, len(seed_fraction_values) // 6), max(1, len(b0_values) // 6)
    log_lower = np.log10(lower)
    _draw_stable_direction_markers(
        ax, X[::stride_y, ::stride_x], Y[::stride_y, ::stride_x], log_lower[::stride_y, ::stride_x], length=0.6,
    )

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(mappable, ax=ax, shrink=0.55, pad=0.05, label=r"load power density $S_L$ [W/m$^3$] at the boundary (log scale)")

    ax.set_xlabel(r"$\log_{10}$(seed fraction)")
    ax.set_ylabel(r"$B_0$ [T]")
    ax.set_zlabel(r"$\log_{10}$(critical $\eta_L$ [$\Omega\cdot$m])")
    ax.set_title(f"Stability boundary: critical $\\eta_L(B_0,\\ \\mathrm{{seed\\ fraction}})$\n{STABILITY_NOTE}", fontsize=11.5)

    fig.tight_layout()
    fig.subplots_adjust(top=0.90, bottom=0.16)
    fig.text(
        0.5, 0.02,
        _fixed_params_text(base, "B0", "seed_fraction", "load_resistivity")
        + "\nblue arrows point toward the stable side (higher $\\eta_L$ than critical)",
        ha="center", fontsize=8.5,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.85),
    )
    return fig


def _matched_load_batch(hall_solver: HallSolver, base: dict, seed_fraction, magnetic_field, gas_temperature, iters: int = 15) -> dict:
    """Vectorized Friedberg (6.10) matched-load (Z=sqrt(1+beta^2)) equilibrium solve --
    the array analogue of the scalar outer fixed-point loop an earlier version of this
    script used (one HallSolver.solve_equilibrium call per grid point per outer
    iteration): `iters` rounds of `solve_equilibrium_batch` instead, solving an entire
    grid's worth of points (any broadcastable shape, e.g. a 3-D meshgrid) in one pass.
    This is what makes a dense 3-D (seed_fraction, B0, Tp) grid affordable at all -- see
    stability_volume_grid's docstring for the measured cost difference."""
    gas_number_density = base["p0"] / (constants.k * gas_temperature)
    seed_number_density = seed_fraction * gas_number_density
    load_resistivity = base["load_resistivity"]
    result = None
    for _ in range(iters):
        result = hall_solver.solve_equilibrium_batch(
            flow_speed=base["v0"], gas_temperature=gas_temperature, gas_number_density=gas_number_density,
            seed_number_density=seed_number_density, magnetic_field=magnetic_field, load_resistivity=load_resistivity,
        )
        load_resistivity = np.sqrt(1.0 + result["hall_parameter"] ** 2) * result["resistivity"]
    return result


def stability_volume_grid(
        hall_solver: HallSolver, base: dict,
        seed_fraction_values: np.ndarray, b0_values: np.ndarray, tp_values: np.ndarray,
) -> dict:
    """Precomputed 3-D equilibrium/stability table over (seed_fraction, B0, Tp), solved
    with the Friedberg (6.10) matched-load Z=sqrt(1+beta^2) policy throughout -- load
    resistivity is never a free axis here either (see _matched_load_batch, and this
    function's git history for why Z=beta^2+1, maximizing power outright, was tried
    first and abandoned: it pushed the whole system toward physically unreasonable
    sub-100K Tp at high field).

    This is the "precompute a shared equilibrium table" this script's redesign was built
    around: solving the WHOLE grid is one vectorized pass via _matched_load_batch,
    however fine -- a 40x40x40 grid (64,000 points) solves in well under a second, versus
    the ~16s an earlier, much coarser (14x14, with an internal per-cell root-search) 2-D
    version of this analysis took. That's what makes a dense enough 3-D grid to resolve
    real structure (like the Tp ~ 2000-5000 K re-entrant unstable pocket found during
    development) actually affordable, where a scan-and-bisect search extended to 3-D
    would not have been.

    Returns a dict of 3-D arrays (shape (len(seed_fraction), len(B0), len(Tp))): margin
    (beta_crit/beta), stable (margin>=1), Te, ionization_fraction, load_power_density."""
    SF, B0, TP = np.meshgrid(seed_fraction_values, b0_values, tp_values, indexing="ij")
    result = _matched_load_batch(hall_solver, base, SF, B0, TP)

    ionization_potential = default_seed_type().ionization_potential
    ionization_fraction = result["electron_number_density"] / result["seed_number_density"]
    beta_crit = critical_hall_parameter(result["electron_temperature"], TP, ionization_potential, ionization_fraction)
    margin = stability_margin(result["hall_parameter"], beta_crit)

    return dict(
        margin=margin,
        stable=margin >= 1.0,
        Te=result["electron_temperature"],
        ionization_fraction=ionization_fraction,
        load_power_density=result["load_power_density"],
    )


def stability_boundary_mesh(
        grid: dict, seed_fraction_values: np.ndarray, b0_values: np.ndarray, tp_values: np.ndarray, level: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Marching-cubes isosurface (skimage.measure.marching_cubes) of the precomputed
    margin field at margin=`level`=1 (the stability boundary), replacing an earlier
    version that extracted a scatter of boundary-adjacent grid cells instead.

    That scatter version showed visible "terracing": it upsampled the coarse `grid`
    with RegularGridInterpolator's trilinear interpolation, then just flagged whole
    fine cells as boundary-adjacent. Trilinear interpolation connects coarse samples
    with straight-line ramps -- wherever the true field is a near-cliff (this system's
    ionisation-avalanche transitions are exactly that), the interpolated crossing
    location within each coarse cell is very sensitive to which cell you're in, and
    stitching cells together traces out the coarse grid's own cell boundaries. Marching
    cubes doesn't have this failure mode: it solves for the exact interpolated crossing
    position along each grid edge from the real corner values directly, giving sub-cell
    accuracy instead of "which cell", and returns a proper triangulated surface (a few
    thousand vertices) rather than a ~100,000-point scatter -- both smoother-looking and
    far lighter to render.

    Returns (vertices, faces, vertex_power): vertices in REAL coordinates
    (log10(seed_fraction), B0, Tp [K, linear]) -- not the fractional grid-index
    coordinates marching_cubes itself works in -- faces as vertex-index triples, and the
    load power density interpolated at each vertex (for coloring)."""
    margin_capped = np.clip(np.nan_to_num(grid["margin"], nan=0.0, posinf=1e6, neginf=0.0), 0.0, 1e6)
    index_vertices, faces, _normals, _values = marching_cubes(margin_capped, level=level)

    log_sf = np.log10(seed_fraction_values)
    log_tp = np.log10(tp_values)
    real_log_sf = np.interp(index_vertices[:, 0], np.arange(len(seed_fraction_values)), log_sf)
    real_b0 = np.interp(index_vertices[:, 1], np.arange(len(b0_values)), b0_values)
    real_tp = 10.0 ** np.interp(index_vertices[:, 2], np.arange(len(tp_values)), log_tp)
    vertices = np.stack([real_log_sf, real_b0, real_tp], axis=-1)

    vertex_power = map_coordinates(grid["load_power_density"], index_vertices.T, order=1, mode="nearest")
    return vertices, faces, vertex_power


def stability_boundary_stable_direction_segments(
        grid: dict, seed_fraction_values: np.ndarray, b0_values: np.ndarray, tp_values: np.ndarray,
        vertices: np.ndarray, faces: np.ndarray,
        num_arrows: int = 30, eps_frac: float = 0.01, arrow_frac: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """For `num_arrows` faces sampled evenly across the mesh, a short line segment from
    that face's centroid pointing toward increasing margin (the stable side) -- unlike
    the earlier surface plots (where "up" always meant "more stable"), this mesh can
    face in genuinely different directions on different sheets, so each arrow needs its
    own locally-estimated direction rather than a single fixed one.

    (marching_cubes' own per-vertex `normals` were checked directly against the margin
    field first -- confirmed, consistently, to point toward the *unstable* side, i.e.
    the negative gradient -- but rather than rely on that sign convention, this instead
    estimates the gradient itself directly from the margin field via a small centered
    finite difference, in (log10(seed_fraction), B0, log10(Tp)) space -- the space the
    grid is actually uniform in -- then converts back to (log10(seed_fraction), B0,
    Tp-linear) only for the final plotted coordinates. Each axis's step/arrow length is
    a fixed fraction of that axis's own range, so arrows stay visually sensible despite
    Tp spanning thousands of K while log10(seed_fraction) spans only ~4.

    Returns (starts, ends), both (N, 3) arrays in the same real coordinates as
    `vertices`."""
    log_sf = np.log10(seed_fraction_values)
    log_tp = np.log10(tp_values)
    margin_capped = np.clip(np.nan_to_num(grid["margin"], nan=0.0, posinf=1e6, neginf=0.0), 0.0, 1e6)
    interpolator = RegularGridInterpolator((log_sf, b0_values, log_tp), margin_capped, bounds_error=False, fill_value=None)
    ranges = np.array([log_sf[-1] - log_sf[0], b0_values[-1] - b0_values[0], log_tp[-1] - log_tp[0]])

    stride = max(1, len(faces) // num_arrows)
    starts, ends = [], []
    for face in faces[::stride]:
        centroid_real = vertices[face].mean(axis=0)
        centroid_query = np.array([centroid_real[0], centroid_real[1], np.log10(centroid_real[2])])

        gradient = np.empty(3)
        for k in range(3):
            step = np.zeros(3)
            step[k] = eps_frac * ranges[k]
            gradient[k] = (interpolator(centroid_query + step)[0] - interpolator(centroid_query - step)[0]) / (2.0 * eps_frac)
        norm = np.linalg.norm(gradient)
        if norm < 1e-9:
            continue
        end_query = centroid_query + (gradient / norm) * arrow_frac * ranges

        starts.append(centroid_real)
        ends.append(np.array([end_query[0], end_query[1], 10.0 ** end_query[2]]))
    return np.array(starts), np.array(ends)


def plot_stability_boundary_mesh(
        vertices: np.ndarray, faces: np.ndarray, vertex_power: np.ndarray,
        starts: np.ndarray, ends: np.ndarray, base: dict,
) -> plt.Figure:
    """Solid triangulated surface (mpl_toolkits.mplot3d.art3d.Poly3DCollection) of the
    stability boundary -- the full shape, including the Tp ~ 2000-5000 K re-entrant
    unstable pocket that made an assumed 2-surface version of this plot unreadable,
    comes through directly as an actual continuous surface rather than a dense scatter.
    Color is load power density per face (log scale, averaged from its vertices) --
    independent information, not a repaint of position. Blue segments mark the stable
    direction at a sample of faces (see stability_boundary_stable_direction_segments)."""
    face_power = vertex_power[faces].mean(axis=1)
    finite_power = face_power[np.isfinite(face_power) & (face_power > 0)]
    norm = mcolors.LogNorm(vmin=np.min(finite_power), vmax=np.max(finite_power)) if finite_power.size else mcolors.LogNorm(1.0, 10.0)
    safe_power = np.where(np.isfinite(face_power) & (face_power > 0), face_power, norm.vmin)
    cmap = plt.get_cmap("plasma")

    fig = plt.figure(figsize=(8, 7.5))
    ax = fig.add_subplot(projection="3d")
    mesh = Poly3DCollection(vertices[faces], facecolors=cmap(norm(safe_power)), edgecolor="none", alpha=0.9)
    ax.add_collection3d(mesh)
    # add_collection3d doesn't auto-scale the axes -- set limits from the data explicitly.
    ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())

    for s, e in zip(starts, ends):
        ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color="steelblue", linewidth=1.5)
    if len(ends):
        ax.scatter(ends[:, 0], ends[:, 1], ends[:, 2], marker="^", color="steelblue", s=12, depthshade=False)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(mappable, ax=ax, shrink=0.55, pad=0.05, label=r"load power density $S_L$ [W/m$^3$] at the boundary (log scale)")

    ax.set_xlabel(r"$\log_{10}$(seed fraction)")
    ax.set_ylabel(r"$B_0$ [T]")
    ax.set_zlabel(r"$T_p$ [K]")
    ax.set_title(
        f"Stability boundary surface: $B_0$, seed fraction, $T_p$ (matched load, $Z=\\sqrt{{1+\\beta^2}}$)\n{STABILITY_NOTE}",
        fontsize=10.5,
    )

    fig.tight_layout()
    fig.subplots_adjust(top=0.88, bottom=0.16)
    fig.text(
        0.5, 0.02,
        _fixed_params_text(base, "B0", "seed_fraction", "Tp", "load_resistivity")
        + "\n(load resistivity fixed everywhere to the matched-load value $Z=\\sqrt{1+\\beta^2}$, eq. 6.10, not swept)"
        "\nblue segments point toward the stable side (a locally-estimated direction, not always \"up\" -- see docstring)",
        ha="center", fontsize=8, style="italic",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.85),
    )
    return fig


def plot_stability_boundary_mesh_interactive(
        vertices: np.ndarray, faces: np.ndarray, vertex_power: np.ndarray, starts: np.ndarray, ends: np.ndarray,
):
    """Interactive (WebGL) counterpart to plot_stability_boundary_mesh, for actually
    rotating it smoothly -- matplotlib's Poly3DCollection has no GPU acceleration
    either, though this mesh (a few thousand vertices) is light enough that it matters
    far less than it did for the ~100,000-point scatter version. Plotly's Mesh3d
    renders through WebGL regardless; this repo already depends on plotly (see
    ui/app.py) for exactly this reason.

    Returns a plotly Figure, saved to a standalone .html file by main() -- openable in
    any browser, no server needed. Only called if `plotly` is importable; main() skips
    this (with a note) otherwise, since plotly isn't a required dependency of this
    script."""
    import plotly.graph_objects as go

    safe_power = np.where(np.isfinite(vertex_power) & (vertex_power > 0), vertex_power, np.nan)
    log_power = np.log10(safe_power)

    segment_x, segment_y, segment_z = [], [], []
    for s, e in zip(starts, ends):
        segment_x += [s[0], e[0], None]
        segment_y += [s[1], e[1], None]
        segment_z += [s[2], e[2], None]

    fig = go.Figure(data=[
        go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            intensity=log_power, colorscale="Plasma", colorbar=dict(title="log10(S_L)<br>[W/m³]"),
            opacity=1.0, flatshading=False,
        ),
        go.Scatter3d(
            x=segment_x, y=segment_y, z=segment_z, mode="lines",
            line=dict(color="steelblue", width=6), showlegend=False,
        ),
    ])
    fig.update_layout(
        title=f"Stability boundary surface (matched load, Z=√(1+β²)) -- {len(faces):,} faces, drag to rotate",
        scene=dict(
            xaxis_title="log10(seed fraction)",
            yaxis_title="B0 [T]",
            zaxis_title="Tp [K]",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def seed_fraction_ceiling_asymptotic(
        electron_temperature_values: np.ndarray, beta_values: np.ndarray, reference_gas_number_density: float,
        ns_bracket: tuple[float, float] = (1e14, 1e28),
) -> np.ndarray:
    """Reproduces Friedberg's (6.27)-style *ceiling* directly: holding Te fixed
    (prescribed here, NOT solved via the energy-balance equilibrium the rest of this
    script uses) and beta fixed as an external design target, finds the largest seed
    density n_s for which the high-ionisation asymptotic criterion (6.23) still holds.
    Verified numerically first (see the conversation this script grew out of): at fixed
    Te, beta_crit_asymptotic decreases monotonically as n_s increases (more seed atoms
    dilutes f_I away from 1), so there's a real ceiling and no matching floor from the
    stability criterion alone -- consistent with (6.27) itself, and with there being no
    stability-side reason for a *minimum* seed density (Friedberg's stated minimum comes
    from a separate design constraint, hitting a target load power density S_C, which
    isn't modelled anywhere in this script).

    This deliberately does NOT go through HallSolver.solve_equilibrium. That iterative
    solve lets Te float to whatever the energy balance demands, which is exactly why
    critical_load_resistivity_surface (and every self-consistent sweep above it) never
    shows this ceiling -- verified numerically: at any seed fraction there, the critical
    point's Te climbs right alongside seed fraction (roughly 4,600 K to 11,200 K across
    that surface's whole range), always staying just far enough above the ceiling to
    keep f_I within reach of 1. Friedberg's (6.27) instead treats Te as prescribed rather
    than solved for, so reproducing it means doing the same here.

    Returns ceiling seed density expressed as a fraction of `reference_gas_number_density`
    -- purely for display alongside the other seed-fraction plots; this analysis has no
    actual primary-gas dependence (Tp, p0 don't enter it at all), unlike everywhere else
    in this script."""
    ionization_model = LocalThermodynamicEquilibrium(seed_type=default_seed_type())
    ionization_potential = default_seed_type().ionization_potential

    def margin_minus_one(ns: float, Te: float, beta: float) -> float:
        f_I = ionization_model.get_electron_density(Te, ns) / ns
        return critical_hall_parameter_asymptotic(Te, ionization_potential, f_I) / beta - 1.0

    shape = (len(electron_temperature_values), len(beta_values))
    ceiling_ns = np.full(shape, np.nan)
    lo, hi = ns_bracket
    for i, Te in enumerate(electron_temperature_values):
        for j, beta in enumerate(beta_values):
            try:
                if np.sign(margin_minus_one(lo, Te, beta)) != np.sign(margin_minus_one(hi, Te, beta)):
                    ceiling_ns[i, j] = brentq(margin_minus_one, lo, hi, args=(Te, beta), xtol=1.0)
            except (ValueError, RuntimeError):
                pass
    return ceiling_ns / reference_gas_number_density


def plot_seed_fraction_ceiling(
        beta_values: np.ndarray, electron_temperature_values: np.ndarray, ceiling_seed_fraction: np.ndarray,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for Te, row in zip(electron_temperature_values, ceiling_seed_fraction):
        ax.plot(beta_values, row, linewidth=2, label=f"$T_e$ = {Te:,.0f} K")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Hall parameter $\beta$ (design target)")
    ax.set_ylabel(r"maximum seed fraction $n_s/n_{p0}$ (ceiling)")
    ax.set_title(f"Maximum allowable seed density at fixed $T_e$ (eq. 6.23/6.27)\n{STABILITY_NOTE}", fontsize=11.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    fig.text(
        0.5, 0.03,
        r"$T_e$ prescribed directly here, unlike every self-consistent sweep above -- see"
        "\nseed_fraction_ceiling_asymptotic()'s docstring for why that distinction is exactly what hides this ceiling elsewhere",
        ha="center", fontsize=8, style="italic",
    )
    return fig


def seed_fraction_min_max_surface(
        electron_temperature_values: np.ndarray, beta_values: np.ndarray,
        reference_gas_number_density: float, reference_flow_speed: float,
        target_power_density: float = 100e6,
        ns_bracket: tuple[float, float] = (1e14, 1e28),
) -> tuple[np.ndarray, np.ndarray]:
    """The 3-D generalisation of seed_fraction_ceiling_asymptotic: MAXIMUM seed density
    (the same eq. 6.23/6.27 stability ceiling) alongside a genuine MINIMUM, both as
    surfaces over (Te, beta).

    The maximum comes from stability alone, exactly as in seed_fraction_ceiling_
    asymptotic. The minimum comes from Friedberg's *other* design constraint: delivering
    at least an "industrially relevant" load power density S_C (~100 MW/m^3 is the
    paper's own figure, see Sec. 2/7), combined with the same matched-load choice of Z
    used in stability_volume_grid/_matched_load_batch (Z = sqrt(1+beta^2), eq. 6.10). At
    that Z, with s = sqrt(1+beta^2), S_L simplifies to S_L = m_e n_e nu_M(Te) v_p^2 beta^4
    / [s(s+1)^2] -- still linear in n_e, so hitting S_C requires
    n_e >= S_C * s(s+1)^2 / (m_e nu_M v_p^2 beta^4). This is closed-form algebra, not a
    root-find, and -- like the ceiling -- never touches HallSolver.solve_equilibrium;
    nu_M and v_p come from a REFERENCE operating point (S_C is an absolute power density,
    not a ratio, so it needs *some* concrete n_p, v_p to anchor to). Callers should pass
    PAPER_REFERENCE_GAS_NUMBER_DENSITY/PAPER_REFERENCE_FLOW_SPEED, not
    base_operating_point()'s -- checked numerically first: with this repo's own (much
    more dilute, slower) default as the reference, the region where max_ns > min_ne
    (i.e. where both constraints can be satisfied at once) covers only ~4% of a typical
    (Te, beta) grid, versus ~75% with Friedberg's own reference conditions. S_C=100 MW/m^3
    is an absolute figure the paper derived to pair with its own reference point; pairing
    it with a much lower-density, slower one just makes it far harder to reach.

    Together these two independently-derived bounds reproduce exactly what Friedberg's
    Introduction describes: "the combination of the marginal stability criterion plus the
    practical requirement of an industrial relevant load power density...represent two
    design constraints. Their simultaneous solution leads to specific values for the
    electron temperature and seed density." Returns (max_ns, min_ne), both as absolute
    number densities [m^-3] (divide by a reference n_p for a seed-fraction-like ratio, as
    plot_seed_fraction_min_max_surface does)."""
    seed_type = default_seed_type()
    ionization_model = LocalThermodynamicEquilibrium(seed_type=seed_type)
    transport_model = MHDTransportModel(seed_type=seed_type, gas_type=default_gas_type())

    def ceiling_margin_minus_one(ns: float, Te: float, beta: float) -> float:
        f_I = ionization_model.get_electron_density(Te, ns) / ns
        return critical_hall_parameter_asymptotic(Te, seed_type.ionization_potential, f_I) / beta - 1.0

    shape = (len(electron_temperature_values), len(beta_values))
    max_ns = np.full(shape, np.nan)
    min_ne = np.full(shape, np.nan)
    lo, hi = ns_bracket
    for i, Te in enumerate(electron_temperature_values):
        nu_M = transport_model.get_momentum_transfer_frequency(Te, reference_gas_number_density)
        for j, beta in enumerate(beta_values):
            try:
                if np.sign(ceiling_margin_minus_one(lo, Te, beta)) != np.sign(ceiling_margin_minus_one(hi, Te, beta)):
                    max_ns[i, j] = brentq(ceiling_margin_minus_one, lo, hi, args=(Te, beta), xtol=1.0)
            except (ValueError, RuntimeError):
                pass
            s = np.sqrt(1.0 + beta ** 2)
            min_ne[i, j] = (
                target_power_density * s * (s + 1.0) ** 2
                / (constants.electron_mass * nu_M * reference_flow_speed ** 2 * beta ** 4)
            )
    return max_ns, min_ne


def plot_seed_fraction_min_max_surface(
        electron_temperature_values: np.ndarray, beta_values: np.ndarray,
        max_ns: np.ndarray, min_ne: np.ndarray, reference_gas_number_density: float, target_power_density: float,
) -> plt.Figure:
    """3-D surfaces bounding the *stable and sufficiently powerful* seed density window
    over (Te, beta): `max_ns` (the stability ceiling) and `min_ne` (the target-power
    floor). The region strictly between the two surfaces, at a given (Te, beta), is where
    BOTH of Friedberg's design constraints hold at once.

    Each surface is a single flat color (matching its legend entry), not a colormap --
    an earlier version colored each by its own height (matplotlib's plot_surface default
    when given `cmap` instead of `color`), which just re-plotted the z-axis as color a
    second time with no independent meaning, and looked like decoration rather than data
    once the two surfaces started crossing each other."""
    X, Y = np.meshgrid(electron_temperature_values, beta_values, indexing="ij")
    log_max = np.log10(max_ns / reference_gas_number_density)
    log_min = np.log10(min_ne / reference_gas_number_density)

    fig = plt.figure(figsize=(8, 7.2))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, log_max, color="darkorange", edgecolor="none", alpha=0.85)
    ax.plot_surface(X, Y, log_min, color="steelblue", edgecolor="none", alpha=0.75)

    ax.plot([], [], color="darkorange", linewidth=6, label="maximum (stability ceiling, eq. 6.23/6.27)")
    ax.plot([], [], color="steelblue", linewidth=6, label=f"minimum ($S_L \\geq$ {target_power_density / 1e6:.0f} MW/m$^3$, matched load, eq. 6.10)")
    ax.legend(loc="upper left", fontsize=8)

    ax.set_xlabel(r"$T_e$ [K]")
    ax.set_ylabel(r"$\beta$")
    ax.set_zlabel(r"$\log_{10}$(seed fraction $n_s/n_{p0}$)")
    ax.set_title(f"Seed density window: stability ceiling vs. power floor\n{STABILITY_NOTE}", fontsize=11.5)

    fig.tight_layout()
    fig.subplots_adjust(top=0.90, bottom=0.16)
    fig.text(
        0.5, 0.02,
        "the region strictly BETWEEN the two surfaces satisfies both constraints (stability AND the target power density)",
        ha="center", fontsize=8.5,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.85),
    )
    return fig


def demo_sweeps(hall_solver: HallSolver, base: dict) -> list[tuple[str, plt.Figure]]:
    figures = []

    # 1. B0 vs seed_fraction: the paper's headline field/ionisation-degree tradeoff.
    b0_values = np.linspace(0.1, 5.0, 250)
    seed_fraction_values = np.logspace(-5, -1, 250)
    grid = sweep_grid(hall_solver, base, "B0", b0_values, "seed_fraction", seed_fraction_values)
    figures.append(("b0_vs_seed_fraction", plot_stability_boundary(
        grid, base, "B0", b0_values, "seed_fraction", seed_fraction_values,
        "Stability boundary: $B_0$ vs. seed fraction", r"$B_0$ [T]", r"seed fraction $n_{s0}/n_{p0}$",
        y_log=True,
    )))

    # 2. B0 vs Tp: isolates the Delta_T dependence unique to the exact criterion (absent
    # from the asymptotic one) -- the exact/asymptotic boundaries should visibly diverge.
    tp_values = np.linspace(500.0, 3000.0, 250)
    grid = sweep_grid(hall_solver, base, "B0", b0_values, "Tp", tp_values)
    figures.append(("b0_vs_Tp", plot_stability_boundary(
        grid, base, "B0", b0_values, "Tp", tp_values,
        "Stability boundary: $B_0$ vs. primary gas temperature", r"$B_0$ [T]", r"$T_p$ [K]",
    )))

    # 3. load_resistivity vs B0: ties back to the paper's S_Omega/S_L figures -- the load
    # choice pulls Te around indirectly through the energy balance, shifting the margin.
    load_resistivity_values = np.logspace(-2, 0.3, 250)
    grid = sweep_grid(hall_solver, base, "B0", b0_values, "load_resistivity", load_resistivity_values)
    figures.append(("load_resistivity_vs_B0", plot_stability_boundary(
        grid, base, "B0", b0_values, "load_resistivity", load_resistivity_values,
        r"Stability boundary: $B_0$ vs. load resistivity", r"$B_0$ [T]", r"$\eta_L$ [$\Omega\cdot$m]",
        y_log=True,
    )))

    # 4. seed_fraction vs p0: does a denser primary gas make it easier or harder to reach
    # (and sustain) the near-full-ionisation stable branch at fixed B0?
    p0_values = np.logspace(3.0, 5.0, 250)
    grid = sweep_grid(hall_solver, base, "p0", p0_values, "seed_fraction", seed_fraction_values)
    figures.append(("seed_fraction_vs_p0", plot_stability_boundary(
        grid, base, "p0", p0_values, "seed_fraction", seed_fraction_values,
        "Stability boundary: inlet pressure vs. seed fraction", r"$p_0$ [Pa]", r"seed fraction $n_{s0}/n_{p0}$",
        y_log=True,
    )))

    # 5. seed_fraction vs v0: same question for flow speed (hence Mach number, which
    # drives Ohmic heating through Delta_T ~ M^2).
    v0_values = np.linspace(20.0, 500.0, 250)
    grid = sweep_grid(hall_solver, base, "v0", v0_values, "seed_fraction", seed_fraction_values)
    figures.append(("seed_fraction_vs_v0", plot_stability_boundary(
        grid, base, "v0", v0_values, "seed_fraction", seed_fraction_values,
        "Stability boundary: flow speed vs. seed fraction", r"$v_0$ [m/s]", r"seed fraction $n_{s0}/n_{p0}$",
        y_log=True,
    )))

    # 6. v0 vs p0 at the default (fixed) seed fraction -- unlike 4 and 5, this pair
    # doesn't touch seed_fraction at all, so it shouldn't hit the ionisation-avalanche
    # bifurcation those two do; a useful contrast for how much smoother the boundary looks
    # when it doesn't.
    grid = sweep_grid(hall_solver, base, "v0", v0_values, "p0", p0_values)
    figures.append(("v0_vs_p0", plot_stability_boundary(
        grid, base, "v0", v0_values, "p0", p0_values,
        "Stability boundary: flow speed vs. inlet pressure", r"$v_0$ [m/s]", r"$p_0$ [Pa]",
        y_log=True,
    )))

    # 7. Genuine 3-D boundary surface over the three inputs most directly implicated in
    # the criterion: seed fraction, B0, and critical load resistivity (color = load power
    # density at the boundary -- independent information from the height).
    seed_fraction_surface_values = np.logspace(-5, -1, 25)
    b0_surface_values = np.linspace(0.1, 5.0, 25)
    lower, upper, lower_power, upper_power = critical_load_resistivity_surface(
        hall_solver, base, seed_fraction_surface_values, b0_surface_values,
    )
    figures.append(("critical_load_resistivity_surface", plot_critical_load_resistivity_surface(
        lower, upper, lower_power, upper_power, seed_fraction_surface_values, b0_surface_values, base,
    )))

    # 8. Direct reproduction of the paper's (6.27)-style maximum seed density ceiling,
    # holding Te fixed rather than solving for it -- see seed_fraction_ceiling_asymptotic's
    # docstring for why that distinction is exactly what hides this ceiling in (7) above.
    beta_values = np.logspace(0.0, 2.5, 60)
    electron_temperature_values = np.array([5000.0, 15000.0, 40000.0])
    reference_gas_number_density = base["p0"] / (constants.k * base["Tp"])
    ceiling_seed_fraction = seed_fraction_ceiling_asymptotic(electron_temperature_values, beta_values, reference_gas_number_density)
    figures.append(("seed_fraction_ceiling", plot_seed_fraction_ceiling(
        beta_values, electron_temperature_values, ceiling_seed_fraction,
    )))

    # 9. The 3-D generalisation of (8): BOTH the stability ceiling (max seed fraction) and,
    # from Friedberg's *other* design constraint -- a target load power density -- a
    # genuine minimum, shown together as two surfaces over (Te, beta). Uses the paper's
    # OWN Sec. 7.1 reference conditions (PAPER_REFERENCE_*), not base_operating_point() --
    # see that constant's comment for why the choice of reference matters here.
    window_te_values = np.linspace(500.0, 10000.0, 30)
    window_beta_values = np.linspace(1.0, 50.0, 30)
    max_ns, min_ne = seed_fraction_min_max_surface(
        window_te_values, window_beta_values, PAPER_REFERENCE_GAS_NUMBER_DENSITY, PAPER_REFERENCE_FLOW_SPEED,
    )
    figures.append(("seed_fraction_min_max_surface", plot_seed_fraction_min_max_surface(
        window_te_values, window_beta_values, max_ns, min_ne, PAPER_REFERENCE_GAS_NUMBER_DENSITY, 100e6,
    )))

    # 10. The full 3-D stability structure over (seed_fraction, B0, Tp), including the
    # Tp ~ 2000-5000 K re-entrant unstable pocket a 2-surface version of this plot
    # couldn't represent -- a precomputed grid (fully vectorized, affordable at much
    # finer resolution than the old per-cell root-search) plus a marching-cubes
    # isosurface, rather than an assumed surface shape or a dense (and, it turned out,
    # visibly terraced) point-cloud scatter.
    volume = stability_volume_grid(hall_solver, base, VOLUME_SEED_FRACTION_VALUES, VOLUME_B0_VALUES, VOLUME_TP_VALUES)
    mesh_vertices, mesh_faces, mesh_vertex_power = stability_boundary_mesh(
        volume, VOLUME_SEED_FRACTION_VALUES, VOLUME_B0_VALUES, VOLUME_TP_VALUES,
    )
    arrow_starts, arrow_ends = stability_boundary_stable_direction_segments(
        volume, VOLUME_SEED_FRACTION_VALUES, VOLUME_B0_VALUES, VOLUME_TP_VALUES, mesh_vertices, mesh_faces,
    )
    figures.append(("stability_boundary_mesh", plot_stability_boundary_mesh(
        mesh_vertices, mesh_faces, mesh_vertex_power, arrow_starts, arrow_ends, base,
    )))

    return figures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show", action="store_true", help="Display each figure interactively.")
    parser.add_argument("--no-save", action="store_true", help="Skip writing PNGs to examples/output/.")
    args = parser.parse_args()

    hall_solver, _ = build_default_hall_solver()
    base = base_operating_point()

    figures = demo_sweeps(hall_solver, base)

    if not args.no_save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for name, fig in figures:
            path = OUTPUT_DIR / f"{name}.png"
            fig.savefig(path, dpi=150)
            print(f"Saved {path}")

        # matplotlib's Poly3DCollection has no GPU acceleration either, so also emit a
        # standalone interactive HTML version via Plotly (WebGL) for actually rotating
        # it; skipped gracefully if plotly isn't installed, since it's optional here
        # (see pyproject.toml's "ui" extra).
        try:
            volume = stability_volume_grid(hall_solver, base, VOLUME_SEED_FRACTION_VALUES, VOLUME_B0_VALUES, VOLUME_TP_VALUES)
            mesh_vertices, mesh_faces, mesh_vertex_power = stability_boundary_mesh(
                volume, VOLUME_SEED_FRACTION_VALUES, VOLUME_B0_VALUES, VOLUME_TP_VALUES,
            )
            arrow_starts, arrow_ends = stability_boundary_stable_direction_segments(
                volume, VOLUME_SEED_FRACTION_VALUES, VOLUME_B0_VALUES, VOLUME_TP_VALUES, mesh_vertices, mesh_faces,
            )
            interactive_fig = plot_stability_boundary_mesh_interactive(
                mesh_vertices, mesh_faces, mesh_vertex_power, arrow_starts, arrow_ends,
            )
        except ImportError:
            print("(plotly not installed -- skipping the interactive HTML surface; `uv sync --extra ui` to enable it)")
        else:
            html_path = OUTPUT_DIR / "stability_boundary_mesh.html"
            interactive_fig.write_html(html_path, include_plotlyjs="cdn")  # don't embed the plotly.js bundle
            print(f"Saved {html_path} (open in a browser to rotate it smoothly)")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
