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
from scipy import constants
from scipy.optimize import brentq

from magnetohydrodynamics.ionization.saha_ionization import LocalThermodynamicEquilibrium
from magnetohydrodynamics.presets import build_default_hall_solver, default_gas_type, default_operating_point, default_seed_type
from magnetohydrodynamics.solver.hall_solver import HallSolver
from magnetohydrodynamics.stability import (
    critical_hall_parameter_asymptotic,
    plasma_critical_hall_parameter,
    plasma_critical_hall_parameter_asymptotic,
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
    stability criteria there. Returns 2-D arrays of shape (len(y_values), len(x_values))."""
    shape = (len(y_values), len(x_values))
    beta = np.empty(shape)
    beta_crit = np.empty(shape)
    beta_crit_asymptotic = np.empty(shape)
    electron_temperature = np.empty(shape)
    ionization_fraction = np.empty(shape)

    for i, y in enumerate(y_values):
        for j, x in enumerate(x_values):
            point = _resolve_point(base, **{x_key: x, y_key: y})
            plasma = hall_solver.solve_equilibrium(**point)
            beta[i, j] = plasma.hall_parameter
            beta_crit[i, j] = plasma_critical_hall_parameter(plasma)
            beta_crit_asymptotic[i, j] = plasma_critical_hall_parameter_asymptotic(plasma)
            electron_temperature[i, j] = plasma.electron_temperature
            ionization_fraction[i, j] = plasma.ionization_fraction

    margin = stability_margin(beta, beta_crit)
    return dict(
        beta=beta,
        beta_crit=beta_crit,
        beta_crit_asymptotic=beta_crit_asymptotic,
        margin=margin,
        margin_asymptotic=stability_margin(beta, beta_crit_asymptotic),
        Te=electron_temperature,
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


def _solve_equilibrium_matched_load(hall_solver: HallSolver, base: dict, overrides: dict, iters: int = 15):
    """Solve the local equilibrium exactly as `_resolve_point` + `solve_equilibrium`
    normally would, except load_resistivity is never a fixed input here -- at every outer
    iteration it's set to Z = sqrt(1+beta^2) at the *current* beta (Friedberg eq. 6.10:
    the "matched load" that minimizes S_Omega/S_L, i.e. wastes the least power to Ohmic
    heating relative to what reaches the load), then the equilibrium is re-solved and
    beta updates, repeating to a fixed point. Iterating is necessary because beta and eta
    both depend on Te, which depends on Z through the Delta_T relation (eq. 3.14) -- so Z,
    beta and Te are mutually coupled.

    An earlier version of this used Z = beta^2+1 instead (maximizing S_L outright, also
    closed-form: differentiating S_L = m_e n_e nu_M v_p^2 beta^4 Z/(beta^2+1+Z)^2 with
    respect to Z and setting it to zero). That policy pushed the whole system toward
    extreme Te at high beta, and the "critical Tp" surface built on it collapsed to below
    100 K (and often below argon's ~87 K boiling point) almost everywhere above ~2 T --
    not a bug, but a real result: simultaneously maximizing power, staying stable, AND
    running at high field just isn't achievable at any physical Tp. Friedberg's own
    matched-load choice is a much gentler policy (Z grows linearly with beta rather than
    quadratically) and lands the same critical-Tp search back in a physically sensible
    ~100-1000s K range across the whole B0 sweep -- verified numerically before switching."""
    load_resistivity = base["load_resistivity"]
    plasma = None
    for _ in range(iters):
        point = _resolve_point(base, **{**overrides, "load_resistivity": load_resistivity})
        plasma = hall_solver.solve_equilibrium(**point)
        load_resistivity = np.sqrt(1.0 + plasma.hall_parameter ** 2) * plasma.resistivity
    return plasma


def _find_margin_crossings(
        hall_solver: HallSolver, base: dict, overrides: dict, solve_key: str, scan_values: np.ndarray,
        solve_equilibrium=None,
) -> list[float]:
    """Every value of `solve_key` (refined by bisection between consecutive samples of
    `scan_values`) where beta_crit/beta crosses 1, holding every other physical knob at
    `base` merged with `overrides`. Scanning for *all* sign changes first -- rather than
    just bisecting between scan_values' two endpoints -- matters because this system can
    have a genuine stability *window* (unstable-stable-unstable) rather than a single
    threshold: checking only the endpoints can't tell "never crosses" apart from "crosses
    an even number of times, same sign at both ends".

    `solve_equilibrium`, if given, replaces the default `hall_solver.solve_equilibrium(
    **_resolve_point(base, **kwargs))` (e.g. `_solve_equilibrium_matched_load`, above) --
    used wherever load_resistivity shouldn't be a fixed input either."""
    solve = solve_equilibrium or (lambda kwargs: hall_solver.solve_equilibrium(**_resolve_point(base, **kwargs)))

    def margin_minus_one(value: float) -> float:
        plasma = solve({**overrides, solve_key: value})
        return plasma_critical_hall_parameter(plasma) / plasma.hall_parameter - 1.0

    values = np.array([margin_minus_one(v) for v in scan_values])
    sign_changes = np.where(np.diff(np.sign(values)) != 0)[0]
    roots = []
    for k in sign_changes:
        try:
            roots.append(brentq(margin_minus_one, scan_values[k], scan_values[k + 1], xtol=1e-4 * scan_values[k] + 1e-8))
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
    genuinely independent piece of information instead of re-deriving the height."""
    scan_values = np.logspace(np.log10(load_resistivity_bracket[0]), np.log10(load_resistivity_bracket[1]), scan_points)
    shape = (len(b0_values), len(seed_fraction_values))
    lower = np.full(shape, np.nan)
    upper = np.full(shape, np.nan)
    lower_power = np.full(shape, np.nan)
    upper_power = np.full(shape, np.nan)

    for i, b0 in enumerate(b0_values):
        for j, seed_fraction in enumerate(seed_fraction_values):
            overrides = dict(B0=b0, seed_fraction=seed_fraction)
            roots = _find_margin_crossings(hall_solver, base, overrides, "load_resistivity", scan_values)
            if not roots:
                continue
            lower[i, j] = roots[0]
            point = _resolve_point(base, **{**overrides, "load_resistivity": roots[0]})
            lower_power[i, j] = hall_solver.solve_equilibrium(**point).load_power_density
            if len(roots) > 1:
                upper[i, j] = roots[-1]
                point = _resolve_point(base, **{**overrides, "load_resistivity": roots[-1]})
                upper_power[i, j] = hall_solver.solve_equilibrium(**point).load_power_density
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


def critical_temperature_surface_matched_load(
        hall_solver: HallSolver, base: dict,
        seed_fraction_values: np.ndarray, b0_values: np.ndarray,
        tp_bracket: tuple[float, float] = (1.0, 6000.0), scan_points: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Like critical_load_resistivity_surface, but load resistivity is no longer a free
    axis at all -- it's fixed, at every point, to the Friedberg (6.10) matched-load value
    Z = sqrt(1+beta^2) (see _solve_equilibrium_matched_load) -- freeing up the height for
    a more physically interesting quantity: the critical primary gas temperature Tp.

    Returns (lower, upper, lower_power, upper_power), exactly as
    critical_load_resistivity_surface does. An earlier version of this used Z = beta^2+1
    (maximizing load power outright) instead, and its critical-Tp surface collapsed to
    below 100 K -- often below argon's ~87 K boiling point -- for almost every B0 above
    ~2 T: not a bug, but a real finding that maximizing power, staying stable, AND running
    at high field together just isn't achievable at any physical Tp under that policy.
    Switching to the gentler matched-load Z (linear in beta, not quadratic) brings the
    lower boundary back into a physically sensible ~100 K-few 1000s K range across the
    whole B0 sweep (verified numerically first). It also revealed real multi-valued
    structure this system didn't show under the max-power policy: at a fixed
    (B0, seed_fraction), the plasma can go stable, then unstable again in a pocket around
    Tp ~ 2000-5000 K, then stable again at higher Tp still -- hence `upper` (and even
    higher-order crossings, silently folded into just "the largest root found") are worth
    keeping here, unlike in the max-power version where they never appeared."""
    scan_values = np.logspace(np.log10(tp_bracket[0]), np.log10(tp_bracket[1]), scan_points)
    shape = (len(b0_values), len(seed_fraction_values))
    lower = np.full(shape, np.nan)
    upper = np.full(shape, np.nan)
    lower_power = np.full(shape, np.nan)
    upper_power = np.full(shape, np.nan)

    def solve(kwargs):
        return _solve_equilibrium_matched_load(hall_solver, base, kwargs)

    for i, b0 in enumerate(b0_values):
        for j, seed_fraction in enumerate(seed_fraction_values):
            overrides = dict(B0=b0, seed_fraction=seed_fraction)
            roots = _find_margin_crossings(hall_solver, base, overrides, "Tp", scan_values, solve_equilibrium=solve)
            if not roots:
                continue
            lower[i, j] = roots[0]
            lower_power[i, j] = solve({**overrides, "Tp": roots[0]}).load_power_density
            if len(roots) > 1:
                upper[i, j] = roots[-1]
                upper_power[i, j] = solve({**overrides, "Tp": roots[-1]}).load_power_density
    return lower, upper, lower_power, upper_power


def plot_critical_temperature_surface_matched_load(
        lower: np.ndarray, upper: np.ndarray, lower_power: np.ndarray, upper_power: np.ndarray,
        seed_fraction_values: np.ndarray, b0_values: np.ndarray, base: dict,
) -> plt.Figure:
    """3-D surface of the critical primary gas temperature as a function of seed fraction
    and B0, with load resistivity eliminated as a free axis entirely (fixed to the
    Friedberg (6.10) matched-load value Z=sqrt(1+beta^2) at every point instead of being
    swept or held at an arbitrary constant). Color is log10(load power density) at the
    boundary, exactly as in plot_critical_load_resistivity_surface -- genuinely new
    information, not the height re-plotted.

    Deliberately plots ONLY the lowest crossing (the primary ignition threshold), even
    though critical_temperature_surface_matched_load can return a second ("upper") one.
    An earlier version rendered both, and the result was close to unreadable: at this
    resolution the two surfaces fold and self-intersect in ways that don't correspond to
    one continuous "stable window" the way the B0 stability-window plot's two surfaces
    do -- this system can go stable, unstable again in a pocket around Tp ~ 2000-5000 K,
    then stable a third time, and collapsing that into just "lower" and "upper" loses the
    pocket in between without actually producing something easier to read. Showing only
    the first (lowest) crossing sacrifices that extra structure but is honest and legible;
    `upper` is still returned by the surface function for anyone who wants to inspect it
    directly, just not plotted here."""
    log_seed_fraction = np.log10(seed_fraction_values)
    X, Y = np.meshgrid(log_seed_fraction, b0_values)  # shape (len(b0_values), len(seed_fraction_values))

    finite_power = lower_power[np.isfinite(lower_power) & (lower_power > 0)]
    norm = mcolors.LogNorm(vmin=np.min(finite_power), vmax=np.max(finite_power)) if finite_power.size else mcolors.LogNorm(1.0, 10.0)
    cmap = plt.get_cmap("plasma")

    fig = plt.figure(figsize=(8, 7.5))
    ax = fig.add_subplot(projection="3d")

    safe_power = np.where(np.isfinite(lower_power) & (lower_power > 0), lower_power, norm.vmin)
    ax.plot_surface(X, Y, np.ma.masked_invalid(lower), facecolors=cmap(norm(safe_power)), edgecolor="none", alpha=0.95)

    # Sparse "stable this way" markers -- verified numerically that higher Tp is the
    # stable side here, same convention as plot_critical_load_resistivity_surface.
    stride_x, stride_y = max(1, len(seed_fraction_values) // 6), max(1, len(b0_values) // 6)
    tp_span = np.nanmax(lower) - np.nanmin(lower) if np.any(np.isfinite(lower)) else 1.0
    _draw_stable_direction_markers(
        ax, X[::stride_y, ::stride_x], Y[::stride_y, ::stride_x], lower[::stride_y, ::stride_x],
        length=0.15 * max(tp_span, 1.0),
    )

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(mappable, ax=ax, shrink=0.55, pad=0.05, label=r"load power density $S_L$ [W/m$^3$] at the boundary (log scale)")

    ax.set_xlabel(r"$\log_{10}$(seed fraction)")
    ax.set_ylabel(r"$B_0$ [T]")
    ax.set_zlabel(r"critical $T_p$ [K]")
    ax.set_title(
        f"Stability boundary: critical $T_p(B_0,\\ \\mathrm{{seed\\ fraction}})$, matched load ($Z=\\sqrt{{1+\\beta^2}}$)\n{STABILITY_NOTE}",
        fontsize=11,
    )

    fig.tight_layout()
    fig.subplots_adjust(top=0.88, bottom=0.2)
    fig.text(
        0.5, 0.02,
        _fixed_params_text(base, "B0", "seed_fraction", "Tp", "load_resistivity")
        + "\n(load resistivity fixed everywhere to the matched-load value $Z=\\sqrt{1+\\beta^2}$, eq. 6.10, not swept)"
        "\nblue arrows: stable just above the lower boundary -- a narrow unstable pocket can reappear at higher $T_p$ still (see docstring)",
        ha="center", fontsize=8, style="italic",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.85),
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
    used in critical_temperature_surface_matched_load (Z = sqrt(1+beta^2), eq. 6.10). At
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

    # 10. critical_load_resistivity_surface's height (7) replaced with the next most
    # consequential free knob, Tp -- load resistivity is no longer a free axis at all,
    # fixed everywhere to the Friedberg (6.10) matched-load Z=sqrt(1+beta^2) instead.
    # Smaller grid than (7): each point here re-solves an inner fixed-point loop (for Z)
    # inside the Tp scan, substantially more expensive per point.
    seed_fraction_tp_values = np.logspace(-5, -1, 14)
    b0_tp_values = np.linspace(0.1, 5.0, 14)
    lower_tp, upper_tp, lower_tp_power, upper_tp_power = critical_temperature_surface_matched_load(
        hall_solver, base, seed_fraction_tp_values, b0_tp_values,
    )
    figures.append(("critical_temperature_surface_matched_load", plot_critical_temperature_surface_matched_load(
        lower_tp, upper_tp, lower_tp_power, upper_tp_power, seed_fraction_tp_values, b0_tp_values, base,
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

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
