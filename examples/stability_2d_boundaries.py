"""
Visualize the Velikhov-ionisation instability's stability boundary (Friedberg 2025, see
magnetohydrodynamics.stability) as a function of two physical inputs at a time, holding
the rest fixed at the default operating point.

A boundary plot needs a self-consistent local equilibrium (Te, ne, f_I, beta) at every
grid point -- exactly what HallSolver.solve_equilibrium computes for a single slice, and
what magnetohydrodynamics.stability.EquilibriumSweep.grid vectorizes over a whole 2-D
sweep at once.

Of the seven candidate inputs (cross section, seed fraction, T_p, p_p, B, v_p, load
resistivity), the load-resistivity-normalised parameter Z from the paper is *derived*
(it falls out of load_resistivity and the self-consistently solved local resistivity
eta(Te)), not a free input -- so it is never a direct sweep axis here, only ever a
byproduct you could read off a solved grid if you wanted it.

The plotted quantity throughout is the stability margin beta_crit/beta: >= 1 is stable,
< 1 is unstable (Friedberg's criterion is beta <= beta_crit for stability).

Run with:
    uv run python examples/stability_2d_boundaries.py            # saves PNGs, no window
    uv run python examples/stability_2d_boundaries.py --show      # also opens windows
    uv run python examples/stability_2d_boundaries.py --no-save   # windows only
"""
from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from magnetohydrodynamics.presets import build_default_hall_solver, default_seed_type
from magnetohydrodynamics.stability import EquilibriumSweep, OperatingPoint

from plotting_utils import STABILITY_NOTE, draw_boundary_contour, fixed_params_text, parse_show_save_args, save_and_show


def plot_stability_boundary(
        grid: dict, base: OperatingPoint, x_key: str, x_values: np.ndarray, y_key: str, y_values: np.ndarray,
        title: str, x_label: str, y_label: str, y_log: bool = False,
        margin_cap: float = 2.0,
) -> plt.Figure:
    """Continuously-shaded stability margin ratio beta_crit/beta: red (<1) unstable,
    white (=1) marginal, blue (>1) stable, with the exact (5.13) boundary as a solid
    black contour and the high-ionisation asymptotic (6.23) boundary overlaid as a dashed
    lime one -- a distinct color, not just dash pattern, since the two often sit almost on
    top of each other and a same-colored halo would otherwise paint over the dashes in
    their gaps wherever that happens (see draw_boundary_contour's docstring). A caption
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
    # unlike pcolormesh(shading="gouraud"), which linearly blends colors *within* each
    # grid cell regardless of how sharply the underlying field actually changes there.
    # This system's stability transition can be very steep (a near step-function in
    # places -- see e.g. the seed-fraction sweeps' ionisation-avalanche cliff), and
    # gouraud's per-cell linear blend showed up as visible diagonal streaking/banding
    # artifacts across that cliff even at high grid resolution; contourf's level-boundary
    # interpolation doesn't have that failure mode.
    margin_filled = np.nan_to_num(margin, nan=0.0, posinf=margin_cap * 10.0, neginf=0.0)
    levels = np.linspace(0.0, margin_cap, 200)
    mesh = ax.contourf(x_values, y_values, margin_filled, levels=levels, cmap="RdBu", norm=norm, extend="max")
    cbar = fig.colorbar(mesh, ax=ax, label=r"stability margin  $\beta_{crit}/\beta$")
    cbar.ax.axhline(1.0, color="black", linewidth=1.0)  # marks the marginal-stability level on the bar itself

    draw_boundary_contour(ax, x_values, y_values, margin, linewidths=2.0)
    draw_boundary_contour(
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
        0.5, 0.04, fixed_params_text(base, x_key, y_key), ha="center", va="bottom", fontsize=8.5,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.85),
    )
    return fig


def build_figures(sweep: EquilibriumSweep, base: OperatingPoint) -> list[tuple[str, plt.Figure]]:
    figures = []

    # 1. B0 vs seed_fraction: the paper's headline field/ionisation-degree tradeoff.
    b0_values = np.linspace(0.1, 5.0, 250)
    seed_fraction_values = np.logspace(-5, -1, 250)
    grid = sweep.grid("B0", b0_values, "seed_fraction", seed_fraction_values)
    figures.append(("b0_vs_seed_fraction", plot_stability_boundary(
        grid, base, "B0", b0_values, "seed_fraction", seed_fraction_values,
        "Stability boundary: $B_0$ vs. seed fraction", r"$B_0$ [T]", r"seed fraction $n_{s0}/n_{p0}$",
        y_log=True,
    )))

    # 2. B0 vs Tp: isolates the Delta_T dependence unique to the exact criterion (absent
    # from the asymptotic one) -- the exact/asymptotic boundaries should visibly diverge.
    tp_values = np.linspace(500.0, 3000.0, 250)
    grid = sweep.grid("B0", b0_values, "Tp", tp_values)
    figures.append(("b0_vs_Tp", plot_stability_boundary(
        grid, base, "B0", b0_values, "Tp", tp_values,
        "Stability boundary: $B_0$ vs. primary gas temperature", r"$B_0$ [T]", r"$T_p$ [K]",
    )))

    # 3. load_resistivity vs B0: ties back to the paper's S_Omega/S_L figures -- the load
    # choice pulls Te around indirectly through the energy balance, shifting the margin.
    load_resistivity_values = np.logspace(-2, 0.3, 250)
    grid = sweep.grid("B0", b0_values, "load_resistivity", load_resistivity_values)
    figures.append(("load_resistivity_vs_B0", plot_stability_boundary(
        grid, base, "B0", b0_values, "load_resistivity", load_resistivity_values,
        r"Stability boundary: $B_0$ vs. load resistivity", r"$B_0$ [T]", r"$\eta_L$ [$\Omega\cdot$m]",
        y_log=True,
    )))

    # 4. seed_fraction vs p0: does a denser primary gas make it easier or harder to reach
    # (and sustain) the near-full-ionisation stable branch at fixed B0?
    p0_values = np.logspace(3.0, 5.0, 250)
    grid = sweep.grid("p0", p0_values, "seed_fraction", seed_fraction_values)
    figures.append(("seed_fraction_vs_p0", plot_stability_boundary(
        grid, base, "p0", p0_values, "seed_fraction", seed_fraction_values,
        "Stability boundary: inlet pressure vs. seed fraction", r"$p_0$ [Pa]", r"seed fraction $n_{s0}/n_{p0}$",
        y_log=True,
    )))

    # 5. seed_fraction vs v0: same question for flow speed (hence Mach number, which
    # drives Ohmic heating through Delta_T ~ M^2).
    v0_values = np.linspace(20.0, 500.0, 250)
    grid = sweep.grid("v0", v0_values, "seed_fraction", seed_fraction_values)
    figures.append(("seed_fraction_vs_v0", plot_stability_boundary(
        grid, base, "v0", v0_values, "seed_fraction", seed_fraction_values,
        "Stability boundary: flow speed vs. seed fraction", r"$v_0$ [m/s]", r"seed fraction $n_{s0}/n_{p0}$",
        y_log=True,
    )))

    # 6. v0 vs p0 at the default (fixed) seed fraction -- unlike 4 and 5, this pair
    # doesn't touch seed_fraction at all, so it shouldn't hit the ionisation-avalanche
    # bifurcation those two do; a useful contrast for how much smoother the boundary looks
    # when it doesn't.
    grid = sweep.grid("v0", v0_values, "p0", p0_values)
    figures.append(("v0_vs_p0", plot_stability_boundary(
        grid, base, "v0", v0_values, "p0", p0_values,
        "Stability boundary: flow speed vs. inlet pressure", r"$v_0$ [m/s]", r"$p_0$ [Pa]",
        y_log=True,
    )))

    return figures


def main() -> None:
    args = parse_show_save_args(__doc__)
    hall_solver, _ = build_default_hall_solver()
    base = OperatingPoint.default()
    sweep = EquilibriumSweep(hall_solver, base, default_seed_type().ionization_potential)

    figures = build_figures(sweep, base)
    save_and_show(figures, args)


if __name__ == "__main__":
    main()
