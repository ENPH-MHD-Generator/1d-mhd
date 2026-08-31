"""
Two closed-form bounds on seed density from Friedberg's design constraints, holding Te
prescribed rather than solved via HallSolver's self-consistent equilibrium (see
magnetohydrodynamics.stability.SeedDensityBounds's docstring for why that distinction
matters -- it's exactly what hides these bounds from the self-consistent sweeps in
stability_2d_boundaries.py/stability_load_resistivity_surface.py):

1. SeedDensityBounds.ceiling: Friedberg's (6.27)-style maximum seed density from the
   stability criterion alone, at fixed Te and design-target beta.
2. SeedDensityBounds.min_max_window: the 3-D generalisation -- that same ceiling
   alongside a genuine minimum from a target load power density, both as surfaces over
   (Te, beta).

Run with:
    uv run python examples/seed_density_bounds.py            # saves PNGs, no window
    uv run python examples/seed_density_bounds.py --show      # also opens windows
    uv run python examples/seed_density_bounds.py --no-save   # windows only
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection
from plotting_utils import STABILITY_NOTE, parse_show_save_args, save_and_show
from scipy import constants

from magnetohydrodynamics.presets import default_gas_type, default_seed_type
from magnetohydrodynamics.stability import SeedDensityBounds

# Friedberg's own Sec. 7.1 worked-example inlet conditions (HTGR + Laval nozzle, M=1.8):
# Tp=481 K, pp=0.801 MPa, vp=735 m/s. Used as the reference n_p/v_p for
# seed_fraction_min_max_window's absolute S_C=100 MW/m^3 power-density constraint --
# checked numerically: with this repo's own (much more dilute, slower) default operating
# point as the reference instead, the region satisfying BOTH the stability ceiling and
# the power floor covers only ~4% of a typical (Te, beta) grid -- a sliver invisible at
# plot resolution -- whereas Friedberg's own S_C=100 MW/m^3 figure was derived to pair
# with HIS reference conditions, under which the valid region covers ~75% of the same
# grid.
PAPER_REFERENCE_GAS_NUMBER_DENSITY = 0.801e6 / (constants.k * 481.0)  # n_p = pp/(k Tp) [m^-3]
PAPER_REFERENCE_FLOW_SPEED = 735.0  # v_p [m/s]


def plot_seed_fraction_ceiling(
        beta_values: np.ndarray, electron_temperature_values: np.ndarray, ceiling_seed_fraction: np.ndarray,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for Te, row in zip(electron_temperature_values, ceiling_seed_fraction, strict=True):
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
        r"$T_e$ prescribed directly here, unlike every self-consistent sweep elsewhere -- see"
        "\nmagnetohydrodynamics.stability.SeedDensityBounds.ceiling's docstring for why that hides this ceiling elsewhere",
        ha="center", fontsize=8, style="italic",
    )
    return fig


def plot_seed_fraction_min_max_window(
        electron_temperature_values: np.ndarray, beta_values: np.ndarray,
        max_ns: np.ndarray, min_ne: np.ndarray, reference_gas_number_density: float, target_power_density: float,
) -> plt.Figure:
    """3-D surfaces bounding the *stable and sufficiently powerful* seed density window
    over (Te, beta): `max_ns` (the stability ceiling) and `min_ne` (the target-power
    floor). The region strictly between the two surfaces, at a given (Te, beta), is where
    BOTH of Friedberg's design constraints hold at once.

    Each surface is a single flat color (matching its legend entry), not a colormap --
    matplotlib's plot_surface default when given `cmap` instead of `color` just re-plots
    the z-axis as color a second time with no independent meaning, which looks like
    decoration rather than data once the two surfaces start crossing each other."""
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


def main() -> None:
    args = parse_show_save_args(__doc__)
    bounds = SeedDensityBounds(default_gas_type(), default_seed_type())

    beta_values = np.logspace(0.0, 2.5, 60)
    electron_temperature_values = np.array([5000.0, 15000.0, 40000.0])
    ceiling_seed_fraction = bounds.ceiling(electron_temperature_values, beta_values, PAPER_REFERENCE_GAS_NUMBER_DENSITY)

    window_te_values = np.linspace(500.0, 10000.0, 30)
    window_beta_values = np.linspace(1.0, 50.0, 30)
    max_ns, min_ne = bounds.min_max_window(
        window_te_values, window_beta_values, PAPER_REFERENCE_GAS_NUMBER_DENSITY, PAPER_REFERENCE_FLOW_SPEED,
    )

    figures = [
        ("seed_fraction_ceiling", plot_seed_fraction_ceiling(beta_values, electron_temperature_values, ceiling_seed_fraction)),
        ("seed_fraction_min_max_surface", plot_seed_fraction_min_max_window(
            window_te_values, window_beta_values, max_ns, min_ne, PAPER_REFERENCE_GAS_NUMBER_DENSITY, 100e6,
        )),
    ]
    save_and_show(figures, args)


if __name__ == "__main__":
    main()
