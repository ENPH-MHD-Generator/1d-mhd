"""
3-D stability boundary surface: critical load resistivity as a function of seed
fraction and B0 -- the next most consequential *free dial* after those two (see
magnetohydrodynamics.stability.EquilibriumSweep.critical_load_resistivity_surface's
docstring for why).

Run with:
    uv run python examples/stability_load_resistivity_surface.py            # saves PNG, no window
    uv run python examples/stability_load_resistivity_surface.py --show      # also opens a window
    uv run python examples/stability_load_resistivity_surface.py --no-save   # window only
"""
from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection

from magnetohydrodynamics.presets import build_default_hall_solver, default_seed_type
from magnetohydrodynamics.stability import EquilibriumSweep, OperatingPoint

from plotting_utils import STABILITY_NOTE, draw_stable_direction_markers, fixed_params_text, parse_show_save_args, save_and_show


def plot_critical_load_resistivity_surface(
        lower: np.ndarray, upper: np.ndarray, lower_power: np.ndarray, upper_power: np.ndarray,
        seed_fraction_values: np.ndarray, b0_values: np.ndarray, base: OperatingPoint,
) -> plt.Figure:
    """3-D surface of the critical (marginal-stability) load resistivity as a function of
    seed fraction and B0. Height is log10(critical eta_L) (it spans orders of
    magnitude); color is log10(load power density S_L) delivered *at* that boundary
    point (also spans orders of magnitude) -- an independent quantity from the height,
    rather than just re-plotting the z-axis as color.

    Increasing eta_L only ever pushes the system *toward* stability here (raising Z
    drives Delta_T -> 0, which drives beta_crit -> infinity -- verified numerically: no
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
    draw_stable_direction_markers(
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
        fixed_params_text(base, "B0", "seed_fraction", "load_resistivity")
        + "\nblue arrows point toward the stable side (higher $\\eta_L$ than critical)",
        ha="center", fontsize=8.5,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.85),
    )
    return fig


def main() -> None:
    args = parse_show_save_args(__doc__)
    hall_solver, _ = build_default_hall_solver()
    base = OperatingPoint.default()
    sweep = EquilibriumSweep(hall_solver, base, default_seed_type().ionization_potential)

    seed_fraction_values = np.logspace(-5, -1, 25)
    b0_values = np.linspace(0.1, 5.0, 25)
    lower, upper, lower_power, upper_power = sweep.critical_load_resistivity_surface(seed_fraction_values, b0_values)
    figures = [("critical_load_resistivity_surface", plot_critical_load_resistivity_surface(
        lower, upper, lower_power, upper_power, seed_fraction_values, b0_values, base,
    ))]
    save_and_show(figures, args)


if __name__ == "__main__":
    main()
