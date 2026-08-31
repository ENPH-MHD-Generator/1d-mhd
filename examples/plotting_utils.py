"""
Plotting helpers shared across the stability-boundary example scripts
(stability_2d_boundaries.py, stability_load_resistivity_surface.py,
stability_boundary_mesh.py, seed_density_bounds.py) -- purely presentational: argument
parsing/figure saving boilerplate, caption text, and small mpl drawing routines. No
physics/math lives here; see magnetohydrodynamics.stability for that.
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

from magnetohydrodynamics.stability import OperatingPoint

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Appended to every stability-boundary plot title as a brief reminder of which way the
# margin ratio reads.
STABILITY_NOTE = r"(stable: $\beta_{crit}/\beta \geq 1$)"

# Display metadata for each physical knob: (symbol, unit, format spec), used both for
# axis labels a sweep doesn't already set explicitly and for the "fixed at..."
# annotation on every plot.
PHYSICAL_META = {
    "B0": (r"$B_0$", "T", "{:.3g}"),
    "Tp": (r"$T_p$", "K", "{:.0f}"),
    "p0": (r"$p_0$", "Pa", "{:.3g}"),
    "v0": (r"$v_0$", "m/s", "{:.3g}"),
    "seed_fraction": ("seed fraction", "", "{:.2e}"),
    "load_resistivity": (r"$\eta_L$", r"$\Omega\cdot$m", "{:.3g}"),
}


def fixed_params_text(base: OperatingPoint, *varied_keys: str) -> str:
    """'fixed: ...' string listing every physical knob in `base` NOT in `varied_keys`,
    for annotating a plot with what's being held constant."""
    parts = []
    for key, value in dataclasses.asdict(base).items():
        if key in varied_keys:
            continue
        symbol, unit, fmt = PHYSICAL_META[key]
        formatted = fmt.format(value) + (f" {unit}" if unit else "")
        parts.append(f"{symbol} = {formatted}")
    return "fixed:  " + ",   ".join(parts)


def draw_boundary_contour(ax, x_values, y_values, margin, color="black", halo_color="white", halo_extra=1.0, linewidths=2.0, **contour_kwargs):
    """A contour line at margin == 1 (marginal stability), with a thin halo so it stays
    visible against the colormap. Kept thin deliberately: the exact and asymptotic
    boundaries often sit almost on top of each other, and a thick halo around one (an
    earlier version used +2.0) paints over the other's dashes in its gaps wherever the
    two nearly coincide. Distinct colors (not just dash pattern) are what actually keeps
    both readable when they overlap."""
    cs = ax.contour(x_values, y_values, margin, levels=[1.0], colors=color, linewidths=linewidths, **contour_kwargs)
    cs.set(path_effects=[pe.withStroke(linewidth=linewidths + halo_extra, foreground=halo_color)])
    return cs


def draw_stable_direction_markers(ax, xs, ys, zs, length: float, color: str = "steelblue") -> None:
    """Short vertical line segments (with a triangle marker on top) from each finite
    (x, y, z) sample point straight up by `length`, indicating the stable side on a 3-D
    surface plot. Plotted one at a time with plain `ax.plot`/`ax.scatter` rather than
    `Axes3D.quiver` -- quiver's arrows came out wildly mis-scaled (stretching across the
    whole plot) whenever the height axis's raw numeric range was very different from the
    other two axes' (e.g. Tp in the hundreds/thousands vs. log10-scaled axes of order
    1-10); this avoids relying on quiver's internal scaling entirely."""
    import numpy as np

    xs, ys, zs = np.ravel(xs), np.ravel(ys), np.ravel(zs)
    finite = np.isfinite(zs)
    for x, y, z in zip(xs[finite], ys[finite], zs[finite], strict=True):
        ax.plot([x, x], [y, y], [z, z + length], color=color, linewidth=1.5)
    ax.scatter(xs[finite], ys[finite], zs[finite] + length, marker="^", color=color, s=12, depthshade=False)


def parse_show_save_args(description: str) -> argparse.Namespace:
    """Shared --show/--no-save argparse setup for every example script's main()."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show", action="store_true", help="Display each figure interactively.")
    parser.add_argument("--no-save", action="store_true", help="Skip writing PNGs to examples/output/.")
    return parser.parse_args()


def save_and_show(figures: list[tuple[str, plt.Figure]], args: argparse.Namespace) -> None:
    """Save each (name, Figure) to OUTPUT_DIR/name.png (unless --no-save) and open
    interactive windows (if --show) -- the common tail end of every example script's
    main()."""
    if not args.no_save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for name, fig in figures:
            path = OUTPUT_DIR / f"{name}.png"
            fig.savefig(path, dpi=150)
            print(f"Saved {path}")

    if args.show:
        plt.show()
