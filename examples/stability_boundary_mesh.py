"""
The full 3-D stability boundary surface over (seed_fraction, B0, Tp), solved on a
precomputed, matched-load (Z=sqrt(1+beta^2)) equilibrium grid and extracted with
marching cubes (magnetohydrodynamics.stability.StabilityBoundaryMesh) -- including the
Tp ~ 2000-5000 K re-entrant unstable pocket a simpler 2-surface plot couldn't represent.

Run with:
    uv run python examples/stability_boundary_mesh.py            # saves PNG (+HTML if plotly installed), no window
    uv run python examples/stability_boundary_mesh.py --show      # also opens a window
    uv run python examples/stability_boundary_mesh.py --no-save   # window only
"""
from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotting_utils import STABILITY_NOTE, fixed_params_text, parse_show_save_args, save_and_show

from magnetohydrodynamics.presets import build_default_hall_solver, default_seed_type
from magnetohydrodynamics.stability import EquilibriumSweep, OperatingPoint, StabilityBoundaryMesh

# Shared between the static matplotlib figure and the optional interactive Plotly
# export, so both describe the same grid without repeating the resolution/range
# numbers in two places.
VOLUME_SEED_FRACTION_VALUES = np.logspace(-5, -1, 45)
VOLUME_B0_VALUES = np.linspace(0.1, 5.0, 45)
VOLUME_TP_VALUES = np.logspace(0.0, np.log10(6000.0), 45)


def plot_stability_boundary_mesh(
        vertices: np.ndarray, faces: np.ndarray, vertex_power: np.ndarray,
        starts: np.ndarray, ends: np.ndarray, base: OperatingPoint,
) -> plt.Figure:
    """Solid triangulated surface (mpl_toolkits.mplot3d.art3d.Poly3DCollection) of the
    stability boundary, colored by load power density per face (log scale, averaged
    from its vertices) -- independent information, not a repaint of position. Blue
    segments mark the stable direction at a sample of faces.

    EARLIER FINDINGS NO LONGER VERIFIED, READ BEFORE TRUSTING THEM: prior versions of
    this docstring described a visible cusp (a fan of ridges near
    log10(seed_fraction) ~ -3 to -2) and a distinct re-entrant unstable pocket around
    Tp ~ 2000-5000 K, both checked carefully at the time and believed genuine. Those
    checks were all run against EquilibriumSweep.matched_load's old Picard-iteration
    solve, which (see that method's docstring) failed to converge to a self-consistent
    matched-load solution at roughly half of a typical grid's points -- so some
    fraction of the surface those checks examined was built from arbitrary,
    non-physical Picard-cycle artifacts, not real equilibria. After switching
    matched_load to the bisection fix, the rendered surface visibly changed shape (a
    single connected sheet where there used to be a visually separate lower sheet) --
    checked only by eye, once, not with the same rigor (resolution-invariance,
    log-vs-linear-Tp re-render) the original cusp/pocket claims got. Until that
    re-investigation is redone against the corrected solver, treat any fold/pocket
    structure in this plot as unconfirmed rather than assuming it's the same genuine
    physics found before."""
    face_power = vertex_power[faces].mean(axis=1)
    finite_power = face_power[np.isfinite(face_power) & (face_power > 0)]
    norm = mcolors.LogNorm(vmin=np.min(finite_power), vmax=np.max(finite_power)) if finite_power.size else mcolors.LogNorm(1.0, 10.0)
    # norm is always constructed above with explicit vmin/vmax (either branch), so vmin is
    # never actually None here -- LogNorm.vmin is just typed float | None in general.
    assert norm.vmin is not None
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

    for s, e in zip(starts, ends, strict=True):
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
        fixed_params_text(base, "B0", "seed_fraction", "Tp", "load_resistivity")
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
    far less than it did for an earlier, ~100,000-point scatter version. Plotly's
    Mesh3d renders through WebGL regardless; this repo already depends on plotly (see
    ui/app.py) for exactly this reason.

    Returns a plotly Figure, saved to a standalone .html file by main() -- openable in
    any browser, no server needed. Only called if `plotly` is importable; main() skips
    this (with a note) otherwise, since plotly isn't a required dependency of this
    script."""
    import plotly.graph_objects as go

    safe_power = np.where(np.isfinite(vertex_power) & (vertex_power > 0), vertex_power, np.nan)
    log_power = np.log10(safe_power)

    segment_x, segment_y, segment_z = [], [], []
    for s, e in zip(starts, ends, strict=True):
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


def main() -> None:
    args = parse_show_save_args(__doc__)
    hall_solver, _ = build_default_hall_solver()
    base = OperatingPoint.default()
    sweep = EquilibriumSweep(hall_solver, base, default_seed_type().ionization_potential)

    volume = sweep.volume_grid(VOLUME_SEED_FRACTION_VALUES, VOLUME_B0_VALUES, VOLUME_TP_VALUES)
    mesh = StabilityBoundaryMesh(volume, VOLUME_SEED_FRACTION_VALUES, VOLUME_B0_VALUES, VOLUME_TP_VALUES)
    mesh_vertices, mesh_faces, mesh_vertex_power = mesh.extract()
    arrow_starts, arrow_ends = mesh.stable_direction_segments(mesh_vertices, mesh_faces)
    figures = [("stability_boundary_mesh", plot_stability_boundary_mesh(
        mesh_vertices, mesh_faces, mesh_vertex_power, arrow_starts, arrow_ends, base,
    ))]

    if not args.no_save:
        # matplotlib's Poly3DCollection has no GPU acceleration either, so also emit a
        # standalone interactive HTML version via Plotly (WebGL) for actually rotating
        # it; skipped gracefully if plotly isn't installed, since it's optional here
        # (see pyproject.toml's "ui" extra).
        try:
            interactive_fig = plot_stability_boundary_mesh_interactive(
                mesh_vertices, mesh_faces, mesh_vertex_power, arrow_starts, arrow_ends,
            )
        except ImportError:
            print("(plotly not installed -- skipping the interactive HTML surface; `uv sync --extra ui` to enable it)")
        else:
            from plotting_utils import OUTPUT_DIR
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            html_path = OUTPUT_DIR / "stability_boundary_mesh.html"
            interactive_fig.write_html(html_path, include_plotlyjs="cdn")  # don't embed the plotly.js bundle
            print(f"Saved {html_path} (open in a browser to rotate it smoothly)")

    save_and_show(figures, args)


if __name__ == "__main__":
    main()
