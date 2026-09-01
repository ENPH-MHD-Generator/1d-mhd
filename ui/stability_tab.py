"""
The "Stability" tab: interactive Velikhov-ionisation stability plots (2-D boundary
sweeps, a critical-load-resistivity surface, the full 3-D mesh, and the seed-density
window) built on magnetohydrodynamics.stability, rendered with Plotly for the same
interactive zoom/pan/hover the rest of this app already has (matplotlib's static 3-D
plots -- see examples/stability_boundary_mesh.py -- needed static "stable direction"
arrows to compensate for not being rotatable; Plotly's are, so this port drops them).

Every plot's *fixed* physical knobs come from the sidebar's current slider values (via
`base_from_ui`), not a hardcoded default operating point -- so these plots track
whatever channel configuration the rest of the app is showing, not a fixed reference
point. Two tab-local controls (not sidebar sliders, since they're specific to this
tab): a target power density (for the seed-density window) and a stability-margin
threshold (>1.0 asks for a stability *buffer* before calling something stable; <1.0
relaxes how close to marginal still counts as "stable enough") -- the threshold shifts
every boundary line/surface in this tab, not just a fixed margin==1 crossing.

Sweep axis bounds default to this module's own ranges, or -- with "Use default
bounds" unchecked -- to whatever bounds the user has set on that variable's own
sidebar slider (the same "⚙️" popover bounds, read directly out of session_state).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import constants

from magnetohydrodynamics.operating_point import OperatingPoint
from magnetohydrodynamics.presets import build_default_hall_solver, default_gas_type, default_seed_type
from magnetohydrodynamics.stability import EquilibriumSweep, SeedDensityBounds, StabilityBoundaryMesh
from magnetohydrodynamics.thermophysics.ideal_gas import IdealGas

COLOR_EXACT = "#111111"
COLOR_ASYMPTOTIC = "#2ca02c"


@dataclass(frozen=True)
class AxisSpec:
    ui_key: str  # the sidebar slider (an ui/app.py UI_PARAMS key) that controls this axis's *fixed* value elsewhere
    label: str
    log: bool
    default: tuple[float, float]
    ui_to_op: Callable[[float], float]  # sidebar slider units -> this axis's OperatingPoint-field units


# Maps each OperatingPoint field this tab can sweep to the sidebar slider that
# controls its *fixed* value elsewhere, so bounds can be derived from that slider's
# own (possibly user-customized) bounds instead of always using a fixed default range.
AXES: dict[str, AxisSpec] = {
    "B0": AxisSpec(ui_key="magnetic_field", label="B₀ [T]", log=False, default=(0.1, 5.0), ui_to_op=lambda v: v),
    # Log-scale, 1-6000 K to match examples/stability_boundary_mesh.py's own
    # VOLUME_TP_VALUES -- that's the validated reference sweep the "lower sheet"
    # investigation ([[matched-load-loop-nonconvergence]]) was run against, and most of
    # the interesting boundary structure lives below the linear 500-3000 K range this
    # used to default to (confirmed: that narrower range finds only 19 mesh faces at
    # resolution 32, vs. 1819 with this range at the same resolution).
    "Tp": AxisSpec(ui_key="inlet_gas_temperature", label="T_p [K]", log=True, default=(1.0, 6000.0), ui_to_op=lambda v: v),
    "p0": AxisSpec(ui_key="inlet_pressure_kpa", label="p₀ [Pa]", log=True, default=(1e3, 1e5), ui_to_op=lambda v: v * 1e3),
    "v0": AxisSpec(ui_key="inlet_speed", label="v₀ [m/s]", log=False, default=(20.0, 500.0), ui_to_op=lambda v: v),
    "seed_fraction": AxisSpec(ui_key="seed_fraction_log10", label="seed fraction", log=True, default=(1e-5, 1e-1), ui_to_op=lambda v: 10.0 ** v),
    "load_resistivity": AxisSpec(ui_key="load_resistivity", label="η_L [Ω·m]", log=True, default=(1e-2, 2.0), ui_to_op=lambda v: v),
}


def base_from_ui(ui_values: dict) -> OperatingPoint:
    """The sidebar's current values, expressed as the OperatingPoint every stability
    sweep in this tab holds fixed except whichever axis/axes it's varying -- so
    "fixed at..." always means "at the sidebar's current setting", not a hardcoded
    default operating point."""
    return OperatingPoint(
        B0=ui_values["magnetic_field"],
        Tp=ui_values["inlet_gas_temperature"],
        p0=ui_values["inlet_pressure_kpa"] * 1e3,
        v0=ui_values["inlet_speed"],
        seed_fraction=10.0 ** ui_values["seed_fraction_log10"],
        load_resistivity=ui_values["load_resistivity"],
    )


def axis_bounds(op_key: str, use_default_bounds: bool) -> tuple[float, float]:
    """(lo, hi) for `op_key`'s sweep axis. Reads st.session_state, so call this
    OUTSIDE any @st.cache_data function and pass the result in as a plain tuple --
    session_state isn't a valid cache key, and a cached function that read it directly
    would silently keep returning a stale result after the user changed a bound."""
    spec = AXES[op_key]
    if not use_default_bounds:
        lo_ui = st.session_state.get(f"slider_bound_lo_{spec.ui_key}")
        hi_ui = st.session_state.get(f"slider_bound_hi_{spec.ui_key}")
        if lo_ui is not None and hi_ui is not None:
            lo, hi = spec.ui_to_op(lo_ui), spec.ui_to_op(hi_ui)
            if lo < hi:
                return lo, hi
    return spec.default


def make_axis_array(op_key: str, bounds: tuple[float, float], n: int) -> np.ndarray:
    lo, hi = bounds
    return np.logspace(np.log10(lo), np.log10(hi), n) if AXES[op_key].log else np.linspace(lo, hi, n)


# --- Cached computation: every function below takes only hashable/primitive args
# (OperatingPoint is a frozen, hashable dataclass) and rebuilds its own HallSolver --
# cheap and deterministic -- rather than taking one as an argument, precisely so these
# signatures stay cacheable. Recomputes only when the sidebar's operating point, the
# sweep bounds, the margin level, or the target power actually change -- not on every
# unrelated interaction elsewhere in the app (Streamlit reruns the whole script on any
# widget interaction, so without this every slider tweak would re-run every sweep).

@st.cache_data(show_spinner="Sweeping the stability boundary...")
def compute_2d_grid(
        base: OperatingPoint, ionization_potential: float,
        x_key: str, x_bounds: tuple[float, float], y_key: str, y_bounds: tuple[float, float],
        resolution: int, fixed_mach_number: float | None,
) -> dict:
    hall_solver, _ = build_default_hall_solver()
    sweep = EquilibriumSweep(hall_solver, base, ionization_potential)
    x_values = make_axis_array(x_key, x_bounds, resolution)
    y_values = make_axis_array(y_key, y_bounds, resolution)
    grid = sweep.grid(x_key, x_values, y_key, y_values, fixed_mach_number=fixed_mach_number)
    return dict(grid=grid, x_values=x_values, y_values=y_values)


@st.cache_data(show_spinner="Solving the critical-load-resistivity surface...")
def compute_load_resistivity_surface(
        base: OperatingPoint, ionization_potential: float, level: float,
        sf_bounds: tuple[float, float], b0_bounds: tuple[float, float], resolution: int,
) -> dict:
    hall_solver, _ = build_default_hall_solver()
    sweep = EquilibriumSweep(hall_solver, base, ionization_potential)
    sf_values = make_axis_array("seed_fraction", sf_bounds, resolution)
    b0_values = make_axis_array("B0", b0_bounds, resolution)
    lower, upper, lower_power, upper_power = sweep.critical_load_resistivity_surface(sf_values, b0_values, level=level)
    return dict(sf_values=sf_values, b0_values=b0_values, lower=lower, upper=upper, lower_power=lower_power, upper_power=upper_power)


@st.cache_data(show_spinner="Building the 3-D stability mesh (a few seconds)...")
def compute_mesh(
        base: OperatingPoint, ionization_potential: float, level: float,
        sf_bounds: tuple[float, float], b0_bounds: tuple[float, float], tp_bounds: tuple[float, float],
        resolution: int, fixed_mach_number: float | None,
) -> dict:
    hall_solver, _ = build_default_hall_solver()
    sweep = EquilibriumSweep(hall_solver, base, ionization_potential)
    sf_values = make_axis_array("seed_fraction", sf_bounds, resolution)
    b0_values = make_axis_array("B0", b0_bounds, resolution)
    tp_values = make_axis_array("Tp", tp_bounds, resolution)
    volume = sweep.volume_grid(sf_values, b0_values, tp_values, fixed_mach_number=fixed_mach_number)
    mesh = StabilityBoundaryMesh(volume, sf_values, b0_values, tp_values)
    try:
        vertices, faces, vertex_power = mesh.extract(level=level)
    except (ValueError, RuntimeError):
        vertices, faces, vertex_power = np.empty((0, 3)), np.empty((0, 3), dtype=int), np.empty(0)
    return dict(vertices=vertices, faces=faces, vertex_power=vertex_power)


@st.cache_data(show_spinner="Computing the seed-density window...")
def compute_seed_density_window(base: OperatingPoint, target_power_density: float, level: float) -> dict:
    gas_type = default_gas_type()
    seed_type = default_seed_type()
    bounds = SeedDensityBounds(gas_type, seed_type)

    beta_values_1d = np.logspace(0.0, 2.5, 60)
    te_values_1d = np.array([5000.0, 15000.0, 40000.0])
    reference_n_p = base.p0 / (constants.k * base.Tp)
    ceiling = bounds.ceiling(te_values_1d, beta_values_1d, reference_n_p, level=level)

    te_values_3d = np.linspace(500.0, 10000.0, 30)
    beta_values_3d = np.linspace(1.0, 50.0, 30)
    max_ns, min_ne = bounds.min_max_window(
        te_values_3d, beta_values_3d, reference_n_p, base.v0, target_power_density=target_power_density, level=level,
    )
    return dict(
        beta_values_1d=beta_values_1d, te_values_1d=te_values_1d, ceiling=ceiling,
        te_values_3d=te_values_3d, beta_values_3d=beta_values_3d, max_ns=max_ns, min_ne=min_ne,
        reference_n_p=reference_n_p,
    )


# --- Plotly figure builders (pure -- take already-computed arrays, no solving here).

def plot_2d_boundary(
        data: dict, base: OperatingPoint, x_key: str, y_key: str,
        x_label: str, y_label: str, x_log: bool, y_log: bool, level: float, margin_cap: float = 2.0,
) -> go.Figure:
    grid, x_values, y_values = data["grid"], data["x_values"], data["y_values"]
    margin_capped = np.clip(np.nan_to_num(grid.margin, nan=0.0, posinf=margin_cap * 10, neginf=0.0), 0.0, margin_cap)

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=x_values, y=y_values, z=margin_capped,
        colorscale="RdBu", zmin=0.0, zmax=margin_cap,
        # showlines=False: Contour's fill mode defaults to also stroking a line at
        # every one of these 60 bands, which (at this size) buries the plot in thin
        # black divider lines with no meaning of their own -- the two traces below
        # already draw the one line that matters (the actual margin==level boundary).
        contours=dict(start=0.0, end=margin_cap, size=margin_cap / 60.0, showlines=False),
        colorbar=dict(title="β_crit/β"),
    ))
    fig.add_trace(go.Contour(
        x=x_values, y=y_values, z=grid.margin, showscale=False,
        contours=dict(coloring="lines", start=level, end=level, size=1.0),
        line=dict(color=COLOR_EXACT, width=2.5), name=f"exact (margin={level:g})",
    ))
    fig.add_trace(go.Contour(
        x=x_values, y=y_values, z=grid.margin_asymptotic, showscale=False,
        contours=dict(coloring="lines", start=level, end=level, size=1.0),
        line=dict(color=COLOR_ASYMPTOTIC, width=2.0, dash="dash"), name=f"asymptotic (margin={level:g})",
    ))
    fig.add_trace(go.Scatter(
        x=[getattr(base, x_key)], y=[getattr(base, y_key)], mode="markers", name="current configuration",
        marker=dict(color="red", size=13, symbol="circle", line=dict(color="white", width=1.5)),
    ))
    fig.update_xaxes(title=x_label, type="log" if x_log else "linear")
    fig.update_yaxes(title=y_label, type="log" if y_log else "linear")
    fig.update_layout(
        title=dict(text=f"Stability boundary: {x_label} vs. {y_label}", font=dict(size=13)),
        height=520, margin=dict(l=60, r=60, t=40, b=40),
        # uirevision keyed on which axes are swept -- switching axes changes what's even
        # being plotted, so THAT should reset zoom/pan, but a re-render at the same axis
        # choice (e.g. triggered by a sidebar slider elsewhere) shouldn't discard it.
        uirevision=f"{x_key}-{y_key}",
    )
    return fig


def plot_load_resistivity_surface(data: dict, base: OperatingPoint) -> go.Figure:
    log_sf = np.log10(data["sf_values"])
    lower, lower_power = data["lower"], data["lower_power"]
    log_power = np.where(np.isfinite(lower_power) & (lower_power > 0), np.log10(lower_power), np.nan)

    fig = go.Figure(data=[
        go.Surface(
            x=log_sf, y=data["b0_values"], z=np.log10(lower),
            surfacecolor=log_power, colorscale="Plasma",
            colorbar=dict(title="log10(S_L)<br>[W/m³]"),
        ),
        go.Scatter3d(
            x=[np.log10(base.seed_fraction)], y=[base.B0], z=[np.log10(base.load_resistivity)],
            mode="markers", name="current configuration",
            marker=dict(color="red", size=6, symbol="circle", line=dict(color="white", width=1)),
        ),
    ])
    fig.update_layout(
        title=dict(text="Critical load resistivity η_L(B₀, seed fraction)", font=dict(size=13)),
        scene=dict(
            xaxis_title="log10(seed fraction)", yaxis_title="B₀ [T]", zaxis_title="log10(η_L [Ω·m])",
        ),
        height=560, margin=dict(l=0, r=0, t=40, b=0),
        uirevision="surface",  # keep the camera's rotation/zoom across re-renders with new data
    )
    return fig


def plot_mesh(data: dict, base: OperatingPoint) -> go.Figure | None:
    vertices, faces, vertex_power = data["vertices"], data["faces"], data["vertex_power"]
    if len(vertices) == 0:
        return None
    safe_power = np.where(np.isfinite(vertex_power) & (vertex_power > 0), vertex_power, np.nan)
    log_power = np.log10(safe_power)

    fig = go.Figure(data=[
        go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            intensity=log_power, colorscale="Plasma", colorbar=dict(title="log10(S_L)<br>[W/m³]"),
            opacity=1.0, flatshading=False,
        ),
        go.Scatter3d(
            x=[np.log10(base.seed_fraction)], y=[base.B0], z=[base.Tp],
            mode="markers", name="current configuration",
            marker=dict(color="red", size=6, symbol="circle", line=dict(color="white", width=1)),
        ),
    ])
    fig.update_layout(
        title=dict(text=f"Stability boundary surface -- {len(faces):,} faces, drag to rotate", font=dict(size=13)),
        scene=dict(xaxis_title="log10(seed fraction)", yaxis_title="B₀ [T]", zaxis_title="T_p [K]"),
        height=600, margin=dict(l=0, r=0, t=40, b=0),
        uirevision="mesh",  # keep the camera's rotation/zoom across re-renders with new data
    )
    return fig


def plot_seed_ceiling(data: dict) -> go.Figure:
    fig = go.Figure()
    for te, row in zip(data["te_values_1d"], data["ceiling"], strict=True):
        fig.add_trace(go.Scatter(x=data["beta_values_1d"], y=row, mode="lines", name=f"T_e = {te:,.0f} K", line=dict(width=2.5)))
    fig.update_xaxes(title="Hall parameter β (design target)", type="log")
    fig.update_yaxes(title="maximum seed fraction (ceiling)", type="log")
    fig.update_layout(
        title=dict(text="Maximum allowable seed density at fixed T_e", font=dict(size=13)),
        height=420, margin=dict(l=60, r=60, t=40, b=40), hovermode="x unified",
        uirevision="ceiling",  # keep zoom/pan across re-renders with new data
    )
    return fig


def plot_seed_window(data: dict) -> go.Figure:
    log_max = np.log10(data["max_ns"] / data["reference_n_p"])
    log_min = np.log10(data["min_ne"] / data["reference_n_p"])
    te, beta = data["te_values_3d"], data["beta_values_3d"]
    fig = go.Figure(data=[
        go.Surface(x=te, y=beta, z=log_max.T, colorscale=[[0, "#ff9d33"], [1, "#ff9d33"]], showscale=False, name="max (ceiling)"),
        go.Surface(x=te, y=beta, z=log_min.T, colorscale=[[0, "#1f77b4"], [1, "#1f77b4"]], showscale=False, name="min (power floor)"),
    ])
    fig.update_layout(
        title=dict(text="Seed density window: stability ceiling (orange) vs. power floor (blue)", font=dict(size=13)),
        scene=dict(xaxis_title="T_e [K]", yaxis_title="β", zaxis_title="log10(seed fraction)"),
        height=560, margin=dict(l=0, r=0, t=40, b=0),
        uirevision="window",  # keep the camera's rotation/zoom across re-renders with new data
    )
    return fig


def render(ui_values: dict) -> None:
    base = base_from_ui(ui_values)
    ionization_potential = default_seed_type().ionization_potential

    st.caption(
        "Every plot below holds its non-swept inputs at the sidebar's current values (not a fixed default "
        "operating point) -- change a slider on the left and these update too."
    )

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 1])
    with ctrl1:
        target_power_mw = st.slider("Target power density S_C [MW/m³]", 10.0, 500.0, 100.0, step=10.0, key="stability_target_power")
    with ctrl2:
        margin_level = st.slider(
            "Stability margin threshold β_crit/β", 0.5, 1.5, 1.0, step=0.05, key="stability_margin_level",
            help="1.0 = exactly marginal stability. >1.0 asks for a safety buffer; <1.0 relaxes how close to "
                 "marginal still counts as \"stable\". Shifts the boundary line/surface in every plot below.",
        )
    with ctrl3:
        use_default_bounds = st.checkbox(
            "Use default bounds", value=True, key="stability_use_default_bounds",
            help="Checked: use this tab's own default sweep ranges. Unchecked: use each variable's own sidebar "
                 "slider bounds (the \"⚙️\" popover next to it) instead.",
        )
    with ctrl4:
        fix_mach_number = st.checkbox(
            "Fix Mach number (not velocity)", value=True, key="stability_fix_mach_number",
            help="Affects any plot below that sweeps T_p (currently the 2-D Boundary plot, if T_p is one of its "
                 "axes, and the 3-D Mesh, which always does): whether v₀ is held at the sidebar's current value "
                 "throughout the sweep, or the *Mach number* implied by the sidebar's current v₀/T_p is held fixed "
                 "instead, letting v₀ vary with T_p to match. Checked (default) is usually what you want -- T_p "
                 "sweeps here span 1-6000 K, and holding a single v₀ fixed across that makes the Mach number swing "
                 "from deep subsonic to hypersonic (M ≈ 8 at T_p = 1 K for a typical v₀), which feeds straight into "
                 "the electron-heating term (ΔT ∝ M²) and distorts where the boundary actually sits at low T_p. "
                 "Uncheck to go back to literally fixing v₀ instead.",
        )
    target_power_density = target_power_mw * 1e6
    ideal_gas = IdealGas(default_gas_type())
    fixed_mach_number = float(ideal_gas.get_mach_number(base.v0, base.Tp)) if fix_mach_number else None

    # st.tabs() looked like the natural fit here, but its active-tab selection is purely
    # frontend state that can reset to the first tab on a rerun triggered by an unrelated
    # sidebar change -- a documented, currently-unresolved Streamlit limitation
    # (https://github.com/streamlit/streamlit/issues/13341), confirmed against this exact
    # app: even key + on_change="rerun" (which the docs suggest tracks tab state) did not
    # survive an unrelated rerun when checked here. st.segmented_control is used instead
    # as a tab-bar substitute -- a normal stateful widget whose value reliably persists in
    # st.session_state across ANY rerun, the same guarantee a slider already has.
    # required=True keeps exactly one option always selected, matching st.tabs()' own
    # behavior. Trade-off: unlike st.tabs() (which computes every tab's content on every
    # rerun), only the SELECTED section's body runs below -- a deliberate bonus, not a side
    # effect: the other three sub-tabs' computations (see [[stability-performance-profiling]]
    # for how much some of them cost) no longer run at all while hidden.
    SUB_TAB_LABELS = ["📐 2-D Boundary", "🌐 Load-Resistivity Surface", "🧊 3-D Mesh", "🪟 Seed-Density Window"]
    active_sub_tab = st.segmented_control(
        "Stability view", SUB_TAB_LABELS, default=SUB_TAB_LABELS[0], required=True,
        key="stability_sub_tab", label_visibility="collapsed",
    )

    if active_sub_tab == "📐 2-D Boundary":
        axis_options = list(AXES)

        def axis_label(key: str) -> str:
            return AXES[key].label

        c1, c2 = st.columns(2)
        with c1:
            x_key = st.selectbox("X axis", axis_options, index=axis_options.index("B0"), format_func=axis_label, key="stability_2d_x")
        with c2:
            y_key = st.selectbox("Y axis", axis_options, index=axis_options.index("seed_fraction"), format_func=axis_label, key="stability_2d_y")

        if x_key == y_key:
            st.warning("Pick two different axes.")
        else:
            x_bounds = axis_bounds(x_key, use_default_bounds)
            y_bounds = axis_bounds(y_key, use_default_bounds)
            data = compute_2d_grid(base, ionization_potential, x_key, x_bounds, y_key, y_bounds, resolution=90, fixed_mach_number=fixed_mach_number)
            st.plotly_chart(
                plot_2d_boundary(
                    data, base, x_key, y_key, AXES[x_key].label, AXES[y_key].label, AXES[x_key].log, AXES[y_key].log, margin_level,
                ),
                width="stretch", key="plot_stability_2d",
            )

    if active_sub_tab == "🌐 Load-Resistivity Surface":
        sf_bounds = axis_bounds("seed_fraction", use_default_bounds)
        b0_bounds = axis_bounds("B0", use_default_bounds)
        data = compute_load_resistivity_surface(base, ionization_potential, margin_level, sf_bounds, b0_bounds, resolution=22)
        st.plotly_chart(plot_load_resistivity_surface(data, base), width="stretch", key="plot_stability_surface")
        st.caption("Everything above the surface has higher load resistivity than critical -- more stable.")

    if active_sub_tab == "🧊 3-D Mesh":
        sf_bounds = axis_bounds("seed_fraction", use_default_bounds)
        b0_bounds = axis_bounds("B0", use_default_bounds)
        tp_bounds = axis_bounds("Tp", use_default_bounds)
        data = compute_mesh(base, ionization_potential, margin_level, sf_bounds, b0_bounds, tp_bounds, resolution=45, fixed_mach_number=fixed_mach_number)
        fig = plot_mesh(data, base)
        if fig is None:
            st.info(
                f"No boundary found at margin = {margin_level:g} within the swept (seed fraction, B₀, T_p) "
                "range -- try widening the bounds or a different margin threshold."
            )
        else:
            st.plotly_chart(fig, width="stretch", key="plot_stability_mesh")
        st.caption(
            "Solved on a matched-load (Z=√(1+β²)) equilibrium grid -- load resistivity itself isn't swept here. "
            "Rendered directly with Plotly's own rotation instead of the static \"stable direction\" arrows the "
            "matplotlib version (examples/stability_boundary_mesh.py) needs."
        )

    if active_sub_tab == "🪟 Seed-Density Window":
        data = compute_seed_density_window(base, target_power_density, margin_level)
        wc1, wc2 = st.columns(2)
        with wc1:
            st.plotly_chart(plot_seed_ceiling(data), width="stretch", key="plot_stability_ceiling")
        with wc2:
            st.plotly_chart(plot_seed_window(data), width="stretch", key="plot_stability_window")
        st.caption(
            "Both hold T_e prescribed rather than solving HallSolver's self-consistent equilibrium for it -- see "
            "SeedDensityBounds' docstring for why. The power floor uses the sidebar's own inlet density/speed as "
            "its reference, so (unlike Friedberg's own reference furnace conditions) the valid window's size is "
            "sensitive to how dense/fast the current sidebar configuration is."
        )
