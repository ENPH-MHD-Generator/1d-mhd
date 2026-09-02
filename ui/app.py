"""
Interactive UI for the 1-D linear Hall MHD generator.

Lets you tweak channel geometry / inlet conditions with sliders and see the
resulting axial profiles and performance figures update live, inspect the
full inlet plasma state, and optimize one chosen parameter -- or several at
once -- (within bounds you set) to maximize a chosen objective (e.g. load
power). Plots are interactive (Plotly): zoom, pan, and hover for exact values.

Run with:
    uv run --extra ui streamlit run ui/app.py

This is intentionally separate from main.py -- it depends on `streamlit`,
which lives in the optional "ui" dependency group so `uv sync` (no extras)
stays lightweight.
"""
import re
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

# This project isn't installed as a package (`package = false` in
# pyproject.toml), and `streamlit run ui/app.py` puts ui/ -- not the repo
# root -- on sys.path. Add the repo root explicitly so `magnetohydrodynamics`
# is importable regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tomli_w
from scipy.optimize import differential_evolution, minimize_scalar

from magnetohydrodynamics.analysis import summarize_performance, summarize_tapered_performance
from magnetohydrodynamics.presets import build_default_hall_solver, default_seed_type
from magnetohydrodynamics.solver.tapered_hall_solver import LinearTaper, TaperedHallSolver
from magnetohydrodynamics.stability import FriedbergAsymptoticCriterion, FriedbergCriterion
from magnetohydrodynamics.thermophysics.ideal_gas import IdealGas

import stability_tab

st.set_page_config(page_title="Hall MHD Generator", layout="wide")

# Scoped to the sidebar only (every rule prefixed) -- tighter vertical rhythm so the whole
# control panel fits on one screen without scrolling. The main content area is untouched.
# Deliberately conservative: blanket-overriding [data-testid="stVerticalBlock"]'s gap and
# [data-testid="stElementContainer"]'s margin (both match at *every* nesting level, including
# inside st.columns()) collapsed the Save/Load rows into an overlapping mess. Only single,
# specific containers are touched here.
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    [data-testid="stSidebar"] h2 { font-size: 1.05rem; margin: 0.6rem 0 0.2rem 0; padding: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Colors (matplotlib's "tab10" hex values, so this still looks like the old plots).
COLOR_BLUE = "#1f77b4"
COLOR_ORANGE = "#ff7f0e"
COLOR_GREEN = "#2ca02c"
COLOR_PURPLE = "#9467bd"
COLOR_RED = "#d62728"

# Parameters the UI exposes, keyed by the name used in both `st.session_state`
# and `build_march_params`. (full label, default min, default max, default value, step)
UI_PARAMS = {
    "channel_side_mm": ("Channel side (square cross-section) [mm]", 10.0, 200.0, 48.0, 1.0),
    "channel_length": ("Channel length [m]", 0.02, 1.0, 0.2, 0.01),
    "inlet_gas_temperature": ("Inlet gas temperature [K]", 300.0, 5000.0, 2000.0, 50.0),
    "inlet_pressure_kpa": ("Inlet pressure [kPa]", 1.0, 500.0, 10.01, 0.5),
    "inlet_speed": ("Inlet speed [m/s]", 1.0, 1000.0, 150.115, 1.0),
    "magnetic_field": ("Magnetic field [T]", 0.0, 3.0, 0.5, 0.05),
    "seed_fraction_log10": ("log10(seed fraction)", -5.0, -1.0, float(np.log10(6.18e-3)), 0.05),
    "load_resistivity": ("Load resistivity [Ω·m]", 0.001, 2.0, 0.1171875, 0.001),
}
NUM_SLICES_PARAMS = ("Axial slices", 20, 400, 200, 10)

# Short labels for the sidebar's inline (label-left-of-slider) layout; the full labels above
# remain in use everywhere else (Optimize tabs, results tables, plot axes).
SIDEBAR_SHORT_LABELS = {
    "channel_side_mm": "Side [mm]",
    "channel_length": "Length [m]",
    "num_slices": "Slices",
    "inlet_gas_temperature": "Inlet T [K]",
    "inlet_pressure_kpa": "Inlet p [kPa]",
    "inlet_speed": "Inlet v [m/s]",
    "magnetic_field": "B [T]",
    "seed_fraction_log10": "log10(seed)",
    "load_resistivity": "R_load [Ω·m]",
}

OBJECTIVES = {
    "Load power (PL)": lambda out, params, perf: perf["PL"],
    "Electrical efficiency": lambda out, params, perf: perf["eta_electrical"],
    "Enthalpy extraction ratio": lambda out, params, perf: perf["enthalpy_extraction_ratio"],
}

# The parameters the multi-parameter optimizer searches over: everything actually
# controllable in practice (seed fraction, load resistivity, and the inlet conditions).
# Channel geometry and magnetic field are fixed by the use case, so they're excluded --
# the multi-optimizer holds them at their current slider values.
MULTI_OPTIMIZABLE_KEYS = [
    "inlet_gas_temperature", "inlet_pressure_kpa", "inlet_speed", "seed_fraction_log10", "load_resistivity",
]

# Everything a "configuration" is: the 8 UI_PARAMS values, the slice count, and the
# tapered-channel setting (see the Geometry sidebar section's dialog) -- CONFIG_KEYS
# grew by the last two after saved configs already existed, so _config_default below
# exists specifically to fill them in when loading an older file that predates them.
CONFIG_KEYS = [*UI_PARAMS, "num_slices", "tapered_channel_enabled", "tapered_area_ratio"]
CONFIG_EXTRA_DEFAULTS = {"tapered_channel_enabled": False, "tapered_area_ratio": 1.0}


def _config_default(key: str):
    """The value a CONFIG_KEYS entry defaults to when a saved config predates it --
    UI_PARAMS' own tuple for keys registered there, NUM_SLICES_PARAMS for the slice
    count, CONFIG_EXTRA_DEFAULTS for everything else CONFIG_KEYS has grown to include."""
    if key in UI_PARAMS:
        return UI_PARAMS[key][3]
    if key == "num_slices":
        return NUM_SLICES_PARAMS[3]
    return CONFIG_EXTRA_DEFAULTS[key]


SAVE_DIR = Path(__file__).resolve().parent / "saved_configs"
SAVE_DIR.mkdir(exist_ok=True)


def list_saved_configs() -> list[Path]:
    return sorted(SAVE_DIR.glob("*.toml"))


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_").lower()
    return slug


def save_config(values: dict, name: str) -> Path:
    """Save a configuration as TOML. A given name overwrites its own file on re-save
    (a "named slot"); leaving it blank always creates a fresh timestamped file."""
    slug = slugify(name)
    if slug:
        path = SAVE_DIR / f"{slug}.toml"
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = SAVE_DIR / f"config_{stamp}.toml"
        counter = 1
        while path.exists():  # only matters if two unnamed saves land in the same second
            path = SAVE_DIR / f"config_{stamp}_{counter}.toml"
            counter += 1

    data = {
        "name": name or path.stem,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "values": {key: values[key] for key in CONFIG_KEYS},
    }
    with open(path, "wb") as f:
        tomli_w.dump(data, f)
    return path


def load_config(path: Path) -> tuple[dict, str]:
    """Returns (values, display_name). Fills in any CONFIG_KEYS entry missing from the
    file with today's default (rather than leaving it out) -- a config saved before
    CONFIG_KEYS grew a new key (e.g. the tapered-channel setting) should load as if
    that setting were at its default, not silently keep whatever value happens to
    currently be in session_state from an unrelated prior action."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    values = data["values"]
    for key in CONFIG_KEYS:
        values.setdefault(key, _config_default(key))
    return values, data.get("name", path.stem)


def pending_apply_for_values(values: dict) -> dict:
    """Build a pending_apply dict that restores `values` exactly, widening any
    user-customized slider bounds that would otherwise clamp a loaded value away from
    what was saved."""
    updates = dict(values)
    for key, value in values.items():
        lo_key, hi_key = f"slider_bound_lo_{key}", f"slider_bound_hi_{key}"
        current_lo = st.session_state.get(lo_key)
        current_hi = st.session_state.get(hi_key)
        if current_lo is not None and value < current_lo:
            updates[lo_key] = value
        if current_hi is not None and value > current_hi:
            updates[hi_key] = value
    return updates


def bounds_label(key: str) -> str:
    """Label used where the *units of the bound itself* matter (log10 vs. linear)."""
    return UI_PARAMS[key][0]


def result_label(key: str) -> str:
    """Label used for a resulting/optimal *value* (already converted out of log-space)."""
    return "Seed fraction" if key == "seed_fraction_log10" else UI_PARAMS[key][0]


def result_value(key: str, raw_value: float) -> str:
    if key == "seed_fraction_log10":
        return f"{10.0 ** raw_value:.3e}"
    return f"{raw_value:.4g}"


# differential_evolution's population is popsize * ndim; it runs up to maxiter generations
# (often fewer, via early convergence). More evaluations -> a more thorough search of the
# 5-parameter space, at the cost of time -- see the live estimate in the Multi-Optimize tab.
SEARCH_THOROUGHNESS_PRESETS = {
    "Quick": dict(maxiter=15, popsize=8),
    "Balanced": dict(maxiter=40, popsize=15),
    "Thorough": dict(maxiter=80, popsize=25),
}


def build_march_params(ui: dict) -> dict:
    """Map UI-level values to HallSolver.march()'s keyword arguments."""
    area = (ui["channel_side_mm"] / 1000.0) ** 2
    length = ui["channel_length"]
    return dict(
        num_slices=int(ui["num_slices"]),
        length=length,
        area=area,
        inlet_speed=ui["inlet_speed"],
        inlet_pressure=ui["inlet_pressure_kpa"] * 1e3,
        inlet_gas_temperature=ui["inlet_gas_temperature"],
        magnetic_field=ui["magnetic_field"],
        load_resistance=ui["load_resistivity"] * length / area,
        inlet_seed_fraction=10.0 ** ui["seed_fraction_log10"],
    )


def solve(hall_solver, gas_type, ui: dict):
    """Marches the channel and summarizes performance -- constant-area
    (HallSolver.march) by default, or a linearly-tapered channel
    (TaperedHallSolver.march) when `ui["tapered_channel_enabled"]` is set (see the
    "Tapered channel..." dialog in the Geometry sidebar section). `.get(...)` with a
    default rather than a bare `ui[...]` lookup, since `ui` can be a saved-config dict
    (Compare tab) that predates this feature and has neither key."""
    params = build_march_params(ui)
    if ui.get("tapered_channel_enabled", False):
        taper = LinearTaper.from_exit_area(
            inlet_area=params["area"],
            exit_area=params["area"] * ui.get("tapered_area_ratio", 1.0),
            length=params["length"],
        )
        tapered_solver = TaperedHallSolver(hall_solver)
        channel = tapered_solver.march(
            num_slices=params["num_slices"], length=params["length"], taper=taper,
            inlet_speed=params["inlet_speed"], inlet_pressure=params["inlet_pressure"],
            inlet_gas_temperature=params["inlet_gas_temperature"], magnetic_field=params["magnetic_field"],
            load_resistance=params["load_resistance"], inlet_seed_fraction=params["inlet_seed_fraction"],
        )
        out = channel.to_dict()
        perf = summarize_tapered_performance(
            out, cp=gas_type.molar_heat_capacity, m_p=gas_type.particle_mass, gamma=gas_type.heat_capacity_ratio,
        )
    else:
        channel = hall_solver.march(**params)
        out = channel.to_dict()
        perf = summarize_performance(
            out, A=params["area"],
            cp=gas_type.molar_heat_capacity,
            m_p=gas_type.particle_mass,
            gamma=gas_type.heat_capacity_ratio,
        )
    return params, channel, out, perf


def regime_note(value: float, low: float, high: float, low_text: str, mid_text: str, high_text: str) -> str:
    if value < low:
        return low_text
    if value > high:
        return high_text
    return mid_text


def bounded_slider(key: str, label: str, default_lo: float, default_hi: float, default_value: float, step) -> None:
    """A slider whose min/max bounds are user-adjustable via a small popover next to it,
    with a reset-to-default for the bounds themselves."""
    # "slider_bound_" (not "bound_") to avoid colliding with the Optimize tabs' own
    # bound_lo_{key}/bound_hi_{key} search-bound widget keys, which share the same key namespace.
    lo_key, hi_key = f"slider_bound_lo_{key}", f"slider_bound_hi_{key}"
    st.session_state.setdefault(lo_key, default_lo)
    st.session_state.setdefault(hi_key, default_hi)
    st.session_state.setdefault(key, default_value)

    lo, hi = st.session_state[lo_key], st.session_state[hi_key]
    bounds_invalid = lo >= hi
    if bounds_invalid:
        lo, hi = default_lo, default_hi

    # Keep the current value inside the (possibly just-narrowed) bounds.
    st.session_state[key] = min(max(st.session_state[key], lo), hi)

    col_label, col_slider, col_gear = st.columns([3, 8, 1], vertical_alignment="center")
    with col_label:
        st.caption(SIDEBAR_SHORT_LABELS.get(key, label))
    with col_slider:
        st.slider(label, lo, hi, key=key, step=step, label_visibility="collapsed")
    with col_gear:
        # Empty label -> just the popover's own built-in chevron, no emoji/text competing
        # for space in this narrow column.
        with st.popover("", width="stretch", help=f"Adjust bounds for {label}"):
            st.markdown(f"**{label}**  \nAdjust slider bounds")
            # Private widget keys with an explicit `value=`, copied into the persistent
            # lo_key/hi_key by hand below -- NOT `key=lo_key`/`key=hi_key` directly. A
            # popover's contents, like a dialog's (see tapered_channel_dialog's own
            # comment), aren't part of the element tree while it's collapsed; a widget
            # bound straight to a persistent key showed 0.0 the first time a popover was
            # opened (the widget's own `value=0.0` default winning on that first mount)
            # rather than the already-correct default_lo/default_hi in session_state.
            new_lo = st.number_input("Min", value=st.session_state[lo_key], step=step, key=f"_bound_lo_widget_{key}")
            new_hi = st.number_input("Max", value=st.session_state[hi_key], step=step, key=f"_bound_hi_widget_{key}")
            st.session_state[lo_key] = new_lo
            st.session_state[hi_key] = new_hi
            if bounds_invalid:
                st.caption("⚠️ Min must be less than max -- using defaults until fixed.")
            if st.button("Reset to default", key=f"reset_bounds_{key}", width="stretch"):
                st.session_state["pending_apply"] = {lo_key: default_lo, hi_key: default_hi}
                st.rerun()


@st.dialog("Tapered Channel", width="small")
def tapered_channel_dialog() -> None:
    """A linearly-tapered channel (one wall pair fixed, the other diverging/converging
    at a constant angle -- see Derivation.md's "Variable-Area (Tapered) Channel"
    section and magnetohydrodynamics.solver.tapered_hall_solver) lives behind this
    dialog rather than its own sidebar sliders, to keep the sidebar itself short: it's
    an occasional, advanced setting (off by default), not something tweaked every run
    the way the inlet conditions are."""
    st.caption(
        "Diverge (or converge) the channel linearly from inlet to outlet, instead of "
        "holding its cross-section constant -- buys back distance from Rayleigh-flow "
        "choking for a supersonic inlet (see the Profiles tab's Mach number plot), at "
        "the cost of an approximation (below) that only holds for gentle angles."
    )
    # The checkbox/slider below are bound to PRIVATE widget keys, not directly to the
    # persistent "tapered_channel_enabled"/"tapered_area_ratio" settings -- st.dialog
    # content only exists in the app's element tree while the dialog is open, and
    # Streamlit resets (not preserves) a widget's value whenever it's removed and later
    # re-added (see the "Widget behavior" docs st.segmented_control's own help text
    # points to). Binding the widget's own `key` straight to the persistent setting
    # silently reset it to `value=` every time this dialog was reopened -- and on any
    # OTHER rerun where it wasn't open at all, since the widget (and, it turns out, the
    # key it owned) didn't exist on that render either. Copying the widget's value into
    # the persistent key by hand, every render, sidesteps that: the persistent key is
    # then a plain session_state entry, never bound to this widget's own lifecycle, so
    # it survives the dialog closing/reopening and every other rerun.
    enabled = st.checkbox(
        "Enable tapered channel", value=st.session_state["tapered_channel_enabled"],
        key="_tapered_channel_enabled_widget",
    )
    st.session_state["tapered_channel_enabled"] = enabled
    area_ratio = st.slider(
        "Area ratio (outlet / inlet)", 0.2, 4.0, st.session_state["tapered_area_ratio"], step=0.05,
        key="_tapered_area_ratio_widget", disabled=not enabled,
        help="> 1 diverges, < 1 converges, = 1 is the constant-area channel above.",
    )
    st.session_state["tapered_area_ratio"] = area_ratio

    inlet_area = (st.session_state["channel_side_mm"] / 1000.0) ** 2
    length = st.session_state["channel_length"]
    taper = LinearTaper.from_exit_area(inlet_area, inlet_area * area_ratio, length)

    max_half_angle_deg = 15.0  # matches TaperedHallSolver's own default guard
    half_angle = taper.half_angle_deg
    if abs(half_angle) <= 0.6 * max_half_angle_deg:
        color = "green"
    elif abs(half_angle) <= max_half_angle_deg:
        color = "orange"
    else:
        color = "red"
    st.metric(
        "Divergence half-angle", f":{color}[{half_angle:+.2f}°]",
        help="Ohm's law's dropped v_py/v_pz terms (Derivation.md) are only a "
             "defensible approximation for GENTLE divergence. Green: comfortably "
             f"under the {max_half_angle_deg:g}° guard. Orange: approaching it. "
             "Red: past it -- the solver will warn (or, in strict mode, refuse to run).",
    )
    if not enabled:
        st.caption("Currently disabled -- the channel above uses a constant area.")

    if enabled:
        st.divider()
        st.caption(
            "Live preview, at the current sidebar inlet conditions -- constant area (above) vs. this taper. A "
            "taper's LOCAL steepness (dA/dx) is the ratio spread over the WHOLE channel length, so the same ratio "
            "does much less over the short distance where the interaction (or choking, for a supersonic inlet) "
            "actually happens if the channel is long compared to that distance -- this is where that shows up."
        )
        # Two full marches, right here, rather than only reporting the taper's geometry: the
        # geometry alone (half-angle, area ratio) doesn't say whether it's steep enough to matter
        # over the distance the flow actually covers before choking, which is exactly what was
        # confusing about "an area ratio of 2 does ~nothing" for a channel far longer than the
        # interaction length. `ui_snapshot` reads every other UI_PARAMS key straight from
        # session_state (set on the initial full script render, same as e.g. `inlet_area` above --
        # this dialog runs before those widgets re-render this rerun, but their session_state
        # entries already exist from the run that rendered the button that opened it).
        ui_snapshot = {key: st.session_state[key] for key in UI_PARAMS}
        ui_snapshot["num_slices"] = st.session_state["num_slices"]
        _, flat_channel, flat_out, flat_perf = solve(hall_solver, gas_type, {**ui_snapshot, "tapered_channel_enabled": False})
        _, tapered_channel, tapered_out, tapered_perf = solve(
            hall_solver, gas_type, {**ui_snapshot, "tapered_channel_enabled": True, "tapered_area_ratio": area_ratio},
        )
        preview_rows = [
            ("Choked?", str(flat_channel.choked), str(tapered_channel.choked)),
            ("Distance traveled [mm]", f"{flat_out['x'][-1] * 1000:.3f}", f"{tapered_out['x'][-1] * 1000:.3f}"),
            ("Load power P_L [W]", f"{flat_perf['PL']:.1f}", f"{tapered_perf['PL']:.1f}"),
            (
                "Enthalpy extraction [%]", f"{flat_perf['enthalpy_extraction_ratio'] * 100:.3f}",
                f"{tapered_perf['enthalpy_extraction_ratio'] * 100:.3f}",
            ),
        ]
        st.dataframe(
            pd.DataFrame(preview_rows, columns=["", "Constant area", f"{taper.half_angle_deg:+.2f}° taper"]),
            hide_index=True, width="stretch",
        )
        flat_x_last = flat_out["x"][-1]
        distance_change = abs(tapered_out["x"][-1] - flat_x_last) / flat_x_last if flat_x_last > 0 else 0.0
        if distance_change < 0.02:
            st.caption(
                f"⚠️ Distance traveled changed by only {distance_change * 100:.1f}% -- with the configured channel "
                f"length ({length:.3g} m), this taper's local steepness (half-angle {taper.half_angle_deg:+.2f}°) "
                "is too gentle to matter over the short distance where the flow actually chokes/interacts. A "
                "shorter channel length, or a larger area ratio, concentrates the same taper over less distance."
            )

    if st.button("Done", width="stretch"):
        st.rerun()


# --- Plotly figure helpers. Every figure gets a shared, compact layout; all comparison
# figures share these conventions: each config gets one fixed color (shared across all its
# lines), each physical quantity gets a fixed dash style (shared across configs) -- so e.g.
# every config's T_p is solid and every config's T_e is dashed, but colored per config. All
# configs plot on the *same* axes (not renormalized), so magnitudes stay comparable.
CONFIG_COLORS = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN]


def _config_color(i: int) -> str:
    return CONFIG_COLORS[i % len(CONFIG_COLORS)]


def _base_layout(fig: go.Figure, title: str, xaxis_title: str = "x [m]", height: int = 360) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis_title=xaxis_title,
        height=height,
        margin=dict(l=60, r=60, t=40, b=40),
        hovermode="x unified",
        legend=dict(font=dict(size=10)),
    )
    return fig


def plot_single_axis(x, series: list[tuple[str, "np.ndarray", str, str]], title: str, yaxis_title: str) -> go.Figure:
    """series: list of (name, y, color, dash)."""
    fig = go.Figure()
    for name, y, color, dash in series:
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color=color, dash=dash, width=2.5)))
    fig.update_yaxes(title_text=yaxis_title)
    return _base_layout(fig, title)


def plot_twin_axis(
        x, left_series: list[tuple[str, "np.ndarray", str, str]], right_series: list[tuple[str, "np.ndarray", str, str]],
        title: str, left_title: str, right_title: str, left_log: bool = False, right_range: tuple | None = None,
) -> go.Figure:
    fig = go.Figure()
    for name, y, color, dash in left_series:
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color=color, dash=dash, width=2.5)))
    for name, y, color, dash in right_series:
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color=color, dash=dash, width=2.5), yaxis="y2"))
    fig.update_layout(
        yaxis=dict(title=left_title, type="log" if left_log else "linear"),
        yaxis2=dict(title=right_title, overlaying="y", side="right", range=list(right_range) if right_range else None),
    )
    return _base_layout(fig, title)


def profile_temperature(out: dict) -> go.Figure:
    return plot_single_axis(
        out["x"], [("T_p (primary gas)", out["Tp"], COLOR_BLUE, "solid"), ("T_e (electron)", out["Te"], COLOR_ORANGE, "dash")],
        "Electron vs. Primary Gas Temperature", "Temperature [K]",
    )


def profile_primary_gas(out: dict) -> go.Figure:
    return plot_twin_axis(
        out["x"], [("n_p", out["np"], COLOR_BLUE, "solid")], [("v_p", out["u"], COLOR_ORANGE, "dash")],
        "Primary Gas: n_p, v_p", "n_p [m⁻³]", "v_p [m/s]",
    )


def profile_mach_number(out: dict, ideal_gas: IdealGas) -> go.Figure:
    """Gas-dynamic Mach number M = u/c_s along the channel -- the same quantity the
    Inlet Summary tab's "Mach number M" row and the choked-flow warning (M=1, the
    Rayleigh-flow limit HallSolver.march stops at) already use."""
    mach = ideal_gas.get_mach_number(out["u"], out["Tp"])
    fig = plot_single_axis(out["x"], [("M", mach, COLOR_RED, "solid")], "Mach Number Along the Channel", "M [-]")
    fig.add_hline(y=1.0, line=dict(color="black", width=1))
    return fig


def profile_channel_area(out: dict) -> go.Figure:
    """Channel cross-sectional area along x -- only meaningful (and only shown) when
    the "Tapered channel..." dialog's setting is enabled; `out["area"]` only exists on
    a TaperedChannel.to_dict(), not the constant-area Channel's."""
    fig = plot_single_axis(
        out["x"], [("A", out["area"] * 1e4, COLOR_GREEN, "solid")], "Channel Area Along the Channel", "Area [cm²]",
    )
    return fig


def profile_density(out: dict, seed_number_density) -> go.Figure:
    ionization_fraction = out["ne"] / seed_number_density
    fig = plot_twin_axis(
        out["x"],
        [("n_s (seed)", seed_number_density, COLOR_BLUE, "solid"), ("n_e (electrons)", out["ne"], COLOR_ORANGE, "dash")],
        [("n_e/n_s (ionization fraction)", ionization_fraction, COLOR_GREEN, "dot")],
        "Seed vs. Electron Density (how much of the seed is ionized)",
        "Number density [m⁻³]", "Ionization fraction",
        left_log=True, right_range=(0.0, 1.05),
    )
    return fig


def profile_power(out: dict) -> go.Figure:
    return plot_single_axis(
        out["x"], [("S_Ω (ohmic)", out["S_ohm"], COLOR_BLUE, "solid"), ("S_L (load)", out["S_load"], COLOR_ORANGE, "dash")],
        "Ohmic Heating vs Load Power", "Power density [W/m³]",
    )


def profile_pressure(out: dict) -> go.Figure:
    return plot_single_axis(out["x"], [("p_p", out["p"], COLOR_GREEN, "solid")], "Primary Gas Pressure", "p_p [Pa]")


def profile_hall_parameter(out: dict) -> go.Figure:
    return plot_single_axis(out["x"], [("β", out["beta"], COLOR_PURPLE, "solid")], "Hall Parameter Along the Channel", "β [-]")


def compute_stability_margin(out: dict, ionization_potential: float) -> tuple[np.ndarray, np.ndarray]:
    """beta_crit/beta at every slice the march visited, both the exact (5.13) and
    high-ionisation asymptotic (6.23) criteria -- shared by profile_stability_margin
    (the full profile plot) and the Performance section's summary metric (its
    minimum along the channel)."""
    exact = FriedbergCriterion()
    asymptotic = FriedbergAsymptoticCriterion()
    margin = exact.stability_margin(out["beta"], out["Te"], out["Tp"], ionization_potential, out["f_I"])
    margin_asymptotic = asymptotic.stability_margin(out["beta"], out["Te"], out["Tp"], ionization_potential, out["f_I"])
    return margin, margin_asymptotic


def profile_stability_margin(out: dict, ionization_potential: float) -> go.Figure:
    """Velikhov-ionisation stability margin beta_crit/beta at every slice the march
    visited (magnetohydrodynamics.stability -- see examples/channel_profile.py for
    the matplotlib counterpart this mirrors). Log-scaled: the margin routinely spans
    several orders of magnitude along one channel."""
    margin, margin_asymptotic = compute_stability_margin(out, ionization_potential)
    fig = plot_single_axis(
        out["x"],
        [("exact (5.13)", margin, COLOR_BLUE, "solid"), ("asymptotic (6.23)", margin_asymptotic, COLOR_GREEN, "dash")],
        "Stability Margin Along the Channel", "β_crit/β",
    )
    fig.update_yaxes(type="log")
    fig.add_hline(y=1.0, line=dict(color="black", width=1))
    return fig


def compare_temperature(configs: list[dict]) -> go.Figure:
    series = []
    for i, cfg in enumerate(configs):
        c = _config_color(i)
        series.append((f"{cfg['label']}: T_p", cfg["out"]["Tp"], c, "solid"))
        series.append((f"{cfg['label']}: T_e", cfg["out"]["Te"], c, "dash"))
    return plot_single_axis(
        configs[0]["out"]["x"], series,
        "Electron vs. Primary Gas Temperature (solid = T_p, dashed = T_e)", "Temperature [K]",
    )


def compare_primary_gas(configs: list[dict]) -> go.Figure:
    left = [(f"{cfg['label']}: n_p", cfg["out"]["np"], _config_color(i), "solid") for i, cfg in enumerate(configs)]
    right = [(f"{cfg['label']}: v_p", cfg["out"]["u"], _config_color(i), "dash") for i, cfg in enumerate(configs)]
    return plot_twin_axis(
        configs[0]["out"]["x"], left, right, "Primary Gas (solid = n_p, dashed = v_p)", "n_p [m⁻³]", "v_p [m/s]",
    )


def compare_density(configs: list[dict]) -> go.Figure:
    left, right = [], []
    for i, cfg in enumerate(configs):
        c = _config_color(i)
        left.append((f"{cfg['label']}: n_s", cfg["seed_number_density"], c, "solid"))
        left.append((f"{cfg['label']}: n_e", cfg["out"]["ne"], c, "dash"))
        ionization_fraction = cfg["out"]["ne"] / cfg["seed_number_density"]
        right.append((f"{cfg['label']}: n_e/n_s", ionization_fraction, c, "dot"))
    return plot_twin_axis(
        configs[0]["out"]["x"], left, right,
        "Seed/Electron Density (solid/dashed), Ionization Fraction (dotted)",
        "Number density [m⁻³]", "Ionization fraction",
        left_log=True, right_range=(0.0, 1.05),
    )


def compare_power(configs: list[dict]) -> go.Figure:
    series = []
    for i, cfg in enumerate(configs):
        c = _config_color(i)
        series.append((f"{cfg['label']}: S_Ω", cfg["out"]["S_ohm"], c, "solid"))
        series.append((f"{cfg['label']}: S_L", cfg["out"]["S_load"], c, "dash"))
    return plot_single_axis(
        configs[0]["out"]["x"], series, "Ohmic Heating (solid) vs Load Power (dashed)", "Power density [W/m³]",
    )


def compare_pressure(configs: list[dict]) -> go.Figure:
    series = [(cfg["label"], cfg["out"]["p"], _config_color(i), "solid") for i, cfg in enumerate(configs)]
    return plot_single_axis(configs[0]["out"]["x"], series, "Primary Gas Pressure", "p_p [Pa]")


def compare_hall_parameter(configs: list[dict]) -> go.Figure:
    series = [(cfg["label"], cfg["out"]["beta"], _config_color(i), "solid") for i, cfg in enumerate(configs)]
    return plot_single_axis(configs[0]["out"]["x"], series, "Hall Parameter Along the Channel", "β [-]")


COMPARE_PLOTS = {
    "Temperature (Tp, Te)": compare_temperature,
    "Primary Gas (np, vp)": compare_primary_gas,
    "Seed/Electron Density + Ionization": compare_density,
    "Ohmic vs Load Power": compare_power,
    "Pressure": compare_pressure,
    "Hall Parameter": compare_hall_parameter,
}


# --- apply a pending "set slider(s)/bound(s) to these value(s)" request, before any widget
# with those keys is created below. {state_key: value, ...} -- used by the single- and
# multi-parameter "apply optimal value" buttons, and by "reset bounds to default".
if "pending_apply" in st.session_state:
    updates = st.session_state.pop("pending_apply")
    for key, value in updates.items():
        st.session_state[key] = value


hall_solver, gas_type = build_default_hall_solver()
ideal_gas = IdealGas(gas_type=gas_type)

st.title("1-D Linear Hall MHD Generator")
st.caption(f"{gas_type.name} primary gas, caesium-seeded -- see Derivation.md")

with st.sidebar:
    st.header("Geometry")
    for key in ("channel_side_mm", "channel_length"):
        label, lo, hi, default, step = UI_PARAMS[key]
        bounded_slider(key, label, lo, hi, default, step)
    bounded_slider("num_slices", *NUM_SLICES_PARAMS)

    # Unlike the sliders above, these two only otherwise get a session_state entry
    # once the dialog below has actually been opened once -- setdefault here
    # guarantees they exist from the first script run, the same way every other
    # CONFIG_KEYS entry already does, so Save (which reads st.session_state[key]
    # directly for every CONFIG_KEYS entry) never KeyErrors on a session that never
    # opened this dialog.
    st.session_state.setdefault("tapered_channel_enabled", False)
    st.session_state.setdefault("tapered_area_ratio", 1.0)
    taper_button_label = (
        f"🔻 Tapered channel: ON ({st.session_state['tapered_area_ratio']:.2f}x)"
        if st.session_state["tapered_channel_enabled"]
        else "Tapered channel..."
    )
    if st.button(taper_button_label, width="stretch"):
        tapered_channel_dialog()

    st.header("Inlet Conditions")
    for key in ("inlet_gas_temperature", "inlet_pressure_kpa", "inlet_speed", "magnetic_field", "seed_fraction_log10"):
        label, lo, hi, default, step = UI_PARAMS[key]
        bounded_slider(key, label, lo, hi, default, step)
    st.caption(f"Seed fraction = {10.0 ** st.session_state['seed_fraction_log10']:.3e}")

    st.header("Load")
    label, lo, hi, default, step = UI_PARAMS["load_resistivity"]
    bounded_slider("load_resistivity", label, lo, hi, default, step)

    st.header("💾 Save / Load")
    name_col, save_col = st.columns([3, 1], vertical_alignment="bottom")
    with name_col:
        save_name = st.text_input(
            "Name", key="save_config_name", placeholder="optional, blank -> timestamp", label_visibility="collapsed",
        )
    with save_col:
        if st.button("Save", key="save_config_button", width="stretch"):
            current_values = {key: st.session_state[key] for key in CONFIG_KEYS}
            saved_path = save_config(current_values, save_name)
            st.toast(f"Saved as `{saved_path.name}`")

    saved_paths = list_saved_configs()
    if saved_paths:
        stem_options = [p.stem for p in saved_paths]
        stem_to_path = {p.stem: p for p in saved_paths}
        load_col, load_btn_col = st.columns([3, 1], vertical_alignment="bottom")
        with load_col:
            selected_stem = st.selectbox("Load", stem_options, key="load_select", label_visibility="collapsed")
        with load_btn_col:
            if st.button("Load", key="load_config_button", width="stretch"):
                loaded_values, loaded_name = load_config(stem_to_path[selected_stem])
                st.session_state["pending_apply"] = pending_apply_for_values(loaded_values)
                st.rerun()
    else:
        st.caption(f"No saved configs yet -- go in `{SAVE_DIR.relative_to(SAVE_DIR.parent.parent)}/`.")

ui_values = {key: st.session_state[key] for key in UI_PARAMS}
ui_values["num_slices"] = st.session_state["num_slices"]
ui_values["tapered_channel_enabled"] = st.session_state["tapered_channel_enabled"]
ui_values["tapered_area_ratio"] = st.session_state["tapered_area_ratio"]

params, channel, out, perf = solve(hall_solver, gas_type, ui_values)
inlet = channel.states[0]
seed_fraction = 10.0 ** ui_values["seed_fraction_log10"]
seed_number_density = seed_fraction * out["np"]  # n_s(x) = seed_fraction * n_p(x), constant-area march

stability_margin, _stability_margin_asymptotic = compute_stability_margin(out, default_seed_type().ionization_potential)
min_stability_margin = float(np.min(stability_margin))
if min_stability_margin <= 1.0:
    margin_color = "red"
elif min_stability_margin <= 50.0:
    margin_color = "orange"
else:
    margin_color = "green"

st.subheader("Performance")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Load power P_L", f"{perf['PL']:.1f} W")
m2.metric("Electrical efficiency", f"{perf['eta_electrical'] * 100:.1f} %")
m3.metric("Isentropic efficiency", f"{perf['eta_isentropic'] * 100:.1f} %")
m4.metric("Enthalpy extraction ratio", f"{perf['enthalpy_extraction_ratio'] * 100:.1f} %")
m5.metric(
    "Min stability margin", f":{margin_color}[{min_stability_margin:.3g}]",
    help="Velikhov-ionisation marginal-stability ratio β_crit/β (exact criterion, eq. 5.13), minimum along the "
         "whole channel -- see the Profiles tab's own plot for the full profile, and the Stability tab to explore "
         "it away from this exact operating point. Red: ≤1, unstable somewhere in the channel. Orange: 1-50, "
         "stable but with little margin. Green: >50, comfortably stable.",
)

if channel.choked:
    st.warning(
        f"⚠️ Flow reached M={hall_solver.max_mach_number:g} (choked, Rayleigh-flow limit) at "
        f"x={channel.x[-1]:.4f} m -- {len(channel)} of the requested {ui_values['num_slices']} slices were "
        "computed. Plots and metrics below only cover the channel up to that point. Reduce the magnetic field, "
        "load resistivity, or seed fraction to push the choke point further down the channel."
    )

# st.tabs() looked like the natural fit here, but its active-tab selection is purely
# frontend state that can reset to the first tab on a rerun triggered by an unrelated
# sidebar change -- especially likely here since the choked-flow st.warning right above
# conditionally appears/disappears, shifting the element tree these sit in. This is a
# documented, currently-unresolved Streamlit limitation
# (https://github.com/streamlit/streamlit/issues/13341); even key + on_change="rerun"
# (which the docs suggest tracks tab state) did not survive an unrelated rerun when
# checked against this exact app. st.segmented_control is used instead as a tab-bar
# substitute: a normal stateful widget whose value reliably persists in st.session_state
# across ANY rerun, the same guarantee a slider already has. required=True keeps exactly
# one option always selected, matching st.tabs()' own behavior -- and, unlike st.tabs()
# (which runs every tab's body on every rerun), only the selected section's body runs
# below, so an unrelated tab's content no longer computes at all while hidden.
TAB_LABELS = ["📈 Profiles", "🔎 Inlet Summary", "🛡️ Stability", "🎯 Optimize", "🧬 Multi-Optimize", "🆚 Compare"]
active_tab = st.segmented_control(
    "Main view", TAB_LABELS, default=TAB_LABELS[0], required=True,
    key="app_main_tab", label_visibility="collapsed",
)

if active_tab == "📈 Profiles":
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(profile_temperature(out), width="stretch", key="plot_temperature")
        st.plotly_chart(profile_primary_gas(out), width="stretch", key="plot_primary_gas")
        st.plotly_chart(profile_mach_number(out, ideal_gas), width="stretch", key="plot_mach_number")
        st.plotly_chart(profile_density(out, seed_number_density), width="stretch", key="plot_density")

    with col2:
        st.plotly_chart(profile_power(out), width="stretch", key="plot_power")
        st.plotly_chart(profile_pressure(out), width="stretch", key="plot_pressure")
        st.plotly_chart(profile_hall_parameter(out), width="stretch", key="plot_hall_parameter")
        st.plotly_chart(profile_stability_margin(out, default_seed_type().ionization_potential), width="stretch", key="plot_stability_margin")
        if "area" in out:  # only present for a tapered channel (see the Geometry sidebar's dialog)
            st.plotly_chart(profile_channel_area(out), width="stretch", key="plot_channel_area")

if active_tab == "🛡️ Stability":
    stability_tab.render(ui_values)

if active_tab == "🔎 Inlet Summary":
    st.markdown("Everything the solver computes for the very first slice (x = 0), where the axial march starts.")

    mach0 = float(ideal_gas.get_mach_number(inlet.flow_speed, inlet.gas_temperature))
    ns0 = seed_number_density[0]
    ionization_fraction0 = inlet.electron_number_density / ns0 if ns0 > 0 else float("nan")
    delta_T0 = inlet.electron_temperature / inlet.gas_temperature - 1.0
    # channel.load_resistivity is a single scalar for the constant-area Channel, but a
    # per-slice array for a tapered TaperedChannel (eta_L = R_load*A(x)/length varies
    # once area does) -- np.atleast_1d(...)[0] reads "the inlet value" uniformly either way.
    Z0 = float(np.atleast_1d(channel.load_resistivity)[0]) / inlet.resistivity
    Z_matched0 = inlet.hall_parameter ** 2 + 1.0
    hall_current_ratio0 = abs(inlet.current_density[1] / inlet.current_density[0])

    rows = [
        ("Primary gas temperature  T_p", f"{inlet.gas_temperature:,.1f}", "K"),
        ("Primary gas pressure  p", f"{inlet.gas_pressure:,.1f}", "Pa"),
        ("Primary gas number density  n_p", f"{inlet.gas_number_density:.3e}", "m^-3"),
        ("Flow speed  u", f"{inlet.flow_speed:,.2f}", "m/s"),
        ("Mach number  M", f"{mach0:.3f}", "-"),
        ("Electron temperature  T_e", f"{inlet.electron_temperature:,.1f}", "K"),
        ("Electron/gas temperature ratio  T_e/T_p", f"{inlet.electron_temperature / inlet.gas_temperature:.3f}", "-"),
        ("Seed number density  n_s", f"{ns0:.3e}", "m^-3"),
        ("Electron number density  n_e", f"{inlet.electron_number_density:.3e}", "m^-3"),
        ("Ionization fraction  n_e/n_s", f"{ionization_fraction0:.4f}", "-"),
        ("Momentum-transfer frequency  ν_M", f"{inlet.momentum_transfer_frequency:.3e}", "1/s"),
        ("Energy-transfer frequency  ν_E", f"{inlet.energy_transfer_frequency:.3e}", "1/s"),
        ("Plasma resistivity  η", f"{inlet.resistivity:.3e}", "Ω·m"),
        ("Plasma conductivity  σ", f"{inlet.conductivity:.3e}", "S/m"),
        ("Hall parameter  β", f"{inlet.hall_parameter:.3f}", "-"),
        ("Ideal channel length  L (Messerle 4.20)", f"{inlet.ideal_channel_length:.4g}", "m"),
        ("Load ratio  Z = R_load/η", f"{Z0:.3f}", "-"),
        ("Power-matched load ratio  Z* = β²+1", f"{Z_matched0:.3f}", "-"),
        ("Current density  J_x (Faraday)", f"{inlet.current_density[0]:,.2f}", "A/m²"),
        ("Current density  J_y (Hall)", f"{inlet.current_density[1]:,.2f}", "A/m²"),
        ("Hall/Faraday current ratio  |J_y/J_x|", f"{hall_current_ratio0:.3f}", "-"),
        ("Axial electric field  E_x", f"{inlet.axial_electric_field:,.2f}", "V/m"),
        ("Ohmic power density  S_Ω", f"{inlet.ohmic_power_density:,.2f}", "W/m³"),
        ("Load power density  S_L", f"{inlet.load_power_density:,.2f}", "W/m³"),
    ]
    st.dataframe(
        pd.DataFrame(rows, columns=["Quantity", "Value", "Units"]),
        hide_index=True, width="stretch",
    )

    st.markdown("#### Why the numbers come out this way")

    mach_note = regime_note(
        mach0, 0.95, 1.05,
        f"**Subsonic inlet** (M = {mach0:.2f} < 1): the primary gas can still respond to downstream pressure changes.",
        f"**Transonic inlet** (M = {mach0:.2f} ≈ 1): right at the sonic boundary.",
        f"**Supersonic inlet** (M = {mach0:.2f} > 1): downstream conditions can't propagate back upstream.",
    )
    beta_note = regime_note(
        inlet.hall_parameter, 0.3, 3.0,
        f"**Collision-dominated** (β = {inlet.hall_parameter:.2f} ≪ 1): electrons collide with neutrals before they can "
        f"gyrate, so current flows almost straight to the load (J_x ≫ J_y).",
        f"**Intermediate regime** (β = {inlet.hall_parameter:.2f}): Faraday and Hall currents are comparable.",
        f"**Hall-dominated** (β = {inlet.hall_parameter:.2f} ≫ 1): electrons gyrate many times between collisions, "
        f"deflecting most of the current into the transverse (Hall, J_y) direction instead of to the load (J_x/J_y = "
        f"{1 / hall_current_ratio0:.2f}).",
    )
    z_ratio = Z0 / Z_matched0
    z_note = regime_note(
        z_ratio, 0.7, 1.4,
        f"**Under-loaded** (Z = {Z0:.2f} vs. the locally power-matched Z* = β²+1 = {Z_matched0:.2f}): the load "
        f"resistivity is too low for this β -- raising it should increase S_L (try the Optimize tab).",
        f"**Near the local power-matched point** (Z = {Z0:.2f} ≈ Z* = {Z_matched0:.2f}): for fixed β and η, S_L is "
        f"maximized when Z = β²+1 (from S_L ∝ Z/(β²+1+Z)²). Note η and T_e also shift with Z, so this is a rule of "
        f"thumb, not the exact global optimum -- that's what the Optimize tab solves for.",
        f"**Over-loaded** (Z = {Z0:.2f} vs. the locally power-matched Z* = β²+1 = {Z_matched0:.2f}): close to "
        f"open-circuit -- lowering the load resistivity should increase S_L (try the Optimize tab).",
    )
    ionization_note = regime_note(
        ionization_fraction0, 0.01, 0.5,
        f"Only **{ionization_fraction0:.2%}** of the seed is ionized -- Saha equilibrium at T_e = "
        f"{inlet.electron_temperature:,.0f} K is still far from fully ionizing the caesium seed.",
        f"**{ionization_fraction0:.2%}** of the seed is ionized -- a moderately ionized seed plasma.",
        f"**{ionization_fraction0:.2%}** of the seed is ionized -- the caesium seed is nearly fully ionized at this "
        f"electron temperature, so n_e is limited mainly by n_s, not by the Saha exponential anymore.",
    )
    delta_T_note = (
        f"Electrons run **{delta_T0:.1%} hotter** than the primary gas (T_e/T_p = {inlet.electron_temperature / inlet.gas_temperature:.2f}) "
        f"because Ohmic heating (S_Ω ∝ β⁴) outpaces how fast elastic collisions (ν_E) can relax that energy back into "
        f"the neutral gas -- this is exactly the ΔT term in `HallSolver.solve_equilibrium`, and it grows with both β and "
        f"the Mach number."
    )
    channel_length_ratio = params["length"] / inlet.ideal_channel_length
    channel_length_note = regime_note(
        channel_length_ratio, 0.5, 2.0,
        f"**Shorter than Messerle's estimate** (configured length {params['length']:.3g} m vs. L ≈ "
        f"{inlet.ideal_channel_length:.3g} m at the inlet, ratio {channel_length_ratio:.2f}): the B-field/flow "
        f"interaction (Sec. 4.3) may not have room to fully develop before the outlet.",
        f"**Close to Messerle's estimate** (configured length {params['length']:.3g} m vs. L ≈ "
        f"{inlet.ideal_channel_length:.3g} m at the inlet, ratio {channel_length_ratio:.2f}).",
        f"**Longer than Messerle's estimate** (configured length {params['length']:.3g} m vs. L ≈ "
        f"{inlet.ideal_channel_length:.3g} m at the inlet, ratio {channel_length_ratio:.2f}): well past the "
        f"interaction length -- note L itself grows fast down the channel too, since u, σ, and p all evolve.",
    )

    for note in (mach_note, beta_note, z_note, ionization_note, delta_T_note, channel_length_note):
        st.markdown(f"- {note}")

if active_tab == "🎯 Optimize":
    st.caption(
        "Sweeps one parameter within the bounds below to maximize the chosen objective, holding everything else at "
        "its current slider value."
    )

    oc1, oc2 = st.columns([1, 1])
    with oc1:
        target_key = st.selectbox("Parameter to optimize", list(UI_PARAMS), format_func=lambda k: UI_PARAMS[k][0])
    with oc2:
        objective_name = st.selectbox("Objective to maximize", list(OBJECTIVES))

    _, default_lo, default_hi, _, target_step = UI_PARAMS[target_key]
    bound_lo, bound_hi = st.slider(
        f"Search bounds for {UI_PARAMS[target_key][0]}",
        min_value=default_lo, max_value=default_hi, value=(default_lo, default_hi),
        step=target_step, key=f"bound_range_{target_key}",
    )

    has_result = (
        "last_optimization" in st.session_state
        and st.session_state["last_optimization"]["target_key"] == target_key
        and st.session_state["last_optimization"]["objective_name"] == objective_name
    )
    run_col, apply_col = st.columns([1, 1])
    with run_col:
        run_clicked = st.button("Run optimization", type="primary", width="stretch")
    with apply_col:
        apply_clicked = st.button("Apply optimal value to slider", disabled=not has_result, width="stretch")

    if run_clicked:
        if bound_lo >= bound_hi:
            st.error("Lower bound must be less than upper bound.")
        else:
            def objective(value: float) -> float:
                trial_ui = dict(ui_values)
                trial_ui[target_key] = value
                trial_params, _, trial_out, trial_perf = solve(hall_solver, gas_type, trial_ui)
                return OBJECTIVES[objective_name](trial_out, trial_params, trial_perf)

            with st.spinner("Optimizing..."):
                result = minimize_scalar(lambda v: -objective(v), bounds=(bound_lo, bound_hi), method="bounded")
                sweep_x = np.linspace(bound_lo, bound_hi, 40)
                sweep_y = [objective(v) for v in sweep_x]

            st.session_state["last_optimization"] = dict(
                target_key=target_key, objective_name=objective_name,
                value=float(result.x), objective=float(-result.fun),
                sweep_x=sweep_x.tolist(), sweep_y=sweep_y,
            )
            st.rerun()

    if apply_clicked:
        st.session_state["pending_apply"] = {target_key: st.session_state["last_optimization"]["value"]}
        st.rerun()

    if "last_optimization" in st.session_state:
        res = st.session_state["last_optimization"]
        if res["target_key"] == target_key and res["objective_name"] == objective_name:
            label = UI_PARAMS[target_key][0]
            st.success(f"Best {label} = {res['value']:.5g}  →  {res['objective_name']} = {res['objective']:.5g}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res["sweep_x"], y=res["sweep_y"], mode="lines", name=objective_name,
                                      line=dict(color=COLOR_BLUE, width=2.5)))
            fig.add_trace(go.Scatter(x=[res["value"]], y=[res["objective"]], mode="markers", name="Optimum",
                                      marker=dict(color=COLOR_RED, size=10)))
            fig.add_vline(x=res["value"], line=dict(color=COLOR_RED, dash="dash"))
            fig.update_layout(xaxis_title=label, yaxis_title=objective_name, height=340,
                               margin=dict(l=60, r=60, t=30, b=40), hovermode="x unified")
            st.plotly_chart(fig, width="stretch", key="plot_single_optimize_sweep")

if active_tab == "🧬 Multi-Optimize":
    st.caption(
        "Optimizes seed fraction, load resistivity, inlet pressure, inlet speed, and inlet gas temperature "
        "**together** -- channel geometry and magnetic field stay fixed at their current slider values, since "
        "those are fixed by the use case. Uncheck any parameter below to hold IT fixed too (at its current "
        "slider value) instead of searching it. This uses a derivative-free global optimizer (the choke limit "
        "makes the objective landscape non-smooth) and takes noticeably longer than the single-parameter tab."
    )

    oc1, oc2 = st.columns([2, 1])
    with oc1:
        multi_objective_name = st.selectbox("Objective to maximize", list(OBJECTIVES), key="multi_objective")
    with oc2:
        thoroughness = st.selectbox("Search thoroughness", list(SEARCH_THOROUGHNESS_PRESETS), index=1, key="multi_thoroughness")
    maxiter = SEARCH_THOROUGHNESS_PRESETS[thoroughness]["maxiter"]
    popsize = SEARCH_THOROUGHNESS_PRESETS[thoroughness]["popsize"]

    st.markdown("**Search bounds**")
    st.caption(
        "Note: the seed fraction bound is in log10 units, matching its sidebar slider (e.g. -3 → 0.001). "
        "Uncheck a parameter to hold it fixed at its current sidebar value instead of searching it."
    )
    multi_bounds: dict[str, tuple[float, float]] = {}
    multi_enabled: dict[str, bool] = {}
    bound_cols = st.columns(len(MULTI_OPTIMIZABLE_KEYS))
    for col, key in zip(bound_cols, MULTI_OPTIMIZABLE_KEYS, strict=True):
        _, default_lo, default_hi, _, key_step = UI_PARAMS[key]
        with col:
            multi_enabled[key] = st.checkbox(bounds_label(key), value=True, key=f"multi_optimize_enabled_{key}")
            multi_bounds[key] = st.slider(
                bounds_label(key), min_value=default_lo, max_value=default_hi,
                value=(default_lo, default_hi), step=key_step, key=f"multi_bound_range_{key}",
                disabled=not multi_enabled[key], label_visibility="collapsed",
            )
            if not multi_enabled[key]:
                st.caption(f"Fixed: {result_value(key, ui_values[key])}")

    # Only parameters left checked above are actually searched; everything else stays at its
    # current sidebar value in every trial (both the timing samples and the real search below),
    # via `dict(ui_values, ...)`'s base already carrying every UI_PARAMS key.
    active_keys = [key for key in MULTI_OPTIMIZABLE_KEYS if multi_enabled[key]]
    invalid_bounds = [key for key in active_keys if multi_bounds[key][0] >= multi_bounds[key][1]]

    # Search at a reduced axial resolution for speed (hundreds-thousands of marches), then
    # re-evaluate the found optimum at the slider's actual resolution for display.
    search_num_slices = min(int(ui_values["num_slices"]), 60)
    ndim = len(active_keys)

    if ndim == 0:
        st.warning("Check at least one parameter above to search over.")
        max_evals = 0
    else:
        max_evals = popsize * ndim * (maxiter + 1)

        # Time evaluations at a few random points drawn from the actual search bounds (not just
        # the current slider point, which is often mid-choke and unrepresentatively fast/slow) to
        # give an upfront estimate. Cheap (a handful of extra marches) -- reruns on every change.
        _rng = np.random.default_rng(0)
        _sample_times = []
        for _ in range(5):
            _trial_ui = dict(ui_values, num_slices=search_num_slices)
            for _key in active_keys:
                _lo, _hi = multi_bounds[_key]
                _trial_ui[_key] = _rng.uniform(_lo, _hi) if _hi > _lo else _lo
            _t0 = time.perf_counter()
            solve(hall_solver, gas_type, _trial_ui)
            _sample_times.append(time.perf_counter() - _t0)
        per_eval_s = float(np.mean(_sample_times))
        # +40% fudge factor: the sampled points tend to under-represent the longer, un-choked
        # evaluations DE spends more time on as it converges, plus DE's own per-generation and
        # polish-step overhead aren't captured by march timing alone.
        estimated_s = per_eval_s * max_evals * 1.4
        st.caption(
            f"Estimated time: up to ~{estimated_s:.0f}s ({max_evals:,} evaluations at {search_num_slices} slices "
            f"each, ~{per_eval_s * 1000:.1f} ms/evaluation). Often finishes sooner via early convergence."
        )

    has_multi_result = (
        "last_multi_optimization" in st.session_state
        and st.session_state["last_multi_optimization"]["objective_name"] == multi_objective_name
    )
    run_col, apply_col = st.columns([1, 1])
    with run_col:
        run_multi_clicked = st.button(
            "Run multi-parameter optimization", type="primary", width="stretch", disabled=ndim == 0,
        )
    with apply_col:
        apply_multi_clicked = st.button(
            "Apply all optimal values to sliders", disabled=not has_multi_result, width="stretch",
        )

    if run_multi_clicked:
        if invalid_bounds:
            st.error(f"Lower bound must be less than upper bound for: {', '.join(bounds_label(k) for k in invalid_bounds)}.")
        else:
            de_bounds = [multi_bounds[key] for key in active_keys]

            def neg_objective(x):
                trial_ui = dict(ui_values, num_slices=search_num_slices)
                for key, value in zip(active_keys, x, strict=True):
                    trial_ui[key] = value
                trial_params, _, trial_out, trial_perf = solve(hall_solver, gas_type, trial_ui)
                return -OBJECTIVES[multi_objective_name](trial_out, trial_params, trial_perf)

            progress_bar = st.progress(0.0, text="Starting search...")
            search_start = time.perf_counter()

            def on_generation(intermediate_result):
                elapsed = time.perf_counter() - search_start
                fraction = min(1.0, intermediate_result.nit / maxiter)
                eta = elapsed * (1.0 - fraction) / fraction if fraction > 0 else 0.0
                progress_bar.progress(
                    fraction,
                    text=(
                        f"Generation {intermediate_result.nit}/{maxiter} "
                        f"({intermediate_result.nfev:,} evaluations) -- best {multi_objective_name} so far: "
                        f"{-intermediate_result.fun:.5g} -- {elapsed:.1f}s elapsed, ~{eta:.1f}s remaining"
                    ),
                )

            result = differential_evolution(
                neg_objective, de_bounds, maxiter=maxiter, popsize=popsize, tol=1e-6, seed=0, polish=True,
                callback=on_generation,
            )

            total_time = time.perf_counter() - search_start
            progress_bar.progress(1.0, text=f"Done in {total_time:.1f}s ({result.nfev:,} evaluations).")

            best_ui = dict(ui_values)
            for key, value in zip(active_keys, result.x, strict=True):
                best_ui[key] = value
            best_params, best_channel, best_out, best_perf = solve(hall_solver, gas_type, best_ui)
            best_objective = OBJECTIVES[multi_objective_name](best_out, best_params, best_perf)

            st.session_state["last_multi_optimization"] = dict(
                objective_name=multi_objective_name,
                values={key: float(value) for key, value in zip(active_keys, result.x, strict=True)},
                objective=float(best_objective),
                choked=best_channel.choked,
            )
            st.rerun()

    if apply_multi_clicked:
        st.session_state["pending_apply"] = dict(st.session_state["last_multi_optimization"]["values"])
        st.rerun()

    if "last_multi_optimization" in st.session_state:
        res = st.session_state["last_multi_optimization"]
        if res["objective_name"] == multi_objective_name:
            current_objective = OBJECTIVES[multi_objective_name](out, params, perf)
            st.success(
                f"Best {multi_objective_name} = {res['objective']:.5g}  "
                f"(current slider values give {current_objective:.5g})"
            )
            if res["choked"]:
                st.caption("⚠️ The optimum found chokes before the channel outlet (see the Profiles tab warning).")

            comparison = pd.DataFrame(
                [
                    (
                        result_label(key), result_value(key, ui_values[key]),
                        result_value(key, res["values"][key]) if key in res["values"] else "-- (fixed)",
                    )
                    for key in MULTI_OPTIMIZABLE_KEYS
                ],
                columns=["Parameter", "Current", "Optimal"],
            )
            st.dataframe(comparison, hide_index=True, width="stretch")

if active_tab == "🆚 Compare":
    st.caption(
        "Compare the current configuration and/or up to three saved files side by side -- performance numbers "
        "plus any two profile plots, all live (no need to re-run anything when you change a selection)."
    )

    saved_paths = list_saved_configs()
    slot_options = ["None", "Current configuration"] + [p.stem for p in saved_paths]
    stem_to_path = {p.stem: p for p in saved_paths}

    slot_cols = st.columns(3)
    slot_choices = []
    for i, col in enumerate(slot_cols):
        with col:
            default_index = 1 if i == 0 else 0  # slot 1 defaults to "Current configuration"
            slot_choices.append(st.selectbox(f"Slot {i + 1}", slot_options, index=default_index, key=f"compare_slot_{i}"))

    configs = []
    for choice in slot_choices:
        if choice == "None":
            continue
        if choice == "Current configuration":
            configs.append(dict(label="Current", out=out, perf=perf, channel=channel, seed_number_density=seed_number_density))
        else:
            c_values, c_name = load_config(stem_to_path[choice])
            c_params, c_channel, c_out, c_perf = solve(hall_solver, gas_type, c_values)
            c_seed_number_density = 10.0 ** c_values["seed_fraction_log10"] * c_out["np"]
            configs.append(dict(label=c_name, out=c_out, perf=c_perf, channel=c_channel, seed_number_density=c_seed_number_density))

    if not configs:
        st.info("Pick at least one configuration above (a slot, or a saved file) to compare.")
    else:
        st.markdown("**Performance**")
        perf_rows = [
            (
                cfg["label"],
                f"{cfg['perf']['PL']:.1f} W",
                f"{cfg['perf']['eta_electrical'] * 100:.1f} %",
                f"{cfg['perf']['eta_isentropic'] * 100:.1f} %",
                f"{cfg['perf']['enthalpy_extraction_ratio'] * 100:.1f} %",
                "Yes" if cfg["channel"].choked else "No",
            )
            for cfg in configs
        ]
        perf_df = pd.DataFrame(
            perf_rows,
            columns=["Configuration", "Load Power", "Electrical Eff.", "Isentropic Eff.", "Enthalpy Extraction", "Choked?"],
        )
        st.dataframe(perf_df, hide_index=True, width="stretch")

        st.markdown("**Plots**")
        plot_names = list(COMPARE_PLOTS)
        pc1, pc2 = st.columns(2)
        with pc1:
            plot_choice_1 = st.selectbox("Plot 1", plot_names, index=0, key="compare_plot_1")
        with pc2:
            # Default to a different plot than Plot 1 (index 1 there == index 1 here would
            # otherwise default both pickers to the same plot).
            plot_choice_2 = st.selectbox("Plot 2", ["None"] + plot_names, index=4, key="compare_plot_2")

        if plot_choice_2 == "None":
            st.plotly_chart(COMPARE_PLOTS[plot_choice_1](configs), width="stretch", key="compare_chart_1")
        else:
            plot_col_1, plot_col_2 = st.columns(2)
            with plot_col_1:
                st.plotly_chart(COMPARE_PLOTS[plot_choice_1](configs), width="stretch", key="compare_chart_1")
            with plot_col_2:
                st.plotly_chart(COMPARE_PLOTS[plot_choice_2](configs), width="stretch", key="compare_chart_2")
