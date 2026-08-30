"""
Interactive UI for the 1-D linear Hall MHD generator.

Lets you tweak channel geometry / inlet conditions with sliders and see the
resulting axial profiles and performance figures update live, inspect the
full inlet plasma state, and optimize one chosen parameter (within bounds you
set) to maximize a chosen objective (e.g. load power).

Run with:
    uv run --extra ui streamlit run ui/app.py

This is intentionally separate from main.py -- it depends on `streamlit`,
which lives in the optional "ui" dependency group so `uv sync` (no extras)
stays lightweight.
"""
import sys
from pathlib import Path

# This project isn't installed as a package (`package = false` in
# pyproject.toml), and `streamlit run ui/app.py` puts ui/ -- not the repo
# root -- on sys.path. Add the repo root explicitly so `magnetohydrodynamics`
# is importable regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize_scalar

from magnetohydrodynamics.analysis import summarize_performance
from magnetohydrodynamics.presets import build_default_hall_solver
from magnetohydrodynamics.thermophysics.ideal_gas import IdealGas

st.set_page_config(page_title="Hall MHD Generator", layout="wide")

# Parameters the UI exposes, keyed by the name used in both `st.session_state`
# and `build_march_params`. (label, min, max, default, step)
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
NUM_SLICES_DEFAULT = 200

OBJECTIVES = {
    "Load power (PL)": lambda out, params, perf: perf["PL"],
    "Electrical efficiency": lambda out, params, perf: perf["eta_electrical"],
    "Enthalpy extraction ratio": lambda out, params, perf: perf["enthalpy_extraction_ratio"],
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
    params = build_march_params(ui)
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


# --- apply a pending "set slider to this optimal value" request, before any
# widget with that key is created below.
if "pending_apply" in st.session_state:
    key, value = st.session_state.pop("pending_apply")
    st.session_state[key] = value


hall_solver, gas_type = build_default_hall_solver()
ideal_gas = IdealGas(gas_type=gas_type)

st.title("1-D Linear Hall MHD Generator")
st.caption(f"{gas_type.name} primary gas, caesium-seeded -- see Derivation.md")

with st.sidebar:
    st.header("Geometry")
    for key in ("channel_side_mm", "channel_length"):
        label, lo, hi, default, step = UI_PARAMS[key]
        st.session_state.setdefault(key, default)
        st.slider(label, lo, hi, key=key, step=step)
    st.session_state.setdefault("num_slices", NUM_SLICES_DEFAULT)
    st.slider("Axial slices", 20, 400, key="num_slices", step=10)

    st.header("Inlet Conditions")
    for key in ("inlet_gas_temperature", "inlet_pressure_kpa", "inlet_speed", "magnetic_field", "seed_fraction_log10"):
        label, lo, hi, default, step = UI_PARAMS[key]
        st.session_state.setdefault(key, default)
        st.slider(label, lo, hi, key=key, step=step)
    st.caption(f"Seed fraction = {10.0 ** st.session_state['seed_fraction_log10']:.3e}")

    st.header("Load")
    label, lo, hi, default, step = UI_PARAMS["load_resistivity"]
    st.session_state.setdefault("load_resistivity", default)
    st.slider(label, lo, hi, key="load_resistivity", step=step)

ui_values = {key: st.session_state[key] for key in UI_PARAMS}
ui_values["num_slices"] = st.session_state["num_slices"]

params, channel, out, perf = solve(hall_solver, gas_type, ui_values)
inlet = channel.states[0]
seed_fraction = 10.0 ** ui_values["seed_fraction_log10"]
seed_number_density = seed_fraction * out["np"]  # n_s(x) = seed_fraction * n_p(x), constant-area march

st.subheader("Performance")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Load power P_L", f"{perf['PL']:.1f} W")
m2.metric("Electrical efficiency", f"{perf['eta_electrical'] * 100:.1f} %")
m3.metric("Isentropic efficiency", f"{perf['eta_isentropic'] * 100:.1f} %")
m4.metric("Enthalpy extraction ratio", f"{perf['enthalpy_extraction_ratio'] * 100:.1f} %")

tab_profiles, tab_inlet, tab_optimize = st.tabs(["📈 Profiles", "🔎 Inlet Summary", "🎯 Optimize"])

with tab_profiles:
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(out["x"], out["Tp"], label=r"$T_p$", linewidth=2)
        ax.plot(out["x"], out["Te"], "--", label=r"$T_e$", linewidth=2)
        ax.set_xlabel("x [m]"); ax.set_ylabel("Temperature [K]")
        ax.set_title("Electron vs. Primary Gas Temperature")
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        fig, ax1 = plt.subplots(figsize=(6, 3.5))
        ax1.set_xlabel("x [m]")
        ax1.set_ylabel(r"$n_p$ [m$^{-3}$]", color="tab:blue")
        ax1.plot(out["x"], out["np"], color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(alpha=0.3)
        ax2 = ax1.twinx()
        ax2.set_ylabel(r"$v_p$ [m/s]", color="tab:orange")
        ax2.plot(out["x"], out["u"], "--", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        ax1.set_title(r"Primary Gas: $n_p$, $v_p$")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.semilogy(out["x"], seed_number_density, label=r"$n_s$ (seed)", linewidth=2)
        ax.semilogy(out["x"], out["ne"], "--", label=r"$n_e$ (electrons)", linewidth=2)
        ax.set_xlabel("x [m]"); ax.set_ylabel(r"Number density [m$^{-3}$]")
        ax.set_title("Seed vs. Electron Density (how much of the seed is ionized)")
        ax.grid(alpha=0.3, which="both"); ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(out["x"], out["S_ohm"], label=r"$S_{\Omega}$ (ohmic)", linewidth=2)
        ax.plot(out["x"], out["S_load"], label=r"$S_L$ (load)", linewidth=2)
        ax.set_xlabel("x [m]"); ax.set_ylabel("Power density [W/m$^3$]")
        ax.set_title("Ohmic Heating vs Load Power")
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(out["x"], out["p"], color="tab:green")
        ax.set_xlabel("x [m]"); ax.set_ylabel(r"$p_p$ [Pa]")
        ax.set_title("Primary Gas Pressure")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(out["x"], out["beta"], color="tab:purple")
        ax.set_xlabel("x [m]"); ax.set_ylabel(r"$\beta$ [-]")
        ax.set_title("Hall Parameter Along the Channel")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with tab_inlet:
    st.markdown("Everything the solver computes for the very first slice (x = 0), where the axial march starts.")

    mach0 = float(ideal_gas.get_mach_number(inlet.flow_speed, inlet.gas_temperature))
    ns0 = seed_number_density[0]
    ionization_fraction0 = inlet.electron_number_density / ns0 if ns0 > 0 else float("nan")
    delta_T0 = inlet.electron_temperature / inlet.gas_temperature - 1.0
    Z0 = channel.load_resistivity / inlet.resistivity
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
        f"the neutral gas -- this is exactly the ΔT term in `HallSolver._solve_slice`, and it grows with both β and "
        f"the Mach number."
    )

    for note in (mach_note, beta_note, z_note, ionization_note, delta_T_note):
        st.markdown(f"- {note}")

with tab_optimize:
    st.caption(
        "Sweeps one parameter within the bounds below to maximize the chosen objective, holding everything else at "
        "its current slider value."
    )

    oc1, oc2 = st.columns([1, 1])
    with oc1:
        target_key = st.selectbox("Parameter to optimize", list(UI_PARAMS), format_func=lambda k: UI_PARAMS[k][0])
    with oc2:
        objective_name = st.selectbox("Objective to maximize", list(OBJECTIVES))

    _, default_lo, default_hi, _, _ = UI_PARAMS[target_key]
    bc1, bc2 = st.columns(2)
    with bc1:
        bound_lo = st.number_input("Lower bound", value=default_lo, key=f"bound_lo_{target_key}")
    with bc2:
        bound_hi = st.number_input("Upper bound", value=default_hi, key=f"bound_hi_{target_key}")

    if st.button("Run optimization", type="primary"):
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

    if "last_optimization" in st.session_state:
        res = st.session_state["last_optimization"]
        if res["target_key"] == target_key and res["objective_name"] == objective_name:
            label = UI_PARAMS[target_key][0]
            st.success(f"Best {label} = {res['value']:.5g}  →  {res['objective_name']} = {res['objective']:.5g}")

            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(res["sweep_x"], res["sweep_y"])
            ax.axvline(res["value"], color="tab:red", linestyle="--")
            ax.scatter([res["value"]], [res["objective"]], color="tab:red", zorder=5)
            ax.set_xlabel(label); ax.set_ylabel(objective_name)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            if st.button("Apply optimal value to slider"):
                st.session_state["pending_apply"] = (target_key, res["value"])
                st.rerun()
