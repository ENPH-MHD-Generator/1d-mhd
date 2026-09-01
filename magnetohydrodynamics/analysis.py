"""Post-processing of a solved Channel -- performance figures of merit."""
from __future__ import annotations

import numpy as np


def summarize_performance(out: dict, A: float, cp: float, m_p: float, gamma: float) -> dict:
    # This function wqs ChatGPT'd, probably check

    x = out['x']
    Tp = out['Tp']
    u = out['u']
    nparr = out['np']  # primary number density
    p = out['p']
    S_L = out['S_load']
    S_Ohm = out['S_ohm']

    # inlet/outlet states
    Tp_in, Tp_out = Tp[0], Tp[-1]
    u_in, u_out = u[0], u[-1]
    p_in, p_out = p[0], p[-1]

    # mass flow (constant area)
    rho_in = m_p * nparr[0]
    mdot = rho_in * u_in * A

    # stagnation enthalpies (per unit mass)
    h0_in = cp * Tp_in + 0.5 * u_in ** 2
    h0_out = cp * Tp_out + 0.5 * u_out ** 2
    dh0 = h0_in - h0_out

    # Powers
    PL = np.trapezoid(S_L, x) * A
    POhm = np.trapezoid(S_Ohm, x) * A

    # Ratios
    enthalpy_extraction_ratio = dh0 / h0_in
    chi_load = (PL / mdot) / h0_in
    eta_electrical = PL / max(1e-30, (PL + POhm))

    pressure_ratio = p_in / p_out
    eta_isentropic = enthalpy_extraction_ratio / (1 - pressure_ratio ** (- (gamma - 1) / gamma))

    return {
        "mdot": mdot,
        "h0_in": h0_in, "h0_out": h0_out, "dh0": dh0,
        "PL": PL, "POhm": POhm,
        "enthalpy_extraction_ratio": enthalpy_extraction_ratio,
        "chi_load": chi_load, "eta_electrical": eta_electrical,
        "pressure_ratio": pressure_ratio, "eta_isentropic": eta_isentropic
    }


def terminal_power(out: dict, area: float) -> float:
    """P = I_in * V, matching main.py's original inlet-current x terminal-voltage estimate."""
    V = -np.trapezoid(out["Ex"], out["x"])  # volts
    Ix_in = out["Jx"][0] * area
    return Ix_in * V


def summarize_tapered_performance(out: dict, cp: float, m_p: float, gamma: float) -> dict:
    """Variable-area counterpart to `summarize_performance`, for a
    `TaperedHallSolver.march()`-produced `out` dict (has an `"area"` key -- a
    per-slice array, unlike the constant-area version's single scalar `A`
    parameter). The only real difference: `PL`/`POhm` integrate the volumetric power
    densities *times the local area* (`S_L(x)*area(x)`) before integrating over x,
    since area can no longer be pulled out of the integral as a constant factor (see
    Derivation.md's "Variable-Area (Tapered) Channel" section) -- everything else
    (stagnation enthalpy, ratios) is unaffected by area varying."""
    x = out["x"]
    area = out["area"]
    Tp = out["Tp"]
    u = out["u"]
    nparr = out["np"]  # primary number density
    p = out["p"]
    S_L = out["S_load"]
    S_Ohm = out["S_ohm"]

    # inlet/outlet states
    Tp_in, Tp_out = Tp[0], Tp[-1]
    u_in, u_out = u[0], u[-1]
    p_in, p_out = p[0], p[-1]

    # mass flow rate (constant along the channel by construction -- rho*u*A=const --
    # evaluated at the inlet, same as the constant-area version)
    rho_in = m_p * nparr[0]
    mdot = rho_in * u_in * area[0]

    # stagnation enthalpies (per unit mass)
    h0_in = cp * Tp_in + 0.5 * u_in ** 2
    h0_out = cp * Tp_out + 0.5 * u_out ** 2
    dh0 = h0_in - h0_out

    # Powers -- area INSIDE the integral, unlike summarize_performance's `* A` outside it.
    PL = np.trapezoid(S_L * area, x)
    POhm = np.trapezoid(S_Ohm * area, x)

    # Ratios
    enthalpy_extraction_ratio = dh0 / h0_in
    chi_load = (PL / mdot) / h0_in
    eta_electrical = PL / max(1e-30, (PL + POhm))

    pressure_ratio = p_in / p_out
    eta_isentropic = enthalpy_extraction_ratio / (1 - pressure_ratio ** (- (gamma - 1) / gamma))

    return {
        "mdot": mdot,
        "h0_in": h0_in, "h0_out": h0_out, "dh0": dh0,
        "PL": PL, "POhm": POhm,
        "enthalpy_extraction_ratio": enthalpy_extraction_ratio,
        "chi_load": chi_load, "eta_electrical": eta_electrical,
        "pressure_ratio": pressure_ratio, "eta_isentropic": eta_isentropic
    }


def terminal_power_tapered(out: dict) -> float:
    """Variable-area counterpart to `terminal_power` -- reads the inlet area straight
    off `out["area"][0]` instead of taking a separate `area` parameter, since a
    tapered channel's area is a per-slice array, not one number for the whole march."""
    V = -np.trapezoid(out["Ex"], out["x"])  # volts
    Ix_in = out["Jx"][0] * out["area"][0]
    return Ix_in * V
