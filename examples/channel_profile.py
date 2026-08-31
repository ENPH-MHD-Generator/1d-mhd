"""
Quick default-operating-point demo: march the channel (HallSolver.march), print a
performance summary, and plot the axial profiles a first look at the simulation would
want -- temperature, power density, primary-gas variables, and (new here, main.py
predates the stability module) the Velikhov-ionisation stability margin along the
channel. This is examples/'s counterpart of main.py's own plots, extended with
magnetohydrodynamics.stability.

Unlike every other examples/*.py script, this always shows every figure and never
saves to examples/output/ -- meant as a quick look, not a batch-generated report, the
same way main.py itself was.

Run with:
    uv run python examples/channel_profile.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from plotting_utils import STABILITY_NOTE

from magnetohydrodynamics.analysis import summarize_performance, terminal_power
from magnetohydrodynamics.presets import build_default_hall_solver, default_operating_point, default_seed_type
from magnetohydrodynamics.stability import FriedbergAsymptoticCriterion, FriedbergCriterion


def plot_temperatures(out: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(out["x"], out["Tp"], label=r"$T_p$", linewidth=2)
    ax.plot(out["x"], out["Te"], label=r"$T_e$", linestyle="--", linewidth=2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Electron vs. primary gas temperature")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_power_density(out: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(out["x"], out["S_ohm"], label=r"$S_\Omega$ (ohmic heating)", linewidth=2)
    ax.plot(out["x"], out["S_load"], label=r"$S_L$ (load power)", linewidth=2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel(r"Power density [W/m$^3$]")
    ax.set_title("Ohmic heating vs. load power")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_primary_gas_variables(out: dict) -> plt.Figure:
    x = out["x"]
    fig, ax1 = plt.subplots(figsize=(7, 4))

    color1 = "tab:blue"
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel(r"$n_p$ [m$^{-3}$]", color=color1)
    ax1.plot(x, out["np"], color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, which="both", alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.set_ylabel(r"$v_p$ [m/s]", color=color2)
    ax2.plot(x, out["u"], color=color2, linestyle="--")
    ax2.tick_params(axis="y", labelcolor=color2)

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.15))
    color3 = "tab:green"
    ax3.set_ylabel(r"$p_p$ [Pa]", color=color3)
    ax3.plot(x, out["p"], color=color3, linestyle=":")
    ax3.tick_params(axis="y", labelcolor=color3)

    ax1.set_title(r"Primary gas variables: $n_p$, $v_p$, $p_p$")
    fig.tight_layout()
    return fig


def plot_stability_margin(out: dict, ionization_potential: float) -> plt.Figure:
    """Velikhov-ionisation stability margin beta_crit/beta at every slice the march
    actually visited -- exact (5.13) and high-ionisation asymptotic (6.23) criteria
    both plotted, matching the convention used throughout examples/stability_*.py.
    Unlike those scripts (which sweep a fixed operating point over external inputs),
    this profile comes from the ACTUAL slice-by-slice state HallSolver.march produced,
    so it answers "is this specific channel design stable all the way through," not
    "where is the boundary in some broader input space.\""""
    exact = FriedbergCriterion()
    asymptotic = FriedbergAsymptoticCriterion()
    margin = exact.stability_margin(out["beta"], out["Te"], out["Tp"], ionization_potential, out["f_I"])
    margin_asymptotic = asymptotic.stability_margin(out["beta"], out["Te"], out["Tp"], ionization_potential, out["f_I"])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(out["x"], margin, label="exact (5.13)", linewidth=2)
    ax.plot(out["x"], margin_asymptotic, label="asymptotic (6.23)", linestyle="--", linewidth=2)
    ax.axhline(1.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_yscale("log")  # margin spans several orders of magnitude along the channel;
    # log keeps the whole shape legible instead of compressing everything below the
    # final approach to the choke point into a flat line near zero.
    ax.set_xlabel("x [m]")
    ax.set_ylabel(r"stability margin $\beta_{crit}/\beta$")
    ax.set_title(f"Stability margin along the channel  {STABILITY_NOTE}", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def main() -> None:
    hall_solver, gas_type = build_default_hall_solver()
    params = default_operating_point()
    channel_area = params["area"]

    channel = hall_solver.march(**params)
    out = channel.to_dict()

    print(f"Power: {terminal_power(out, channel_area):.3e} W")
    print(f"n_e (inlet) = {out['ne'][0]:.3e} m^-3")
    if channel.choked:
        print(f"(channel choked at x={out['x'][-1]:.4f} m of {params['length']} m requested)")

    perf = summarize_performance(
        out, A=channel_area, cp=gas_type.molar_heat_capacity,
        m_p=gas_type.particle_mass, gamma=gas_type.heat_capacity_ratio,
    )
    print()
    for key, value in perf.items():
        print(f"{key:10s}: {value}")

    plot_temperatures(out)
    plot_power_density(out)
    plot_primary_gas_variables(out)
    plot_stability_margin(out, default_seed_type().ionization_potential)

    plt.show()


if __name__ == "__main__":
    main()
