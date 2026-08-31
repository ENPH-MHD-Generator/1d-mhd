import time

import matplotlib.pyplot as plt
import numpy as np

from magnetohydrodynamics.analysis import summarize_performance
from magnetohydrodynamics.presets import build_default_hall_solver as build_hall_solver
from magnetohydrodynamics.presets import default_operating_point as operating_point


def main():
    hall_solver, gas_type = build_hall_solver()
    params = operating_point()
    channel_area = params["area"]

    t0 = time.perf_counter()
    channel = hall_solver.march(**params)
    print(time.perf_counter() - t0)

    out = channel.to_dict()

    x = out["x"]
    Ex = out["Ex"]

    V = -np.trapezoid(Ex, x)  # volts
    A = channel_area

    Ix_profile = out["Jx"] * A
    Ix_in = Ix_profile[0]

    P = Ix_in * V

    print(f"Power: {P}")

    print(f"n_e = {out['ne'][0]:.3e}")

    perf = summarize_performance(
        out, A=channel_area,
        cp=gas_type.molar_heat_capacity,
        m_p=gas_type.particle_mass,
        gamma=gas_type.heat_capacity_ratio,
    )

    for k, v in perf.items():
        print(f"{k:10s}: {v}")

    plot_results(out)


def plot_results(out):
    x = out['x']
    Tp = out['Tp']
    Te = out['Te']
    S_ohm = out['S_ohm']
    S_load = out['S_load']
    np_arr = out['np']
    u = out['u']
    p = out['p']

    plt.figure(figsize=(7, 4))
    plt.plot(x, Tp, label=r"$T_p$", linewidth=2)
    plt.plot(x, Te, label=r"$T_e$", linestyle='--', linewidth=2)
    plt.xlabel("x [m]")
    plt.ylabel("Temperature [K]")
    plt.title("Electron vs. Primary Gas Temperature")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(x, S_ohm, label=r"$S_{\Omega}$ (ohmic heating)", linewidth=2)
    plt.plot(x, S_load, label=r"$S_L$ (load power)", linewidth=2)
    plt.xlabel("x [m]")
    plt.ylabel("Power Density [W/m$^3$]")
    plt.title("Ohmic Heating vs Load Power")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    fig, ax1 = plt.subplots(figsize=(7, 4))

    color1 = "tab:blue"
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel(r"$n_p$ [m$^{-3}$]", color=color1)
    ax1.plot(x, np_arr, color=color1, label=r"$n_p$")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, which='both', alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.set_ylabel(r"$v_p$ [m/s]", color=color2)
    ax2.plot(x, u, color=color2, linestyle="--", label=r"$v_p$")
    ax2.tick_params(axis='y', labelcolor=color2)

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.15))
    color3 = "tab:green"
    ax3.set_ylabel(r"$p_p$ [Pa]", color=color3)
    ax3.plot(x, p, color=color3, linestyle=":", label=r"$p_p$")
    ax3.tick_params(axis='y', labelcolor=color3)

    plt.title("Primary Gas Variables: $n_p$, $v_p$, $p_p$")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()