import time

import numpy as np
import matplotlib.pyplot as plt

import magnetohydrodynamics as mhd


def build_hall_solver() -> tuple[mhd.HallSolver, mhd.GasType]:
    """Argon primary gas, caesium-seeded -- see Derivation.md."""
    gas_type = mhd.GasType(name="Argon", molar_mass=39.948e-3, heat_capacity_ratio=5.0 / 3.0)
    seed_type = mhd.SeedType(
        name="Caesium",
        ionization_potential=3.894,                           # eV
        electron_neutral_cross_section=3.93994730526347e-21,  # m^2
        degeneracy_ratio=1.0,
    )
    transport_model = mhd.MHDTransportModel(seed_type=seed_type, gas_type=gas_type)
    ionization_model = mhd.LocalThermodynamicEquilibrium(seed_type=seed_type)
    hall_solver = mhd.HallSolver(
        gas_type=gas_type,
        seed_type=seed_type,
        transport_model=transport_model,
        ionization_model=ionization_model,
    )
    return hall_solver, gas_type


def operating_point() -> dict:
    """Default operating point (see Derivation.md's worked example)."""
    load_resistivity = 1.5 / 2 * 50 / 2 / 2 / 2 / 2 / 20  # Ohm * m
    channel_area = 48e-3 * 48e-3  # 2in * 2in in meters
    # channel_area = (8 / 100) * (8 / 100) # 1 cm x 1 cm
    channel_length = 0.2  # m

    return dict(
        num_slices=200,
        length=channel_length,
        area=channel_area,
        inlet_speed=150.115,          # m/s
        inlet_pressure=10.01e3,       # Pa
        inlet_gas_temperature=2000.0,  # K
        magnetic_field=0.5,           # T
        load_resistance=load_resistivity * channel_length / channel_area,
        inlet_seed_fraction=6.18e-3,
        # inlet_seed_fraction=0.00004166666667 * 1.12,
    )


def channel_to_dict(channel: mhd.Channel) -> dict:
    """Flatten a solved Channel into the axial-profile dict the plotting/summary helpers expect."""
    Jx = np.array([state.current_density[0] for state in channel.states])
    Jy = np.array([state.current_density[1] for state in channel.states])
    return dict(
        x=channel.x,
        u=channel["flow_speed"],
        Tp=channel["gas_temperature"],
        p=channel["gas_pressure"],
        np=channel["gas_number_density"],
        Te=channel["electron_temperature"],
        ne=channel["electron_number_density"],
        beta=channel["hall_parameter"],
        Jx=Jx,
        Jy=Jy,
        Ex=channel["axial_electric_field"],
        S_ohm=channel["ohmic_power_density"],
        S_load=channel["load_power_density"],
        eta_L=channel.load_resistivity,
    )


def main():
    hall_solver, gas_type = build_hall_solver()
    params = operating_point()
    channel_area = params["area"]

    t0 = time.perf_counter()
    channel = hall_solver.march(**params)
    print(time.perf_counter() - t0)
    # exit()

    out = channel_to_dict(channel)

    x = out["x"]
    Ex = out["Ex"]

    V = -np.trapezoid(Ex, x)  # volts
    A = channel_area

    Ix_profile = out["Jx"] * A
    Ix_in = Ix_profile[0]
    Ix_out = Ix_profile[-1]

    P = Ix_in * V

    print(f"Power: {P}")

    print(f"n_e = {out['ne'][0]:.3e}")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))

    plt.plot(out['x'], out['S_load'], label='Te', linestyle='--')
    plt.xlabel('x [m]'); plt.ylabel('Temperature [K]'); plt.legend(); plt.tight_layout(); plt.show()

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


def summarize_performance(out, A, cp, m_p, gamma):
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
    u_in, u_out = u[0],  u[-1]
    p_in, p_out = p[0],  p[-1]

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


if __name__ == "__main__":
    main()