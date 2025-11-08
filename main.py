import numpy as np
from scipy import constants as C
import matplotlib.pyplot as plt


m_p = 39.948 * 1.66053906660e-27   # Argon
m_e = C.electron_mass
gamma = 5.0 / 3.0
R = C.k / m_p
C_p = gamma / (gamma - 1.0) * R
eps_eV = 3.894
sigma_ep = 3.93994730526347e-21


def get_electron_temperature(primary_gas_temperature, delta_t):
    return primary_gas_temperature * (delta_t + 1.0)


def get_electron_density(Te, ns, eps_ion_eV=eps_eV, g_ratio=1.0):
    ns = np.asarray(ns, dtype=np.float64)
    Te = np.asarray(Te, dtype=np.float64)
    ns, Te = np.broadcast_arrays(ns, Te)

    eps_J = eps_ion_eV * C.e
    pref = (2.0 * np.pi * m_e * C.k * Te) / (C.h**2)
    pref = np.power(pref, 1.5)
    expo = np.exp(-eps_J / (C.k * Te))
    S = g_ratio * pref * expo

    disc = np.sqrt(S * (S + 4.0 * ns))
    ne = 0.5 * (disc - S)
    ne = np.clip(ne, 0.0, np.maximum(0.0, ns * (1.0 - 1e-12)))
    return ne


def get_frequencies(primary_gas_density, sigma_ep, electron_temperature):
    nu_M = primary_gas_density * sigma_ep * np.sqrt(2.0 * C.k * electron_temperature / m_e)
    nu_E = 2.0 * m_e / m_p * nu_M
    return nu_E, nu_M


def get_resistivity(nu_M, electron_density):
    return m_e * nu_M / (C.e**2 * electron_density)


def get_hall_parameter(magnetic_field, nu_M):
    return (C.e * magnetic_field) / (m_e * nu_M)


def get_mach_number(primary_gas_speed, primary_gas_temperature):
    return np.sqrt((m_p * primary_gas_speed**2) / (gamma * C.k * primary_gas_temperature))


def solve_plasma_properties(u, Tp, np_local, ns_local, B0, eta_L,
                            relax=0.5, max_iter=12):
    """
    Local micro/EM solve for a Hall generator slice (Ey=0, Jz=Ez=0, B=B0*ẑ).
    Returns dict with Te, ne, nuM, nuE, eta, beta, Z, Jx, Jy, Ex, S_ohm, S_load.
    """
    # initial guesses
    Te = Tp * 8.783
    ne = np.clip(1.0 * ns_local, 1e12, 1.0 * ns_local)

    for _ in range(max_iter):
        nu_E, nu_M = get_frequencies(np_local, sigma_ep, Te)
        eta = get_resistivity(nu_M, ne)
        beta = get_hall_parameter(B0, nu_M)
        Z = eta_L / eta

        denom = (beta**2 + 1.0 + Z)
        Jx = (beta**2 / denom) * C.e * ne * u
        Jy = -(beta * (1.0 + Z) / denom) * C.e * ne * u
        Ex_over_eta = -(beta**2 * Z / denom) * C.e * ne * u
        Ex = Ex_over_eta * eta

        J2 = Jx**2 + Jy**2
        S_ohm = eta * J2                  # W/m^3
        S_load = -Ex * Jx                 # W/m^3

        # Freidberg Eq. 3.14: ΔT = (5 M^2 / 9) * β^2(β^2+(1+Z)^2)/(β^2+1+Z)^2
        M = np.sqrt((3.0/5.0) * (m_p * u**2) / (C.k * Tp))
        DeltaT = (5.0 * M**2 / 9.0) * (beta**2 * (beta**2 + (1.0 + Z)**2)) / (denom**2)
        Te_new = Tp * (1.0 + DeltaT)

        # update ne via Saha at new Te
        ne_new = get_electron_density(Te_new, ns_local)

        # relax
        Te = relax * Te_new + (1.0 - relax) * Te
        ne = relax * ne_new + (1.0 - relax) * ne

    return dict(Te=Te, ne=ne, nuM=nu_M, nuE=nu_E, eta=eta, beta=beta, Z=Z,
                Jx=Jx, Jy=Jy, Ex=Ex, S_ohm=S_ohm, S_load=S_load)


def march_channel(num_slices,
                  L,
                  A,
                  u0,
                  p0,
                  Tp0,
                  B0,
                  R_L,
                  seed_frac0
  ):
    """
    Constant-area 1-D march with inner plasma solution per slice.
    Returns a dict of axial profiles.
    """
    x = np.linspace(0.0, L, num_slices)
    dx = L / max(1, (num_slices - 1))

    # distributed load resistivity: eta_L = R_L * A / L
    eta_L = R_L * A / L
    eta_L = 1.60

    # inlet state
    rho0 = p0 / (R * Tp0)
    np0 = rho0 / m_p
    ns0 = seed_frac0 * np0
    Phi_n = np0 * u0  # number flux = n_p u (constant for constant area)

    u = np.zeros_like(x)
    Tp = np.zeros_like(x)
    p = np.zeros_like(x)
    np_arr = np.zeros_like(x)
    ns_arr = np.zeros_like(x)
    Te = np.zeros_like(x)
    ne = np.zeros_like(x)
    beta = np.zeros_like(x)
    Z = np.zeros_like(x)
    Jx = np.zeros_like(x)
    Jy = np.zeros_like(x)
    Ex = np.zeros_like(x)
    S_ohm = np.zeros_like(x)
    S_load = np.zeros_like(x)

    # set inlet
    u[0] = u0
    Tp[0] = Tp0
    np_arr[0] = np0
    ns_arr[0] = ns0
    p[0] = np_arr[0] * C.k * Tp[0]

    for i in range(num_slices):
        # continuity: n_p = phi_n / u
        np_arr[i] = Phi_n / max(1e-6, u[i])
        ns_arr[i] = ns0 * (np_arr[i] / np0)
        p[i] = np_arr[i] * C.k * Tp[i]

        plasma = solve_plasma_properties(u[i], Tp[i], np_arr[i], ns_arr[i], B0, eta_L)
        Te[i] = plasma['Te']
        ne[i] = plasma['ne']
        beta[i] = plasma['beta']
        Z[i] = plasma['Z']
        Jx[i] = plasma['Jx']
        Jy[i] = plasma['Jy']
        Ex[i] = plasma['Ex']
        S_ohm[i] = plasma['S_ohm']
        S_load[i] = plasma['S_load']

        # march to next slice (skip after last)
        if i == num_slices - 1:
            break

        dTpdx = (S_ohm[i] - S_load[i]) / (m_p * np_arr[i] * u[i] * C_p)

        # couple momentum with dp/dx self-consistently (constant area)
        # p = phi_n kB Tp / u
        # dp/dx = phi_n kB[(1/u) dTp/dx - (Tp/u^2) du/dx]

        denom = (m_p * Phi_n + (Phi_n * C.k * Tp[i]) / (u[i]**2))
        dudx = (Jy[i] * B0 - (Phi_n * C.k / u[i]) * dTpdx) / denom

        # explicit Euler step
        Tp[i+1] = max(50.0, Tp[i] + dTpdx * dx)
        u[i+1] = max(1e-3, u[i] + dudx * dx)

    return dict(x=x, u=u, Tp=Tp, p=p, np=np_arr, ns=ns_arr,
                Te=Te, ne=ne, beta=beta, Z=Z, Jx=Jx, Jy=Jy, Ex=Ex,
                S_ohm=S_ohm, S_load=S_load, eta_L=eta_L)


def main():
    num_slices = 200

    inlet_primary_gas_temperature = 488  # K
    inlet_primary_gas_pressure = 8.01e5  # Pa
    inlet_primary_gas_speed = 735.115  # m/s
    magnetic_field = 8.0  # T
    load_resistance = 0.14  # Ohm
    channel_area = 4.0e-1  # 1 cm x 1 cm
    channel_length = 1.0  # m
    seed_gas_fraction = 6.18e-7

    out = march_channel(
        num_slices=num_slices,
        L=channel_length,
        A=channel_area,
        u0=inlet_primary_gas_speed,
        p0=inlet_primary_gas_pressure,
        Tp0=inlet_primary_gas_temperature,
        B0=magnetic_field,
        R_L=load_resistance,
        seed_frac0=seed_gas_fraction
    )

    print(f"n_e = {out['ne'][0]:.3e}")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))

    plt.plot(out['x'], out['S_load'], label='Te', linestyle='--')
    plt.xlabel('x [m]'); plt.ylabel('Temperature [K]'); plt.legend(); plt.tight_layout(); plt.show()

    perf = summarize_performance(out, A=channel_area, cp=C_p, m_p=m_p, gamma=gamma)

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

    plt.figure(figsize=(7,4))
    plt.plot(x, Tp, label=r"$T_p$", linewidth=2)
    plt.plot(x, Te, label=r"$T_e$", linestyle='--', linewidth=2)
    plt.xlabel("x [m]")
    plt.ylabel("Temperature [K]")
    plt.title("Electron vs. Primary Gas Temperature")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7,4))
    plt.plot(x, S_ohm, label=r"$S_{\Omega}$ (ohmic heating)", linewidth=2)
    plt.plot(x, S_load, label=r"$S_L$ (load power)", linewidth=2)
    plt.xlabel("x [m]")
    plt.ylabel("Power Density [W/m$^3$]")
    plt.title("Ohmic Heating vs Load Power")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    fig, ax1 = plt.subplots(figsize=(7,4))

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
    x     = out['x']
    Tp    = out['Tp']
    u     = out['u']
    nparr = out['np']          # primary number density
    p     = out['p']
    S_L   = out['S_load']
    S_Ohm = out['S_ohm']

    # inlet/outlet states
    Tp_in, Tp_out = Tp[0], Tp[-1]
    u_in,  u_out  = u[0],  u[-1]
    p_in,  p_out  = p[0],  p[-1]

    # mass flow (constant area)
    rho_in = m_p * nparr[0]
    mdot = rho_in * u_in * A

    # stagnation enthalpies (per unit mass)
    h0_in = cp*Tp_in + 0.5*u_in**2
    h0_out = cp*Tp_out + 0.5*u_out**2
    dh0 = h0_in - h0_out

    # Powers
    PL = np.trapezoid(S_L, x) * A
    POhm = np.trapezoid(S_Ohm, x) * A

    # Ratios
    enthalpy_extraction_ratio = dh0 / h0_in
    chi_load = (PL / mdot) / h0_in
    eta_electrical = PL / max(1e-30, (PL + POhm))  # delivered / extracted

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