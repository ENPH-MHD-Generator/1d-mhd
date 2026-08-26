from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import constants


def solve_local_hall_slice(
    self,
    B0: float,
    eta_L: float,
    relax: float = 0.5,
    max_iter: int = 12,
    *,
    force_eta: Optional[float] = None,
) -> "Plasma":
    """
    Local plasma properties solver for a Hall generator slice (Ey=0, Jz=Ez=0, B=B0).
    Updates this Plasma instance in-place and returns self.

    NOTE: Your original code overwrote eta with `eta = 1/500`. To preserve that behavior,
    use `force_eta=1/500` (or keep it None if you want the computed eta).
    """
    # initial guesses (same as your code)
    Te = self.Tp * 8.783
    ns_local = self.n_s

    ne = np.clip(1.0 * ns_local, 1e12, 1.0 * ns_local)

    for _ in range(max_iter):
        nu_E, nu_M = self.get_frequencies_for_Te(Te)
        eta = self.resistivity(nu_M, ne)

        if force_eta is not None:
            eta = force_eta

        beta = self.hall_parameter(B0, nu_M)
        Z = eta_L / eta

        denom = (beta**2 + 1.0 + Z)
        Jx = (beta**2 / denom) * C.e * ne * self.u
        Jy = -(beta * (1.0 + Z) / denom) * C.e * ne * self.u
        Ex_over_eta = -(beta**2 * Z / denom) * C.e * ne * self.u
        Ex = Ex_over_eta * eta

        J2 = Jx**2 + Jy**2
        S_ohm = eta * J2
        S_load = -Ex * Jx

        M = np.sqrt((3.0 / 5.0) * (self.gas_type.m_particle * self.u**2) / (C.k * self.Tp))
        DeltaT = (5.0 * M**2 / 9.0) * (beta**2 * (beta**2 + (1.0 + Z)**2)) / (denom**2)
        Te_new = self.Tp * (1.0 + DeltaT)

        ne_new = self.seed_type.saha_electron_density(Te_new, ns_local)

        Te = relax * Te_new + (1.0 - relax) * Te
        ne = relax * ne_new + (1.0 - relax) * ne

    # store results
    self.Te = float(Te)
    self.ne = float(ne)
    self.nuE = float(nu_E)
    self.nu_M = float(nu_M)  # keep name parity if you want; but we store also nuM below
    self.nuE = float(nu_E)
    self.nuM = float(nu_M)
    self.eta = float(eta)
    self.beta = float(beta)
    self.Z = float(Z)

    self.Jx = float(Jx)
    self.Jy = float(Jy)
    self.Ex = float(Ex)

    self.S_ohm = float(S_ohm)
    self.S_load = float(S_load)

    return self


# -----------------------------
# Marcher: keeps your 1-D logic
# -----------------------------

def march_channel(
    num_slices: int,
    L: float,
    A: float,
    u0: float,
    p0: float,
    Tp0: float,
    B0: float,
    R_L: float,
    seed_frac0: float,
    gas_type: GasType,
    seed_type: SeedType,
    *,
    relax: float = 0.5,
    max_iter: int = 12,
    force_eta: Optional[float] = 1.0 / 500.0,  # matches your original eta override by default
) -> Dict[str, np.ndarray]:
    """
    Constant-area 1-D march with inner plasma solution per slice.
    Returns the same kind of dict of axial profiles you had, but the state is managed by Plasma objects.
    """
    x = np.linspace(0.0, L, num_slices)
    dx = L / max(1, (num_slices - 1))

    eta_L = R_L * A / L

    # inlet number density from ideal gas
    n_p0 = p0 / (C.k * Tp0)
    Phi_n = n_p0 * u0  # number flux n_p u (constant for constant area)

    # storage arrays
    u = np.zeros_like(x)
    Tp = np.zeros_like(x)
    p = np.zeros_like(x)
    n_p = np.zeros_like(x)
    n_s = np.zeros_like(x)

    Te = np.zeros_like(x)
    ne = np.zeros_like(x)
    beta = np.zeros_like(x)
    Z = np.zeros_like(x)
    Jx = np.zeros_like(x)
    Jy = np.zeros_like(x)
    Ex = np.zeros_like(x)
    S_ohm = np.zeros_like(x)
    S_load = np.zeros_like(x)

    # inlet
    u[0] = u0
    Tp[0] = Tp0
    n_p[0] = n_p0
    n_s[0] = seed_frac0 * n_p0
    p[0] = n_p[0] * C.k * Tp[0]

    for i in range(num_slices):
        # update primary density from constant number flux and current u
        n_p[i] = Phi_n / max(1e-6, u[i])
        n_s[i] = (seed_frac0 * n_p0) * (n_p[i] / n_p0)  # same scaling as your code
        p[i] = n_p[i] * C.k * Tp[i]

        slice_state = Plasma(
            gas_type=gas_type,
            seed_type=seed_type,
            Tp=float(Tp[i]),
            p=float(p[i]),
            u=float(u[i]),
            n_p=float(n_p[i]),
            seed_frac=float(n_s[i] / max(1e-30, n_p[i])),
        )

        slice_state.solve_local_hall_slice(
            B0=B0,
            eta_L=eta_L,
            relax=relax,
            max_iter=max_iter,
            force_eta=force_eta,
        )

        Te[i] = slice_state.Te
        ne[i] = slice_state.ne
        beta[i] = slice_state.beta
        Z[i] = slice_state.Z
        Jx[i] = slice_state.Jx
        Jy[i] = slice_state.Jy
        Ex[i] = slice_state.Ex
        S_ohm[i] = slice_state.S_ohm
        S_load[i] = slice_state.S_load

        if i == num_slices - 1:
            break

        # march equations (unchanged)
        dTpdx = (S_ohm[i] - S_load[i]) / (gas_type.m_particle * n_p[i] * u[i] * gas_type.cp)
        denom = (gas_type.m_particle * Phi_n + (Phi_n * C.k * Tp[i]) / (u[i] ** 2))
        dudx = (Jy[i] * B0 - (Phi_n * C.k / u[i]) * dTpdx) / denom

        Tp[i + 1] = max(50.0, Tp[i] + dTpdx * dx)
        u[i + 1] = max(1e-3, u[i] + dudx * dx)

    return dict(
        x=x, u=u, Tp=Tp, p=p, np=n_p, ns=n_s,
        Te=Te, ne=ne, beta=beta, Z=Z, Jx=Jx, Jy=Jy, Ex=Ex,
        S_ohm=S_ohm, S_load=S_load, eta_L=np.full_like(x, eta_L),
    )
