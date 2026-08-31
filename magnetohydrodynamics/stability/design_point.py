from __future__ import annotations

import numpy as np
from scipy import constants
from scipy.optimize import brentq

from magnetohydrodynamics.ionization.saha_ionization import LocalThermodynamicEquilibrium
from magnetohydrodynamics.ionization.seed_type import SeedType
from magnetohydrodynamics.stability.friedberg_criterion import FriedbergCriterion
from magnetohydrodynamics.thermophysics.gas_type import GasType
from magnetohydrodynamics.transport.mhd_transport_model import MHDTransportModel


class MarginalDesignPointSolver:
    """Friedberg's Sec. 7.2 iterative design procedure (eq. 7.1-7.2): for a given B0
    and target converted power density S_C, finds the electron temperature at which
    the plasma sits EXACTLY at marginal stability while delivering EXACTLY S_C -- the
    paper's two simultaneous design constraints (marginal stability + an
    "industrially relevant" load power density, see its Introduction), solved
    together for the one unknown, Te.

    Bundles the primary-gas/seed species and the fixed inlet condition
    (gas_temperature, gas_number_density, flow_speed, mach_number -- e.g. Friedberg's
    own Sec. 7.1 reference furnace: Tp=481 K, pp=0.801 MPa, vp=735 m/s, M=1.8) that
    `solve`/`sweep` both hold fixed while varying B0 and/or the target power density --
    this class has no built-in default reference point, callers supply one explicitly
    at construction."""

    def __init__(
            self, gas_type: GasType, seed_type: SeedType,
            gas_temperature: float, gas_number_density: float, flow_speed: float, mach_number: float,
    ):
        self._seed_type = seed_type
        self._gas_temperature = gas_temperature
        self._gas_number_density = gas_number_density
        self._flow_speed = flow_speed
        self._mach_number = mach_number
        self._ionization_model = LocalThermodynamicEquilibrium(seed_type=seed_type)
        self._transport_model = MHDTransportModel(seed_type=seed_type, gas_type=gas_type)
        self._criterion = FriedbergCriterion()

    def _evaluate(self, electron_temperature: float, magnetic_field: float, target_power_density: float) -> dict:
        delta_t = electron_temperature / self._gas_temperature - 1.0
        nu_M = self._transport_model.get_momentum_transfer_frequency(electron_temperature, self._gas_number_density)
        beta = constants.e * magnetic_field / (constants.electron_mass * nu_M)

        # Energy balance (eq. 3.14, solved for Z given beta and Delta_T -- eq. 7.1's mu/xi/Z group).
        mu = (9.0 / (5.0 * self._mach_number ** 2 * beta ** 2)) * delta_t
        xi = beta ** 2 * mu / (1.0 - mu) - 1.0
        Z = xi + np.sqrt(xi * (xi + 1.0) / mu)

        # Power density (eq. 3.12, S_L = target_power_density, inverted for n_e).
        n_e = ((beta ** 2 + 1.0 + Z) / (beta ** 2 * (1.0 + Z))) * target_power_density / (
            constants.electron_mass * nu_M * self._flow_speed ** 2
        )

        # Saha, rearranged (eq. 7.1): 1-f_I = r/(1+r), r = n_e/N(Te).
        r = n_e / self._ionization_model.saha_coefficient(electron_temperature)
        one_minus_f_I = r / (1.0 + r)
        ionization_fraction = 1.0 - one_minus_f_I

        beta_crit = self._criterion.critical_hall_parameter(electron_temperature, self._gas_temperature, self._seed_type.ionization_potential, ionization_fraction)

        # Figures of merit (eq. 6.9 for the ratio; sigma = 1/eta, e^2 n_e/(me nu_M)).
        ohmic_to_load_ratio = (beta ** 2 + (1.0 + Z) ** 2) / (beta ** 2 * Z)
        conductivity = constants.e ** 2 * n_e / (constants.electron_mass * nu_M)

        return dict(
            electron_temperature=electron_temperature, delta_t=delta_t, nu_M=nu_M, beta=beta, Z=Z,
            n_e=n_e, ionization_fraction=ionization_fraction, one_minus_f_I=one_minus_f_I,
            beta_crit=beta_crit, ohmic_to_load_ratio=ohmic_to_load_ratio, conductivity=conductivity,
        )

    def solve(
            self, magnetic_field: float, target_power_density: float,
            te_scan_range: tuple[float, float] = (600.0, 2.0e5),
            te_scan_points: int = 80,
    ) -> dict:
        """Reuses `FriedbergCriterion.critical_hall_parameter` directly for the
        marginal-stability side of the equation (eq. 5.13, with the more-precise
        (4.3)/(7.1) alpha that class already uses) -- no stability algebra is
        reimplemented here, only the electromagnetic/energy-balance/Saha algebra (eq.
        7.1) that turns a Te guess into a full equilibrium is specific to this class.
        Root-found via brentq rather than the paper's own under-relaxed fixed-point
        iteration -- more robust, and this codebase already uses brentq elsewhere for
        the same reason.

        Returns a dict: electron_temperature, delta_t, nu_M, beta, Z, n_e,
        ionization_fraction (f_I), one_minus_f_I, beta_crit, ohmic_to_load_ratio
        (S_Omega/S_L, eq. 6.9), conductivity (S/m)."""

        def residual(electron_temperature: float) -> float:
            result = self._evaluate(electron_temperature, magnetic_field, target_power_density)
            return float(result["beta"] - result["beta_crit"])

        # eq. 7.1's closed-form Z (a quadratic root) has a real discriminant only once
        # Te is enough above Tp -- xi*(xi+1)/mu goes negative for Te too close to Tp (a
        # genuine domain restriction of that formula, not a numerical issue), so a
        # fixed low bracket endpoint can land in that invalid region and return NaN.
        # Scanning with np.errstate-suppressed warnings and skipping non-finite points
        # finds the first actually-valid bracket instead of guessing a safe constant.
        candidates = np.logspace(np.log10(te_scan_range[0]), np.log10(te_scan_range[1]), te_scan_points)
        with np.errstate(invalid="ignore", divide="ignore"):
            residuals = np.array([residual(te) for te in candidates])
        finite = np.isfinite(residuals)
        sign_changes = np.where(finite[:-1] & finite[1:] & (np.sign(residuals[:-1]) != np.sign(residuals[1:])))[0]
        if len(sign_changes) == 0:
            raise ValueError(
                f"No valid sign change in beta-beta_crit found scanning Te in {te_scan_range} for "
                f"B0={magnetic_field}, S_C={target_power_density}, seed={self._seed_type.name} -- widen te_scan_range."
            )
        lo, hi = candidates[sign_changes[0]], candidates[sign_changes[0] + 1]
        converged_te = brentq(residual, lo, hi, xtol=1e-2)
        return self._evaluate(converged_te, magnetic_field, target_power_density)

    def sweep(self, b0_values: np.ndarray, target_power_density: float) -> dict[str, np.ndarray]:
        """`solve` at every B0 in b0_values, at this solver's fixed target power
        density and inlet condition. Returns a dict of arrays, one entry per
        quantity."""
        results = [self.solve(float(b0), target_power_density) for b0 in b0_values]
        return {key: np.array([result[key] for result in results]) for key in results[0]}
