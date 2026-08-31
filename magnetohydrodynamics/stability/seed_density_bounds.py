from __future__ import annotations

import numpy as np
from scipy import constants
from scipy.optimize import brentq

from magnetohydrodynamics.ionization.saha_ionization import LocalThermodynamicEquilibrium
from magnetohydrodynamics.ionization.seed_type import SeedType
from magnetohydrodynamics.stability.friedberg_criterion import FriedbergAsymptoticCriterion
from magnetohydrodynamics.thermophysics.gas_type import GasType
from magnetohydrodynamics.transport.mhd_transport_model import MHDTransportModel


class SeedDensityBounds:
    """Closed-form / brentq-based bounds on seed density from Friedberg's two design
    constraints (marginal stability and a target load power density), holding Te
    prescribed rather than solved via HallSolver's self-consistent equilibrium (see
    each method's docstring for why that distinction matters -- it's exactly what
    hides these bounds from EquilibriumSweep's self-consistent sweeps).

    Bundles `gas_type`/`seed_type` and builds their ionisation/transport/criterion
    models once, rather than re-constructing them on every call -- `ceiling` doesn't
    actually need `gas_type` (only `min_max_window` does, for the transport model), but
    both live on the same operating condition (a fixed seed species in a fixed primary
    gas) so one object holding both is simpler than splitting them further."""

    def __init__(self, gas_type: GasType, seed_type: SeedType):
        self._seed_type = seed_type
        self._ionization_model = LocalThermodynamicEquilibrium(seed_type=seed_type)
        self._transport_model = MHDTransportModel(seed_type=seed_type, gas_type=gas_type)
        self._criterion = FriedbergAsymptoticCriterion()

    def _ceiling_margin_minus_level(self, seed_number_density: float, electron_temperature: float, beta: float, level: float) -> float:
        """Shared by `ceiling` and `min_max_window`: at fixed Te and beta,
        beta_crit_asymptotic(n_s)/beta - level. `gas_temperature` is passed as NaN --
        FriedbergAsymptoticCriterion ignores it, and NaN makes that explicit rather
        than passing a real-looking value that isn't actually used. `level`
        generalizes past marginal stability (1.0, the default) the same way
        EquilibriumSweep.margin_minus_level does -- see its docstring."""
        f_I = self._ionization_model.get_electron_density(electron_temperature, seed_number_density) / seed_number_density
        beta_crit = self._criterion.critical_hall_parameter(electron_temperature, np.nan, self._seed_type.ionization_potential, f_I)
        return float(beta_crit) / beta - level

    def ceiling(
            self, electron_temperature_values: np.ndarray, beta_values: np.ndarray, reference_gas_number_density: float,
            level: float = 1.0, ns_bracket: tuple[float, float] = (1e14, 1e28),
    ) -> np.ndarray:
        """Reproduces Friedberg's (6.27)-style *ceiling* directly: holding Te fixed
        (prescribed here, NOT solved via the energy-balance equilibrium HallSolver
        uses) and beta fixed as an external design target, finds the largest seed
        density n_s for which the high-ionisation asymptotic criterion (6.23) still
        holds. Verified numerically: at fixed Te, beta_crit_asymptotic decreases
        monotonically as n_s increases (more seed atoms dilutes f_I away from 1), so
        there's a real ceiling and no matching floor from the stability criterion
        alone -- consistent with (6.27) itself, and with there being no
        stability-side reason for a *minimum* seed density (Friedberg's stated
        minimum comes from a separate design constraint, hitting a target load power
        density S_C -- see `min_max_window`).

        This deliberately does NOT go through HallSolver.solve_equilibrium. That
        iterative solve lets Te float to whatever the energy balance demands, which is
        exactly why a self-consistent sweep (e.g.
        `EquilibriumSweep.critical_load_resistivity_surface`) never shows this
        ceiling -- verified numerically: at any seed fraction there, the critical
        point's Te climbs right alongside seed fraction (roughly 4,600 K to 11,200 K
        across that surface's whole range), always staying just far enough above the
        ceiling to keep f_I within reach of 1. Friedberg's (6.27) instead treats Te as
        prescribed rather than solved for, so reproducing it means doing the same
        here.

        Returns ceiling seed density expressed as a fraction of
        `reference_gas_number_density` -- purely for display alongside other
        seed-fraction plots; this analysis has no actual primary-gas dependence (Tp,
        p0 don't enter it at all)."""
        shape = (len(electron_temperature_values), len(beta_values))
        ceiling_ns = np.full(shape, np.nan)
        lo, hi = ns_bracket
        for i, Te in enumerate(electron_temperature_values):
            for j, beta in enumerate(beta_values):
                try:
                    if np.sign(self._ceiling_margin_minus_level(lo, Te, beta, level)) != np.sign(self._ceiling_margin_minus_level(hi, Te, beta, level)):
                        ceiling_ns[i, j] = brentq(self._ceiling_margin_minus_level, lo, hi, args=(Te, beta, level), xtol=1.0)
                except (ValueError, RuntimeError):
                    pass
        return ceiling_ns / reference_gas_number_density

    def min_max_window(
            self,
            electron_temperature_values: np.ndarray, beta_values: np.ndarray,
            reference_gas_number_density: float, reference_flow_speed: float,
            target_power_density: float = 100e6, level: float = 1.0,
            ns_bracket: tuple[float, float] = (1e14, 1e28),
    ) -> tuple[np.ndarray, np.ndarray]:
        """The 3-D generalisation of `ceiling`: MAXIMUM seed density (the same eq.
        6.23/6.27 stability ceiling) alongside a genuine MINIMUM, both as surfaces
        over (Te, beta).

        The maximum comes from stability alone, exactly as in `ceiling`. The minimum
        comes from Friedberg's *other* design constraint: delivering at least an
        "industrially relevant" load power density S_C (~100 MW/m^3 is the paper's own
        figure, see Sec. 2/7), combined with the same matched-load choice of Z used
        elsewhere (Z = sqrt(1+beta^2), eq. 6.10). At that Z, with s = sqrt(1+beta^2),
        S_L simplifies to S_L = m_e n_e nu_M(Te) v_p^2 beta^4 / [s(s+1)^2] -- still
        linear in n_e, so hitting S_C requires n_e >= S_C * s(s+1)^2 / (m_e nu_M v_p^2
        beta^4). This is closed-form algebra, not a root-find, and -- like the
        ceiling -- never touches HallSolver.solve_equilibrium; nu_M and v_p come from a
        REFERENCE operating point (S_C is an absolute power density, not a ratio, so
        it needs *some* concrete n_p, v_p to anchor to). Callers should generally pass
        Friedberg's own Sec. 7.1 reference conditions here, not an arbitrary/dilute
        one: with a much more dilute, slower reference, the region where
        max_ns > min_ne (i.e. where both constraints can be satisfied at once) can
        shrink to a sliver invisible at plot resolution.

        Together these two independently-derived bounds reproduce exactly what
        Friedberg's Introduction describes: "the combination of the marginal
        stability criterion plus the practical requirement of an industrial relevant
        load power density...represent two design constraints. Their simultaneous
        solution leads to specific values for the electron temperature and seed
        density." Returns (max_ns, min_ne), both as absolute number densities [m^-3]
        (divide by a reference n_p for a seed-fraction-like ratio)."""
        shape = (len(electron_temperature_values), len(beta_values))
        max_ns = np.full(shape, np.nan)
        min_ne = np.full(shape, np.nan)
        lo, hi = ns_bracket
        for i, Te in enumerate(electron_temperature_values):
            nu_M = self._transport_model.get_momentum_transfer_frequency(Te, reference_gas_number_density)
            for j, beta in enumerate(beta_values):
                try:
                    if np.sign(self._ceiling_margin_minus_level(lo, Te, beta, level)) != np.sign(self._ceiling_margin_minus_level(hi, Te, beta, level)):
                        max_ns[i, j] = brentq(self._ceiling_margin_minus_level, lo, hi, args=(Te, beta, level), xtol=1.0)
                except (ValueError, RuntimeError):
                    pass
                s = np.sqrt(1.0 + beta ** 2)
                min_ne[i, j] = (
                    target_power_density * s * (s + 1.0) ** 2
                    / (constants.electron_mass * nu_M * reference_flow_speed ** 2 * beta ** 4)
                )
        return max_ns, min_ne
