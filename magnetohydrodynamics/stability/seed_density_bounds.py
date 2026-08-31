from __future__ import annotations

import numpy as np
from scipy import constants

from magnetohydrodynamics.ionization.saha_ionization import LocalThermodynamicEquilibrium
from magnetohydrodynamics.ionization.seed_type import SeedType
from magnetohydrodynamics.stability.friedberg_criterion import FriedbergAsymptoticCriterion
from magnetohydrodynamics.thermophysics.gas_type import GasType
from magnetohydrodynamics.transport.mhd_transport_model import MHDTransportModel
from magnetohydrodynamics.typing import Scalar


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

    def _ceiling_margin_minus_level(self, seed_number_density: Scalar, electron_temperature: Scalar, beta: Scalar, level: float) -> Scalar:
        """Shared by `ceiling`/`min_max_window` (via `_bisect_ceiling`): at fixed Te and
        beta, beta_crit_asymptotic(n_s)/beta - level. `gas_temperature` is passed as
        NaN -- FriedbergAsymptoticCriterion ignores it, and NaN makes that explicit
        rather than passing a real-looking value that isn't actually used. `level`
        generalizes past marginal stability (1.0, the default) the same way
        EquilibriumSweep.margin_minus_level does -- see its docstring. Elementwise, so
        `seed_number_density`/`electron_temperature`/`beta` may be scalars or
        broadcastable arrays -- `_bisect_ceiling` calls this with whole grids."""
        f_I = self._ionization_model.get_electron_density(electron_temperature, seed_number_density) / seed_number_density
        beta_crit = self._criterion.critical_hall_parameter(electron_temperature, np.nan, self._seed_type.ionization_potential, f_I)
        return beta_crit / beta - level

    def _bisect_ceiling(
            self, electron_temperature_values: np.ndarray, beta_values: np.ndarray,
            level: float, ns_bracket: tuple[float, float], iters: int = 60,
    ) -> np.ndarray:
        """Vectorized log-space bisection for the seed-density ceiling, over the full
        (Te, beta) grid at once -- the same approach as
        EquilibriumSweep.matched_load, applicable here for the same reason: at fixed
        (Te, beta), `_ceiling_margin_minus_level` has (verified numerically, see
        `ceiling`'s docstring) exactly one root in n_s, since beta_crit_asymptotic(n_s)
        is monotonic. This used to be a per-(Te, beta) Python loop calling
        `scipy.optimize.brentq` once per cell (with a try/except RuntimeError/ValueError
        around each) -- profiled as this module's dominant cost (~1000s of scalar
        HallSolver-adjacent calls for a modest grid) well out of proportion to the
        actual amount of math involved; one vectorized bisection over the whole grid
        removes essentially all of that Python-level overhead.

        Unlike `matched_load`, this doesn't assume which endpoint is positive (that
        wasn't asserted anywhere for this criterion) -- each bisection step compares
        the midpoint's sign against `ns_bracket[0]`'s own sign instead of a hardcoded
        one, so it stays correct regardless of whether the root-finder function happens
        to be increasing or decreasing in n_s.

        Returns a 2-D array of shape (len(electron_temperature_values),
        len(beta_values)), NaN wherever `ns_bracket`'s endpoints don't actually
        sign-change (checked once, vectorized, in place of the old per-cell
        try/except)."""
        Te, beta = np.meshgrid(electron_temperature_values, beta_values, indexing="ij")
        lo, hi = ns_bracket
        f_lo = self._ceiling_margin_minus_level(np.full_like(Te, lo), Te, beta, level)
        f_hi = self._ceiling_margin_minus_level(np.full_like(Te, hi), Te, beta, level)
        valid = np.sign(f_lo) != np.sign(f_hi)
        positive_at_lo = f_lo > 0.0

        log_lo = np.full_like(Te, np.log10(lo))
        log_hi = np.full_like(Te, np.log10(hi))
        for _ in range(iters):
            log_mid = 0.5 * (log_lo + log_hi)
            ns_mid = 10.0 ** log_mid
            positive = self._ceiling_margin_minus_level(ns_mid, Te, beta, level) > 0.0
            go_high = positive == positive_at_lo  # sign(mid) matches sign(lo) => root is in [mid, hi]
            log_lo = np.where(go_high, log_mid, log_lo)
            log_hi = np.where(go_high, log_hi, log_mid)
        return np.where(valid, 10.0 ** (0.5 * (log_lo + log_hi)), np.nan)

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
        ceiling_ns = self._bisect_ceiling(electron_temperature_values, beta_values, level, ns_bracket)
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
        max_ns = self._bisect_ceiling(electron_temperature_values, beta_values, level, ns_bracket)

        Te, beta = np.meshgrid(electron_temperature_values, beta_values, indexing="ij")
        # nu_M depends on Te only -- broadcasting it across the full (Te, beta) grid (rather
        # than computing it once per Te row, as the old per-cell loop did) is harmless.
        nu_M = self._transport_model.get_momentum_transfer_frequency(Te, reference_gas_number_density)
        s = np.sqrt(1.0 + beta ** 2)
        min_ne = (
            target_power_density * s * (s + 1.0) ** 2
            / (constants.electron_mass * nu_M * reference_flow_speed ** 2 * beta ** 4)
        )
        return max_ns, min_ne
