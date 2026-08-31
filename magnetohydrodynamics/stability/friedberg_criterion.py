from __future__ import annotations

import numpy as np
from scipy import constants

from magnetohydrodynamics.stability.stability_model import StabilityModel
from magnetohydrodynamics.typing import Scalar


class FriedbergCriterion(StabilityModel):
    """The exact Velikhov-ionisation marginal-stability criterion, Friedberg (2025) eq.
    5.13, valid at any ionisation fraction f_I. See `FriedbergAsymptoticCriterion` for
    the high-ionisation (f_I -> 1) limit, eq. 6.23 -- the paper's headline practical
    result (nearly-full seed ionisation is what makes high-beta, i.e. high-field,
    operation stable).

    Diverges to +inf as Te -> Tp (Delta_T -> 0) or f_I -> 1: physically, no
    electron/gas temperature difference (nothing to drive the mode) or a fully-ionised
    seed (Saha's exponential sensitivity vanishes) are unconditionally stable. This is
    intentional -- callers should not clamp the result -- so the internal 1/Delta_T and
    1/(1-f_I) divisions are wrapped in `np.errstate` to suppress the resulting
    (harmless, expected) divide-by-zero warnings rather than treating them as
    numerical noise to hide.
    """

    def alpha(
            self,
            electron_temperature: Scalar,
            ionization_potential: Scalar,
            ionization_fraction: Scalar,
    ) -> Scalar:
        """Friedberg (4.3)/(7.1)'s alpha = (kTe / (3kTe + 2E_I)) * (2 - f_I)/(1 - f_I).

        Te : electron temperature [K]. E_I : seed ionisation potential [eV]. f_I : seed
        ionisation fraction n_e/n_s [-]. Note (5.13) itself is *stated* with the further
        simplified alpha = (kTe/2E_I)*(2-f_I)/(1-f_I), valid only when kTe/E_I << 1; that
        simplification is not always accurate (e.g. it's off by a factor of ~1.35 at the
        paper's own worked numeric example in Sec. 7.3, where kTe/E_I ~ 0.09, not <<1).
        This uses the more exact (4.3)/(7.1) form throughout -- it reduces to (5.13)'s
        simplified form whenever 3kTe << 2E_I anyway, and is materially more accurate when
        it doesn't. Public (not a private detail of `critical_hall_parameter`) because
        some callers want alpha itself, not just the derived beta_crit.
        """
        electron_temperature = np.asarray(electron_temperature, dtype=np.float64)
        ionization_fraction = np.asarray(ionization_fraction, dtype=np.float64)
        ionization_potential_j = np.asarray(ionization_potential, dtype=np.float64) * constants.e

        with np.errstate(divide="ignore", invalid="ignore"):
            return (
                constants.k * electron_temperature
                / (3.0 * constants.k * electron_temperature + 2.0 * ionization_potential_j)
            ) * ((2.0 - ionization_fraction) / (1.0 - ionization_fraction))

    def critical_hall_parameter(
            self,
            electron_temperature: Scalar,
            gas_temperature: Scalar,
            ionization_potential: Scalar,
            ionization_fraction: Scalar,
    ) -> Scalar:
        """beta_crit from the exact marginal stability criterion, Friedberg (5.13):

            beta_crit^2 = 4*alpha*(2 + 1/Delta_T) * [1 + alpha*(1 + 1/Delta_T)]
            Delta_T = Te/Tp - 1

        Returns beta_crit (not squared), directly comparable with Plasma.hall_parameter.
        """
        electron_temperature = np.asarray(electron_temperature, dtype=np.float64)
        gas_temperature = np.asarray(gas_temperature, dtype=np.float64)
        a = self.alpha(electron_temperature, ionization_potential, ionization_fraction)

        with np.errstate(divide="ignore", invalid="ignore"):
            delta_t = electron_temperature / gas_temperature - 1.0
            inv_delta_t = 1.0 / delta_t
            beta_crit_sq = 4.0 * a * (2.0 + inv_delta_t) * (1.0 + a * (1.0 + inv_delta_t))
            return np.sqrt(beta_crit_sq)


class FriedbergAsymptoticCriterion(StabilityModel):
    """The high-ionisation asymptotic limit of the marginal-stability criterion,
    Friedberg (2025) eq. 6.23:

        beta_crit = sqrt(2) * (kTe/E_I) / (1 - f_I)

    Valid for f_I -> 1; unlike `FriedbergCriterion`, this has no Tp/Delta_T dependence
    (dropped in this limit's derivation) -- `gas_temperature` is still accepted, to
    satisfy `StabilityModel`'s interface and let callers treat both criteria
    identically, but is ignored. Diverges to +inf as f_I -> 1, for the same reason as
    `FriedbergCriterion` -- see its docstring.
    """

    def critical_hall_parameter(
            self,
            electron_temperature: Scalar,
            gas_temperature: Scalar,
            ionization_potential: Scalar,
            ionization_fraction: Scalar,
    ) -> Scalar:
        electron_temperature = np.asarray(electron_temperature, dtype=np.float64)
        ionization_fraction = np.asarray(ionization_fraction, dtype=np.float64)
        ionization_potential_j = np.asarray(ionization_potential, dtype=np.float64) * constants.e

        with np.errstate(divide="ignore", invalid="ignore"):
            return np.sqrt(2.0) * (constants.k * electron_temperature / ionization_potential_j) / (
                1.0 - ionization_fraction
            )
