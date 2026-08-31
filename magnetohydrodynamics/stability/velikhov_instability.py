"""
Marginal stability criterion for the Velikhov-ionisation instability (Friedberg 2025,
"The Velikhov-ionisation instability revisited", J. Plasma Phys. 91, E115 -- PDF at repo root).

A plasma slice is unstable to short-wavelength ionisation perturbations once its Hall
parameter beta exceeds a critical value beta_crit that depends only on the electron/gas
temperature ratio and the seed ionisation fraction. This module implements two forms:

- `critical_hall_parameter` -- the exact criterion (Friedberg 5.13), valid at any f_I.
- `critical_hall_parameter_asymptotic` -- the high-ionisation limit (Friedberg 6.23),
  valid as f_I -> 1; this is the paper's headline practical result (nearly-full seed
  ionisation is what makes high-beta, i.e. high-field, operation stable).

Both diverge to +inf as Te -> Tp (Delta_T -> 0) or f_I -> 1: physically, no electron/gas
temperature difference (nothing to drive the mode) or a fully-ionised seed (Saha's
exponential sensitivity vanishes) are unconditionally stable. This is intentional --
callers should not clamp the result -- so the internal 1/Delta_T and 1/(1-f_I) divisions
are wrapped in `np.errstate` to suppress the resulting (harmless, expected) divide-by-zero
warnings rather than treating them as numerical noise to hide.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import constants

from magnetohydrodynamics.plasma.plasma import Plasma


def alpha(
        electron_temperature: float | Iterable,
        ionization_potential: float | Iterable,
        ionization_fraction: float | Iterable,
) -> float | Iterable:
    """Friedberg (4.3)/(7.1)'s alpha = (kTe / (3kTe + 2E_I)) * (2 - f_I)/(1 - f_I).

    Te : electron temperature [K]. E_I : seed ionisation potential [eV]. f_I : seed
    ionisation fraction n_e/n_s [-]. Note (5.13) itself is *stated* with the further
    simplified alpha = (kTe/2E_I)*(2-f_I)/(1-f_I), valid only when kTe/E_I << 1; that
    simplification is not always accurate (e.g. it's off by a factor of ~1.35 at the
    paper's own worked numeric example in Sec. 7.3, where kTe/E_I ~ 0.09, not <<1).
    This module uses the more exact (4.3)/(7.1) form throughout -- it reduces to (5.13)'s
    simplified form whenever 3kTe << 2E_I anyway, and is materially more accurate when it
    doesn't.
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
        electron_temperature: float | Iterable,
        gas_temperature: float | Iterable,
        ionization_potential: float | Iterable,
        ionization_fraction: float | Iterable,
) -> float | Iterable:
    """beta_crit from the exact marginal stability criterion, Friedberg (5.13):

        beta_crit^2 = 4*alpha*(2 + 1/Delta_T) * [1 + alpha*(1 + 1/Delta_T)]
        Delta_T = Te/Tp - 1

    Returns beta_crit (not squared), directly comparable with Plasma.hall_parameter.
    Diverges to +inf as Te -> Tp or f_I -> 1 (both unconditionally stable) -- see module
    docstring.
    """
    electron_temperature = np.asarray(electron_temperature, dtype=np.float64)
    gas_temperature = np.asarray(gas_temperature, dtype=np.float64)
    a = alpha(electron_temperature, ionization_potential, ionization_fraction)

    with np.errstate(divide="ignore", invalid="ignore"):
        delta_t = electron_temperature / gas_temperature - 1.0
        inv_delta_t = 1.0 / delta_t
        beta_crit_sq = 4.0 * a * (2.0 + inv_delta_t) * (1.0 + a * (1.0 + inv_delta_t))
        return np.sqrt(beta_crit_sq)


def critical_hall_parameter_asymptotic(
        electron_temperature: float | Iterable,
        ionization_potential: float | Iterable,
        ionization_fraction: float | Iterable,
) -> float | Iterable:
    """beta_crit from the high-ionisation asymptotic limit, Friedberg (6.23):

        beta_crit = sqrt(2) * (kTe/E_I) / (1 - f_I)

    Valid for f_I -> 1; unlike the exact form, this has no Tp/Delta_T dependence (dropped
    in this limit's derivation). Diverges to +inf as f_I -> 1.
    """
    electron_temperature = np.asarray(electron_temperature, dtype=np.float64)
    ionization_fraction = np.asarray(ionization_fraction, dtype=np.float64)
    ionization_potential_j = np.asarray(ionization_potential, dtype=np.float64) * constants.e

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(2.0) * (constants.k * electron_temperature / ionization_potential_j) / (
            1.0 - ionization_fraction
        )


def stability_margin(
        hall_parameter: float | Iterable,
        critical_hall_parameter: float | Iterable,
) -> float | Iterable:
    """critical/actual: > 1 stable, < 1 unstable, == 1 marginal."""
    return np.asarray(critical_hall_parameter, dtype=np.float64) / np.asarray(hall_parameter, dtype=np.float64)


def is_stable(
        hall_parameter: float | Iterable,
        critical_hall_parameter: float | Iterable,
) -> bool | Iterable:
    """hall_parameter <= critical_hall_parameter."""
    return np.asarray(hall_parameter) <= np.asarray(critical_hall_parameter)


def plasma_alpha(plasma: Plasma) -> float:
    return alpha(
        plasma.electron_temperature, plasma.seed_type.ionization_potential, plasma.ionization_fraction,
    )


def plasma_critical_hall_parameter(plasma: Plasma) -> float:
    return critical_hall_parameter(
        plasma.electron_temperature, plasma.gas_temperature,
        plasma.seed_type.ionization_potential, plasma.ionization_fraction,
    )


def plasma_critical_hall_parameter_asymptotic(plasma: Plasma) -> float:
    return critical_hall_parameter_asymptotic(
        plasma.electron_temperature, plasma.seed_type.ionization_potential, plasma.ionization_fraction,
    )


def plasma_stability_margin(plasma: Plasma) -> float:
    return stability_margin(plasma.hall_parameter, plasma_critical_hall_parameter(plasma))


def is_plasma_stable(plasma: Plasma) -> bool:
    return is_stable(plasma.hall_parameter, plasma_critical_hall_parameter(plasma))
