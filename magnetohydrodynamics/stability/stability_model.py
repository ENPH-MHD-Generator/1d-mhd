from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np

from magnetohydrodynamics.plasma.plasma import Plasma


class StabilityModel(ABC):
    """Interface for a Velikhov-ionisation marginal-stability criterion.

    A plasma slice is unstable to short-wavelength ionisation perturbations once its
    Hall parameter beta exceeds a critical value beta_crit that a concrete criterion
    computes from the local electron/gas temperature and the seed ionisation fraction.
    Friedberg (2025, "The Velikhov-ionisation instability revisited") derives two such
    criteria -- see friedberg_criterion.py's `FriedbergCriterion` (the exact result, eq.
    5.13, valid at any ionisation fraction) and `FriedbergAsymptoticCriterion` (the
    high-ionisation limit, eq. 6.23, the paper's headline practical result) -- both
    implementing this same interface, which is what lets the sweep/search code
    elsewhere in this package (`equilibrium_sweep`, `boundary_search`,
    `seed_density_bounds`, ...) evaluate either one identically.

    `stability_margin`/`is_stable`/the `plasma_*` convenience wrappers are concrete
    here (not abstract) -- they're pure comparisons built on whatever
    `critical_hall_parameter` a subclass provides, so every criterion gets them for
    free without re-implementing the comparison logic.
    """

    @abstractmethod
    def critical_hall_parameter(
            self,
            electron_temperature: float | Iterable,
            gas_temperature: float | Iterable,
            ionization_potential: float | Iterable,
            ionization_fraction: float | Iterable,
    ) -> float | Iterable:
        """beta_crit: the Hall parameter above which the plasma is unstable, directly
        comparable with Plasma.hall_parameter. Diverges to +inf wherever a concrete
        criterion judges the plasma unconditionally stable (e.g. Te->Tp or f_I->1) --
        implementations should not clamp that away, and callers should expect it."""

    def stability_margin(
            self,
            hall_parameter: float | Iterable,
            electron_temperature: float | Iterable,
            gas_temperature: float | Iterable,
            ionization_potential: float | Iterable,
            ionization_fraction: float | Iterable,
    ) -> float | Iterable:
        """beta_crit/beta: > 1 stable, < 1 unstable, == 1 marginal."""
        beta_crit = self.critical_hall_parameter(electron_temperature, gas_temperature, ionization_potential, ionization_fraction)
        return np.asarray(beta_crit, dtype=np.float64) / np.asarray(hall_parameter, dtype=np.float64)

    def is_stable(
            self,
            hall_parameter: float | Iterable,
            electron_temperature: float | Iterable,
            gas_temperature: float | Iterable,
            ionization_potential: float | Iterable,
            ionization_fraction: float | Iterable,
    ) -> bool | Iterable:
        """hall_parameter <= critical_hall_parameter."""
        beta_crit = self.critical_hall_parameter(electron_temperature, gas_temperature, ionization_potential, ionization_fraction)
        return np.asarray(hall_parameter) <= np.asarray(beta_crit)

    def plasma_critical_hall_parameter(self, plasma: Plasma) -> float:
        return self.critical_hall_parameter(
            plasma.electron_temperature, plasma.gas_temperature,
            plasma.seed_type.ionization_potential, plasma.ionization_fraction,
        )

    def plasma_stability_margin(self, plasma: Plasma) -> float:
        return self.stability_margin(
            plasma.hall_parameter, plasma.electron_temperature, plasma.gas_temperature,
            plasma.seed_type.ionization_potential, plasma.ionization_fraction,
        )

    def is_plasma_stable(self, plasma: Plasma) -> bool:
        return self.is_stable(
            plasma.hall_parameter, plasma.electron_temperature, plasma.gas_temperature,
            plasma.seed_type.ionization_potential, plasma.ionization_fraction,
        )
