import numpy as np
from scipy import constants

from magnetohydrodynamics.ionization.ionization_model import IonizationModel
from magnetohydrodynamics.ionization.seed_type import SeedType
from magnetohydrodynamics.typing import Scalar


class LocalThermodynamicEquilibrium(IonizationModel):
    def __init__(self, seed_type: SeedType):
        self._seed_type = seed_type

    def saha_coefficient(self, electron_temperature: Scalar) -> Scalar:
        """
        N(Te), the Saha prefactor: n_e^2/(n_s-n_e) = N(Te) at equilibrium.
        N(Te) = degeneracy_ratio * (2*pi*me*k*Te/h^2)^(3/2) * exp(-E_I/kTe).

        Exposed standalone (not just inline inside get_electron_density) because some
        callers need N(Te) directly rather than the full Saha quadratic solved for n_e --
        e.g. Friedberg's Sec. 7.2 design procedure, which computes n_e from a target power
        density first and only then needs N(Te) to get the ionisation fraction from it.
        """
        electron_temperature = np.asarray(electron_temperature, dtype=np.float64)
        ionization_potential = self._seed_type.ionization_potential * constants.e
        pref = (2.0 * np.pi * constants.electron_mass * constants.k * electron_temperature) / (constants.h**2)
        pref = np.power(pref, 1.5)
        expo = np.exp(-ionization_potential / (constants.k * electron_temperature))
        return self._seed_type.degeneracy_ratio * pref * expo

    def get_electron_density(
            self,
            electron_temperature: Scalar,
            seed_number_density: Scalar
    ) -> Scalar:
        """
        Saha-like model used in your original get_electron_density().
        Inputs:
          Te : electron temperature [K]
          ns : seed number density [1/m^3]
        Returns:
          ne : electron density [1/m^3]
        """
        seed_number_density = np.asarray(seed_number_density, dtype=np.float64)
        electron_temperature = np.asarray(electron_temperature, dtype=np.float64)
        seed_number_density, electron_temperature = np.broadcast_arrays(seed_number_density, electron_temperature)

        s = self.saha_coefficient(electron_temperature)

        disc = np.sqrt(s * (s + 4.0 * seed_number_density))
        ne = 0.5 * (disc - s)
        ne = np.clip(ne, 0.0, np.maximum(0.0, seed_number_density * (1.0 - 1e-12)))

        return ne
