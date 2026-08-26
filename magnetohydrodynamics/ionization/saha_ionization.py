from magnetohydrodynamics.ionization.ionization_model import IonizationModel
from magnetohydrodynamics.ionization.seed_type import SeedType
from scipy.constants import constants
from typing import Iterable
import numpy as np


class LocalThermodynamicEquilibrium(IonizationModel):
    def __init__(self, seed_type: SeedType):
        self._seed_type = seed_type

    def get_electron_density(
            self,
            electron_temperature: float | Iterable,
            seed_number_density: float | Iterable
    ) -> float | Iterable:
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

        ionization_potential = self._seed_type.ionization_potential * constants.e
        pref = (2.0 * np.pi * constants.electron_mass * constants.k * electron_temperature) / (constants.h**2)
        pref = np.power(pref, 1.5)
        expo = np.exp(-ionization_potential / (constants.k * electron_temperature))
        s = self._seed_type.degeneracy_ratio * pref * expo

        disc = np.sqrt(s * (s + 4.0 * seed_number_density))
        ne = 0.5 * (disc - s)
        ne = np.clip(ne, 0.0, np.maximum(0.0, seed_number_density * (1.0 - 1e-12)))

        return ne
