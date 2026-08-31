import numpy as np
from scipy import constants

from magnetohydrodynamics.ionization.seed_type import SeedType
from magnetohydrodynamics.thermophysics.gas_type import GasType
from magnetohydrodynamics.transport.transport_model import TransportModel
from magnetohydrodynamics.typing import Scalar


class MHDTransportModel(TransportModel):
    def __init__(self, seed_type: SeedType, gas_type: GasType):
        self._seed_type = seed_type
        self._gas_type = gas_type

    def get_momentum_transfer_frequency(
            self, electron_temperature: Scalar,
            gas_number_density: Scalar
    ) -> Scalar:
        sqrt_term = np.sqrt(2.0 * constants.k * electron_temperature / constants.electron_mass)
        return gas_number_density * self._seed_type.electron_neutral_cross_section * sqrt_term

    def get_energy_transfer_frequency(
            self, electron_temperature: Scalar,
            gas_number_density: Scalar
    ) -> Scalar:
        momentum_transfer_frequency = self.get_momentum_transfer_frequency(
            electron_temperature,
            gas_number_density
        )

        return 2.0 * constants.electron_mass / self._gas_type.particle_mass * momentum_transfer_frequency
