from magnetohydrodynamics.thermophysics.gas_model import GasModel
from magnetohydrodynamics.thermophysics.gas_type import GasType
from scipy import constants
from typing import Iterable
import numpy as np


class IdealGas(GasModel):
    def __init__(self, gas_type: GasType):
        self.gas_type = gas_type

    def compute_number_density(
            self,
            gas_pressure: float | Iterable,
            gas_temperature: float | Iterable
    ) -> float | Iterable:
        """Ideal gas number density n_p = p / (kT) [1/m^3]."""
        return gas_pressure / (constants.k * gas_temperature)

    def get_mass_density(
            self,
            gas_pressure: float | Iterable,
            gas_temperature: float | Iterable
    ) -> float | Iterable:
        """Mass density [kg/m^3] from the ideal gas law."""
        return gas_pressure / (self.gas_type.specific_gas_constant * gas_temperature)

    def get_mach_number(
            self,
            flow_speed: float | Iterable,
            gas_temperature: float | Iterable
    ) -> float | Iterable:
        """Gas-dynamic Mach number M = sqrt(m v^2 / (gamma k T))."""
        m_particle = self.gas_type.particle_mass
        gamma = self.gas_type.heat_capacity_ratio
        return np.sqrt((m_particle * flow_speed**2) / (gamma * constants.k * gas_temperature))
