import numpy as np
from scipy import constants

from magnetohydrodynamics.thermophysics.gas_model import GasModel
from magnetohydrodynamics.thermophysics.gas_type import GasType
from magnetohydrodynamics.typing import Scalar


class IdealGas(GasModel):
    def __init__(self, gas_type: GasType):
        self.gas_type = gas_type

    def compute_number_density(
            self,
            gas_pressure: Scalar,
            gas_temperature: Scalar
    ) -> Scalar:
        """Ideal gas number density n_p = p / (kT) [1/m^3]."""
        return gas_pressure / (constants.k * gas_temperature)

    def get_mass_density(
            self,
            gas_pressure: Scalar,
            gas_temperature: Scalar
    ) -> Scalar:
        """Mass density [kg/m^3] from the ideal gas law."""
        return gas_pressure / (self.gas_type.specific_gas_constant * gas_temperature)

    def get_mach_number(
            self,
            flow_speed: Scalar,
            gas_temperature: Scalar
    ) -> Scalar:
        """Gas-dynamic Mach number M = sqrt(m v^2 / (gamma k T))."""
        m_particle = self.gas_type.particle_mass
        gamma = self.gas_type.heat_capacity_ratio
        return np.sqrt((m_particle * flow_speed**2) / (gamma * constants.k * gas_temperature))

    def get_flow_speed(
            self,
            mach_number: Scalar,
            gas_temperature: Scalar
    ) -> Scalar:
        """Inverse of get_mach_number: v = M * sqrt(gamma k T / m) -- the flow speed
        that holds Mach number fixed at `mach_number` as `gas_temperature` varies
        (used by EquilibriumSweep's fixed_mach_number sweeps, see its docstring)."""
        m_particle = self.gas_type.particle_mass
        gamma = self.gas_type.heat_capacity_ratio
        return mach_number * np.sqrt((gamma * constants.k * gas_temperature) / m_particle)
