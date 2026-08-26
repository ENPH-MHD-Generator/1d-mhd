from magnetohydrodynamics.thermophysics.gas_model import GasModel
from scipy.constants import k
import numpy as np


class IdealGas(GasModel):
    def compute_number_density(self) -> float:
        """Ideal gas number density n_p = p/(kT) [1/m^3]."""
        self.n_p = self.p / (k * self.Tp)
        return self.n_p

    def get_mass_density(self) -> float:
        """Mass density [kg/m^3] from ideal gas law."""
        return self.p / (self.gas_type.specific_gas_constant * self.Tp)

    def get_mach_number(self) -> float:
        """A simple Mach-like parameter from your code (monatomic-ish)."""
        m_p = self.gas_type.particle_mass
        return np.sqrt((m_p * self.u**2) / (self.gas_type.heat_capacity_ratio * k * self.Tp))
