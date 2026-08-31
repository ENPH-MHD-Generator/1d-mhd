from dataclasses import dataclass

from scipy import constants


@dataclass(frozen=True)
class GasType:
    """Thermophysical constants for an ideal gas."""
    name: str
    molar_mass: float           # [kg / mol]
    heat_capacity_ratio: float  # gamma = C_p / C_v

    @property
    def particle_mass(self) -> float:
        """Particle mass [kg] assuming monatomic ideal gas or per-molecule basis."""
        return self.molar_mass / constants.N_A

    @property
    def specific_gas_constant(self) -> float:
        """Specific gas constant [J / (kg K)]."""
        return constants.R / self.molar_mass

    @property
    def molar_heat_capacity(self) -> float:
        """Ideal-gas molar heat capacity at constant pressure [J / (kg K)]."""
        return self.heat_capacity_ratio / (self.heat_capacity_ratio - 1.0) * self.specific_gas_constant
