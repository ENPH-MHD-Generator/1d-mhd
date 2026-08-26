from magnetohydrodynamics.thermophysics.gas_type import GasType
from magnetohydrodynamics.state import State
from dataclasses import dataclass


@dataclass
class GasState(State):
    gas_type: GasType

    gas_temperature: float      # Primary Gas Temperature [K]
    gas_pressure: float         # Primary Gas Pressure [Pa]
    flow_speed: float           # Axial Speed [m/s]
    gas_number_density: float   # Primary Gas Number Density [1/m^3]
    magnetic_field: float       # External Applied Magnetic Field [T]
