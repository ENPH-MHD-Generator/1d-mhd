from dataclasses import dataclass

from magnetohydrodynamics.ionization.seed_type import SeedType
from magnetohydrodynamics.state.state import State


@dataclass
class IonizationState(State):
    seed_type: SeedType
    seed_number_density: float              # Seed Number Density [1/m^3]
    electron_temperature: float             # Electron Temperature [K]
    electron_number_density: float          # Electron Density [1/m^3]

    momentum_transfer_frequency: float      # Momentum-Transfer Frequency [1/s]
    energy_transfer_frequency: float        # Energy-Transfer Frequency [1/s]

    current_density: tuple[float, float]    # (J_x, J_y) [A/m^3]
    axial_electric_field: float             # E_x [N/C]

    ohmic_power_density: float  # Ohmic Power Density [W/m^3]
    load_power_density: float   # Load Power Density [W/m^3]
