from dataclasses import dataclass

from scipy import constants

from magnetohydrodynamics.ionization.seed_type import SeedType
from magnetohydrodynamics.state.gas_state import GasState
from magnetohydrodynamics.state.ionization_state import IonizationState
from magnetohydrodynamics.state.state import State
from magnetohydrodynamics.thermophysics.gas_model import GasType


@dataclass
class Plasma(State):
    """
    Plasma slice.
    """
    ionization_state: IonizationState
    gas_state: GasState

    @property
    def gas_type(self) -> GasType:
        return self.gas_state.gas_type

    @property
    def gas_temperature(self) -> float:
        return self.gas_state.gas_temperature

    @property
    def gas_pressure(self) -> float:
        return self.gas_state.gas_pressure

    @property
    def flow_speed(self) -> float:
        return self.gas_state.flow_speed

    @property
    def gas_number_density(self) -> float:
        return self.gas_state.gas_number_density

    @property
    def magnetic_field(self) -> float:
        return self.gas_state.magnetic_field

    @property
    def seed_type(self) -> SeedType:
        return self.ionization_state.seed_type

    @property
    def seed_number_density(self) -> float:
        return self.ionization_state.seed_number_density

    @property
    def electron_temperature(self) -> float:
        return self.ionization_state.electron_temperature

    @property
    def electron_number_density(self) -> float:
        return self.ionization_state.electron_number_density

    @property
    def momentum_transfer_frequency(self) -> float:
        return self.ionization_state.momentum_transfer_frequency

    @property
    def energy_transfer_frequency(self) -> float:
        return self.ionization_state.energy_transfer_frequency

    @property
    def current_density(self) -> tuple[float, float]:
        return self.ionization_state.current_density

    @property
    def axial_electric_field(self) -> float:
        return self.ionization_state.axial_electric_field

    @property
    def ohmic_power_density(self) -> float:
        return self.ionization_state.ohmic_power_density

    @property
    def load_power_density(self) -> float:
        return self.ionization_state.load_power_density

    @property
    def seed_fraction(self) -> float:
        """n_e/n_p -- NOT Friedberg's ionisation fraction f_I; see `ionization_fraction`."""
        return self.electron_number_density / self.gas_number_density

    @property
    def ionization_fraction(self) -> float:
        """Friedberg's f_I = n_e/n_s: the fraction of the SEED gas that's ionised."""
        return self.electron_number_density / self.seed_number_density

    @property
    def resistivity(self):
        denominator = constants.e ** 2 * self.electron_number_density
        return constants.electron_mass * self.momentum_transfer_frequency / denominator

    @property
    def conductivity(self):
        return 1 / self.resistivity

    @property
    def hall_parameter(self):
        denominator = constants.electron_mass * self.momentum_transfer_frequency
        return constants.elementary_charge * self.magnetic_field / denominator

    def __repr__(self) -> str:
        return f"{self.seed_type.name}-seeded {self.gas_type.name} plasma with {self.conductivity:.2f} S/m."
