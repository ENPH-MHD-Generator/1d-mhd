from magnetohydrodynamics.thermophysics.gas_type import GasType
from abc import ABC, abstractmethod


class GasModel(ABC):
    gas_type: GasType

    @abstractmethod
    def compute_number_density(self) -> float: ...

    @abstractmethod
    def get_mass_density(self) -> float: ...

    @abstractmethod
    def get_mach_number(self) -> float: ...
