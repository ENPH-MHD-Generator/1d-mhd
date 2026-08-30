from magnetohydrodynamics.thermophysics.gas_type import GasType
from abc import ABC, abstractmethod
from typing import Iterable


class GasModel(ABC):
    gas_type: GasType

    @abstractmethod
    def compute_number_density(
            self,
            gas_pressure: float | Iterable,
            gas_temperature: float | Iterable
    ) -> float | Iterable: ...

    @abstractmethod
    def get_mass_density(
            self,
            gas_pressure: float | Iterable,
            gas_temperature: float | Iterable
    ) -> float | Iterable: ...

    @abstractmethod
    def get_mach_number(
            self,
            flow_speed: float | Iterable,
            gas_temperature: float | Iterable
    ) -> float | Iterable: ...
