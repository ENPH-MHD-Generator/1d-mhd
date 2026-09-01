from abc import ABC, abstractmethod

from magnetohydrodynamics.thermophysics.gas_type import GasType
from magnetohydrodynamics.typing import Scalar


class GasModel(ABC):
    gas_type: GasType

    @abstractmethod
    def compute_number_density(
            self,
            gas_pressure: Scalar,
            gas_temperature: Scalar
    ) -> Scalar: ...

    @abstractmethod
    def get_mass_density(
            self,
            gas_pressure: Scalar,
            gas_temperature: Scalar
    ) -> Scalar: ...

    @abstractmethod
    def get_mach_number(
            self,
            flow_speed: Scalar,
            gas_temperature: Scalar
    ) -> Scalar: ...

    @abstractmethod
    def get_flow_speed(
            self,
            mach_number: Scalar,
            gas_temperature: Scalar
    ) -> Scalar:
        """Inverse of get_mach_number: the flow speed that gives `mach_number` at
        `gas_temperature`."""
        ...
