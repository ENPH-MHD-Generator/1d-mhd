from abc import ABC, abstractmethod

from magnetohydrodynamics.typing import Scalar


class TransportModel(ABC):
    @abstractmethod
    def get_momentum_transfer_frequency(
            self, electron_temperature: Scalar,
            gas_number_density: Scalar
    ) -> Scalar: ...

    @abstractmethod
    def get_energy_transfer_frequency(
            self, electron_temperature: Scalar,
            gas_number_density: Scalar
    ) -> Scalar: ...
