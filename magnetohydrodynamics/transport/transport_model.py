from abc import ABC, abstractmethod
from typing import Iterable


class TransportModel(ABC):
    @abstractmethod
    def get_momentum_transfer_frequency(
            self, electron_temperature: float | Iterable,
            gas_number_density: float | Iterable
    ) -> float | Iterable: ...

    @abstractmethod
    def get_energy_transfer_frequency(
            self, electron_temperature: float | Iterable,
            gas_number_density: float | Iterable
    ) -> float | Iterable: ...