from abc import ABC, abstractmethod
from typing import Iterable


class IonizationModel(ABC):
    @abstractmethod
    def get_electron_density(
            self,
            electron_temperature: float | Iterable,
            seed_number_density: float | Iterable
    ) -> float | Iterable: ...
