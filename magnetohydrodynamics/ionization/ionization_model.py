from abc import ABC, abstractmethod

from magnetohydrodynamics.typing import Scalar


class IonizationModel(ABC):
    @abstractmethod
    def get_electron_density(
            self,
            electron_temperature: Scalar,
            seed_number_density: Scalar
    ) -> Scalar: ...
