from abc import ABC
from typing import Iterable
from magnetohydrodynamics.state.state import State
import numpy as np

class Geometry(ABC):
    states: Iterable[State]

    def __getitem__(self, item) -> Iterable[float]:
        return np.fromiter([state.__getattribute__(item) for state in self.states], dtype=float)

    def __iter__(self):
        return iter(self.states)
