from abc import ABC


class State(ABC):  # noqa: B024 -- deliberately a marker/tag base (no interface to enforce),
    """Common ancestor for GasState/IonizationState/Plasma, used only so Geometry can
    type its `states` collection as `Iterable[State]` -- not meant to define a shared
    interface, so it has no abstract methods."""
