from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from scipy import constants

from magnetohydrodynamics.presets import default_operating_point


@dataclass(frozen=True)
class OperatingPoint:
    """The physical knobs a stability sweep can vary: B0, Tp, p0, v0, seed_fraction,
    load_resistivity. Not `HallSolver.solve_equilibrium`'s raw args directly --
    `resolve()` derives gas_number_density/seed_number_density from these, since
    sweeping e.g. Tp at fixed p0 must recompute n_p = p0/(k Tp) rather than holding n_p
    fixed."""
    B0: float
    Tp: float
    p0: float
    v0: float
    seed_fraction: float
    load_resistivity: float

    @classmethod
    def default(cls) -> OperatingPoint:
        """The default operating point (presets.default_operating_point()), expressed
        in these physical-knob terms."""
        params = default_operating_point()
        return cls(
            B0=params["magnetic_field"],
            Tp=params["inlet_gas_temperature"],
            p0=params["inlet_pressure"],
            v0=params["inlet_speed"],
            seed_fraction=params["inlet_seed_fraction"],
            load_resistivity=params["load_resistance"] * params["area"] / params["length"],
        )

    def resolve(self, **overrides) -> dict:
        """This point (or one of its fields overridden with a scalar or an array --
        e.g. a whole meshgrid, for a batch sweep) -> the six raw args
        HallSolver.solve_equilibrium(_batch) needs. `**overrides` are applied via
        `dataclasses.replace`, so an unknown field name raises immediately rather than
        being silently ignored."""
        point = dataclasses.replace(self, **overrides) if overrides else self
        gas_number_density = point.p0 / (constants.k * point.Tp)
        seed_number_density = point.seed_fraction * gas_number_density
        return dict(
            flow_speed=point.v0,
            gas_temperature=point.Tp,
            gas_number_density=gas_number_density,
            seed_number_density=seed_number_density,
            magnetic_field=point.B0,
            load_resistivity=point.load_resistivity,
        )
