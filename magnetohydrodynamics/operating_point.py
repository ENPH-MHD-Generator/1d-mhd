from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from scipy import constants

from magnetohydrodynamics.solver.equilibrium import EquilibriumInputs


@dataclass(frozen=True)
class OperatingPoint:
    """The physical knobs a stability sweep (or, more generally, any single-point
    equilibrium solve) can vary: B0, Tp, p0, v0, seed_fraction, load_resistivity. Not
    HallSolver.solve_equilibrium's raw args directly -- `resolve()` derives
    gas_number_density/seed_number_density from these, since sweeping e.g. Tp at fixed
    p0 must recompute n_p = p0/(k Tp) rather than holding n_p fixed.

    Lives here (a dependency-free leaf module) rather than in magnetohydrodynamics.stability
    (where it originated) so it can be used anywhere a HallSolver equilibrium is
    -- solver code included -- without pulling in stability's own machinery
    (FriedbergCriterion, EquilibriumSweep, ...), and without risking a circular import if
    something in solver/ ever wants this type: this module deliberately doesn't import
    presets.py (which imports solver.hall_solver) for that reason -- see
    presets.default_channel_operating_point() for the equivalent of what used to be an
    OperatingPoint.default() classmethod here."""
    B0: float
    Tp: float
    p0: float
    v0: float
    seed_fraction: float
    load_resistivity: float

    def resolve(self, **overrides) -> EquilibriumInputs:
        """This point (or one of its fields overridden with a scalar or an array --
        e.g. a whole meshgrid, for a batch sweep) -> the six raw args
        HallSolver.solve_equilibrium(_batch) needs. `**overrides` are applied via
        `dataclasses.replace`, so an unknown field name raises immediately rather than
        being silently ignored."""
        point = dataclasses.replace(self, **overrides) if overrides else self
        gas_number_density = point.p0 / (constants.k * point.Tp)
        seed_number_density = point.seed_fraction * gas_number_density
        return EquilibriumInputs(
            flow_speed=point.v0,
            gas_temperature=point.Tp,
            gas_number_density=gas_number_density,
            seed_number_density=seed_number_density,
            magnetic_field=point.B0,
            load_resistivity=point.load_resistivity,
        )
