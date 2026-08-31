from __future__ import annotations

from dataclasses import dataclass

from scipy import constants

from magnetohydrodynamics.typing import Scalar


@dataclass(frozen=True)
class EquilibriumInputs:
    """The six raw args HallSolver.solve_equilibrium(_batch) needs -- each a scalar or
    a numpy array broadcastable to a common shape. This is what OperatingPoint.resolve()
    produces, replacing the plain dict that used to be built (and typo-prone-string-key
    read back) throughout magnetohydrodynamics.stability."""
    flow_speed: Scalar
    gas_temperature: Scalar
    gas_number_density: Scalar
    seed_number_density: Scalar
    magnetic_field: Scalar
    load_resistivity: Scalar

    def as_kwargs(self) -> dict[str, Scalar]:
        """A shallow (no data copied) kwargs dict for solve_equilibrium(_batch) --
        deliberately not dataclasses.asdict(self), which deep-copies every field; that's
        cheap for OperatingPoint's own plain floats but not for the large meshgrid-shaped
        arrays these fields typically hold once resolved for a batch sweep."""
        return dict(
            flow_speed=self.flow_speed,
            gas_temperature=self.gas_temperature,
            gas_number_density=self.gas_number_density,
            seed_number_density=self.seed_number_density,
            magnetic_field=self.magnetic_field,
            load_resistivity=self.load_resistivity,
        )


@dataclass
class Equilibrium:
    """One (or, with array-valued fields, a whole batch/grid of) HallSolver equilibrium
    solve(s) -- the fixed-point loop's converged state, replacing the dict
    solve_equilibrium/solve_equilibrium_batch/EquilibriumSweep.matched_load used to build
    and hand back.

    resistivity/hall_parameter are derived @properties, computed on demand from the
    stored fields (the same formulas Plasma.resistivity/Plasma.hall_parameter already use
    for the scalar, per-slice case) rather than separate stored fields solve_equilibrium_batch
    used to recompute by hand after the fact -- one formula, not two copies of it."""
    flow_speed: Scalar
    gas_temperature: Scalar
    gas_number_density: Scalar
    seed_number_density: Scalar
    magnetic_field: Scalar
    electron_temperature: Scalar
    electron_number_density: Scalar
    momentum_transfer_frequency: Scalar
    energy_transfer_frequency: Scalar
    current_x: Scalar
    current_y: Scalar
    axial_electric_field: Scalar
    ohmic_power_density: Scalar
    load_power_density: Scalar

    @property
    def resistivity(self) -> Scalar:
        return constants.electron_mass * self.momentum_transfer_frequency / (constants.e ** 2 * self.electron_number_density)

    @property
    def hall_parameter(self) -> Scalar:
        return constants.e * self.magnetic_field / (constants.electron_mass * self.momentum_transfer_frequency)
