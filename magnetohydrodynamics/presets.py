"""
Example Argon / caesium-seeded operating point (see Derivation.md's worked
example). Shared by main.py and the interactive UI (ui/app.py) so both stay
in sync with a single source of truth.
"""
from __future__ import annotations

from magnetohydrodynamics.ionization.saha_ionization import LocalThermodynamicEquilibrium
from magnetohydrodynamics.ionization.seed_type import SeedType
from magnetohydrodynamics.solver.hall_solver import HallSolver
from magnetohydrodynamics.thermophysics.gas_type import GasType
from magnetohydrodynamics.transport.mhd_transport_model import MHDTransportModel


def default_gas_type() -> GasType:
    return GasType(name="Argon", molar_mass=39.948e-3, heat_capacity_ratio=5.0 / 3.0)


def default_seed_type() -> SeedType:
    return SeedType(
        name="Caesium",
        ionization_potential=3.894,                           # eV
        electron_neutral_cross_section=3.93994730526347e-21,  # m^2
        degeneracy_ratio=1.0,
    )


def build_default_hall_solver() -> tuple[HallSolver, GasType]:
    """Argon primary gas, caesium-seeded -- see Derivation.md."""
    gas_type = default_gas_type()
    seed_type = default_seed_type()
    transport_model = MHDTransportModel(seed_type=seed_type, gas_type=gas_type)
    ionization_model = LocalThermodynamicEquilibrium(seed_type=seed_type)
    hall_solver = HallSolver(
        gas_type=gas_type,
        seed_type=seed_type,
        transport_model=transport_model,
        ionization_model=ionization_model,
    )
    return hall_solver, gas_type


def default_operating_point() -> dict:
    """Default operating point (see Derivation.md's worked example)."""
    load_resistivity = 1.5 / 2 * 50 / 2 / 2 / 2 / 2 / 20  # Ohm * m
    channel_area = 48e-3 * 48e-3  # 2in * 2in in meters
    channel_length = 0.2  # m

    return dict(
        num_slices=200,
        length=channel_length,
        area=channel_area,
        inlet_speed=150.115,           # m/s
        inlet_pressure=10.01e3,        # Pa
        inlet_gas_temperature=2000.0,  # K
        magnetic_field=0.5,            # T
        load_resistance=load_resistivity * channel_length / channel_area,
        inlet_seed_fraction=6.18e-3,
    )