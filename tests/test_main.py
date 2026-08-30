"""
Checks main.py's own wiring (build_hall_solver()/operating_point()) against
tests/reference_main.py's module constants, and its behavior under the
corrected axial march.

main.py's own march_channel() is NOT used as an oracle for full axial
profiles here -- see tests/reference_main.py's caveat and
tests/test_energy_accounting.py. main.py's default operating point (a low
10.01 kPa inlet pressure at B0=0.5 T) actually chokes under the corrected,
energy-conserving march (see Derivation.md) -- test_default_operating_point_
chokes locks that new, real behavior in rather than silently regressing it.
"""
import pytest


def test_build_hall_solver_matches_reference_species(main_ref, main_module):
    """main.py's Argon/caesium constants should match tests/reference_main.py's exactly."""
    _, gas_type = main_module.build_hall_solver()
    assert gas_type.name == "Argon"
    assert gas_type.particle_mass == pytest.approx(main_ref.m_p, rel=1e-9)
    assert gas_type.heat_capacity_ratio == main_ref.gamma


def test_default_operating_point_chokes(main_module):
    """main.py's default operating point drives the flow to M=1 (Rayleigh-flow choking --
    see Derivation.md) well before the requested channel length, under the corrected
    energy-conserving march. This documents that real, new behavior."""
    hall_solver, gas_type = main_module.build_hall_solver()
    params = main_module.operating_point()

    channel = hall_solver.march(**params)

    assert channel.choked
    assert len(channel) < params["num_slices"]

    out = channel.to_dict()
    perf = main_module.summarize_performance(
        out, A=params["area"],
        cp=gas_type.molar_heat_capacity, m_p=gas_type.particle_mass, gamma=gas_type.heat_capacity_ratio,
    )
    # Even truncated at the choke point, extraction should be physically sensible now.
    assert perf["enthalpy_extraction_ratio"] >= 0.0


def test_operating_point_away_from_choking_conserves_energy(main_module):
    """Same operating point, weak enough field to stay subsonic the whole channel: the
    stagnation-enthalpy identity (d(mdot h0) == -S_L, see Derivation.md) should hold end
    to end, exercised through main.py's own imports/wiring."""
    hall_solver, gas_type = main_module.build_hall_solver()
    params = dict(main_module.operating_point(), magnetic_field=0.02, num_slices=1600)

    channel = hall_solver.march(**params)
    assert not channel.choked

    out = channel.to_dict()
    perf = main_module.summarize_performance(
        out, A=params["area"],
        cp=gas_type.molar_heat_capacity, m_p=gas_type.particle_mass, gamma=gas_type.heat_capacity_ratio,
    )
    cp = gas_type.molar_heat_capacity
    h0 = cp * out["Tp"] + 0.5 * out["u"] ** 2
    mdot = gas_type.particle_mass * out["np"][0] * out["u"][0] * params["area"]
    stagnation_enthalpy_drop = mdot * (h0[0] - h0[-1])

    assert stagnation_enthalpy_drop == pytest.approx(perf["PL"], rel=2e-3)
    assert perf["enthalpy_extraction_ratio"] >= 0.0
