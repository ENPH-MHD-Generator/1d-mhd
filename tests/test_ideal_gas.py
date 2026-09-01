import numpy as np
import pytest

from magnetohydrodynamics.thermophysics.ideal_gas import IdealGas


@pytest.fixture
def ideal_gas(gas_type) -> IdealGas:
    return IdealGas(gas_type=gas_type)


def test_number_density_matches_reference_inlet_calc(main_ref, ideal_gas):
    """main.py computes n_p0 as rho0/m_p with rho0 = p0/(R*Tp0); check p/(kT) agrees."""
    p0, Tp0 = 101.01e3, 2000.0
    rho0 = p0 / (main_ref.R * Tp0)
    expected_n_p0 = rho0 / main_ref.m_p

    actual = ideal_gas.compute_number_density(p0, Tp0)
    assert actual == pytest.approx(expected_n_p0, rel=1e-12)


def test_mass_density_matches_reference(main_ref, ideal_gas):
    # rel=1e-9: main.py's m_p is a hardcoded CODATA-rounded amu literal, while
    # GasType derives particle_mass as molar_mass / N_A -- both correct, but
    # they diverge at the ~1e-10 relative level from independent rounding.
    p0, Tp0 = 101.01e3, 2000.0
    expected_rho0 = p0 / (main_ref.R * Tp0)
    assert ideal_gas.get_mass_density(p0, Tp0) == pytest.approx(expected_rho0, rel=1e-9)


def test_mach_number_matches_reference(main_ref, ideal_gas):
    for u, Tp in [(150.115, 2000.0), (50.0, 500.0), (300.0, 3000.0)]:
        expected = main_ref.get_mach_number(u, Tp)
        assert ideal_gas.get_mach_number(u, Tp) == pytest.approx(expected, rel=1e-9)


def test_get_flow_speed_is_the_inverse_of_get_mach_number(ideal_gas):
    for u, Tp in [(150.115, 2000.0), (50.0, 500.0), (300.0, 3000.0)]:
        mach_number = ideal_gas.get_mach_number(u, Tp)
        assert ideal_gas.get_flow_speed(mach_number, Tp) == pytest.approx(u, rel=1e-12)


def test_get_flow_speed_holds_mach_number_fixed_across_temperature(ideal_gas):
    """The actual use case (EquilibriumSweep's fixed_mach_number sweeps): pick a Mach
    number once, then confirm the flow speed this returns really does read back the
    same Mach number at every temperature, not just the one it was derived from."""
    mach_number = 0.18
    temperatures = np.array([1.0, 100.0, 500.0, 1900.0, 6000.0])
    flow_speeds = ideal_gas.get_flow_speed(mach_number, temperatures)
    recovered = ideal_gas.get_mach_number(flow_speeds, temperatures)
    np.testing.assert_allclose(recovered, mach_number, rtol=1e-12)
