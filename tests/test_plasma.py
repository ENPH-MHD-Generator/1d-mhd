import pytest

from magnetohydrodynamics.plasma.plasma import Plasma
from magnetohydrodynamics.state.gas_state import GasState
from magnetohydrodynamics.state.ionization_state import IonizationState


@pytest.fixture
def plasma(gas_type, seed_type) -> Plasma:
    gas_state = GasState(
        gas_type=gas_type,
        gas_temperature=2000.0,
        gas_pressure=101.01e3,
        flow_speed=150.115,
        gas_number_density=3.66e23,
        magnetic_field=0.5,
    )
    ionization_state = IonizationState(
        seed_type=seed_type,
        seed_number_density=2.0e21,
        electron_temperature=7000.0,
        electron_number_density=1.2e21,
        momentum_transfer_frequency=5.0e7,
        energy_transfer_frequency=5.0e3,
        current_density=(100.0, -50.0),
        axial_electric_field=-25.0,
        ohmic_power_density=1.5e6,
        load_power_density=1.1e6,
    )
    return Plasma(ionization_state=ionization_state, gas_state=gas_state)


def test_resistivity_matches_reference(main_ref, plasma):
    expected = main_ref.get_resistivity(plasma.momentum_transfer_frequency, plasma.electron_number_density)
    assert plasma.resistivity == pytest.approx(expected, rel=1e-12)


def test_hall_parameter_matches_reference(main_ref, plasma):
    expected = main_ref.get_hall_parameter(plasma.magnetic_field, plasma.momentum_transfer_frequency)
    assert plasma.hall_parameter == pytest.approx(expected, rel=1e-12)


def test_conductivity_is_inverse_of_resistivity(plasma):
    assert plasma.conductivity == pytest.approx(1.0 / plasma.resistivity, rel=1e-12)


def test_properties_delegate_to_component_states(plasma):
    assert plasma.gas_temperature == plasma.gas_state.gas_temperature
    assert plasma.electron_temperature == plasma.ionization_state.electron_temperature
    assert plasma.current_density == plasma.ionization_state.current_density
