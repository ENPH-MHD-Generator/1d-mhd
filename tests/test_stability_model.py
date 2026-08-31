"""
Tests for magnetohydrodynamics.stability.stability_model.StabilityModel -- the concrete
comparison/wrapper logic (`stability_margin`, `is_stable`, the `plasma_*` convenience
methods) that every concrete criterion inherits for free. Exercised through a minimal
fake subclass with an arbitrary, easy-to-hand-check `critical_hall_parameter` (not
`FriedbergCriterion`'s real physics) so these tests check the base class's own logic in
isolation, independent of Friedberg's formulas -- those are covered separately in
test_friedberg_criterion.py.
"""
import pytest

from magnetohydrodynamics.plasma.plasma import Plasma
from magnetohydrodynamics.stability.stability_model import StabilityModel
from magnetohydrodynamics.state.gas_state import GasState
from magnetohydrodynamics.state.ionization_state import IonizationState


class ConstantCriticalHallParameter(StabilityModel):
    """Trivial concrete StabilityModel: beta_crit is always the given constant,
    regardless of the other inputs -- makes stability_margin/is_stable trivial to
    hand-check."""

    def __init__(self, beta_crit: float):
        self._beta_crit = beta_crit

    def critical_hall_parameter(self, electron_temperature, gas_temperature, ionization_potential, ionization_fraction) -> float:
        return self._beta_crit


def test_stability_model_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        StabilityModel()


class TestStabilityMargin:
    def test_greater_than_one_when_stable(self):
        model = ConstantCriticalHallParameter(beta_crit=10.0)
        assert model.stability_margin(1.0, 6000.0, 2000.0, 3.894, 0.5) == pytest.approx(10.0)

    def test_less_than_one_when_unstable(self):
        model = ConstantCriticalHallParameter(beta_crit=1.0)
        assert model.stability_margin(2.0, 6000.0, 2000.0, 3.894, 0.5) == pytest.approx(0.5)

    def test_equals_one_at_marginal_stability(self):
        model = ConstantCriticalHallParameter(beta_crit=5.0)
        assert model.stability_margin(5.0, 6000.0, 2000.0, 3.894, 0.5) == pytest.approx(1.0)


class TestIsStable:
    def test_true_when_hall_parameter_below_critical(self):
        model = ConstantCriticalHallParameter(beta_crit=10.0)
        assert bool(model.is_stable(1.0, 6000.0, 2000.0, 3.894, 0.5)) is True

    def test_false_when_hall_parameter_above_critical(self):
        model = ConstantCriticalHallParameter(beta_crit=1.0)
        assert bool(model.is_stable(2.0, 6000.0, 2000.0, 3.894, 0.5)) is False

    def test_true_at_exact_equality(self):
        model = ConstantCriticalHallParameter(beta_crit=5.0)
        assert bool(model.is_stable(5.0, 6000.0, 2000.0, 3.894, 0.5)) is True


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
        electron_number_density=1.2e21,  # ionization_fraction = 0.6
        momentum_transfer_frequency=5.0e7,
        energy_transfer_frequency=5.0e3,
        current_density=(100.0, -50.0),
        axial_electric_field=-25.0,
        ohmic_power_density=1.5e6,
        load_power_density=1.1e6,
    )
    return Plasma(ionization_state=ionization_state, gas_state=gas_state)


class TestPlasmaWrappers:
    """Each plasma_* wrapper should be a pure pass-through of the plasma's own fields
    into the corresponding scalar method."""

    def test_plasma_critical_hall_parameter_is_a_pure_pass_through(self, plasma):
        model = ConstantCriticalHallParameter(beta_crit=7.5)
        assert model.plasma_critical_hall_parameter(plasma) == pytest.approx(7.5)

    def test_plasma_stability_margin_is_a_pure_pass_through(self, plasma):
        model = ConstantCriticalHallParameter(beta_crit=7.5)
        expected = 7.5 / plasma.hall_parameter
        assert model.plasma_stability_margin(plasma) == pytest.approx(expected, rel=1e-12)

    def test_is_plasma_stable_matches_margin(self, plasma):
        stable_model = ConstantCriticalHallParameter(beta_crit=plasma.hall_parameter * 10.0)
        unstable_model = ConstantCriticalHallParameter(beta_crit=plasma.hall_parameter * 0.1)
        assert bool(stable_model.is_plasma_stable(plasma)) is True
        assert bool(unstable_model.is_plasma_stable(plasma)) is False
