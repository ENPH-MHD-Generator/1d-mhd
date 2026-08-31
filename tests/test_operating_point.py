"""
OperatingPoint.resolve() is the ideal-gas-law unit conversion every sweep in
magnetohydrodynamics.stability depends on, and default_channel_operating_point() is a
thin wrapper over presets.default_operating_point() -- both worth pinning down
directly.
"""
import dataclasses

import pytest
from scipy import constants

from magnetohydrodynamics.operating_point import OperatingPoint
from magnetohydrodynamics.presets import default_channel_operating_point, default_operating_point


def test_default_matches_default_operating_point():
    params = default_operating_point()
    base = default_channel_operating_point()
    assert params["magnetic_field"] == base.B0
    assert base.Tp == params["inlet_gas_temperature"]
    assert base.p0 == params["inlet_pressure"]
    assert base.v0 == params["inlet_speed"]
    assert base.seed_fraction == params["inlet_seed_fraction"]
    expected_load_resistivity = params["load_resistance"] * params["area"] / params["length"]
    assert base.load_resistivity == pytest.approx(expected_load_resistivity)


def test_resolve_derives_number_densities_from_ideal_gas_law():
    base = OperatingPoint(B0=0.5, Tp=2000.0, p0=10.0e3, v0=150.0, seed_fraction=6.18e-3, load_resistivity=0.5)
    point = base.resolve()

    expected_gas_number_density = base.p0 / (constants.k * base.Tp)
    assert point.gas_number_density == pytest.approx(expected_gas_number_density)
    assert point.seed_number_density == pytest.approx(base.seed_fraction * expected_gas_number_density)
    assert point.flow_speed == base.v0
    assert point.gas_temperature == base.Tp
    assert point.magnetic_field == base.B0
    assert point.load_resistivity == base.load_resistivity


def test_resolve_override_recomputes_dependent_densities():
    """Overriding Tp (at fixed p0) must recompute n_p = p0/(k*Tp), not hold n_p fixed --
    this is the whole reason resolve() exists instead of just reading the dataclass
    fields directly."""
    base = OperatingPoint(B0=0.5, Tp=2000.0, p0=10.0e3, v0=150.0, seed_fraction=6.18e-3, load_resistivity=0.5)
    baseline = base.resolve()
    overridden = base.resolve(Tp=4000.0)

    assert overridden.gas_temperature == 4000.0
    assert overridden.gas_number_density == pytest.approx(base.p0 / (constants.k * 4000.0))
    assert overridden.gas_number_density != pytest.approx(baseline.gas_number_density)


def test_resolve_applies_a_single_override():
    base = OperatingPoint(B0=0.5, Tp=2000.0, p0=10.0e3, v0=150.0, seed_fraction=6.18e-3, load_resistivity=0.5)
    point = base.resolve(B0=3.5)
    assert point.magnetic_field == 3.5


def test_resolve_rejects_an_unknown_field_name():
    """Unlike the old dict-merge implementation, an unknown override key should raise
    immediately (dataclasses.replace's own behavior) rather than being silently
    ignored -- catches typos like resolve(b0=...) instead of B0=...."""
    base = OperatingPoint(B0=0.5, Tp=2000.0, p0=10.0e3, v0=150.0, seed_fraction=6.18e-3, load_resistivity=0.5)
    with pytest.raises(TypeError):
        base.resolve(b0=3.5)


def test_is_immutable():
    base = OperatingPoint(B0=0.5, Tp=2000.0, p0=10.0e3, v0=150.0, seed_fraction=6.18e-3, load_resistivity=0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        base.B0 = 1.0
