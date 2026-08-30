"""
Checks that the axial march conserves energy correctly: the only energy that
should ever leave the primary gas stream is the load power S_L (see
Derivation.md's "Correction" note on the stagnation-enthalpy energy equation).

These tests use magnetic_field=0.1 for `channel_params`' operating point,
which stays subsonic for the whole channel (HallSolver.march never sets
Channel.choked) -- see test_flow_chokes_at_higher_field below for the
choking case. Explicit Euler has O(dx) global error, so exact equality isn't
expected at finite num_slices; the tests check the identity holds to a loose
tolerance at a moderate slice count, and that the error shrinks as expected
when the step size shrinks.
"""
import numpy as np
import pytest

from magnetohydrodynamics.analysis import summarize_performance
from scipy import constants


def _march_and_measure(hall_solver, gas_type, channel_params, num_slices, magnetic_field=0.1):
    channel = hall_solver.march(
        num_slices=num_slices,
        length=channel_params["L"],
        area=channel_params["A"],
        inlet_speed=channel_params["u0"],
        inlet_pressure=channel_params["p0"],
        inlet_gas_temperature=channel_params["Tp0"],
        magnetic_field=magnetic_field,
        load_resistance=channel_params["R_L"],
        inlet_seed_fraction=channel_params["seed_frac0"],
    )
    out = channel.to_dict()
    perf = summarize_performance(
        out, A=channel_params["A"],
        cp=gas_type.molar_heat_capacity, m_p=gas_type.particle_mass, gamma=gas_type.heat_capacity_ratio,
    )
    cp = gas_type.molar_heat_capacity
    h0 = cp * out["Tp"] + 0.5 * out["u"] ** 2
    mdot = gas_type.particle_mass * out["np"][0] * out["u"][0] * channel_params["A"]
    stagnation_enthalpy_drop = mdot * (h0[0] - h0[-1])
    return channel, perf, stagnation_enthalpy_drop


def test_mechanical_power_removed_equals_ohmic_plus_load(hall_solver, channel_params):
    """Foundational identity the whole energy accounting rests on: the Lorentz retarding
    force removes exactly S_Omega + S_L from the flow's kinetic energy at every slice
    (Ohm's law is energy-consistent -- this is the literature-derived closure, unchanged
    by the axial-march fix)."""
    channel, _, _ = _march_and_measure(hall_solver, hall_solver._gas_type, channel_params, num_slices=50)
    for state in channel.states:
        mechanical_power_removed = -state.current_density[1] * state.magnetic_field * state.flow_speed
        assert mechanical_power_removed == pytest.approx(
            state.ohmic_power_density + state.load_power_density, rel=1e-9
        )


def test_stagnation_enthalpy_drop_matches_load_power(hall_solver, channel_params):
    """d(mdot * h0) should equal -S_L integrated over the channel -- i.e. only the load
    power actually leaves the primary gas stream."""
    channel, perf, stagnation_enthalpy_drop = _march_and_measure(
        hall_solver, hall_solver._gas_type, channel_params, num_slices=400
    )
    assert not channel.choked
    assert stagnation_enthalpy_drop == pytest.approx(perf["PL"], rel=1e-3)


def test_energy_balance_error_shrinks_with_more_slices(hall_solver, channel_params):
    """Explicit Euler's O(dx) truncation error should shrink roughly linearly with the
    step size -- confirms the identity holds in the continuum limit, not by coincidence
    at one particular resolution."""
    errors = []
    for num_slices in (100, 400):
        channel, perf, stagnation_enthalpy_drop = _march_and_measure(
            hall_solver, hall_solver._gas_type, channel_params, num_slices=num_slices
        )
        assert not channel.choked
        errors.append(abs(stagnation_enthalpy_drop - perf["PL"]) / perf["PL"])

    # 4x more slices (4x smaller dx) should cut a first-order error by well over half;
    # generous margin to avoid flakiness while still checking the right direction/order.
    assert errors[1] < errors[0] / 2


def test_enthalpy_extraction_ratio_is_nonnegative_away_from_choking(hall_solver, channel_params):
    """The original bug report: enthalpy_extraction_ratio could come out negative (the
    primary gas's stagnation enthalpy increasing instead of decreasing). Away from
    choking, it should now be >= 0."""
    channel, perf, _ = _march_and_measure(hall_solver, hall_solver._gas_type, channel_params, num_slices=200)
    assert not channel.choked
    assert perf["enthalpy_extraction_ratio"] >= 0.0


def test_flow_chokes_at_higher_field(hall_solver, channel_params):
    """At channel_params' default B0=0.5, Ohmic heating drives the subsonic flow toward
    M=1 (Rayleigh-flow choking) well before the outlet; the march must stop there rather
    than step through the m_p*v^2 - gamma*k*T == 0 singularity."""
    channel = hall_solver.march(
        num_slices=channel_params["num_slices"],
        length=channel_params["L"],
        area=channel_params["A"],
        inlet_speed=channel_params["u0"],
        inlet_pressure=channel_params["p0"],
        inlet_gas_temperature=channel_params["Tp0"],
        magnetic_field=channel_params["B0"],
        load_resistance=channel_params["R_L"],
        inlet_seed_fraction=channel_params["seed_frac0"],
    )
    assert channel.choked
    assert len(channel) < channel_params["num_slices"]

    gas_type = hall_solver._gas_type
    for state in channel.states:
        mach = np.sqrt(
            gas_type.particle_mass * state.flow_speed ** 2
            / (gas_type.heat_capacity_ratio * constants.k * state.gas_temperature)
        )
        assert mach <= hall_solver.max_mach_number + 1e-9
