"""
Variable-area counterpart to tests/test_energy_accounting.py: checks that
TaperedHallSolver.march() still conserves energy correctly -- the only energy that
should ever leave the primary gas stream is the load power S_L, exactly as for the
constant-area case (see Derivation.md's "Correction" note and its "Variable-Area
(Tapered) Channel" section). Since the area-varying terms enter only the momentum
equation (see that section), and the stagnation-energy identity these tests check is
independent of area varying, these tests use a genuinely nonzero half-angle throughout
-- not the half_angle_deg=0 case tests/test_tapered_hall_solver.py's
TestMarchReducesToHallSolver already covers.

Uses magnetic_field=0.1, same as tests/test_energy_accounting.py, to stay subsonic for
the whole channel (see test_flow_chokes_at_higher_field there for the choking case,
mirrored here by test_flow_chokes_at_higher_field).
"""
import numpy as np
import pytest
from scipy import constants

from magnetohydrodynamics.analysis import summarize_tapered_performance
from magnetohydrodynamics.solver.tapered_hall_solver import LinearTaper, TaperedHallSolver

HALF_ANGLE_DEG = 5.0


def _march_and_measure(hall_solver, gas_type, channel_params, num_slices, magnetic_field=0.1):
    solver = TaperedHallSolver(hall_solver)
    taper = LinearTaper(inlet_area=channel_params["A"], half_angle_deg=HALF_ANGLE_DEG)
    channel = solver.march(
        num_slices=num_slices,
        length=channel_params["L"],
        taper=taper,
        inlet_speed=channel_params["u0"],
        inlet_pressure=channel_params["p0"],
        inlet_gas_temperature=channel_params["Tp0"],
        magnetic_field=magnetic_field,
        load_resistance=channel_params["R_L"],
        inlet_seed_fraction=channel_params["seed_frac0"],
    )
    out = channel.to_dict()
    perf = summarize_tapered_performance(
        out, cp=gas_type.molar_heat_capacity, m_p=gas_type.particle_mass, gamma=gas_type.heat_capacity_ratio,
    )
    cp = gas_type.molar_heat_capacity
    h0 = cp * out["Tp"] + 0.5 * out["u"] ** 2
    mdot = gas_type.particle_mass * out["np"][0] * out["u"][0] * out["area"][0]
    stagnation_enthalpy_drop = mdot * (h0[0] - h0[-1])
    return channel, perf, stagnation_enthalpy_drop


def test_mechanical_power_removed_equals_ohmic_plus_load(hall_solver, channel_params):
    """Same foundational identity as the constant-area case -- Ohm's law's own
    energy-consistency is completely unaffected by area, since TaperedHallSolver
    reuses HallSolver's per-slice closure unchanged."""
    channel, _, _ = _march_and_measure(hall_solver, hall_solver.gas_type, channel_params, num_slices=50)
    for state in channel.states:
        mechanical_power_removed = -state.current_density[1] * state.magnetic_field * state.flow_speed
        assert mechanical_power_removed == pytest.approx(
            state.ohmic_power_density + state.load_power_density, rel=1e-9
        )


def test_stagnation_enthalpy_drop_matches_load_power(hall_solver, channel_params):
    """d(mdot * h0) should equal -S_L integrated over the (now area-weighted) channel
    -- the same identity as the constant-area case, but only holds here if
    summarize_tapered_performance's PL = trapezoid(S_L*area, x) is actually integrating
    area correctly (a bug that pulled area outside the integral, or omitted it, would
    show up as a real discrepancy here, not just a wrong number). Needs more slices
    than the constant-area version's equivalent test to reach a comparably tight
    tolerance -- confirmed (see test_energy_balance_error_shrinks_with_more_slices)
    that this is the same O(dx) explicit-Euler truncation error the constant-area case
    has, just with a larger error constant here (the extra dA/dx-driven curvature),
    not a slower convergence order -- 400 slices alone give ~5% error, not a bug."""
    channel, perf, stagnation_enthalpy_drop = _march_and_measure(
        hall_solver, hall_solver.gas_type, channel_params, num_slices=6400
    )
    assert not channel.choked
    assert stagnation_enthalpy_drop == pytest.approx(perf["PL"], rel=5e-3)


def test_energy_balance_error_shrinks_with_more_slices(hall_solver, channel_params):
    """Explicit Euler's O(dx) truncation error should shrink roughly linearly with the
    step size, same as the constant-area case."""
    errors = []
    for num_slices in (100, 400):
        channel, perf, stagnation_enthalpy_drop = _march_and_measure(
            hall_solver, hall_solver.gas_type, channel_params, num_slices=num_slices
        )
        assert not channel.choked
        errors.append(abs(stagnation_enthalpy_drop - perf["PL"]) / perf["PL"])

    assert errors[1] < errors[0] / 2


def test_enthalpy_extraction_ratio_is_nonnegative_away_from_choking(hall_solver, channel_params):
    channel, perf, _ = _march_and_measure(hall_solver, hall_solver.gas_type, channel_params, num_slices=200)
    assert not channel.choked
    assert perf["enthalpy_extraction_ratio"] >= 0.0


def test_flow_chokes_at_higher_field(hall_solver, channel_params):
    """Same choking check as the constant-area case, at channel_params' own B0=0.5 --
    the Mach number bound must still hold even though it's now evaluated against a
    growing area/local flux along the way."""
    solver = TaperedHallSolver(hall_solver)
    taper = LinearTaper(inlet_area=channel_params["A"], half_angle_deg=HALF_ANGLE_DEG)
    channel = solver.march(
        num_slices=channel_params["num_slices"],
        length=channel_params["L"],
        taper=taper,
        inlet_speed=channel_params["u0"],
        inlet_pressure=channel_params["p0"],
        inlet_gas_temperature=channel_params["Tp0"],
        magnetic_field=channel_params["B0"],
        load_resistance=channel_params["R_L"],
        inlet_seed_fraction=channel_params["seed_frac0"],
    )
    assert channel.choked
    assert len(channel) < channel_params["num_slices"]

    gas_type = hall_solver.gas_type
    for state in channel.states:
        mach = np.sqrt(
            gas_type.particle_mass * state.flow_speed ** 2
            / (gas_type.heat_capacity_ratio * constants.k * state.gas_temperature)
        )
        assert mach <= hall_solver.max_mach_number + 1e-9
