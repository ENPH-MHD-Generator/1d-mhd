"""
Tests for magnetohydrodynamics.solver.tapered_hall_solver -- the variable-area
(linearly-tapered) counterpart to HallSolver.march(). Since this is a genuinely new
solver (not a modification of HallSolver), there's no frozen reference_main.py-style
oracle to compare full marches against; correctness instead comes from (a) the
half_angle_deg=0 case reproducing HallSolver.march() exactly (TestMarchReducesToHallSolver),
(b) tests/test_tapered_derivation_sympy.py's symbolic check of the ODEs themselves, and
(c) tests/test_tapered_energy_accounting.py's numerical energy-conservation check.
"""
import numpy as np
import pytest

from magnetohydrodynamics.solver.tapered_hall_solver import DivergenceAngleWarning, LinearTaper, TaperedHallSolver


class TestLinearTaper:
    def test_zero_half_angle_gives_constant_area(self):
        taper = LinearTaper(inlet_area=0.01, half_angle_deg=0.0)
        assert taper.dA_dx == 0.0
        x = np.linspace(0.0, 1.0, 5)
        np.testing.assert_array_equal(taper.area(x), np.full_like(x, 0.01))

    def test_positive_half_angle_increases_area_with_x(self):
        taper = LinearTaper(inlet_area=0.01, half_angle_deg=10.0)
        assert taper.dA_dx > 0.0
        assert taper.area(0.0) == pytest.approx(0.01)
        assert taper.area(1.0) > taper.area(0.0)

    def test_negative_half_angle_decreases_area_with_x(self):
        taper = LinearTaper(inlet_area=0.01, half_angle_deg=-5.0)
        assert taper.dA_dx < 0.0
        assert taper.area(1.0) < taper.area(0.0)

    def test_area_is_exactly_linear_in_x(self):
        """The whole point of "linear" -- dA/dx must be a single constant, not itself
        a function of x (unlike a self-similar square taper, where it wouldn't be)."""
        taper = LinearTaper(inlet_area=0.02, half_angle_deg=7.0)
        x = np.linspace(0.0, 0.5, 50)
        slopes = np.diff(np.asarray(taper.area(x))) / np.diff(x)
        np.testing.assert_allclose(slopes, taper.dA_dx, rtol=1e-10)

    def test_local_half_angle_deg_is_constant_and_matches_half_angle_deg(self):
        taper = LinearTaper(inlet_area=0.01, half_angle_deg=6.5)
        x = np.linspace(0.0, 1.0, 4)
        np.testing.assert_array_equal(taper.local_half_angle_deg(x), np.full_like(x, 6.5))

    def test_from_exit_area_round_trips_through_area(self):
        taper = LinearTaper.from_exit_area(inlet_area=0.01, exit_area=0.015, length=0.2)
        assert taper.area(0.0) == pytest.approx(0.01)
        assert taper.area(0.2) == pytest.approx(0.015)

    def test_from_exit_area_with_equal_areas_gives_zero_half_angle(self):
        taper = LinearTaper.from_exit_area(inlet_area=0.01, exit_area=0.01, length=0.2)
        assert taper.half_angle_deg == pytest.approx(0.0, abs=1e-9)


class TestDivergenceAngleGuard:
    def test_gentle_angle_does_not_warn(self, hall_solver, channel_params, recwarn):
        solver = TaperedHallSolver(hall_solver, max_half_angle_deg=15.0)
        taper = LinearTaper(inlet_area=channel_params["A"], half_angle_deg=5.0)
        solver.march(
            num_slices=10, length=channel_params["L"], taper=taper,
            inlet_speed=channel_params["u0"], inlet_pressure=channel_params["p0"],
            inlet_gas_temperature=channel_params["Tp0"], magnetic_field=0.1,
            load_resistance=channel_params["R_L"], inlet_seed_fraction=channel_params["seed_frac0"],
        )
        assert not any(issubclass(w.category, DivergenceAngleWarning) for w in recwarn.list)

    def test_steep_angle_warns_by_default(self, hall_solver, channel_params):
        solver = TaperedHallSolver(hall_solver, max_half_angle_deg=15.0)
        taper = LinearTaper(inlet_area=channel_params["A"], half_angle_deg=30.0)
        with pytest.warns(DivergenceAngleWarning):
            solver.march(
                num_slices=10, length=channel_params["L"], taper=taper,
                inlet_speed=channel_params["u0"], inlet_pressure=channel_params["p0"],
                inlet_gas_temperature=channel_params["Tp0"], magnetic_field=0.1,
                load_resistance=channel_params["R_L"], inlet_seed_fraction=channel_params["seed_frac0"],
            )

    def test_steep_angle_raises_in_strict_mode(self, hall_solver, channel_params):
        solver = TaperedHallSolver(hall_solver, max_half_angle_deg=15.0, strict=True)
        taper = LinearTaper(inlet_area=channel_params["A"], half_angle_deg=30.0)
        with pytest.raises(DivergenceAngleWarning):
            solver.march(
                num_slices=10, length=channel_params["L"], taper=taper,
                inlet_speed=channel_params["u0"], inlet_pressure=channel_params["p0"],
                inlet_gas_temperature=channel_params["Tp0"], magnetic_field=0.1,
                load_resistance=channel_params["R_L"], inlet_seed_fraction=channel_params["seed_frac0"],
            )

    def test_negative_angle_beyond_threshold_also_warns(self, hall_solver, channel_params):
        """abs(half_angle_deg) is what's checked -- a steep converging taper is just
        as far outside the gentle-divergence approximation as a steep diverging one."""
        solver = TaperedHallSolver(hall_solver, max_half_angle_deg=15.0)
        taper = LinearTaper(inlet_area=channel_params["A"], half_angle_deg=-30.0)
        with pytest.warns(DivergenceAngleWarning):
            solver.march(
                num_slices=10, length=channel_params["L"] * 0.01, taper=taper,
                inlet_speed=channel_params["u0"], inlet_pressure=channel_params["p0"],
                inlet_gas_temperature=channel_params["Tp0"], magnetic_field=0.1,
                load_resistance=channel_params["R_L"], inlet_seed_fraction=channel_params["seed_frac0"],
            )


class TestAreaPositivityGuard:
    def test_converging_taper_that_pinches_shut_raises(self, hall_solver, channel_params):
        solver = TaperedHallSolver(hall_solver, max_half_angle_deg=90.0)
        # half-angle steep and negative enough, over this length, to close the channel
        taper = LinearTaper.from_exit_area(inlet_area=channel_params["A"], exit_area=-1e-6, length=channel_params["L"])
        with pytest.raises(ValueError, match=r"taper\.area"):
            solver.march(
                num_slices=10, length=channel_params["L"], taper=taper,
                inlet_speed=channel_params["u0"], inlet_pressure=channel_params["p0"],
                inlet_gas_temperature=channel_params["Tp0"], magnetic_field=0.1,
                load_resistance=channel_params["R_L"], inlet_seed_fraction=channel_params["seed_frac0"],
            )


class TestMarchReducesToHallSolver:
    """half_angle_deg=0 (constant area) must reproduce HallSolver.march() exactly --
    the single strongest correctness check available, since it needs no independent
    oracle: the two solvers describe the same physics in that limit by construction."""

    def _march_both(self, hall_solver, channel_params, magnetic_field, inlet_speed=None):
        solver = TaperedHallSolver(hall_solver)
        kwargs = dict(
            num_slices=channel_params["num_slices"],
            length=channel_params["L"],
            inlet_speed=inlet_speed if inlet_speed is not None else channel_params["u0"],
            inlet_pressure=channel_params["p0"],
            inlet_gas_temperature=channel_params["Tp0"],
            magnetic_field=magnetic_field,
            load_resistance=channel_params["R_L"],
            inlet_seed_fraction=channel_params["seed_frac0"],
        )
        original = hall_solver.march(area=channel_params["A"], **kwargs)
        taper = LinearTaper(inlet_area=channel_params["A"], half_angle_deg=0.0)
        tapered = solver.march(taper=taper, **kwargs)
        return original, tapered

    def test_not_choked_case_matches_exactly(self, hall_solver, channel_params):
        # B0=0.1 keeps this subsonic all the way through (see test_hall_solver.py).
        original, tapered = self._march_both(hall_solver, channel_params, magnetic_field=0.1)
        assert not original.choked and not tapered.choked
        assert len(original) == len(tapered)
        np.testing.assert_array_equal(tapered.x, original.x)
        for o, t in zip(original.states, tapered.states, strict=True):
            assert t.flow_speed == o.flow_speed
            assert t.gas_temperature == o.gas_temperature
            assert t.electron_temperature == o.electron_temperature
            assert t.load_power_density == o.load_power_density

    def test_choked_case_matches_exactly(self, hall_solver, channel_params):
        # channel_params' own B0=0.5 chokes well before the outlet (see test_hall_solver.py).
        original, tapered = self._march_both(hall_solver, channel_params, magnetic_field=channel_params["B0"])
        assert original.choked and tapered.choked
        assert len(original) == len(tapered)
        np.testing.assert_array_equal(tapered.x, original.x)
        assert tapered.states[-1].flow_speed == original.states[-1].flow_speed
        assert tapered.states[-1].gas_temperature == original.states[-1].gas_temperature

    def test_supersonic_choked_case_matches_exactly(self, hall_solver, channel_params):
        """Same check, but starting supersonic -- confirms the subsonic/supersonic
        mirrored-threshold bisection carries over identically too. Uses a tight
        tolerance rather than bare equality: the tapered solver computes the same
        dTdx/dudx via a different floating-point evaluation order (e.g. `local_flux`
        via `psi/area` rather than a directly-carried `number_flux`), which can nudge
        the choking bisection's landing point by a few ULPs -- confirmed negligible
        (~1e-11 relative) rather than assumed."""
        original, tapered = self._march_both(
            hall_solver, channel_params, magnetic_field=channel_params["B0"], inlet_speed=1200.0,
        )
        assert original.choked and tapered.choked
        np.testing.assert_allclose(tapered.x, original.x, rtol=1e-9, atol=1e-12)
        assert tapered.states[-1].flow_speed == pytest.approx(original.states[-1].flow_speed, rel=1e-9)


class TestTaperedChannel:
    def test_area_property_matches_taper(self, hall_solver, channel_params):
        solver = TaperedHallSolver(hall_solver)
        taper = LinearTaper(inlet_area=channel_params["A"], half_angle_deg=5.0)
        channel = solver.march(
            num_slices=20, length=channel_params["L"], taper=taper,
            inlet_speed=channel_params["u0"], inlet_pressure=channel_params["p0"],
            inlet_gas_temperature=channel_params["Tp0"], magnetic_field=0.1,
            load_resistance=channel_params["R_L"], inlet_seed_fraction=channel_params["seed_frac0"],
        )
        np.testing.assert_allclose(channel.area, taper.area(channel.x))

    def test_load_resistivity_is_per_slice_and_tracks_area(self, hall_solver, channel_params):
        solver = TaperedHallSolver(hall_solver)
        taper = LinearTaper(inlet_area=channel_params["A"], half_angle_deg=8.0)
        channel = solver.march(
            num_slices=20, length=channel_params["L"], taper=taper,
            inlet_speed=channel_params["u0"], inlet_pressure=channel_params["p0"],
            inlet_gas_temperature=channel_params["Tp0"], magnetic_field=0.1,
            load_resistance=channel_params["R_L"], inlet_seed_fraction=channel_params["seed_frac0"],
        )
        assert channel.load_resistivity.shape == channel.x.shape
        expected = channel_params["R_L"] * channel.area / channel_params["L"]
        np.testing.assert_allclose(channel.load_resistivity, expected)
        # Area grows -> load resistivity grows too (fixed load_resistance, growing area).
        assert np.all(np.diff(channel.load_resistivity) > 0.0)

    def test_to_dict_includes_area_and_matches_channel_to_dict_shape(self, hall_solver, channel_params):
        solver = TaperedHallSolver(hall_solver)
        taper = LinearTaper(inlet_area=channel_params["A"], half_angle_deg=5.0)
        channel = solver.march(
            num_slices=10, length=channel_params["L"], taper=taper,
            inlet_speed=channel_params["u0"], inlet_pressure=channel_params["p0"],
            inlet_gas_temperature=channel_params["Tp0"], magnetic_field=0.1,
            load_resistance=channel_params["R_L"], inlet_seed_fraction=channel_params["seed_frac0"],
        )
        out = channel.to_dict()
        assert "area" in out
        for key in ("x", "u", "Tp", "p", "np", "Te", "ne", "beta", "Jx", "Jy", "Ex", "S_ohm", "S_load", "eta_L", "ns", "f_I"):
            assert key in out


class TestSupersonicTaperBuysBackDistance:
    """The motivating case: a supersonic inlet that chokes almost immediately under
    HallSolver's constant-area march should travel measurably further -- and extract
    measurably more enthalpy -- under a modest diverging taper."""

    def test_diverging_taper_travels_further_and_extracts_more_than_constant_area(self, hall_solver, channel_params):
        from magnetohydrodynamics.analysis import summarize_tapered_performance

        area = channel_params["A"]
        length = channel_params["L"]
        eta_L = 0.0016  # near-marginal-stability point identified for this exact case
        common = dict(
            num_slices=2000, length=length, inlet_speed=1200.0, inlet_pressure=channel_params["p0"],
            inlet_gas_temperature=channel_params["Tp0"], magnetic_field=0.5,
            load_resistance=eta_L * length / area, inlet_seed_fraction=channel_params["seed_frac0"],
        )
        solver = TaperedHallSolver(hall_solver)
        flat = solver.march(taper=LinearTaper(inlet_area=area, half_angle_deg=0.0), **common)
        diverging = solver.march(taper=LinearTaper(inlet_area=area, half_angle_deg=10.0), **common)

        assert diverging.x[-1] > flat.x[-1]

        gas_type = hall_solver.gas_type
        flat_perf = summarize_tapered_performance(
            flat.to_dict(), cp=gas_type.molar_heat_capacity, m_p=gas_type.particle_mass, gamma=gas_type.heat_capacity_ratio,
        )
        diverging_perf = summarize_tapered_performance(
            diverging.to_dict(), cp=gas_type.molar_heat_capacity, m_p=gas_type.particle_mass, gamma=gas_type.heat_capacity_ratio,
        )
        assert diverging_perf["PL"] > flat_perf["PL"]
