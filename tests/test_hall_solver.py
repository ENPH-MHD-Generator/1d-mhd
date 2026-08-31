import numpy as np
import pytest
from scipy import constants


class TestSolveEquilibriumBatch:
    """solve_equilibrium_batch runs the identical _iterate_equilibrium fixed-point loop
    as solve_equilibrium, just vectorized -- it must agree with looping the scalar
    version point-by-point, for both a simple grid and irregular/broadcast shapes."""

    def test_matches_scalar_loop_on_a_grid(self, hall_solver, channel_params):
        B0 = np.linspace(0.2, 3.0, 5)
        Tp = np.linspace(1000.0, 3000.0, 4)
        B0_grid, Tp_grid = np.meshgrid(B0, Tp)  # shape (4, 5)

        n_p0 = channel_params["p0"] / (
            constants.k * channel_params["Tp0"]
        )
        n_s0 = channel_params["seed_frac0"] * n_p0
        eta_L = channel_params["R_L"] * channel_params["A"] / channel_params["L"]

        result = hall_solver.solve_equilibrium_batch(
            flow_speed=channel_params["u0"],
            gas_temperature=Tp_grid,
            gas_number_density=n_p0,
            seed_number_density=n_s0,
            magnetic_field=B0_grid,
            load_resistivity=eta_L,
        )

        for i in range(Tp_grid.shape[0]):
            for j in range(Tp_grid.shape[1]):
                plasma = hall_solver.solve_equilibrium(
                    flow_speed=channel_params["u0"],
                    gas_temperature=float(Tp_grid[i, j]),
                    gas_number_density=n_p0,
                    seed_number_density=n_s0,
                    magnetic_field=float(B0_grid[i, j]),
                    load_resistivity=eta_L,
                )
                assert result.electron_temperature[i, j] == pytest.approx(plasma.electron_temperature, rel=1e-9)
                assert result.electron_number_density[i, j] == pytest.approx(plasma.electron_number_density, rel=1e-9)
                assert result.hall_parameter[i, j] == pytest.approx(plasma.hall_parameter, rel=1e-9)
                assert result.resistivity[i, j] == pytest.approx(plasma.resistivity, rel=1e-9)
                assert result.current_x[i, j] == pytest.approx(plasma.current_density[0], rel=1e-9)
                assert result.current_y[i, j] == pytest.approx(plasma.current_density[1], rel=1e-9)
                assert result.axial_electric_field[i, j] == pytest.approx(plasma.axial_electric_field, rel=1e-9)
                assert result.ohmic_power_density[i, j] == pytest.approx(plasma.ohmic_power_density, rel=1e-9)
                assert result.load_power_density[i, j] == pytest.approx(plasma.load_power_density, rel=1e-9)

    def test_scalar_inputs_match_solve_equilibrium(self, hall_solver, channel_params):
        n_p0 = channel_params["p0"] / (
            constants.k * channel_params["Tp0"]
        )
        n_s0 = channel_params["seed_frac0"] * n_p0
        eta_L = channel_params["R_L"] * channel_params["A"] / channel_params["L"]

        result = hall_solver.solve_equilibrium_batch(
            flow_speed=channel_params["u0"], gas_temperature=channel_params["Tp0"],
            gas_number_density=n_p0, seed_number_density=n_s0,
            magnetic_field=channel_params["B0"], load_resistivity=eta_L,
        )
        plasma = hall_solver.solve_equilibrium(
            flow_speed=channel_params["u0"], gas_temperature=channel_params["Tp0"],
            gas_number_density=n_p0, seed_number_density=n_s0,
            magnetic_field=channel_params["B0"], load_resistivity=eta_L,
        )
        assert float(result.electron_temperature) == pytest.approx(plasma.electron_temperature, rel=1e-9)
        assert float(result.hall_parameter) == pytest.approx(plasma.hall_parameter, rel=1e-9)


class TestSingleSlice:
    """The local plasma solve for a single slice, isolated from the axial march."""

    def _solve_reference_inlet(self, main_ref, params):
        eta_L = params["R_L"] * params["A"] / params["L"]
        n_p0 = params["p0"] / (main_ref.C.k * params["Tp0"])
        n_s0 = params["seed_frac0"] * n_p0
        return main_ref.solve_plasma_properties(
            params["u0"], params["Tp0"], n_p0, n_s0, params["B0"], eta_L
        ), n_p0, n_s0

    def test_matches_reference_at_inlet(self, main_ref, hall_solver, channel_params):
        expected, n_p0, n_s0 = self._solve_reference_inlet(main_ref, channel_params)
        eta_L = channel_params["R_L"] * channel_params["A"] / channel_params["L"]

        plasma = hall_solver.solve_equilibrium(
            flow_speed=channel_params["u0"],
            gas_temperature=channel_params["Tp0"],
            gas_number_density=n_p0,
            seed_number_density=n_s0,
            magnetic_field=channel_params["B0"],
            load_resistivity=eta_L,
        )

        assert plasma.electron_temperature == pytest.approx(expected["Te"], rel=1e-9)
        assert plasma.electron_number_density == pytest.approx(expected["ne"], rel=1e-9)
        assert plasma.hall_parameter == pytest.approx(expected["beta"], rel=1e-9)
        assert plasma.axial_electric_field == pytest.approx(expected["Ex"], rel=1e-9)
        assert plasma.ohmic_power_density == pytest.approx(expected["S_ohm"], rel=1e-9)
        assert plasma.load_power_density == pytest.approx(expected["S_load"], rel=1e-9)
        assert plasma.current_density[0] == pytest.approx(expected["Jx"], rel=1e-9)
        assert plasma.current_density[1] == pytest.approx(expected["Jy"], rel=1e-9)


class TestFullMarch:
    """
    The complete axial march.

    tests/reference_main.py's march_channel() is NOT a valid oracle for full axial
    profiles: it has a known energy-accounting bug in its own axial ODEs (fixed here,
    see Derivation.md's "Correction" note), so its trajectory now genuinely differs
    from HallSolver.march()'s. What's still unchanged -- and still checked here -- is
    the local, literature-derived Ohm's-law/Saha closure `solve_equilibrium` uses at every
    point the (corrected) march actually visits. Energy conservation of the new axial
    ODEs is checked separately in tests/test_energy_accounting.py.
    """

    def test_local_closure_matches_reference_at_every_visited_state(self, main_ref, hall_solver, channel_params):
        """Feed main_ref.solve_plasma_properties the exact (u, Tp, np, ns) our march visits
        at each slice -- the local closure it computes should still match ours exactly,
        even though the trajectory used to reach each state now differs."""
        # B0=0.1 keeps this operating point subsonic all the way through (see
        # test_energy_accounting.py for how that was determined), so the full requested
        # trajectory is available to compare.
        magnetic_field = 0.1
        num_slices = 40
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
        assert not channel.choked
        assert len(channel) == num_slices

        eta_L = channel_params["R_L"] * channel_params["A"] / channel_params["L"]
        for state in channel.states:
            seed_number_density = channel_params["seed_frac0"] * state.gas_number_density
            expected = main_ref.solve_plasma_properties(
                state.flow_speed, state.gas_temperature, state.gas_number_density,
                seed_number_density, magnetic_field, eta_L,
            )
            assert state.electron_temperature == pytest.approx(expected["Te"], rel=1e-9)
            assert state.electron_number_density == pytest.approx(expected["ne"], rel=1e-9)
            assert state.hall_parameter == pytest.approx(expected["beta"], rel=1e-9)
            assert state.axial_electric_field == pytest.approx(expected["Ex"], rel=1e-9)
            assert state.ohmic_power_density == pytest.approx(expected["S_ohm"], rel=1e-9)
            assert state.load_power_density == pytest.approx(expected["S_load"], rel=1e-9)
            assert state.current_density[0] == pytest.approx(expected["Jx"], rel=1e-9)
            assert state.current_density[1] == pytest.approx(expected["Jy"], rel=1e-9)

    def test_channel_length_and_positions_when_not_choked(self, hall_solver, channel_params):
        channel = hall_solver.march(
            num_slices=channel_params["num_slices"],
            length=channel_params["L"],
            area=channel_params["A"],
            inlet_speed=channel_params["u0"],
            inlet_pressure=channel_params["p0"],
            inlet_gas_temperature=channel_params["Tp0"],
            magnetic_field=0.1,  # stays subsonic (see test above)
            load_resistance=channel_params["R_L"],
            inlet_seed_fraction=channel_params["seed_frac0"],
        )
        assert not channel.choked
        assert len(channel) == channel_params["num_slices"]
        assert channel.x[0] == 0.0
        assert channel.x[-1] == pytest.approx(channel_params["L"])

    def test_choked_flag_and_truncated_channel(self, hall_solver, channel_params):
        """At channel_params' default B0=0.5, Ohmic heating drives the flow to M=1 well
        before the outlet (Rayleigh-flow choking -- see Derivation.md). The march must
        stop there rather than step through the singularity."""
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
        assert 0 < len(channel) < channel_params["num_slices"]
        assert len(channel.x) == len(channel)
        assert channel.x[-1] < channel_params["L"]

        gas_type = hall_solver._gas_type
        last = channel.states[-1]
        from scipy import constants
        mach_last = np.sqrt(
            gas_type.particle_mass * last.flow_speed ** 2
            / (gas_type.heat_capacity_ratio * constants.k * last.gas_temperature)
        )
        assert mach_last == pytest.approx(hall_solver.max_mach_number, abs=1e-6)
