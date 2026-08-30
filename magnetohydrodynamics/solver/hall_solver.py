from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import constants

from magnetohydrodynamics.ionization.ionization_model import IonizationModel
from magnetohydrodynamics.ionization.seed_type import SeedType
from magnetohydrodynamics.plasma.plasma import Plasma
from magnetohydrodynamics.state.gas_state import GasState
from magnetohydrodynamics.state.geometry import Geometry
from magnetohydrodynamics.state.ionization_state import IonizationState
from magnetohydrodynamics.thermophysics.gas_type import GasType
from magnetohydrodynamics.transport.transport_model import TransportModel


@dataclass
class Channel(Geometry):
    """The solved axial profile of a linear Hall generator: one Plasma slice per station."""
    x: np.ndarray
    states: list[Plasma] = field(default_factory=list)
    load_resistivity: float = 0.0

    def __len__(self) -> int:
        return len(self.states)


class HallSolver:
    """
    1-D marcher for a constant-area linear Hall MHD generator.

    Reproduces, slice by slice, the local closure of Ohm's law and the electron
    energy balance (Friedberg secs. 3.7-3.14) together with the axial mass /
    momentum / energy conservation used to step primary-gas speed and
    temperature down the channel. See Derivation.md.
    """

    def __init__(
            self,
            gas_type: GasType,
            seed_type: SeedType,
            transport_model: TransportModel,
            ionization_model: IonizationModel,
            *,
            relax: float = 0.5,
            max_iter: int = 12,
    ):
        self._gas_type = gas_type
        self._seed_type = seed_type
        self._transport_model = transport_model
        self._ionization_model = ionization_model
        self._relax = relax
        self._max_iter = max_iter

    def _solve_slice(
            self,
            flow_speed: float,
            gas_temperature: float,
            gas_number_density: float,
            seed_number_density: float,
            magnetic_field: float,
            load_resistivity: float,
    ) -> Plasma:
        """Local plasma properties for a single slice (Ey=0, Jz=Ez=0, B=B0)."""
        m_particle = self._gas_type.particle_mass

        # initial guesses
        electron_temperature = gas_temperature * 8.783
        electron_number_density = np.clip(1.0 * seed_number_density, 1e12, 1.0 * seed_number_density)

        nu_M = nu_E = hall_parameter = load_ratio = 0.0
        current_x = current_y = axial_field = 0.0
        ohmic_power_density = load_power_density = 0.0

        for _ in range(self._max_iter):
            nu_M = self._transport_model.get_momentum_transfer_frequency(electron_temperature, gas_number_density)
            nu_E = self._transport_model.get_energy_transfer_frequency(electron_temperature, gas_number_density)

            resistivity = constants.electron_mass * nu_M / (constants.e ** 2 * electron_number_density)
            hall_parameter = constants.e * magnetic_field / (constants.electron_mass * nu_M)
            load_ratio = load_resistivity / resistivity

            denom = hall_parameter ** 2 + 1.0 + load_ratio
            current_x = (hall_parameter ** 2 / denom) * constants.e * electron_number_density * flow_speed
            current_y = -(hall_parameter * (1.0 + load_ratio) / denom) * constants.e * electron_number_density * flow_speed
            axial_field_over_resistivity = -(hall_parameter ** 2 * load_ratio / denom) * constants.e * electron_number_density * flow_speed
            axial_field = axial_field_over_resistivity * resistivity

            ohmic_power_density = resistivity * (current_x ** 2 + current_y ** 2)
            load_power_density = -axial_field * current_x

            mach = np.sqrt((3.0 / 5.0) * (m_particle * flow_speed ** 2) / (constants.k * gas_temperature))
            delta_t = (5.0 * mach ** 2 / 9.0) * (hall_parameter ** 2 * (hall_parameter ** 2 + (1.0 + load_ratio) ** 2)) / (denom ** 2)
            electron_temperature_new = gas_temperature * (1.0 + delta_t)

            electron_number_density_new = self._ionization_model.get_electron_density(
                electron_temperature_new, seed_number_density
            )

            electron_temperature = self._relax * electron_temperature_new + (1.0 - self._relax) * electron_temperature
            electron_number_density = self._relax * electron_number_density_new + (1.0 - self._relax) * electron_number_density

        gas_state = GasState(
            gas_type=self._gas_type,
            gas_temperature=float(gas_temperature),
            gas_pressure=float(gas_number_density * constants.k * gas_temperature),
            flow_speed=float(flow_speed),
            gas_number_density=float(gas_number_density),
            magnetic_field=float(magnetic_field),
        )
        ionization_state = IonizationState(
            seed_type=self._seed_type,
            electron_temperature=float(electron_temperature),
            electron_number_density=float(electron_number_density),
            momentum_transfer_frequency=float(nu_M),
            energy_transfer_frequency=float(nu_E),
            current_density=(float(current_x), float(current_y)),
            axial_electric_field=float(axial_field),
            ohmic_power_density=float(ohmic_power_density),
            load_power_density=float(load_power_density),
        )
        return Plasma(ionization_state=ionization_state, gas_state=gas_state)

    def march(
            self,
            num_slices: int,
            length: float,
            area: float,
            inlet_speed: float,
            inlet_pressure: float,
            inlet_gas_temperature: float,
            magnetic_field: float,
            load_resistance: float,
            inlet_seed_fraction: float,
    ) -> Channel:
        """Constant-area 1-D march with an inner plasma solve per slice."""
        x = np.linspace(0.0, length, num_slices)
        dx = length / max(1, (num_slices - 1))

        load_resistivity = load_resistance * area / length

        inlet_gas_number_density = inlet_pressure / (constants.k * inlet_gas_temperature)
        inlet_seed_number_density = inlet_seed_fraction * inlet_gas_number_density
        number_flux = inlet_gas_number_density * inlet_speed  # n_p * u, constant for constant area

        m_particle = self._gas_type.particle_mass
        molar_heat_capacity = self._gas_type.molar_heat_capacity

        flow_speed = inlet_speed
        gas_temperature = inlet_gas_temperature

        states: list[Plasma] = []
        for i in range(num_slices):
            gas_number_density = number_flux / max(1e-6, flow_speed)
            seed_number_density = inlet_seed_number_density * (gas_number_density / inlet_gas_number_density)

            plasma = self._solve_slice(
                flow_speed=flow_speed,
                gas_temperature=gas_temperature,
                gas_number_density=gas_number_density,
                seed_number_density=seed_number_density,
                magnetic_field=magnetic_field,
                load_resistivity=load_resistivity,
            )
            states.append(plasma)

            if i == num_slices - 1:
                break

            current_x, current_y = plasma.current_density
            dTdx = (plasma.ohmic_power_density - plasma.load_power_density) / (
                m_particle * gas_number_density * flow_speed * molar_heat_capacity
            )
            denom = m_particle * number_flux + (number_flux * constants.k * gas_temperature) / flow_speed ** 2
            dudx = (current_y * magnetic_field - (number_flux * constants.k / flow_speed) * dTdx) / denom

            # explicit Euler step
            gas_temperature = max(50.0, gas_temperature + dTdx * dx)
            flow_speed = max(1e-3, flow_speed + dudx * dx)

        return Channel(x=x, states=states, load_resistivity=load_resistivity)
