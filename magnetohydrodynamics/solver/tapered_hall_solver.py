from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import constants

from magnetohydrodynamics.plasma.plasma import Plasma
from magnetohydrodynamics.solver.hall_solver import HallSolver
from magnetohydrodynamics.state.geometry import Geometry
from magnetohydrodynamics.typing import Scalar


@dataclass(frozen=True)
class LinearTaper:
    """A channel whose cross-sectional area is exactly linear in x: one wall pair of
    the square inlet held fixed, the other diverging at a constant half-angle -- NOT a
    self-similar square taper (which would make area quadratic in x, and dA/dx a
    function of x rather than a single constant). See Derivation.md's "Variable-Area
    (Tapered) Channel" section for why this geometry was chosen and what it implies
    (the channel is only literally square at the inlet; it's rectangular downstream)."""
    inlet_area: float
    half_angle_deg: float = 0.0
    """Positive = diverging (area grows with x), negative = converging, 0 = the
    constant-area limit that TaperedHallSolver.march() must reproduce exactly."""

    @property
    def dA_dx(self) -> float:
        """Constant slope (this profile is linear in x, so the slope doesn't vary) --
        d(side)/dx = tan(half_angle) on the one diverging wall pair, and
        A = inlet_side * side(x), so dA/dx = inlet_side * tan(half_angle) = 2*sqrt(inlet_area)*tan(half_angle)."""
        return 2.0 * np.sqrt(self.inlet_area) * np.tan(np.radians(self.half_angle_deg))

    def area(self, x: Scalar) -> Scalar:
        return self.inlet_area + self.dA_dx * np.asarray(x, dtype=float)

    def local_half_angle_deg(self, x: Scalar) -> Scalar:
        """Constant here (linear profile) -- takes x so a future non-linear taper is a
        drop-in replacement without changing call sites that read this."""
        return np.full_like(np.asarray(x, dtype=float), self.half_angle_deg)

    @classmethod
    def from_exit_area(cls, inlet_area: float, exit_area: float, length: float) -> LinearTaper:
        """The more intuitive "what area do I want at the outlet" input style,
        converted to the canonical (inlet_area, half_angle_deg) form."""
        slope = (exit_area - inlet_area) / length
        half_angle_rad = np.arctan(slope / (2.0 * np.sqrt(inlet_area)))
        return cls(inlet_area=inlet_area, half_angle_deg=float(np.degrees(half_angle_rad)))


class DivergenceAngleWarning(UserWarning):
    """Raised as a warning (or, in TaperedHallSolver's `strict` mode, as an actual
    error -- Warning subclasses Exception, so `raise DivergenceAngleWarning(...)` is
    valid) when a LinearTaper's half-angle exceeds the solver's configured
    `max_half_angle_deg`: past that point, Ohm's law's dropped v_py/v_pz terms (see
    Derivation.md) are no longer a defensible approximation, and results shouldn't be
    trusted without re-deriving that closure for real."""


@dataclass
class TaperedChannel(Geometry):
    """The solved axial profile of a linearly-tapered linear Hall generator -- the
    variable-area counterpart to HallSolver's own `Channel`. Stores the `LinearTaper`
    itself (not a separately-tracked area array) and derives `area` as a property, so
    it can't drift out of sync with `x` -- the same "derive, don't duplicate" pattern
    as e.g. `Equilibrium.resistivity`/`Plasma.resistivity`."""
    x: np.ndarray
    taper: LinearTaper
    states: list[Plasma] = field(default_factory=list)
    load_resistivity: np.ndarray = field(default_factory=lambda: np.array([]))
    """Per-slice, unlike Channel.load_resistivity's single scalar -- eta_L = R_load*A(x)/length
    varies station to station once area does."""
    choked: bool = False
    """See Channel.choked's docstring -- identical meaning and cause here."""

    @property
    def area(self) -> np.ndarray:
        return np.asarray(self.taper.area(self.x))

    def __len__(self) -> int:
        return len(self.states)

    def to_dict(self) -> dict:
        """Same key names as Channel.to_dict() (so plotting helpers written against
        one work against the other wherever they don't touch `eta_L`/`area`
        specifically), plus the new `area` key. `eta_L` is an array here, unlike
        Channel.to_dict()'s scalar -- a real, accepted asymmetry between the two
        shapes (see Derivation.md)."""
        current_x = np.array([state.current_density[0] for state in self.states])
        current_y = np.array([state.current_density[1] for state in self.states])
        return dict(
            x=self.x,
            area=self.area,
            u=self["flow_speed"],
            Tp=self["gas_temperature"],
            p=self["gas_pressure"],
            np=self["gas_number_density"],
            Te=self["electron_temperature"],
            ne=self["electron_number_density"],
            beta=self["hall_parameter"],
            Jx=current_x,
            Jy=current_y,
            Ex=self["axial_electric_field"],
            S_ohm=self["ohmic_power_density"],
            S_load=self["load_power_density"],
            eta_L=self.load_resistivity,
            ns=self["seed_number_density"],
            f_I=self["ionization_fraction"],
        )


class TaperedHallSolver:
    """Variable-area (tapered-channel) counterpart to `HallSolver.march()` -- a
    SEPARATE solver, not a modification of `HallSolver`: it reuses `HallSolver`'s
    per-slice local closure (Ohm's law + Saha + electron energy balance, Friedberg
    3.2-3.14) unchanged, by composition, and only replaces the axial mass/momentum/
    energy stepping. See Derivation.md's "Variable-Area (Tapered) Channel" section for
    the full derivation this implements, and for why the mass/momentum/energy
    generalization is exact quasi-1D while Ohm's law's unchanged v_py=v_pz=0
    assumption is the one genuinely new approximation being made here."""

    def __init__(self, hall_solver: HallSolver, max_half_angle_deg: float = 15.0, *, strict: bool = False):
        """`hall_solver`: an EXISTING HallSolver (e.g. from build_default_hall_solver())
        -- reused as-is, not reconfigured, so this class never re-plumbs
        gas_type/seed_type/transport_model/ionization_model. `max_half_angle_deg`
        (default 15 degrees, a standard real-diffuser rule of thumb -- NOT
        independently sourced from Messerle or Sutton & Sherman, see Derivation.md) is
        the Ohm's-law validity guard: `march()` warns (or, if `strict`, raises)
        `DivergenceAngleWarning` when a taper's half-angle exceeds it."""
        self._hall_solver = hall_solver
        self._max_half_angle_deg = max_half_angle_deg
        self._strict = strict

    def march(
            self,
            num_slices: int,
            length: float,
            taper: LinearTaper,
            inlet_speed: float,
            inlet_pressure: float,
            inlet_gas_temperature: float,
            magnetic_field: float,
            load_resistance: float,
            inlet_seed_fraction: float,
    ) -> TaperedChannel:
        """Variable-area counterpart to HallSolver.march() -- same choking-aware
        stepping structure (including the subsonic/supersonic mirrored-threshold
        bisection), stepping the tapered dT_p/dx, dv_p/dx from Derivation.md instead
        of the zero-divergence ones."""
        if abs(taper.half_angle_deg) > self._max_half_angle_deg:
            message = (
                f"LinearTaper half_angle_deg={taper.half_angle_deg:g} exceeds max_half_angle_deg="
                f"{self._max_half_angle_deg:g} -- Ohm's law's dropped v_py/v_pz terms (Derivation.md) are no "
                "longer a defensible approximation at this divergence."
            )
            if self._strict:
                raise DivergenceAngleWarning(message)
            warnings.warn(message, DivergenceAngleWarning, stacklevel=2)

        # Area is linear (monotonic) in x, so only the far endpoint needs checking --
        # a converging (half_angle_deg < 0) taper over a long enough length can pinch
        # the channel to zero or negative area.
        if taper.area(length) <= 0.0:
            raise ValueError(
                f"taper.area(length={length:g}) = {taper.area(length):g} <= 0 -- this converging taper "
                "closes the channel before reaching the requested length."
            )

        dx = length / max(1, (num_slices - 1))

        inlet_gas_number_density = inlet_pressure / (constants.k * inlet_gas_temperature)
        inlet_seed_number_density = inlet_seed_fraction * inlet_gas_number_density
        # Psi = n_p*v_p*A, the TOTAL particle rate through the duct (particles/s) --
        # constant in x, unlike the local flux Phi(x) = Psi/A(x) computed per slice
        # below (which is what HallSolver.march()'s constant `number_flux` becomes
        # once area varies; Phi(x) reduces to that same constant when taper.dA_dx=0).
        psi = inlet_gas_number_density * inlet_speed * taper.inlet_area

        m_particle = self._hall_solver.gas_type.particle_mass
        gamma = self._hall_solver.gas_type.heat_capacity_ratio

        flow_speed = inlet_speed
        gas_temperature = inlet_gas_temperature
        position = 0.0

        states: list[Plasma] = []
        x_visited: list[float] = []
        load_resistivity_visited: list[float] = []
        choked = False
        for i in range(num_slices):
            area = float(taper.area(position))
            local_flux = psi / area
            gas_number_density = local_flux / max(1e-6, flow_speed)
            seed_number_density = inlet_seed_number_density * (gas_number_density / inlet_gas_number_density)
            load_resistivity = load_resistance * area / length

            plasma = self._hall_solver.solve_equilibrium(
                flow_speed=flow_speed,
                gas_temperature=gas_temperature,
                gas_number_density=gas_number_density,
                seed_number_density=seed_number_density,
                magnetic_field=magnetic_field,
                load_resistivity=load_resistivity,
            )
            states.append(plasma)
            x_visited.append(position)
            load_resistivity_visited.append(load_resistivity)

            if choked or i == num_slices - 1:
                break

            _current_x, current_y = plasma.current_density
            lorentz_work = current_y * magnetic_field  # J_y B_z
            a_over_a = taper.dA_dx / area  # A'(x)/A(x) at this slice

            # See Derivation.md's "Variable-Area (Tapered) Channel" section (sympy-verified
            # in tests/test_tapered_derivation_sympy.py) -- the second term on each line is
            # the new A'/A correction; setting it to 0 (taper.dA_dx=0) recovers
            # HallSolver.march()'s own dTdx/dudx exactly.
            denom = m_particle * flow_speed ** 2 - gamma * constants.k * gas_temperature
            dTdx = (gamma - 1.0) / constants.k * (
                plasma.ohmic_power_density * m_particle * flow_speed ** 2
                - gas_temperature * constants.k * (lorentz_work * flow_speed + plasma.ohmic_power_density)
            ) / (local_flux * denom) - (gamma - 1.0) * m_particle * flow_speed ** 2 * gas_temperature / denom * a_over_a
            dudx = flow_speed * (lorentz_work * flow_speed - plasma.ohmic_power_density * (gamma - 1.0)) / (
                local_flux * denom
            ) + gamma * constants.k * gas_temperature * flow_speed / denom * a_over_a

            def step_to(step_size, _T=gas_temperature, _u=flow_speed, _dT=dTdx, _du=dudx):
                T_new = max(50.0, _T + _dT * step_size)
                u_new = max(1e-3, _u + _du * step_size)
                mach_new = np.sqrt(m_particle * u_new ** 2 / (gamma * constants.k * T_new))
                return T_new, u_new, mach_new

            def bisect_step_size(threshold: float, safe_below: bool) -> float:
                """Same idiom as HallSolver.march()'s own helper of the same name --
                largest step_size in [0, dx] for which step_to's Mach number is still
                on the safe side of `threshold`."""
                lo, hi = 0.0, dx
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    _, _, mach_mid = step_to(mid)
                    is_safe = mach_mid < threshold if safe_below else mach_mid > threshold
                    if is_safe:
                        lo = mid
                    else:
                        hi = mid
                return lo

            # Identical structure to HallSolver.march()'s own choking logic -- the
            # shared denominator above is untouched by the area terms (see
            # Derivation.md), so the sign-flip-at-M=1 bisection carries over verbatim.
            current_mach = np.sqrt(m_particle * flow_speed ** 2 / (gamma * constants.k * gas_temperature))
            _, _, mach_full_step = step_to(dx)
            step_size = dx
            if current_mach > 1.0:
                min_mach_number = 2.0 - self._hall_solver.max_mach_number
                if mach_full_step <= min_mach_number:
                    step_size = bisect_step_size(min_mach_number, safe_below=False)
                    choked = True
            elif mach_full_step >= self._hall_solver.max_mach_number:
                step_size = bisect_step_size(self._hall_solver.max_mach_number, safe_below=True)
                choked = True

            gas_temperature, flow_speed, _ = step_to(step_size)
            position += step_size

        return TaperedChannel(
            x=np.array(x_visited), taper=taper, states=states,
            load_resistivity=np.array(load_resistivity_visited), choked=choked,
        )
