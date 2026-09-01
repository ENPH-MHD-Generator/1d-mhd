"""
The motivating demonstration for magnetohydrodynamics.solver.tapered_hall_solver:
a supersonic inlet that chokes almost immediately in a constant-area channel
(HallSolver.march) survives measurably longer -- and extracts measurably more
enthalpy -- once the channel is allowed to diverge gently (TaperedHallSolver.march),
the same mechanism a real supersonic diffuser uses to hold off M=1. See Derivation.md's
"Variable-Area (Tapered) Channel" section for the physics.

Both cases share every input except channel shape, so the comparison is apples to
apples: same inlet speed/pressure/temperature/field/seed fraction/load resistance/
channel length, only `half_angle_deg` differs (0 for the constant-area case).

Run with:
    uv run python examples/tapered_channel_profile.py            # saves PNGs, no window
    uv run python examples/tapered_channel_profile.py --show      # also opens windows
    uv run python examples/tapered_channel_profile.py --no-save   # windows only
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from channel_profile import plot_power_density, plot_stability_margin, plot_temperatures
from plotting_utils import parse_show_save_args, save_and_show

from magnetohydrodynamics.analysis import summarize_performance, summarize_tapered_performance
from magnetohydrodynamics.presets import build_default_hall_solver, default_seed_type
from magnetohydrodynamics.solver.tapered_hall_solver import LinearTaper, TaperedHallSolver
from magnetohydrodynamics.thermophysics.ideal_gas import IdealGas

# A short, strongly-coupled channel -- deliberately not the library's own default
# operating point (which is comfortably subsonic): this is the exact supersonic,
# almost-instant-choking regime the tapered solver exists to help with (see the
# conversation this script documents). Every input except channel shape is shared
# between the two marches below.
LENGTH = 0.02  # m -- about a tenth of the default operating point's 0.2 m
AREA = 48e-3 * 48e-3  # m^2, same square inlet as the default operating point
LOAD_RESISTIVITY = 0.0016  # Ohm*m -- near the marginal-stability point found while investigating this
COMMON_KWARGS: dict = dict(
    num_slices=2000,
    length=LENGTH,
    inlet_speed=1200.0,  # m/s -- M ~ 1.44 at the inlet, well supersonic
    inlet_pressure=10.01e3,
    inlet_gas_temperature=2000.0,
    magnetic_field=0.5,
    load_resistance=LOAD_RESISTIVITY * LENGTH / AREA,
    inlet_seed_fraction=6.18e-3,
)
HALF_ANGLE_DEG = 12.0  # comfortably inside TaperedHallSolver's default 15-degree "gentle divergence" guard


def plot_mach_number_comparison(flat_out: dict, tapered_out: dict, ideal_gas: IdealGas) -> plt.Figure:
    """The core demonstration: Mach number along the channel for both cases,
    overlaid. Both decelerate from the same supersonic inlet condition (Rayleigh-flow
    heat addition always pushes toward M=1 -- see Derivation.md), but the tapered
    case's M'(x) = 2 - max_mach_number floor (or simply not reaching it at all) sits
    much further down the channel."""
    flat_mach = ideal_gas.get_mach_number(flat_out["u"], flat_out["Tp"])
    tapered_mach = ideal_gas.get_mach_number(tapered_out["u"], tapered_out["Tp"])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(flat_out["x"] * 1000, flat_mach, label="constant area", linewidth=2)
    ax.plot(tapered_out["x"] * 1000, tapered_mach, label=f"{HALF_ANGLE_DEG:g}° taper", linewidth=2, linestyle="--")
    ax.axhline(1.0, color="black", linewidth=1.0, alpha=0.6, label="M = 1 (choking)")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("Mach number M")
    ax.set_title("Supersonic deceleration: constant area vs. a gentle taper")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_area_profile(taper: LinearTaper, length: float) -> plt.Figure:
    """What the taper actually looks like -- one wall pair fixed, the other
    diverging at a constant half-angle (see Derivation.md), so area is linear in x."""
    x = np.linspace(0.0, length, 100)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(x * 1000, np.asarray(taper.area(x)) * 1e4, linewidth=2)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel(r"Area [cm$^2$]")
    ax.set_title(f"Channel area profile ({taper.half_angle_deg:g}° half-angle)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_show_save_args(__doc__)
    hall_solver, gas_type = build_default_hall_solver()
    ideal_gas = IdealGas(gas_type)
    ionization_potential = default_seed_type().ionization_potential

    flat_channel = hall_solver.march(area=AREA, **COMMON_KWARGS)
    flat_out = flat_channel.to_dict()

    tapered_solver = TaperedHallSolver(hall_solver)
    taper = LinearTaper(inlet_area=AREA, half_angle_deg=HALF_ANGLE_DEG)
    tapered_channel = tapered_solver.march(taper=taper, **COMMON_KWARGS)
    tapered_out = tapered_channel.to_dict()

    flat_perf = summarize_performance(
        flat_out, A=AREA, cp=gas_type.molar_heat_capacity, m_p=gas_type.particle_mass, gamma=gas_type.heat_capacity_ratio,
    )
    tapered_perf = summarize_tapered_performance(
        tapered_out, cp=gas_type.molar_heat_capacity, m_p=gas_type.particle_mass, gamma=gas_type.heat_capacity_ratio,
    )

    print(f"Inlet Mach number: {float(ideal_gas.get_mach_number(COMMON_KWARGS['inlet_speed'], COMMON_KWARGS['inlet_gas_temperature'])):.3f}")
    print(f"Requested channel length: {LENGTH * 1000:.1f} mm\n")
    print(f"{'':25s} {'constant area':>15s} {f'{HALF_ANGLE_DEG:g}° taper':>15s}")
    print(f"{'choked?':25s} {flat_channel.choked!s:>15s} {tapered_channel.choked!s:>15s}")
    print(f"{'distance traveled [mm]':25s} {flat_out['x'][-1] * 1000:15.3f} {tapered_out['x'][-1] * 1000:15.3f}")
    print(f"{'load power [W]':25s} {flat_perf['PL']:15.1f} {tapered_perf['PL']:15.1f}")
    print(f"{'enthalpy extraction [%]':25s} {flat_perf['enthalpy_extraction_ratio'] * 100:15.3f} {tapered_perf['enthalpy_extraction_ratio'] * 100:15.3f}")

    figures = [
        ("tapered_demo_mach_comparison", plot_mach_number_comparison(flat_out, tapered_out, ideal_gas)),
        ("tapered_demo_area_profile", plot_area_profile(taper, LENGTH)),
        ("tapered_demo_flat_temperatures", plot_temperatures(flat_out)),
        ("tapered_demo_tapered_temperatures", plot_temperatures(tapered_out)),
        ("tapered_demo_flat_power", plot_power_density(flat_out)),
        ("tapered_demo_tapered_power", plot_power_density(tapered_out)),
        ("tapered_demo_flat_stability_margin", plot_stability_margin(flat_out, ionization_potential)),
        ("tapered_demo_tapered_stability_margin", plot_stability_margin(tapered_out, ionization_potential)),
    ]
    save_and_show(figures, args)


if __name__ == "__main__":
    main()
