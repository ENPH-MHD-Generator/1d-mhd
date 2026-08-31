"""
Reproduce Friedberg's own Sec. 7 design-curve figures (Figs. 2-4): for a target
converted power density S_C and a scan of B0, find the electron temperature at which
the plasma sits EXACTLY at marginal stability while delivering EXACTLY S_C, and see how
the resulting design point moves. This is a different question from every sweep in
stability_2d_boundaries.py/etc. -- those all ask "is this local state stable"; this asks
"what's the design point that's exactly at the edge, for a given field and power
target." The actual procedure (magnetohydrodynamics.stability.MarginalDesignPointSolver)
doubles as a validation of this whole codebase's physics against the paper's own
published numbers -- see print_section_7_3_validation.

Also adds a seed-species comparison (Caesium/Potassium/Sodium, via
magnetohydrodynamics.presets.all_seed_types) that isn't in the paper -- a natural
"what if we picked a different seed gas" question a real design process would ask.

Run with:
    uv run python examples/design_curves.py            # saves PNGs, no window
    uv run python examples/design_curves.py --show      # also opens windows
    uv run python examples/design_curves.py --no-save   # windows only
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from plotting_utils import parse_show_save_args, save_and_show
from scipy import constants

from magnetohydrodynamics.presets import all_seed_types, default_gas_type, default_seed_type
from magnetohydrodynamics.stability import MarginalDesignPointSolver
from magnetohydrodynamics.thermophysics.gas_type import GasType

# Friedberg Sec. 7.1's own reference furnace conditions (HTGR + Laval nozzle, M=1.8).
REFERENCE_GAS_TEMPERATURE = 481.0    # Tp [K]
REFERENCE_GAS_PRESSURE = 0.801e6     # pp [Pa]
REFERENCE_MACH_NUMBER = 1.8          # M
REFERENCE_FLOW_SPEED = 735.0         # vp [m/s]
REFERENCE_GAS_NUMBER_DENSITY = REFERENCE_GAS_PRESSURE / (constants.k * REFERENCE_GAS_TEMPERATURE)  # np [m^-3]
REFERENCE_KWARGS = dict(
    gas_temperature=REFERENCE_GAS_TEMPERATURE, gas_number_density=REFERENCE_GAS_NUMBER_DENSITY,
    flow_speed=REFERENCE_FLOW_SPEED, mach_number=REFERENCE_MACH_NUMBER,
)

B0_VALUES = np.linspace(3.0, 20.0, 60)          # matches the paper's own Figs. 2-4 scan range
POWER_DENSITIES = [100e6, 200e6, 300e6]          # S_C [W/m^3], matching the paper's three curves


def plot_ohmic_to_load_ratio(solver: MarginalDesignPointSolver, seed_name: str) -> plt.Figure:
    """Fig. 2 analog: S_Omega/S_L vs B0, one curve per target power density."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for target_power_density in POWER_DENSITIES:
        curve = solver.sweep(B0_VALUES, target_power_density)
        ax.plot(B0_VALUES, curve["ohmic_to_load_ratio"], linewidth=2, label=f"$S_C$={target_power_density / 1e6:.0f} MW/m$^3$")
    ax.set_xlabel(r"$B_0$ [T]")
    ax.set_ylabel(r"$S_\Omega/S_L$")
    ax.set_title(f"Ohmic-to-load power ratio at marginal stability ({seed_name} seed) -- Fig. 2 analog")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_conductivity(solver: MarginalDesignPointSolver, seed_name: str) -> plt.Figure:
    """Fig. 3 analog: sigma vs B0, one curve per target power density."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for target_power_density in POWER_DENSITIES:
        curve = solver.sweep(B0_VALUES, target_power_density)
        ax.plot(B0_VALUES, curve["conductivity"], linewidth=2, label=f"$S_C$={target_power_density / 1e6:.0f} MW/m$^3$")
    ax.set_xlabel(r"$B_0$ [T]")
    ax.set_ylabel(r"$\sigma$ [S/m]")
    ax.set_title(f"Electrical conductivity at marginal stability ({seed_name} seed) -- Fig. 3 analog")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_design_point_summary(solver: MarginalDesignPointSolver, seed_name: str) -> plt.Figure:
    """Fig. 4 analog: 2x2 panel of (n_e, Te, beta, 1-f_I) vs B0, one curve per target
    power density."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    panels = [
        (axes[0, 0], "n_e", r"$n_e$ [m$^{-3}$]"),
        (axes[0, 1], "electron_temperature", r"$T_e$ [K]"),
        (axes[1, 0], "beta", r"$\beta$"),
        (axes[1, 1], "one_minus_f_I", r"$1-f_I$"),
    ]
    for target_power_density in POWER_DENSITIES:
        curve = solver.sweep(B0_VALUES, target_power_density)
        for ax, key, _ylabel in panels:
            ax.plot(B0_VALUES, curve[key], linewidth=2, label=f"$S_C$={target_power_density / 1e6:.0f} MW/m$^3$")
    for ax, _key, ylabel in panels:
        ax.set_xlabel(r"$B_0$ [T]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[1, 1].set_yscale("log")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Design point vs $B_0$ at marginal stability ({seed_name} seed) -- Fig. 4 analog")
    fig.tight_layout()
    return fig


def plot_seed_species_comparison(gas_type: GasType, target_power_density: float = 100e6) -> plt.Figure:
    """Not in the paper: S_Omega/S_L and sigma vs B0, one curve per seed species, at a
    fixed target power density -- directly answers "how much does the seed gas choice
    shift the design curves"."""
    fig, (ax_ratio, ax_sigma) = plt.subplots(1, 2, figsize=(12, 5))
    for seed_type in all_seed_types():
        solver = MarginalDesignPointSolver(gas_type, seed_type, **REFERENCE_KWARGS)
        curve = solver.sweep(B0_VALUES, target_power_density)
        ax_ratio.plot(B0_VALUES, curve["ohmic_to_load_ratio"], linewidth=2, label=seed_type.name)
        ax_sigma.plot(B0_VALUES, curve["conductivity"], linewidth=2, label=seed_type.name)
    ax_ratio.set_xlabel(r"$B_0$ [T]")
    ax_ratio.set_ylabel(r"$S_\Omega/S_L$")
    ax_ratio.grid(True, alpha=0.3)
    ax_ratio.legend()
    ax_sigma.set_xlabel(r"$B_0$ [T]")
    ax_sigma.set_ylabel(r"$\sigma$ [S/m]")
    ax_sigma.grid(True, alpha=0.3)
    ax_sigma.legend()
    fig.suptitle(f"Seed species comparison at marginal stability ($S_C$={target_power_density / 1e6:.0f} MW/m$^3$)")
    fig.tight_layout()
    return fig


def print_section_7_3_validation(solver: MarginalDesignPointSolver) -> None:
    """Solves the paper's own worked example (B0=8T, S_C=100 MW/m^3, Sec. 7.3) and
    prints it alongside the paper's stated values. ~10-15% deviation is expected here,
    not an implementation error -- already established in
    tests/test_design_point.py's own Sec. 7.3 check, which traced this to rounding
    amplified through the alpha^2 term, starting from only 4-significant-figure
    published intermediates."""
    result = solver.solve(magnetic_field=8.0, target_power_density=100e6)
    print()
    print("Sec. 7.3 validation (B0=8T, S_C=100 MW/m^3, Caesium seed)")
    print("~10-15% deviation expected -- see this function's docstring:")
    print(f"  Te           : {result.electron_temperature:9.1f} K        (paper: 4223 K)")
    print(f"  Te/Tp        : {result.electron_temperature / REFERENCE_GAS_TEMPERATURE:9.3f}         (paper: 8.783)")
    print(f"  beta         : {result.beta:9.3f}         (paper: 8.249)")
    print(f"  S_Omega/S_L  : {result.ohmic_to_load_ratio:9.4f}         (paper: 0.3720)")
    print(f"  sigma        : {result.conductivity:9.2f} S/m     (paper: 12.35 mho/m)")
    print(f"  1-f_I        : {result.one_minus_f_I:9.5f}         (paper: 0.01679)")
    print(f"  n_e          : {result.n_e:9.3e} m^-3    (paper: 7.478e19)")
    print(f"  n_e/n_p      : {result.n_e / REFERENCE_GAS_NUMBER_DENSITY:9.3e}    (paper: 6.180e-7)")


def main() -> None:
    args = parse_show_save_args(__doc__)
    gas_type = default_gas_type()
    seed_type = default_seed_type()
    solver = MarginalDesignPointSolver(gas_type, seed_type, **REFERENCE_KWARGS)

    figures = [
        ("ohmic_to_load_ratio", plot_ohmic_to_load_ratio(solver, seed_type.name)),
        ("conductivity", plot_conductivity(solver, seed_type.name)),
        ("design_point_summary", plot_design_point_summary(solver, seed_type.name)),
        ("seed_species_comparison", plot_seed_species_comparison(gas_type)),
    ]
    print_section_7_3_validation(solver)
    save_and_show(figures, args)


if __name__ == "__main__":
    main()
