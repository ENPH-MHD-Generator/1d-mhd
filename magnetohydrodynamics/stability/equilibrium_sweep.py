from __future__ import annotations

import numpy as np
from scipy import constants
from scipy.optimize import brentq

from magnetohydrodynamics.solver.hall_solver import HallSolver
from magnetohydrodynamics.stability.friedberg_criterion import FriedbergAsymptoticCriterion, FriedbergCriterion
from magnetohydrodynamics.stability.operating_point import OperatingPoint


class EquilibriumSweep:
    """Everything this package computes by solving a HallSolver equilibrium across a
    sweep of physical-knob values, bundled with the three inputs every one of these
    computations shares: which solver, which base OperatingPoint to vary from, and
    which seed's ionisation potential to evaluate the stability criteria at."""

    def __init__(self, hall_solver: HallSolver, base: OperatingPoint, ionization_potential: float):
        self._hall_solver = hall_solver
        self._base = base
        self._ionization_potential = ionization_potential

    def _solve(self, **overrides) -> tuple[dict, dict]:
        """resolve `overrides` against `base` and solve the batch equilibrium there --
        the shared first step of every method below. Returns (resolved_point, result)."""
        point = self._base.resolve(**overrides)
        result = self._hall_solver.solve_equilibrium_batch(**point)
        return point, result

    def grid(self, x_key: str, x_values: np.ndarray, y_key: str, y_values: np.ndarray) -> dict[str, np.ndarray]:
        """Solve the local equilibrium at every (x, y) combination of two physical-knob
        sweep axes and evaluate both stability criteria there, via a single vectorized
        `solve_equilibrium_batch` call instead of one equilibrium (and one Plasma
        object) per grid point. Measured speedup on a 250x250 grid: ~5.5s -> ~0.02s --
        the physics is identical either way (`solve_equilibrium_batch` runs the exact
        same `_iterate_equilibrium` fixed-point loop `HallSolver.solve_equilibrium`
        does), the win is purely from replacing 62,500 Python-level calls with one call
        operating on 62,500-element arrays.

        Returns 2-D arrays of shape (len(y_values), len(x_values))."""
        X, Y = np.meshgrid(x_values, y_values)
        point, result = self._solve(**{x_key: X, y_key: Y})

        ionization_fraction = result["electron_number_density"] / result["seed_number_density"]
        beta = result["hall_parameter"]
        gas_temperature = point["gas_temperature"]

        exact = FriedbergCriterion()
        asymptotic = FriedbergAsymptoticCriterion()
        margin = exact.stability_margin(beta, result["electron_temperature"], gas_temperature, self._ionization_potential, ionization_fraction)
        margin_asymptotic = asymptotic.stability_margin(beta, result["electron_temperature"], gas_temperature, self._ionization_potential, ionization_fraction)

        return dict(
            beta=beta,
            beta_crit=exact.critical_hall_parameter(result["electron_temperature"], gas_temperature, self._ionization_potential, ionization_fraction),
            beta_crit_asymptotic=asymptotic.critical_hall_parameter(result["electron_temperature"], gas_temperature, self._ionization_potential, ionization_fraction),
            margin=margin,
            margin_asymptotic=margin_asymptotic,
            Te=result["electron_temperature"],
            ionization_fraction=ionization_fraction,
            stable=margin >= 1.0,
        )

    def matched_load(self, seed_fraction, magnetic_field, gas_temperature, iters: int = 15) -> dict:
        """Vectorized Friedberg (6.10) matched-load (Z=sqrt(1+beta^2)) equilibrium
        solve -- the array analogue of a scalar outer fixed-point loop (one
        HallSolver.solve_equilibrium call per grid point per outer iteration): `iters`
        rounds of `solve_equilibrium_batch` instead, solving an entire grid's worth of
        points (any broadcastable shape, e.g. a 3-D meshgrid) in one pass. This is what
        makes a dense 3-D (seed_fraction, B0, Tp) grid affordable at all -- see
        `volume_grid`'s docstring for the measured cost difference.

        KNOWN LIMITATION (found while adding test coverage, not fixed here -- it's a
        numerics question needing its own investigation, out of scope for the refactor
        that surfaced it): this is a plain, undamped Picard iteration on
        load_resistivity, with no relaxation. Checked numerically over a grid spanning
        volume_grid's typical ranges: roughly HALF of points do not converge to a fixed
        point at all, even after 200 iterations or with load_resistivity itself
        relaxed as low as 0.02 -- they settle into a stable period-2 limit cycle
        (load_resistivity alternates between two distinct values indefinitely) instead.
        Other points converge cleanly within ~10 iterations with no relaxation at all
        (e.g. seed_fraction~2e-4, B0~1.7, Tp~18 K), so this isn't uniformly slow
        convergence -- it looks like genuine bistability in Z at fixed
        (seed_fraction, B0, Tp) that the naive map bounces between rather than a single
        map that just needs more/gentler iterations. Whatever `result` this method
        returns for a non-converged point reflects an arbitrary phase of that cycle
        (whichever `iters` lands on), not a self-consistent matched-load solution -- so
        plots built from `volume_grid` may currently be silently wrong at close to half
        their grid points. Flagged for follow-up, not addressed here."""
        gas_number_density = self._base.p0 / (constants.k * gas_temperature)
        seed_number_density = seed_fraction * gas_number_density
        load_resistivity = self._base.load_resistivity
        result = None
        for _ in range(iters):
            result = self._hall_solver.solve_equilibrium_batch(
                flow_speed=self._base.v0, gas_temperature=gas_temperature, gas_number_density=gas_number_density,
                seed_number_density=seed_number_density, magnetic_field=magnetic_field, load_resistivity=load_resistivity,
            )
            load_resistivity = np.sqrt(1.0 + result["hall_parameter"] ** 2) * result["resistivity"]
        return result

    def volume_grid(self, seed_fraction_values: np.ndarray, b0_values: np.ndarray, tp_values: np.ndarray) -> dict:
        """Precomputed 3-D equilibrium/stability table over (seed_fraction, B0, Tp),
        solved with the Friedberg (6.10) matched-load Z=sqrt(1+beta^2) policy
        throughout -- load resistivity is never a free axis here (see `matched_load`;
        Z=beta^2+1, maximizing power outright, was tried first and abandoned: it
        pushed the whole system toward physically unreasonable sub-100K Tp at high
        field).

        Solving the WHOLE grid is one vectorized pass via `matched_load`, however
        fine -- a 40x40x40 grid (64,000 points) solves in well under a second, versus
        the ~16s an earlier, much coarser (14x14, with an internal per-cell
        root-search) 2-D version of this analysis took. That's what makes a dense
        enough 3-D grid to resolve real structure (like the Tp ~ 2000-5000 K
        re-entrant unstable pocket found during development) actually affordable,
        where a scan-and-bisect search extended to 3-D would not have been.

        Returns a dict of 3-D arrays (shape (len(seed_fraction), len(B0), len(Tp))):
        margin (beta_crit/beta), stable (margin>=1), Te, ionization_fraction,
        load_power_density."""
        SF, B0, TP = np.meshgrid(seed_fraction_values, b0_values, tp_values, indexing="ij")
        result = self.matched_load(SF, B0, TP)

        ionization_fraction = result["electron_number_density"] / result["seed_number_density"]
        criterion = FriedbergCriterion()
        margin = criterion.stability_margin(result["hall_parameter"], result["electron_temperature"], TP, self._ionization_potential, ionization_fraction)

        return dict(
            margin=margin,
            stable=margin >= 1.0,
            Te=result["electron_temperature"],
            ionization_fraction=ionization_fraction,
            load_power_density=result["load_power_density"],
        )

    def margin_minus_one(self, **overrides) -> np.ndarray:
        """beta_crit/beta - 1, evaluated via solve_equilibrium_batch for whatever
        combination of scalar/array physical-knob `overrides` is given (see
        `OperatingPoint.resolve`). Used both for big vectorized scans and, with scalar
        args, as the objective handed to brentq for individual root refinement."""
        point, result = self._solve(**overrides)
        ionization_fraction = result["electron_number_density"] / result["seed_number_density"]
        margin = FriedbergCriterion().stability_margin(
            result["hall_parameter"], result["electron_temperature"], point["gas_temperature"],
            self._ionization_potential, ionization_fraction,
        )
        return margin - 1.0

    @staticmethod
    def find_crossings(margin_minus_one_values: np.ndarray, axis_values: np.ndarray, objective) -> list[float]:
        """Every value of `axis_values` (refined by bisection) where a precomputed 1-D
        array of (beta_crit/beta - 1) changes sign -- the vectorized-scan counterpart
        of scanning-then-bisecting one point at a time. `objective(value) ->
        beta_crit/beta - 1` is only called a handful of times per detected crossing
        (brentq's own refinement), not once per scan point. A static method: it's a
        generic sign-change/bisection utility with no dependency on this sweep's own
        hall_solver/base/ionization_potential, grouped here because it's always used
        alongside `critical_load_resistivity_surface`.

        Scanning for *all* sign changes -- not just checking the two endpoints --
        matters because this system can have a genuine stability *window*
        (unstable-stable-unstable) rather than a single threshold: checking only the
        endpoints can't tell "never crosses" apart from "crosses an even number of
        times, same sign at both ends"."""
        sign_changes = np.where(np.diff(np.sign(margin_minus_one_values)) != 0)[0]
        roots = []
        for k in sign_changes:
            try:
                roots.append(brentq(objective, axis_values[k], axis_values[k + 1], xtol=1e-4 * axis_values[k] + 1e-8))
            except (ValueError, RuntimeError):
                pass
        return sorted(roots)

    def critical_load_resistivity_surface(
            self, seed_fraction_values: np.ndarray, b0_values: np.ndarray,
            load_resistivity_bracket: tuple[float, float] = (1e-4, 20.0), scan_points: int = 60,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """For each (seed_fraction, B0), finds the critical load resistivity -- the
        stability boundary surface over the three inputs that matter most directly:
        seed fraction and B0 together set beta and (via Saha/energy balance) f_I, and
        load resistivity is the next most consequential *free dial* (as opposed to Tp
        or the flow conditions, which are usually fixed by the working
        gas/environment rather than tuned) -- it sets Z = eta_L/eta, which Delta_T
        depends on through the whole current/heating loop (eq. 3.14).

        Returns (lower, upper, lower_power, upper_power): the boundary height(s) in
        Ohm*m -- `upper` is all-NaN unless the margin actually re-crosses 1 at higher
        resistivity (checked numerically: not observed in the range tested here, but
        handled the same way as a B0 stability window for robustness) -- plus the load
        power density S_L *at* each boundary point, so the surface's color can carry a
        genuinely independent piece of information instead of re-deriving the height.

        The scan itself (the expensive part -- every (B0, seed_fraction,
        load_resistivity) combination) is one vectorized `margin_minus_one` call;
        only the brentq refinement of each detected crossing falls back to a handful
        of individual (still batch-based, just scalar-shaped) evaluations."""
        scan_values = np.logspace(np.log10(load_resistivity_bracket[0]), np.log10(load_resistivity_bracket[1]), scan_points)
        B0, SF, LR = np.meshgrid(b0_values, seed_fraction_values, scan_values, indexing="ij")
        values = self.margin_minus_one(B0=B0, seed_fraction=SF, load_resistivity=LR)  # shape (len(b0), len(sf), scan_points)

        shape = (len(b0_values), len(seed_fraction_values))
        lower = np.full(shape, np.nan)
        upper = np.full(shape, np.nan)
        lower_power = np.full(shape, np.nan)
        upper_power = np.full(shape, np.nan)

        for i, b0 in enumerate(b0_values):
            for j, seed_fraction in enumerate(seed_fraction_values):
                def objective(lr, b0=b0, seed_fraction=seed_fraction) -> float:
                    return float(self.margin_minus_one(B0=b0, seed_fraction=seed_fraction, load_resistivity=lr))

                roots = self.find_crossings(values[i, j, :], scan_values, objective)
                if not roots:
                    continue
                lower[i, j] = roots[0]
                lower_power[i, j] = float(self._solve(B0=b0, seed_fraction=seed_fraction, load_resistivity=roots[0])[1]["load_power_density"])
                if len(roots) > 1:
                    upper[i, j] = roots[-1]
                    upper_power[i, j] = float(self._solve(B0=b0, seed_fraction=seed_fraction, load_resistivity=roots[-1])[1]["load_power_density"])
        return lower, upper, lower_power, upper_power
