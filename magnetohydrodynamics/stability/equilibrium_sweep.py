from __future__ import annotations

import numpy as np
from scipy import constants

from magnetohydrodynamics.operating_point import OperatingPoint
from magnetohydrodynamics.solver.equilibrium import Equilibrium, EquilibriumInputs
from magnetohydrodynamics.solver.hall_solver import HallSolver
from magnetohydrodynamics.stability.friedberg_criterion import FriedbergAsymptoticCriterion, FriedbergCriterion
from magnetohydrodynamics.stability.stability_grid import StabilityGrid, VolumeGrid
from magnetohydrodynamics.thermophysics.ideal_gas import IdealGas
from magnetohydrodynamics.typing import Scalar


class EquilibriumSweep:
    """Everything this package computes by solving a HallSolver equilibrium across a
    sweep of physical-knob values, bundled with the three inputs every one of these
    computations shares: which solver, which base OperatingPoint to vary from, and
    which seed's ionisation potential to evaluate the stability criteria at."""

    def __init__(self, hall_solver: HallSolver, base: OperatingPoint, ionization_potential: float):
        self._hall_solver = hall_solver
        self._base = base
        self._ionization_potential = ionization_potential

    def _solve(self, **overrides) -> tuple[EquilibriumInputs, Equilibrium]:
        """resolve `overrides` against `base` and solve the batch equilibrium there --
        the shared first step of every method below. Returns (resolved_point, result)."""
        point = self._base.resolve(**overrides)
        result = self._hall_solver.solve_equilibrium_batch(**point.as_kwargs())
        return point, result

    def grid(
            self, x_key: str, x_values: np.ndarray, y_key: str, y_values: np.ndarray,
            fixed_mach_number: float | None = None,
    ) -> StabilityGrid:
        """Solve the local equilibrium at every (x, y) combination of two physical-knob
        sweep axes and evaluate both stability criteria there, via a single vectorized
        `solve_equilibrium_batch` call instead of one equilibrium (and one Plasma
        object) per grid point. Measured speedup on a 250x250 grid: ~5.5s -> ~0.02s --
        the physics is identical either way (`solve_equilibrium_batch` runs the exact
        same `_iterate_equilibrium` fixed-point loop `HallSolver.solve_equilibrium`
        does), the win is purely from replacing 62,500 Python-level calls with one call
        operating on 62,500-element arrays.

        `fixed_mach_number`: when given AND "Tp" is one of `x_key`/`y_key`, v0 is
        overridden to hold this Mach number fixed as Tp sweeps, instead of holding v0
        itself fixed at `base.v0` -- see `volume_grid`'s docstring for why (the same
        reasoning applies to any Tp sweep, not just the 3-D one). Has no effect if
        "Tp" isn't actually being swept here (base.Tp is a single fixed value, so
        there's no "wide range" for the Mach number to swing across).

        Returns 2-D arrays of shape (len(y_values), len(x_values))."""
        X, Y = np.meshgrid(x_values, y_values)
        overrides: dict[str, Scalar] = {x_key: X, y_key: Y}
        if fixed_mach_number is not None and "Tp" in (x_key, y_key):
            gas_temperature = X if x_key == "Tp" else Y
            ideal_gas = IdealGas(self._hall_solver.gas_type)
            overrides["v0"] = ideal_gas.get_flow_speed(fixed_mach_number, gas_temperature)
        point, result = self._solve(**overrides)

        ionization_fraction = result.electron_number_density / result.seed_number_density
        beta = result.hall_parameter
        gas_temperature = point.gas_temperature

        exact = FriedbergCriterion()
        asymptotic = FriedbergAsymptoticCriterion()
        margin = exact.stability_margin(beta, result.electron_temperature, gas_temperature, self._ionization_potential, ionization_fraction)
        margin_asymptotic = asymptotic.stability_margin(beta, result.electron_temperature, gas_temperature, self._ionization_potential, ionization_fraction)

        # StabilityGrid's fields are plain arrays (X, Y above are always np.meshgrid
        # output, so every quantity derived from them genuinely is one) -- the
        # np.asarray() calls below are for mypy's benefit only, not a runtime
        # conversion: everything here is already an ndarray, just typed Scalar
        # (float | ndarray) by the general HallSolver/FriedbergCriterion signatures
        # that produced it.
        return StabilityGrid(
            beta=np.asarray(beta),
            beta_crit=np.asarray(exact.critical_hall_parameter(
                result.electron_temperature, gas_temperature, self._ionization_potential, ionization_fraction,
            )),
            beta_crit_asymptotic=np.asarray(asymptotic.critical_hall_parameter(
                result.electron_temperature, gas_temperature, self._ionization_potential, ionization_fraction,
            )),
            margin=np.asarray(margin),
            margin_asymptotic=np.asarray(margin_asymptotic),
            Te=np.asarray(result.electron_temperature),
            ionization_fraction=np.asarray(ionization_fraction),
            stable=np.asarray(margin >= 1.0),
        )

    def matched_load(
            self, seed_fraction, magnetic_field, gas_temperature, flow_speed: Scalar | None = None,
            bracket: tuple[float, float] = (1e-8, 1e4), iters: int = 50,
    ) -> Equilibrium:
        """Vectorized Friedberg (6.10) matched-load (Z=sqrt(1+beta^2)) equilibrium
        solve: at fixed (seed_fraction, magnetic_field, gas_temperature, flow_speed --
        `flow_speed` defaults to `base.v0` if not given, but callers sweeping a wide
        `gas_temperature` range may want a Tp-dependent flow_speed instead, e.g. one
        holding Mach number fixed -- see `volume_grid`'s `fixed_mach_number`), finds the
        load_resistivity for which sqrt(1+beta^2)*eta(load_resistivity) ==
        load_resistivity, i.e. the load resistivity self-consistent with its own
        matched-load condition. Solved by vectorized log-space bisection on
        h(load_resistivity) = sqrt(1+beta^2)*eta(load_resistivity) - load_resistivity
        over `bracket` -- any broadcastable shape (e.g. a 3-D meshgrid) is solved in
        one pass of `iters` rounds of `solve_equilibrium_batch`, the same way `grid`
        and `volume_grid` vectorize their own solves. This is what makes a dense 3-D
        (seed_fraction, B0, Tp) grid affordable at all -- see `volume_grid`'s
        docstring for the measured cost difference.

        This used to be a plain, undamped Picard iteration (`load_resistivity =
        g(load_resistivity)` repeated `iters` times) instead of a proper root-find.
        That failed badly: checked numerically over a grid spanning `volume_grid`'s
        typical ranges, roughly HALF of points never converged to a fixed point at
        all, even after 200 iterations or with load_resistivity relaxed as low as
        0.02 -- they settled into a stable period-2 limit cycle instead. The cause,
        found by tabulating h(load_resistivity) directly at a few of those points: h
        has exactly ONE root (no bistability), but right at that root the map g is
        an almost-vertical cliff -- this system's ionisation-avalanche transition
        (see e.g. sweep_stability_grid's docstring) -- where g changes by ~2 orders
        of magnitude while load_resistivity moves by a few tens of percent. Any fixed
        relaxation factor overshoots a cliff that steep every step; bisection doesn't
        care how steep the root is, only that h changes sign across it, which was
        verified to hold (h(bracket[0]) > 0, h(bracket[1]) < 0) at every point tested
        across the actual grids this method is used for -- if some future caller's
        (seed_fraction, B0, Tp) genuinely falls outside where that holds, this will
        silently converge to one of the bracket endpoints rather than raise; widen
        `bracket` if that's ever suspected.

        50 iterations of bisection on a 12-decade bracket resolves the root to
        relative precision roughly 1e-3*2^-50 ~ 1e-18, far past floating-point noise
        long before reaching this default -- checked directly: 49 vs. 50 iterations
        agree to within 1e-4 relative at every point of a 45x45x45 grid spanning
        `volume_grid`'s typical ranges (worst case ~6e-6), where the old Picard
        version disagreed at ~50% of points even between iteration 199 and 200."""
        if flow_speed is None:
            flow_speed = self._base.v0
        gas_number_density = self._base.p0 / (constants.k * gas_temperature)
        seed_number_density = seed_fraction * gas_number_density
        log_lo, log_hi = np.broadcast_arrays(
            np.full_like(np.asarray(seed_fraction, dtype=float), np.log10(bracket[0])),
            np.full_like(np.asarray(seed_fraction, dtype=float), np.log10(bracket[1])),
        )
        log_lo, log_hi = log_lo.copy(), log_hi.copy()  # broadcast_arrays gives read-only views
        result: Equilibrium | None = None
        for _ in range(iters):
            log_mid = 0.5 * (log_lo + log_hi)
            load_resistivity = 10.0 ** log_mid
            result = self._hall_solver.solve_equilibrium_batch(
                flow_speed=flow_speed, gas_temperature=gas_temperature, gas_number_density=gas_number_density,
                seed_number_density=seed_number_density, magnetic_field=magnetic_field, load_resistivity=load_resistivity,
            )
            h = np.sqrt(1.0 + result.hall_parameter ** 2) * result.resistivity - load_resistivity
            positive = h > 0.0  # h(lo) > 0, h(hi) < 0 by construction of `bracket` -- root is in [mid, hi] iff h(mid) > 0
            log_lo = np.where(positive, log_mid, log_lo)
            log_hi = np.where(positive, log_hi, log_mid)
        assert result is not None, "iters must be >= 1"
        return result

    def volume_grid(
            self, seed_fraction_values: np.ndarray, b0_values: np.ndarray, tp_values: np.ndarray,
            fixed_mach_number: float | None = None,
    ) -> VolumeGrid:
        """Precomputed 3-D equilibrium/stability table over (seed_fraction, B0, Tp),
        solved with the Friedberg (6.10) matched-load Z=sqrt(1+beta^2) policy
        throughout -- load resistivity is never a free axis here (see `matched_load`;
        Z=beta^2+1, maximizing power outright, was tried first and abandoned: it
        pushed the whole system toward physically unreasonable sub-100K Tp at high
        field).

        `fixed_mach_number`: by default (None) v0 is held fixed at `base.v0`
        throughout, the same as every other input here. But this sweep's own Tp range
        typically spans several orders of magnitude (e.g. 1-6000 K) -- holding a
        single v0 fixed across that means the gas-dynamic Mach number
        M = v0/sqrt(gamma*k*Tp/m) swings just as wildly (checked: ~8 at the low end of
        a typical range down to ~0.1 at the high end, for a v0 around 150 m/s), which
        feeds directly into HallSolver's electron-heating term (ΔT ∝ M^2) and so
        distorts where the computed stability boundary actually sits at low Tp -- not
        a cosmetic issue. Passing a Mach number here instead (e.g. the one implied by
        the sidebar's own current v0/Tp) holds THAT fixed via IdealGas.get_flow_speed,
        letting v0 vary with the swept Tp so the flow stays at a consistent regime
        across the whole sweep -- closed-form, no extra root-find, unlike trying to
        pin down v0 from something that itself depends on the solved equilibrium
        (e.g. Messerle's interaction-length estimate, Plasma.ideal_channel_length --
        considered and set aside for exactly that reason: sigma there depends on the
        equilibrium solve, which depends on v0, which the same equation is trying to
        solve for).

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
        flow_speed = None
        if fixed_mach_number is not None:
            flow_speed = IdealGas(self._hall_solver.gas_type).get_flow_speed(fixed_mach_number, TP)
        result = self.matched_load(SF, B0, TP, flow_speed=flow_speed)

        ionization_fraction = result.electron_number_density / result.seed_number_density
        criterion = FriedbergCriterion()
        margin = criterion.stability_margin(result.hall_parameter, result.electron_temperature, TP, self._ionization_potential, ionization_fraction)

        # See grid()'s comment on these np.asarray() calls -- SF/B0/TP are always
        # np.meshgrid output, so everything derived from them already is an array.
        return VolumeGrid(
            margin=np.asarray(margin),
            stable=np.asarray(margin >= 1.0),
            Te=np.asarray(result.electron_temperature),
            ionization_fraction=np.asarray(ionization_fraction),
            load_power_density=np.asarray(result.load_power_density),
        )

    def margin_minus_level(self, level: float = 1.0, **overrides) -> Scalar:
        """beta_crit/beta - level, evaluated via solve_equilibrium_batch for whatever
        combination of scalar/array physical-knob `overrides` is given (see
        `OperatingPoint.resolve`). Used for big vectorized scans -- e.g. the coarse
        scan `critical_load_resistivity_surface` refines with its own vectorized
        bisection.

        `level` generalizes past the marginal-stability level (1.0, the default,
        margin == 1): callers wanting some safety buffer -- "stable with margin >=
        1.5" -- or the reverse -- "still counts as stable-ish with margin >= 0.5" --
        find that boundary by searching for margin_minus_level's zero crossing at
        `level` instead of 1.0, everything else about the search unchanged."""
        point, result = self._solve(**overrides)
        ionization_fraction = result.electron_number_density / result.seed_number_density
        margin = FriedbergCriterion().stability_margin(
            result.hall_parameter, result.electron_temperature, point.gas_temperature,
            self._ionization_potential, ionization_fraction,
        )
        return margin - level

    def critical_load_resistivity_surface(
            self, seed_fraction_values: np.ndarray, b0_values: np.ndarray, level: float = 1.0,
            load_resistivity_bracket: tuple[float, float] = (1e-4, 20.0), scan_points: int = 60, refine_iters: int = 40,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """For each (seed_fraction, B0), finds the critical load resistivity -- the
        boundary surface (at margin == `level`; the default, 1.0, is marginal
        stability itself) over the three inputs that matter most directly: seed
        fraction and B0 together set beta and (via Saha/energy balance) f_I, and load
        resistivity is the next most consequential *free dial* (as opposed to Tp or
        the flow conditions, which are usually fixed by the working gas/environment
        rather than tuned) -- it sets Z = eta_L/eta, which Delta_T depends on through
        the whole current/heating loop (eq. 3.14).

        Returns (lower, upper, lower_power, upper_power): the boundary height(s) in
        Ohm*m -- `upper` is all-NaN unless the margin actually re-crosses `level` at
        higher resistivity (checked numerically at level=1: not observed in the range
        tested here, but handled the same way as a B0 stability window for
        robustness) -- plus the load power density S_L *at* each boundary point, so
        the surface's color can carry a genuinely independent piece of information
        instead of re-deriving the height.

        The scan (every (B0, seed_fraction, load_resistivity) combination) is one
        vectorized `margin_minus_level` call. This used to refine each detected
        crossing with a scalar `scipy.optimize.brentq` call in a Python-level
        (B0, seed_fraction) double loop -- profiled as this method's dominant cost
        (thousands of scalar HallSolver-adjacent calls, one full fixed-point solve
        each, for a modest grid). Refinement is now a second, fully vectorized
        log-space bisection (`refine_iters` rounds, same idiom as `matched_load`/
        `SeedDensityBounds._bisect_ceiling`) over every cell's own scan-bracketed
        interval at once -- collapsing that Python loop into a handful of batch calls.

        Still finds *every* sign change per cell, not just the endpoints -- the same
        reason the old per-cell version scanned first rather than just bisecting the
        full bracket directly (this system can have a genuine stability window,
        unstable-stable-unstable, not just a single threshold): `lower`/`upper` are
        the first/last sign change the scan found, matching what the old
        `roots[0]`/`roots[-1]` selection did."""
        scan_values = np.logspace(np.log10(load_resistivity_bracket[0]), np.log10(load_resistivity_bracket[1]), scan_points)
        B0, SF, LR = np.meshgrid(b0_values, seed_fraction_values, scan_values, indexing="ij")
        values: np.ndarray = np.asarray(self.margin_minus_level(level, B0=B0, seed_fraction=SF, load_resistivity=LR))  # shape (len(b0), len(sf), scan_points)

        # changes[..., k] is True where the scan crosses zero between scan_values[k] and
        # scan_values[k+1] -- first_index/last_index locate the first/last such interval
        # per (B0, seed_fraction) cell (argmax finds the first True; reversing first finds
        # the last), exactly the two crossings the old per-cell roots[0]/roots[-1] kept.
        changes = np.diff(np.sign(values), axis=-1) != 0
        any_change = changes.any(axis=-1)
        first_index = np.argmax(changes, axis=-1)
        last_index = changes.shape[-1] - 1 - np.argmax(changes[..., ::-1], axis=-1)
        has_window = any_change & (last_index != first_index)

        B0_2d, SF_2d = np.meshgrid(b0_values, seed_fraction_values, indexing="ij")

        def refine(bracket_index: np.ndarray, valid: np.ndarray) -> np.ndarray:
            """Vectorized log-space bisection of [scan_values[bracket_index],
            scan_values[bracket_index + 1]] at every (B0, seed_fraction) cell at once.
            Doesn't assume which bracket endpoint is positive -- each step compares the
            midpoint's sign against the lower endpoint's own sign instead of a hardcoded
            one (same reasoning as `SeedDensityBounds._bisect_ceiling`)."""
            bracket_lo = scan_values[bracket_index]
            bracket_hi = scan_values[np.minimum(bracket_index + 1, scan_points - 1)]
            positive_at_lo = np.asarray(self.margin_minus_level(level, B0=B0_2d, seed_fraction=SF_2d, load_resistivity=bracket_lo)) > 0.0

            log_lo, log_hi = np.log10(bracket_lo), np.log10(bracket_hi)
            for _ in range(refine_iters):
                log_mid = 0.5 * (log_lo + log_hi)
                lr_mid = 10.0 ** log_mid
                positive = np.asarray(self.margin_minus_level(level, B0=B0_2d, seed_fraction=SF_2d, load_resistivity=lr_mid)) > 0.0
                go_high = positive == positive_at_lo  # sign(mid) matches sign(lo) => root is in [mid, hi]
                log_lo = np.where(go_high, log_mid, log_lo)
                log_hi = np.where(go_high, log_hi, log_mid)
            return np.where(valid, 10.0 ** (0.5 * (log_lo + log_hi)), np.nan)

        lower = refine(first_index, any_change)
        upper = refine(last_index, has_window)

        # NaN load_resistivity propagates to NaN load_power_density cleanly (no exception,
        # just the elementwise fixed-point math computing NaN throughout) -- errstate just
        # silences the resulting (harmless, expected) invalid-value warnings.
        with np.errstate(invalid="ignore"):
            lower_power = np.asarray(self._solve(B0=B0_2d, seed_fraction=SF_2d, load_resistivity=lower)[1].load_power_density)
            upper_power = np.asarray(self._solve(B0=B0_2d, seed_fraction=SF_2d, load_resistivity=upper)[1].load_power_density)

        return lower, upper, lower_power, upper_power
