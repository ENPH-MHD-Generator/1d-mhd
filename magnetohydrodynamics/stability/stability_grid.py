from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class StabilityGrid:
    """EquilibriumSweep.grid()'s per-point stability picture over a 2-D sweep of two
    physical-knob axes: both stability criteria (exact and the high-ionisation
    asymptotic), their margins, and the underlying equilibrium quantities plots need
    alongside them. Replaces the ad hoc dict `grid()` used to return. Every field has
    shape (len(y_values), len(x_values)) -- see `grid()`'s own docstring.

    Fields are typed as plain arrays (not magnetohydrodynamics.typing.Scalar) because
    grid() always solves over an np.meshgrid -- unlike EquilibriumSweep/HallSolver's own
    solve methods, there's no genuinely scalar-input case here to make a Union honest."""
    beta: npt.NDArray[np.float64]
    beta_crit: npt.NDArray[np.float64]
    beta_crit_asymptotic: npt.NDArray[np.float64]
    margin: npt.NDArray[np.float64]
    margin_asymptotic: npt.NDArray[np.float64]
    Te: npt.NDArray[np.float64]
    ionization_fraction: npt.NDArray[np.float64]
    stable: npt.NDArray[np.bool_]


@dataclass
class VolumeGrid:
    """EquilibriumSweep.volume_grid()'s precomputed 3-D (seed_fraction, B0, Tp) table,
    solved at the Friedberg (6.10) matched-load condition throughout. Replaces the ad
    hoc dict `volume_grid()` used to return; also what StabilityBoundaryMesh extracts
    an isosurface from (see its own docstring). Every field has shape
    (len(seed_fraction_values), len(b0_values), len(tp_values)) -- see StabilityGrid's
    docstring for why these are plain arrays, not the Scalar union."""
    margin: npt.NDArray[np.float64]
    stable: npt.NDArray[np.bool_]
    Te: npt.NDArray[np.float64]
    ionization_fraction: npt.NDArray[np.float64]
    load_power_density: npt.NDArray[np.float64]
