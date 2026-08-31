"""
Shared type alias for this library's numpy-vectorized physics functions: every one
accepts a plain float or a numpy array of floats and returns the same kind of thing,
elementwise -- which is what every arithmetic operation on this union type-checks
against under mypy. `typing.Iterable` (this module's previous choice, throughout the
library) does not support arithmetic operators at all, so it type-checked nothing --
`Scalar` is what these functions were actually always being called with.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

Scalar = float | npt.NDArray[np.float64]

# The boolean counterpart -- e.g. StabilityModel.is_stable's elementwise comparison
# result, for a plain float comparison or an elementwise array comparison alike.
BoolScalar = bool | npt.NDArray[np.bool_]
