from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from .array import CoordsArray
from .coords import Coordinates
from .typing import CoordinateLike


def zeros(shape, dtype=None, *, coords: CoordinateLike | None = None):
    return CoordsArray(np.zeros(shape, dtype=dtype), coords=coords)


def empty(shape, dtype=None, *, coords: CoordinateLike | None = None):
    return CoordsArray(np.empty(shape, dtype=dtype), coords=coords)
