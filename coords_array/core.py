from __future__ import annotations
from typing import Any, Hashable, Sequence, SupportsIndex, TypeVar, Union, Iterable, TYPE_CHECKING
import numpy as np
from numpy.typing import ArrayLike
from .array import CoordsArray
from .coords import Coordinates
from .typing import CoordinateLike

def zeros(shape, dtype=float, *, coords=None):
    return CoordsArray(np.zeros(shape, dtype=dtype), coords=coords)