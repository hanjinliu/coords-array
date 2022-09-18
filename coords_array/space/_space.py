from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Sequence, SupportsIndex
import numpy as np

from ..coords import IndexLike, Coordinates
from ..array import CoordsArray
from .._shared_types import ShapeLike, AxisOptions, SpaceOptions

if TYPE_CHECKING:
    from typing_extensions import Self


class AxisType(Enum):
    linear = "linear"
    category = "category"
    periodic = "periodic"


# Space(3)  # 3D linear space
# Space(["y", "x"])
# Space(y=["B", "G", "R"], x={})


class Space:
    def __init__(self, input: SpaceOptions | None = None, /, **options: AxisOptions):
        # normalize input
        if isinstance(input, int):
            _options = {f"axis_{i}": {} for i in range(input)}
        elif isinstance(input, dict):
            _options = input
        elif hasattr(input, "__iter__"):
            _options = {k: {} for k in input}
        elif input is None:
            _options = options
        else:
            raise TypeError(f"Invalid input: {input!r}")

        self._ndim = len(_options)

        if self._ndim == 0:
            raise ValueError("Cannot create a space with no dimensions.")

        self._options = _options

    @property
    def ndim(self) -> int:
        return self._ndim

    def build_coords(self, shape: ShapeLike) -> Coordinates:
        """Build coordinates with the given shape."""
        self._check_ndim(shape)
        return Coordinates.from_dict(self._options, shape)

    def partial(self, include: Sequence[SupportsIndex | str]) -> Space:
        """Return a space with a subset of the coordinates."""
        _keys = list(self._options.keys())
        _include_keys = [i if isinstance(i, str) else _keys[i] for i in include]
        options = {k: self._options[k] for k in _include_keys}
        return type(self)(options)

    def zeros(self, shape: ShapeLike, dtype=None) -> CoordsArray:
        return CoordsArray(
            np.zeros(shape, dtype=dtype), coords=self.build_coords(shape)
        )

    def ones(self, shape: ShapeLike, dtype=None) -> CoordsArray:
        return CoordsArray(np.ones(shape, dtype=dtype), coords=self.build_coords(shape))

    def empty(self, shape: ShapeLike, dtype=None) -> CoordsArray:
        return CoordsArray(
            np.empty(shape, dtype=dtype), coords=self.build_coords(shape)
        )

    def __repr__(self):
        _cls = type(self).__name__
        return f"<{self._ndim}D {_cls}>"

    @property
    def random(self):
        """The random generator for this space."""
        from ._random import RandomGenerator

        return RandomGenerator(self)

    def _check_ndim(self, shape: ShapeLike) -> None:
        if isinstance(shape, int):
            err = self._ndim != 1
        else:
            err = self._ndim != len(shape)
        if err:
            raise ValueError(f"Invalid shape for {self._ndim}-D space.")
