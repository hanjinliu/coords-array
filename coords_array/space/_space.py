from __future__ import annotations

from enum import Enum
from typing import (
    TYPE_CHECKING,
    Hashable,
    Iterator,
    Sequence,
    SupportsIndex,
    TypeVar,
    Mapping,
    ValuesView,
)
import numpy as np

from ..coords import Coordinates
from ..array import CoordsArray
from .._shared_types import ShapeLike, PartialShapeLike, AxisOptions, SpaceOptions

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
            _options = IndexedMap((f"axis_{i}", {}) for i in range(input))
        elif isinstance(input, dict):
            _options = IndexedMap(input)
        elif hasattr(input, "__iter__"):
            _options = IndexedMap((k, {}) for k in input)
        elif input is None:
            _options = IndexedMap(options)
        else:
            raise TypeError(f"Invalid input: {input!r}")

        self._ndim = len(_options)

        if self._ndim == 0:
            raise ValueError("Cannot create a space with no dimensions.")

        self._options = _options

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self._ndim

    @property
    def axes(self) -> tuple[str, ...]:
        return tuple(self._options.keys())

    def __repr__(self):
        _cls = type(self).__name__
        return f"<{self._ndim}D {_cls} with axes {self.axes!r}>"

    def build_coords(self, shape: PartialShapeLike) -> Coordinates:
        """Build coordinates with the given shape."""
        shape, options = _normaize_shape_and_options(shape, self._options)
        return Coordinates.from_dict(options, shape)

    def partial(self, include: Sequence[SupportsIndex | str]) -> Space:
        """Return a space with a subset of the coordinates."""
        _include_keys = [
            i if isinstance(i, str) else self._options.value_at(i) for i in include
        ]
        options = IndexedMap((k, self._options[k]) for k in _include_keys)
        return type(self)(options)

    def zeros(self, shape: PartialShapeLike, dtype=None) -> CoordsArray:
        shape, options = _normaize_shape_and_options(shape, self._options)
        return CoordsArray(
            np.zeros(shape, dtype=dtype),
            coords=Coordinates.from_dict(options, shape),
        )

    def ones(self, shape: PartialShapeLike, dtype=None) -> CoordsArray:
        shape, options = _normaize_shape_and_options(shape, self._options)
        return CoordsArray(
            np.ones(shape, dtype=dtype),
            coords=Coordinates.from_dict(options, shape),
        )

    def empty(self, shape: PartialShapeLike, dtype=None) -> CoordsArray:
        shape, options = _normaize_shape_and_options(shape, self._options)
        return CoordsArray(
            np.empty(shape, dtype=dtype),
            coords=Coordinates.from_dict(options, shape),
        )

    def full(self, shape: PartialShapeLike, fill_value, dtype=None) -> CoordsArray:
        shape, options = _normaize_shape_and_options(shape, self._options)
        return CoordsArray(
            np.full(shape, fill_value, dtype=dtype),
            coords=Coordinates.from_dict(options, shape),
        )

    @property
    def random(self):
        """The random generator for this space."""
        from ._random import RandomGenerator

        return RandomGenerator(self)


_K = TypeVar("_K", bound=Hashable)
_V = TypeVar("_V")


class IndexedMap(Mapping[_K, _V]):
    def __init__(self, *args, **kwargs):
        self._map = dict(*args, **kwargs)
        self._keys = list(self.keys())

    def __getitem__(self, key: _K) -> _V:
        return self._map[key]

    def __iter__(self) -> Iterator[_K]:
        return iter(self._map)

    def values(self) -> ValuesView[_V]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._map)

    def key_at(self, index: int) -> _K:
        return self._keys[index]

    def value_at(self, index: SupportsIndex) -> _V:
        """Get value at index."""
        return self._keys[index]

    def item_at(self, index: SupportsIndex) -> tuple[_K, _V]:
        """Get item at index."""
        return self._keys[index], self._map[self._keys[index]]

    def subset(self, indices: Sequence[SupportsIndex]) -> IndexedMap[_K, _V]:
        """Return a subset of the map."""
        return IndexedMap(self.item_at(i) for i in indices)


def _normaize_shape_and_options(
    shape: PartialShapeLike,
    options: IndexedMap,
) -> tuple[ShapeLike, IndexedMap]:
    if isinstance(shape, int):
        if len(options) != 1:
            raise ValueError(f"Invalid shape ({shape},) for {len(options)}D space.")
        return shape, options

    if len(shape) != len(options):
        raise ValueError(f"Invalid shape {shape!r} for {len(options)}D space.")
    accept = [size is not None for size in shape]
    shape = tuple(size for size in shape if size is not None)
    return shape, options.subset(accept)
