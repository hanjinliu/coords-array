from __future__ import annotations
from abc import ABC, abstractmethod
from typing import (
    Hashable,
    Mapping,
    Sequence,
    SupportsIndex,
    TypeVar,
    Union,
    Iterable,
    TYPE_CHECKING,
    TypedDict,
)
import numpy as np

if TYPE_CHECKING:
    from typing_extensions import Self

_T = TypeVar("_T", bound=Hashable)
_Slicable = Union[SupportsIndex, slice, list[int], np.ndarray]
_Real = Union[int, float]


class Index(Sequence[_T], ABC):
    @abstractmethod
    def to_indexer(self, coords: _T | slice) -> _Slicable:
        """Convert input into a slicable object."""

    @abstractmethod
    def get_size(self) -> int:
        """Length of the index."""

    @abstractmethod
    def get_scale(self) -> float:
        """Return the scale of the index."""

    @abstractmethod
    def rescaled(self, scale: float) -> Index[_T]:
        """Return a rescaled index."""

    def get_unit(self) -> str:
        return self._unit

    def set_unit(self, unit: str) -> None:
        self._unit = unit

    def _repr_short(self) -> str:
        return self.__repr__()

    def __len__(self) -> int:
        return self.get_size()

    @abstractmethod
    def shifted(self, shift: _T) -> Index[_T]:
        """Return a shifted index."""

    @abstractmethod
    def inverted(self) -> Index[_T]:
        """Return a inverted index."""

    @abstractmethod
    def subset(self, subset: Iterable[int]) -> Index[_T]:
        """Return a subset of the index."""

    @abstractmethod
    def copy(self: Self) -> Self:
        """Return a copy of the index."""


class ScaledIndex(Index[_Real]):
    """Decimal range."""

    def __init__(self, start: float, step: float, size: int, unit=None):
        if size < 0:
            raise ValueError("Size must be non-negative.")
        self._start = start
        self._step = step
        self._size = size
        self._unit = unit

    @classmethod
    def arange(cls: type[ScaledIndex], size: int, step: float = 1.0) -> Self:
        return cls(0.0, step, size)

    @property
    def _stop(self) -> float:
        return self._start + self._step * self._size

    def rescaled(self, scale: float) -> ScaledIndex:
        return ScaledIndex(
            start=self._start,
            step=scale,
            size=self._size,
            unit=self._unit,
        )

    def shifted(self, shift: float) -> ScaledIndex:
        return ScaledIndex(
            start=self._start + shift,
            step=self._step,
            size=self._size,
            unit=self._unit,
        )

    def inverted(self) -> ScaledIndex:
        return ScaledIndex(
            start=self._start,
            step=-self._step,
            size=self._size,
            unit=self._unit,
        )

    def copy(self: Self) -> Self:
        return type(self)(self._start, self._step, self._size, self._unit)

    def __repr__(self) -> str:
        _cls = type(self).__name__
        return f"{_cls}<start={self._start}, stop={self._stop}, step={self._step}>"

    def __getitem__(self, key):
        if not isinstance(key, slice):
            if self._step > 0:
                val = self._start + key * self._step
                if val >= self._stop:
                    raise IndexError("Index out of range.")
            else:
                val = self._stop + key * self._step
                if val <= self._start:
                    raise IndexError("Index out of range.")
        else:
            start, stop, step = key.indices(self._size)
            if step > 0:
                m_start = self.__getitem__(start)
            else:
                m_start = self.__getitem__(stop - 1)
            m_step = step * self._step
            size = max((stop - start) // step, 0)
            val = ScaledIndex(m_start, m_step, size, self._unit)
        return val

    def get_scale(self) -> float:
        return self._step

    def get_size(self) -> int:
        return self._size

    def to_indexer(self, coords: _Real | slice) -> _Slicable:
        if isinstance(coords, slice):
            if coords.step not in (None, 1, -1):
                raise ValueError("Step size must be 1 or -1.")
            start = coords.start
            stop = coords.stop
            if start is not None:
                start = int(np.ceil((start - self._start) / self._step))
            if stop is not None:
                stop = int((stop - self._start) / self._step)
            return slice(start, stop, coords.step)
        else:
            return int(np.round((coords - self._start) / self._step))

    def subset(self, subset: Iterable[int]) -> CategoricalIndex[_T]:
        return CategoricalIndex([self._start + key * self._step for key in subset])


# class DatetimeRangeIndex(Index[pd.Timestamp]):
#     def __init__(self, index: pd.DatetimeIndex):
#         self._pd_index = index

#     @classmethod
#     def from_range(
#         cls: type[DatetimeRangeIndex],
#         start,
#         stop,
#         periods=None,
#         freq=None,
#     ) -> Self:
#         trange = pd.date_range(start, stop, periods=periods, freq=freq)
#         return cls(trange)

#     def size(self) -> int:
#         return len(self._pd_index)

#     def to_indexer(self, key: pd.Timestamp | slice) -> _Slicable:
#         if isinstance(key, slice):
#             if (start := key.start) is not None:
#                 start = self._pd_index.get_loc(key.start)
#             if (stop := key.stop) is not None:
#                 stop = self._pd_index.get_loc(key.stop)
#             return slice(start, stop, key.step)
#         return self._pd_index.get_loc(key)

_NO_SCALE = object()


class CategoricalIndex(Index[_T]):
    """An index with categorical labels."""

    def __init__(self, seq: Iterable[_T], scale: float | object = _NO_SCALE, unit=None):
        self._labels = tuple(seq)
        self._hash_map: dict[_T, int] = {}
        for i, label in enumerate(self._labels):
            self._hash_map.setdefault(label, i)
        self._scale = scale
        self._unit = unit

    def __repr__(self) -> str:
        _cls = type(self).__name__
        return f"{_cls}<{self._labels!r}>"

    def _repr_short(self) -> str:
        _cls = type(self).__name__
        if len(self._labels) < 4:
            return f"{_cls}<{self._labels!r}>"
        else:
            _l = self._labels
            return f"{_cls}<{_l[0]!r}, {_l[1]!r}, ..., {_l[-1]!r}>"

    def get_size(self) -> int:
        """Length of labels"""
        return len(self._labels)

    def get_scale(self) -> float:
        return self._scale

    def __getitem__(self, key):
        out = self._labels[key]
        if isinstance(key, slice):
            return CategoricalIndex(out)
        return out

    def copy(self) -> CategoricalIndex:
        return self.__class__(self._labels, scale=self._scale, unit=self._unit)

    def __eq__(self, other: Sequence[_T]) -> bool:
        return self._labels == other

    @property
    def has_duplicate(self) -> bool:
        """True if self has duplicated labels."""
        return len(self._labels) != len(self._hash_map)

    def to_indexer(self, coords: _T | slice) -> int | slice:
        if self.has_duplicate:
            raise ValueError("Labels have duplicate.")

        if isinstance(coords, slice):
            if coords.start is None:
                start = None
            else:
                start = self._hash_map[coords.start]
            if coords.stop is None:
                stop = None
            else:
                stop = self._hash_map[coords.stop]
            idx = slice(start, stop, coords.step)
        else:
            idx = self._hash_map[coords]
        return idx

    def rescaled(self, scale: float) -> CategoricalIndex:
        return type(self)(self._labels, scale=scale, unit=self._unit)

    def shifted(self, shift: float) -> CategoricalIndex:
        return self.copy()

    def inverted(self) -> CategoricalIndex:
        return CategoricalIndex(
            self._labels[::-1],
            scale=self._scale,
            unit=self._unit,
        )

    def subset(self, subset: Iterable[int]) -> CategoricalIndex[_T]:
        """Return a subset of the coordinates."""
        return CategoricalIndex(
            (self._labels[i] for i in subset),
            unit=self._unit,
        )


class IndexOptions(TypedDict):
    scale: float
    unit: str
    labels: Iterable[Hashable]


IndexLike = Union[
    IndexOptions,
    range,
    Iterable[Hashable],
    None,
]


def as_index(obj: IndexLike, size: int) -> Index:
    """Convert input object to an Index with given size."""

    if isinstance(obj, Index):
        index = obj
    elif obj is None:
        index = ScaledIndex.arange(size)
    elif isinstance(obj, range):
        index = ScaledIndex(obj.start, obj.stop, obj.step)
    elif isinstance(obj, Mapping):
        index = ScaledIndex.arange(size).rescaled(obj.get("scale", 1.0))
        index.set_unit(obj.get("unit", None))
    elif hasattr(obj, "__iter__"):
        index = CategoricalIndex(obj)
    else:
        raise TypeError(f"Cannot convert {type(obj)} to Coordinates.")

    if len(index) != size:
        raise ValueError(f"Length of coordinates must be {size}.")
    return index
