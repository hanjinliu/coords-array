from __future__ import annotations
from typing import (
    Any, Mapping, Sequence, Iterable, overload, MutableMapping, TypeVar, TYPE_CHECKING
)
import weakref
import numpy as np

from ._axis import Axis, AxisLike, as_axis, UndefAxis, MetricLike
from ._metric import as_metric
from ._slicer import Slicer
from ._misc import CoordinateError

if TYPE_CHECKING:
    from ._axes_tuple import AxesTuple
    from ..typing import CoordinateLike, CoordinateOptions

_T = TypeVar("_T")
AxesLike = Iterable[AxisLike]


class Coordinates(Sequence[Axis]):
    """
    A sequence of axes.
    
    This object behaves like a string as much as possible.
    """
    def __init__(self, value: Sequence[Axis] | Coordinates) -> None:
        if not isinstance(value, self.__class__):
            ndim = len(value)
            
            # check duplication
            if ndim > len(set(value)):
                raise CoordinateError(f"Duplicated axes found: {value}.")
            
            self._axis_list = list(value)
            
        else:
            self._axis_list = [a.__copy__() for a in value._axis_list]
    
    @classmethod
    def undef(cls, ndim: int):
        """Construct an Axes object initialized with undefined axes."""
        return cls([UndefAxis() for _ in range(ndim)])
    
    @classmethod
    def from_iterable(cls, axes: Iterable[AxisLike], shape: tuple[int, ...]) -> Coordinates:
        """Construct an Axes object from an iterable of AxisLike objects."""
        if len(axes) != len(shape):
            raise CoordinateError(
                f"Length of input ({len(axes)}) and shape ({len(shape)}) do not match."
            )
        return cls([as_axis(a, size) for a, size in zip(axes, shape)])
    
    @classmethod
    def from_dict(
        cls: type[Coordinates],
        input: Mapping[AxisLike, CoordinateOptions], 
        shape: tuple[int, ...],
    ) -> Coordinates:
        """Construct an Axes object from a dictionary."""
        if len(input) != len(shape):
            raise CoordinateError(
                f"Length of input ({len(input)}) and shape ({len(shape)}) do not match."
            )
        axes: list[Axis] = []
        for (a, options), size in zip(input.items(), shape):
            if not isinstance(options, Mapping):
                raise TypeError(f"Options for {a} must be a dictionary.")
            axes.append(Axis(a, metric=as_metric(options, size)))
        return cls(axes)
    
    def update_scales(
        self,
        other: Mapping[AxisLike, float] | None = None,
        **kwargs: dict[str, float],
    ) -> None:
        if other is not None:
            for k, v in other.items():
                self._axis_list[self.find(k)].scale = v
        for k, v in kwargs.items():
            self._axis_list[self.find(k)].scale = v

    def __str__(self):
        return "".join(map(str, self._axis_list))
    
    def __repr__(self) -> str:
        return self._repr()
    
    def _repr(self, n_indent: int = 0) -> str:
        """Return a string representation of the axes."""
        clsname = type(self).__name__
        indent = " " * n_indent
        args = [repr(axis) for axis in self._axis_list]
        args_str = f",\n{indent}  ".join(args)
        return f"{indent}{clsname}(\n{indent}  {args_str}\n{indent})"
    
    def __len__(self):
        return len(self._axis_list)

    @overload
    def __getitem__(self, key: int | str | Axis) -> Axis:
        ...
        
    @overload
    def __getitem__(self, key: slice) -> Coordinates:
        ...
        
    def __getitem__(self, key):
        """Get an axis."""
        if isinstance(key, (str, Axis)):
            return self._axis_list[self.find(key)]
        elif isinstance(key, slice):
            l = self._axis_list[key]
            return self.__class__(l)
        else:
            return self._axis_list[key]
    
    def __getattr__(self, key: str) -> Axis:
        """Return an axis with name `key`."""
        try:
            idx = self._axis_list.index(key)
        except:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {key!r}.")
        return self._axis_list[idx]
    
    def __iter__(self):
        return iter(self._axis_list)
    
    def __eq__(self, other):
        if isinstance(other, str):
            return str(self) == other
        elif isinstance(other, self.__class__):
            return other._axis_list == self._axis_list
        return self._axis_list == other

    def __contains__(self, other: AxisLike) -> bool:
        return other in self._axis_list
    
    def __repr__(self):
        s = ", ".join(map(lambda x: repr(str(x)), self))
        return f"{self.__class__.__name__}[{s}]"

    def __hash__(self) -> int:
        """Hash as a tuple of strings."""
        return hash(tuple(map(str, self._axis_list)))
    
    @overload
    def find(self, axis: str | Axis) -> int:
        ...
    
    @overload
    def find(self, axis: str | Axis, default: _T) -> _T:
        ...
        
    def find(self, axis: str | Axis, *args) -> int:
        """Find the index of an axis."""
        if len(args) > 1:
            raise TypeError(f"Expected 2 or 3 arguments but got {len(args) + 2}.")
        try:
            return self._axis_list.index(axis)
        except ValueError:
            if args:
                return args[0]
            _axes = tuple(str(a) for a in self._axis_list)
            raise CoordinateError(
                f"Image does not have {axis}-axis: {_axes}."
            ) from None
    
    def has_undef(self) -> bool:
        return any(isinstance(a, UndefAxis) for a in self._axis_list)

    def copy(self):
        """Make a copy of Axes object."""
        return self.__class__(self)

    def replace(self, old: AxisLike, new: AxisLike) -> Coordinates:
        """
        Create a new Axes object with `old` axis replaced by `new`.
        
        To avoid unexpected effect between images, new scale attribute will be copied.

        Parameters
        ----------
        old : str
            Old symbol.
        new : str
            New symbol.
        """        
        i = self.index(old)
        if new in self._axis_list and old != new:
            raise CoordinateError(f"Axes {new} already exists: {self}")
        
        if isinstance(new, str):
            new_axis = Axis(new, metadata=self[i].metadata.copy())
        else:
            new_axis = new
        axis_list = self._axis_list.copy()
        axis_list[i] = new_axis
        return self.__class__(axis_list)
    
    def contains(self, chars: AxesLike, *, ignore_undef: bool = False) -> bool:
        """True if self contains all the characters in ``chars``."""
        if ignore_undef:
            return all(a in self._axis_list for a in chars if not isinstance(a, UndefAxis))
        return all(a in self._axis_list for a in chars)
    
    def drop(self, axes: AxisLike | AxesLike | int | Iterable[int]) -> Coordinates:
        """Drop an axis or a list of axes."""
        if not isinstance(axes, (list, tuple, str)):
            axes = (axes,)
        
        drop_list = []
        for a in axes:
            if isinstance(a, int):
                drop_list.append(self._axis_list[a])
            else:
                drop_list.append(a)
        
        return Coordinates(a for a in self._axis_list if a not in drop_list)
    
    def extend(self, axes: AxesLike) -> Coordinates:
        """Extend axes with given axes."""
        return self + axes

    @overload
    def create_slice(self, sl: Mapping[str, Any] | Slicer) -> tuple[Any, ...]:
        ...
    
    @overload
    def create_slice(self, **kwargs: dict[str, Any]) -> tuple[Any, ...]:
        ...
    
    def create_slice(self, sl = None, /, **kwargs):
        if sl is None:
            sl = kwargs
        elif isinstance(sl, Slicer):
            sl = sl._dict
            
        if not sl:
            raise TypeError("Slice not given.")
        
        sl_list = [slice(None)] * len(self)
    
        for k, v in sl.items():
            idx = self.index(k)
            sl_list[idx] = v
        
        return tuple(sl_list)

    def tuple(self, iterable: Iterable[_T], /) -> AxesTuple[_T]:
        """Convert iterable to AxesTuple."""
        from ._axes_tuple import get_axes_tuple
        try:
            out = get_axes_tuple(self)(*iterable)
        except CoordinateError:
            out = tuple(iterable)
        return out
    
    def update_coords(self, coords: Mapping[str, MetricLike] = {}, /, **kwargs) -> Coordinates:
        """Update coordinates."""
        if kwargs:
            coords = dict(coords, **kwargs)
        if not coords:
            return self
        
        pairs: list[tuple[Axis, MetricLike]] = []
        for k, v in coords.items():
            axis = self._axis_list[self.find(k)]
            _crds = as_metric(v, size=axis.size)
            pairs.append((axis, _crds))
        for axis, crds in pairs:
            axis._set_metric(crds)
        return self
    
    def update_scales(self, scales: Mapping[str, float] = {}, /, **kwargs) -> Coordinates:
        """Update scales."""
        _scales = {}
        for k, v in dict(scales, **kwargs).items():
            v = float(v)
            if v <= 0:
                raise ValueError(f"Scale of {k} must be positive: {v}")
            _scales[k] = v
            
        if not _scales:
            return self
        
        for k, v in _scales.items():
            axis = self._axis_list[self.find(k)]
            axis.scale = v
        return self


def as_coordinates(coords: CoordinateLike, shape: tuple[int, ...]) -> Coordinates:
    if coords is None:
        return Coordinates.undef(len(shape))
    elif isinstance(coords, Coordinates):
        return coords
    elif isinstance(coords, Mapping):
        return Coordinates.from_dict(coords, shape)
    elif isinstance(coords, Iterable):
        return Coordinates.from_iterable(coords, shape)
    else:
        raise TypeError(f"Invalid type for coords: {type(coords)}")


def _broadcast_two(coords0: CoordinateLike, coords1: CoordinateLike, rule=None) -> Coordinates:
    coords0 = as_coordinates(coords0)
    coords1 = as_coordinates(coords1)
    
    arg_idx: list[int] = []
    out = list(coords0)
    for a in coords1:
        if type(a) is UndefAxis:
            raise TypeError("Cannot broadcast coordinates with UndefAxis.")
        arg_idx.append(coords0.find(a, -1))
    
    stack = []
    n_insert = 0
    iter = enumerate(arg_idx.copy())
    for i, idx in iter:
        if idx < 0:
            stack.append(i)
        else:
            for j in stack:
                out.insert(idx + n_insert, coords1[j])
                n_insert += 1
            stack.clear()
    for j in stack:
        out.append(coords1[j])
        
    return Coordinates(out)

def broadcast(*axes_objects: AxesLike) -> Coordinates:
    """
    Broadcast two or more axes objects and returns their consensus.
    
    This function is designed for more flexible ``numpy`` broadcasting using axes.
    
    Examples
    --------
    >>> broadcast("zyx", "tzyx")  # Axes "tzyx"
    >>> broadcast("tzyx", "tcyx")  # Axes "tzcyx"
    >>> broadcast("yx", "xy")  # Axes "yx"
    """
    n_axes = len(axes_objects)
    
    if n_axes == 2:
        return _broadcast_two(*axes_objects)
    elif n_axes < 2:
        raise TypeError("Less than two axes objects were given.")
    
    it = iter(axes_objects)
    axes0 = next(it)
    for axes1 in it:
        axes0 = _broadcast_two(axes0, axes1)
    return axes0
