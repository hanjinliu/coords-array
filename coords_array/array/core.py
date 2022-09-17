from __future__ import annotations
from typing import Any, Mapping, SupportsInt, TYPE_CHECKING, Union
from enum import Enum
import numpy as np
from numpy.typing import DTypeLike

from ._mixin import CoordinatesMixin
from . import _axes_ops as axesop

from ..coords import CoordinateError, Coordinates, Slicer
from ..typing import CoordinateLike

if TYPE_CHECKING:
    from typing_extensions import Self

SupportOneSlicing = Union[SupportsInt, slice]
SupportSlicing = Union[
    SupportsInt,
    str,
    slice,
    tuple[SupportOneSlicing, ...],
    Mapping[str, SupportOneSlicing],
]


class CoordsArray(np.ndarray, CoordinatesMixin):
    _additional_props = ("_source", "_metadata", "_name")
    NP_DISPATCH = {}

    def __new__(
        cls: type[CoordsArray],
        obj,
        *,
        coords: CoordinateLike | None = None,
        dtype: DTypeLike = None,
    ) -> Self:
        if isinstance(obj, cls):
            return obj

        self = np.asarray(obj, dtype=dtype).view(cls)
        self.coords = coords
        return self

    @property
    def value(self) -> np.ndarray:
        """Numpy view of the array."""
        return np.asarray(self)

    def __repr__(self) -> str:
        if self.ndim > 0:
            return CoordinatesMixin.__repr__(self)
        return self.value[()]

    @property
    def shape(self):
        return self.coords.tuple(super().shape)

    def __getitem__(self, key: SupportSlicing) -> Self:

        if isinstance(key, (Mapping, Slicer)):
            key = self.coords.create_slice(key)

        if isinstance(key, np.ndarray):
            key = self._broadcast(key)

        out = super().__getitem__(key)  # get item as np.ndarray

        if isinstance(
            out, self.__class__
        ):  # cannot set attribution to such as numpy.int32
            out._inherit_coordinates(self, coords=axesop.slice_axes(self.coords, key))

        return out

    def __setitem__(self, key: SupportSlicing, value):
        if isinstance(key, (Mapping, Slicer)):
            key = self.coords.create_slice(key)

        if isinstance(key, CoordsArray) and key.dtype == bool:
            key = axesop.add_axes(self.coords, self.shape, key, key.coords)

        elif isinstance(key, np.ndarray) and key.dtype == bool and key.ndim == 2:
            # img[arr] ... where arr is 2-D boolean array
            key = axesop.add_axes(self.coords, self.shape, key)

        super().__setitem__(key, value)

    def sel(self, indexer=None, /, **kwargs: dict[str, Any]) -> Self:
        """."""
        if indexer is not None:
            kwargs.update(indexer)
        axes = self.coords
        slices = [slice(None)] * self.ndim
        for k, v in kwargs.items():
            idx = axes.find(k)
            axis = axes[idx]
            if lbl := axis.index:
                if isinstance(v, list):
                    slices[idx] = [lbl.index(each) for each in v]
                else:
                    slices[idx] = lbl.to_indexer(v)
            else:
                raise ValueError(f"Cannot select {k} because it has no coordinates.")
        return self[tuple(slices)]

    def isel(self, indexer=None, /, **kwargs: dict[str, Any]) -> Self:
        if indexer is not None:
            kwargs = dict(indexer, **kwargs)

        key = self.coords.create_slice(kwargs)
        out = super().__getitem__(key)  # get item as np.ndarray

        if isinstance(
            out, self.__class__
        ):  # cannot set attribution to such as numpy.int32
            out._inherit_coordinates(self, coords=axesop.slice_axes(self.coords, key))
        return out

    def __array_finalize__(self, obj):
        """
        Every time an np.ndarray object is made by numpy functions inherited to ImgArray,
        this function will be called to set essential attributes. Therefore, you can use
        such as img.copy() and img.astype("int") without problems (maybe...).
        """
        if obj is None:
            return None

        try:
            self.coords = getattr(obj, "axes", None)
        except Exception:
            self.coords = None
        else:
            if len(self.coords) != self.ndim:
                self.coords = None

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        """
        Every time a numpy universal function (add, subtract, ...) is called,
        this function will be called to set/update essential attributes.
        """
        args_, _ = _replace_inputs(self, args, kwargs)

        result = getattr(ufunc, method)(*args_, **kwargs)

        if result is NotImplemented:
            return NotImplemented

        result = result.view(self.__class__)

        # in the case result is such as np.float64
        if not isinstance(result, self.__class__):
            return result

        result._process_output(ufunc, args, kwargs)
        return result

    def __array_function__(self, func, types, args, kwargs):
        """
        Every time a numpy function (np.mean...) is called, this function will be called. Essentially numpy
        function can be overloaded with this method.
        """
        if func in self.__class__.NP_DISPATCH and all(
            issubclass(t, CoordsArray) for t in types
        ):
            return self.__class__.NP_DISPATCH[func](*args, **kwargs)

        args_, _ = _replace_inputs(self, args, kwargs)

        result = func(*args_, **kwargs)

        if result is NotImplemented:
            return NotImplemented

        if isinstance(result, (tuple, list)):
            _as_meta_array = (
                lambda a: a.view(self.__class__)._process_output(func, args, kwargs)
                if type(a) is np.ndarray
                else a
            )
            result = list(_as_meta_array(r) for r in result)

        else:
            if isinstance(result, np.ndarray):
                result = result.view(self.__class__)
            # in the case result is such as np.float64
            if isinstance(result, self.__class__):
                result._process_output(func, args, kwargs)

        return result

    @classmethod
    def implements(cls, numpy_function):
        """
        Add functions to NP_DISPATCH so that numpy functions can be overloaded.
        """

        def decorator(func):
            cls.NP_DISPATCH[numpy_function] = func
            func.__name__ = numpy_function.__name__
            return func

        return decorator

    def argmax_nd(self) -> tuple[int, ...]:
        """
        N-dimensional version of argmax.

        For instance, if yx-array takes its maximum at (5, 8), this function returns
        ``AxesShape(y=5, x=8)``.

        Returns
        -------
        AxesShape
            Argmax of the array.
        """
        argmax = np.unravel_index(np.argmax(self), self.shape)
        return self.coords.tuple(argmax)

    def transpose(self, axis_names) -> Self:
        """
        change the order of image dimensions.
        'axes' will also be arranged.
        """
        _axes = [self.axisof(a) for a in axis_names]
        new_coords = [self.coords[i] for i in list(axis_names)]
        out: np.ndarray = np.transpose(self.value, _axes)
        out: Self = out.view(self.__class__)
        out._inherit_coordinates(self, coords=new_coords)
        return out

    @property
    def T(self) -> Self:
        out = super().T
        out.coords = out.coords[::-1]
        return out

    def _broadcast(self, value: Any):
        """Broadcasting method used in most of the mathematical operations."""
        if not isinstance(value, CoordsArray):
            return value
        current_axes = self.coords
        if (
            current_axes == value.coords
            or current_axes.has_undef()
            or value.coords.has_undef()
        ):
            # In most cases arrays don't need broadcasting. Check axes first to
            # avoid spending time on broadcasting.
            return value
        value = value.broadcast_to(self.shape, current_axes)
        return value

    def broadcast_to(
        self,
        shape: tuple[int, ...],
        coords: CoordinateLike | None = None,
    ) -> Self:
        """
        Broadcast array to specified shape and coordinates.

        Parameters
        ----------
        shape : shape-like
            Shape of output array.
        coords : coordinate-like, optional
            Coordinates of output array. If given, it must match the dimensionality
            of input shape.

        Returns
        -------
        CoordsArray
            Broadcasted array.
        """
        if coords is None:
            return np.broadcast_to(self, shape)
        elif len(shape) != len(coords):
            raise ValueError(f"Dimensionality mismatch: {shape=} and {coords=}")
        current_axes = self.coords
        if self.shape == shape and current_axes == coords:
            return self
        if any(a not in coords for a in current_axes):
            ax0 = [a.name for a in current_axes]
            ax1 = [a.name for a in coords]
            raise CoordinateError(f"Cannot broadcast array with axes {ax0} to {ax1}.")

        out = self.value
        for i, axis in enumerate(coords):
            if axis not in current_axes:
                out = np.stack([out] * shape[i], axis=i)

        out = out.view(self.__class__)

        if out.shape != shape:
            raise ValueError(f"Shape {shape} required but returned {out.shape}.")

        if not isinstance(coords, Coordinates):
            coords = Coordinates(coords)
        out._inherit_coordinates(self, coords=coords)
        return out

    def __add__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__add__(value)

    def __sub__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__sub__(value)

    def __mul__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__mul__(value)

    def __truediv__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__truediv__(value)

    def __mod__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__mod__(value)

    def __floordiv__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__floordiv__(value)

    def __gt__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__gt__(value)

    def __ge__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__ge__(value)

    def __lt__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__lt__(value)

    def __le__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__le__(value)

    def __eq__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__eq__(value)

    def __ne__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__ne__(value)

    def __and__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__and__(value)

    def __or__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__or__(value)

    def __ne__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__ne__(value)

    def __iadd__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__iadd__(value)

    def __isub__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__isub__(value)

    def __imul__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__imul__(value)

    def __itruediv__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__itruediv__(value)

    def __imod__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__imod__(value)

    def __ifloordiv__(self, value) -> Self:
        value = self._broadcast(value)
        return super().__ifloordiv__(value)

    def _process_output(self, func, args, kwargs):
        # find the largest MetaArray. Largest because of broadcasting.
        arr = None
        for arg in args:
            if isinstance(arg, self.__class__):
                if arr is None or arr.ndim < arg.ndim:
                    arr = arg

        if isinstance(arr, self.__class__):
            self._inherit_meta(arr, func, **kwargs)

        return self

    def _inherit_meta(self, obj: CoordinatesMixin, ufunc, **kwargs):
        """
        Copy axis etc. from obj.
        This is called in __array_ufunc__(). Unlike _set_info(), keyword `axis` must be
        considered because it changes `ndim`.
        """
        if "axis" in kwargs.keys():
            new_axes = obj.coords.drop(kwargs["axis"])
        else:
            new_axes = self._INHERIT
        self._inherit_coordinates(obj, coords=new_axes)
        return self

    if TYPE_CHECKING:

        def astype(self, dtype) -> Self:
            ...

        def flatten(self, order="C") -> Self:
            ...

        def ravel(self, order="C") -> Self:
            ...


def _replace_inputs(img: CoordsArray, args: tuple[Any], kwargs: dict[str, Any]):
    _as_np_ndarray = lambda a: a.value if isinstance(a, CoordsArray) else a
    # convert arguments
    args = tuple(_as_np_ndarray(a) for a in args)
    if kwargs.get("axis", None) is not None:
        axis = kwargs["axis"]
        if not hasattr(axis, "__iter__"):
            axis = [axis]
        kwargs["axis"] = tuple(map(img.axisof, axis))

    if kwargs.get("axes", None) is not None:
        # used in such as np.rot90
        axes = kwargs["axes"]
        kwargs["axes"] = tuple(map(img.axisof, axes))

    if kwargs.get("out", None) is not None:
        kwargs["out"] = tuple(_as_np_ndarray(a) for a in kwargs["out"])

    return args, kwargs


def solve_slicer(key: Any, axes: Coordinates) -> tuple[slice, ...]:
    if isinstance(key, (Mapping, Slicer)):
        key = axes.create_slice(key)

    return key


class BroadCastingRule(Enum):
    same_size = "same_size"
    same_name = "same_name"
    same_scale = "same_scale"
