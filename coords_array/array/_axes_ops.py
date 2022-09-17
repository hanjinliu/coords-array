from __future__ import annotations
from math import prod
import numpy as np
from ..coords import Coordinates, Axis, pick_axis


def add_axes(axes: Coordinates, shape: tuple[int, ...], key: np.ndarray, key_axes="yx"):
    """
    Stack `key` to make its shape key_axes-> axes.
    """
    key_axes = list(key_axes)
    if shape == key.shape:
        return key

    for i, o in enumerate(axes):
        if o not in key_axes:
            key = np.stack([key] * shape[i], axis=i)
    return key


def slice_coordinates(coords: Coordinates, key, shape: tuple[int, ...]) -> Coordinates:
    """Slice a coordinates and return a new coordinates with given shape."""

    ndim = len(coords)
    names = {a.name for a in coords}

    if isinstance(key, tuple):
        ndim += sum(k is None for k in key)
        rest = ndim - len(key)
        if any(k is ... for k in key):
            idx = key.index(...)
            _keys = key[:idx] + (slice(None),) * (rest + 1) + key[idx + 1 :]
        else:
            _keys = key + (slice(None),) * rest

    elif isinstance(key, np.ndarray) or hasattr(key, "__array__"):
        ndim = key.ndim
        if ndim == 1:
            new_coords = coords
        else:
            new_size = prod(shape[:ndim])
            new_coords = Coordinates(
                pick_axis(new_size, names) + coords._axis_list[ndim:]
            )
        return new_coords

    elif key is None:
        return Coordinates([pick_axis(shape[0], names)] + coords._axis_list)

    elif key is ...:
        return coords

    else:
        _keys = (key,) + (slice(None),) * (ndim - 1)

    # "_keys" is the same length as "shape" and "_new_axes_list"

    _new_axes_list: list[Axis] = []
    list_idx: list[int] = []

    axes_iter = iter(coords)
    shape_iter = iter(shape)
    for sl in _keys:
        if sl is not None:
            a = next(axes_iter)
            if isinstance(sl, (slice, np.ndarray)):
                size = next(shape_iter)
                _new_axes_list.append(a.slice_axis(sl))
            elif isinstance(sl, list):
                size = next(shape_iter)
                _new_axes_list.append(a.slice_axis(sl))
                list_idx.append(a)
        else:
            size = next(shape_iter)
            _new_axes_list.append(pick_axis(size, names))  # new axis

    if len(list_idx) > 1:
        added = False
        out: list[Axis] = []
        for size, a in zip(shape, _new_axes_list):
            if a not in list_idx:
                out.append(a)
            elif not added:
                out.append(pick_axis(size, names))
                added = True
        _new_axes_list = out

    return Coordinates(_new_axes_list)
