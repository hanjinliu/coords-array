from __future__ import annotations
import numpy as np
from ..coords import Coordinates, Axis, AxisLike


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


def complement_axes(axes, all_axes="ptzcyx") -> list[AxisLike]:
    c_axes = []
    axes_list = list(axes)
    for a in all_axes:
        if a not in axes_list:
            c_axes.append(a)
    return c_axes


def switch_slice(axes, all_axes, ifin=np.newaxis, ifnot=":"):
    axes = list(axes)
    if ifnot == ":":
        ifnot = [slice(None)] * len(all_axes)
    elif not hasattr(ifnot, "__iter__"):
        ifnot = [ifnot] * len(all_axes)

    if not hasattr(ifin, "__iter__"):
        ifin = [ifin] * len(all_axes)

    sl = []
    for a, slin, slout in zip(all_axes, ifin, ifnot):
        if a in axes:
            sl.append(slin)
        else:
            sl.append(slout)
    sl = tuple(sl)
    return sl


def slice_axes(coords: Coordinates, key) -> Coordinates:
    ndim = len(coords)
    if isinstance(key, tuple):
        ndim += sum(k is None for k in key)
        rest = ndim - len(key)
        if any(k is ... for k in key):
            idx = key.index(...)
            _keys = key[:idx] + (slice(None),) * (rest + 1) + key[idx + 1 :]
        else:
            _keys = key + (slice(None),) * rest

    elif isinstance(key, np.ndarray) or hasattr(key, "__array__"):
        if key.ndim == 1:
            new_coords = coords
        else:
            new_coords = Coordinates([None] + coords._axis_list[key.ndim :])
        return new_coords

    elif key is None:
        return Coordinates([None] + coords._axis_list)

    elif key is ...:
        return coords

    else:
        _keys = (key,) + (slice(None),) * (ndim - 1)

    _new_axes_list: list[Axis] = []
    list_idx: list[int] = []

    axes_iter = iter(coords)
    for sl in _keys:
        if sl is not None:
            a = next(axes_iter)
            if isinstance(sl, (slice, np.ndarray)):
                _new_axes_list.append(a.slice_axis(sl))
            elif isinstance(sl, list):
                _new_axes_list.append(a.slice_axis(sl))
                list_idx.append(a)
        else:
            _new_axes_list.append(None)  # new axis

    if len(list_idx) > 1:
        added = False
        out: list[Axis] = []
        for a in _new_axes_list:
            if a not in list_idx:
                out.append(a)
            elif not added:
                out.append(None)
                added = True
        _new_axes_list = out

    return Coordinates(_new_axes_list)
