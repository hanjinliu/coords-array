from __future__ import annotations
import numpy as np
import operator
from .core import CoordsArray
from ..coords import Axis, Coordinates, AxisLike, UndefAxis

# Overloading numpy functions using __array_function__.
# https://numpy.org/devdocs/reference/arrays.classes.html


@CoordsArray.implements(np.squeeze)
def _(img: CoordsArray):
    out = np.squeeze(img.value).view(img.__class__)
    new_axes = [a for a in img.coords if img.sizeof(a) > 1]
    out._set_info(img, new_axes)
    return out

@CoordsArray.implements(np.take)
def _(a: CoordsArray, indices, axis=None, out=None, mode="raise"):
    new_axes = a.coords.drop(axis)
    if isinstance(axis, str):
        axis = a.coords.find(axis)
    out = np.take(a.value, indices, axis=axis, out=out, mode=mode).view(a.__class__)
    if isinstance(out, a.__class__):
        out._set_info(a, coords=new_axes)
    return out

@CoordsArray.implements(np.stack)
def _(imgs: list[CoordsArray], axis: AxisLike = 0, dtype=None):
    old_axes = imgs[0].coords
    
    idx = axis
    axis = "#"
    new_axes = old_axes[:idx] + [axis] + old_axes[idx:]
        
    if dtype is None:
        dtype = imgs[0].dtype

    arrs = [img.value.astype(dtype) for img in imgs]

    out = np.stack(arrs, axis=0)
    out = np.moveaxis(out, 0, idx)
    out = out.view(imgs[0].__class__)
    out._set_info(imgs[0], new_axes)
    return out

@CoordsArray.implements(np.concatenate)
def _(imgs: list[CoordsArray], axis=0, dtype=None, casting="same_kind"):
    if not isinstance(axis, (int, str)):
        raise TypeError(f"`axis` must be int or str, but got {type(axis)}")
    axis = imgs[0].axisof(axis)
    out: np.ndarray = np.concatenate(
        [img.value for img in imgs], axis=axis, dtype=dtype, casting=casting
    )
    out = out.view(imgs[0].__class__)
    out._set_info(imgs[0], imgs[0].coords)
    return out

@CoordsArray.implements(np.block)
def _(imgs: list[CoordsArray]):
    def _recursive_view(obj):
        if isinstance(obj, CoordsArray):
            return obj.value
        else:
            return [_recursive_view(a) for a in obj]
    
    def _recursive_get0(obj):
        first = obj[0]
        if isinstance(first, CoordsArray):
            return first
        else:
            return _recursive_get0(first)
    
    img0 = _recursive_get0(imgs)
    
    imgs = _recursive_view(imgs)
    out = np.block(imgs).view(img0.__class__)
    out._set_info(img0, img0.coords)
    return out


@CoordsArray.implements(np.zeros_like)
def _(img: CoordsArray, name: str = None):
    out = np.zeros_like(img.value).view(img.__class__)
    out._set_info(img, coords=img.coords)
    if isinstance(name, str):
        out.name = name
    return out

@CoordsArray.implements(np.empty_like)
def _(img: CoordsArray, name: str = None):
    out = np.empty_like(img.value).view(img.__class__)
    out._set_info(img, coords=img.coords)
    if isinstance(name, str):
        out.name = name
    return out

@CoordsArray.implements(np.expand_dims)
def _(img: CoordsArray, axis):
    if isinstance(axis, str):
        new_axes = Coordinates(axis + str(img.coords))
        axisint = tuple(new_axes.find(a) for a in axis)
    else:
        axisint = axis
        new_axes = list(img.coords)
        new_axes.insert(axis, UndefAxis())
    
    out: np.ndarray = np.expand_dims(img.value, axisint)
    out = out.view(img.__class__)
    out._set_info(img, new_axes)
    return out

@CoordsArray.implements(np.transpose)
def _(img: CoordsArray, axes):
    return img.transpose(axes)

@CoordsArray.implements(np.split)
def _(img: CoordsArray, indices_or_sections, axis=0):
    if not isinstance(axis, (int, str)):
        raise TypeError(f"`axis` must be int or str, but got {type(axis)}")
    axis = img.axisof(axis)
    
    imgs: list[CoordsArray] = np.split(img.value, indices_or_sections, axis=axis)
    out = []
    for each in imgs:
        each = each.view(img.__class__)
        each._set_info(img)
        out.append(each)
    return out

@CoordsArray.implements(np.broadcast_to)
def _(img: CoordsArray, shape: tuple[int, ...]):
    out: np.ndarray = np.broadcast_to(img.value, shape)
    nexpand = len(shape) - img.ndim
    new_axes = [UndefAxis()] * nexpand + list(img.coords)
    out = out.view(img.__class__)
    out._set_info(img, coords=new_axes)
    return out

@CoordsArray.implements(np.moveaxis)
def _(img: CoordsArray, source, destination):
    out = np.moveaxis(img.value, source, destination)
    
    if not hasattr(source, "__iter__"):
        source = [source]
    if not hasattr(destination, "__iter__"):
        destination = [destination]
    
    order = [n for n in range(img.ndim) if n not in source]

    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)

    new_axes = [img.coords[i] for i in order]
    out = out.view(img.__class__)
    out._set_info(img, coords=new_axes)
    return out

@CoordsArray.implements(np.swapaxes)
def _(img: CoordsArray, axis1: int | AxisLike, axis2: int = AxisLike):
    if isinstance(axis1, (str, Axis)):
        axis1 = img.axisof(axis1)
    if isinstance(axis2, (str, Axis)):
        axis2 = img.axisof(axis2)
    out = np.swapaxes(img.value, axis1, axis2)
    out = out.view(img.__class__)
    
    axes_list = list(img.coords)
    axes_list[axis1], axes_list[axis2] = axes_list[axis2], axes_list[axis1]
    
    out._set_info(img, new_axes=axes_list)
    return out

# This function is ported from numpy.core.numeric.normalize_axis_tuple
def np_normalize_axis_tuple(axis, ndim, argname=None, allow_duplicate=False):
    # Optimization to speed-up the most common cases.
    if type(axis) not in (tuple, list):
        try:
            axis = [operator.index(axis)]
        except TypeError:
            pass
    # Going via an iterator directly is slower than via list comprehension.
    axis = tuple([np_normalize_axis_tuple(ax, ndim, argname) for ax in axis])
    if not allow_duplicate and len(set(axis)) != len(axis):
        if argname:
            raise ValueError('repeated axis in `{}` argument'.format(argname))
        else:
            raise ValueError('repeated axis')
    return axis
