from __future__ import annotations
import numpy as np
from typing import Literal

from coords_array.typing import CoordinateLike

from .array import CoordsArray

def wraps(npfunc):
    def _wraps(ipfunc):
        ipfunc.__doc__ = npfunc.__doc__
        return ipfunc
    return _wraps

def __getattr__(name: str):
    npfunc = getattr(np.random, name)
    @wraps(npfunc)
    def _func(*args, **kwargs) -> CoordsArray:
        coords = kwargs.pop("coords", None)
        out = npfunc(*args, **kwargs)
        return CoordsArray(out, coords=coords)
    return _func

@wraps(np.random.random)
def random(
    size, 
    *,
    coords: CoordinateLike | None = None,
) -> CoordsArray:
    return CoordsArray(np.random.random(size), coords=coords)

@wraps(np.random.normal)
def normal(
    loc: float = 0.0, 
    scale: float = 1.0,
    size=None, 
    *,
    coords: CoordinateLike | None = None,
) -> CoordsArray:
    return CoordsArray(np.random.normal(loc, scale, size), coords=coords)

def random_uint8(
    size: int | tuple[int], 
    *, 
    coords: CoordinateLike | None = None,
) -> CoordsArray:
    """
    Return a random uint8 image, ranging 0-255.

    Parameters
    ----------
    size : int or tuple of int
        Image shape.
    axes : str, optional
        Image axes.
        
    Returns
    -------
    CoordsArray
        Random Image in dtype ``np.uint8``.
    """
    arr = np.random.randint(0, 255, size, dtype=np.uint8)
    return CoordsArray(arr, coords=coords)

def random_uint16(
    size,
    *, 
    coords: CoordinateLike | None = None,
) -> CoordsArray:
    """
    Return a random uint16 image, ranging 0-65535.

    Parameters
    ----------
    size : int or tuple of int
        Image shape.
    axes : str, optional
        Image axes.
        
    Returns
    -------
    CoordsArray
        Random Image in dtype ``np.uint16``.
    """
    arr = np.random.randint(0, 65535, size, dtype=np.uint16)
    return CoordsArray(arr, coords=coords)


def default_rng(seed) -> ImageGenerator:
    return ImageGenerator(np.random.default_rng(seed))

class ImageGenerator:
    def __init__(self, rng: np.random.Generator):
        self._rng = rng
    
    def standard_normal(
        self,
        size: int | tuple[int, ...] | None = None,
        dtype = None,
        *,
        coords: CoordinateLike | None = None,
    ) -> CoordsArray:
        arr = self._rng.standard_normal(size=size, dtype=dtype)
        if np.isscalar(arr):
            return arr
        return CoordsArray(arr, coords=coords)
    
    def standard_exponential(
        self,
        size: int | tuple[int, ...] | None = None, 
        dtype = None,
        method: Literal["zig", "inv"] = None,
        *,
        coords: CoordinateLike | None = None,
    ) -> CoordsArray:
        arr = self._rng.standard_exponential(size=size, dtype=dtype, method=method)
        if np.isscalar(arr):
            return arr
        return CoordsArray(arr, coords=coords)
    
    def random(
        self,
        size: int | tuple[int, ...] | None = None, 
        dtype = None,
        *,
        coords: CoordinateLike | None = None,
    ) -> CoordsArray:
        arr = self._rng.random(size=size, dtype=dtype)
        if np.isscalar(arr):
            return arr
        return CoordsArray(arr, coords=coords)
    
    def normal(
        self,
        loc: float | np.ndarray = 0.,
        scale: float | np.ndarray = 1.,
        size: int | tuple[int, ...] | None = None,
        *,
        coords: CoordinateLike | None = None,
    ) -> CoordsArray:
        arr = self._rng.normal(loc=loc, scale=scale, size=size)
        if np.isscalar(arr):
            return arr
        return CoordsArray(arr, coords=coords)

    def poisson(
        self,
        lam: float,
        size: int | tuple[int, ...] | None = None,
        *,
        coords: CoordinateLike | None = None,
    ) -> CoordsArray:
        arr = self._rng.poisson(lam=lam, size=size)
        if np.isscalar(arr):
            return arr
        return CoordsArray(arr, coords=coords)
    
    def random_uint8(
        self,
        size: int | tuple[int], 
        *, 
        coords: CoordinateLike | None = None,
    ) -> CoordsArray:
        arr = self._rng.integers(0, 255, size, dtype=np.uint8)
        return CoordsArray(arr, coords=coords)

    def random_uint16(
        self,
        size,
        *, 
        coords: CoordinateLike | None = None,
    ) -> CoordsArray:
        arr = self._rng.integers(0, 65535, size, dtype=np.uint16)
        return CoordsArray(arr, coords=coords)

del wraps
