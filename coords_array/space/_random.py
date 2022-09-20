from __future__ import annotations

import weakref
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal, overload
import numpy as np

from ..array import CoordsArray
from ..coords import Coordinates
from .._shared_types import ShapeLike, DTypeLike

if TYPE_CHECKING:
    from ._space import Space


class RandomGenerator:
    def __init__(self, space: Space):
        self._space_ref = weakref.ref(space)
        self.seed()

    def seed(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed=seed)

    @contextmanager
    def seeded(self, seed: int | None = None):
        old_rng = self._rng
        self.seed(seed)
        try:
            yield
        finally:
            self._rng = old_rng

    # fmt: off
    @overload
    def standard_normal(self, size: Literal[None], dtype: DTypeLike = None) -> np.number: ...
    @overload
    def standard_normal(self, size: ShapeLike, dtype: DTypeLike = None) -> CoordsArray: ...
    @overload
    def uniform(self, low: float, high: float, size: Literal[None], dtype: DTypeLike = None) -> np.number: ...
    @overload
    def uniform(self, low: float, high: float, size: ShapeLike, dtype: DTypeLike = None) -> CoordsArray: ...
    @overload
    def normal(self, loc: float = 0.0, scale: float = 1.0, size: Literal[None] = None, dtype: DTypeLike = None) -> np.number: ...
    @overload
    def normal(self, loc: float, scale: float, size: ShapeLike, dtype: DTypeLike = None) -> CoordsArray: ...
    @overload
    def random(self, size: Literal[None], dtype: DTypeLike = None) -> np.number: ...
    @overload
    def random(self, size: ShapeLike, dtype: DTypeLike = None) -> CoordsArray: ...
    @overload
    def poisson(self, lam: float, size: Literal[None]) -> np.number: ...
    @overload
    def poisson(self, lam: float, size: ShapeLike) -> CoordsArray: ...
    # fmt: on

    def standard_normal(
        self,
        size=None,
        dtype=None,
    ) -> CoordsArray:
        arr = self._rng.standard_normal(size=size, dtype=dtype)
        if np.isscalar(arr):
            return arr
        return CoordsArray(arr, coords=self._build_coords(arr.shape))

    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: int | tuple[int, ...] | None = None,
        dtype=None,
    ) -> CoordsArray:
        arr = self._rng.uniform(low=low, high=high, size=size, dtype=dtype)
        if np.isscalar(arr):
            return arr
        return CoordsArray(arr, coords=self._build_coords(arr.shape))

    def normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: int | tuple[int, ...] | None = None,
        dtype=None,
    ) -> CoordsArray:
        arr = self._rng.normal(loc=loc, scale=scale, size=size, dtype=dtype)
        if np.isscalar(arr):
            return arr
        return CoordsArray(arr, coords=self._build_coords(arr.shape))

    def random(
        self, size: int | tuple[int, ...] | None = None, dtype=None
    ) -> CoordsArray:
        arr = self._rng.random(size=size, dtype=dtype)
        if np.isscalar(arr):
            return arr
        return CoordsArray(arr, coords=self._build_coords(arr.shape))

    def poisson(
        self,
        lam: float | np.ndarray = 1.0,
        size: int | tuple[int, ...] | None = None,
    ) -> CoordsArray:
        arr = self._rng.poisson(lam=lam, size=size)
        if np.isscalar(arr):
            return arr
        return CoordsArray(arr, coords=self._build_coords(arr.shape))

    def _build_coords(self, shape) -> Coordinates:
        return self._space_ref().build_coords(shape)
