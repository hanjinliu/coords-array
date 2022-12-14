from __future__ import annotations
from typing import TYPE_CHECKING, Any

from ..coords import Coordinates, CoordinateError, build_coords
from ..coords._coordinates import CoordinateLike

if TYPE_CHECKING:
    from typing_extensions import Self


class CoordinatesMixin:
    """Abstract class that shape and coordinates are defined."""

    _INHERIT = object()

    @property
    def coords(self) -> Coordinates:
        """Axes of the array."""
        return self._coords

    @coords.setter
    def coords(self, value: CoordinateLike | None):
        if value is None:
            self._coords = Coordinates.undef(self._get_shape())
        else:
            self._coords = build_coords(value, self._get_shape())

    @property
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError()

    def _get_shape(self) -> tuple[int, ...]:
        raise NotImplementedError()

    @property
    def ndim(self) -> int:
        return len(self._get_shape())

    @property
    def value(self) -> Any:
        raise NotImplementedError()

    def __repr__(self) -> str:
        _cls = f"{self.__class__.__name__}"
        _shape = f"shape:\n  {self.shape!r}"
        _coords = f"coords:\n{self.coords._repr(2)}"
        _value = f"value:\n{self.value!r}".replace("\n", "\n  ")
        return f"{_cls} object with\n\n{_shape}\n\n{_coords}\n\n{_value}"

    def _inherit_coordinates(self, other: Self, coords: Any = _INHERIT):
        try:
            if coords is not self._INHERIT:
                self.coords = coords
            else:
                self.coords = other.coords.copy()
        except CoordinateError:
            self.coords = None

        return None
