from ._coordinates import (
    Coordinates,
    CoordinateError,
    broadcast,
    as_index,
    as_coordinates,
)
from ._axis import Axis, as_axis, AxisLike, IndexLike, pick_axis
from ._slicer import Slicer
from ._axes_tuple import AxesTuple

slicer = Slicer()  # default slicer object

__all__ = [
    "Coordinates",
    "IndexLike",
    "CoordinateError",
    "broadcast",
    "Axis",
    "as_axis",
    "AxisLike",
    "AxesTuple",
    "slicer",
]
