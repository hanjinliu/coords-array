from ._coordinates import Coordinates, CoordinateError, broadcast, as_metric, as_coordinates
from ._axis import Axis, as_axis, UndefAxis, AxisLike, MetricLike
from ._slicer import Slicer
from ._axes_tuple import AxesTuple

slicer = Slicer()  # default slicer object

__all__ = [
    "Coordinates",
    "MetricLike",
    "CoordinateError",
    "broadcast",
    "Axis",
    "as_axis",
    "UndefAxis",
    "AxisLike",
    "AxesTuple",
    "slicer",
]