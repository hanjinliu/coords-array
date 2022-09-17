from __future__ import annotations
from typing import Any, Hashable, Dict, Union, Iterable, TypedDict
from numpy.typing import ArrayLike
from .coords import Coordinates

class CoordinateOptions(TypedDict):
    scale: float
    unit: str
    labels: Iterable[Hashable]

MetricLike = Union[
    CoordinateOptions,
    range,
    Iterable[Hashable],
]

CoordinateLike = Union[
    Coordinates, 
    Iterable[Hashable], 
    Dict[Hashable, MetricLike],  # coordinate options
]
