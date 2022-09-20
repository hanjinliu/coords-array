from typing import Mapping, Sequence, SupportsIndex, Tuple, Union
from typing_extensions import TypedDict
from numpy.typing import DTypeLike

ShapeLike = Union[SupportsIndex, Tuple[SupportsIndex, ...]]
PartialShapeLike = Union[SupportsIndex, Tuple[Union[SupportsIndex, None], ...]]


class AxisOptions(TypedDict):
    scale: float
    unit: float
    labels: float
    border: Tuple[float, float]


SpaceOptions = Union[
    int,
    Sequence[str],
    # Sequence[Tuple[str, AxisOptions]]
    Mapping[str, AxisOptions],
]
