from coords_array import Space
from coords_array.typing import PartialShapeLike, ShapeLike
import pytest

@pytest.mark.parametrize(
    "input, axes",
    [(["y", "x"], ("y", "x")),
      (2, ("axis_0", "axis_1")),
      ],
)
def test_space_construction(input, axes):
    space = Space(input)
    assert space.axes == axes

@pytest.mark.parametrize(
    "input_shape, output_shape, axes",
    [((3, 4, 5), (3, 4, 5), ("z", "y", "x")),
     ((3, 4, None), (3, 4), ("z", "y")),
     ((3, None, 5), (3, 5), ("z", "x")),
     ((None, 4, 5), (4, 5), ("y", "x")),]
)
def test_array_construction(
    input_shape: PartialShapeLike,
    output_shape: ShapeLike,
    axes: tuple[str, ...],
):
    space = Space(["z", "y", "x"])
    arr = space.zeros(input_shape)
    assert arr.shape == output_shape
    assert arr.coords.axes == axes
