import pytest
import coords_array as cr
import numpy as np
from coords_array.coords import CoordinateError, broadcast, Axis

@pytest.mark.parametrize("axes", [["t", "z", "y", "x"], ["time", "z", ":y", ":x"]])
def test_axes(axes):
    arr = cr.random.random_uint8((10, 10, 10))
    arr.coords = axes[1:]
    
    tyx = axes[0:1] + axes[2:]
    zyx = axes[1:]
    zx = axes[1:2] + axes[3:4]
    
    arr.set_scale({axes[1]: 0.3})
    assert arr.coords == zyx
    assert list(arr.scale.keys()) == zyx
    assert arr.coords[axes[1]].scale == 0.3
    
    arr1 = arr.gaussian_filter(dims=axes[1:])
    assert arr1.axes == zyx
    assert list(arr1.scale.keys()) == zyx
    assert arr1.axes[axes[1]].scale == 0.3
    
    arr1.axes = arr1.axes.replace(axes[1], axes[0])
    assert arr1.axes == tyx
    assert arr.coords == zyx
        
    arr2 = arr.proj(axes[2:3])
    assert arr2.axes == zx
    assert list(arr2.scale.keys()) == zx
    assert arr2.axes[axes[1]].scale == 0.3

def test_getattr():
    arr = cr.zeros((10, 10, 10), axes="zyx")
    assert arr.axes[0] == arr.axes.z == arr.axes["z"]
    assert arr.axes[1] == arr.axes.y == arr.axes["y"]
    assert arr.axes[2] == arr.axes.x == arr.axes["x"]
    
@pytest.mark.parametrize("axes", [["t", "z", "y", "x"], ["time", "z", ":y", ":x"]])
def test_set_axes(axes):
    img = cr.random.random_uint8((10, 10, 10), axes="zyx")
    tyx = axes[0:1] + axes[2:]
    img.coords = tyx
    assert img.coords == tyx
    with pytest.raises(CoordinateError):
        img.coords = axes[-1:]
    assert img.coords == tyx
    img.coords = None
    assert str(img.coords) == "###"

    
def test_set_scale():
    arr = cr.random.random_uint8((10, 10, 10), axes="zyx")
    assert arr.scale.z == arr.scale.y == arr.scale.x == 1
    arr.scale = {"z": 0.5, "y": 0.4, "x": 0.4}
    assert arr.scale.z == 0.5
    assert arr.scale.y == arr.scale.x == 0.4
    
    with pytest.raises(Exception):
        arr.scale["t"] = 1  # cannot set scale to an axis that image does not have.
    with pytest.raises(Exception):
        arr.scale.t = 1  # cannot set scale to an axis that image does not have.
    with pytest.raises(ValueError):
        arr.scale["z"] = 0  # cannot set zero
    
    arr.scale.z = 0.3
    assert arr.scale.z == 0.3
        
    arr1 = arr.gaussian_filter()
    arr1.scale.z = 0.4
    assert arr.scale.z == 0.3
    assert arr1.scale.z == 0.4
    
    arr2 = arr.binning(2)
    assert arr2.scale.y == arr2.scale.x == 0.8

@pytest.mark.parametrize("axes", [["z", "y", "x"], [":z", ":y", ":x"]])
def test_numpy(axes):
    img = cr.random.random_uint8((10, 10, 10), axes=axes)
    assert np.all(np.array(img.scale) == np.ones(3))

@pytest.mark.parametrize("axes", [["t", "z", "y", "x"], ["time", ":z", ":y", ":x"]])
def test_slicing(axes):
    arr = cr.random.random_uint8((10, 10, 10, 10), axes=axes)
    tzyx = axes
    tyx = axes[0:1] + axes[2:]
    zyx = axes[1:]
    yx = axes[2:]
    zx = axes[1:2] + axes[3:4]
    _yx = ["#"] + yx
    _zx = ["#"] + zx
    
    assert arr.coords == tzyx
    assert arr[0].coords == zyx
    assert arr[0, 0].coords == yx
    assert arr[1, 1, 2].coords == axes[3]
    assert arr[1, :, 2].coords == zx
    assert arr[1, :, 5:7].coords == zyx
    assert arr[:, 0].coords == tyx
    assert arr[[1, 3, 5]].coords == tzyx
    assert arr[5, [1, 3, 5]].coords == zyx
    assert arr[[1, 2, 3], [1, 2, 3]].coords == _yx
    assert arr[[1, 2, 3], :, [1, 2, 3]].coords == _zx
    assert arr[:, [1, 2, 3], :, [1, 2, 3]].coords == tzyx[0:1] + ["#"] + tzyx[2:3]
    assert arr[[1, 3, 5], [1, 2, 3], :, [1, 2, 3]].coords == ["#"] + tzyx[2:3]
    
    # test new axis
    assert arr[np.newaxis].coords == ["#"] + tzyx
    assert arr[:, :, np.newaxis].coords == axes[0:2] + ["#"] + axes[2:]
    assert arr[np.newaxis, :, np.newaxis].coords == ["#", axes[0], "#"] + zyx
    
    # test array slicing
    sl = arr[:, 0, 0, 0].value > 120
    assert arr[sl].coords == tzyx
    assert arr[0, sl].coords == zyx
    
    sl = arr[:, :, 0, 0].value > 120
    assert arr[sl].coords == _yx
    
    sl = arr[:, :, :, 0].value > 120
    assert arr[sl].coords == ["#"] + tzyx[3:4]
    
    # test ellipsis
    assert arr[..., 0].coords == tzyx[:-1]
    assert arr[..., 0, :].coords == tzyx[:2] + tzyx[3:4]
    assert arr[0, ..., 0].coords == tzyx[1:3]

def test_axis_coordinates_str():
    arr = cr.zeros((4, 10, 10), axes="cyx")
    with pytest.raises(ValueError):
        arr.axes["c"].coords =["c0", "c1", "c2"]
    arr.axes["c"].coords = ["c0", "c1", "c2", "c3"]
    assert arr.axes["c"].coords == ("c0", "c1", "c2", "c3")
    assert arr[:2].axes["c"].coords == ("c0", "c1")
    assert arr[2:].axes["c"].coords == ("c2", "c3")
    assert arr[[0, 2]].axes["c"].coords == ("c0", "c2")
    assert arr[2::-1].axes["c"].coords == ("c2", "c1", "c0")

def test_transformed_axis():
    x = Axis("x", scale=2, unit="nm")
    y = Axis("y", scale=1, unit="nm")
    u = 2 * x + y
    v = x - 2 * y
    assert u.name == "2x+1y"
    assert v.name == "1x-2y"
    a = u + v
    assert a.name == "3x-1y"
    
    assert u.scale == pytest.approx(np.linalg.norm([2*2, 1*1]))
    assert v.scale == pytest.approx(np.linalg.norm([1*2, 2*1]))
    assert a.scale == pytest.approx(np.linalg.norm([3*2, 1*1]))
    
    assert u.unit == "nm"
    assert v.unit == "nm"
    assert a.unit == "nm"

def test_broadcast():
    assert broadcast("zyx", "tzyx") == "tzyx"
    assert broadcast("yx", "tzyx") == "tzyx"
    assert broadcast("z", "tzyx") == "tzyx"
    assert broadcast("tzcyx", "tyx") == "tzcyx"
    assert broadcast("tzyx", "tcyx") == "tzcyx"
    assert broadcast("yx", "xy") == "yx"
    assert broadcast("tyx", "xy") == "tyx"
    assert broadcast("y", "x") == "yx"
    assert broadcast("z", "y", "x") == "zyx"
    assert broadcast("dz", "dy", "dx") == "dzyx"
    assert broadcast("tzyx", "tyx", "tzyx", "yzx") == "tzyx"
