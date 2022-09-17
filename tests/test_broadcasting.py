import numpy as np
import coords_array as cr
from coords_array.coords import CoordinateError
import pytest

def test_operators():
    arr = cr.random.normal(size=(3, 4, 10), coords="tyx")
    fit = cr.CoordsArray(np.arange(1, 41).reshape(4, 10), coords="yx")
    fit0 = np.stack([fit]*3, axis="t")
    assert np.all(arr/fit == arr/fit0)
    assert np.all(arr[fit>12] == arr[fit0>12])

def test_mismatch():
    img0 = cr.zeros((3, 3), coords="yx")
    img1 = cr.zeros((3, 3), coords="zy")
    
    with pytest.raises(CoordinateError):
        img0 + img1

def test_broadcasting_arrays():
    out = cr.broadcast_arrays(
        cr.zeros((3, 4, 5), coords="zyx"),
        cr.zeros((4, 5), coords="yx"),
        cr.zeros((3, 4), coords="zy"),
    )
    for i in range(3):
        assert out[i].shape == (3, 4, 5)
        assert out[i].coords == ["z", "y", "x"]
    
    out = cr.broadcast_arrays(
        cr.zeros((3,), coords="z"),
        cr.zeros((4,), coords="y"),
        cr.zeros((5,), coords="x"),
    )
    for i in range(3):
        assert out[i].shape == (3, 4, 5)
        assert out[i].coords == ["z", "y", "x"]
    
    out = cr.broadcast_arrays(
        cr.zeros((3, 3), coords="dz"),
        cr.zeros((3, 4), coords="dy"),
        cr.zeros((3, 5), coords="dx"),
    )
    for i in range(3):
        assert out[i].shape == (3, 3, 4, 5)
        assert out[i].coords == ["d", "z", "y", "x"]

def test_error():
    with pytest.raises(Exception):
        cr.broadcast_arrays(
            cr.zeros((3, 4, 5), coords="zyx"),
            cr.zeros((4, 6), coords="yx"),
        )

    with pytest.raises(Exception):
        cr.broadcast_arrays(
            cr.zeros((3, 4, 5), coords="zyx"),
            cr.zeros((5, 5), coords="yt"),
        )
        
    with pytest.raises(Exception):
        cr.broadcast_arrays(
            cr.zeros((3, 4, 5), coords="zyx"),
            cr.zeros((4, 5), coords="zy"),
        )