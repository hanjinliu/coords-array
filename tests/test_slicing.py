import coords_array as cr
import numpy as np
from numpy.testing import assert_equal
import pytest

def test_slicer_slicing():
    img = cr.random.normal(size=(10, 2, 30, 40), coords = "tcyx")
    assert_equal(img[cr.slicer.t[4].c[0]], img.value[4,0])
    assert_equal(img[cr.slicer.c[0].x[10:30]], img.value[:,0,:,10:30])
    assert_equal(img[cr.slicer.y[3, 6, 20, 26].x[7, 3, 4, 13]], img.value[:,:,[3,6,20,26],[7,3,4,13]])
    assert_equal(img[cr.slicer.t[::-1].x[2:-1:3]], img.value[::-1,:,:,2:-1:3])

def test_sel_slicing():
    coords = dict(
        t=[f"t={i}" for i in range(10)],
        c=["blue", "red"],
        y=[str(i) for i in range(30)],
        x=[str(i) for i in range(40)],
    )
    arr = cr.random.normal(size=(10, 2, 30, 40), coords=coords)
    assert_equal(arr.sel(t="t=4", c="blue"), arr.value[4,0])
    assert_equal(arr.sel(c="blue", x=slice("10", "30")), arr.value[:,0,:,10:30])
    assert_equal(arr.sel(y=["3", "6", "20", "26"], x=["7", "3", "4", "13"]), arr.value[:,:,[3,6,20,26],[7,3,4,13]])
    assert_equal(arr.sel(t=slice(None, None, -1), x=slice("2", "39", 3)), arr.value[::-1,:,:,2:40:3])


def test_formatter():
    img = cr.random.normal(size=(10, 2, 30, 40))
    fmt = cr.slicer.y[4].get_formatter("tx")

    with pytest.raises(Exception):
        img[fmt]

    # test repr
    repr(cr.slicer)
    repr(fmt)

    assert_equal(img[fmt[0, 0]], img["t=0;y=4;x=0"])
    assert_equal(img[fmt[0, 3:6]], img["t=0;y=4;x=3:6"])
    assert_equal(img[fmt[:5][:6]], img["t=:5;y=4;x=:6"])

    fmt = cr.slicer.get_formatter("tx")
    with pytest.raises(Exception):
        img[fmt]

    # test repr
    repr(cr.slicer)
    repr(fmt)

    assert_equal(img[fmt[0, 0]], img["t=0;x=0"])
    assert_equal(img[fmt[0, 3:6]], img["t=0;x=3:6"])
    assert_equal(img[fmt[:5][:6]], img["t=:5;x=:6"])

def test_scale():
    arr = cr.zeros((10, 10), coords=[])
    arr.set_scale(xy=0.2)
    assert arr[0].coords[-1].scale == arr.coords[-1].scale
    assert arr[0, 2:4].coords[-1].scale == arr.coords[-1].scale
    assert arr[0, ::-1].coords[-1].scale == arr.coords[-1].scale
    assert arr[0, ::2].coords[-1].scale == arr.coords[-1].scale * 2
    assert arr[0, ::-3].coords[-1].scale == arr.coords[-1].scale * 3
    # assert img.binning(2).coords[-1].scale == img.coords[-1].scale * 2
    # assert img.rescale(1/4).coords[-1].scale == img.coords[-1].scale * 4
