import coords_array as cr

def test_shapes():
    shape = (4, 5, 6)
    img = cr.zeros(shape, coords="zyx")
    assert img.shape == shape
    assert img.shape.z == img.shape["z"] == shape[0]
    assert img.shape.y == img.shape["y"] == shape[1]
    assert img.shape.x == img.shape["x"] == shape[2]
    
    shape = (9, 3, 12)
    img = cr.zeros(shape, coords="zyx")
    assert img.shape == shape
    assert img.shape.z == img.shape["z"] == shape[0]
    assert img.shape.y == img.shape["y"] == shape[1]
    assert img.shape.x == img.shape["x"] == shape[2]
