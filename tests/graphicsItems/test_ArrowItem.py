import pytest

import pyqtgraph as pg

app = pg.mkQApp()

@pytest.mark.skip("Debugging skipping test")
def test_ArrowItem_parent():
    parent = pg.GraphicsObject()
    a = pg.ArrowItem(parent=parent, pos=(10, 10))
    assert a.parentItem() is parent
    assert a.pos() == pg.Point(10, 10)
