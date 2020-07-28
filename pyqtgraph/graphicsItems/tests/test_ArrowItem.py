import pyqtgraph as pg

app = pg.mkQApp()


def test_ArrowItem():
    parent = pg.GraphicsObject()
    a = pg.ArrowItem(parent=parent, pos=(10, 10))
    assert a.parentItem() is parent
    assert a.pos() == pg.Point(10, 10)
