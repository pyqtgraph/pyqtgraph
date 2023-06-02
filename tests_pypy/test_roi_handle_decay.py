import pyqtgraph as pg

app = pg.mkQApp()

def test_roi_handle_decay():
    view = pg.GraphicsView()

    parent = pg.GraphicsObject()        # "roi"
    child = pg.GraphicsObject(parent)   # "handle"
    view.addItem(parent)
    items = view.scene().items()

    for item in items:
        print(item)

    # on PyPy, we get QGraphicsItem instead of pg.GraphicsObject
    assert child in items
