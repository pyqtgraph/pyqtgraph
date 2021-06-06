import pyqtgraph as pg

app = pg.mkQApp()


def test_TextItem_setAngle():
    plt = pg.plot()
    plt.setXRange(-10, 10)
    plt.setYRange(-20, 20)
    item = pg.TextItem(text="test")
    plt.addItem(item)

    t1 = item.transform()

    item.setAngle(30)
    app.processEvents()

    t2 = item.transform()

    assert t1 != t2
    assert not t1.isRotating()
    assert t2.isRotating()
