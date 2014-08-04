import pyqtgraph as pg
import gc, os

app = pg.mkQApp()

def test_isQObjectAlive():
    o1 = pg.QtCore.QObject()
    o2 = pg.QtCore.QObject()
    o2.setParent(o1)
    del o1
    gc.collect()
    assert not pg.Qt.isQObjectAlive(o2)


def test_loadUiType():
    path = os.path.dirname(__file__)
    formClass, baseClass = pg.Qt.loadUiType(os.path.join(path, 'uictest.ui'))
    w = baseClass()
    ui = formClass()
    ui.setupUi(w)
    w.show()
    app.processEvents()

