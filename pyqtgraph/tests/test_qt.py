import pyqtgraph as pg
import gc

def test_isQObjectAlive():
    o1 = pg.QtCore.QObject()
    o2 = pg.QtCore.QObject()
    o2.setParent(o1)
    del o1
    gc.collect()
    assert not pg.Qt.isQObjectAlive(o2)
