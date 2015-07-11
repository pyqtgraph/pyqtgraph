import gc
import weakref
# try:
#     import faulthandler
#     faulthandler.enable()
# except ImportError:
#     pass

from pyqtgraph.Qt import QtCore, QtGui, QtTest
import numpy as np
import pyqtgraph as pg
app = pg.mkQApp()


def test_dividebyzero():
    import pyqtgraph as pg
    im = pg.image(pg.np.random.normal(size=(100,100)))
    im.imageItem.setAutoDownsample(True)
    im.view.setRange(xRange=[-5+25, 5e+25],yRange=[-5e+25, 5e+25])
    app.processEvents()
    QtTest.QTest.qWait(1000)
    # must manually call im.imageItem.render here or the exception
    # will only exist on the Qt event loop
    im.imageItem.render()


if __name__ == "__main__":
    test_dividebyzero()


# def test_getViewWidget():
#     view = pg.PlotWidget()
#     vref = weakref.ref(view)
#     item = pg.InfiniteLine()
#     view.addItem(item)
#     assert item.getViewWidget() is view
#     del view
#     gc.collect()
#     assert vref() is None
#     assert item.getViewWidget() is None
#
# def test_getViewWidget_deleted():
#     view = pg.PlotWidget()
#     item = pg.InfiniteLine()
#     view.addItem(item)
#     assert item.getViewWidget() is view
#
#     # Arrange to have Qt automatically delete the view widget
#     obj = pg.QtGui.QWidget()
#     view.setParent(obj)
#     del obj
#     gc.collect()
#
#     assert not pg.Qt.isQObjectAlive(view)
#     assert item.getViewWidget() is None


#if __name__ == '__main__':
    #view = pg.PlotItem()
    #vref = weakref.ref(view)
    #item = pg.InfiniteLine()
    #view.addItem(item)
    #del view
    #gc.collect()
    
