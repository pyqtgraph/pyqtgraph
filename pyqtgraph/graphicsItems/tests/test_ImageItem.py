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
