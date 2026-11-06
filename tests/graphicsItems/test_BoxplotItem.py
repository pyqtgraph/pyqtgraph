import numpy as np

import pyqtgraph as pg
from pyqtgraph import functions as fn
from pyqtgraph.Qt import QtGui


def render(**opts):
    item = pg.BoxplotItem(data=[[1, 2, 3, 4, 5, 6, 7, 8, 50]], pen="k", **opts)
    image = QtGui.QImage(100, 100, QtGui.QImage.Format.Format_RGBA8888)
    image.fill(QtGui.QColor("white"))
    painter = QtGui.QPainter(image)
    painter.translate(50, 90)
    painter.scale(50, -1.5)
    item.paint(painter)
    painter.end()
    return fn.ndarray_from_qimage(image).copy()


def test_outliers_drawn_by_default():
    app = pg.mkQApp()
    assert not np.array_equal(render(), render(outlier=False))
    # an explicit None still hides them
    assert np.array_equal(render(symbolPen=None, symbolBrush=None), render(outlier=False))
