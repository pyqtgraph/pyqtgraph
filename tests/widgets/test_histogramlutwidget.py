"""
HistogramLUTWidget test:

Tests the creation of a HistogramLUTWidget.
"""

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets


def testHistogramLUTWidget():
    pg.mkQApp()
    
    win = QtWidgets.QMainWindow()
    win.show()

    cw = QtWidgets.QWidget()
    win.setCentralWidget(cw)

    l = QtWidgets.QGridLayout()
    cw.setLayout(l)
    l.setSpacing(0)

    v = pg.GraphicsView()
    vb = pg.ViewBox()
    vb.setAspectLocked()
    v.setCentralItem(vb)
    l.addWidget(v, 0, 0, 3, 1)

    w = pg.HistogramLUTWidget(background='w')
    l.addWidget(w, 0, 1)

    data = pg.gaussianFilter(np.random.normal(size=(256, 256, 3)), (20, 20, 0))
    for i in range(32):
        for j in range(32):
            data[i*8, j*8] += .1
    img = pg.ImageItem(data)
    vb.addItem(img)
    vb.autoRange()

    w.setImageItem(img)
    
    QtWidgets.QApplication.processEvents()
    win.close()
