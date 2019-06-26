"""
HistogramLUTWidget test:

Tests the creation of a HistogramLUTWidget.
"""

import pyqtgraph as pg
from ..Qt import QtGui
import numpy as np

def testHistogramLUTWidget():
    pg.mkQApp()
    
    win = QtGui.QMainWindow()
    win.show()

    cw = QtGui.QWidget()
    win.setCentralWidget(cw)

    l = QtGui.QGridLayout()
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
    
    QtGui.QApplication.processEvents()
    pg.exit()
    
