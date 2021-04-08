# -*- coding: utf-8 -*-
"""
Use a HistogramLUTWidget to control the contrast / coloration of an image.
"""

## Add path to library (just for examples; you do not need this)                                                                           
import initExample

import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg


app = pg.mkQApp("Histogram Lookup Table Example")
win = QtGui.QMainWindow()
win.resize(800,600)
win.show()
win.setWindowTitle('pyqtgraph example: Histogram LUT')

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

w = pg.HistogramLUTWidget()
l.addWidget(w, 0, 1)

monoRadio = QtGui.QRadioButton('mono')
rgbaRadio = QtGui.QRadioButton('rgba')
l.addWidget(monoRadio, 1, 1)
l.addWidget(rgbaRadio, 2, 1)
monoRadio.setChecked(True)

def setLevelMode():
    mode = 'mono' if monoRadio.isChecked() else 'rgba'
    w.setLevelMode(mode)
monoRadio.toggled.connect(setLevelMode)

data = pg.gaussianFilter(np.random.normal(size=(256, 256, 3)), (20, 20, 0))
for i in range(32):
    for j in range(32):
        data[i*8, j*8] += .1
img = pg.ImageItem(data)
vb.addItem(img)
vb.autoRange()

w.setImageItem(img)

if __name__ == '__main__':
    pg.mkQApp().exec_()
