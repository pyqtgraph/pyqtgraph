# -*- coding: utf-8 -*-
import sys, os
## Add path to library (just for examples; you do not need this)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
import numpy as np

app = QtGui.QApplication([])
mw = QtGui.QMainWindow()
cw = pg.PlotWidget()
mw.setCentralWidget(cw)
mw.show()

s1 = pg.ScatterPlotItem(size=10, pen=QtGui.QPen(QtCore.Qt.NoPen), brush=QtGui.QBrush(QtGui.QColor(255, 255, 255, 20)))
pos = np.random.normal(size=(2,3000))
spots = [{'pos': pos[:,i]} for i in range(3000)]
s1.addPoints(spots)

cw.addDataItem(s1)

## Start Qt event loop unless running in interactive mode.
if sys.flags.interactive != 1:
    app.exec_()

