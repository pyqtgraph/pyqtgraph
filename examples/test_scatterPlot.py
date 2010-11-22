# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from PyQt4 import QtGui, QtCore
from pyqtgraph.PlotWidget import *
from pyqtgraph.graphicsItems import *


app = QtGui.QApplication([])
mw = QtGui.QMainWindow()
cw = PlotWidget()
mw.setCentralWidget(cw)
mw.show()


#s1 = SpotItem(5, pxMode=True, brush=QtGui.QBrush(QtGui.QColor(0, 0, 200)), pen=QtGui.QPen(QtGui.QColor(100,100,100)))
#s1.setPos(1, 0)
#s2 = SpotItem(.1, pxMode=False, brush=QtGui.QBrush(QtGui.QColor(0, 200, 0)), pen=QtGui.QPen(QtGui.QColor(100,100,100)))
#s2.setPos(0, 1)
#cw.addItem(s1)
#cw.addItem(s2)

import numpy as np
s1 = ScatterPlotItem(size=10, pen=QtGui.QPen(QtCore.Qt.NoPen), brush=QtGui.QBrush(QtGui.QColor(255, 255, 255, 20)))
pos = np.random.normal(size=(2,3000))
spots = [{'pos': pos[:,i]} for i in range(3000)]
s1.addPoints(spots)

cw.addDataItem(s1)


