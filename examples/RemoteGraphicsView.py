# -*- coding: utf-8 -*-
import initExample ## Add path to library (just for examples; you do not need this)
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
app = pg.mkQApp()

v = pg.RemoteGraphicsView()
v.show()

plt = v.pg.PlotItem()
v.setCentralItem(plt)
plt.plot([1,4,2,3,6,2,3,4,2,3], pen='g')



## Start Qt event loop unless running in interactive mode or using pyside.
import sys
if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    QtGui.QApplication.instance().exec_()
