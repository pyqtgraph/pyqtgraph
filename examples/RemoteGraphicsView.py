# -*- coding: utf-8 -*-
import initExample ## Add path to library (just for examples; you do not need this)
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
app = pg.mkQApp()

v = pg.RemoteGraphicsView()
v.show()

QtGui = v.pg.QtGui
rect = QtGui.QGraphicsRectItem(0,0,10,10)
rect.setPen(QtGui.QPen(QtGui.QColor(255,255,0)))
v.scene().addItem(rect)



## Start Qt event loop unless running in interactive mode or using pyside.
import sys
if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    QtGui.QApplication.instance().exec_()
