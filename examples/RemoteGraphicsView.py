# -*- coding: utf-8 -*-
"""
Very simple example demonstrating RemoteGraphicsView
"""
import initExample ## Add path to library (just for examples; you do not need this)
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph.widgets.RemoteGraphicsView import RemoteGraphicsView
app = pg.mkQApp()

v = RemoteGraphicsView()
v.show()
v.setWindowTitle('pyqtgraph example: RemoteGraphicsView')

## v.pg is a proxy to the remote process' pyqtgraph module. All attribute 
## requests and function calls made with this object are forwarded to the
## remote process and executed there.
plt = v.pg.PlotItem()
v.setCentralItem(plt)
plt.plot([1,4,2,3,6,2,3,4,2,3], pen='g')


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
