# -*- coding: utf-8 -*-

## This example demonstrates the ability to link the axes of views together
## Views can be linked manually using the context menu, but only if they are given names.


import initExample ## Add path to library (just for examples; you do not need this)


from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

#QtGui.QApplication.setGraphicsSystem('raster')
app = QtGui.QApplication([])
#mw = QtGui.QMainWindow()
#mw.resize(800,800)

x = np.linspace(-50, 50, 1000)
y = np.sin(x) / x

win = pg.GraphicsWindow(title="View Linking Examples")
win.resize(800,600)

win.addLabel("Views linked at runtime:", colspan=2)
win.nextRow()

p1 = win.addPlot(x=x, y=y, name="Plot1", title="Plot1")
p2 = win.addPlot(x=x, y=y, name="Plot2", title="Plot2 - Y linked with Plot1")
p2.setLabel('bottom', "Label to test offset")
p2.setYLink(p1)

win.nextRow()

p3 = win.addPlot(x=x, y=y, name="Plot3", title="Plot3 - X linked with Plot1")
p4 = win.addPlot(x=x, y=y, name="Plot4", title="Plot4 - X and Y linked with Plot1")
p3.setLabel('left', "Label to test offset")
QtGui.QApplication.processEvents()
p3.setXLink(p1)
p4.setXLink(p1)
p4.setYLink(p1)


## Start Qt event loop unless running in interactive mode or using pyside.
import sys
if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    app.exec_()

