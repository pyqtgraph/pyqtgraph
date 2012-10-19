# -*- coding: utf-8 -*-
import initExample ## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

plt = pg.plot()

l = pg.LegendItem((100,60), (60,10))  # args are (size, position)
l.setParentItem(plt.graphicsItem())   # Note we do NOT call plt.addItem in this case

c1 = plt.plot([1,3,2,4], pen='r')
c2 = plt.plot([2,1,4,3], pen='g')
l.addItem(c1, 'red plot')
l.addItem(c2, 'green plot')


## Start Qt event loop unless running in interactive mode or using pyside.
import sys
if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    QtGui.QApplication.instance().exec_()
