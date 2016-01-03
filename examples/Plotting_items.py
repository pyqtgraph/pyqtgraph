# -*- coding: utf-8 -*-
"""
This example demonstrates some of the new plotting items.
"""

import initExample ## Add path to library (just for examples; you do not need this)


from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

#QtGui.QApplication.setGraphicsSystem('raster')
app = QtGui.QApplication([])
#mw = QtGui.QMainWindow()
#mw.resize(800,800)

win = pg.GraphicsWindow(title="Basic plotting with some added items")
win.resize(1000,600)
win.setWindowTitle('pyqtgraph example: Plotting with items')

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

p1 = win.addPlot(title="Basic array plotting", y=np.random.normal(size=100))
infiniteLine = pg.InfiniteLine(label=True)
linearRegion = pg.LinearRegionItem(values = [30,70], labels=True, unit=u"°C")
crossHair = pg.CrossHair(textBorder=(200,200,200), bounds=((-2,1), (10,80)))
crossHair.setUnits(x="mm", y="°C")
crossHair.setFormat(y="{:.4f}")
p1.addItem(infiniteLine)
p1.addItem(linearRegion)
p1.addItem(crossHair)
## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
