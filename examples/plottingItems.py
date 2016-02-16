# -*- coding: utf-8 -*-
"""
This example demonstrates some of the plotting items available in pyqtgraph.
"""

import initExample ## Add path to library (just for examples; you do not need this)
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg


app = QtGui.QApplication([])
win = pg.GraphicsWindow(title="Plotting items examples")
win.resize(1000,600)

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

p1 = win.addPlot(title="Plot Items example", y=np.random.normal(size=100))
inf1 = pg.InfiniteLine(movable=True, angle=90, label=True, textPosition=[0.5, 0.2], textColor=(200,200,100), textFill=(200,200,200,50))
inf2 = pg.InfiniteLine(movable=True, angle=0, label=True, pen=(0, 0, 200), textColor=(200,0,0), bounds = [-2, 2], suffix="mm", hoverPen=(0,200,0))
inf3 = pg.InfiniteLine(movable=True, angle=45)
inf1.setPos([2,2])
inf1.setTextLocation(position=0.75)
inf2.setTextLocation(shift=0.8)
p1.addItem(inf1)
p1.addItem(inf2)
p1.addItem(inf3)
lr = pg.LinearRegionItem(values=[0, 10])
p1.addItem(lr)

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
