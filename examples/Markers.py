# -*- coding: utf-8 -*-
"""
This example shows all the markers available into pyqtgraph.
"""

import initExample ## Add path to library (just for examples; you do not need this)
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
from pyqtgraph.graphicsItems.ScatterPlotItem import makeRegularPolygon

app = QtGui.QApplication([])
win = pg.GraphicsWindow(title="Pyqtgraph markers")
win.resize(1000, 600)

pg.setConfigOptions(antialias=True)

plot = win.addPlot(title="Plotting with markers")

data = np.array([0, 1, 2, 3, 4])
size = 20

plot.plot(data+1, pen=(0,0,200), symbolBrush=(0,0,200), symbolPen='w', symbol='o', symbolSize=size)
plot.plot(data+2, pen=(0,128,0), symbolBrush=(0,128,0), symbolPen='w', symbol='t', symbolSize=size)
plot.plot(data+3, pen=(19,234,201), symbolBrush=(19,234,201), symbolPen='w', symbol='t1', symbolSize=size)
plot.plot(data+4, pen=(195,46,212), symbolBrush=(195,46,212), symbolPen='w', symbol='t2', symbolSize=size)
plot.plot(data+5, pen=(250,194,5), symbolBrush=(250,194,5), symbolPen='w', symbol='t3', symbolSize=size)
plot.plot(data+6, pen=(54,55,55), symbolBrush=(55,55,55), symbolPen='w', symbol='s', symbolSize=size)
plot.plot(data+7, pen=(0,114,189), symbolBrush=(0,114,189), symbolPen='w', symbol='p', symbolSize=size)
plot.plot(data+8, pen=(217,83,25), symbolBrush=(217,83,25), symbolPen='w', symbol='h', symbolSize=size)
plot.plot(data+9, pen=(237,177,32), symbolBrush=(237,177,32), symbolPen='w', symbol='star', symbolSize=size)
plot.plot(data+10, pen=(126,47,142), symbolBrush=(126,47,142), symbolPen='w', symbol='+', symbolSize=size)
plot.plot(data+11, pen=(119,172,48), symbolBrush=(119,172,48), symbolPen='w', symbol='d', symbolSize=size)
plot.plot(data+12, pen=(119,172,48), symbolBrush=(119,172,48), symbolPen='w', symbol=makeRegularPolygon(3), symbolSize=size)
plot.plot(data+13, pen=(119,172,48), symbolBrush=(119,172,48), symbolPen='w', symbol=makeRegularPolygon(4), symbolSize=size)
plot.plot(data+14, pen=(119,172,48), symbolBrush=(119,172,48), symbolPen='w', symbol=makeRegularPolygon(5), symbolSize=size)
plot.plot(data+15, pen=(119,172,48), symbolBrush=(119,172,48), symbolPen='w', symbol=makeRegularPolygon(6), symbolSize=size)
plot.plot(data+16, pen=(119,172,48), symbolBrush=(119,172,48), symbolPen='w', symbol=makeRegularPolygon(7), symbolSize=size)
plot.plot(data+17, pen=(119,172,48), symbolBrush=(119,172,48), symbolPen='w', symbol=makeRegularPolygon(100), symbolSize=size)

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
