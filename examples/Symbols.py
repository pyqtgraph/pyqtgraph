# -*- coding: utf-8 -*-
"""
This example shows all the scatter plot symbols available in pyqtgraph.

These symbols are used to mark point locations for scatter plots and some line
plots, similar to "markers" in matplotlib and vispy.
"""

import initExample ## Add path to library (just for examples; you do not need this)
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

app = pg.mkQApp("Symbols Examples")
win = pg.GraphicsLayoutWidget(show=True, title="Scatter Plot Symbols")
win.resize(1000,600)

pg.setConfigOptions(antialias=True)

plot = win.addPlot(title="Plotting with symbols")
plot.addLegend()
plot.plot([0, 1, 2, 3, 4], pen=(0,0,200), symbolBrush=(0,0,200), symbolPen='w', symbol='o', symbolSize=14, name="symbol='o'")
plot.plot([1, 2, 3, 4, 5], pen=(0,128,0), symbolBrush=(0,128,0), symbolPen='w', symbol='t', symbolSize=14, name="symbol='t'")
plot.plot([2, 3, 4, 5, 6], pen=(19,234,201), symbolBrush=(19,234,201), symbolPen='w', symbol='t1', symbolSize=14, name="symbol='t1'")
plot.plot([3, 4, 5, 6, 7], pen=(195,46,212), symbolBrush=(195,46,212), symbolPen='w', symbol='t2', symbolSize=14, name="symbol='t2'")
plot.plot([4, 5, 6, 7, 8], pen=(250,194,5), symbolBrush=(250,194,5), symbolPen='w', symbol='t3', symbolSize=14, name="symbol='t3'")
plot.plot([5, 6, 7, 8, 9], pen=(54,55,55), symbolBrush=(55,55,55), symbolPen='w', symbol='s', symbolSize=14, name="symbol='s'")
plot.plot([6, 7, 8, 9, 10], pen=(0,114,189), symbolBrush=(0,114,189), symbolPen='w', symbol='p', symbolSize=14, name="symbol='p'")
plot.plot([7, 8, 9, 10, 11], pen=(217,83,25), symbolBrush=(217,83,25), symbolPen='w', symbol='h', symbolSize=14, name="symbol='h'")
plot.plot([8, 9, 10, 11, 12], pen=(237,177,32), symbolBrush=(237,177,32), symbolPen='w', symbol='star', symbolSize=14, name="symbol='star'")
plot.plot([9, 10, 11, 12, 13], pen=(126,47,142), symbolBrush=(126,47,142), symbolPen='w', symbol='+', symbolSize=14, name="symbol='+'")
plot.plot([10, 11, 12, 13, 14], pen=(119,172,48), symbolBrush=(119,172,48), symbolPen='w', symbol='d', symbolSize=14, name="symbol='d'")
plot.plot([11, 12, 13, 14, 15], pen=(253, 216, 53), symbolBrush=(253, 216, 53), symbolPen='w', symbol='arrow_down', symbolSize=22, name="symbol='arrow_down'")
plot.plot([12, 13, 14, 15, 16], pen=(189, 189, 189), symbolBrush=(189, 189, 189), symbolPen='w', symbol='arrow_left', symbolSize=22, name="symbol='arrow_left'")
plot.plot([13, 14, 15, 16, 17], pen=(187, 26, 95), symbolBrush=(187, 26, 95), symbolPen='w', symbol='arrow_up', symbolSize=22, name="symbol='arrow_up'")
plot.plot([14, 15, 16, 17, 18], pen=(248, 187, 208), symbolBrush=(248, 187, 208), symbolPen='w', symbol='arrow_right', symbolSize=22, name="symbol='arrow_right'")
plot.setXRange(-2, 4)

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
