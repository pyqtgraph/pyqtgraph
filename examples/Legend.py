# -*- coding: utf-8 -*-
"""
Demonstrates basic use of LegendItem

"""
import initExample ## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

win = pg.plot()
win.setWindowTitle('pyqtgraph example: BarGraphItem')

# # option1: only for .plot(), following c1,c2 for example-----------------------
# win.addLegend(frame=False, colCount=2)

# bar graph
x = np.arange(10)
y = np.sin(x+2) * 3
bg1 = pg.BarGraphItem(x=x, height=y, width=0.3, brush='b', pen='w', name='bar')
win.addItem(bg1)

# curve
c1 = win.plot([np.random.randint(0,8) for i in range(10)], pen='r', symbol='t', symbolPen='r', symbolBrush='g', name='curve1')
c2 = win.plot([2,1,4,3,1,3,2,4,3,2], pen='g', fillLevel=0, fillBrush=(255,255,255,30), name='curve2')

# scatter plot
s1 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120), name='scatter')
spots = [{'pos': [i, np.random.randint(-3, 3)], 'data': 1} for i in range(10)]
s1.addPoints(spots)
win.addItem(s1)

# # option2: generic method------------------------------------------------
legend = pg.LegendItem((80,60), offset=(70,20))
legend.setParentItem(win.graphicsItem())
legend.addItem(bg1, 'bar')
legend.addItem(c1, 'curve1')
legend.addItem(c2, 'curve2')
legend.addItem(s1, 'scatter')


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
