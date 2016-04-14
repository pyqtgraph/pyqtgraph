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

# Create a plot with some random data
p1 = win.addPlot(title="Plot Items example", y=np.random.normal(size=100, scale=10), pen=0.5)
p1.setYRange(-40, 40)

# Add three infinite lines with labels
inf1 = pg.InfiniteLine(movable=True, angle=90, label='x={value:0.2f}', 
                       labelOpts={'position':0.1, 'color': (200,200,100), 'fill': (200,200,200,50), 'movable': True})
inf2 = pg.InfiniteLine(movable=True, angle=0, pen=(0, 0, 200), bounds = [-20, 20], hoverPen=(0,200,0), label='y={value:0.2f}mm', 
                       labelOpts={'color': (200,0,0), 'movable': True, 'fill': (0, 0, 200, 100)})
inf3 = pg.InfiniteLine(movable=True, angle=45, pen='g', label='diagonal',
                       labelOpts={'rotateAxis': [1, 0], 'fill': (0, 200, 0, 100), 'movable': True})
inf1.setPos([2,2])
p1.addItem(inf1)
p1.addItem(inf2)
p1.addItem(inf3)

# Add a linear region with a label
lr = pg.LinearRegionItem(values=[70, 80])
p1.addItem(lr)
label = pg.InfLineLabel(lr.lines[1], "region 1", position=0.95, rotateAxis=(1,0), anchor=(1, 1))

# Add a cursor (without label)
cursor = pg.CursorItem(pos=[50, 20])
p1.addItem(cursor)

cursorLabel = pg.CursorItem(pos=[20,-20], radius=3, cursor='c', pen=(0, 0, 255), label='x={:.3f}\ny={:.3f}',
                            labelOpts={'color': (200, 200, 200), 'fill': (20, 20, 20, 20)})
p1.addItem(cursorLabel)




## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
