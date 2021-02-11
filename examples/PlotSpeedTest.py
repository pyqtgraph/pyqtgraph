#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Update a simple plot as rapidly as possible to measure speed.
"""

## Add path to library (just for examples; you do not need this)
from time import perf_counter

import initExample

from pyqtgraph.Qt import QtGui, QtCore

import numpy as np
import pyqtgraph as pg

app = pg.mkQApp("Plot Speed Test")

p = pg.plot()
p.setWindowTitle('pyqtgraph example: PlotSpeedTest')
p.setRange(QtCore.QRectF(0, -10, 5000, 20))
p.setLabel('bottom', 'Index', units='B')
curve = p.plot()

data = np.random.normal(size=(50, 5000))
ptr = 0
elapsed = 0


def update():
    global curve, data, ptr, elapsed, ptr

    ptr += 1
    # Empty the eventloop stack before!
    app.processEvents(QtCore.QEventLoop.AllEvents, 20)

    # Measure
    t_start = perf_counter()
    curve.setData(data[ptr % 10])
    app.processEvents(QtCore.QEventLoop.AllEvents)

    # Statistics
    elapsed += perf_counter() - t_start
    average = elapsed / ptr
    fps = 1 / average
    p.setTitle('%0.2f fps - %0.5fs avg' % (fps, average))


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
