#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Update a simple plot as rapidly as possible to measure speed.
"""

## Add path to library (just for examples; you do not need this)
import initExample

from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
from time import perf_counter

app = pg.mkQApp("Plot Speed Test")

p = pg.plot()
p.setWindowTitle('pyqtgraph example: PlotSpeedTest')
p.setRange(QtCore.QRectF(0, -10, 5000, 20))
p.setLabel('bottom', 'Index', units='B')
curve = p.plot()

data = np.random.normal(size=(50, 5000))
ptr = 0
elapsed = 0


def timeit(func):
    def wrapper(*args, **kwargs):
        global elapsed
        # Empty the eventloop stack before!
        app.processEvents(QtCore.QEventLoop.AllEvents, 20)
        t_start = perf_counter()
        ret = func(*args, **kwargs)
        # And process now
        app.processEvents(QtCore.QEventLoop.AllEvents)

        elapsed += perf_counter() - t_start
        average = elapsed / ptr
        print("{} average {} seconds".format(func.__name__, average))
        p.setTitle('%0.5f sec avg' % average)
        return ret

    return wrapper


@timeit
def update():
    global curve, data, ptr
    curve.setData(data[ptr % 10])
    ptr += 1


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
