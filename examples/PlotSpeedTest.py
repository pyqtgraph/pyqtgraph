#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Update a simple plot as rapidly as possible to measure speed.
"""

## Add path to library (just for examples; you do not need this)
import initExample

from collections import deque
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
rollingAverageSize = 1000

elapsed = deque(maxlen=rollingAverageSize)

def update():
    global curve, data, ptr, elapsed, ptr

    ptr += 1
    # Measure
    t_start = perf_counter()
    curve.setData(data[ptr % 10])
    app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
    elapsed.append(perf_counter() - t_start)

    # update display every 50-updates
    if ptr % 50 == 0:
        average = np.mean(elapsed)
        fps = 1 / average
        p.setTitle('%0.2f fps - %0.1f ms avg' % (fps, average * 1_000))


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)
    
if __name__ == '__main__':
    pg.exec()
