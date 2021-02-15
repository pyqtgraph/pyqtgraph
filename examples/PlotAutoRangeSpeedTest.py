#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
from time import perf_counter

import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

app = pg.mkQApp("Autorange Speed Test")

p = pg.plot()
p.setWindowTitle('pyqtgraph example: Autorange speed test')
p.setRange(QtCore.QRectF(0, -10, 5000, 20))
p.setLabel('bottom', 'Index', units='B')

# Enable Autorange and show all axis, but `right` and `top` without values
for ax in ["right", "top"]:
    axis = p.plotItem.getAxis(ax)
    axis.setStyle(showValues=False)
    axis.show()
p.plotItem.vb.enableAutoRange(enable=True, x=True, y=True)
curve = p.plot()
data = np.random.normal(size=(50, 5000))
ptr = 0
lastTime = perf_counter()
fps = None


def update():
    global curve, data, ptr, p, lastTime, fps
    curve.setData(data[ptr % 10])
    ptr += 1
    now = perf_counter()
    dt = now - lastTime
    lastTime = now
    if fps is None:
        fps = 1.0 / dt
    else:
        s = np.clip(dt * 3., 0, 1)
        fps = fps * (1 - s) + (1.0 / dt) * s
    p.setTitle('%0.2f fps' % fps)
    app.processEvents()  ## force complete redraw for every plot


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
