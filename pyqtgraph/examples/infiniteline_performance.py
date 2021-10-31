#!/usr/bin/python

from time import perf_counter

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

app = pg.mkQApp("Infinite Line Performance")

p = pg.plot()
p.setWindowTitle('pyqtgraph performance: InfiniteLine')
p.setRange(QtCore.QRectF(0, -10, 5000, 20))
p.setLabel('bottom', 'Index', units='B')
curve = p.plot()

# Add a large number of horizontal InfiniteLine to plot
for i in range(100):
    line = pg.InfiniteLine(pos=np.random.randint(5000), movable=True)
    p.addItem(line)

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
        fps = 1.0/dt
    else:
        s = np.clip(dt*3., 0, 1)
        fps = fps * (1-s) + (1.0/dt) * s
    p.setTitle('%0.2f fps' % fps)
    app.processEvents()  # force complete redraw for every plot


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

if __name__ == '__main__':
    pg.exec()
