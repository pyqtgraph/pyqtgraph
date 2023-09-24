#!/usr/bin/python

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from utils import FrameCounter

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

def update():
    global ptr
    curve.setData(data[ptr % 10])
    ptr += 1
    framecnt.update()


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

framecnt = FrameCounter()
framecnt.sigFpsUpdate.connect(lambda fps: p.setTitle(f'{fps:.1f} fps'))

if __name__ == '__main__':
    pg.exec()
