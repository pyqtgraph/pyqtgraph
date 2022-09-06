
"""
This example demonstrates the different auto-ranging capabilities of ViewBoxes
"""

import time

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

app = pg.mkQApp("Plot Auto Range Example")

win = pg.GraphicsLayoutWidget(show=True, title="Plot auto-range examples")
win.resize(800,600)
win.setWindowTitle('pyqtgraph example: PlotAutoRange')

d = np.random.normal(size=100)
d[50:54] += 10
p1 = win.addPlot(title="95th percentile range", y=d)
p1.enableAutoRange('y', 0.95)


p2 = win.addPlot(title="Auto Pan Only")
p2.setAutoPan(y=True)
curve = p2.plot()
t0 = time.time()

def update():
    t = time.time() - t0
    
    data = np.ones(100) * np.sin(t)
    data[50:60] += np.sin(t)
    curve.setData(data)
    
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)

if __name__ == '__main__':
    pg.exec()
