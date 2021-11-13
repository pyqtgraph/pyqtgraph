
"""
This example demonstrates the different auto-ranging capabilities of ViewBoxes
"""

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

app = pg.mkQApp("Plot Auto Range Example")
#mw = QtWidgets.QMainWindow()
#mw.resize(800,800)

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
def update():
    t = pg.time()
    
    data = np.ones(100) * np.sin(t)
    data[50:60] += np.sin(t)
    global curve
    curve.setData(data)
    
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)

if __name__ == '__main__':
    pg.exec()
