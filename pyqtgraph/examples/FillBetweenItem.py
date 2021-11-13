"""
Demonstrates use of FillBetweenItem to fill the space between two plot curves.
"""

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

#FIXME: When running on Qt5, not as perfect as on Qt4

win = pg.plot()
win.setWindowTitle('pyqtgraph example: FillBetweenItem')
win.setXRange(-10, 10)
win.setYRange(-10, 10)

N = 200
x = np.linspace(-10, 10, N)
gauss = np.exp(-x**2 / 20.)
mn = mx = np.zeros(len(x))
curves = [win.plot(x=x, y=np.zeros(len(x)), pen='k') for i in range(4)]
brushes = [0.5, (100, 100, 255), 0.5]
fills = [pg.FillBetweenItem(curves[i], curves[i+1], brushes[i]) for i in range(3)]
for f in fills:
    win.addItem(f)

def update():
    global mx, mn, curves, gauss, x
    a = 5 / abs(np.random.normal(loc=1, scale=0.2))
    y1 = -np.abs(a*gauss + np.random.normal(size=len(x)))
    y2 =  np.abs(a*gauss + np.random.normal(size=len(x)))
    
    s = 0.01
    mn = np.where(y1<mn, y1, mn) * (1-s) + y1 * s
    mx = np.where(y2>mx, y2, mx) * (1-s) + y2 * s
    curves[0].setData(x, mn)
    curves[1].setData(x, y1)
    curves[2].setData(x, y2)
    curves[3].setData(x, mx)
    

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(30)


if __name__ == '__main__':
    pg.exec()
