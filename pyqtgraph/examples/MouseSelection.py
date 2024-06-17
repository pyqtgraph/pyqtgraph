"""
Demonstrates selecting plot curves by mouse click
"""

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
pg.setConfigOptions(useOpenGL=True, enableExperimental=True)

pg.mkQApp()
plt = pg.PlotWidget()
plt.setWindowTitle('pyqtgraph example: Plot data selection')
# shift plot area by adding title to test that the OpenGL code handles it
plt.setTitle('try clicking the curves')
plt.show()

x = np.arange(1000, dtype=np.float32)
x[[250, 500, 750]] = np.nan
connect = np.ones(1000, dtype=bool)
connect[3::4] = False
curves = [
    pg.PlotCurveItem(y=np.sin(np.linspace(0, 20, 1000)), pen='r', clickable=True, connect="pairs"),
    pg.PlotCurveItem(x=x, y=np.sin(np.linspace(1, 21, 1000)), pen='g', clickable=True, connect="finite"),
    pg.PlotCurveItem(x=x, y=np.sin(np.linspace(2, 22, 1000)), pen='b', clickable=True, connect="all"),
    pg.PlotCurveItem(y=np.sin(np.linspace(3, 23, 1000)), pen='y', clickable=True, connect=connect),
]
              
def plotClicked(curve):
    for i, c in enumerate(curves):
        width = 3 if c is curve else 1
        c.setPen('rgby'[i], width=width)
    
for c in curves:
    plt.addItem(c)
    c.sigClicked.connect(plotClicked)

# reparent the PlotWidget to test that the OpenGL code is able to handle it
win = QtWidgets.QMainWindow()
win.setCentralWidget(plt)
win.show()

if __name__ == '__main__':
    pg.exec()
