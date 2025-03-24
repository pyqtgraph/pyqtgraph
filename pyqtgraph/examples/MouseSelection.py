"""
Demonstrates selecting plot curves by mouse click
"""

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
pg.setConfigOptions(useOpenGL=True)

app = pg.mkQApp("MouseSelection Example")
plt = pg.PlotWidget()
plt.setWindowTitle('pyqtgraph example: Plot data selection')
# shift plot area by adding title to test that the OpenGL code handles it
plt.setTitle('try clicking the curves')
plt.show()

xint = np.arange(0, 1000) + 100_000_000
xfloat = np.arange(0, 1000) + 1e8
xfloat[(250, 500, 750),] = np.nan
connect = np.ones(1000, dtype=bool)
connect[3::4] = False
curves = [
    pg.PlotCurveItem(x=xint, y=np.sin(np.linspace(0, 20, 1000)), pen='r', clickable=True, connect="pairs"),
    pg.PlotCurveItem(x=xfloat, y=np.sin(np.linspace(1, 21, 1000)), pen='g', clickable=True, connect="finite", fillLevel=0, brush=(50,50,200,100)),
    pg.PlotCurveItem(x=xfloat, y=np.sin(np.linspace(2, 22, 1000)), pen='c', clickable=True, connect="all"),
    pg.PlotCurveItem(x=xint, y=np.sin(np.linspace(3, 23, 1000)), pen='y', clickable=True, connect=connect),
]
              
def plotClicked(curve):
    for i, c in enumerate(curves):
        width = 1
        color = pg.mkColor('rgcy'[i])
        if c is curve:
            width = 4
            color = color.darker()
        c.setPen(color, width=width)
    
for c in curves:
    plt.addItem(c)
    c.sigClicked.connect(plotClicked)

# force a render followed by a reparent of the PlotWidget
# to test that the OpenGL code is able to handle it
app.processEvents()
win = QtWidgets.QMainWindow()
win.setCentralWidget(plt)
win.show()

if __name__ == '__main__':
    pg.exec()
