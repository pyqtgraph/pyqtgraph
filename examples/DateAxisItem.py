"""
Demonstrates the usage of DateAxisItem to display properly-formatted 
timestamps on x-axis which automatically adapt to current zoom level.

"""
import initExample ## Add path to library (just for examples; you do not need this)

import time
from datetime import datetime, timedelta

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

app = QtGui.QApplication([])

# Create a plot with a date-time axis
bax = pg.DateAxisItem()
bax.setLabel('Usage of DateAxisItem')
bax.setStyle(nudge=5.45)
w = pg.PlotWidget(axisItems={'bottom': bax})
w.showGrid(x=True, y=True)

# Plot sin(1/x^2) with timestamps in the last 100 years
now = time.time()
x = np.linspace(2*np.pi, 1000*2*np.pi, 8301)
w.plot(now-(2*np.pi/x)**2*100*np.pi*1e7, np.sin(x), symbol='o')

w.setWindowTitle('pyqtgraph example: DateAxisItem')
w.show()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        app.exec_()
