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

# Create a plot with the date-time axis
w = pg.PlotWidget(axisItems = {'bottom': pg.DateAxisItem()})

# Plot some random data with timestamps in the last hour
now = time.time()
timestamps = np.linspace(now - 3600, now, 100)
w.plot(x=timestamps, y=np.random.rand(100), symbol='o')

w.show()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        app.exec_()
