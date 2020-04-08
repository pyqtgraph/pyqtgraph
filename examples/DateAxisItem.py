import time
import sys
from datetime import datetime, timedelta

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

if __name__ == '__main__':
    app = QtGui.QApplication([])

    # Create a plot with the Date-time axis
    w = pg.PlotWidget(axisItems = {'bottom': pg.DateAxisItem()})

    # plot some random data with timestamps in the last hour
    now = time.time()
    timestamps = np.linspace(now - 3600, now, 100)
    w.plot(x=timestamps, y=np.random.rand(100), symbol='o')
    
    w.show()

    sys.exit(app.exec_())
