import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

plt = pg.plot(np.random.normal(size=100), title="View limit example")
plt.centralWidget.vb.setLimits(xMin=-20, xMax=120, minXRange=5, maxXRange=100)

if __name__ == '__main__':
    pg.exec()
