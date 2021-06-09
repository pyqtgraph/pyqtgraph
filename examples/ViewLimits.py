import initExample ## Add path to library (just for examples; you do not need this)

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np

plt = pg.plot(np.random.normal(size=100), title="View limit example")
plt.centralWidget.vb.setLimits(xMin=-20, xMax=120, minXRange=5, maxXRange=100)

if __name__ == '__main__':
    pg.exec()
