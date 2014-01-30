import initExample ## Add path to library (just for examples; you do not need this)

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np

plt = pg.plot(np.random.normal(size=100), title="View limit example")
plt.centralWidget.vb.setLimits(xRange=[-100, 100], minRange=[0.1, None], maxRange=[50, None])


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
        pg.QtGui.QApplication.exec_()
