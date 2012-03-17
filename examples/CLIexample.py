import initExample ## Add path to library (just for examples; you do not need this)

from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

app = QtGui.QApplication([])


data = np.random.normal(size=1000)
pg.plot(data, title="Simplest possible plotting example")

data = np.random.normal(size=(500,500))
pg.show(data, title="Simplest possible image example")


## Start Qt event loop unless running in interactive mode or using pyside.
import sys
if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
    app.exec_()
