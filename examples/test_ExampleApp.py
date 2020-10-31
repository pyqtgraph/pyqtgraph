import initExample ## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

from examples.ExampleApp import ExampleLoader

loader = ExampleLoader()

if __name__ == "__main__":
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()