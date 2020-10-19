import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

from .ExampleApp import ExampleLoader

loader = ExampleLoader()

if __name__ == "__main__":
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()