import initExample ## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

from examples.ExampleApp import ExampleLoader

loader = ExampleLoader()

if __name__ == '__main__':
    pg.mkQApp().exec_()
