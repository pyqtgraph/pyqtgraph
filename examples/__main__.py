import sys, os
if __name__ == "__main__" and (__package__ is None or __package__==''):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    import examples
    __package__ = "examples"
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from .ExampleApp import ExampleLoader


def run():
    app = pg.mkQApp()
    loader = ExampleLoader()
    app.exec_()

if __name__ == '__main__':
    run()
