import sys, os

# Set up path to contain pyqtgraph module when run without installation
if __name__ == "__main__" and (__package__ is None or __package__==''):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    import examples
    __package__ = "examples"

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui


if __name__ == '__main__':
    from .ExampleApp import main as run
    run()
