import sys, os
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui



if __name__ == '__main__':
    if __package__ is None or __package__ == "":
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, parent_dir)
        import examples
        __package__ = "examples"

    from .ExampleApp import main as run
    run()
