# -*- coding: utf-8 -*-
import initExample ## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph.console

app = pg.mkQApp()

## build an initial namespace for console commands to be executed in (this is optional;
## the user can always import these modules manually)
namespace = {'pg': pg, 'np': np}

## initial text to display in the console
text = """
This is an interactive python console. The numpy and pyqtgraph modules have already been imported 
as 'np' and 'pg'. 

Go, play.
"""
c = pyqtgraph.console.ConsoleWidget(namespace=namespace, text=text)
c.show()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
