#!/usr/bin/python
# -*- coding: utf-8 -*-
## Add path to library (just for examples; you do not need this)
import initExample


from scipy import random
from numpy import linspace
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph import MultiPlotWidget
try:
    from pyqtgraph.metaarray import *
except:
    print("MultiPlot is only used with MetaArray for now (and you do not have the metaarray package)")
    exit()

app = QtGui.QApplication([])
mw = QtGui.QMainWindow()
mw.resize(800,800)
pw = MultiPlotWidget()
mw.setCentralWidget(pw)
mw.show()

data = random.normal(size=(3, 1000)) * np.array([[0.1], [1e-5], [1]])
ma = MetaArray(data, info=[
    {'name': 'Signal', 'cols': [
        {'name': 'Col1', 'units': 'V'}, 
        {'name': 'Col2', 'units': 'A'}, 
        {'name': 'Col3'},
        ]}, 
    {'name': 'Time', 'values': linspace(0., 1., 1000), 'units': 's'}
    ])
pw.plot(ma)

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

