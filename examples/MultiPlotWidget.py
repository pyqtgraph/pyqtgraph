#!/usr/bin/python
# -*- coding: utf-8 -*-
## Add path to library (just for examples; you do not need this)
import initExample

import numpy as np
from numpy import linspace
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph import MultiPlotWidget
try:
    from pyqtgraph.metaarray import *
except:
    print("MultiPlot is only used with MetaArray for now (and you do not have the metaarray package)")
    exit()

app = pg.mkQApp("MultiPlot Widget Example")
mw = QtGui.QMainWindow()
mw.resize(800,800)
pw = MultiPlotWidget()
mw.setCentralWidget(pw)
mw.show()

data = np.random.normal(size=(3, 1000)) * np.array([[0.1], [1e-5], [1]])
ma = MetaArray(data, info=[
    {'name': 'Signal', 'cols': [
        {'name': 'Col1', 'units': 'V'}, 
        {'name': 'Col2', 'units': 'A'}, 
        {'name': 'Col3'},
        ]}, 
    {'name': 'Time', 'values': linspace(0., 1., 1000), 'units': 's'}
    ])
pw.plot(ma, pen='y')

if __name__ == '__main__':
    pg.Qt.QtWidgets.QApplication.exec_()

