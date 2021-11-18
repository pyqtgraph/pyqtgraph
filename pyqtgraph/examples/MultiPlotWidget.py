#!/usr/bin/python

import numpy as np
from numpy import linspace

import pyqtgraph as pg
from pyqtgraph import MultiPlotWidget
from pyqtgraph.Qt import QtWidgets

try:
    from pyqtgraph.metaarray import *
except:
    print("MultiPlot is only used with MetaArray for now (and you do not have the metaarray package)")
    exit()

app = pg.mkQApp("MultiPlot Widget Example")
mw = QtWidgets.QMainWindow()
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
    pg.exec()
