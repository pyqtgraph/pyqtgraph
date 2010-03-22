#!/usr/bin/python
# -*- coding: utf-8 -*-
## Add path to library (just for examples; you do not need this)
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scipy import random
from numpy import linspace
from PyQt4 import QtGui, QtCore
from pyqtgraph.MultiPlotWidget import *
try:
    from metaarray import *
except:
    print "MultiPlot is only used with MetaArray for now (and you do not have the metaarray module)"
    exit()
    
app = QtGui.QApplication([])
mw = QtGui.QMainWindow()
pw = MultiPlotWidget()
mw.setCentralWidget(pw)
mw.show()

ma = MetaArray(random.random((3, 1000)), info=[{'name': 'Signal', 'cols': [{'name': 'Col1'}, {'name': 'Col2'}, {'name': 'Col3'}]}, {'name': 'Time', 'vals': linspace(0., 1., 1000)}])
pw.plot(ma)

app.exec_()