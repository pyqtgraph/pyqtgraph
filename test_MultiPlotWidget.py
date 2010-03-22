#!/usr/bin/python
# -*- coding: utf-8 -*-

from scipy import random
from numpy import linspace
from PyQt4 import QtGui, QtCore
from MultiPlotWidget import *
from metaarray import *

app = QtGui.QApplication([])
mw = QtGui.QMainWindow()
pw = MultiPlotWidget()
mw.setCentralWidget(pw)
mw.show()

ma = MetaArray(random.random((3, 1000)), info=[{'name': 'Signal', 'cols': [{'name': 'Col1'}, {'name': 'Col2'}, {'name': 'Col3'}]}, {'name': 'Time', 'vals': linspace(0., 1., 1000)}])
pw.plot(ma)
