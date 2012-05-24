# -*- coding: utf-8 -*-

## Display an animated arrowhead following a curve.
## This example uses the CurveArrow class, which is a combination
## of ArrowItem and CurvePoint. 
##
## To place a static arrow anywhere in a scene, use ArrowItem.
## To attach other types of item to a curve, use CurvePoint.

import initExample ## Add path to library (just for examples; you do not need this)

import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg


app = QtGui.QApplication([])
mw = QtGui.QMainWindow()
mw.resize(300,300)

p = pg.PlotWidget()
mw.setCentralWidget(p)
c = p.plot(x=np.sin(np.linspace(0, 2*np.pi, 1000)), y=np.cos(np.linspace(0, 6*np.pi, 1000)))
a = pg.CurveArrow(c)
p.addItem(a)

mw.show()

anim = a.makeAnimation(loop=-1)
anim.start()

## Start Qt event loop unless running in interactive mode or using pyside.
import sys
if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    app.exec_()
