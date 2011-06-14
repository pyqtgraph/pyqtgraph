# -*- coding: utf-8 -*-
## Add path to library (just for examples; you do not need this)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


import numpy as np
from PyQt4 import QtGui, QtCore
import pyqtgraph as pg


app = QtGui.QApplication([])
mw = QtGui.QMainWindow()
mw.resize(800,800)

p = pg.PlotWidget()
mw.setCentralWidget(p)
c = p.plot(x=np.sin(np.linspace(0, 2*np.pi, 100)), y=np.cos(np.linspace(0, 2*np.pi, 100)))
a = pg.CurveArrow(c)
p.addItem(a)

mw.show()

anim = a.makeAnimation(loop=-1)
anim.start()

## Start Qt event loop unless running in interactive mode.
if sys.flags.interactive != 1:
    app.exec_()
