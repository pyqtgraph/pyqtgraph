# -*- coding: utf-8 -*-
## Add path to library (just for examples; you do not need this)                                                                           
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from PyQt4 import QtGui, QtCore
import pyqtgraph as pg


app = QtGui.QApplication([])
mw = pg.GraphicsView()
mw.resize(800,800)
mw.show()

#ts = pg.TickSliderItem()
#mw.setCentralItem(ts)
#ts.addTick(0.5, 'r')
#ts.addTick(0.9, 'b')

ge = pg.GradientEditorItem()
mw.setCentralItem(ge)


## Start Qt event loop unless running in interactive mode.
if sys.flags.interactive != 1:
    app.exec_()
