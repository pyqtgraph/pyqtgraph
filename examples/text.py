# -*- coding: utf-8 -*-
## This example shows how to insert text into a scene using QTextItem


import initExample ## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np


x = np.linspace(-100, 100, 1000)
y = np.sin(x) / x
plot = pg.plot(x, y)

## Create text object, use HTML tags to specify color (default is black; won't be visible)
text = pg.TextItem(html='<div style="text-align: center"><span style="color: #FFF;">This is the</span><br><span style="color: #FF0;">PEAK</span></div>')
plot.addItem(text)
text.setPos(0, y.max())


## Start Qt event loop unless running in interactive mode or using pyside.
import sys
if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    QtGui.QApplication.instance().exec_()
