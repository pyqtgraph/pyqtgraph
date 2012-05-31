# -*- coding: utf-8 -*-
import initExample ## Add path to library (just for examples; you do not need this)

"""
Simple example demonstrating a button which displays a colored rectangle
and allows the user to select a new color by clicking on the button.
"""

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

app = QtGui.QApplication([])
win = QtGui.QMainWindow()
btn = pg.ColorButton()
win.setCentralWidget(btn)
win.show()

def change(btn):
    print("change", btn.color())
def done(btn):
    print("done", btn.color())

btn.sigColorChanging.connect(change)
btn.sigColorChanged.connect(done)

## Start Qt event loop unless running in interactive mode or using pyside.
import sys
if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    QtGui.QApplication.instance().exec_()
