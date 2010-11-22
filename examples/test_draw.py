# -*- coding: utf-8 -*-
## Add path to library (just for examples; you do not need this)
import sys, os
sys.path = [os.path.join(os.path.dirname(__file__), '..', '..')] + sys.path

from pyqtgraph.GraphicsView import *
from pyqtgraph.graphicsItems import *
#from numpy import random
from PyQt4 import QtCore, QtGui
from scipy.ndimage import *
import numpy as np

app = QtGui.QApplication([])

## Create window with GraphicsView widget
win = QtGui.QMainWindow()
view = GraphicsView()
#view.useOpenGL(True)
win.setCentralWidget(view)
win.show()

## Allow mouse scale/pan
view.enableMouse()

## ..But lock the aspect ratio
view.setAspectLocked(True)

## Create image item
img = ImageItem(np.zeros((200,200)))
view.scene().addItem(img)

## Set initial view bounds
view.setRange(QtCore.QRectF(0, 0, 200, 200))

img.setDrawKernel(1)
img.setLevels(10,0)

#app.exec_()
