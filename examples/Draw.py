# -*- coding: utf-8 -*-
import initExample ## Add path to library (just for examples; you do not need this)


from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg

app = QtGui.QApplication([])

## Create window with GraphicsView widget
win = QtGui.QMainWindow()
win.resize(800,800)
view = pg.GraphicsView()
#view.useOpenGL(True)
win.setCentralWidget(view)
win.show()

## Allow mouse scale/pan
view.enableMouse()
## ..But lock the aspect ratio
view.setAspectLocked(True)

## Create image item
img = pg.ImageItem(np.zeros((200,200)))
view.scene().addItem(img)

## Set initial view bounds
view.setRange(QtCore.QRectF(0, 0, 200, 200))

## start drawing with 3x3 brush
kern = np.array([
    [0.0, 0.5, 0.0],
    [0.5, 1.0, 0.5],
    [0.0, 0.5, 0.0]
])
img.setDrawKernel(kern, mask=kern, center=(1,1), mode='add')
img.setLevels([0, 10])

## Start Qt event loop unless running in interactive mode or using pyside.
import sys
if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    app.exec_()
