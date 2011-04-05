# -*- coding: utf-8 -*-
## Add path to library (just for examples; you do not need this)
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from PyQt4 import QtCore, QtGui
import numpy as np
import pyqtgraph as pg

app = QtGui.QApplication([])

## Create window with GraphicsView widget
win = QtGui.QMainWindow()
view = pg.GraphicsView()
#view.useOpenGL(True)
win.setCentralWidget(view)
win.show()

## Allow mouse scale/pan
view.enableMouse()

## ..But lock the aspect ratio
view.setAspectLocked(True)

## Create image item
img = pg.ImageItem()
view.scene().addItem(img)

## Set initial view bounds
view.setRange(QtCore.QRectF(0, 0, 200, 200))

## Create random image
data = np.random.normal(size=(50, 200, 200))
i = 0

def updateData():
    global img, data, i

    ## Display the data
    img.updateImage(data[i])
    i = (i+1) % data.shape[0]
    

# update image data every 20ms (or so)
t = QtCore.QTimer()
t.timeout.connect(updateData)
t.start(20)

## Start Qt event loop unless running in interactive mode.
if sys.flags.interactive != 1:
    app.exec_()
