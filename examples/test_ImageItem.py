# -*- coding: utf-8 -*-
## Add path to library (just for examples; you do not need this)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


from PyQt4 import QtCore, QtGui
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

    QtCore.QTimer.singleShot(20, updateData)
    

# update image data every 20ms (or so)
#t = QtCore.QTimer()
#t.timeout.connect(updateData)
#t.start(20)
updateData()


def doWork():
    while True:
        x = '.'.join(['%f'%i for i in range(100)])  ## some work for the thread to do
        if time is None:  ## main thread has started cleaning up, bail out now
            break
        time.sleep(1e-3)

import thread
thread.start_new_thread(doWork, ())


## Start Qt event loop unless running in interactive mode.
if sys.flags.interactive != 1:
    app.exec_()
