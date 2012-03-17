# -*- coding: utf-8 -*-
## Add path to library (just for examples; you do not need this)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg
import pyqtgraph.ptime as ptime

app = QtGui.QApplication([])

## Create window with GraphicsView widget
view = pg.GraphicsView()
view.show()  ## show view alone in its own window

## Allow mouse scale/pan. Normally we use a ViewBox for this, but
## for simple examples this is easier.
view.enableMouse()

## lock the aspect ratio so pixels are always square
view.setAspectLocked(True)

## Create image item
img = pg.ImageItem(border='w')
view.scene().addItem(img)

## Set initial view bounds
view.setRange(QtCore.QRectF(0, 0, 600, 600))

## Create random image
data = np.random.normal(size=(15, 600, 600), loc=1024, scale=64).astype(np.uint16)
i = 0

updateTime = ptime.time()
fps = 0

def updateData():
    global img, data, i, updateTime, fps

    ## Display the data
    img.setImage(data[i])
    i = (i+1) % data.shape[0]

    QtCore.QTimer.singleShot(1, updateData)
    now = ptime.time()
    fps2 = 1.0 / (now-updateTime)
    updateTime = now
    fps = fps * 0.9 + fps2 * 0.1
    
    #print "%0.1f fps" % fps
    

updateData()

## Start Qt event loop unless running in interactive mode.
if sys.flags.interactive != 1:
    app.exec_()
