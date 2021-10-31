"""
Demonstrates very basic use of ImageItem to display image data inside a ViewBox.
"""

from time import perf_counter

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

app = pg.mkQApp("ImageItem Example")

## Create window with GraphicsView widget
win = pg.GraphicsLayoutWidget()
win.show()  ## show widget alone in its own window
win.setWindowTitle('pyqtgraph example: ImageItem')
view = win.addViewBox()

## lock the aspect ratio so pixels are always square
view.setAspectLocked(True)

## Create image item
img = pg.ImageItem(border='w')
view.addItem(img)

## Set initial view bounds
view.setRange(QtCore.QRectF(0, 0, 600, 600))

## Create random image
data = np.random.normal(size=(15, 600, 600), loc=1024, scale=64).astype(np.uint16)
i = 0

updateTime = perf_counter()
elapsed = 0

timer = QtCore.QTimer()
timer.setSingleShot(True)
# not using QTimer.singleShot() because of persistence on PyQt. see PR #1605

def updateData():
    global img, data, i, updateTime, elapsed

    ## Display the data
    img.setImage(data[i])
    i = (i+1) % data.shape[0]

    timer.start(1)
    now = perf_counter()
    elapsed_now = now - updateTime
    updateTime = now
    elapsed = elapsed * 0.9 + elapsed_now * 0.1

    # print(f"{1 / elapsed:.1f} fps")
    
timer.timeout.connect(updateData)
updateData()

if __name__ == '__main__':
    pg.exec()
