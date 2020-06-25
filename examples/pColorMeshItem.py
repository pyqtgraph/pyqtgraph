# -*- coding: utf-8 -*-
"""
Demonstrates very basic use of pColorMeshItem
"""

## Add path to library (just for examples; you do not need this)
import initExample

from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg
import pyqtgraph.ptime as ptime

app = QtGui.QApplication([])

## Create window with GraphicsView widget
win = pg.GraphicsLayoutWidget()
win.show()  ## show widget alone in its own window
win.setWindowTitle('pyqtgraph example: pColorMeshItem')
view = win.addViewBox()


## Create data
x = np.array([[1,1,1,1],
              [2,2,2,2],
              [3,3,3,3],
              [4,4,4,4],
              [5,5,5,5]])
y = np.array([[4,8,12,16],
              [2,4,6,8],
              [3,6,9,12],
              [5,10,15,20],
              [6,12,18,24]])
z = np.array([[1,2,3],
              [5,6,7],
              [9,10,11],
              [13,14,15]])

## Create image item
pcmi = pg.PColorMeshItem(x, y, z)
view.addItem(pcmi)



fps = 1
i = 0

def updateData():
    global pcmi, x, y, z, i

    ## Display the data
    pcmi.setData(x-i, y, z)

    QtCore.QTimer.singleShot(fps*1000, updateData)
    i += 1
    print(i)
    

updateData()

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
