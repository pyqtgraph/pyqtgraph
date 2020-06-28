# -*- coding: utf-8 -*-
"""
Demonstrates very basic use of PColorMeshItem
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

# x and y being the coordinates of the polygons, they share the same shape
# However the shape can be different in both dimension
xn = 50 # nb points along x
yn = 40 # nb points along y
x = np.repeat(np.arange(1, xn+1), yn).reshape(xn, yn)
y = np.tile(np.arange(1, yn+1), xn).reshape(xn, yn)

# z being the color of the polygons its shape must be decreased by one in each dimension
z = np.exp(-(x*xn)**2/1000)[:-1,:-1]

## Create image item
pcmi = pg.PColorMeshItem()
view.addItem(pcmi)


## Set the animation
fps = 25 # Frame per second of the animation

# Wave parameters
wave_amplitude  = 3
wave_speed      = 0.3
wave_length     = 10
color_speed     = 0.3

i=0
def updateData():
    global i
    
    ## Display the new data set
    new_x = x
    new_y = y+wave_amplitude*np.cos(x/wave_length+i)
    new_z = np.exp(-(x-np.cos(i*color_speed)*xn)**2/1000)[:-1,:-1]
    pcmi.setData(new_x,
                 new_y,
                 new_z)

    i += wave_speed
    QtCore.QTimer.singleShot(1000/fps, updateData)

updateData()

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
