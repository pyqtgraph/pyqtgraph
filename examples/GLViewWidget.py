# -*- coding: utf-8 -*-
"""
Very basic 3D graphics example; create a view widget and add a few items.

"""
## Add path to library (just for examples; you do not need this)
import initExample

from pyqtgraph.Qt import QtCore, QtGui, mkQApp
import pyqtgraph.opengl as gl

app = mkQApp("GLViewWidget Example")
w = gl.GLViewWidget()
w.opts['distance'] = 20
w.show()
w.setWindowTitle('pyqtgraph example: GLViewWidget')

ax = gl.GLAxisItem()
ax.setSize(5,5,5)
w.addItem(ax)

b = gl.GLBoxItem()
w.addItem(b)

ax2 = gl.GLAxisItem()
ax2.setParentItem(b)

b.translate(1,1,1)

if __name__ == '__main__':
    pg.mkQApp().exec_()
