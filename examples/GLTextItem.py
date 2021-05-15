# -*- coding: utf-8 -*-
"""
Simple examples demonstrating the use of GLTextItem.

"""

from pyqtgraph.Qt import QtCore, QtGui, mkQApp
import pyqtgraph.opengl as gl

app = mkQApp("GLTextItem Example")

gvw = gl.GLViewWidget()
gvw.show()
gvw.setWindowTitle('pyqtgraph example: GLTextItem')

txtitem1 = gl.GLTextItem(pos=(0.0, 0.0, 0.0), text='text1')
gvw.addItem(txtitem1)

txtitem2 = gl.GLTextItem()
txtitem2.setData(pos=(1.0, -1.0, 2.0), color=(0.5, 1.0, 0.5, 1.0), text='text2')
gvw.addItem(txtitem2)

if __name__ == '__main__':
  mkQApp().exec_()
