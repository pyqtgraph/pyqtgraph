# -*- coding: utf-8 -*-
"""
Use GLTextItem to display text in GLViewWidget.
"""
## Add path to library (just for examples; you do not need this)
import initExample

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.opts['distance'] = 40
w.show()
w.setWindowTitle('pyqtgraph example: GLTextItem')

t = gl.GLTextItem(X=0, Y=5, Z=10, text="Hello World")
t.setGLViewWidget(w)
w.addItem(t)

g = gl.GLGridItem()
w.addItem(g)

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
