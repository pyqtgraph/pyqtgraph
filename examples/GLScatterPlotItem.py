# -*- coding: utf-8 -*-
## Add path to library (just for examples; you do not need this)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.opts['distance'] = 20
w.show()

g = gl.GLGridItem()
w.addItem(g)

pts = [
    {'pos': (1,0,0), 'size':0.5, 'color':(1.0, 0.0, 0.0, 0.5)}, 
    {'pos': (0,1,0), 'size':0.2, 'color':(0.0, 0.0, 1.0, 0.5)}, 
    {'pos': (0,0,1), 'size':2./3., 'color':(0.0, 1.0, 0.0, 0.5)},
]
z = 0.5
d = 6.0
for i in range(50):
    pts.append({'pos': (0,0,z), 'size':2./d, 'color':(0.0, 1.0, 0.0, 0.5)})
    z *= 0.5
    d *= 2.0
sp = gl.GLScatterPlotItem(pts)
w.addItem(sp)

## Start Qt event loop unless running in interactive mode.
if sys.flags.interactive != 1:
    app.exec_()
