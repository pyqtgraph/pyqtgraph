# -*- coding: utf-8 -*-
## Add path to library (just for examples; you do not need this)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.opts['distance'] = 20
w.show()

g = gl.GLGridItem()
w.addItem(g)

#pos = np.empty((53, 3))
#size = np.empty((53))
#color = np.empty((53, 4))
#pos[0] = (1,0,0); size[0] = 0.5;   color[0] = (1.0, 0.0, 0.0, 0.5)
#pos[1] = (0,1,0); size[1] = 0.2;   color[1] = (0.0, 0.0, 1.0, 0.5)
#pos[2] = (0,0,1); size[2] = 2./3.; color[2] = (0.0, 1.0, 0.0, 0.5)

#z = 0.5
#d = 6.0
#for i in range(3,53):
    #pos[i] = (0,0,z)
    #size[i] = 2./d
    #color[i] = (0.0, 1.0, 0.0, 0.5)
    #z *= 0.5
    #d *= 2.0
    
#sp = gl.GLScatterPlotItem(pos=pos, sizes=size, colors=color, pxMode=False)


pos = (np.random.random(size=(100000,3)) * 10) - 5
color = np.ones((pos.shape[0], 4))
d = (pos**2).sum(axis=1)**0.5
color[:,3] = np.clip(-np.cos(d*2) * 0.2, 0, 1)
sp = gl.GLScatterPlotItem(pos=pos, color=color, size=5)
phase = 0.

def update():
    global phase, color, sp, d
    s = -np.cos(d*2+phase)
    color[:,3] = np.clip(s * 0.2, 0, 1)
    color[:,0] = np.clip(s * 3.0, 0, 1)
    color[:,1] = np.clip(s * 1.0, 0, 1)
    color[:,2] = np.clip(s ** 3, 0, 1)
    
    sp.setData(color=color)
    phase -= 0.1
    
t = QtCore.QTimer()
t.timeout.connect(update)
t.start(50)

w.addItem(sp)

## Start Qt event loop unless running in interactive mode.
if sys.flags.interactive != 1:
    app.exec_()
