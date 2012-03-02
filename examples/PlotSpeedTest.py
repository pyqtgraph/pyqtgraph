#!/usr/bin/python
# -*- coding: utf-8 -*-
## Add path to library (just for examples; you do not need this)
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

#QtGui.QApplication.setGraphicsSystem('raster')
app = QtGui.QApplication([])
#mw = QtGui.QMainWindow()
#mw.resize(800,800)

p = pg.plot()

curve = p.plot()
data = np.random.normal(size=(10,50000))
ptr = 0
lastTime = time.time()
fps = None
def update():
    global curve, data, ptr, p, lastTime, fps
    curve.setData(data[ptr%10])
    ptr += 1
    now = time.time()
    dt = now - lastTime
    lastTime = now
    if fps is None:
        fps = 1.0/dt
    else:
        s = np.clip(dt*3., 0, 1)
        fps = fps * (1-s) + (1.0/dt) * s
    p.setTitle('%0.2f fps' % fps)
    app.processEvents()  ## force complete redraw for every plot
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)
    


## Start Qt event loop unless running in interactive mode.
if sys.flags.interactive != 1:
    app.exec_()
