#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
For testing rapid updates of ScatterPlotItem under various conditions.

(Scatter plots are still rather slow to draw; expect about 20fps)
"""



## Add path to library (just for examples; you do not need this)
import initExample


from pyqtgraph.Qt import QtGui, QtCore, QT_LIB
import numpy as np
import pyqtgraph as pg
from pyqtgraph.ptime import time
#QtGui.QApplication.setGraphicsSystem('raster')
app = QtGui.QApplication([])
#mw = QtGui.QMainWindow()
#mw.resize(800,800)
import importlib
ui_template = importlib.import_module(
    f'ScatterPlotSpeedTestTemplate_{QT_LIB.lower()}')

win = QtGui.QWidget()
win.setWindowTitle('pyqtgraph example: ScatterPlotSpeedTest')
ui = ui_template.Ui_Form()
ui.setupUi(win)
win.show()

p = ui.plot
p.setRange(xRange=[-500, 500], yRange=[-500, 500])

count = 500
data = np.random.normal(size=(50,count), scale=100)
sizeArray = (np.random.random(count) * 20.).astype(int)
brushArray = [pg.mkBrush(x) for x in np.random.randint(0, 256, (count, 3))]
ptr = 0
lastTime = time()
fps = None


def update():
    global curve, data, ptr, p, lastTime, fps
    p.clear()
    if ui.randCheck.isChecked():
        size = sizeArray
        brush = brushArray
    else:
        size = ui.sizeSpin.value()
        brush = 'b'
    curve = pg.ScatterPlotItem(x=data[ptr % 50], y=data[(ptr+1) % 50],
                               pen='w', brush=brush, size=size,
                               pxMode=ui.pixelModeCheck.isChecked())
    p.addItem(curve)
    ptr += 1
    now = time()
    dt = now - lastTime
    lastTime = now
    if fps is None:
        fps = 1.0/dt
    else:
        s = np.clip(dt*3., 0, 1)
        fps = fps * (1-s) + (1.0/dt) * s
    p.setTitle('%0.2f fps' % fps)
    p.repaint()
    #app.processEvents()  ## force complete redraw for every plot
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)
    


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
