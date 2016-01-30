#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Test the speed of rapidly updating multiple plot curves
"""

## Add path to library (just for examples; you do not need this)
import initExample

import cProfile, pstats, StringIO
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
from pyqtgraph.ptime import time

app = QtGui.QApplication([])

p = pg.plot()
p.setWindowTitle('pyqtgraph example: MultiPlotRedrawSpeedTest')
p.setLabel('bottom', 'Index', units='B')

nPlots = 10
nSamples = 500000
curves = []
for i in xrange(nPlots):
    c = pg.PlotCurveItem(pen=(i, nPlots*1.3))
    p.addItem(c)
    c.setPos(0, i*6)
    curves.append(c)

data = np.random.normal(size=(nPlots*23, nSamples))
for i in xrange(nPlots):
    curves[i].setData(data[i % data.shape[0]])

p.setYRange(0, nPlots*6)
p.setXRange(0, nSamples)
p.resize(600, 900)

lastTime = time()
fps = None
count = 0
pr = cProfile.Profile()
timer = QtCore.QTimer()


def update():
    global curve, p, lastTime, fps, nPlots, count, pr, timer
    count += 1

    p.viewTransformChanged()
    for i in xrange(nPlots):
        curves[i].viewTransformChanged()

    now = time()
    dt = now - lastTime
    lastTime = now
    if fps is None:
        fps = 1.0/dt
    else:
        s = np.clip(dt*3., 0, 1)
        fps = fps * (1-s) + (1.0/dt) * s
    p.setTitle('%0.2f fps' % fps)
    '''
    if count > 20:
        timer.stop()
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'tottime'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        open('pstats.txt', 'w').write(s.getvalue())
        exit()
    '''



#pr.enable()


timer.timeout.connect(update)
timer.start(0)


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
