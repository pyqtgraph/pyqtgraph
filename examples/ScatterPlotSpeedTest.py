#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
For testing rapid updates of ScatterPlotItem under various conditions.

(Scatter plots are still rather slow to draw; expect about 20fps)
"""

# Add path to library (just for examples; you do not need this)
import initExample

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from pyqtgraph.ptime import time
import pyqtgraph.parametertree as ptree
import pyqtgraph.graphicsItems.ScatterPlotItem

app = pg.mkQApp()
pt = ptree.ParameterTree(showHeader=False)
param = ptree.Parameter.create(name='Parameters', type='group', children=[
    dict(name='_USE_QRECT', title='Use QRect:    ', type='bool', value=pyqtgraph.graphicsItems.ScatterPlotItem._USE_QRECT),
    dict(name='pxMode', title='pxMode:    ', type='bool', value=True),
    dict(name='useCache', title='useCache:    ', type='bool', value=True),
    dict(name='count', title='Count:    ', type='int', limits=[1, None], value=500),
    dict(name='size', title='Size:    ', type='int', limits=[1, None], value=10),
    dict(name='randomize', title='Randomize:    ', type='bool', value=False),
    dict(name='mkItem', title='Remake Item:    ', type='bool', value=False),
    dict(name='paused', title='Paused:    ', type='bool', value=False)
])

for c in param.children():
    c.setDefault(c.value())

pt.setParameters(param)
p = pg.PlotWidget()

sp = QtWidgets.QSplitter()
sp.addWidget(pt)
sp.addWidget(p)
sp.show()

data = {}
item = pg.ScatterPlotItem()
ptr = 0
lastTime = time()
fps = None

timer = QtCore.QTimer()
timer.start(0)


def mkData():
    global data, fps
    scale = 100
    data = {
        'pos': np.random.normal(size=(50, param['count']), scale=scale),
        'pen': [pg.mkPen(x) for x in np.random.randint(0, 256, (param['count'], 3))],
        'brush': [pg.mkBrush(x) for x in np.random.randint(0, 256, (param['count'], 3))],
        'size': (np.random.random(param['count']) * param['size']).astype(int)
    }
    data['pen'][0] = pg.mkPen('w')
    data['size'][0] = param['size']
    data['brush'][0] = pg.mkBrush('b')
    bound = 5 * scale
    p.setRange(xRange=[-bound, bound], yRange=[-bound, bound])


def mkItem():
    global item
    pyqtgraph.graphicsItems.ScatterPlotItem._USE_QRECT = param['_USE_QRECT']
    item = pg.ScatterPlotItem(pxMode=param['pxMode'])
    item.opts['useCache'] = param['useCache']
    p.clear()
    p.addItem(item)


def update():
    global ptr, lastTime, fps
    pos = data['pos']
    pen = data['pen']
    size = data['size']
    brush = data['brush']
    if not param['randomize']:
        pen = pen[0]
        size = size[0]
        brush = brush[0]
    if param['mkItem']:
        mkItem()
    item.setData(x=pos[ptr % 50], y=pos[(ptr+1) % 50], pen=pen, brush=brush, size=size)
    ptr += 1
    now = time()
    dt = now - lastTime
    lastTime = now
    if fps is None:
        fps = 1.0 / dt
    else:
        s = np.clip(dt * 3., 0, 1)
        fps = fps * (1 - s) + (1.0 / dt) * s
    p.setTitle('%0.2f fps' % fps)
    p.repaint()
    # app.processEvents()  # force complete redraw for every plot


def refresh(*args):
    if args and args[1][0][0].name() == 'paused':
        if args[1][0][2]:
            timer.stop()
        else:
            timer.start()
    else:
        mkData()
        mkItem()
        update()


refresh()
param.sigTreeStateChanged.connect(refresh)
timer.timeout.connect(update)

# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
