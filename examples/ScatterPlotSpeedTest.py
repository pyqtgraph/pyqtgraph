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

translate = QtCore.QCoreApplication.translate

app = pg.mkQApp()
param = ptree.Parameter.create(name=translate('ScatterPlot', 'Parameters'), type='group', children=[
    dict(name='paused', title=translate('ScatterPlot', 'Paused:    '), type='bool', value=False),
    dict(name='count', title=translate('ScatterPlot', 'Count:    '), type='int', limits=[1, None], value=500, step=100),
    dict(name='size', title=translate('ScatterPlot', 'Size:    '), type='int', limits=[1, None], value=10),
    dict(name='randomize', title=translate('ScatterPlot', 'Randomize:    '), type='bool', value=False),
    dict(name='_USE_QRECT', title='_USE_QRECT:    ', type='bool', value=pyqtgraph.graphicsItems.ScatterPlotItem._USE_QRECT),
    dict(name='pxMode', title='pxMode:    ', type='bool', value=True),
    dict(name='useCache', title='useCache:    ', type='bool', value=True),
    dict(name='mode', title=translate('ScatterPlot', 'Mode:    '), type='list', values={translate('ScatterPlot', 'New Item'): 'newItem', translate('ScatterPlot', 'Reuse Item'): 'reuseItem', translate('ScatterPlot', 'Simulate Pan/Zoom'): 'panZoom', translate('ScatterPlot', 'Simulate Hover'): 'hover'}, value='reuseItem'),
])
for c in param.children():
    c.setDefault(c.value())

pt = ptree.ParameterTree(showHeader=False)
pt.setParameters(param)
p = pg.PlotWidget()
splitter = QtWidgets.QSplitter()
splitter.addWidget(pt)
splitter.addWidget(p)
splitter.show()

data = {}
item = pg.ScatterPlotItem()
hoverBrush = pg.mkBrush('y')
ptr = 0
lastTime = time()
fps = None
timer = QtCore.QTimer()


def mkDataAndItem():
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
    mkItem()


def mkItem():
    global item
    pyqtgraph.graphicsItems.ScatterPlotItem._USE_QRECT = param['_USE_QRECT']
    item = pg.ScatterPlotItem(pxMode=param['pxMode'], **getData())
    item.opts['useCache'] = param['useCache']
    p.clear()
    p.addItem(item)


def getData():
    pos = data['pos']
    pen = data['pen']
    size = data['size']
    brush = data['brush']
    if not param['randomize']:
        pen = pen[0]
        size = size[0]
        brush = brush[0]
    return dict(x=pos[ptr % 50], y=pos[(ptr + 1) % 50], pen=pen, brush=brush, size=size)


def update():
    global ptr, lastTime, fps
    mode = param['mode']
    if mode == 'newItem':
        mkItem()
    elif mode == 'reuseItem':
        item.setData(**getData())
    elif mode == 'panZoom':
        item.viewTransformChanged()
        item.update()
    elif mode == 'hover':
        pts = item.points()
        old = pts[(ptr - 1) % len(pts)]
        new = pts[ptr % len(pts)]
        item.pointsAt(new.pos())
        old.resetBrush()  # reset old's brush before setting new's to better simulate hovering
        new.setBrush(hoverBrush)

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


mkDataAndItem()
for name in ['count', 'size']:
    param.child(name).sigValueChanged.connect(mkDataAndItem)
for name in ['_USE_QRECT', 'useCache', 'pxMode', 'randomize']:
    param.child(name).sigValueChanged.connect(mkItem)
param.child('paused').sigValueChanged.connect(lambda _, v: timer.stop() if v else timer.start())
timer.timeout.connect(update)
timer.start(0)

if __name__ == '__main__':
    pg.Qt.QtWidgets.QApplication.exec_()
