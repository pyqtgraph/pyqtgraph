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
import pyqtgraph.parametertree as ptree
from pyqtgraph.parametertree.parameterTypes import GroupParameter as GP, Parameter
import pyqtgraph.graphicsItems.ScatterPlotItem
from time import perf_counter
import re
from contextlib import ExitStack

translate = QtCore.QCoreApplication.translate

app = pg.mkQApp()

pt = ptree.ParameterTree(showHeader=False)
param = Parameter.create(name=translate('ScatterPlot', 'Parameters'), type='group')
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
lastTime = perf_counter()
fps = None
timer = QtCore.QTimer()

def fmt(name):
    replace = r'\1 \2'
    name = re.sub(r'(\w)([A-Z])', replace, name)
    name = name.replace('_', ' ')
    return translate('ScatterPlot', name.title().strip() + ':    ')

# Exit stack to avoid indentation for this entire file
stack = ExitStack()
stack.enter_context(GP.interactiveOptsContext(runTitleFormat=fmt))

@param.interactDecorator(nest=False,
                          count=dict(limits=[1, None], step=100),
                          size=dict(limits=[1, None])
                          )
def mkDataAndItem(count=500, size=10):
    global data, fps
    scale = 100
    data = {
        'pos': np.random.normal(size=(50, count), scale=scale),
        'pen': [pg.mkPen(x) for x in np.random.randint(0, 256, (count, 3))],
        'brush': [pg.mkBrush(x) for x in np.random.randint(0, 256, (count, 3))],
        'size': (np.random.random(count) * size).astype(int)
    }
    data['pen'][0] = pg.mkPen('w')
    data['size'][0] = size
    data['brush'][0] = pg.mkBrush('b')
    bound = 5 * scale
    p.setRange(xRange=[-bound, bound], yRange=[-bound, bound])
    mkItem()


@param.interactDecorator(nest=False)
def mkItem(pxMode=True, useCache=True):
    global item
    item = pg.ScatterPlotItem(pxMode=pxMode, **getData())
    item.opts['useCache'] = useCache
    p.clear()
    p.addItem(item)

@param.interactDecorator(nest=False)
def getData(randomize=False):
    pos = data['pos']
    pen = data['pen']
    size = data['size']
    brush = data['brush']
    if not randomize:
        pen = pen[0]
        size = size[0]
        brush = brush[0]
    return dict(x=pos[ptr % 50], y=pos[(ptr + 1) % 50], pen=pen, brush=brush, size=size)

modeOpts = dict(name='mode',
                type='list',
                values={translate('ScatterPlot', 'New Item'): 'newItem',
                        translate('ScatterPlot', 'Reuse Item'): 'reuseItem',
                        translate('ScatterPlot', 'Simulate Pan/Zoom'): 'panZoom',
                        translate('ScatterPlot', 'Simulate Hover'): 'hover'},
                )
@param.interactDecorator(nest=False, mode=modeOpts)
def update(mode='reuseItem'):
    global ptr, lastTime, fps
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
    now = perf_counter()
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

param.addChild(dict(name='paused', title=translate('ScatterPlot', 'Paused:    '), type='bool', value=False))

mkDataAndItem()
param.child('paused').sigValueChanged.connect(lambda _, v: timer.stop() if v else timer.start())
timer.timeout.connect(update)
timer.start(0)
stack.close()

if __name__ == '__main__':
    pg.exec()
