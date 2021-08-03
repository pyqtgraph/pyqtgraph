#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Update a simple plot as rapidly as possible to measure speed.
"""

## Add path to library (just for examples; you do not need this)
import initExample

from collections import deque
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets, QT_LIB
import numpy as np
import pyqtgraph as pg
from time import perf_counter
import pyqtgraph.parametertree as ptree
import pyqtgraph.functions as fn
import itertools
import argparse

from pyqtgraph.parametertree import interact, InteractiveFunction

if QT_LIB.startswith('PyQt'):
    wrapinstance = pg.Qt.sip.wrapinstance
else:
    wrapinstance = pg.Qt.shiboken.wrapInstance

# defaults here result in the same configuration as the original PlotSpeedTest
parser = argparse.ArgumentParser()
parser.add_argument('--noise', dest='noise', action='store_true')
parser.add_argument('--no-noise', dest='noise', action='store_false')
parser.set_defaults(noise=True)
parser.add_argument('--nsamples', default=5000, type=int)
parser.add_argument('--frames', default=50, type=int)
parser.add_argument('--fsample', default=1000, type=float)
parser.add_argument('--frequency', default=0, type=float)
parser.add_argument('--amplitude', default=5, type=float)
parser.add_argument('--opengl', dest='use_opengl', action='store_true')
parser.add_argument('--no-opengl', dest='use_opengl', action='store_false')
parser.set_defaults(use_opengl=None)
parser.add_argument('--allow-opengl-toggle', action='store_true',
    help="""Allow on-the-fly change of OpenGL setting. This may cause unwanted side effects.
    """)
args = parser.parse_args()

if args.use_opengl is not None:
    pg.setConfigOption('useOpenGL', args.use_opengl)
    pg.setConfigOption('enableExperimental', args.use_opengl)

# don't limit frame rate to vsync
sfmt = QtGui.QSurfaceFormat()
sfmt.setSwapInterval(0)
QtGui.QSurfaceFormat.setDefaultFormat(sfmt)

class LineInstances:
    def __init__(self):
        self.alloc(0)

    def alloc(self, size):
        self.arr = np.empty((size, 4), dtype=np.float64)
        self.ptrs = list(map(wrapinstance,
            itertools.count(self.arr.ctypes.data, self.arr.strides[0]),
            itertools.repeat(QtCore.QLineF, self.arr.shape[0])))

    def array(self, size):
        if size > self.arr.shape[0]:
            self.alloc(size + 16)
        return self.arr[:size]

    def instances(self, size):
        return self.ptrs[:size]

class MonkeyCurveItem(pg.PlotCurveItem):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.monkey_mode = ''
        self._lineInstances = LineInstances()

    def setMethod(self, method):
        self.monkey_mode = method

    def paint(self, painter, opt, widget):
        if self.monkey_mode not in ['drawPolyline', 'drawLines']:
            return super().paint(painter, opt, widget)

        painter.setRenderHint(painter.RenderHint.Antialiasing, self.opts['antialias'])
        painter.setPen(pg.mkPen(self.opts['pen']))

        if self.monkey_mode == 'drawPolyline':
            painter.drawPolyline(fn.arrayToQPolygonF(self.xData, self.yData))
        elif self.monkey_mode == 'drawLines':
            lines = self._lineInstances
            npts = len(self.xData)
            even_slice = slice(0, 0+(npts-0)//2*2)
            odd_slice = slice(1, 1+(npts-1)//2*2)
            for sl in [even_slice, odd_slice]:
                npairs = (sl.stop - sl.start) // 2
                memory = lines.array(npairs).reshape((-1, 2))
                memory[:, 0] = self.xData[sl]
                memory[:, 1] = self.yData[sl]
                painter.drawLines(lines.instances(npairs))

app = pg.mkQApp("Plot Speed Test")

default_pen = pg.mkPen()

params = ptree.Parameter.create(name='Parameters', type='group')
pt = ptree.ParameterTree(showHeader=False)
pt.setParameters(params)
pw = pg.PlotWidget()
splitter = QtWidgets.QSplitter()
splitter.addWidget(pt)
splitter.addWidget(pw)
splitter.setSizes([375, pw.width()])
splitter.show()

pw.setWindowTitle('pyqtgraph example: PlotSpeedTest')
pw.setLabel('bottom', 'Index', units='B')
curve = MonkeyCurveItem(pen=default_pen)
pw.addItem(curve)

rollingAverageSize = 1000
elapsed = deque(maxlen=rollingAverageSize)

def resetTimings(*args):
    elapsed.clear()

@params.interactDecorator(title='Signal Options')
def makeData(nsamples=args.nsamples, frames=args.frames, fsample=args.fsample, amplitude=args.amplitude,
             frequency=args.frequency, noise=args.noise):
    """
    [nsamples.options]
    limits = [0, None]

    [frames.options]
    limits = [1, None]

    [fsample.options]
    title = 'sample rate'
    units = 'Hz'

    [frequency.options]
    units = 'Hz'
    """
    global data, connect_array, ptr
    ttt = np.arange(frames * nsamples, dtype=np.float64) / fsample
    data = amplitude*np.sin(2*np.pi*frequency*ttt).reshape((frames, nsamples))
    if noise:
        data += np.random.normal(size=data.shape)
    connect_array = np.ones(data.shape[-1], dtype=bool)
    ptr = 0
    pw.setRange(QtCore.QRectF(0, -10, nsamples, 20))

# Use function with same name so interacted params are under the same menu
def update(useOpenGL=pg.getConfigOption('useOpenGL'), enableExperimental=pg.getConfigOption('enableExperimental')):
    pw.useOpenGL(useOpenGL)
    pg.setConfigOption('enableExperimental', enableExperimental)
interact(update, parent=params)

@params.interactDecorator(title='Curve Options')
def setCurveOpts(pen=default_pen, method='pyqtgraph'):
    """
    [pen.options]
    type = pen

    [method.options]
    limits = ['pyqtgraph', 'drawPolyline', 'drawLines']
    type = 'list'
    title = 'Plot Method'
    """
    curve.setPen(pen)
    curve.setMethod(method)

def update(antialias=pg.getConfigOption('antialias'), connect='all', skipfiniteCheck=False):
    """
    [connect.options]
    type = list
    limits = ['all', 'pairs', 'finite', 'array']
    """
    global curve, data, ptr, elapsed, fpsLastUpdate

    if connect == 'array':
        connect = connect_array

    # Measure
    t_start = perf_counter()
    curve.setData(data[ptr], antialias=antialias, connect=connect, skipfiniteCheck=skipfiniteCheck)
    app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
    t_end = perf_counter()
    elapsed.append(t_end - t_start)
    ptr = (ptr + 1) % data.shape[0]

    # update fps at most once every 0.2 secs
    if t_end - fpsLastUpdate > 0.2:
        fpsLastUpdate = t_end
        average = np.mean(elapsed)
        fps = 1 / average
        pw.setTitle('%0.2f fps - %0.1f ms avg' % (fps, average * 1_000))

params.sigTreeStateChanged.connect(resetTimings)
makeData()
fpsLastUpdate = perf_counter()

# Wrap update as interactive to preserve values across calls
update_interactive = InteractiveFunction(update)
interact(update_interactive, parent=params, title='Display Options')
timer = QtCore.QTimer()
timer.timeout.connect(update_interactive)
timer.start(0)

if __name__ == '__main__':
    pg.exec()
