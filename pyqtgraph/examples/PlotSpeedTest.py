#!/usr/bin/python
"""
Update a simple plot as rapidly as possible to measure speed.
"""

import argparse
from collections import deque
from time import perf_counter

import numpy as np

import pyqtgraph as pg
import pyqtgraph.functions as fn
import pyqtgraph.parametertree as ptree
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

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


class MonkeyCurveItem(pg.PlotCurveItem):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.monkey_mode = ''

    def setMethod(self, value):
        self.monkey_mode = value

    def paint(self, painter, opt, widget):
        if self.monkey_mode not in ['drawPolyline']:
            return super().paint(painter, opt, widget)

        painter.setRenderHint(painter.RenderHint.Antialiasing, self.opts['antialias'])
        painter.setPen(pg.mkPen(self.opts['pen']))

        if self.monkey_mode == 'drawPolyline':
            painter.drawPolyline(fn.arrayToQPolygonF(self.xData, self.yData))

app = pg.mkQApp("Plot Speed Test")

default_pen = pg.mkPen()

params = ptree.Parameter.create(name='Parameters', type='group')
pt = ptree.ParameterTree(showHeader=False)
pt.setParameters(params)
pw = pg.PlotWidget()
splitter = QtWidgets.QSplitter()
splitter.addWidget(pt)
splitter.addWidget(pw)
splitter.show()

interactor = ptree.Interactor(
    parent=params, nest=False, runOptions=ptree.RunOptions.ON_CHANGED
)

pw.setWindowTitle('pyqtgraph example: PlotSpeedTest')
pw.setLabel('bottom', 'Index', units='B')
curve = MonkeyCurveItem(pen=default_pen, brush='b')
pw.addItem(curve)

rollingAverageSize = 1000
elapsed = deque(maxlen=rollingAverageSize)

def resetTimings(*args):
    elapsed.clear()

@interactor.decorate(
    nest=True,
    nsamples={'limits': [0, None]},
    frames={'limits': [1, None]},
    fsample={'units': 'Hz'},
    frequency={'units': 'Hz'}
)
def makeData(noise=True, nsamples=5000, frames=50, fsample=1000.0, frequency=0.0, amplitude=5.0):
    global data, connect_array, ptr
    ttt = np.arange(frames * nsamples, dtype=np.float64) / fsample
    data = amplitude*np.sin(2*np.pi*frequency*ttt).reshape((frames, nsamples))
    if noise:
        data += np.random.normal(size=data.shape)
    connect_array = np.ones(data.shape[-1], dtype=bool)
    ptr = 0
    pw.setRange(QtCore.QRectF(0, -10, nsamples, 20))

params.child('makeData').setOpts(title='Plot Options')

@interactor.decorate(
    connect={'type': 'list', 'limits': ['all', 'pairs', 'finite', 'array']}
)
def update(
    antialias=pg.getConfigOption('antialias'),
    connect='all',
    skipFiniteCheck=False
):
    global curve, data, ptr, elapsed, fpsLastUpdate

    if connect == 'array':
        connect = connect_array

    # Measure
    t_start = perf_counter()
    curve.setData(data[ptr], antialias=antialias, connect=connect, skipFiniteCheck=skipFiniteCheck)
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

@interactor.decorate(
    useOpenGL={'readonly': not args.allow_opengl_toggle},
    plotMethod={'limits': ['pyqtgraph', 'drawPolyline'], 'type': 'list'},
    curvePen={'type': 'pen'}
)
def updateOptions(
    curvePen=pg.mkPen(),
    plotMethod='pyqtgraph',
    fillLevel=False,
    enableExperimental=False,
    useOpenGL=False,
):
    pg.setConfigOption('enableExperimental', enableExperimental)
    pg.setConfigOption('useOpenGL', useOpenGL)
    curve.setPen(curvePen)
    curve.setFillLevel(0.0 if fillLevel else None)
    curve.setMethod(plotMethod)

params.sigTreeStateChanged.connect(resetTimings)

makeData()

fpsLastUpdate = perf_counter()

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

if __name__ == '__main__':
    # Splitter by default gives too small of a width to the parameter tree,
    # so fix that right before the event loop
    pt.setMinimumSize(225,0)
    pg.exec()
