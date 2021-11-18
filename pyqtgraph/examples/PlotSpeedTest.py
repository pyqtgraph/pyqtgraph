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

    def setMethod(self, param, value):
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

children = [
    dict(name='sigopts', title='Signal Options', type='group', children=[
        dict(name='noise', type='bool', value=args.noise),
        dict(name='nsamples', type='int', limits=[0, None], value=args.nsamples),
        dict(name='frames', type='int', limits=[1, None], value=args.frames),
        dict(name='fsample', title='sample rate', type='float', value=args.fsample, units='Hz'),
        dict(name='frequency', type='float', value=args.frequency, units='Hz'),
        dict(name='amplitude', type='float', value=args.amplitude),
    ]),
    dict(name='useOpenGL', type='bool', value=pg.getConfigOption('useOpenGL'),
        readonly=not args.allow_opengl_toggle),
    dict(name='enableExperimental', type='bool', value=pg.getConfigOption('enableExperimental')),
    dict(name='pen', type='pen', value=default_pen),
    dict(name='antialias', type='bool', value=pg.getConfigOption('antialias')),
    dict(name='connect', type='list', limits=['all', 'pairs', 'finite', 'array'], value='all'),
    dict(name='fill', type='bool', value=False),
    dict(name='skipFiniteCheck', type='bool', value=False),
    dict(name='plotMethod', title='Plot Method', type='list', limits=['pyqtgraph', 'drawPolyline'])
]

params = ptree.Parameter.create(name='Parameters', type='group', children=children)
pt = ptree.ParameterTree(showHeader=False)
pt.setParameters(params)
pw = pg.PlotWidget()
splitter = QtWidgets.QSplitter()
splitter.addWidget(pt)
splitter.addWidget(pw)
splitter.show()

pw.setWindowTitle('pyqtgraph example: PlotSpeedTest')
pw.setLabel('bottom', 'Index', units='B')
curve = MonkeyCurveItem(pen=default_pen, brush='b')
pw.addItem(curve)

rollingAverageSize = 1000
elapsed = deque(maxlen=rollingAverageSize)

def resetTimings(*args):
    elapsed.clear()

def makeData(*args):
    global data, connect_array, ptr
    sigopts = params.child('sigopts')
    nsamples = sigopts['nsamples']
    frames = sigopts['frames']
    Fs = sigopts['fsample']
    A = sigopts['amplitude']
    F = sigopts['frequency']
    ttt = np.arange(frames * nsamples, dtype=np.float64) / Fs
    data = A*np.sin(2*np.pi*F*ttt).reshape((frames, nsamples))
    if sigopts['noise']:
        data += np.random.normal(size=data.shape)
    connect_array = np.ones(data.shape[-1], dtype=bool)
    ptr = 0
    pw.setRange(QtCore.QRectF(0, -10, nsamples, 20))

def onUseOpenGLChanged(param, enable):
    pw.useOpenGL(enable)

def onEnableExperimentalChanged(param, enable):
    pg.setConfigOption('enableExperimental', enable)

def onPenChanged(param, pen):
    curve.setPen(pen)

def onFillChanged(param, enable):
    curve.setFillLevel(0.0 if enable else None)

params.child('sigopts').sigTreeStateChanged.connect(makeData)
params.child('useOpenGL').sigValueChanged.connect(onUseOpenGLChanged)
params.child('enableExperimental').sigValueChanged.connect(onEnableExperimentalChanged)
params.child('pen').sigValueChanged.connect(onPenChanged)
params.child('fill').sigValueChanged.connect(onFillChanged)
params.child('plotMethod').sigValueChanged.connect(curve.setMethod)
params.sigTreeStateChanged.connect(resetTimings)

makeData()

fpsLastUpdate = perf_counter()
def update():
    global curve, data, ptr, elapsed, fpsLastUpdate

    options = ['antialias', 'connect', 'skipFiniteCheck']
    kwds = { k : params[k] for k in options }
    if kwds['connect'] == 'array':
        kwds['connect'] = connect_array

    # Measure
    t_start = perf_counter()
    curve.setData(data[ptr], **kwds)
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

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

if __name__ == '__main__':
    pg.exec()
