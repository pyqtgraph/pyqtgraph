#!/usr/bin/python
"""
Update a simple plot as rapidly as possible to measure speed.
"""

import argparse
import itertools

import numpy as np
from utils import FrameCounter

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
parser.add_argument('--iterations', default=float('inf'), type=float,
    help="Number of iterations to run before exiting"
)
args = parser.parse_args()

if args.use_opengl is not None:
    pg.setConfigOption('useOpenGL', args.use_opengl)
    pg.setConfigOption('enableExperimental', args.use_opengl)
use_opengl = pg.getConfigOption('useOpenGL')

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
iterations_counter = itertools.count()

@interactor.decorate(
    nest=True,
    nsamples={'limits': [0, None]},
    frames={'limits': [1, None]},
    fsample={'units': 'Hz'},
    frequency={'units': 'Hz'}
)
def makeData(
    noise=args.noise,
    nsamples=args.nsamples,
    frames=args.frames,
    fsample=args.fsample,
    frequency=args.frequency,
    amplitude=args.amplitude,
):
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
    global ptr

    if next(iterations_counter) > args.iterations:
        # cleanly close down benchmark
        timer.stop()
        app.quit()
        return None

    if connect == 'array':
        connect = connect_array

    curve.setData(
        data[ptr],
        antialias=antialias,
        connect=connect,
        skipFiniteCheck=skipFiniteCheck
    )
    ptr = (ptr + 1) % data.shape[0]
    framecnt.update()


@interactor.decorate(
    useOpenGL={'readonly': not args.allow_opengl_toggle},
    plotMethod={'limits': ['pyqtgraph', 'drawPolyline'], 'type': 'list'},
    curvePen={'type': 'pen'}
)
def updateOptions(
    curvePen=pg.mkPen(),
    plotMethod='pyqtgraph',
    fillLevel=False,
    enableExperimental=use_opengl,
    useOpenGL=use_opengl,
):
    pg.setConfigOption('enableExperimental', enableExperimental)
    pw.useOpenGL(useOpenGL)
    curve.setPen(curvePen)
    curve.setFillLevel(0.0 if fillLevel else None)
    curve.setMethod(plotMethod)


makeData()

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

framecnt = FrameCounter()
framecnt.sigFpsUpdate.connect(lambda fps: pw.setTitle(f'{fps:.1f} fps'))

if __name__ == '__main__':
    # Splitter by default gives too small of a width to the parameter tree,
    # so fix that right before the event loop
    pt.setMinimumSize(225,0)
    pg.exec()
