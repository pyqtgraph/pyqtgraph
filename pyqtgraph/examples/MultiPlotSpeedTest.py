#!/usr/bin/python
"""
Test the speed of rapidly updating multiple plot curves
"""
import argparse
import itertools

import numpy as np
from utils import FrameCounter

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

parser = argparse.ArgumentParser()
parser.add_argument('--iterations', default=float('inf'), type=float,
    help="Number of iterations to run before exiting"
)
args = parser.parse_args()
iterations_counter = itertools.count()

# pg.setConfigOptions(useOpenGL=True)
app = pg.mkQApp("MultiPlot Speed Test")

plot = pg.plot()
plot.setWindowTitle('pyqtgraph example: MultiPlotSpeedTest')
plot.setLabel('bottom', 'Index', units='B')

nPlots = 100
nSamples = 500
curves = []
for idx in range(nPlots):
    curve = pg.PlotCurveItem(pen=({'color': (idx, nPlots*1.3), 'width': 1}), skipFiniteCheck=True)
    plot.addItem(curve)
    curve.setPos(0,idx*6)
    curves.append(curve)

plot.setYRange(0, nPlots*6)
plot.setXRange(0, nSamples)
plot.resize(600,900)

rgn = pg.LinearRegionItem([nSamples/5.,nSamples/3.])
plot.addItem(rgn)


data = np.random.normal(size=(nPlots*23,nSamples))
ptr = 0
def update():
    global ptr
    if next(iterations_counter) > args.iterations:
        timer.stop()
        app.quit()
        return None
    for i in range(nPlots):
        curves[i].setData(data[(ptr+i)%data.shape[0]])

    ptr += nPlots
    framecnt.update()

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

framecnt = FrameCounter()
framecnt.sigFpsUpdate.connect(lambda fps: plot.setTitle(f'{fps:.1f} fps'))

if __name__ == '__main__':
    pg.exec()
