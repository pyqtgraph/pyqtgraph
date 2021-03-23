#!/usr/bin/python
# -*- coding: utf-8 -*-
# Add path to library (just for examples; you do not need this)
import initExample
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

app = pg.mkQApp()
mw = QtGui.QMainWindow()
mw.resize(800, 800)
pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")

mpw = pg.MultiAxisPlotWidget()

mw.setCentralWidget(mpw)
mw.show()

# LEGEND
mpw.addLegend(offset=(0, 0))
# TITLE
mpw.setTitle("MultiAxisPlotWidget Example")
# AXYS
a1 = mpw.addAxis("samples1", "bottom", text="Samples1", units="samples1")
a2 = mpw.addAxis("samples2", "bottom", text="Samples2", units="samples2")
as1 = mpw.addAxis("sin1", "left", text="Data1", units="sin1")
as2 = mpw.addAxis("sin2", "left", text="Data2", units="sin2")
# CHARTS
c1, pi1 = mpw.addChart("Dataset 1", "sin1", "samples1")
c2, pi2 = mpw.addChart("Dataset 2", "sin1", "samples2")
c3, pi3 = mpw.addChart("Dataset 3", "sin2", "samples2")
# make and display chart
mpw.makeLayout()

data1 = np.array(np.sin(np.linspace(0, 2 * np.pi, num=1000)))
c1.setData(data1)
data2 = data1 * 2
c2.setData(data2)
data3 = np.array(np.sin(np.linspace(0, 4 * np.pi, num=500))) * 3
c3.setData(data3)

mpw.update()


# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    pg.mkQApp().exec_()
