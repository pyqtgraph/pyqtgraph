# -*- coding: utf-8 -*-
"""
ViewBox is the general-purpose graphical container that allows the user to 
zoom / pan to inspect any area of a 2D coordinate system. 

This example demonstrates many of the features ViewBox provides.
"""

import initExample ## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

x = np.arange(1000, dtype=float)
y = np.random.normal(size=1000)
y += 5 * np.sin(x/100) 

win = pg.GraphicsLayoutWidget(show=True)
win.setWindowTitle('pyqtgraph example: ____')
win.resize(1000, 800)
win.ci.setBorder((50, 50, 100))

sub1 = win.addLayout()
sub1.addLabel("<b>Standard mouse interaction:</b><br>left-drag to pan, right-drag to zoom.")
sub1.nextRow()
v1 = sub1.addViewBox()
l1 = pg.PlotDataItem(y)
v1.addItem(l1)


sub2 = win.addLayout()
sub2.addLabel("<b>One-button mouse interaction:</b><br>left-drag zoom to box, wheel to zoom out.")
sub2.nextRow()
v2 = sub2.addViewBox()
v2.setMouseMode(v2.RectMode)
l2 = pg.PlotDataItem(y)
v2.addItem(l2)

win.nextRow()

sub3 = win.addLayout()
sub3.addLabel("<b>Locked aspect ratio when zooming.</b>")
sub3.nextRow()
v3 = sub3.addViewBox()
v3.setAspectLocked(1.0)
l3 = pg.PlotDataItem(y)
v3.addItem(l3)

sub4 = win.addLayout()
sub4.addLabel("<b>View limits:</b><br>prevent panning or zooming past limits.")
sub4.nextRow()
v4 = sub4.addViewBox()
v4.setLimits(xMin=-100, xMax=1100, 
             minXRange=20, maxXRange=500, 
             yMin=-10, yMax=10,
             minYRange=1, maxYRange=10)
l4 = pg.PlotDataItem(y)
v4.addItem(l4)

win.nextRow()

sub5 = win.addLayout()
sub5.addLabel("<b>Linked axes:</b> Data in this plot is always X-aligned to<br>the plot above.")
sub5.nextRow()
v5 = sub5.addViewBox()
v5.setXLink(v3)
l5 = pg.PlotDataItem(y)
v5.addItem(l5)

sub6 = win.addLayout()
sub6.addLabel("<b>Disable mouse:</b> Per-axis control over mouse input.<br>"
              "<b>Auto-scale-visible:</b> Automatically fit *visible* data within view<br>"
              "(try panning left-right).")
sub6.nextRow()
v6 = sub6.addViewBox()
v6.setMouseEnabled(x=True, y=False)
v6.enableAutoRange(x=False, y=True)
v6.setXRange(300, 450)
v6.setAutoVisible(x=False, y=True)
l6 = pg.PlotDataItem(y)
v6.addItem(l6)

if __name__ == '__main__':
    pg.exec()
