# -*- coding: utf-8 -*-
"""
Display a non-uniform image.
This example displays 2-d data as an image with non-uniformly
distributed sample points.
"""

import initExample ## Add path to library (just for examples; you do not need this)

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
from pyqtgraph.graphicsItems.GradientEditorItem import Gradients
from pyqtgraph.graphicsItems.NonUniformImage import NonUniformImage


RPM2RADS = 2 * np.pi / 60
RADS2RPM = 1 / RPM2RADS

kfric  = 1       # [Ws/rad] angular damping coefficient [0;100]
kfric3 = 1.5e-6  # [Ws3/rad3] angular damping coefficient (3rd order) [0;10-3]
psi    = 0.2     # [Vs] flux linkage [0.001;10]
res    = 5e-3    # [Ohm] resistance [0;100]
v_ref  = 200     # [V] reference DC voltage [0;1000]
k_v    = 5       # linear voltage coefficient [-100;100]

# create the (non-uniform) scales
tau = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220], dtype=np.float32)
w = np.array([0, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], dtype=np.float32) * RPM2RADS
v = 380

# calculate the power losses
TAU, W = np.meshgrid(tau, w, indexing='ij')
V = np.ones_like(TAU) * v

P_loss = kfric * W + kfric3 * W ** 3 + (res * (TAU / psi) ** 2) + k_v * (V - v_ref)

P_mech = TAU * W
P_loss[P_mech > 1.5e5] = np.NaN

# green - orange - red
Gradients['gor'] = {'ticks': [(0.0, (74, 158, 71)), (0.5, (255, 230, 0)), (1, (191, 79, 76))], 'mode': 'rgb'}

app = pg.mkQApp("NonUniform Image Example")

win = QtGui.QMainWindow()
cw = pg.GraphicsLayoutWidget()
win.show()
win.resize(600, 400)
win.setCentralWidget(cw)
win.setWindowTitle('pyqtgraph example: Non-uniform Image')

p = cw.addPlot(title="Power Losses [W]", row=0, col=0)

lut = pg.HistogramLUTItem(orientation="horizontal")

p.setMouseEnabled(x=False, y=False)

cw.nextRow()
cw.addItem(lut)

# load the gradient
lut.gradient.loadPreset('gor')

image = NonUniformImage(w * RADS2RPM, tau, P_loss.T)
image.setLookupTable(lut, autoLevel=True)
image.setZValue(-1)
p.addItem(image)

h = image.getHistogram()
lut.plot.setData(*h)

p.showGrid(x=True, y=True)

p.setLabel(axis='bottom', text='Speed [rpm]')
p.setLabel(axis='left', text='Torque [Nm]')

# elevate the grid lines
p.axes['bottom']['item'].setZValue(1000)
p.axes['left']['item'].setZValue(1000)

if __name__ == '__main__':
    pg.exec()
