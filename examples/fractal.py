# -*- coding: utf-8 -*-
"""
Displays an interactive Koch fractal
"""
import initExample ## Add path to library (just for examples; you do not need this)

from functools import reduce
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

app = pg.mkQApp("Fractal Example")

# Set up UI widgets
win = pg.QtGui.QWidget()
win.setWindowTitle('pyqtgraph example: fractal demo')
layout = pg.QtGui.QGridLayout()
win.setLayout(layout)
layout.setContentsMargins(0, 0, 0, 0)
depthLabel = pg.QtGui.QLabel('fractal depth:')
layout.addWidget(depthLabel, 0, 0)
depthSpin = pg.SpinBox(value=5, step=1, bounds=[1, 10], delay=0, int=True)
depthSpin.resize(100, 20)
layout.addWidget(depthSpin, 0, 1)
w = pg.GraphicsLayoutWidget()
layout.addWidget(w, 1, 0, 1, 2)
win.show()

# Set up graphics
v = w.addViewBox()
v.setAspectLocked()
baseLine = pg.PolyLineROI([[0, 0], [1, 0], [1.5, 1], [2, 0], [3, 0]], pen=(0, 255, 0, 100), movable=False)
v.addItem(baseLine)
fc = pg.PlotCurveItem(pen=(255, 255, 255, 200), antialias=True)
v.addItem(fc)
v.autoRange()


transformMap = [0, 0, None]


def update():
    # recalculate and redraw the fractal curve
    
    depth = depthSpin.value()
    pts = baseLine.getState()['points']
    nbseg = len(pts) - 1
    nseg = nbseg**depth
    
    # Get a transformation matrix for each base segment
    trs = []
    v1 = pts[-1] - pts[0]
    l1 = v1.length()
    for i in range(len(pts)-1):
        p1 = pts[i]
        p2 = pts[i+1]
        v2 = p2 - p1
        t = p1 - pts[0]
        r = v1.angle(v2)
        s = v2.length() / l1
        trs.append(pg.SRTTransform({'pos': t, 'scale': (s, s), 'angle': r}))

    basePts = [np.array(list(pt) + [1]) for pt in baseLine.getState()['points']]
    baseMats = np.dstack([tr.matrix().T for tr in trs]).transpose(2, 0, 1)

    # Generate an array of matrices to transform base points
    global transformMap
    if transformMap[:2] != [depth, nbseg]:
        # we can cache the transform index to save a little time..
        nseg = nbseg**depth
        matInds = np.empty((depth, nseg), dtype=int)
        for i in range(depth):
            matInds[i] = np.tile(np.repeat(np.arange(nbseg), nbseg**(depth-1-i)), nbseg**i)
        transformMap = [depth, nbseg, matInds]
        
    # Each column in matInds contains the indices referring to the base transform
    # matrices that must be multiplied together to generate the final transform
    # for each segment of the fractal 
    matInds = transformMap[2]
    
    # Collect all matrices needed for generating fractal curve
    mats = baseMats[matInds]
    
    # Magic-multiply stacks of matrices together
    def matmul(a, b):
        return np.sum(np.transpose(a,(0,2,1))[..., None] * b[..., None, :], axis=-3)
    mats = reduce(matmul, mats)
    
    # Transform base points through matrix array
    pts = np.empty((nseg * nbseg + 1, 2))
    for l in range(len(trs)):
        bp = basePts[l]
        pts[l:-1:len(trs)] = np.dot(mats, bp)[:, :2]
    
    # Finish the curve with the last base point
    pts[-1] = basePts[-1][:2]

    # update fractal curve with new points
    fc.setData(pts[:,0], pts[:,1])


# Update the fractal whenever the base shape or depth has changed
baseLine.sigRegionChanged.connect(update)
depthSpin.valueChanged.connect(update)

# Initialize
update()

if __name__ == '__main__':
    pg.exec()
