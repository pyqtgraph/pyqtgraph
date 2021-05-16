# -*- coding: utf-8 -*-
"""
Demonstrates GLVolumeItem for displaying volumetric data.

"""

## Add path to library (just for examples; you do not need this)
import initExample

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph import functions as fn

app = pg.mkQApp("GLVolumeItem Example")
w = gl.GLViewWidget()
w.opts['distance'] = 200
w.show()
w.setWindowTitle('pyqtgraph example: GLVolumeItem')

g = gl.GLGridItem()
g.scale(10, 10, 1)
w.addItem(g)

## Hydrogen electron probability density
def psi(i, j, k, offset=(50,50,100)):
    x = i-offset[0]
    y = j-offset[1]
    z = k-offset[2]
    th = np.arctan2(z, np.hypot(x, y))
    r = np.sqrt(x**2 + y**2 + z **2)
    a0 = 2
    return (
        (1.0 / 81.0)
        * 1.0 / (6.0 * np.pi) ** 0.5
        * (1.0 / a0) ** (3 / 2)
        * (r / a0) ** 2
        * np.exp(-r / (3 * a0))
        * (3 * np.cos(th) ** 2 - 1)
    )


data = np.fromfunction(psi, (100,100,200))
with np.errstate(divide = 'ignore'):
    positive = np.log(fn.clip_array(data, 0, data.max())**2)
    negative = np.log(fn.clip_array(-data, 0, -data.min())**2)

d2 = np.empty(data.shape + (4,), dtype=np.ubyte)
d2[..., 0] = positive * (255./positive.max())
d2[..., 1] = negative * (255./negative.max())
d2[..., 2] = d2[...,1]
d2[..., 3] = d2[..., 0]*0.3 + d2[..., 1]*0.3
d2[..., 3] = (d2[..., 3].astype(float) / 255.) **2 * 255

d2[:, 0, 0] = [255,0,0,100]
d2[0, :, 0] = [0,255,0,100]
d2[0, 0, :] = [0,0,255,100]

v = gl.GLVolumeItem(d2)
v.translate(-50,-50,-100)
w.addItem(v)

ax = gl.GLAxisItem()
w.addItem(ax)

if __name__ == '__main__':
    pg.exec()
