"""Demonstrate GLGridAxisItem with GLSurfacePlotItem."""

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import mkQApp
from pyqtgraph.opengl import GLViewWidget, GLGridAxisItem, GLSurfacePlotItem


def float_to_str(*args, decimals=1):
    return [f'{x:.{decimals}f}' for x in args]

mkQApp("GLGridAxisItem Example")

w = GLViewWidget()
w.show()

grid_axes = GLGridAxisItem()
w.addItem(grid_axes)

extent = 10
points = 36
amplitude = 10
frequency = 1
x = np.linspace(-extent, extent, points)
y = np.linspace(-extent, extent, points)
z = np.zeros((points, points))
for i in range(points):
    yi = y[i]
    d = np.hypot(x, yi)
    z[:,i] = amplitude * np.cos(frequency*d) / (d+1)

surface = GLSurfacePlotItem(
    x=x, y=y, z=z,
    shader='heightColor',
    showGrid=True,
    lineColor=(0.25,0.25,0.25,1)
)
surface.shader()['colorMap'] = np.array([0.2, 2, 0.5, 0.2, 1, 1, 0.2, 0, 2])
w.addItem(surface)

coords_limit =extent
limit_factor = 1.05
coords = {
    'x': np.linspace(-coords_limit, coords_limit, 5),
    'y': np.linspace(-coords_limit, coords_limit, 5),
    'z': np.linspace(-4, 8, 7),
}
coords_labels = {c: float_to_str(*coords[c]) for c in 'xyz'}
limits = {c: (limit_factor*coords[c][0], limit_factor*coords[c][-1]) for c in 'xyz'}

grid_axes.setData(coords=coords, coords_labels=coords_labels, limits=limits)
w.setCameraPosition(**grid_axes.best_camera())


if __name__ == '__main__':
    pg.exec()
