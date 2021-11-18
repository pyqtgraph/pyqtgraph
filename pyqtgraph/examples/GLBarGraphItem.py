"""
This example demonstrates the use of GLBarGraphItem.

"""

import numpy as np

import pyqtgraph as pg
import pyqtgraph.opengl as gl

app = pg.mkQApp("GLBarGraphItem Example")
w = gl.GLViewWidget()
w.show()
w.setWindowTitle('pyqtgraph example: GLBarGraphItem')
w.setCameraPosition(distance=40)

gx = gl.GLGridItem()
gx.rotate(90, 0, 1, 0)
gx.translate(-10, 0, 10)
w.addItem(gx)
gy = gl.GLGridItem()
gy.rotate(90, 1, 0, 0)
gy.translate(0, -10, 10)
w.addItem(gy)
gz = gl.GLGridItem()
gz.translate(0, 0, 0)
w.addItem(gz)

# regular grid of starting positions
pos = np.mgrid[0:10, 0:10, 0:1].reshape(3,10,10).transpose(1,2,0)
# fixed widths, random heights
size = np.empty((10,10,3))
size[...,0:2] = 0.4
size[...,2] = np.random.normal(size=(10,10))

bg = gl.GLBarGraphItem(pos, size)
w.addItem(bg)

if __name__ == '__main__':
    pg.exec()
