"""
Very basic 3D graphics example; create a view widget and add a few items.

"""

import pyqtgraph as pg
import pyqtgraph.opengl as gl

pg.mkQApp("GLViewWidget Example")
w = gl.GLViewWidget()
w.show()
w.setWindowTitle('pyqtgraph example: GLViewWidget')
w.setCameraPosition(distance=20)

ax = gl.GLAxisItem()
ax.setSize(5,5,5)
w.addItem(ax)

b = gl.GLBoxItem()
w.addItem(b)

ax2 = gl.GLAxisItem()
ax2.setParentItem(b)

b.translate(1,1,1)

if __name__ == '__main__':
    pg.exec()
