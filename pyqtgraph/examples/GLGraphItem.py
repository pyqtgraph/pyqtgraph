"""
Demonstrates use of GLGraphItem
"""

import numpy as np

import pyqtgraph as pg
import pyqtgraph.opengl as gl

app = pg.mkQApp("GLGraphItem Example")
w = gl.GLViewWidget()
w.setCameraPosition(distance=20)
w.show()

edges = np.array([
    [0, 2],
    [0, 3],
    [1, 2],
    [1, 3],
    [2, 3]
])

nodes = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 1]
    ]
)

edgeColor=pg.glColor("w")

gi = gl.GLGraphItem(
    edges=edges,
    nodePositions=nodes,
    edgeWidth=1.,
    nodeSize=10.
)

w.addItem(gi)

if __name__ == "__main__":
    pg.exec()
