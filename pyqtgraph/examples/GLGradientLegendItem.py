import numpy

import pyqtgraph as pg
import pyqtgraph.opengl as gl

app = pg.mkQApp()
w = gl.GLViewWidget()
w.show()
w.setWindowTitle("pyqtgraph example: GLGradientLegendItem")
w.setCameraPosition(distance=60)

gx = gl.GLGridItem()
gx.rotate(90, 0, 1, 0)
w.addItem(gx)

md = gl.MeshData.cylinder(rows=10, cols=20, radius=[5.0, 5], length=20.0)
md._vertexes[:, 2] = md._vertexes[:, 2] - 10

# set color based on z coordinates
color_map = pg.colormap.get("CET-L10")

h = md.vertexes()[:, 2]
# remember these
h_max, h_min = h.max(), h.min()
h = (h - h_min) / (h_max - h_min)
colors = color_map.map(h, mode="float")
md.setFaceColors(colors)
m = gl.GLMeshItem(meshdata=md, smooth=True)
w.addItem(m)

legendLabels = numpy.linspace(h_max, h_min, 5)
legendPos = numpy.linspace(1, 0, 5)
legend = dict(zip(map(str, legendLabels), legendPos))

gll = gl.GLGradientLegendItem(
    pos=(10, 10), size=(50, 300), gradient=color_map, labels=legend
)
w.addItem(gll)

## Start Qt event loop unless running in interactive mode.
if __name__ == "__main__":
    pg.exec()
