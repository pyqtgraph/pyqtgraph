import sys
sys.path.insert(0,'C:/Python27/Lib/site-packages/pg__dev/pyqtgraph/')

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()
w.setWindowTitle('pyqtgraph example: GLGradientLegendItem')
w.setCameraPosition(distance=60)

gx = gl.GLGridItem()
gx.rotate(90, 0, 1, 0)
w.addItem(gx)

md = gl.MeshData.cylinder(rows=10, cols=20, radius=[5., 5], length=20.)
md._vertexes[:,2] = md._vertexes[:,2]-10

# set color based on z coordinates
cmap = pg.ColorMap((0,.25,.5,.75,1),((0,0,255),(0,255,255),(0,255,0),(255,255,0),(255,0,0)))
h = md.vertexes()[:,2]
#remember these
hmax, hmin = h.max(), h.min()
h = h-hmin
h = h/h.max()
colors = cmap.map(h)/255.0
colors = numpy.c_[colors,numpy.ones(len(colors))]
md.setFaceColors(colors)
m = gl.GLMeshItem(meshdata=md, smooth=True)
w.addItem(m)

legendLabels = numpy.linspace(hmax,hmin,5)
legendPos = numpy.linspace(1,0,5)
legend = dict(zip(map(str,legendLabels),legendPos))

gll = gl.GLGradientLegendItem(pos=(10,10), size=(50,300), gradient=cmap, labels=legend)
w.addItem(gll)

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
