"""
Demonstrates use of GLScatterPlotItem with rapidly-updating plots.
"""
import sys
import time

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import pyqtgraph.opengl as gl

if 'darwin' in sys.platform:
    fmt = QtGui.QSurfaceFormat()
    fmt.setRenderableType(fmt.RenderableType.OpenGL)
    fmt.setProfile(fmt.OpenGLContextProfile.CoreProfile)
    fmt.setVersion(4, 1)
    QtGui.QSurfaceFormat.setDefaultFormat(fmt)

app = pg.mkQApp("GLScatterPlotItem Example")
w = gl.GLViewWidget()
w.show()
w.setWindowTitle('pyqtgraph example: GLScatterPlotItem')
w.setCameraPosition(distance=20)

g = gl.GLGridItem()
w.addItem(g)

rng = np.random.default_rng()

##
##  First example is a set of points with pxMode=False
##  These demonstrate the ability to have points with real size down to a very small scale 
## 
pos = np.empty((53, 3))
size = np.empty((53))
color = np.empty((53, 4))
pos[0] = (1,0,0); size[0] = 0.5;   color[0] = (1.0, 0.0, 0.0, 0.5)
pos[1] = (0,1,0); size[1] = 0.2;   color[1] = (0.0, 0.0, 1.0, 0.5)
pos[2] = (0,0,1); size[2] = 2./3.; color[2] = (0.0, 1.0, 0.0, 0.5)

z = 0.5
d = 6.0
for i in range(3,53):
    pos[i] = (0,0,z)
    size[i] = 2./d
    color[i] = (0.0, 1.0, 0.0, 0.5)
    z *= 0.5
    d *= 2.0
    
sp1 = gl.GLScatterPlotItem(pos=pos, size=size, color=color, pxMode=False)
sp1.translate(5,5,0)
w.addItem(sp1)


##
##  Second example shows a volume of points with rapidly updating color
##  and pxMode=True
##

pos = rng.random(size=(100000,3), dtype=np.float32)
pos *= [10,-10,10]
pos[0] = (0,0,0)
d2 = (pos**2).sum(axis=1)**0.5
size = rng.random(size=pos.shape[0], dtype=np.float32) * 10
sp2 = gl.GLScatterPlotItem(pos=pos, color=(1,1,1,1), size=size)

w.addItem(sp2)


##
##  Third example shows a grid of points with rapidly updating position
##  and pxMode = False
##

pos3 = np.zeros((100,100,3), dtype=np.float32)
pos3[:,:,:2] = np.mgrid[:100, :100].transpose(1,2,0) * [-0.1,0.1]
pos3 = pos3.reshape((-1, 3))
d3 = (pos3**2).sum(axis=1)**0.5

sp3 = gl.GLScatterPlotItem(pos=pos3, color=(1,1,1,.3), size=0.1, pxMode=False)

w.addItem(sp3)

time_start = time.perf_counter()
def update():
    elapsed = time.perf_counter() - time_start
    phase = 2*np.pi * (1.0 - (elapsed % 2.0) / 2.0)

    ## update volume colors
    global sp2, d2
    s = -np.cos(d2*2+phase)
    color = np.empty((len(d2),4), dtype=np.float32)
    color[:,3] = s * 0.1
    color[:,0] = s * 3.0
    color[:,1] = s * 1.0
    color[:,2] = s ** 3
    np.clip(color, 0, 1, out=color)
    sp2.setData(color=color)
    
    ## update surface positions and colors
    global sp3, d3, pos3
    z = -np.cos(d3*2+phase)
    pos3[:,2] = z
    color = np.empty((len(d3),4), dtype=np.float32)
    color[:,3] = 0.3
    color[:,0] = z * 3.0
    color[:,1] = z * 1.0
    color[:,2] = z ** 3
    np.clip(color, 0, 1, out=color)
    sp3.setData(pos=pos3, color=color)
    
w.frameSwapped.connect(update)

if __name__ == '__main__':
    pg.exec()
