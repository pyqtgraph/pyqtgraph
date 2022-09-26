"""
Use GLImageItem to display image data on rectangular planes.

In this example, the image data is sampled from a volume and the image planes 
placed as if they slice through the volume.
"""

import numpy as np

import pyqtgraph as pg
import pyqtgraph.opengl as gl

app = pg.mkQApp("GLImageItem Example")
w = gl.GLViewWidget()
w.show()
w.setWindowTitle('pyqtgraph example: GLImageItem')
w.setCameraPosition(distance=200)

## create volume data set to slice three images from
shape = (100,100,70)
data = pg.gaussianFilter(np.random.normal(size=shape), (4,4,4))
data += pg.gaussianFilter(np.random.normal(size=shape), (15,15,15))*15

## slice out three planes, convert to RGBA for OpenGL texture
levels = (-0.08, 0.08)
tex1 = pg.makeRGBA(data[shape[0]//2], levels=levels)[0]       # yz plane
tex2 = pg.makeRGBA(data[:,shape[1]//2], levels=levels)[0]     # xz plane
tex3 = pg.makeRGBA(data[:,:,shape[2]//2], levels=levels)[0]   # xy plane
#tex1[:,:,3] = 128
#tex2[:,:,3] = 128
#tex3[:,:,3] = 128

## Create three image items from textures, add to view
v1 = gl.GLImageItem(tex1)
v1.translate(-shape[1]/2, -shape[2]/2, 0)
v1.rotate(90, 0,0,1)
v1.rotate(-90, 0,1,0)
w.addItem(v1)
v2 = gl.GLImageItem(tex2)
v2.translate(-shape[0]/2, -shape[2]/2, 0)
v2.rotate(-90, 1,0,0)
w.addItem(v2)
v3 = gl.GLImageItem(tex3)
v3.translate(-shape[0]/2, -shape[1]/2, 0)
w.addItem(v3)

ax = gl.GLAxisItem()
w.addItem(ax)

if __name__ == '__main__':
    pg.exec()
