# -*- coding: utf-8 -*-

##  This example uses the isosurface function to convert a scalar field
##  (a hydrogen orbital) into a mesh for 3D display.

## Add path to library (just for examples; you do not need this)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()

g = gl.GLGridItem()
g.scale(2,2,1)
w.addItem(g)

import numpy as np

def psi(i, j, k, offset=(25, 25, 50)):
    x = i-offset[0]
    y = j-offset[1]
    z = k-offset[2]
    th = np.arctan2(z, (x**2+y**2)**0.5)
    phi = np.arctan2(y, x)
    r = (x**2 + y**2 + z **2)**0.5
    a0 = 1
    #ps = (1./81.) * (2./np.pi)**0.5 * (1./a0)**(3/2) * (6 - r/a0) * (r/a0) * np.exp(-r/(3*a0)) * np.cos(th)
    ps = (1./81.) * 1./(6.*np.pi)**0.5 * (1./a0)**(3/2) * (r/a0)**2 * np.exp(-r/(3*a0)) * (3 * np.cos(th)**2 - 1)
    
    return ps
    
    #return ((1./81.) * (1./np.pi)**0.5 * (1./a0)**(3/2) * (r/a0)**2 * (r/a0) * np.exp(-r/(3*a0)) * np.sin(th) * np.cos(th) * np.exp(2 * 1j * phi))**2 


print "Generating scalar field.."
data = np.abs(np.fromfunction(psi, (50,50,100)))


#data = np.fromfunction(lambda i,j,k: np.sin(0.2*((i-25)**2+(j-15)**2+k**2)**0.5), (50,50,50)); 
print "Generating isosurface.."
faces = pg.isosurface(data, data.max()/4.)
m = gl.GLMeshItem(faces)
w.addItem(m)
m.translate(-25, -25, -50)
    

    
#data = np.zeros((5,5,5))
#data[2,2,1:4] = 1
#data[2,1:4,2] = 1
#data[1:4,2,2] = 1
#tr.translate(-2.5, -2.5, 0)
#data = np.ones((2,2,2))
#data[0, 1, 0] = 0
#faces = pg.isosurface(data, 0.5)
#m = gl.GLMeshItem(faces)
#w.addItem(m)
#m.setTransform(tr)

## Start Qt event loop unless running in interactive mode.
if sys.flags.interactive != 1:
    app.exec_()
