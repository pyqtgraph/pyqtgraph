"""
Demonstrates very basic use of PColorMeshItem
"""

import time

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

app = pg.mkQApp("PColorMesh Example")

## Create window with GraphicsView widget
win = pg.GraphicsLayoutWidget()
win.show()  ## show widget alone in its own window
win.setWindowTitle('pyqtgraph example: pColorMeshItem')
view = win.addViewBox()


## Create data

# To enhance the non-grid meshing, we randomize the polygon vertices per and 
# certain amount
randomness = 5

# x and y being the vertices of the polygons, they share the same shape
# However the shape can be different in both dimension
xn = 50 # nb points along x
yn = 40 # nb points along y


x = np.repeat(np.arange(1, xn+1), yn).reshape(xn, yn)\
    + np.random.random((xn, yn))*randomness
y = np.tile(np.arange(1, yn+1), xn).reshape(xn, yn)\
    + np.random.random((xn, yn))*randomness
x.sort(axis=0)
y.sort(axis=0)


# z being the color of the polygons its shape must be decreased by one in each dimension
z = np.exp(-(x*xn)**2/1000)[:-1,:-1]

## Create image item
edgecolors   = None
antialiasing = False
# edgecolors = {'color':'w', 'width':2} # May be uncommened to see edgecolor effect
# antialiasing = True # May be uncommened to see antialiasing effect
pcmi = pg.PColorMeshItem(edgecolors=edgecolors, antialiasing=antialiasing)
view.addItem(pcmi)
textitem = pg.TextItem(anchor=(1, 0))
view.addItem(textitem)


## Set the animation
fps = 25 # Frame per second of the animation

# Wave parameters
wave_amplitude  = 3
wave_speed      = 0.3
wave_length     = 10
color_speed     = 0.3

timer = QtCore.QTimer()
timer.setSingleShot(True)
# not using QTimer.singleShot() because of persistence on PyQt. see PR #1605

textpos = None
i=0
def updateData():
    global i
    global textpos
    
    ## Display the new data set
    t0 = time.perf_counter()
    new_x = x
    new_y = y+wave_amplitude*np.cos(x/wave_length+i)
    new_z = np.exp(-(x-np.cos(i*color_speed)*xn)**2/1000)[:-1,:-1]
    t1 = time.perf_counter()
    pcmi.setData(new_x,
                 new_y,
                 new_z)
    t2 = time.perf_counter()

    i += wave_speed

    # display info in top-right corner
    textitem.setText(f'{(t2 - t1)*1000:.1f} ms')
    if textpos is None:
        textpos = pcmi.width(), pcmi.height()
        textitem.setPos(*textpos)

    # cap update rate at fps
    delay = max(1000/fps - (t2 - t0), 0)
    timer.start(int(delay))

timer.timeout.connect(updateData)
updateData()

if __name__ == '__main__':
    pg.exec()
