"""
Demonstrates very basic use of PColorMeshItem
"""

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from utils import FrameCounter

# pg.setConfigOptions(useOpenGL=True)
app = pg.mkQApp("PColorMesh Example")

## Create window with GraphicsView widget
win = pg.GraphicsLayoutWidget()
win.show()  ## show widget alone in its own window
win.setWindowTitle('pyqtgraph example: pColorMeshItem')
view_auto_scale = win.addPlot(0,0,1,1, title="Auto-scaling colorscheme", enableMenu=False)
view_consistent_scale = win.addPlot(1,0,1,1, title="Consistent colorscheme", enableMenu=False)


## Create data

# To enhance the non-grid meshing, we randomize the polygon vertices per and 
# certain amount
randomness = 5

# x and y being the vertices of the polygons, they share the same shape
# However the shape can be different in both dimension
xrange = 50
yrange = 40

# xrange, yrange are the bounds of the mesh
# whereas xn, yn give the number of points evaluated within those bounds.
# increasing density will increase the processing workload.
density = 1
xn = xrange * density # nb points along x
yn = yrange * density # nb points along y

rng = np.random.default_rng()
x = np.repeat(np.linspace(1, xrange, xn), yn).reshape(xn, yn)\
    + rng.random((xn, yn))*randomness
y = np.tile(np.linspace(1, yrange, yn), xn).reshape(xn, yn)\
    + rng.random((xn, yn))*randomness
x.sort(axis=0)
y.sort(axis=0)


# z being the color of the polygons its shape must be decreased by one in each dimension
z = np.exp(-(x*xrange)**2/1000)[:-1,:-1]

## Create autoscaling image item
edgecolors   = None
antialiasing = False
cmap         = pg.colormap.get('viridis')
levels       = (-2,2) # Will be overwritten unless enableAutoLevels is set to False
# edgecolors = {'color':'w', 'width':2} # May be uncommented to see edgecolor effect
# antialiasing = True # May be uncommented to see antialiasing effect
# cmap         = pg.colormap.get('plasma') # May be uncommented to see a different colormap than the default 'viridis'
pcmi_auto = pg.PColorMeshItem(edgecolors=edgecolors, antialiasing=antialiasing, colorMap=cmap, levels=levels, enableAutoLevels=True)
view_auto_scale.addItem(pcmi_auto)

# Add colorbar
bar = pg.ColorBarItem(
    label = "Z value [arbitrary unit]",
    interactive=False, # Setting `interactive=True` would override `enableAutoLevels=True` of pcmi_auto (resulting in consistent colors)
    rounding=0.1)
bar.setImageItem( [pcmi_auto] )
win.addItem(bar, 0,1,1,1)

# Create image item with consistent colors and an interactive colorbar
pcmi_consistent = pg.PColorMeshItem(edgecolors=edgecolors, antialiasing=antialiasing, colorMap=cmap, levels=levels, enableAutoLevels=False)
view_consistent_scale.addItem(pcmi_consistent)

# Add colorbar
bar_static = pg.ColorBarItem(
    label = "Z value [arbitrary unit]",
    interactive=True,
    rounding=0.1)
bar_static.setImageItem( [pcmi_consistent] )
win.addItem(bar_static,1,1,1,1)

# Add timing label to the autoscaling view
textitem = pg.TextItem(anchor=(1, 0))
view_auto_scale.addItem(textitem)

# Wave parameters
wave_amplitude  = 3
wave_speed      = 0.3
wave_length     = 10
color_speed     = 0.3
color_noise_freq = 0.05

# display info in top-right corner
miny = np.min(y) - wave_amplitude
maxy = np.max(y) + wave_amplitude
view_auto_scale.setYRange(miny, maxy)
textitem.setPos(np.max(x), maxy)

i=0
def updateData():
    global i
    
    ## Display the new data set
    color_noise = np.sin(i * 2*np.pi*color_noise_freq) 
    new_x = x
    new_y = y+wave_amplitude*np.cos(x/wave_length+i)
    new_z = np.exp(-(x-np.cos(i*color_speed)*xrange)**2/1000)[:-1,:-1] + color_noise
    pcmi_auto.setData(new_x, new_y, new_z)
    pcmi_consistent.setData(new_x, new_y, new_z)

    i += wave_speed
    framecnt.update()

timer = QtCore.QTimer()
timer.timeout.connect(updateData)
timer.start()

framecnt = FrameCounter()
framecnt.sigFpsUpdate.connect(lambda fps: textitem.setText(f'{fps:.1f} fps'))

if __name__ == '__main__':
    pg.exec()
