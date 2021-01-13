# -*- coding: utf-8 -*-
"""
This example demonstrates the use of ImageView, which is a high-level widget for 
displaying and analyzing 2D and 3D data. ImageView provides:

  1. A zoomable region (ViewBox) for displaying the image
  2. A combination histogram and gradient editor (HistogramLUTItem) for
     controlling the visual appearance of the image
  3. A timeline for selecting the currently displayed frame (for 3D data only).
  4. Tools for very basic analysis of image data (see ROI and Norm buttons)

"""
## Add path to library (just for examples; you do not need this)
import initExample

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

app = pg.mkQApp()

## Create window with ImageView widget
win = QtGui.QMainWindow()
win.resize(1000,800)

lw = pg.GraphicsLayoutWidget()
lw.setFixedWidth(1000)
lw.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

scr = QtGui.QScrollArea()
scr.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
scr.setWidget(lw)
win.setCentralWidget(scr)
win.show()
win.setWindowTitle('pyqtgraph example: Color maps')

## Create color map test image
width  = 3*256
height =  32
img = np.zeros( (width, height) )
gradient   = np.linspace(0.05, 0.95, width)
modulation = np.zeros(width)
for idx in range(width):
    modulation[idx] = -0.05 * np.sin( 200 * np.pi * idx/width )
for idx in range(height):
    img[:,idx] = gradient + (idx/(height-1)) * modulation

num_bars = 0

lw.addLabel('=== local color maps ===')
num_bars += 1
lw.nextRow()
list_of_maps = pg.colormap.listMaps()
for map_name in list_of_maps:
    num_bars += 1
    lw.addLabel(map_name)
    cmap = pg.colormap.get(map_name)
    imi = pg.ImageItem()
    imi.setImage(img)
    imi.setLookupTable( cmap.getLookupTable(alpha=True) )
    vb = lw.addViewBox(lockAspect=True, enableMouse=False)
    vb.addItem(imi)
    lw.nextRow()

lw.addLabel('=== Matplotlib import ===')
num_bars += 1
lw.nextRow()
list_of_maps = pg.colormap.listMaps('matplotlib')
for map_name in list_of_maps:
    num_bars += 1
    lw.addLabel(map_name)
    cmap = pg.colormap.get(map_name, source='matplotlib', skipCache=True)
    if cmap is not None:
        imi = pg.ImageItem()
        imi.setImage(img)
        imi.setLookupTable( cmap.getLookupTable(alpha=True) )
        vb = lw.addViewBox(lockAspect=True, enableMouse=False)
        vb.addItem(imi)
    lw.nextRow()

lw.addLabel('=== ColorCET import ===')
num_bars += 1
lw.nextRow()
list_of_maps = pg.colormap.listMaps('colorcet')   
for map_name in list_of_maps:
    num_bars += 1
    lw.addLabel(map_name)
    cmap = pg.colormap.get(map_name, source='colorcet', skipCache=True)
    if cmap is not None:
        imi = pg.ImageItem()
        imi.setImage(img)
        imi.setLookupTable( cmap.getLookupTable(alpha=True) )
        vb = lw.addViewBox(lockAspect=True, enableMouse=False)
        vb.addItem(imi)
    lw.nextRow()
    
lw.setFixedHeight(num_bars * (height+5) )

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
