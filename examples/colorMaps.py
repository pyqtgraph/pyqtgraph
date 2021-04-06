# -*- coding: utf-8 -*-
"""
This example demonstrates generating ColorMap objects from external data.
It displays the full list of color maps available as local files or by import 
from Matplotlib or ColorCET.
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

if __name__ == '__main__':
    pg.mkQApp().exec_()
