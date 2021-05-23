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

win = QtGui.QMainWindow()
win.resize(1000,800)

lw = pg.GraphicsLayoutWidget()
lw.setFixedWidth(1000)
lw.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

scr = QtGui.QScrollArea()
scr.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
scr.setWidget(lw)
win.setCentralWidget(scr)
win.setWindowTitle('pyqtgraph example: Color maps')
win.show()

bar_width = 32
bar_data = pg.colormap.testBarData(width=bar_width)

num_bars = 0

def add_heading(lw, name):
    global num_bars
    lw.addLabel('=== '+name+' ===')
    num_bars += 1
    lw.nextRow()

def add_bar(lw, name, cm):
    global num_bars
    lw.addLabel(name)
    imi = pg.ImageItem( bar_data )
    imi.setLookupTable( cm.getLookupTable(alpha=True) )
    vb = lw.addViewBox(lockAspect=True, enableMouse=False)
    vb.addItem( imi )
    num_bars += 1
    lw.nextRow()

add_heading(lw, 'local color maps')
list_of_maps = pg.colormap.listMaps()
for map_name in list_of_maps:
    cm = pg.colormap.get(map_name)
    add_bar(lw, map_name, cm)

add_heading(lw, 'Matplotlib import')
list_of_maps = pg.colormap.listMaps('matplotlib')
for map_name in list_of_maps:
    cm = pg.colormap.get(map_name, source='matplotlib', skipCache=True)
    if cm is not None:
        add_bar(lw, map_name, cm)

add_heading(lw, 'ColorCET import')
list_of_maps = pg.colormap.listMaps('colorcet')   
for map_name in list_of_maps:
    cm = pg.colormap.get(map_name, source='colorcet', skipCache=True)
    if cm is not None:
        add_bar(lw, map_name, cm)

lw.setFixedHeight(num_bars * (bar_width+5) )

if __name__ == '__main__':
    pg.exec()
